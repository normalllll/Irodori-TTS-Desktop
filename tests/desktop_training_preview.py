"""Opt-in offscreen Qt/GPU regression for pause -> sample -> safe training resume."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from irodori_tts.desktop.app import MainWindow
from irodori_tts.desktop.materials import import_materials
from irodori_tts.desktop.resources import configure_environment
from irodori_tts.desktop.storage import AppStore, create_project, digest, read_json, write_json
from irodori_tts.desktop.tasks import TrainingPreviewPhase
from irodori_tts.desktop.training import complete_checkpoint
from tests.qt_cleanup import cleanup_qt
from tests.run import guarded_main

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    data = args.data_dir.resolve()
    if data.exists():
        raise ValueError("Use a new data directory to preserve earlier evidence")
    # Never creates a native window or uses desktop input, including during capture.
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["HF_HUB_OFFLINE"] = "1"
    configure_environment(REPO)
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    store = AppStore(data)
    store.settings.update(language="zh_CN", theme="dark", reduce_motion=True)
    project = create_project(data / "projects", "Training preview regression")
    import_materials(project, REPO / "examples/desktop", REPO / "examples/desktop/transcripts.csv")
    store.register(project)
    window = MainWindow(store)
    window.resize(1520, 920)
    window.show()
    window.navigate(2)
    window.training.tabs.setCurrentIndex(2)
    manager = window.tasks
    finished = []
    errors = []
    requested = False
    requested_checkpoint = None
    captured = False
    recovery = None
    recovery_snapshot = {}
    recovery_unchanged = False

    def record_finished(task):
        nonlocal recovery, recovery_snapshot, recovery_unchanged
        finished.append(dict(task))
        if task["kind"] == "train" and task["status"] == "stopped":
            recovery = project / task["result"]["checkpoint"]
            recovery_snapshot = {
                name: digest(recovery / name)
                for name in read_json(recovery / "complete.json")["files"]
            }
        elif task["kind"] == "preview" and recovery and not recovery_unchanged:
            # Resumed training waits for the inference process to release its GPU.
            # Sample inference must leave all saved recovery state untouched.
            assert all(
                digest(recovery / name) == value for name, value in recovery_snapshot.items()
            )
            recovery_unchanged = True

    manager.finished.connect(record_finished)

    def wait_until(condition, timeout=180):
        deadline = time.monotonic() + timeout
        while not condition():
            assert not errors, errors
            assert time.monotonic() < deadline, (
                f"Timed out: {manager.current}; shown: {window.training.start_error.text()!r}"
            )
            # QTest.qWait keeps the GIL and starves the application's pool threads.
            loop = QEventLoop()
            QTimer.singleShot(30, loop.quit)
            loop.exec()

    def request_sample(task):
        nonlocal requested, requested_checkpoint, captured
        if (
            not requested
            and task["kind"] == "train"
            and task["status"] == "running"
            and task["step"] >= 5
        ):
            selected = Path(task["run"]) / "checkpoints/checkpoint_0000002"
            if complete_checkpoint(selected):
                requested = True
                requested_checkpoint = selected
                window.training.preview(selected)
                print(f"Requested sample at training step {task['step']}: {selected}", flush=True)
        if (
            not captured
            and manager.training_preview
            and manager.training_preview.phase == TrainingPreviewPhase.GENERATING
        ):
            captured = True
            window.grab().save(str(data / "training-preview-dark.png"))

    def record_failure(task):
        if task["status"] in {"failed", "interrupted"}:
            errors.append(task)

    manager.changed.connect(request_sample)
    manager.finished.connect(record_failure)
    try:
        window.training.initialize()
        wait_until(lambda: len(finished) == 1 and not manager.busy)
        assert finished[0]["status"] == "completed", finished[0]
        parameters = {
            "max_steps": 8,
            "warmup_steps": 1,
            "save_every": 2,
            "valid_every": 0,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True,
        }
        for key, value in parameters.items():
            window.training.parameters[key].setValue(value)
        window.training.voice_name.setText("Training with sample")
        window.training.start_training()
        wait_until(lambda: requested and not manager.busy and len(finished) >= 5)
        assert not errors, errors
        training_tasks = [task for task in finished if task["kind"] == "train"]
        preview_tasks = [task for task in finished if task["kind"] == "preview"]
        assert [task["status"] for task in training_tasks] == ["stopped", "completed"], finished
        stopped, completed = training_tasks
        assert 2 <= stopped["result"]["step"] < 8, stopped
        assert completed["result"]["step"] == 8, completed
        assert len(preview_tasks) == 2, (
            preview_tasks
        )  # requested sample, then ordinary final sample
        sample = preview_tasks[0]["result"]
        assert Path(sample["audio"]).is_file(), sample
        assert sample["request"]["lora_adapter"] == str(requested_checkpoint), sample
        assert complete_checkpoint(Path(sample["checkpoint_snapshot"])), sample
        # Every save is kept for comparison; older ones keep a usable adapter.
        assert complete_checkpoint(requested_checkpoint), "The requested checkpoint should be kept"
        assert sample["request"]["seed"] == 42 and sample["request"]["num_steps"] == 40
        run_root = Path(completed["run"])
        run = read_json(run_root / "run.json")
        assert run["status"] == "completed" and run["step"] == 8
        assert len(window.training.samples_by_checkpoint(run_root)) >= 2
        assert manager.training_preview is None
        window.grab().save(str(data / "training-completed-dark.png"))
        print("Real sample completed and training automatically reached step 8", flush=True)

        assert recovery_unchanged
        resumed_request = read_json(manager.task_dir(completed) / "request.json")
        assert resumed_request["resume"] and resumed_request["run"] == str(run_root)
        events = [
            json.loads(line)
            for line in (manager.task_dir(completed) / "events.jsonl")
            .read_text("utf-8")
            .splitlines()
        ]
        assert any(
            event["type"] == "running" and event.get("step") == stopped["result"]["step"]
            for event in events
        ), events
        log = (manager.task_dir(completed) / "worker.log").read_text("utf-8")
        assert "Restored dataloader state for mid-epoch resume." in log
        report = {
            "project": str(project),
            "run": str(run_root),
            "sample": sample,
            "stopped_step": stopped["result"]["step"],
            "completed_step": completed["result"]["step"],
            "recovery_checkpoint_unchanged_by_sample": recovery_unchanged,
            "recovery_checkpoint_files": list(recovery_snapshot),
            "resumed_from_saved_step": True,
            "dataloader_state_restored": True,
            "selected_checkpoint_survived_retention_as_snapshot": True,
            "task_sequence": [(t["kind"], t["status"]) for t in finished],
            "offscreen": True,
        }
        write_json(data / "result.json", report)
        print(
            "TRAINING PREVIEW PASSED; recovery state unchanged and saved training restored",
            flush=True,
        )
    finally:
        cleanup_qt()


if __name__ == "__main__":
    raise SystemExit(guarded_main(main, timeout=900))
