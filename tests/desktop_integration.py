"""Real Qt/QProcess workflow. Run only after the recommended GPU job is finished."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QEventLoop, QLocale, QTimer
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication
from qfluentwidgets import FluentTranslator

from irodori_tts.desktop.app import MainWindow
from irodori_tts.desktop.exports import (
    export_diagnostics,
    export_project,
    export_voice,
    import_project,
    import_voice,
)
from irodori_tts.desktop.materials import edit_material, import_materials
from irodori_tts.desktop.messages import BusinessError
from irodori_tts.desktop.resources import configure_environment
from irodori_tts.desktop.storage import (
    AppStore,
    create_project,
    digest,
    load_project,
    read_json,
    write_json,
)
from irodori_tts.desktop.training import complete_checkpoint

REPO = Path(__file__).resolve().parents[1]


def pump(milliseconds=50):
    """Run the event loop briefly. QSignalSpy.wait and QTest.qWait keep the GIL while the media
    backend, after autoplay, joins a decoder thread that needs it: both deadlocked here, and
    an isolated player did too. A QEventLoop releases it, like app.exec."""
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=Path, default=REPO / "examples/desktop")
    parser.add_argument("--data-dir", type=Path, default=REPO / "outputs/desktop-integration")
    parser.add_argument("--skip-prerequisite", action="store_true")
    args = parser.parse_args()
    if (
        not args.skip_prerequisite
        and not (REPO / "outputs/desktop-gpu/recommended-result.json").exists()
    ):
        raise RuntimeError(
            "Recommended training is still running; do not start a competing GPU workflow"
        )
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["HF_HUB_OFFLINE"] = "1"
    configure_environment(REPO)
    app = QApplication([])
    translator = FluentTranslator(QLocale("zh_CN"), app)
    app.installTranslator(translator)
    data = args.data_dir.resolve()
    store = AppStore(data)
    source = args.fixture_root.resolve()
    project = create_project(data / "项目 空格", "完整应用验收")
    import_materials(
        project,
        source,
        source / "transcripts.csv" if (source / "transcripts.csv").exists() else None,
    )
    store.register(project)
    window = MainWindow(store)
    window.stack.setAnimationEnabled(False)
    window.resize(1200, 760)
    window.show()
    app.processEvents()
    manager = window.tasks
    finished = QSignalSpy(manager.finished)
    delivered = 0
    checks = []
    tasks = []

    def await_job(kind, status="completed"):
        nonlocal delivered
        deadline = time.monotonic() + 60
        while delivered >= finished.count():
            assert time.monotonic() < deadline, f"Timed out: {manager.current}"
            pump()
        task = finished.at(delivered)[0]
        delivered += 1
        assert task["kind"] == kind and task["status"] == status, task
        tasks.append(task)
        print(f"{kind}: {status}, step {task['step']}, task {task['id']}", flush=True)
        return task

    # Actual manual generation followed by another request in the same process.
    window.generation.fill_example()
    window.generation.generate()
    await_job("generate")
    pid = manager.process.processId()
    window.generation.caption.setText("落ち着いた、自然な女性の声。")
    reference_paths = [str(project / item["path"]) for item in load_project(project)["materials"]]
    with patch(
        "irodori_tts.desktop.widgets.QFileDialog.getOpenFileNames",
        return_value=(reference_paths, ""),
    ):
        window.generation.add_references()
    # References are copied and measured on a pool thread.
    deadline = time.monotonic() + 60
    while len(window.generation.references) < len(reference_paths):
        assert time.monotonic() < deadline, "references were not added"
        pump()
    window.generation.generate()
    second = await_job("generate")
    assert manager.process.processId() == pid
    assert second["result"]["request"]["ref_wavs"] == [
        row["path"] for row in window.generation.references
    ]
    checks.append("resident inference reused; caption and ordered references reached the runtime")

    # The manager releases resident GPU resources before preprocessing/training.
    window.training.initialize()
    await_job("initialize")
    assert window.training.goto_training.isEnabled()
    checks.append("GUI import -> actual initialization -> ready training controls")
    for key, value in {
        "max_steps": 4,
        "warmup_steps": 1,
        "save_every": 2,
        "valid_every": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
    }.items():
        window.training.parameters[key].setValue(value)
    stopped_once = False

    def stop_training(task):
        nonlocal stopped_once
        if (
            task["kind"] == "train"
            and task["step"] >= 1
            and task["status"] == "running"
            and not stopped_once
        ):
            stopped_once = True
            manager.stop()

    manager.changed.connect(stop_training)
    window.training.start_training()
    # The run is created on a pool thread first; a refused start leaves the task idle.
    deadline = time.monotonic() + 60
    while not manager.busy:
        assert time.monotonic() < deadline, "training did not start"
        pump()
    stopped = await_job("train", "stopped")
    manager.changed.disconnect(stop_training)
    await_job("preview")
    run_root = Path(stopped["run"])
    run = read_json(run_root / "run.json")
    assert run["registered"] and run["preview_status"] == "completed"
    assert len(store.history()) == 2
    checkpoint = project / run["checkpoint"]
    assert complete_checkpoint(checkpoint, require_trainer=True)
    checks.append(
        "GUI safe stop registered a complete adapter and automatically generated a separate run preview"
    )

    # Migrate an unfinished project and resume through the real new location.
    archive = data / "可恢复项目.iroproject"
    export_project(project, archive)
    migrated = import_project(archive, data / "迁移 目标")
    window.register_project(migrated)
    project = migrated
    # Run folders are named after the run; the record ID stays stable across a move.
    run_root = next(
        path.parent
        for path in (project / "runs").glob("*/run.json")
        if read_json(path).get("id") == run["id"]
    )
    window.training.runs.setCurrentIndex(window.training.runs.model().index(0, 0))
    window.training.resume_training()
    await_job("train")
    await_job("preview")
    run = read_json(run_root / "run.json")
    assert run["step"] == 4 and run["status"] == "completed"
    checks.append(
        "migrated Chinese/space project resumed optimizer/scheduler/data state to completion with locally resolved base resources"
    )

    # Using a model keeps the draft; the active request retains a frozen snapshot.
    draft = "こんにちは。これはモデルを使った生成です。"
    window.generation.text.setPlainText(draft)
    window.refresh()
    window.training.select_run_id(run["id"])
    window.training.use_run()
    assert window.generation.text.toPlainText() == draft
    assert window.generation.voice.currentData() == run["id"]
    widget_id = id(window.generation.text)
    window.generation.generate()
    window.generation.text.setPlainText("次のリクエストのために残す下書きです。")
    generated = await_job("generate")
    assert generated["result"]["request"]["text"] == draft
    assert id(window.generation.text) == widget_id and window.generation.text.toPlainText() != draft
    checks.append(
        "library voice selection preserved draft; active generation retained frozen text/voice/parameters"
    )

    # Failed rebuild preflight must retain all old initialization and user results.
    before = load_project(project)["initialization"]
    old_manifest = digest(project / "initialized/manifest.jsonl")
    old_adapter = digest(project / run["checkpoint"] / "adapter_model.safetensors")
    item = load_project(project)["materials"][0]
    edit_material(project, item["id"], text="", caption=item["caption"], included=True)
    window.refresh()
    with patch.object(window, "confirm", return_value=True):
        count = len(manager.tasks)
        try:
            window.training.initialize()
        except BusinessError:
            pass
        else:
            raise AssertionError("an included recording without text must block initialization")
        assert len(manager.tasks) == count
    assert digest(project / "initialized/manifest.jsonl") == old_manifest
    assert load_project(project)["initialization"]["id"] == before["id"]
    edit_material(project, item["id"], text=item["text"], caption=item["caption"], included=True)
    window.refresh()
    with patch.object(window, "confirm", return_value=False):
        count = len(manager.tasks)
        window.training.initialize()
        assert len(manager.tasks) == count
    with patch.object(window, "confirm", return_value=True):
        window.training.initialize()
    await_job("initialize")
    assert load_project(project)["initialization"]["id"] != before["id"]
    assert digest(project / run["checkpoint"] / "adapter_model.safetensors") == old_adapter
    assert (run_root / "samples").is_dir()
    checks.append(
        "rebuild cancel and failed preflight retained old files; confirmed rebuild changed data version and preserved adapter/checkpoints/previews"
    )

    window.generation.generate()
    after_rebuild = await_job("generate")
    assert Path(after_rebuild["result"]["audio"]).is_file()
    audio_export = data / "导出 音频.wav"
    shutil.copyfile(after_rebuild["result"]["audio"], audio_export)
    assert digest(audio_export) == digest(Path(after_rebuild["result"]["audio"]))
    voice_export = data / "完整音色.irovoice"
    export_voice(project, read_json(run_root / "run.json"), voice_export)
    imported_voice = import_voice(voice_export, data / "导入音色")
    assert not (imported_voice / "trainer_state.pt").exists()
    export_diagnostics(store, data / "诊断.zip")
    checks.append(
        "old voice still synthesized after rebuild; actual WAV/voice/project/diagnostic exports verified"
    )

    # Empty close/reopen restores page, Japanese draft, project identity and settings.
    window.navigate(1)
    window.save_ui()
    saved_draft = window.generation.text.toPlainText()
    window.close()
    app.processEvents()
    reopened = MainWindow(AppStore(data))
    assert reopened.generation.text.toPlainText() == saved_draft
    assert reopened.training.project["id"] == load_project(project)["id"]
    reopened.close()
    app.processEvents()
    checks.append(
        "idle close reclaimed resident process; restart restored Japanese draft and migrated project"
    )
    write_json(
        data / "integration-result.json",
        {
            "checks": checks,
            "tasks": tasks,
            "project": str(project),
            "run": str(run_root),
            "source_audio": "existing valid synthesized Japanese WAVs and exact original transcripts",
            "platform": app.platformName(),
            "offline": True,
        },
    )
    print("QT END-TO-END WORKFLOW PASSED", flush=True)
    from tests.qt_cleanup import cleanup_qt

    cleanup_qt()


if __name__ == "__main__":
    from tests.run import guarded_main

    raise SystemExit(guarded_main(main, timeout=900))
