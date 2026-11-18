"""Training/sample scheduling, cancellation and UI integration without GPU work."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PySide6.QtCore import QCoreApplication, QEvent, QProcess
from PySide6.QtWidgets import QApplication

from irodori_tts.desktop.app import MainWindow
from irodori_tts.desktop.base_models import base_record
from irodori_tts.desktop.messages import BusinessError
from irodori_tts.desktop.storage import (
    AppStore,
    create_project,
    load_project,
    read_json,
    save_project,
    write_json,
)
from irodori_tts.desktop.tasks import TaskManager, TaskState, TrainingPreviewPhase
from irodori_tts.desktop.training import RECOMMENDED
from irodori_tts.desktop.worker import copy_progress
from tests.base_fixtures import install_base
from tests.qt_cleanup import cleanup_qt, stub_connection_check

REPO = Path(__file__).resolve().parents[1]


def checkpoint(path, step, *, trainer=True):
    files = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"adapter"}
    if trainer:
        files["trainer_state.pt"] = b"state"
    path.mkdir(parents=True)
    for name, data in files.items():
        (path / name).write_bytes(data)
    write_json(
        path / "complete.json", {"step": step, "files": {k: len(v) for k, v in files.items()}}
    )


def setUpModule():
    unittest.addModuleCleanup(stub_connection_check().stop)


class TrainingPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "projects/p/runs/r"
        self.selected = self.run / "checkpoints/checkpoint_0000002"
        self.recovery = self.run / "checkpoints/checkpoint_0000004"
        checkpoint(self.selected, 2)
        checkpoint(self.recovery, 4)
        self.project = self.run.parents[1]
        self.manager = TaskManager(AppStore(self.root), REPO)
        self.manager.timer.stop()
        self.launch = patch.object(self.manager, "_launch").start()
        self.addCleanup(patch.stopall)
        self.training = self.manager.start(
            {"kind": "train", "project": str(self.project), "run": str(self.run), "resume": False}
        )
        self.training["status"] = TaskState.RUNNING
        self.training["total"] = 10
        self.payload = {
            "kind": "preview",
            "project": str(self.project),
            "run": str(self.run),
            "output": str(self.run / "samples/test"),
            "request": {"lora_adapter": str(self.selected), "seed": 42, "num_steps": 40},
        }

    def tearDown(self):
        self.manager.timer.stop()
        self.manager.finish_timer.stop()
        self.manager.training_preview = None
        self.manager.process = None
        self.manager.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        cleanup_qt()
        self.temp.cleanup()

    def test_poll_shows_everything_read_in_one_pass_once(self):
        task = self.manager.current
        with (self.manager.task_dir() / "events.jsonl").open("a", encoding="utf-8") as stream:
            for step in range(1, 501):
                event = {"task_id": task["id"], "sequence": step, "type": "progress", "step": step}
                stream.write(json.dumps(event) + "\n")
        changed = Mock()
        self.manager.changed.connect(changed)
        with (
            patch.object(self.manager, "_owned_process", return_value=Mock()),
            patch("irodori_tts.desktop.tasks.write_json") as written,
        ):
            self.manager.poll()
        changed.assert_called_once()
        written.assert_called_once()
        self.assertEqual(task["step"], 500)
        self.assertEqual(task["sequence"], 500)
        self.assertEqual([step for _when, step in self.manager.samples[task["id"]]], [500])

    def test_release_waits_for_the_exit_without_blocking_and_gives_up_in_time(self):
        self.manager.current["status"] = TaskState.COMPLETED
        process = Mock()
        self.manager.process = process
        self.manager.resident = True
        first, second = Mock(), Mock()
        self.manager.release(first)
        self.manager.release(second)
        process.closeWriteChannel.assert_called_once()
        process.waitForFinished.assert_not_called()
        first.assert_not_called()
        # A task started now waits for the exit instead of writing to the closing process.
        self.assertFalse(self.manager.resident)
        self.manager._released(True)
        first.assert_called_once_with(True)
        second.assert_called_once_with(True)
        self.assertFalse(self.manager.release_timer.isActive())
        self.manager.release(first)
        self.assertTrue(self.manager.release_timer.isActive())
        self.manager.release_timer.stop()
        self.manager._released(False)
        first.assert_called_with(False)
        self.manager.process = None
        late = Mock()
        self.manager.release(late)
        late.assert_called_once_with(True)

    def test_copy_progress_reports_a_few_times_a_second_and_always_the_end(self):
        channel = Mock()
        report = copy_progress(channel, lambda done, total: f"{done}/{total}")
        total = 1000 * 4 * 1024**2
        for chunk in range(1, 1001):
            report(chunk * 4 * 1024**2, total)
        self.assertLess(channel.emit.call_count, 10)
        self.assertEqual(channel.emit.call_args.kwargs["step"], round(total / 1000**2))

    def event(self, kind, **values):
        task = self.manager.current
        event = {"task_id": task["id"], "sequence": task["sequence"] + 1, "type": kind, **values}
        with (self.manager.task_dir() / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        with patch.object(self.manager, "_owned_process", return_value=Mock()):
            self.manager.poll()
        self.manager.finish_timer.stop()
        self.manager._deliver_finished()

    def pause(self, *, status="stopped", recovery=None):
        self.event(
            "result",
            result={
                "status": status,
                "checkpoint": str((recovery or self.recovery).relative_to(self.project)),
                "step": 4,
            },
        )

    def test_sample_uses_selected_checkpoint_and_training_resumes_latest_saved_state(self):
        self.manager.preview(self.payload)
        self.assertTrue((self.manager.task_dir(self.training) / "stop").exists())
        self.assertTrue(self.training["preview_pause"])
        self.assertEqual(self.launch.call_count, 1)
        self.pause()
        preview = self.manager.current
        request = read_json(self.manager.task_dir() / "request.json")
        self.assertEqual(request["request"], self.payload["request"])
        self.assertEqual(self.manager.training_preview.phase, TrainingPreviewPhase.GENERATING)
        self.event("result", result={"audio": "sample.wav"})
        resumed = self.manager.current
        request = read_json(self.manager.task_dir() / "request.json")
        self.assertEqual(request["run"], str(self.run))
        self.assertTrue(request["resume"])
        self.assertNotIn("lora_adapter", request)
        self.assertNotEqual(resumed["id"], self.training["id"])
        self.assertEqual(preview["status"], TaskState.COMPLETED)
        self.assertEqual(self.launch.call_count, 3)
        self.event("progress", step=5, total=10)
        self.assertIsNone(self.manager.training_preview)
        self.assertTrue(self.manager.busy)

    def test_duplicate_requests_and_other_runs_cannot_replace_the_interlude(self):
        self.manager.preview(self.payload)
        interlude = self.manager.training_preview
        for payload in (self.payload, {**self.payload, "run": str(self.run / "other")}):
            with self.assertRaises(BusinessError):
                self.manager.preview(payload)
            self.assertIs(self.manager.training_preview, interlude)
        self.assertIs(self.manager.start({"kind": "generate"}), self.training)
        self.assertEqual(len(self.manager.tasks), 1)

    def test_save_failure_does_not_generate_or_resume(self):
        self.manager.preview(self.payload)
        self.event("error", message="Disk full")
        self.assertIsNone(self.manager.training_preview)
        self.assertFalse(self.manager.busy)
        self.assertEqual(self.launch.call_count, 1)
        self.assertEqual(self.training["message"], "Disk full")

    def test_adapter_only_checkpoint_cannot_be_used_as_training_recovery(self):
        adapter_only = self.run / "checkpoints/adapter-only"
        checkpoint(adapter_only, 4, trainer=False)
        self.manager.preview(self.payload)
        self.pause(recovery=adapter_only)
        self.assertEqual(self.training["status"], TaskState.FAILED)
        self.assertIsNone(self.manager.training_preview)
        self.assertEqual(self.launch.call_count, 1)

    def test_cancel_before_training_starts_does_not_claim_it_saved_a_checkpoint(self):
        self.manager.preview(self.payload)
        self.event("cancelled", message="Stopped before starting")
        self.assertEqual(self.training["status"], TaskState.FAILED)
        self.assertEqual(self.launch.call_count, 1)
        self.assertIsNone(self.manager.training_preview)

    def test_sample_failure_resumes_training_and_preserves_the_error(self):
        self.manager.preview(self.payload)
        self.pause()
        self.event("error", message="Sample failed")
        self.assertEqual(self.manager.current["kind"], "train")
        self.assertEqual(self.manager.current["preview_error"], "Sample failed")
        self.assertTrue(read_json(self.manager.task_dir() / "request.json")["resume"])

    def test_explicit_stop_in_each_phase_cancels_automatic_continuation(self):
        for phase in TrainingPreviewPhase:
            with self.subTest(phase=phase):
                self.payload["output"] = str(self.run / "samples" / phase.value)
                if phase != TrainingPreviewPhase.PAUSING:
                    self.training = self.manager.start(
                        {"kind": "train", "project": str(self.project), "run": str(self.run)}
                    )
                    self.training["status"] = TaskState.RUNNING
                    self.training["total"] = 10
                self.manager.preview(self.payload)
                if phase != TrainingPreviewPhase.PAUSING:
                    self.pause()
                if phase == TrainingPreviewPhase.RESUMING:
                    self.event("result", result={"audio": "sample.wav"})
                launches = self.launch.call_count
                self.manager.stop()
                self.assertIsNone(self.manager.training_preview)
                self.event("cancelled", message="Explicit stop")
                self.assertEqual(self.launch.call_count, launches)

    def test_naturally_completed_training_gets_sample_without_redundant_resume(self):
        self.manager.preview(self.payload)
        self.pause(status="completed")
        self.event("result", result={"audio": "sample.wav"})
        self.assertEqual(self.launch.call_count, 2)
        self.assertIsNone(self.manager.training_preview)
        self.assertFalse(self.manager.busy)

    def test_force_stop_cancels_the_interlude(self):
        self.manager.preview(self.payload)
        with patch.object(self.manager, "_owned_process", return_value=Mock()) as owned:
            self.manager.force_stop()
            owned.return_value.terminate.assert_called_once()
        self.assertIsNone(self.manager.training_preview)
        self.event("error", message="Terminated")
        self.assertEqual(self.launch.call_count, 1)

    def test_stop_during_gpu_handoff_never_launches_the_pending_task(self):
        for force in (False, True):
            with self.subTest(force=force):
                if force:
                    self.manager.process = None
                    self.training = self.manager.start(
                        {"kind": "train", "project": str(self.project), "run": str(self.run)}
                    )
                    self.training["status"] = TaskState.RUNNING
                    self.training["total"] = 10
                    self.payload["output"] = str(self.run / "samples/force")
                self.manager.preview(self.payload)
                previous = Mock()
                self.manager.process = previous
                self.pause()
                self.assertIsNotNone(self.manager.pending)
                launches = self.launch.call_count
                if force:
                    self.manager.force_stop()
                else:
                    self.manager.stop()
                self.assertIsNone(self.manager.pending)
                self.assertIsNone(self.manager.training_preview)
                with patch.object(self.manager, "sender", return_value=previous):
                    self.manager._exited(0, QProcess.ExitStatus.NormalExit)
                self.manager.finish_timer.stop()
                self.manager._deliver_finished()
                self.assertEqual(self.launch.call_count, launches)
                self.assertFalse(self.manager.busy)

    def test_resumed_training_validation_failure_ends_workflow(self):
        self.manager.preview(self.payload)
        self.pause()
        self.event("result", result={"audio": "sample.wav"})
        self.event("error", message="Training data changed")
        self.assertEqual(self.manager.current["status"], TaskState.FAILED)
        self.assertIsNone(self.manager.training_preview)
        self.assertEqual(self.launch.call_count, 3)

    def test_preview_process_failed_to_start_can_still_restore_training(self):
        self.manager.preview(self.payload)
        self.pause()
        process = Mock()
        process.errorString.return_value = "Cannot start sample process"
        self.manager.process = process
        with patch.object(self.manager, "sender", return_value=process):
            self.manager._process_error(QProcess.ProcessError.FailedToStart)
        self.manager.finish_timer.stop()
        self.manager._deliver_finished()
        self.assertEqual(self.manager.current["kind"], "train")
        self.assertIsNone(self.manager.pending)
        self.assertEqual(self.launch.call_count, 3)
        process.deleteLater.assert_called_once()

    def test_selected_checkpoint_is_preserved_when_safe_save_prunes_the_original(self):
        self.manager.preview(self.payload)
        snapshot = Path(self.manager.training_preview.preview["checkpoint_snapshot"])
        self.assertEqual((snapshot / "adapter_model.safetensors").read_bytes(), b"adapter")
        self.assertFalse((snapshot / "trainer_state.pt").exists())
        self.assertTrue(snapshot.is_relative_to(self.run / "samples"))
        shutil.rmtree(self.selected)
        self.pause()
        request = read_json(self.manager.task_dir() / "request.json")
        self.assertEqual(request["checkpoint_snapshot"], str(snapshot))
        self.assertTrue(snapshot.is_dir())
        self.assertEqual(request["request"]["lora_adapter"], str(self.selected))

    def test_unproduced_samples_leave_no_pinned_adapter_behind(self):
        output = self.run / "samples/test"
        self.manager.preview(self.payload)
        self.assertTrue((output / "adapter").is_dir())
        self.manager.stop()  # still pausing: no sample task exists yet
        self.assertFalse(output.exists())
        self.event("cancelled", message="Stopped")
        for outcome in ("pause_failed", "sample_failed"):
            with self.subTest(outcome=outcome):
                self.training = self.manager.start(
                    {"kind": "train", "project": str(self.project), "run": str(self.run)}
                )
                self.training["status"] = TaskState.RUNNING
                self.training["total"] = 10
                self.manager.preview(self.payload)
                self.assertTrue((output / "adapter").is_dir())
                if outcome == "pause_failed":
                    self.event("error", message="Disk full")
                else:
                    self.pause()
                    self.event("error", message="Sample failed")
                    self.event("cancelled", message="Stopped")
                self.assertFalse(output.exists())

    def test_damaged_task_records_do_not_stop_startup(self):
        tasks = self.root / "tasks"
        (tasks / "partial").mkdir(parents=True)
        (tasks / "partial/task.json").write_text('{"id": "partial", "kin', encoding="utf-8")
        (tasks / "old").mkdir()
        write_json(tasks / "old/task.json", {"id": "old", "kind": "generate"})
        manager = TaskManager(AppStore(self.root), REPO)
        try:
            manager.timer.stop()
            ids = {task["id"] for task in manager.tasks}
            self.assertIn("old", ids)
            self.assertNotIn("partial", ids)
        finally:
            manager.deleteLater()

    def test_restart_adopted_training_releases_the_gpu_before_the_sample_starts(self):
        # No QProcess handle: the training process was adopted after an app restart.
        self.training["pid"] = 4242
        self.manager.preview(self.payload)
        adopted = Mock()
        adopted.status.return_value = "running"
        with patch.object(self.manager, "_owned_process", return_value=adopted):
            self.pause()  # the adopted process is still alive while it tears down
        self.assertEqual(self.launch.call_count, 1)
        self.assertIsNotNone(self.manager.pending)
        self.assertIs(self.manager.handoff[0], adopted)
        with patch.object(self.manager, "_owned_process", return_value=None):
            self.manager.poll()  # waiting for the GPU is not an interrupted task
        self.assertEqual(self.manager.current["status"], TaskState.PREPARING)
        self.manager._check_handoff()
        self.assertEqual(self.launch.call_count, 1)
        adopted.is_running.return_value = False
        self.manager._check_handoff()
        self.assertEqual(self.launch.call_count, 2)
        self.assertIsNone(self.manager.pending)
        self.assertFalse(self.manager.handoff_timer.isActive())

    def test_snapshot_rejects_output_outside_the_training_project_before_stopping(self):
        with self.assertRaises(BusinessError):
            self.manager.preview({**self.payload, "output": str(self.root / "outside")})
        self.assertIsNone(self.manager.training_preview)
        self.assertEqual(self.training["status"], TaskState.RUNNING)
        self.assertFalse((self.manager.task_dir(self.training) / "stop").exists())

    def test_training_page_buttons_start_one_interlude_and_keep_saved_samples_playable(self):
        project = create_project(self.root / "ui_projects", "UI project")
        record = load_project(project)
        record["initialization"] = {
            "id": "init",
            "revision": record["revision"],
            "status": "completed",
            "count": 0,
            "created": "2026-10-28T01:00:00+00:00",
        }
        save_project(project, record)
        run_root = project / "runs/ui-run"
        selected = run_root / "checkpoints/checkpoint_0000002"
        saved = run_root / "checkpoints/checkpoint_0000004"
        for path, step in ((selected, 2), (saved, 4), (saved.with_name(".checkpoint.previous"), 4)):
            checkpoint(path, step)
        run = {
            "id": "ui-run",
            "name": "UI training",
            "created": "2026-10-28T01:00:00+00:00",
            "status": "running",
            "config": {"train": {**RECOMMENDED, "max_steps": 10}},
            "step": 4,
            "checkpoint": str(saved.relative_to(project)),
            "registered": False,
            "initialization_id": "init",
        }
        store = AppStore(self.root / "ui_data")
        # A sample speaks on the run's base, which must be installed.
        base = install_base(store.root / "resources")
        run["base"] = base_record(base)
        write_json(run_root / "run.json", run)
        store.register(project)
        with patch.object(MainWindow, "detect_device", lambda self, error=None: None):
            window = MainWindow(store)
            window.training.runs.setCurrentIndex(window.training.runs.model().index(0, 0))
            manager = window.tasks
            manager.timer.stop()
            window.timer.stop()
            try:
                with patch.object(manager, "_launch"):
                    task = manager.start(
                        {"kind": "train", "project": str(project), "run": str(run_root)}
                    )
                    task.update(status=TaskState.RUNNING, step=4, total=10)
                    manager.changed.emit(task)
                    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                    page = window.training

                    # Checkpoints are points on the loss curve; the hidden atomic-save
                    # backup is omitted.
                    self.assertEqual(len(page.loss_chart.checkpoints), 2)
                    page.checkpoint_chosen(str(selected))
                    self.assertEqual(
                        page.sample_button.property("sample_checkpoint"), str(selected)
                    )
                    self.assertTrue(page.sample_button.isEnabled())
                    page.sample_button.click()
                    self.assertIsNotNone(manager.training_preview)
                    self.assertTrue(window.stop.isEnabled())
                    self.assertFalse(page.sample_button.isEnabled())
                    self.assertEqual(len(manager.tasks), 1)
                    run.update(status="stopped")
                    write_json(run_root / "run.json", run)
                    task["result"] = run
                    manager._finish(TaskState.STOPPED, "Saved")
                    manager.finish_timer.stop()
                    manager._deliver_finished()
                    self.assertEqual(len(manager.tasks), 2)  # no duplicate automatic final sample
                    audio = REPO / "examples/desktop/basic.wav"
                    write_json(
                        run_root / "samples/ready/generation.json",
                        {
                            "audio": str(audio),
                            "request": {"lora_adapter": str(selected)},
                            "checkpoint_snapshot": manager.training_preview.preview[
                                "checkpoint_snapshot"
                            ],
                        },
                    )
                    shutil.rmtree(selected)
                    page.refresh_checkpoints(page.runs.item())
                    # The sample's own copy of the removed checkpoint stays on the curve.
                    kept = next(
                        key
                        for key in page.checkpoints_shown
                        if page.sample_sources.get(key) == str(selected)
                    )
                    with patch.object(page.sample_player, "load") as play:
                        page.checkpoint_chosen(kept)
                        play.assert_called_once_with(audio)
                    self.assertIs(page.sample_stack.currentWidget(), page.sample_player)
                    self.assertEqual(
                        page.sample_button.property("sample_checkpoint"), str(selected)
                    )
                    with patch.object(window, "confirm", return_value=True):
                        window.close()
                    self.assertTrue(window.closing_after_stop)
                    self.assertIsNone(manager.training_preview)
                    manager._finish(TaskState.STOPPED, "Stopped sample")
                    manager.finish_timer.stop()
                    manager._deliver_finished()
                    self.assertEqual(len(manager.tasks), 2)
            finally:
                manager.training_preview = None
                manager.timer.stop()
                manager.finish_timer.stop()
                manager.current["status"] = TaskState.STOPPED
                window.close()
                for _ in range(5):
                    self.app.processEvents()


if __name__ == "__main__":
    from tests.run import guarded_main

    raise SystemExit(guarded_main(lambda: unittest.main()))
