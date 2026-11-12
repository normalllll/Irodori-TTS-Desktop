from __future__ import annotations

import collections
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal, Slot

from irodori_tts.desktop.messages import BusinessError, error_message, fmt, t

from .asr import launch_environment
from .messages import restore
from .storage import now, read_json, stamp, write_json
from .training import complete_checkpoint, discard_preview_snapshot, snapshot_preview_checkpoint

# How long a launch waits for an adopted compute process (one that outlived an app
# restart) to release the GPU after its task reported the end.
HANDOFF_TIMEOUT = 60.0


class TrainingPreviewPhase(str, Enum):
    PAUSING = "pausing"
    GENERATING = "generating"
    RESUMING = "resuming"


@dataclass
class TrainingPreview:
    training_task_id: str
    training: dict
    preview: dict
    phase: TrainingPreviewPhase = TrainingPreviewPhase.PAUSING
    task_id: str | None = None
    resume: bool = True


class TaskState(str, Enum):
    PREPARING = "preparing"
    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    SAVING = "saving"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


ACTIVE = {TaskState.PREPARING, TaskState.RUNNING, TaskState.STOP_REQUESTED, TaskState.SAVING}
STATE_TEXT = {
    "preparing": t("tasks.preparing"),
    "running": t("tasks.running"),
    "stop_requested": t("tasks.waiting_for_safe_stop"),
    "saving": t("tasks.saving"),
    "completed": t("app.completed"),
    "stopped": t("tasks.stopped"),
    "failed": t("tasks.failed"),
    "interrupted": t("tasks.previous_task_interrupted"),
    "pending": t("tasks.awaiting_preview"),
}
KIND_TEXT = {
    "initialize": t("tasks.data_initialization"),
    "train": t("tasks.lora_training"),
    "recommend": t("tasks.training_recommendation"),
    "generate": t("app.generate"),
    "preview": t("tasks.voice_preview"),
    "resources": t("settings_page.prepare_resources"),
    "transcribe": t("tasks.transcription"),
    "addon": t("tasks.addon_install"),
    "models": t("tasks.model_download"),
    "import_models": t("tasks.import_models"),
    "export_models": t("tasks.export_models"),
}
# The model-file task downloads, verifies or downloads again; each is named for what it does.
RESOURCE_TEXT = {
    "download": t("settings_page.prepare_resources"),
    "verify": t("tasks.verify_resources"),
    "repair": t("tasks.repair_resources"),
}


def resource_action(payload: dict) -> str:
    return "repair" if payload.get("repair") else "verify" if payload.get("verify") else "download"


# How long the compute process gets to exit when asked, in milliseconds.
RELEASE_TIMEOUT = 10000


class TaskManager(QObject):
    changed = Signal(dict)
    finished = Signal(dict)

    def __init__(self, store, repo: Path, parent=None):
        super().__init__(parent)
        self.store, self.repo = store, repo
        self.process = None
        self.current = None
        self.resident = False
        self.pending = None
        # An interlude belongs to this session. After an app crash, the normal task
        # recovery keeps the saved training resumable without starting work on its own.
        self.training_preview: TrainingPreview | None = None
        self.offsets = {}
        self.tasks = []
        self.log = None
        self.finished_tasks = []
        # Live step timestamps (for the remaining-time estimate) and per-step losses read from
        # each task's event log. Neither is persisted beyond the logs the worker already writes.
        self.samples = {}
        self.series = {}
        self.finish_timer = QTimer(self)
        self.finish_timer.setSingleShot(True)
        self.finish_timer.timeout.connect(self._deliver_finished)
        # Waits for an adopted process to exit before the pending launch (no QProcess
        # handle exists for it, so there is no finished signal).
        self.handoff = None
        self.handoff_timer = QTimer(self)
        self.handoff_timer.setInterval(200)
        self.handoff_timer.timeout.connect(self._check_handoff)
        # Who waits for the compute process to exit (release).
        self.releasing = []
        self.release_timer = QTimer(self)
        self.release_timer.setSingleShot(True)
        self.release_timer.setInterval(RELEASE_TIMEOUT)
        self.release_timer.timeout.connect(lambda: self._released(False))
        self._load()
        self.timer = QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.poll)
        self.timer.start()

    def _load(self):
        for path in sorted((self.store.root / "tasks").glob("*/task.json"), reverse=True):
            # One damaged or half-written record must not stop the app from starting.
            try:
                task = read_json(path)
                if not isinstance(task, dict) or not task.get("id") or not task.get("kind"):
                    continue
                if task.get("status") in ACTIVE:
                    # Only this app's verified command line can remain active after a crash.
                    alive = self._owned_process(task)
                    if alive:
                        self.current = task
                    else:
                        task.update(
                            status=TaskState.INTERRUPTED,
                            message=t("tasks.previous_process_ended_check_the_complete_recovery"),
                        )
                        write_json(path, task)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            task.setdefault("status", TaskState.INTERRUPTED)
            task.setdefault("created", "")
            self.tasks.append(task)

        self.tasks.sort(key=lambda task: task.get("created") or "", reverse=True)

    def _owned_process(self, task):
        try:
            import psutil

            process = psutil.Process(task.get("pid", -1))
            command = process.cmdline()
            return (
                process
                if "irodori_tts.desktop.worker" in command
                and any(task["id"] in part for part in command)
                else None
            )
        except (ValueError, OSError, psutil.Error):
            return None

    @property
    def busy(self):
        return self.training_preview is not None or (
            self.current is not None and self.current["status"] in ACTIVE
        )

    def can_preview_training(self, run_root):
        task = self.current
        return bool(
            self.training_preview is None
            and task
            and task["kind"] == "train"
            and task["status"] == TaskState.RUNNING
            and task.get("total", 0) > 0
            and task.get("run") == str(run_root)
        )

    def preview(self, payload):
        if not self.busy:
            return self.start(payload)
        if not self.can_preview_training(payload["run"]):
            raise BusinessError(t("tasks.preview_wait_for_training"))
        checkpoint = Path(payload["request"]["lora_adapter"])
        if not complete_checkpoint(checkpoint):
            raise BusinessError(t("training_page.select_a_complete_checkpoint"))
        training = read_json(self.task_dir() / "request.json")
        snapshot = snapshot_preview_checkpoint(
            Path(training["project"]), checkpoint, Path(payload["output"]) / "adapter"
        )
        self.training_preview = TrainingPreview(
            self.current["id"],
            {**training, "resume": True},
            {**payload, "checkpoint_snapshot": str(snapshot)},
        )
        # Suppress the ordinary end-of-training preview for this intentional pause.
        self.current["preview_pause"] = True
        self._request_stop()
        return self.current

    def preview_message(self):
        if self.training_preview is None:
            return None
        match self.training_preview.phase:
            case TrainingPreviewPhase.PAUSING:
                return t("tasks.preview_saving_training")
            case TrainingPreviewPhase.GENERATING:
                return t("tasks.preview_training_paused")
            case TrainingPreviewPhase.RESUMING:
                return t("tasks.preview_resuming_training")

    def task_dir(self, task=None):
        return self.store.root / "tasks" / (task or self.current)["id"]

    def start(self, payload: dict):
        if self.busy:
            self.changed.emit(self.current)
            return self.current
        return self._start(payload)

    def _start(self, payload: dict):
        previous = self.current
        task_id = stamp()
        task = {
            "id": task_id,
            "kind": payload["kind"],
            "status": TaskState.PREPARING,
            "created": now(),
            "step": 0,
            "total": 0,
            "message": t("tasks.preparing"),
            "sequence": 0,
            "project": payload.get("project"),
            "run": payload.get("run"),
        }
        if payload["kind"] == "resources":
            task["action"] = resource_action(payload)
        directory = self.task_dir(task)
        directory.mkdir(parents=True)
        payload = {**payload, "resource_root": str(self.store.root / "resources")}
        write_json(directory / "request.json", payload)
        write_json(directory / "task.json", task)
        self.tasks.insert(0, task)
        self.current = task
        adopted = (
            None
            if self.process or not previous or not previous.get("pid")
            else self._owned_process(previous)
        )
        if self.process and self.resident:
            if payload["kind"] in {"generate", "preview"}:
                self.process.write((json.dumps({"task": str(directory)}) + "\n").encode())
                # Output that follows belongs to the task the resident worker now runs.
                self.process.setProperty("task_directory", str(directory))
                task["pid"] = self.process.processId()
                self.changed.emit(task)
                return task
            self.pending = (task, payload)
            self.process.write(b'{"command":"shutdown"}\n')
            self.resident = False
            task["message"] = t("tasks.releasing_inference_resources")
        elif self.process or adopted:
            self.pending = (task, payload)
            task["message"] = t("tasks.waiting_for_the_compute_process_to_release")
            if adopted:
                self.handoff = (adopted, time.monotonic() + HANDOFF_TIMEOUT)
                self.handoff_timer.start()
        else:
            self._launch(task, payload)
        self.changed.emit(task)
        return task

    @Slot()
    def _check_handoff(self):
        if self.pending is None or self.handoff is None:
            self.handoff = None
            self.handoff_timer.stop()
            return
        process, deadline = self.handoff
        try:
            alive = process.is_running() and process.status() != "zombie"
        except Exception:
            alive = False
        if alive and time.monotonic() < deadline:
            return
        self.handoff = None
        self.handoff_timer.stop()
        task, payload = self.pending
        self.pending = None
        self._launch(task, payload)

    def _launch(self, task, payload):
        directory = self.task_dir(task)
        process = QProcess(self)
        self.process = process
        env = QProcessEnvironment.systemEnvironment()
        for key, value in {
            "IRODORI_TASK_ID": task["id"],
            "IRODORI_TASK_EVENTS": str(directory / "events.jsonl"),
            "IRODORI_TASK_STOP": str(directory / "stop"),
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }.items():
            env.insert(key, value)
        # Backends whose packages conflict with the application's run with their add-on in front.
        inherited = {"PYTHONPATH": env.value("PYTHONPATH")} if env.contains("PYTHONPATH") else {}
        for key, value in launch_environment(payload, inherited).items():
            env.insert(key, value)
        self.resident = payload["kind"] in {"generate", "preview"}
        # Generating only reads what the resource task verified. Online, silentcipher would
        # fetch the newest revision of its whole repository instead of the pinned files.
        if self.resident:
            env.insert("HF_HUB_OFFLINE", "1")
        process.setProcessEnvironment(env)
        process.setWorkingDirectory(str(self.repo))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.setProperty("task_id", task["id"])
        process.setProperty("task_directory", str(directory))
        # QObject slots have an owner context; teardown disconnects them before
        # child-process destruction can call Python with a deleted wrapper.
        process.readyReadStandardOutput.connect(self._read_log)
        process.finished.connect(self._exited)
        process.errorOccurred.connect(self._process_error)
        process.started.connect(self._started)
        python = Path(sys.executable)
        pythonw = python.with_name("pythonw.exe")
        # pythonw and QProcess pipes require no console window on Windows.
        process.setProgram(str(pythonw if os.name == "nt" and pythonw.exists() else python))
        process.setArguments(
            ["-m", "irodori_tts.desktop.worker", str(directory)]
            + (["--serve"] if self.resident else [])
        )
        process.start()

    @Slot()
    def _started(self):
        process = self.sender()
        task = next(task for task in self.tasks if task["id"] == process.property("task_id"))
        task["pid"] = process.processId()
        write_json(self.task_dir(task) / "task.json", task)

    @Slot()
    def _read_log(self):
        process = self.sender()
        # Each process writes to the task it runs, also while a pending task waits for it.
        directory = Path(process.property("task_directory"))
        data = bytes(process.readAllStandardOutput())
        with (directory / "worker.log").open("ab") as stream:
            stream.write(data)

    @Slot(QProcess.ProcessError)
    def _process_error(self, error):
        process = self.sender()
        if error == QProcess.ProcessError.FailedToStart and self.current:
            self._finish(TaskState.FAILED, process.errorString())
            # FailedToStart does not emit finished. Release it here so an interlude
            # can recover training instead of waiting forever for that signal.
            if self.process is process:
                self.process = None
                self.resident = False
            process.deleteLater()

    @Slot(int, QProcess.ExitStatus)
    def _exited(self, code, status):
        process = self.sender()
        self.poll()
        process.deleteLater()
        if self.process is process:
            self.process = None
            self._released(True)
        if self.pending:
            task, payload = self.pending
            self.pending = None
            self._launch(task, payload)
            return
        self.resident = False
        if self.busy:
            self._finish(
                TaskState.INTERRUPTED,
                t("tasks.compute_process_exited_recover_from_the_latest", p1=fmt(code, "", -1)),
            )

    @Slot()
    def poll(self):
        if not self.current:
            return
        task = self.current
        path = self.task_dir(task) / "events.jsonl"
        # A file copy reports faster than the window can draw. Everything read in one pass
        # is shown once: per event, the GUI thread would never get back to the event loop.
        updated = stepped = False
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(self.offsets.get(task["id"], 0))
                while line := stream.readline():
                    if not line.endswith(b"\n"):
                        break
                    self.offsets[task["id"]] = stream.tell()
                    event = restore(json.loads(line))
                    if event["task_id"] != task["id"] or event["sequence"] <= task["sequence"]:
                        continue
                    task["sequence"] = event["sequence"]
                    kind = event["type"]
                    if kind == "result":
                        updated = False
                        task["result"] = event["result"]
                        result_status = event["result"].get("status")
                        self._finish(
                            TaskState.STOPPED
                            if result_status == "stopped"
                            else TaskState.COMPLETED,
                            t("tasks.task_complete"),
                        )
                    elif kind in {"error", "cancelled"}:
                        # Where it stopped: the step and the file, when the worker knows them.
                        for key in ("phase", "item", "items"):
                            if key in event:
                                task[key] = event[key]
                        updated = False
                        self._finish(
                            TaskState.FAILED if kind == "error" else TaskState.STOPPED,
                            event.get("message", ""),
                        )
                    elif kind == "stopped":
                        task.update(
                            step=event["step"], message=t("tasks.complete_checkpoint_saved")
                        )
                    else:
                        if (
                            kind == "progress"
                            and self.training_preview
                            and self.training_preview.phase == TrainingPreviewPhase.RESUMING
                            and task["id"] == self.training_preview.task_id
                        ):
                            self.training_preview = None
                        if (
                            kind in {"running", "saving"}
                            and task["status"] != TaskState.STOP_REQUESTED
                        ):
                            task["status"] = kind
                        for key in ("step", "total", "message", "loss", "phase", "item"):
                            if key in event:
                                task[key] = event[key]
                        stepped = stepped or "step" in event
                        updated = True
        if updated:
            if stepped:
                queue = self.samples.setdefault(task["id"], collections.deque(maxlen=200))
                if not queue or queue[-1][1] != task["step"]:
                    queue.append((time.monotonic(), task["step"]))
            write_json(self.task_dir(task) / "task.json", task)
            self.changed.emit(task)
        if (
            self.busy
            and self.process is None
            and self.pending is None
            and not self._owned_process(task)
        ):
            self._finish(
                TaskState.INTERRUPTED, t("tasks.previous_compute_process_ended_check_the_complete")
            )

    def _finish(self, status, message):
        if self.current is None or self.current["status"] not in ACTIVE:
            return
        task = self.current
        task.update(status=status, message=message, ended=now())
        write_json(self.task_dir(task) / "task.json", task)
        self.changed.emit(task)
        # Avoid launching a new process from inside the old process's event iteration.
        self.finished_tasks.append(task)
        self.finish_timer.start(0)

    @Slot()
    def _deliver_finished(self):
        tasks, self.finished_tasks = self.finished_tasks, []
        for task in tasks:
            if task["kind"] == "preview" and task["status"] != TaskState.COMPLETED:
                self._discard_snapshot(read_json(self.task_dir(task) / "request.json", {}))
            try:
                self._advance_training_preview(task)
            except Exception as exc:
                interlude, self.training_preview = self.training_preview, None
                if interlude and interlude.task_id is None:
                    self._discard_snapshot(interlude.preview)
                # The task itself ended as recorded; only the automatic continuation failed.
                task.update(
                    message=t("tasks.preview_transition_failed", reason=error_message(exc)),
                    preview_error=t("tasks.preview_transition_failed", reason=error_message(exc)),
                )
                write_json(self.task_dir(task) / "task.json", task)
                self.changed.emit(task)
            self.finished.emit(task)
            self._continue_chain(task)

    def _continue_chain(self, task):
        """A task may carry the one that follows it (installing an add-on before the work that
        needs it). It starts only when this one completed."""
        if task["status"] != TaskState.COMPLETED or self.busy:
            return
        follow = (read_json(self.task_dir(task) / "request.json", {}) or {}).get("then")
        if follow:
            self.start(follow)

    @staticmethod
    def _discard_snapshot(payload):
        """Remove the adapter pinned for a sample that will not be produced."""
        snapshot, project = payload.get("checkpoint_snapshot"), payload.get("project")
        if snapshot and project:
            try:
                discard_preview_snapshot(Path(project), Path(snapshot))
            except (OSError, ValueError):
                pass

    def _advance_training_preview(self, task):
        interlude = self.training_preview
        if interlude is None:
            return
        if interlude.phase == TrainingPreviewPhase.PAUSING:
            if task["id"] != interlude.training_task_id:
                return
            result = task.get("result", {})
            checkpoint = result.get("checkpoint")
            if (
                task["status"] not in {TaskState.STOPPED, TaskState.COMPLETED}
                or not checkpoint
                or not complete_checkpoint(
                    Path(interlude.training["project"]) / checkpoint, require_trainer=True
                )
            ):
                self.training_preview = None
                self._discard_snapshot(interlude.preview)
                if task["status"] in {TaskState.STOPPED, TaskState.COMPLETED}:
                    task.update(status=TaskState.FAILED, message=t("tasks.preview_pause_failed"))
                    write_json(self.task_dir(task) / "task.json", task)
                    self.changed.emit(task)
                return
            interlude.resume = task["status"] == TaskState.STOPPED
            interlude.phase = TrainingPreviewPhase.GENERATING
            preview = self._start(interlude.preview)
            interlude.task_id = preview["id"]
        elif task["id"] == interlude.task_id:
            if interlude.phase == TrainingPreviewPhase.GENERATING and interlude.resume:
                interlude.phase = TrainingPreviewPhase.RESUMING
                resumed = self._start(interlude.training)
                interlude.task_id = resumed["id"]
                if task["status"] != TaskState.COMPLETED:
                    resumed["preview_error"] = task["message"]
                    self.changed.emit(resumed)
            else:
                self.training_preview = None
                self.changed.emit(task)

    def eta_seconds(self, task):
        """Seconds left, from the recent step rate of a running task; None until it is known."""
        queue = self.samples.get(task["id"])
        total = task.get("total") or 0
        if task["status"] != TaskState.RUNNING or not queue or len(queue) < 3 or not total:
            return None
        (first_time, first_step), (last_time, last_step) = queue[0], queue[-1]
        if last_step <= first_step or last_time <= first_time:
            return None
        return max(0.0, (total - last_step) * (last_time - first_time) / (last_step - first_step))

    def rate(self, task, window=15.0):
        """Steps per second over the last seconds of a running task (MB/s for downloads),
        or None until two samples are known."""
        queue = self.samples.get(task["id"])
        if not queue or task["status"] != TaskState.RUNNING:
            return None
        last_time, last_step = queue[-1]
        recent = [(when, step) for when, step in queue if last_time - when <= window]
        first_time, first_step = recent[0]
        if last_time - first_time < 1.0 or last_step <= first_step:
            return None
        return (last_step - first_step) / (last_time - first_time)

    def loss_points(self, task):
        """(step, loss) pairs from a task's event log, read incrementally and cached."""
        state = self.series.setdefault(task["id"], {"offset": 0, "points": []})
        path = self.task_dir(task) / "events.jsonl"
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(state["offset"])
                while line := stream.readline():
                    if not line.endswith(b"\n"):
                        break
                    state["offset"] = stream.tell()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("type") == "progress" and "loss" in event:
                        state["points"].append((event["step"], event["loss"]))
        return state["points"]

    def run_tasks(self, run_root):
        return [
            task
            for task in self.tasks
            if task["kind"] == "train" and task.get("run") == str(run_root)
        ]

    def run_series(self, run_root):
        """Losses of every task that trained this run (a resumed run has several), by step."""
        merged = {}
        for task in reversed(self.run_tasks(run_root)):
            merged.update(self.loss_points(task))
        return sorted(merged.items())

    def run_elapsed(self, run_root):
        """Seconds spent training this run, summed over its tasks."""
        total = 0.0
        for task in self.run_tasks(run_root):
            if not task.get("created"):
                continue
            start = datetime.fromisoformat(task["created"])
            if task.get("ended"):
                end = datetime.fromisoformat(task["ended"])
            elif task["status"] in ACTIVE:
                end = datetime.now(timezone.utc)
            else:
                continue
            total += max(0.0, (end - start).total_seconds())
        return total

    def _cancel_interlude(self):
        interlude, self.training_preview = self.training_preview, None
        if interlude and interlude.task_id is None:
            # Still pausing: the sample task does not exist yet to clean up after itself.
            self._discard_snapshot(interlude.preview)

    def stop(self):
        if self.training_preview and self.current and self.current["kind"] == "train":
            self.current["preview_pause"] = True
        self._cancel_interlude()
        self._request_stop()

    def _request_stop(self):
        if self.pending:
            # GPU handoff may still be waiting for the previous process to exit.
            # Cancelling here must remove the launch, not just stop its future worker.
            self.pending = None
            self.handoff = None
            self.current["preview_pause"] = True
            self._finish(TaskState.STOPPED, t("tasks.stopped"))
            return
        if self.busy:
            (self.task_dir() / "stop").touch()
            self.current.update(
                status=TaskState.STOP_REQUESTED,
                message=t("tasks.waiting_for_a_safe_compute_boundary"),
            )
            write_json(self.task_dir() / "task.json", self.current)
            self.changed.emit(self.current)

    def force_stop(self):
        self._cancel_interlude()
        if self.pending:
            self.pending = None
            self.handoff = None
            self._finish(TaskState.INTERRUPTED, t("tasks.stopped"))
        if self.process:
            self.process.kill()
        elif self.current:
            owned = self._owned_process(self.current)
            if owned:
                owned.terminate()

    def release(self, done):
        """Ask an idle compute process to exit, which lets go of the files it holds open. The
        window keeps running meanwhile; done(released) says how it went."""
        if self.busy:
            done(False)
        elif not self.process:
            done(True)
        else:
            self.releasing.append(done)
            if len(self.releasing) > 1:
                return
            # A task started while it exits waits for it (pending) instead of writing to it.
            self.resident = False
            self.release_timer.start()
            self.process.write(b'{"command":"shutdown"}\n')
            self.process.closeWriteChannel()

    def _released(self, released):
        self.release_timer.stop()
        waiting, self.releasing = self.releasing, []
        for done in waiting:
            done(released)
