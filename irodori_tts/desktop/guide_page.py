"""Getting started: everything the application needs from this computer, as one checklist.

A user may stay here a long while, as models download, so the top of the page always says
what is still missing and holds the one button that gets it; each check and each model has a
line of its own with its state, its progress, the reason when it failed and the way on.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
from qfluentwidgets import FluentIcon, ProgressBar

from irodori_tts.desktop.messages import BusinessError, fmt, t

from . import asr, model_files
from .asr_dialog import human_size
from .connection import check_hugging_face
from .i18n import bind, set_text
from .resources import cache_root, hf_home, paths_too_long
from .training_page import rough_duration
from .widgets import (
    Page,
    SplitRow,
    accent_drawn,
    button,
    card,
    card_header,
    choose_folder,
    elided_label,
    ghost_button,
    label,
    open_file,
    row,
    save_file,
    show_menu,
    step_icon,
)

# Tasks whose progress the page shows while they run.
GUIDE_TASKS = {"resources", "models", "addon", "import_models", "export_models"}
TIERS = (
    ("required", t("guide.tier_required")),
    ("recommended", t("guide.tier_recommended")),
    ("optional", t("guide.tier_optional")),
)


class CheckRow(QWidget):
    """One line of the checklist: a state marker, a title with a short value at its right, a
    sentence under it (wrapped), an optional progress bar, and one action."""

    def __init__(self, title, *, progress=False):
        super().__init__()
        self.marker = QLabel()
        self.marker.setFixedSize(20, 20)
        self.title = elided_label(title, emphasis=True)
        self.value = label("", "muted", wrap=False)
        self.detail = label("", "muted")
        self.problem = label("", "error")
        self.problem.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.problem.hide()
        self.bar = ProgressBar() if progress else None
        self.bar_text = elided_label() if progress else None
        self.callback = None
        self.action = button("", lambda: self.callback and self.callback())
        self.action.hide()
        lines = QVBoxLayout()
        lines.setSpacing(3)
        lines.addWidget(row((self.title, 1), self.value))
        lines.addWidget(self.detail)
        lines.addWidget(self.problem)
        if self.bar:
            self.bar.setFixedHeight(4)
            self.bar.hide()
            self.bar_text.hide()
            lines.addSpacing(2)
            lines.addWidget(self.bar)
            lines.addWidget(self.bar_text)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(12)
        layout.addWidget(self.marker, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(lines, 1)
        layout.addWidget(self.action, 0, Qt.AlignmentFlag.AlignVCenter)

    def show_state(self, state, detail="", *, value="", problem="", action=None):
        """state: done, warn, failed, active or pending. action: (text, callback) or None."""
        self.state = state
        accent_drawn(
            self.marker,
            lambda: self.marker.setPixmap(step_icon(None, state, size=20).pixmap(20, 20)),
        )
        set_text(self.detail, detail)
        self.detail.setVisible(bool(str(detail)))
        set_text(self.value, value)
        set_text(self.problem, problem)
        self.problem.setVisible(bool(str(problem)))
        self.action.setVisible(action is not None)
        if action is not None:
            set_text(self.action, action[0])
            self.callback = action[1]

    def show_progress(self, done=None, total=None, text=""):
        if self.bar is None:
            return
        shown = done is not None
        self.bar.setVisible(shown)
        self.bar_text.setVisible(shown and bool(str(text)))
        if shown:
            self.bar.setRange(0, max(1, total))
            self.bar.setValue(min(done, total))
            set_text(self.bar_text, text)


class GuidePage(QWidget):
    def __init__(self, host):
        super().__init__()
        self.host = host
        self.rows = {}
        self.task_state = None
        self.memory = None
        self.stale = True
        self.connection_result = None
        self.connection_checking = False
        # The downloads a running task (chain) is fetching, why one failed, and the stopped ones.
        self.downloading = set()
        self.failures = {}
        self.paused = set()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        top = QWidget()
        head = QVBoxLayout(top)
        head.setContentsMargins(24, 20, 24, 0)
        head.setSpacing(14)
        head.addWidget(
            row(
                label(t("guide.title"), "page", wrap=False),
                None,
                ghost_button(
                    t("tour.restart"), lambda: self.host.start_tour(), icon=FluentIcon.FLAG
                ),
            )
        )
        # The summary stays at the top, outside the scrolled checklist: what is missing and
        # the button that gets it are in view at any window size.
        self.build_summary(head)
        outer.addWidget(top)
        self.scroll = Page()
        self.scroll.body.setContentsMargins(24, 14, 24, 20)
        outer.addWidget(self.scroll, 1)
        left, right = QWidget(), QWidget()
        for column, build in ((left, self.build_computer), (right, self.build_models)):
            lines = QVBoxLayout(column)
            lines.setContentsMargins(0, 0, 0, 0)
            lines.setSpacing(14)
            build(lines)
            lines.addStretch(1)
        self.scroll.body.addWidget(SplitRow(left, right, spacing=14, stretch=(2, 3)), 1)

    # ----- summary -----

    def build_summary(self, parent):
        _frame, body = card(parent)
        self.summary_icon = QLabel()
        self.summary_icon.setFixedSize(28, 28)
        self.summary_title = elided_label(strong=True)
        self.summary_text = label("", "muted")
        texts = QVBoxLayout()
        texts.setSpacing(2)
        texts.addWidget(self.summary_title)
        texts.addWidget(self.summary_text)
        lead = QWidget()
        lead_layout = QHBoxLayout(lead)
        lead_layout.setContentsMargins(0, 0, 0, 0)
        lead_layout.setSpacing(12)
        lead_layout.addWidget(self.summary_icon, 0, Qt.AlignmentFlag.AlignTop)
        lead_layout.addLayout(texts, 1)
        self.import_button = button(
            t("guide.import_files"), self.choose_import, icon=FluentIcon.FOLDER
        )
        self.secondary_button = button(t("guide.go_generate"), lambda: self.host.navigate(1))
        self.download_button = button(
            t("guide.nothing_to_download"),
            lambda: self.host.perform(self.primary),
            primary=True,
            icon=FluentIcon.DOWNLOAD,
        )
        self.pause_button = button(t("guide.pause"), self.host.tasks.stop, icon=FluentIcon.PAUSE)
        actions = row(self.import_button, self.secondary_button, self.pause_button)
        actions.layout().addWidget(self.download_button)
        body.addWidget(SplitRow(lead, actions))
        self.progress = ProgressBar()
        self.progress.setFixedHeight(4)
        self.progress_text = elided_label()
        self.progress_box = QWidget()
        lines = QVBoxLayout(self.progress_box)
        lines.setContentsMargins(0, 2, 0, 0)
        lines.setSpacing(6)
        lines.addWidget(self.progress)
        lines.addWidget(self.progress_text)
        lines.addWidget(label(t("guide.background_note"), "muted"))
        self.progress_box.hide()
        body.addWidget(self.progress_box)

    def guide_task(self):
        task = self.host.tasks.current
        return task if task and self.host.tasks.busy and task["kind"] in GUIDE_TASKS else None

    def todo(self):
        """What still needs doing, as (models to download, blocking computer checks)."""
        missing = self.chosen()
        blocking = []
        report = self.host.device
        if report is not None and not report.get("gpu"):
            blocking.append("gpu")
        needed = sum(item.missing_size() for item in missing)
        if needed > model_files.disk_free(hf_home()):
            blocking.append("disk")
        return missing, blocking

    def complete(self):
        missing, blocking = self.todo()
        return not missing and not blocking

    def refresh_summary(self):
        task = self.guide_task()
        busy = self.host.tasks.busy
        missing, blocking = self.todo()
        needed = sum(item.missing_size() for item in missing)
        free = model_files.disk_free(hf_home())
        if task:
            state, title = "active", t("guide.summary_downloading")
            text = task["message"]
        elif not missing and not blocking:
            state, title, text = "done", t("guide.summary_ready"), t("guide.summary_ready_hint")
        else:
            count = len(missing) + len(blocking)
            state, title = "warn", t("guide.summary_todo", n=count)
            text = (
                t("guide.summary", needed=human_size(needed), free=human_size(free))
                if missing
                else t("guide.summary_blocking")
            )
        accent_drawn(
            self.summary_icon,
            lambda: self.summary_icon.setPixmap(step_icon(None, state, size=28).pixmap(28, 28)),
        )
        set_text(self.summary_title, title)
        set_text(self.summary_text, text)
        ready = not missing and model_files.speech_ready()
        self.pause_button.setVisible(task is not None and task["kind"] in {"resources", "models"})
        self.download_button.setVisible(task is None)
        self.download_button.setEnabled(not busy and (bool(missing) or ready))
        set_text(
            self.download_button,
            t("guide.download", size=human_size(needed))
            if missing
            else t("models_page.new_model")
            if ready
            else t("guide.nothing_to_download"),
        )
        self.download_button.setIcon(FluentIcon.DOWNLOAD if missing else FluentIcon.ADD)
        bind(
            self.download_button,
            t("asr.wait_for_the_running_task") if busy else t("guide.download_tooltip"),
            "setToolTip",
        )
        self.secondary_button.setVisible(ready and task is None)
        self.import_button.setVisible(not ready and task is None)
        self.import_button.setEnabled(not busy)
        self.progress_box.setVisible(task is not None)
        if task:
            total = task.get("total") or 0
            self.progress.setRange(0, max(1, total))
            self.progress.setValue((task.get("step") or 0) if total else 0)
            set_text(self.progress_text, self.speed_text(task) or task["message"])

    def primary(self):
        missing, _blocking = self.todo()
        if missing:
            self.download()
        elif model_files.speech_ready():
            self.host.new_project()

    def speed_text(self, task):
        """'1.20 / 3.40 GB · 12.3 MB/s · about 3 min left' for a download, else ""."""
        total = task.get("total") or 0
        if task["kind"] not in {"resources", "models", "import_models", "export_models"}:
            return ""
        if not total:
            return ""
        done = task.get("step") or 0
        text = t(
            "guide.progress_size",
            done=fmt(done / 1000, ".2f"),
            total=fmt(total / 1000, ".2f"),
        )
        rate = self.host.tasks.rate(task)
        if rate:
            text = t("guide.progress_rate", text=text, rate=fmt(rate, ".1f"))
            remaining = (total - done) / rate
            text = t("guide.progress_left", text=text, left=fmt(rough_duration(remaining), "", -1))
        return text

    # ----- this computer -----

    def build_computer(self, parent):
        _frame, body = card(parent)
        body.setSpacing(4)
        self.check_button = button(t("guide.check_again"), self.check_all, icon=FluentIcon.SYNC)
        body.addWidget(card_header(t("guide.computer"), self.check_button))
        self.gpu = CheckRow(t("guide.check_gpu"))
        self.disk = CheckRow(t("guide.check_disk"))
        self.path = CheckRow(t("guide.check_path"))
        self.network = CheckRow(t("guide.connection"))
        for check in (self.gpu, self.disk, self.path, self.network):
            body.addWidget(check)

    def check_all(self):
        self.host.detect_device()
        self.check_connection()
        self.refresh()

    def refresh_computer(self):
        report = self.host.device
        if report is None:
            self.gpu.show_state("pending", t("guide.checking"))
        elif report.get("gpu"):
            headline, detail, _short = self.host.device_lines()
            self.gpu.show_state(
                "done",
                t("common.task_state", name=fmt(headline, "", -1), state=fmt(detail, "", -1)),
            )
        else:
            self.gpu.show_state(
                "failed",
                t("guide.no_gpu_hint"),
                value=t("app.no_gpu"),
                problem=report.get("gpu_error") or "",
                action=(t("guide.check_again"), self.host.detect_device),
            )
        free = model_files.disk_free(hf_home())
        needed = sum(item.missing_size() for item in self.chosen())
        where = fmt(hf_home().parent, "", -1)
        if needed > free:
            self.disk.show_state(
                "failed",
                t("guide.disk_free", size=human_size(free), path=where),
                value=human_size(free),
                problem=t("guide.disk_short", needed=human_size(needed)),
            )
        else:
            self.disk.show_state("done", t("guide.disk_free", size=human_size(free), path=where))
        crowded = paths_too_long(self.host.store.root, cache_root(self.host.repo))
        if crowded:
            self.path.show_state("warn", t("app.path_too_long", length=len(str(crowded[0]))))
            self.path.setToolTip(str(crowded[0]))
        else:
            self.path.show_state("done", t("guide.path_ok"))
            self.path.setToolTip("")
        self.refresh_connection()

    def check_connection(self):
        if self.connection_checking:
            return
        self.connection_checking = True
        self.connection_result = None
        self.check_button.setEnabled(False)
        self.refresh_connection()
        self.host.run_io(check_hugging_face, self.show_connection, None, blocking=False)

    def show_connection(self, result):
        self.connection_checking = False
        self.check_button.setEnabled(True)
        self.connection_result = result
        self.refresh_connection()

    def refresh_connection(self):
        result = self.connection_result
        if result is None:
            self.network.show_state("pending", t("guide.checking"))
        elif result["reachable"]:
            problem = result.get("status") not in (None, "operational")
            self.network.show_state(
                "warn" if problem else "done",
                t("guide.reachable", ms=result["latency_ms"]),
                problem=t("guide.official_problem") if problem else "",
            )
        else:
            self.network.show_state(
                "failed",
                t("guide.offline_hint"),
                problem=t("guide.unreachable", reason=result["reason"]),
                action=(t("guide.import_files"), lambda: self.choose_import(self.network.action)),
            )

    # ----- models -----

    def build_models(self, parent):
        _frame, body = card(parent)
        body.setSpacing(4)
        body.addWidget(card_header(t("guide.models")))
        for tier, title in TIERS:
            body.addSpacing(6)
            body.addWidget(label(title, "muted", wrap=False))
            for item in model_files.downloads():
                if item.tier == tier:
                    check = CheckRow(item.title, progress=True)
                    self.rows[item.key] = check
                    body.addWidget(check)
        body.addSpacing(6)
        self.no_transcription = label(t("guide.no_transcription"), "muted")
        body.addWidget(self.no_transcription)
        body.addWidget(label(t("guide.import_hint"), "muted"))

    def recommended(self):
        """Recommended transcription models (and the configured defaults) that fit the card.
        When one does not fit, the most accurate ones that do take its place, two at most."""
        main, check = asr.choice(self.host.store.settings)
        items = [i for i in model_files.downloads() if i.backend]
        wanted = [i for i in items if i.tier == "recommended" or i.key in {main, check}]
        chosen = [i.key for i in wanted if model_files.fits(i, self.memory)]
        for item in items:
            if len(chosen) >= 2:
                break
            if item.key not in chosen and model_files.fits(item, self.memory):
                chosen.append(item.key)
        return set(chosen)

    def device_known(self, report):
        gpu = (report or {}).get("gpu")
        self.memory = model_files.card_memory_gb(gpu["total_mib"]) if gpu else None
        self.refresh()

    def chosen(self):
        """What the main button downloads: the required models and the recommended ones that
        fit the card, if they are not downloaded yet."""
        wanted = self.recommended()
        return [
            item
            for item in model_files.downloads()
            if not item.installed() and (item.tier == "required" or item.key in wanted)
        ]

    def refresh_models(self):
        task = self.guide_task()
        busy = self.host.tasks.busy
        for item in model_files.downloads():
            self.refresh_model(item, task, busy)
        transcription = [i for i in model_files.downloads() if i.backend and i.installed()]
        self.no_transcription.setVisible(not transcription)

    def refresh_model(self, item, task, busy, installed=None):
        check = self.rows[item.key]
        installed = item.installed() if installed is None else installed
        size = human_size(item.size)
        current = bool(task and task.get("item") == item.key)
        if installed:
            check.show_state("done", item.reason, value=t("guide.downloaded", size=size))
            check.show_progress()
            return
        if current:
            check.show_state(
                "active",
                item.reason,
                value=t("guide.downloading"),
                action=(t("guide.pause"), self.host.tasks.stop),
            )
            total = task.get("total") or 0
            check.show_progress(task.get("step") or 0, total, self.speed_text(task))
            return
        check.show_progress()
        if item.key in self.downloading and task:
            check.show_state("pending", item.reason, value=t("guide.queued"))
            return
        fits = model_files.fits(item, self.memory)
        problem = self.failures.get(item.key, "")
        text = (
            t("guide.retry")
            if problem
            else t("guide.resume")
            if item.key in self.paused
            else t("guide.download_one")
        )
        action = (text, lambda key=item.key: self.host.perform(lambda: self.download([key])))
        check.action.setEnabled(not busy)
        value = (
            size
            if fits
            else t(
                "guide.too_large",
                size=size,
                need=fmt(f"{asr.backend_for(item.backend).vram:.0f}"),
            )
        )
        state = "failed" if problem else "pending" if fits else "warn"
        check.show_state(state, item.reason, value=value, problem=problem, action=action)
        check.action.setEnabled(not busy)

    # ----- actions -----

    def remember_transcription(self):
        """When the default transcription models are not downloaded, the downloaded ones take
        their place: the most accurate one writes, the next one checks."""
        installed = [
            backend.key
            for backend in asr.BACKENDS
            if model_files.download_for(backend.key).installed()
        ]
        main, check = asr.choice(self.host.store.settings)
        if not installed or (main in installed and (check == asr.NO_CHECK or check in installed)):
            return
        chosen = installed
        self.host.store.settings["asr"] = {
            "main": chosen[0],
            "check": chosen[1] if len(chosen) > 1 else asr.NO_CHECK,
        }
        self.host.store.save()
        self.host.settings_page.show_asr_choice()

    def download(self, keys=None):
        """Download these models, or the required and recommended ones; one task chain."""
        if self.host.tasks.busy:
            raise BusinessError(t("asr.wait_for_the_running_task"))
        if keys is None:
            items = self.chosen()
        else:
            items = [model_files.download_for(key) for key in keys]
            items = [item for item in items if not item.installed()]
        if not items:
            return
        self.downloading = {item.key for item in items}
        for item in items:
            self.failures.pop(item.key, None)
            self.paused.discard(item.key)
        backends = [asr.backend_for(item.backend) for item in items if item.backend]
        payload = None
        keys = [b.key for b in backends]
        keys += [item.key for item in items if item.key in model_files.GENERATION_MODELS]
        if keys:
            payload = {"kind": "models", "backends": keys}
        if any(item.key == model_files.SPEECH for item in items):
            payload = {"kind": "resources", **({"then": payload} if payload else {})}
        # A source checkout installs add-ons here; a package already carries them.
        if missing := asr.missing_addons(backends):
            payload = {"kind": "addon", "keys": missing, **({"then": payload} if payload else {})}
        self.host.tasks.start(payload)
        self.refresh()

    def choose_import(self, anchor=None):
        """Zip from another user's export, or a folder of model files (a copied download
        cache included)."""
        if not isinstance(anchor, QWidget):
            anchor = self.import_button
        show_menu(
            anchor,
            [
                (t("guide.import_zip"), self.import_zip, True),
                (t("guide.import_folder"), self.import_folder, True),
            ],
        )

    def import_zip(self):
        path = open_file(
            self, self.host.store.settings, t("guide.import_files"), t("guide.package_filter")
        )
        if path:
            self.host.perform(lambda: self.start_import(Path(path)))

    def import_folder(self):
        path = choose_folder(self, self.host.store.settings, t("guide.import_files"))
        if path:
            self.host.perform(lambda: self.start_import(Path(path)))

    def start_import(self, source):
        if self.host.tasks.busy:
            raise BusinessError(t("asr.wait_for_the_running_task"))
        self.host.tasks.start({"kind": "import_models", "source": str(source)})

    def export(self):
        """Every downloaded model into one zip, for someone who cannot download."""
        if self.host.tasks.busy:
            raise BusinessError(t("asr.wait_for_the_running_task"))
        items = [item for item in model_files.downloads() if item.installed()]
        if not items:
            raise BusinessError(t("models.nothing_to_export"))
        size = sum(item.size for item in items)
        names = fmt("\n".join(f"· {item.title}" for item in items), "", -1)
        if not self.host.confirm(
            t("models.export_title"),
            t("models.export_message", names=names, size=human_size(size)),
            t("models.export_action"),
        ):
            return
        path = save_file(
            self,
            self.host.store.settings,
            t("models.export_title"),
            "irodori-models.zip",
            t("guide.zip_filter"),
        )
        if path:
            self.host.tasks.start(
                {
                    "kind": "export_models",
                    "keys": [item.key for item in items],
                    "destination": str(path),
                }
            )

    # ----- tasks -----

    def task_changed(self, task):
        if self.stale:
            return
        state = (task["id"], task["status"])
        step, self.task_state = state == self.task_state, state
        if step and task["kind"] not in GUIDE_TASKS:
            return
        guide = task["kind"] in GUIDE_TASKS and self.host.tasks.busy
        if guide and task.get("item") in self.rows:
            # A progress step: only the line being downloaded and the summary change.
            item = model_files.download_for(task["item"])
            self.refresh_model(item, task, True, installed=False)
        elif not (step and task["kind"] in {"import_models", "export_models"}):
            # Copying packages names no line; the lines change when it ends.
            self.refresh_models()
        self.refresh_summary()

    def task_finished(self, task):
        if task["kind"] not in GUIDE_TASKS:
            return
        unfinished = [
            key for key in self.downloading if not model_files.download_for(key).installed()
        ]
        if task["status"] in {"failed", "interrupted"}:
            failed = task.get("item") if task.get("item") in self.rows else None
            for key in [failed] if failed else unfinished:
                self.failures[key] = task["message"]
            if not unfinished and not failed:
                self.host.notify(task["message"], error=True)
            self.downloading = set()
        elif task["status"] == "stopped":
            self.paused |= set(unfinished)
            self.downloading = set()
        else:
            self.downloading = set(unfinished)
            self.remember_transcription()
        if task["kind"] == "export_models" and task["status"] == "completed":
            result = task.get("result") or {}
            self.host.notify(
                t("models.exported", path=fmt(result.get("destination", ""), "", -1)), duration=-1
            )
        if task["kind"] == "import_models" and task["status"] == "completed":
            result = task.get("result") or {}
            names = [str(model_files.download_for(key).title) for key in result.get("complete", [])]
            note = t(
                "guide.import_result",
                imported=len(result.get("imported", [])),
                present=len(result.get("present", [])),
                rejected=len(result.get("rejected", [])),
                complete=fmt("、".join(names) or "—", "", -1),
            )
            if result.get("rejected"):
                first, why = result["rejected"][0]
                note = t(
                    "common.concatenate",
                    p0=note,
                    p1=t("guide.import_rejected", name=fmt(first, "", -1), reason=fmt(why, "", -1)),
                )
            self.host.notify(note, error=bool(result.get("rejected")), duration=-1)
        self.refresh(force=True)

    def showEvent(self, event):
        super().showEvent(event)
        if self.stale:
            self.refresh()
        # Opened from the navigation, not through show_guide: the connection was never checked
        # and the line would say "Checking" for good.
        if self.connection_result is None:
            self.check_connection()

    def refresh(self, force=False):
        # Model states, disk space and device lines are read from disk: only for a page in view.
        self.stale = not (force or self.isVisible())
        if self.stale:
            return
        self.refresh_computer()
        self.refresh_models()
        self.refresh_summary()
