from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    FluentIcon,
    FluentWindow,
    IndeterminateProgressBar,
    InfoBar,
    InfoBarPosition,
    NavigationItemPosition,
    ProgressBar,
    TeachingTipTailPosition,
    isDarkTheme,
    qconfig,
)
from qfluentwidgets.common.style_sheet import CustomStyleSheet, setStyleSheet
from shiboken6 import isValid

from irodori_tts.desktop.messages import LANGUAGES, BusinessError, error_message, fmt, t

from . import base_models, model_files
from .exports import cuda_of, device_report, import_project
from .generation_page import GenerationPage
from .guide_page import GuidePage
from .i18n import bind, clear_text, locale_manager, set_text, system_language
from .library_page import ModelsPage
from .resources import (
    adopt_cached_resources,
    cache_root,
    configure_environment,
    hide_child_consoles,
    paths_too_long,
)
from .settings_page import SettingsPage
from .storage import (
    AppStore,
    create_project,
    load_project,
    project_lock,
    read_json,
    reconcile_sources,
    save_project,
    stamp,
    write_json,
)
from .tasks import KIND_TEXT, RESOURCE_TEXT, STATE_TEXT, TaskManager, resource_action
from .tour import Step, Tour
from .training import complete_checkpoint, latest_checkpoint
from .training_page import rough_duration
from .widgets import (
    ICON,
    AppIcon,
    Confirmation,
    Table,
    accent,
    accent_line,
    app_icon,
    apply_theme,
    elided_label,
    ghost_button,
    icon_button,
    label,
    open_file,
    prompt_text,
    set_accent,
    show_menu,
    toast,
)

REPO = Path(__file__).resolve().parents[2]
# The library's StackedWidget panel (qss/{light,dark}/fluent_window.qss), on the content container.
CONTENT_PANEL_QSS = {
    theme: f"""#contentPanel {{
    border: 1px solid {border};
    border-right: none;
    border-bottom: none;
    border-top-left-radius: 10px;
    background-color: {fill};
}}"""
    for theme, border, fill in (
        ("light", "rgba(0, 0, 0, 0.068)", "rgba(255, 255, 255, 0.5)"),
        ("dark", "rgba(0, 0, 0, 0.18)", "rgba(255, 255, 255, 0.0314)"),
    )
}
# AllowSetForegroundWindow: any process may take the foreground next.
ASFW_ANY = -1
# Every page is laid out to work from the minimum size up; a 1080p screen at 150% scaling
# leaves about 1280x688 for a window. A first start opens at the preferred size, shrunk to
# fit the screen with a margin.
MINIMUM_SIZE = (1024, 640)
PREFERRED_SIZE = (1600, 1000)
COMPACT_BELOW = 1366

WM_NCACTIVATE = 0x0086
if sys.platform == "win32":
    import ctypes.wintypes

    _user32 = ctypes.windll.user32
    _def_window_proc = _user32.DefWindowProcW
    _def_window_proc.argtypes = (
        ctypes.wintypes.HWND,
        ctypes.wintypes.UINT,
        ctypes.wintypes.WPARAM,
        ctypes.wintypes.LPARAM,
    )
    _def_window_proc.restype = ctypes.c_ssize_t


def own_window_in_front() -> bool:
    """Whether the foreground window belongs to this process (a dialog of the app)."""
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return False
    pid = ctypes.wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value == os.getpid()


def window_size(saved, available_width, available_height):
    """The saved size, or the preferred one at most 92% of the screen's free area; never
    larger than that area nor smaller than the minimum."""
    width, height = saved or (
        min(PREFERRED_SIZE[0], round(available_width * 0.92)),
        min(PREFERRED_SIZE[1], round(available_height * 0.92)),
    )
    return (
        max(MINIMUM_SIZE[0], min(width, available_width)),
        max(MINIMUM_SIZE[1], min(height, available_height)),
    )


class JobSignals(QObject):
    result = Signal(object)
    error = Signal(object)


class Job(QRunnable):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.signals = JobSignals()

    def run(self):
        try:
            outcome = (self.signals.result, self.operation())
        except Exception as exc:
            outcome = (self.signals.error, error_message(exc))
        signal, value = outcome
        try:
            signal.emit(value)
        except RuntimeError:
            # The window closed while this ran; there is no one left to tell.
            pass


class MainWindow(FluentWindow):
    def __init__(self, store: AppStore):
        # The window's background color is taken at construction and only animates
        # to a later theme, so the theme must be set before the window exists.
        apply_theme(QApplication.instance(), store.settings.get("theme", "system"))
        super().__init__()
        self.locale = locale_manager()
        language = store.settings.get("language")
        if language not in LANGUAGES:
            # First run (or an unknown value): follow the system, and remember the choice.
            language = store.settings["language"] = system_language()
        self.locale.set_language(language)
        self.repo = REPO
        self.store = store
        self.closing_after_stop = False
        project_states = store.settings.get("ui", {}).get("project_states", {})
        self.project_states = project_states if isinstance(project_states, dict) else {}
        self.device = None
        # File work the user started (closing waits for it), and background probes (it does not).
        self.jobs = set()
        self.background_jobs = set()
        # What the running file jobs say in the status bar, per job.
        self.file_work = {}
        bind(self, t("app.irodori_voice_studio"), "setWindowTitle")
        self.setMinimumSize(*MINIMUM_SIZE)
        self.resize(*PREFERRED_SIZE)
        self.tasks = TaskManager(store, REPO, self)
        self.recover_records()
        # The library's FluentWindow draws the Windows 11 Mica backdrop (wallpaper-tinted, like
        # the system's own apps); Windows 10 keeps the plain background.
        self.apply_theme()
        # After the library's own handler, which re-applies the backdrop on a theme change.
        qconfig.themeChangedFinished.connect(self.sync_backdrop)
        self.setWindowIcon(app_icon())
        self.navigationInterface.setExpandWidth(224)
        # Narrower windows show the navigation as icons, which leaves the pages the width
        # their list and detail columns need.
        self.navigationInterface.setMinimumExpandWidth(COMPACT_BELOW)
        self.navigationInterface.setReturnButtonVisible(False)
        self.stack = self.stackedWidget
        self.tour = None
        self.guide = GuidePage(self)
        self.generation = GenerationPage(self)
        # Models: the ones trained here (projects with their steps) and the imported ones.
        self.models = ModelsPage(self)
        self.training = self.models.training
        self.settings_page = SettingsPage(self)
        # Getting started comes first, so a new user sees where to begin; it stays, as the
        # place to come back to while models download.
        self.pages = (self.guide, self.generation, self.models, self.settings_page)
        for index, (page, name, icon) in enumerate(
            zip(
                self.pages,
                (
                    t("guide.title"),
                    t("app.generate_speech"),
                    t("app.my_models"),
                    t("navigation.settings"),
                ),
                (FluentIcon.FLAG, AppIcon.INFERENCE, AppIcon.MODELS, FluentIcon.SETTING),
                strict=True,
            )
        ):
            page.setObjectName(f"page_{index}")
            self.addSubInterface(
                page,
                icon,
                name,
                position=NavigationItemPosition.BOTTOM
                if page is self.settings_page
                else NavigationItemPosition.TOP,
            )
            bind(self.navigationInterface.widget(page.objectName()), name)
        self.guide_badge = None
        if self.width() >= COMPACT_BELOW:
            self.navigationInterface.expand(useAni=False)
        # The library draws the content panel (border, rounded top-left corner, translucent
        # fill) on the page stack, which normally reaches the window's bottom edge. The status
        # strip sits under the pages, so the panel moves to the container that holds both;
        # otherwise the stack's square bottom-left corner shows above the strip.
        container = QWidget()
        container.setObjectName("contentPanel")
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        # A plain QWidget is not registered with Fluent's style manager. Register its
        # theme source when applying it so the panel has a fill on the very first frame.
        setStyleSheet(
            container,
            CustomStyleSheet(container).setCustomStyleSheet(
                CONTENT_PANEL_QSS["light"], CONTENT_PANEL_QSS["dark"]
            ),
        )
        self.stack.setProperty("isTransparent", True)
        self.stack.style().unpolish(self.stack)
        self.stack.style().polish(self.stack)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.widgetLayout.removeWidget(self.stack)
        self.widgetLayout.addWidget(container)
        outer.addWidget(self.stack, 1)
        outer.addWidget(self.build_status_bar())
        self.tasks.changed.connect(self.task_changed)
        self.tasks.finished.connect(self.task_finished)
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.save_ui)
        self.timer.start()
        # Bases the download cache already holds count from the first frame, so a new data
        # folder does not open on a download it does not need. Registering them reads file
        # sizes and checkpoint headers only; the shared files are adopted after the window.
        if not self.bases():
            try:
                base_models.adopt_cached_bases(self.resource_root)
            except OSError:
                pass
        self.refresh()
        ui = store.settings.get("ui", {})
        generation = ui.get("generation", {})
        self.generation.restore(generation if isinstance(generation, dict) else {})
        page = ui.get("page", 1)
        if model_files.speech_ready(self.resource_root):
            self.navigate(page if isinstance(page, int) else 1)
        else:
            # Nothing works without the speech model: open where it is obtained, directly
            # (no switch from another page first, which flickered).
            self.stackedWidget.view.setCurrentWidget(self.guide, duration=0)
            self.guide.refresh(force=True)
            QTimer.singleShot(0, self.guide.check_connection)
        size = ui.get("size")
        if not (
            isinstance(size, list) and len(size) == 2 and all(isinstance(v, int) for v in size)
        ):
            size = None
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(*window_size(size, available.width(), available.height()))
        # Restore only the size. Qt's default position was left in place for saved
        # sizes, so a larger window could reopen beyond the available screen area.
        self.move(available.center() - self.rect().center())
        views = ui.get("views", {})
        QTimer.singleShot(0, lambda: self.restore_views(views if isinstance(views, dict) else {}))
        self.update_status()
        QTimer.singleShot(0, self.detect_device)
        QTimer.singleShot(0, self.adopt_resources)
        if store.notice:
            QTimer.singleShot(0, lambda: self.notify(store.notice, error=True, duration=-1))
        if crowded := paths_too_long(store.root, cache_root(REPO)):
            folder = crowded[0]
            notice = t("app.path_too_long", length=len(str(folder)))
            QTimer.singleShot(0, lambda: self.notify(notice, duration=-1))
        QApplication.instance().styleHints().colorSchemeChanged.connect(lambda: self.apply_theme())

    def kind_text(self, task):
        if task["kind"] != "resources":
            return KIND_TEXT[task["kind"]]
        # Tasks from before the action was recorded name it in their request.
        action = task.get("action") or resource_action(
            read_json(self.tasks.task_dir(task) / "request.json", {}) or {}
        )
        return RESOURCE_TEXT[action]

    def task_title(self, task):
        """ "LoRA training · my-lora": the kind plus the project, run or model it concerns."""
        kind = self.kind_text(task)
        subject = None
        if task["kind"] == "train" and task.get("run"):
            subject = read_json(Path(task["run"]) / "run.json", {}).get("name")
        elif task.get("project"):
            subject = self.project_names.get(task["project"])
        elif task["kind"] in {"generate", "preview"}:
            subject = read_json(self.tasks.task_dir(task) / "request.json", {}).get("voice")
        if not subject:
            return kind
        return t("common.task_state", name=fmt(kind, "", -1), state=fmt(subject, "", -1))

    def reduced_motion(self):
        return self.store.settings.get("reduce_motion", False)

    def apply_theme(self):
        set_accent(self.store.settings.get("accent", "aqua"))
        apply_theme(QApplication.instance(), self.store.settings.get("theme", "system"))
        QApplication.instance().setProperty("reduce_motion", self.reduced_motion())
        if hasattr(self, "stack"):
            self.stack.setAnimationEnabled(not self.reduced_motion())
        self.sync_backdrop()

    def apply_accent(self):
        """Switch the theme colour alone (widgets.set_accent), and what shows it here."""
        set_accent(self.store.settings.get("accent", "aqua"))
        self.update_status()

    def sync_backdrop(self):
        """Give the Mica backdrop the theme's colours. The window library turns the backdrop's
        dark colours on for a dark theme but never off again, and it first applies them while
        the window is built, before the saved theme is set: on a dark system the light theme
        kept a dark frame and greyed pages."""
        if sys.platform != "win32" or not self.isMicaEffectEnabled():
            return
        from qframelesswindow.windows.c_structures import (
            ACCENT_POLICY,
            WINDOWCOMPOSITIONATTRIB,
            WINDOWCOMPOSITIONATTRIBDATA,
        )

        flag = ACCENT_POLICY()
        flag.AccentState = int(isDarkTheme())  # read as the attribute's BOOL
        data = WINDOWCOMPOSITIONATTRIBDATA()
        data.Attribute = WINDOWCOMPOSITIONATTRIB.WCA_USEDARKMODECOLORS.value
        data.Data = ctypes.pointer(flag)
        data.SizeOfData = ctypes.sizeof(ctypes.c_int)
        self.windowEffect.SetWindowCompositionAttribute(int(self.winId()), ctypes.pointer(data))
        # The colours take effect when the backdrop is applied after them. Windows redraws the
        # backdrop only when the window's dark-mode attribute changes, which it does not when the
        # library writes the value it already has: a switch while the window was open kept the
        # old backdrop. Flip it once and refresh the frame.
        hwnd = int(self.winId())
        self.windowEffect.setMicaEffect(hwnd, isDarkTheme())
        for dark in (not isDarkTheme(), isDarkTheme()):
            value = ctypes.c_int(int(dark))
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), 4)
        flags = 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020  # no size/move/z-order/activate; frame
        _user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, flags)

    def showEvent(self, event):
        super().showEvent(event)
        # The library applies the backdrop again once the window is shown.
        self.sync_backdrop()

    def switchTo(self, interface):
        # The library's convenience route resets ScrollArea offsets. Our pages own
        # their scroll state, so use its standard animated view without that reset.
        self.stackedWidget.view.setCurrentWidget(interface, duration=220)

    def date(self, text):
        return t("common.date", value=text)

    def perform(self, operation, error=None):
        try:
            return operation()
        except Exception as exc:
            self.report(error_message(exc), error)
            return None

    def report(self, message, error=None):
        """An action failed: say so in a toast. A toast label keeps the text (and shows it
        again for a second click); a card's state label is left as it is."""
        if getattr(error, "toast", False):
            clear_text(error)
            set_text(error, message)
        else:
            toast(self, message)

    def inform(self, message):
        """An action's outcome worth a sentence, as a toast rather than a line on the page."""
        toast(self, message, kind="success")

    def notify(self, message, *, error=False, duration=5000):
        """A message shown on top of whatever page is open."""
        (InfoBar.error if error else InfoBar.warning)(
            "",
            str(message),
            duration=duration,
            position=InfoBarPosition.TOP_RIGHT,
            parent=self,
        )

    def run_io(self, operation, completed, error, *, blocking=True, busy=None, settled=None):
        """Run file work off the GUI thread. A blocking job keeps the window open until it
        finishes; a background probe does not. busy is what the status bar says meanwhile;
        settled runs when the job ended either way, before completed."""
        job = Job(operation)
        jobs = self.jobs if blocking else self.background_jobs
        jobs.add(job)
        if busy is not None:
            self.file_work[job] = busy
            self.show_file_work()

        def ended():
            jobs.discard(job)
            self.file_work.pop(job, None)
            if not isValid(self):
                return False
            self.show_file_work()
            if settled is not None:
                self.perform(settled, error)
            return True

        def done(value):
            if not ended():
                return
            if error is not None and isValid(error):
                clear_text(error)
            self.perform(lambda: completed(value), error)

        def failed(message):
            if ended() and (error is None or isValid(error)):
                self.report(message, error)

        job.signals.result.connect(done)
        job.signals.error.connect(failed)
        QThreadPool.globalInstance().start(job)

    def show_file_work(self):
        working = bool(self.file_work)
        if working:
            set_text(self.file_work_text, list(self.file_work.values())[-1])
            self.file_work_bar.start()
        else:
            self.file_work_bar.stop()
        self.file_work_text.setVisible(working)
        self.file_work_bar.setVisible(working)

    @property
    def resource_root(self) -> Path:
        """The data folder's resources: the manifest and the base model registry."""
        return self.store.root / "resources"

    def bases(self) -> list:
        """The installed base models, catalog ones first."""
        return base_models.load_registry(self.resource_root)

    def adopt_resources(self):
        """A new data folder whose resources the shared cache already holds starts ready."""
        if not isValid(self) or self.settings_page.resources_ready or self.tasks.busy:
            return
        root = self.resource_root
        self.run_io(
            lambda: adopt_cached_resources(root),
            lambda adopted: self.refresh() if adopted else None,
            None,
            blocking=False,
        )

    def confirmation(self, title, message, action):
        box = Confirmation(str(title), str(message), self)
        bind(box.titleLabel, title)
        accent(box.yesButton)
        set_text(box.yesButton, action)
        set_text(box.cancelButton, t("app.cancel"))
        box.yesButton.setDefault(False)
        box.yesButton.setAutoDefault(False)
        box.cancelButton.setDefault(True)
        box.cancelButton.setFocus(Qt.FocusReason.OtherFocusReason)
        return box

    def confirm(self, title, message, action):
        return bool(self.confirmation(title, message, action).exec())

    def navigate(self, index):
        self.switchTo(self.pages[min(max(0, index), len(self.pages) - 1)])

    def use_model(self, model_id):
        """Open Generate with this model chosen."""
        self.generation.refresh()
        index = self.generation.voice.findData(model_id)
        if index < 0:
            raise BusinessError(t("library_page.model_files_are_missing"))
        self.generation.voice.setCurrentIndex(index)
        self.navigate(1)

    def show_guide(self):
        # Filled in before it slides in, not after (which made it jump while appearing).
        self.guide.refresh(force=True)
        self.guide.check_connection()
        self.switchTo(self.guide)

    def start_tour(self):
        """Walk through where things are: the window dims except the control being explained."""
        nav = self.navigationInterface.widget
        top, bottom = TeachingTipTailPosition.LEFT_TOP, TeachingTipTailPosition.LEFT_BOTTOM
        steps = [
            Step(
                lambda: self.guide.download_button,
                t("tour.download_title"),
                t("tour.download_content"),
                TeachingTipTailPosition.TOP,
                before=self.show_guide,
            ),
            Step(lambda: nav("page_0"), t("tour.guide_title"), t("tour.guide_content"), top),
            Step(lambda: nav("page_2"), t("tour.train_title"), t("tour.train_content"), top),
            Step(lambda: nav("page_1"), t("tour.generate_title"), t("tour.generate_content"), top),
            Step(
                lambda: nav("page_3"), t("tour.settings_title"), t("tour.settings_content"), bottom
            ),
        ]
        if self.tour is not None and self.tour.active:
            self.tour.finish()
        self.tour = Tour(self, steps)
        self.tour.finished.connect(self.tour_finished)
        self.tour.start()

    def tour_finished(self):
        self.store.settings["tour_done"] = True
        self.store.save()

    def new_project(self):
        """Ask for a name only; the project goes under the default location from Settings."""
        name, accepted = prompt_text(
            self, t("app.new_project"), t("app.project_name"), t("app.my_japanese_voice")
        )
        if accepted and name.strip():
            self.perform(
                lambda: self.register_project(create_project(self.store.project_root(), name))
            )

    def open_project(self):
        """Put a project folder back on the list (one taken off it, or copied here). A folder
        that holds project folders (the data folder's projects) brings back each of them."""
        path = QFileDialog.getExistingDirectory(
            self, str(t("app.open_project")), str(self.store.project_root())
        )
        if path:
            self.perform(lambda: self.add_projects(Path(path)))

    def add_projects(self, folder):
        roots = (
            [folder]
            if (folder / "project.json").is_file()
            else sorted(path.parent for path in folder.glob("*/project.json"))
        )
        if not roots:
            raise BusinessError(t("app.no_project_in_folder", path=fmt(folder, "", -1)))
        failed = None
        added = []
        for root in roots:
            try:
                self.store.register(root)
                added.append(root)
            except (OSError, ValueError, KeyError) as exc:
                failed = failed or exc
        if not added:
            raise failed
        self.refresh()
        self.models.select_project(added[0])
        self.navigate(2)

    def register_project(self, root):
        self.store.register(root)
        self.refresh()
        self.models.select_project(root)
        self.navigate(2)

    def import_project(self):
        path = open_file(
            self,
            self.store.settings,
            t("app.import_project_package"),
            t("app.project_packages_iroproject"),
        )
        if path:
            parent = self.store.project_root()
            projects = {project["id"] for _root, project in self.store.projects()}
            runs = {run["id"] for _root, _project, _run_root, run in self.store.runs()}
            self.run_io(
                lambda: import_project(
                    Path(path), parent, taken_projects=projects, taken_runs=runs
                ),
                self.register_project,
                None,
                busy=t("app.work_import_project"),
            )

    def locate_task(self, task=None):
        task = task if isinstance(task, dict) else self.tasks.current
        if not task:
            return
        if task.get("project"):
            self.models.select_project(task["project"])
            self.training.tabs.setCurrentIndex(
                {"initialize": 1, "transcribe": 0, "addon": 0, "models": 0}.get(task["kind"], 2)
            )
            self.navigate(2)
        elif task["kind"] == "generate" and not task.get("voice_id"):
            self.navigate(1)
        elif task["kind"] in {"generate", "preview"}:
            self.navigate(2)
        else:
            # Downloads and model files: where models are obtained.
            self.navigate(0)

    def project_busy(self, root):
        return self.tasks.busy and self.tasks.current.get("project") == str(root)

    def voice_entries(self):
        """Every model Generate can speak with: each training run with a complete checkpoint
        (its latest), then the imported models. Each names the installed base it runs on
        ("base_id", None when none has its architecture) and why not ("base_problem")."""
        bases = self.bases()
        entries = {**self.run_models(), **self.imported_models()}
        for entry in entries.values():
            entry.update(base_models.voice_base(entry.get("base"), bases))
        return entries

    def run_models(self):
        result = {}
        for root, project, run_root, run in self.store.runs():
            if run.get("checkpoint"):
                adapter = root / run["checkpoint"]
                if complete_checkpoint(adapter):
                    result[run["id"]] = {
                        "base": run.get("base"),
                        "name": run["name"],
                        "adapter": str(adapter),
                        "project_name": project["name"],
                        "project_root": str(root),
                        "run_root": str(run_root),
                        # Where the model keeps what belongs to it, such as its inference settings.
                        "folder": str(run_root),
                        "step": run["step"],
                        "metadata": str(run_root / "run.json"),
                        "created": run["created"],
                        "preview": str(run_root / run["preview"]) if run.get("preview") else None,
                        "preview_status": run.get("preview_status"),
                        "preview_error": run.get("preview_error", ""),
                        "state": t("app.completed")
                        if run["status"] == "completed"
                        else t("app.intermediate_result"),
                    }
        return result

    def imported_models(self):
        """Models in the data folder's voices: imported packages and checkpoints saved apart."""
        result = {}
        for path in sorted((self.store.root / "voices").glob("*/voice.json")):
            try:
                voice = read_json(path)
            except (OSError, ValueError):
                continue
            if isinstance(voice, dict) and voice.get("registered", True):
                sample = path.parent / voice["preview"] if voice.get("preview") else None
                result[path.parent.name] = {
                    **voice,
                    "name": voice.get("name") or path.parent.name,
                    "step": voice.get("step", "—"),
                    "adapter": str(path.parent),
                    "folder": str(path.parent),
                    "project_name": voice.get("project_name", t("app.imported_voice")),
                    "metadata": str(path),
                    "created": voice.get("created", ""),
                    "preview": str(sample) if sample else None,
                    "state": t("app.available"),
                }
        return result

    @staticmethod
    def preview_reference(root):
        """A clean included recording of the project's speaker, closest to 7.5 s: training
        conditions on a same-speaker reference, so samples should too. None without one."""
        try:
            materials = load_project(root)["materials"]
        except (OSError, ValueError, KeyError):
            return None
        best = None
        for item in materials:
            duration = item.get("duration") or 0
            if not item.get("included") or item.get("error") or not 3 <= duration <= 12:
                continue
            path = root / item["path"]
            if path.is_file() and (best is None or abs(duration - 7.5) < best[0]):
                best = (abs(duration - 7.5), path)
        return str(best[1]) if best else None

    def preview(self, root, run_root, checkpoint=None):
        run = read_json(run_root / "run.json")
        base = base_models.voice_base(run.get("base"), self.bases())
        if base["base_id"] is None:
            raise BusinessError(base["base_problem"])
        if checkpoint is None:
            if not run.get("checkpoint"):
                raise BusinessError(t("training_page.select_a_complete_checkpoint"))
            checkpoint = root / run["checkpoint"]
        request = {
            "text": "こんにちは。今日はとてもいい天気ですね。ゆっくりお話ししましょう。",
            "seed": 42,
            "num_steps": 40,
            "no_ref": True,
            "lora_adapter": str(checkpoint),
        }
        if reference := self.preview_reference(root):
            request.update(no_ref=False, ref_wavs=[reference], max_ref_seconds=10.0)
        self.tasks.preview(
            {
                "kind": "preview",
                "project": str(root),
                "run": str(run_root),
                "voice": run["name"],
                "voice_id": run["id"],
                "base": base["base_id"],
                "output": str(run_root / "samples" / stamp()),
                "request": request,
            }
        )

    def preview_adapter(self, voice):
        """A new sample for a model without a training run, kept in the model's own folder."""
        if self.tasks.busy:
            raise BusinessError(t("library_page.sample_waits_for_the_current_task"))
        base = base_models.voice_base(voice.get("base"), self.bases())
        if base["base_id"] is None:
            raise BusinessError(base["base_problem"])
        self.tasks.start(
            {
                "kind": "generate",
                "voice": voice["name"],
                "voice_id": voice["id"],
                "base": base["base_id"],
                "output": str(Path(voice["adapter"]) / "samples" / stamp()),
                "request": {
                    "text": "こんにちは。今日はとてもいい天気ですね。",
                    "seed": 42,
                    "num_steps": 40,
                    "no_ref": True,
                    "lora_adapter": voice["adapter"],
                },
            }
        )

    def model_sample_finished(self, task):
        """Record a finished sample of a model in its voice.json; True if it was one."""
        audio = Path((task.get("result") or {}).get("audio") or "")
        voices = (self.store.root / "voices").resolve()
        try:
            relative = audio.resolve().relative_to(voices)
        except (ValueError, OSError):
            return False
        if not relative.parts:
            return False
        folder = voices / relative.parts[0]
        record = read_json(folder / "voice.json")
        if isinstance(record, dict):
            record["preview"] = audio.resolve().relative_to(folder).as_posix()
            write_json(folder / "voice.json", record)
        return True

    def retry_task(self, task=None):
        task = task or self.tasks.current
        if task and not self.tasks.busy:
            if task["status"] == "completed":
                raise BusinessError(t("app.task_finished_start_a_new_task_on"))
            payload = read_json(self.tasks.task_dir(task) / "request.json")
            if payload["kind"] == "initialize":
                self.locate_task(task)
                self.training.initialize()
                return
            if payload["kind"] == "train":
                run = read_json(Path(payload["run"]) / "run.json")
                payload["resume"] = bool(run.get("checkpoint"))
            elif payload["kind"] == "generate":
                payload["output"] = str(self.store.root / "generations" / stamp())
            elif payload["kind"] == "preview" and payload.get("run"):
                payload["output"] = str(Path(payload["run"]) / "samples" / stamp())
            self.tasks.start(payload)

    def build_status_bar(self):
        """The 34px strip under every page: idle summary, or the running task with its controls."""
        bar = QWidget()
        bar.setObjectName("statusBar")
        bar.setFixedHeight(34)
        bar.setStyleSheet("#statusBar { border-top: 1px solid rgba(128, 128, 128, 0.28); }")
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(8, 8)
        self.task_label = label(t("app.ready"), wrap=False)
        self.task_progress = ProgressBar()
        self.task_progress.setFixedSize(220, 4)
        self.file_work_text = label("", "muted", wrap=False)
        self.file_work_bar = IndeterminateProgressBar(start=False)
        self.file_work_bar.setFixedSize(120, 4)
        self.file_work_text.hide()
        self.file_work_bar.hide()
        self.task_detail = elided_label()
        self.status_info = elided_label()
        self.status_info.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.locate = ghost_button(t("app.view_task"), self.locate_task)
        self.stop = ghost_button(t("app.save_and_stop"), self.tasks.stop)
        # A failed task can be run again from where it is reported.
        self.retry = ghost_button(t("app.retry"), lambda: self.perform(self.retry_task))
        for widget in (self.locate, self.retry, self.stop):
            widget.setMinimumHeight(0)
            widget.setFixedHeight(26)
        self.force = icon_button(
            FluentIcon.MORE, t("app.more_force_stop"), self.status_menu, size=26
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 12, 0)
        layout.setSpacing(12)
        layout.addWidget(self.status_dot)
        layout.addWidget(self.task_label)
        layout.addWidget(self.task_progress)
        layout.addWidget(self.file_work_text)
        layout.addWidget(self.file_work_bar)
        layout.addWidget(self.task_detail, 1)
        layout.addWidget(self.status_info, 1)
        for widget in (self.locate, self.retry, self.stop, self.force):
            layout.addWidget(widget)
        return bar

    def status_menu(self):
        show_menu(self.force, [(t("app.force_stop"), self.force_stop, self.tasks.busy)])

    def update_status(self, task=None):
        task = task or self.tasks.current
        busy = bool(task) and self.tasks.busy
        failed = bool(task) and not busy and task["status"] in {"failed", "interrupted"}
        shown = busy or failed
        color = (
            accent_line().name()
            if busy
            else "#ff99a4"
            if failed and task["status"] == "failed"
            else ("#f2c14e" if failed else "#6ccb5f")
        )
        self.status_dot.setStyleSheet(f"background: {color}; border-radius: 4px;")
        total = task.get("total", 0) if shown else 0
        set_text(self.task_label, self.task_title(task) if shown else t("app.ready"))
        bind(self.task_label, task["message"] if shown else "", "setToolTip")
        self.task_progress.setVisible(busy and total > 0)
        if busy and total:
            self.task_progress.setRange(0, total)
            self.task_progress.setValue(task["step"])
        if shown:
            state = STATE_TEXT[task["status"]]
            text = state
            if busy and total:
                text = t(
                    "app.task_steps_short", step=fmt(task["step"], "", -1), total=fmt(total, "", -1)
                )
                if task["kind"] == "train" and "loss" in task:
                    text = text + t("training.loss", value=fmt(task["loss"], ".4f"))
                eta = self.tasks.eta_seconds(task) if task["kind"] == "train" else None
                if eta is not None:
                    text = text + t("app.eta_left", duration=fmt(rough_duration(eta), "", -1))
                if task["status"] != "running":
                    text = t("common.task_state", name=fmt(text, "", -1), state=fmt(state, "", -1))
            if (
                busy
                and task["status"] == "running"
                and (
                    task["kind"]
                    in {"transcribe", "addon", "models", "import_models", "export_models"}
                    or not total
                )
            ):
                # Downloads, installs and recognition describe themselves ("Qwen3-ASR: 12 / 130"),
                # as does a step before progress is counted ("Loading the voice model").
                text = task["message"]
            set_text(self.task_detail, self.tasks.preview_message() or text)
        page = self.settings_page
        models = (
            t("app.status_model_ready") if page.resources_ready else t("app.status_model_missing")
        )
        short = self.device_lines()[2]
        set_text(
            self.status_info,
            t("common.task_state", name=fmt(models, "", -1), state=fmt(short, "", -1))
            if short
            else models,
        )
        self.task_detail.setVisible(shown)
        self.status_info.setVisible(not shown)
        self.locate.setVisible(shown)
        self.retry.setVisible(failed)
        self.stop.setVisible(busy)
        self.force.setVisible(busy)
        if busy:
            interlude = self.tasks.training_preview is not None
            self.stop.setEnabled(interlude or task["status"] not in {"saving", "stop_requested"})
            set_text(
                self.stop,
                t("tasks.preview_cancel_resume")
                if interlude
                else t("app.save_and_stop")
                if task["kind"] == "train"
                else t("app.stop_task"),
            )

    def task_changed(self, task):
        self.update_status(task)
        for page in (self.generation, self.training, self.settings_page):
            page.task_changed(task)
        # The models list says "Training 35%": its lines change with the percent or the state,
        # not with every step (each update reads the projects from disk).
        total = task.get("total") or 0
        state = (
            task.get("project"),
            task["kind"],
            task["status"],
            round(100 * task.get("step", 0) / total) if total else 0,
        )
        if task.get("project") and state != getattr(self, "listed_state", None):
            self.listed_state = state
            self.models.update_states()
        self.guide.task_changed(task)

    def task_finished(self, task):
        if task.get("project"):
            self.recover_records(Path(task["project"]).resolve())
        self.refresh()
        self.guide.task_finished(task)
        self.training.transcribed(task)
        self.training.recommendation_finished(task)
        if (
            task.get("preview_pause")
            and task["status"] == "failed"
            and str(self.training.root) == task.get("project")
        ):
            set_text(self.training.train_error, task["message"])
        if task["kind"] == "generate" and task["status"] == "completed":
            if self.model_sample_finished(task):
                self.refresh()
            else:
                self.generation.show_new(task["result"])
        if self.closing_after_stop:
            if task["status"] in {"failed", "interrupted"}:
                self.closing_after_stop = False
            else:
                QTimer.singleShot(0, self.close)
            return
        if task["kind"] == "initialize":
            self.training.continue_training(task)
        if (
            task["kind"] == "train"
            and task["status"] in {"completed", "stopped"}
            and not task.get("preview_pause")
            and not self.tasks.busy
            and (read_json(Path(task["run"]) / "run.json", {}) or {}).get("checkpoint")
        ):
            self.perform(
                lambda: self.preview(Path(task["project"]), Path(task["run"])),
                self.training.train_error,
            )

    def force_stop(self):
        task = self.tasks.current
        recovery = t("app.no_saved_training_checkpoint")
        if task and task.get("run"):
            checkpoint, saved_step = latest_checkpoint(Path(task["run"]))
            recovery = (
                t(
                    "app.latest_complete_checkpoint_step",
                    p1=fmt(saved_step, "", -1),
                    p3=fmt(checkpoint, "", -1),
                )
                if checkpoint
                else recovery
            )
        if (
            task
            and self.tasks.busy
            and self.confirm(
                t("app.force_stop_the_compute_task"),
                t(
                    "app.current_step_unsaved_progress_will_be_lost",
                    p1=fmt(task["step"], "", -1),
                    p3=fmt(recovery, "", -1),
                ),
                t("app.force_stop"),
            )
        ):
            self.tasks.force_stop()

    def recover_records(self, only=None):
        """Settle what an ended process left behind: every project at startup, then the one
        whose task just ended (it checks each recording, too much for all of them each time)."""
        for root, _project in self.store.projects():
            if self.project_busy(root) or (only and root != only):
                continue
            try:
                # A project another job holds (an import in progress) is checked next time.
                with project_lock(root):
                    project = load_project(root)
                    reconcile_sources(root, project)
                    init = project.get("initialization")
                    if init and init["status"] == "running":
                        init.update(
                            status="interrupted",
                            error=t("app.previous_initialization_was_interrupted_initialize_again"),
                        )
                        save_project(root, project)
            except (OSError, ValueError):
                continue
        for root, _project, run_root, run in self.store.runs(only):
            if run["status"] in {"preparing", "running"} and not self.project_busy(root):
                checkpoint, step = latest_checkpoint(run_root)
                run.update(
                    status="completed"
                    if step >= run["config"]["train"]["max_steps"]
                    else "interrupted",
                    step=step,
                )
                if checkpoint:
                    run.update(checkpoint=checkpoint.relative_to(root).as_posix(), registered=True)
                write_json(run_root / "run.json", run)

    def refresh(self):
        self.project_names = {str(root): project["name"] for root, project in self.store.projects()}
        self.models.refresh()
        self.generation.refresh()
        self.settings_page.refresh()
        self.guide.refresh()
        self.update_guide_badge()
        self.update_status()

    def update_guide_badge(self):
        """A dot on Getting started while something it lists still needs doing."""
        from qfluentwidgets import InfoBadge, InfoBadgePosition

        wanted = not self.guide.complete()
        if wanted and self.guide_badge is None:
            self.guide_badge = InfoBadge.attension(
                "",
                parent=self.navigationInterface,
                target=self.navigationInterface.widget("page_0"),
                position=InfoBadgePosition.NAVIGATION_ITEM,
            )
            self.guide_badge.setFixedSize(8, 8)
        # Hidden, never deleted: the badge's manager keeps moving it whenever the navigation
        # item resizes (as it does when the navigation collapses), and would reach a deleted one.
        if self.guide_badge is not None:
            self.guide_badge.setVisible(wanted)

    def detect_device(self, error=None):
        """Look at the GPU and system in the background; the result is cached for the tiles,
        the status strip and the Settings page, and refreshed from Settings."""
        self.run_io(
            device_report,
            self.device_detected,
            error if error is not None else self.settings_page.environment_error,
        )

    def device_detected(self, report):
        if not isValid(self):
            return
        self.device = report
        self.settings_page.show_environment(report)
        self.guide.device_known(report)
        self.update_guide_badge()
        self.update_status()

    def device_lines(self):
        """(headline, detail) for the device tile, or the GPU name and CUDA for the strip."""
        report = self.device
        if report is None:
            return t("app.tile_device_pending"), "", None
        gpu = report.get("gpu")
        if not gpu:
            return (
                t("app.no_gpu"),
                report.get("gpu_error") or "",
                None,
            )
        cuda = cuda_of(report.get("torch")) or "—"
        used, total = (f"{gpu[key] / 1024:.1f}" for key in ("used_mib", "total_mib"))
        return (
            gpu["name"],
            t(
                "app.device_detail",
                used=fmt(used, "", -1),
                total=fmt(total, "", -1),
                cuda=fmt(cuda, "", -1),
            ),
            t("app.device_short", gpu=fmt(gpu["name"], "", -1), cuda=fmt(cuda, "", -1)),
        )

    def save_ui(self):
        if self.training.root:
            self.project_states[self.training.project["id"]] = self.training.snapshot()
        views = {}
        for name, widget in self.view_widgets().items():
            if isinstance(widget, Table):
                views[name] = {
                    "selection": widget.item()["id"] if widget.item() else None,
                    "scroll": widget.verticalScrollBar().value(),
                    "widths": [widget.columnWidth(i) for i in range(widget.model().columnCount())],
                }
            else:
                views[name] = {"scroll": widget.verticalScrollBar().value()}
        self.store.settings["ui"] = {
            "page": self.pages.index(current)
            if (current := self.stack.currentWidget()) in self.pages
            else 0,
            "size": [self.width(), self.height()],
            "model": self.models.selection(),
            "generation": self.generation.snapshot(),
            "project_states": self.project_states,
            "views": views,
        }
        self.store.save()

    def view_widgets(self):
        return {
            "guide": self.guide.scroll,
            "generation": self.generation,
            "settings": self.settings_page,
            "materials": self.training.materials,
            "runs": self.training.runs,
            "history": self.generation.history,
            "references": self.generation.reference_table,
        }

    def restore_views(self, views):
        for name, widget in self.view_widgets().items():
            state = views.get(name, {})
            if isinstance(widget, Table):
                widths = state.get("widths", [])
                # Column layouts change between versions; widths only apply to the same shape.
                if len(widths) == widget.model().columnCount():
                    for index, width in enumerate(widths):
                        widget.setColumnWidth(index, width)
                for index, row in enumerate(widget.rows_model.rows):
                    if row["id"] == state.get("selection"):
                        widget.setCurrentIndex(widget.model().index(index, 0))
            widget.verticalScrollBar().setValue(state.get("scroll", 0))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # The library folds the navigation to icons below COMPACT_BELOW but, with its menu
        # button shown, never unfolds it: a window made wider again gets its labels back.
        if event.oldSize().width() < COMPACT_BELOW <= event.size().width():
            self.navigationInterface.expand(useAni=False)

    def nativeEvent(self, eventType, message):
        # Windows draws an inactive window's Mica backdrop flat grey, so a file dialog or any
        # other window of this app flashed the whole main window when it opened and again
        # when it closed. While the window taking over belongs to this process, the frame
        # keeps its active look; another application still makes it inactive.
        if sys.platform == "win32":
            msg = ctypes.wintypes.MSG.from_address(int(message))
            if msg.message == WM_NCACTIVATE and not msg.wParam and own_window_in_front():
                return True, _def_window_proc(msg.hWnd, WM_NCACTIVATE, 1, msg.lParam)
        return super().nativeEvent(eventType, message)

    def closeEvent(self, event):
        if self.tasks.busy:
            event.ignore()
            if self.confirm(
                t("app.task_still_running"),
                t("app.keep_running_or_save_and_stop_at"),
                t("app.save_stop_and_quit"),
            ):
                self.closing_after_stop = True
                self.tasks.stop()
            return
        if self.jobs:
            event.ignore()
            self.notify(t("app.file_processing_is_still_running_wait_for"))
            return
        self.save_ui()
        if not getattr(self, "players_stopped", False):
            event.ignore()
            if not getattr(self, "players_stopping", False):
                self.players_stopping = True
                players = (
                    self.generation.player,
                    self.models.imported.player,
                    self.training.material_player,
                    self.training.sample_player,
                )
                self.players_waiting = set(players)
                for player in players:
                    player.stopped.connect(lambda player=player: self._player_stopped(player))
                    player.stop()
            return
        if self.tasks.process:
            # Closes again once it has exited; the window stays alive meanwhile.
            event.ignore()
            self.tasks.release(self.compute_released)
        else:
            event.accept()

    def compute_released(self, released):
        if released:
            # Not from inside the exiting process's signal.
            QTimer.singleShot(0, self.close)
        else:
            self.notify(t("app.compute_process_did_not_exit"), error=True, duration=-1)

    def _player_stopped(self, player):
        self.players_waiting.discard(player)
        if not self.players_waiting:
            self.players_stopped = True
            self.close()


APP_ID = "Irodori.TTS.Desktop"
APP_NAME = "Irodori TTS Desktop"
LAUNCHER = "Irodori-TTS-Desktop.exe"


def package_root() -> Path | None:
    """The package folder when the app runs from a built package: its launcher sits next to
    the app folder. None for a source checkout."""
    root = REPO.parent
    return root if (root / LAUNCHER).is_file() else None


def default_data_root() -> Path:
    """Data lives next to the software, where users find it: beside the launcher in a package
    (not inside app/, which an update replaces), else in the checkout."""
    return (package_root() or REPO) / "data"


def set_taskbar_identity(window):
    """Give the window's taskbar button its icon, name and the command that starts the app.

    With an ID of its own and no shortcut carrying one, the taskbar shows a blank icon, and a
    pinned button would start a bare pythonw. A package starts again through its launcher."""
    import pythoncom
    from win32com.propsys import propsys, pscon

    package = package_root()
    if package:
        launcher = package / LAUNCHER
        command, icon = f'"{launcher}"', f"{launcher},0"
    else:
        python = Path(sys.executable).with_name("pythonw.exe")
        command, icon = f'"{python}" "{REPO / "desktop_app.py"}"', f"{ICON},0"
    store = propsys.SHGetPropertyStoreForWindow(int(window.winId()), propsys.IID_IPropertyStore)
    for key, value in (
        (pscon.PKEY_AppUserModel_ID, APP_ID),
        (pscon.PKEY_AppUserModel_RelaunchCommand, command),
        (pscon.PKEY_AppUserModel_RelaunchDisplayNameResource, APP_NAME),
        (pscon.PKEY_AppUserModel_RelaunchIconResource, icon),
    ):
        store.SetValue(key, propsys.PROPVARIANTType(value, pythoncom.VT_LPWSTR))
    store.Commit()


def main():
    parser = argparse.ArgumentParser(description=t("app.irodori_voice_studio_22"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--background-check", action="store_true")
    args = parser.parse_args()
    if args.background_check:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    configure_environment(REPO)
    hide_child_consoles()
    if sys.platform == "win32":
        # Its own taskbar identity: without it the taskbar groups the window under Python
        # and shows Python's icon.
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    app = QApplication(sys.argv[:1])
    app.setWindowIcon(app_icon())
    app.setOrganizationName("Irodori")
    app.setApplicationName("IrodoriDesktop")
    locale_manager()
    data_root = args.data_dir or default_data_root()
    server_name = (
        "IrodoriDesktop-" + hashlib.sha256(str(data_root.resolve()).encode()).hexdigest()[:20]
    )
    socket = QLocalSocket()
    socket.connectToServer(server_name)
    if socket.waitForConnected(300):
        if sys.platform == "win32":
            # Windows lets only the process the user just started take the foreground; pass
            # that on, or the running window merely flashes in the taskbar.
            ctypes.windll.user32.AllowSetForegroundWindow(ASFW_ANY)
        socket.write(b"locate")
        socket.waitForBytesWritten(300)
        # Stay until the running window has come forward, which ends the connection.
        socket.waitForDisconnected(2000)
        return 0
    QLocalServer.removeServer(server_name)
    server = QLocalServer()
    if not server.listen(server_name):
        raise RuntimeError(t("app.unable_to_start_the_single_instance_service"))
    window = MainWindow(AppStore(data_root))

    def locate_existing():
        connection = server.nextPendingConnection()
        if not args.background_check:
            # Restore from the taskbar only; a maximized window stays maximized.
            if window.isMinimized():
                window.setWindowState(window.windowState() & ~Qt.WindowState.WindowMinimized)
            window.show()
            window.raise_()
            window.activateWindow()
        if connection is not None:
            connection.disconnectFromServer()

    server.newConnection.connect(locate_existing)
    if args.background_check:
        window.resize(*PREFERRED_SIZE)
        window.stack.setAnimationEnabled(False)
    elif sys.platform == "win32":
        set_taskbar_identity(window)
    window.show()
    if not args.background_check and not window.store.settings.get("tour_done"):
        # The first real start shows where things are; Settings can show it again.
        QTimer.singleShot(800, window.start_tour)
    if args.background_check:

        def capture():
            directory = data_root / "screenshots"
            directory.mkdir(exist_ok=True)
            for index, name in enumerate(("guide", "generation", "models", "settings")):
                window.navigate(index)
                app.processEvents()
                window.grab().save(str(directory / f"{name}.png"))
            window.close()

        QTimer.singleShot(800, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
