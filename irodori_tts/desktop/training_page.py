from __future__ import annotations

import csv
import time
from pathlib import Path

from PySide6.QtCore import QModelIndex, QPointF, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QShortcut,
)
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    FluentIcon,
    LineEdit,
    Pivot,
    ProgressBar,
    SpinBox,
    StrongBodyLabel,
    TextEdit,
    getFont,
    isDarkTheme,
)

from irodori_tts.desktop.messages import BusinessError, error_message, fmt, t

from . import asr, base_models, model_files
from .asr_dialog import ask_transcription, choose_recordings_folder, recordings_start
from .exports import export_project, export_voice, register_checkpoint_voice
from .i18n import add_item, bind, clear_items, clear_text, set_text
from .materials import (
    MAX_TEXT_LENGTH,
    clean_field,
    edit_material,
    import_materials,
    material_problem,
    materials_fingerprint,
    needs_review,
    remove_materials,
)
from .storage import (
    load_project,
    project_lock,
    read_json,
    remove_tree,
    save_project,
    tree_size,
)
from .tasks import KIND_TEXT, STATE_TEXT
from .training import (
    RECOMMENDED,
    TUNED,
    budget_schedule,
    complete_checkpoint,
    create_run,
    delete_run,
    extend_run,
    initialization_current,
    normalize_preset,
    resume_eligibility,
    training_count,
    unique_run_name,
)
from .widgets import (
    DETAIL_MINIMUM,
    AppIcon,
    Callout,
    ChoiceBox,
    Dialog,
    Expand,
    LearningRateBox,
    ModelNotice,
    OptionalSpinBox,
    Page,
    Player,
    RowPlayback,
    SplitRow,
    Table,
    accent_drawn,
    accent_line,
    button,
    card,
    card_header,
    combo_box,
    elided_label,
    empty_state,
    error_label,
    field,
    grid,
    icon_button,
    inner_scroll,
    label,
    menu_button,
    open_folder,
    prompt_text,
    row,
    save_file,
    scientific,
    search_box,
    show_log,
    show_menu,
    smooth,
    step_icon,
    switch,
    toggle_button,
)

TOO_LONG = "materials.recording_exceeds_12_seconds_split_the_audio"
STATE_TONES = {
    "completed": "ok",
    "running": "accent",
    "saving": "accent",
    "preparing": "accent",
    "stopped": "warn",
    "interrupted": "warn",
    "failed": "bad",
}
STEPS = ("training_page.recordings", "tasks.data_initialization", "tasks.lora_training")


def human_duration(seconds):
    seconds = round(seconds)
    if seconds < 60:
        return t("app.seconds", p0=fmt(seconds, "", -1))
    if seconds < 3600:
        return t("training_page.minutes_seconds", m=seconds // 60, s=f"{seconds % 60:02d}")
    return t("training_page.hours_minutes", h=seconds // 3600, m=f"{seconds % 3600 // 60:02d}")


def clock(seconds):
    seconds = max(0, round(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60}:{seconds % 60:02d}"


def rough_duration(seconds):
    """ "6 min" style, for estimates that should not look precise."""
    seconds = round(seconds)
    if seconds < 90:
        return t("app.seconds", p0=fmt(seconds, "", -1))
    if seconds < 5400:
        return t("training_page.rough_minutes", n=fmt(round(seconds / 60), "", -1))
    return t("training_page.hours_minutes", h=seconds // 3600, m=f"{seconds % 3600 // 60:02d}")


def problem_of(item):
    """The reason a recording cannot be trained on as it is, or None."""
    error = item.get("error")
    if error and getattr(error, "key", None) == TOO_LONG:
        return t("generation_page.longer_than_12_seconds")
    return material_problem(item)


def problem_summary(items):
    """'reason ×n' for each distinct problem among the items, most common first."""
    counts = {}
    for item in items:
        reason = problem_of(item) or t("training_page.excluded")
        counts.setdefault(str(reason), [reason, 0])[1] += 1
    ordered = sorted(counts.values(), key=lambda entry: -entry[1])
    parts = [t("training_page.problem_count", reason=reason, count=n) for reason, n in ordered]
    summary = parts[0] if parts else ""
    for part in parts[1:]:
        summary = t("training_page.problem_list", first=summary, next=part)
    return summary


class MoreSteps(Dialog):
    """How far a finished run should train on, and how often it saves on the way."""

    def __init__(self, parent, step, total, save_every):
        super().__init__(parent)
        self.viewLayout.addWidget(label(t("training_page.train_more_title"), "section"))
        self.viewLayout.addWidget(label(t("training_page.train_more_caption", step=step)))
        self.total = smooth(SpinBox())
        self.total.setRange(step + 1, 1_000_000)
        self.total.setSingleStep(max(1, save_every))
        self.total.setValue(step + max(1, total))
        self.save_every = smooth(SpinBox())
        self.save_every.setRange(1, 1_000_000)
        self.save_every.setValue(max(1, save_every))
        self.viewLayout.addWidget(
            grid(
                field(t("training_page.train_more_total"), self.total),
                field(t("training_page.save_interval"), self.save_every),
            )
        )
        set_text(self.yesButton, t("training_page.train_more_start"))
        set_text(self.cancelButton, t("app.cancel"))
        self.widget.setMinimumWidth(440)


class LossChart(QWidget):
    """Training loss against step, with the run's checkpoints as points on it to pick from.
    Raw per-step loss is noisy, so the line is a running average. Hovering anywhere marks the
    nearest checkpoint; a click or the arrow keys pick it."""

    chosen = Signal(str)  # the picked checkpoint's key (its folder)

    def __init__(self):
        super().__init__()
        self.points = []
        self.limit = 0
        self.checkpoints = []  # (step, key), in step order
        self.selected = None
        self.hover = None
        self.dragged_from = None  # the pick when the button went down, while it is held
        self.setMinimumHeight(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        bind(self, t("training_page.pick_on_curve"), "setAccessibleName")

    def set_points(self, points, limit):
        self.points = list(points)
        self.limit = limit
        self.update()

    def set_checkpoints(self, checkpoints, selected):
        self.checkpoints = sorted(checkpoints)
        keys = {key for _step, key in self.checkpoints}
        self.selected = selected if selected in keys else None
        self.hover = self.hover if self.hover in keys else None
        self.update()

    def smoothed(self):
        values, current = [], None
        alpha = min(0.3, max(0.02, 20 / max(1, len(self.points))))
        for _step, loss in self.points:
            current = loss if current is None else current + alpha * (loss - current)
            values.append(current)
        stride = max(1, len(values) // 400)
        return [(self.points[i][0], values[i]) for i in range(0, len(values), stride)] + (
            [(self.points[-1][0], values[-1])] if (len(values) - 1) % stride else []
        )

    def frame(self):
        """(plot area, smoothed line, lowest and highest loss shown, last step)."""
        area = QRectF(self.rect().adjusted(44, 8, -8, -8))
        line = self.smoothed() if len(self.points) >= 2 else []
        low, high = 0.0, 1.0
        if line:
            low, high = min(v for _, v in line), max(v for _, v in line)
            if high - low < 1e-9:
                low, high = low - 0.5, high + 0.5
            pad = (high - low) * 0.08
            low, high = low - pad, high + pad
        last = max([self.limit, line[-1][0] if line else 0, *(s for s, _ in self.checkpoints)])
        return area, line, low, high, last or 1

    @staticmethod
    def value_at(line, step):
        """The running average at step: its last value at or before step."""
        if not line:
            return None
        value = line[0][1]
        for at, current in line:
            if at > step:
                break
            value = current
        return value

    def loss_at(self, step):
        return self.value_at(self.frame()[1], step)

    def marker(self, frame, step):
        area, line, low, high, last = frame
        x = area.left() + step / last * area.width()
        value = self.value_at(line, step)
        if value is None:
            return QPointF(x, area.center().y())
        return QPointF(x, area.bottom() - (value - low) / (high - low) * area.height())

    def nearest(self, x):
        if not self.checkpoints:
            return None
        frame = self.frame()
        return min(self.checkpoints, key=lambda c: abs(self.marker(frame, c[0]).x() - x))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        dark = isDarkTheme()
        muted = QColor("#9a9a9a" if dark else "#5b6068")
        painter.setFont(getFont(11))
        frame = self.frame()
        area, line, low, high, last = frame
        if (not line and not self.checkpoints) or area.width() < 20 or area.height() < 20:
            painter.setPen(muted)
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, str(t("training_page.no_loss_yet"))
            )
            return
        grid = QPen(QColor(128, 128, 128, 70), 1)
        if line:
            for fraction in (0.0, 0.5, 1.0):
                y = area.bottom() - fraction * area.height()
                painter.setPen(grid)
                painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
                painter.setPen(muted)
                painter.drawText(
                    QRectF(0, y - 8, 38, 16),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    f"{low + fraction * (high - low):.2f}",
                )
            path = QPainterPath()
            for index, (step, _value) in enumerate(line):
                point = self.marker(frame, step)
                path.moveTo(point) if index == 0 else path.lineTo(point)
            painter.setPen(QPen(accent_line(), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
        else:
            # Checkpoints without a loss log still sit on the step axis.
            middle = area.center().y()
            painter.setPen(grid)
            painter.drawLine(QPointF(area.left(), middle), QPointF(area.right(), middle))
            painter.setPen(muted)
            painter.drawText(
                QRectF(area.left(), area.top(), area.width(), area.height() / 2 - 8),
                Qt.AlignmentFlag.AlignCenter,
                str(t("training_page.no_loss_yet")),
            )
        # Each checkpoint: a tick on the step axis and a white point ringed in the accent on
        # the curve. The picked one: an accent point with a white ring and a halo (stronger
        # while the chart has keyboard focus), on a thin line up from its tick.
        accent = accent_line()
        white = QColor("#ffffff")
        tick = QColor(154, 160, 172) if dark else QColor(138, 144, 153)
        marked = QColor(205, 214, 229) if dark else QColor(50, 58, 69)
        for step, key in self.checkpoints:
            x = self.marker(frame, step).x()
            strong = key in (self.selected, self.hover)
            painter.setPen(QPen(marked if strong else tick, 2 if strong else 1.5))
            painter.drawLine(QPointF(x, area.bottom() - 7), QPointF(x, area.bottom()))
        for step, key in self.checkpoints:
            point = self.marker(frame, step)
            if key == self.selected:
                guide = QColor(accent)
                guide.setAlpha(110)
                painter.setPen(QPen(guide, 1))
                painter.drawLine(QPointF(point.x(), area.top()), QPointF(point.x(), area.bottom()))
                halo = QColor(accent)
                halo.setAlpha(100 if self.hasFocus() else 60)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(halo)
                radius = 12 if self.hasFocus() else 11
                painter.drawEllipse(point, radius, radius)
                painter.setPen(QPen(white, 2))
                painter.setBrush(accent)
                painter.drawEllipse(point, 6, 6)
            else:
                painter.setPen(QPen(accent, 1.8))
                painter.setBrush(white)
                radius = 5 if key == self.hover else 3.5
                painter.drawEllipse(point, radius, radius)

    def tip(self, step):
        loss = self.loss_at(step)
        return t(
            "training_page.curve_point",
            step=step,
            loss=f"{loss:.3f}" if loss is not None else "—",
        )

    def mouseMoveEvent(self, event):
        found = self.nearest(event.position().x())
        key = found[1] if found else None
        if self.dragged_from is not None and key is not None and key != self.selected:
            # Dragging moves the pick along; the sample follows when the button is let go.
            self.selected = key
        if key != self.hover:
            self.hover = key
        self.update()
        self.setCursor(Qt.CursorShape.PointingHandCursor if key else Qt.CursorShape.ArrowCursor)
        if found:
            QToolTip.showText(event.globalPosition().toPoint(), str(self.tip(found[0])), self)

    def leaveEvent(self, event):
        self.hover = None
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        found = self.nearest(event.position().x())
        if event.button() == Qt.MouseButton.LeftButton and found:
            self.dragged_from = self.selected or ""
            self.selected = found[1]
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.dragged_from is not None:
            start, self.dragged_from = self.dragged_from, None
            if self.selected and self.selected != start:
                self.chosen.emit(self.selected)

    def keyPressEvent(self, event):
        moves = {Qt.Key.Key_Left: -1, Qt.Key.Key_Right: 1}
        keys = [key for _step, key in self.checkpoints]
        if event.key() in moves and keys:
            index = keys.index(self.selected) if self.selected in keys else len(keys)
            self.choose(keys[min(len(keys) - 1, max(0, index + moves[event.key()]))])
        else:
            super().keyPressEvent(event)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.update()

    def choose(self, key):
        if key != self.selected:
            self.selected = key
            self.update()
            self.chosen.emit(key)


class Metric(QWidget):
    def __init__(self, title):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(label(title, "muted", wrap=False))
        self.value = StrongBodyLabel("—")
        layout.addWidget(self.value)

    def set(self, value):
        set_text(self.value, value)


class StepRow(QWidget):
    """One preprocessing step: marker, title, detail, and an optional progress bar."""

    def __init__(self, number, title, hint="", *, progress=False):
        super().__init__()
        self.number = number
        self.hint = hint
        self.badge = QLabel()
        self.badge.setFixedSize(26, 26)
        self.title = label(title, wrap=False)
        self.detail = label(hint, "muted")
        self.count = label("", "muted", wrap=False)
        self.bar = ProgressBar() if progress else None
        if self.bar:
            self.bar.setRange(0, 1)
            self.bar.setFixedHeight(6)
        text = QVBoxLayout()
        text.setSpacing(4)
        text.addWidget(row(self.title, None, self.count) if progress else self.title)
        if self.bar:
            text.addWidget(self.bar)
        text.addWidget(self.detail)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(text, 1)
        self.set_state("pending")

    def set_state(self, state):
        self.state = state
        if self.bar:
            self.bar.setVisible(state != "pending")
        accent_drawn(
            self.badge,
            lambda: self.badge.setPixmap(step_icon(self.number, state, size=24).pixmap(24, 24)),
        )

    def set_detail(self, text):
        set_text(self.detail, text)
        self.detail.setVisible(bool(str(text)))


class TrainingPage(QWidget):
    def __init__(self, host):
        super().__init__()
        self.host = host
        self.root = None
        self.project = None
        self.ready = False
        self.task_state = None
        self.task_listed = 0.0
        # Starting a run checks its files in the background; one start at a time.
        self.starting = False
        self.material_id = None
        self.run_cache = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # The model's name heads the page, beside what can be done to the model as a whole.
        self.title = elided_label(heading=True)
        # The project path is long; it lives in this button's tooltip instead of the header.
        self.folder_button = icon_button(
            FluentIcon.FOLDER, t("library_page.open_location"), lambda: open_folder(self.root)
        )
        self.project_more = icon_button(FluentIcon.MORE, t("library_page.more"), self.project_menu)
        # In a narrow window the models page hides its list; this button opens it as a menu.
        self.switch_button = icon_button(
            FluentIcon.MENU, t("models_page.all_models"), lambda: host.models.model_menu(self)
        )
        self.switch_button.hide()
        header = row(
            self.switch_button,
            (self.title, 1),
            icon_button(
                FluentIcon.EDIT, t("models_page.rename_model"), lambda: host.perform(self.rename)
            ),
            self.folder_button,
            self.project_more,
            spacing=4,
        )
        header.layout().setContentsMargins(14, 20, 24, 0)
        layout.addWidget(header)
        self.header = header

        self.pivot = Pivot()
        self.summary = elided_label()
        self.summary.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        steps = row(self.pivot, (self.summary, 1), spacing=16)
        steps.layout().setContentsMargins(14, 2, 24, 0)
        layout.addWidget(steps)
        self.steps = steps
        self.tabs = QStackedWidget()
        layout.addWidget(self.tabs, 1)
        self.materials_page = Page()
        self.init_page = Page()
        self.train_page = Page()
        for index, (page, title) in enumerate(
            zip(
                (self.materials_page, self.init_page, self.train_page),
                (
                    t("training_page.recordings"),
                    t("tasks.data_initialization"),
                    t("training_page.training_tab"),
                ),
                strict=True,
            )
        ):
            page.body.setContentsMargins(14, 10, 24, 20)
            self.tabs.addWidget(page)
            # PivotItem.itemClicked carries a bool; keep it out of the bound index.
            self.pivot.addItem(
                str(index),
                title,
                onClick=lambda _checked=False, i=index: self.tabs.setCurrentIndex(i),
            )
            bind(self.pivot.widget(str(index)), title)
            self.pivot.widget(str(index)).setStyleSheet("font-size: 14px;")
        for position in (1, 3):
            arrow = label("", wrap=False)
            arrow.setPixmap(FluentIcon.CHEVRON_RIGHT.icon().pixmap(12, 12))
            arrow.setContentsMargins(6, 0, 6, 0)
            self.pivot.hBoxLayout.insertWidget(position, arrow)
        self.tabs.currentChanged.connect(self.tab_changed)
        self.pivot.setCurrentItem("0")
        self.build_materials()
        self.build_initialization()
        self.build_training()
        QShortcut(QKeySequence.StandardKey.Save, self, activated=self.save_shortcut)
        self.refresh_steps()

    def set_inset(self, left):
        """The left margin: beside the models list, or at the page's edge without it."""
        layouts = [self.header.layout(), self.steps.layout()]
        layouts += [page.body for page in (self.materials_page, self.init_page, self.train_page)]
        for layout in layouts:
            margins = layout.contentsMargins()
            layout.setContentsMargins(left, margins.top(), margins.right(), margins.bottom())

    def require(self):
        if self.root is None:
            raise BusinessError(t("training_page.create_or_open_a_project_first"))
        return self.root

    def tab_changed(self, index):
        self.pivot.setCurrentItem(str(index))
        self.refresh_steps()
        self.update_summary()

    def refresh_steps(self):
        """Step markers: current step filled, finished steps checked, a dot when data went stale."""
        current = self.tabs.currentIndex()
        changed = bool(self.project and self.project.get("initialization") and not self.ready)
        finished = any(item.get("status") == "completed" for item in self.run_cache)
        done = (False, self.ready, finished)
        for index in range(3):
            state = "active" if index == current else "done" if done[index] else "pending"
            item = self.pivot.widget(str(index))
            accent_drawn(
                item,
                lambda item=item, index=index, state=state, dot=index == 1 and changed: (
                    item.setIcon(step_icon(index + 1, state, dot=dot))
                ),
            )
        self.pivot.widget("1").setToolTip(
            str(t("training_page.recordings_changed_initialize_again")) if changed else ""
        )

    def update_summary(self):
        self.folder_button.setEnabled(bool(self.project))
        if not self.project:
            set_text(self.summary, "")
            return
        bind(
            self.folder_button,
            t("training_page.stored_at", path=fmt(self.root, "", -1)),
            "setToolTip",
        )
        items = self.project["materials"]
        detail = (
            t("app.initialized")
            if self.ready
            else t("training_page.recordings_changed_initialize_again")
            if self.project.get("initialization")
            else t("training_page.prepare_recordings_before_initialization")
        )
        set_text(
            self.summary,
            t(
                "training_page.project_summary",
                n=len(items),
                duration=fmt(human_duration(sum(item["duration"] for item in items)), "", -1),
                detail=fmt(detail, "", -1),
            ),
        )

    # ----- dataset -----

    def build_materials(self):
        columns = QHBoxLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(14)
        self.materials_page.body.addLayout(columns, 1)

        left, body = card()
        columns.addWidget(left, 1)
        # Recordings arrive without transcripts: importing is transcribing. Transcripts the user
        # already has, the example and transcribing again are the other ways in.
        self.import_button = button(
            t("training_page.import_recordings"),
            lambda: self.host.perform(self.import_transcribed, self.material_error),
            primary=True,
            icon=FluentIcon.ADD,
        )
        self.more_imports = menu_button(
            t("training_page.more_actions"),
            [
                (
                    t("training_page.import_audio_and_csv"),
                    lambda: self.host.perform(lambda: self.import_audio(True), self.material_error),
                ),
                (
                    t("training_page.import_audio_and_txt"),
                    lambda: self.host.perform(
                        lambda: self.import_audio(False), self.material_error
                    ),
                ),
                (
                    t("training_page.import_local_example"),
                    lambda: self.host.perform(self.import_example, self.material_error),
                ),
                (
                    t("training_page.csv_template"),
                    lambda: self.host.perform(self.csv_template, self.material_error),
                ),
                (
                    t("asr.transcribe_listed_again"),
                    lambda: self.host.perform(self.retranscribe_listed, self.material_error),
                ),
                # Clearing is out of the way, behind the menu, not a button beside Import.
                (
                    t("training_page.clear_recordings"),
                    lambda: self.host.perform(self.clear_materials, self.material_error),
                ),
            ],
        )
        self.only_problems = toggle_button()
        self.only_problems.setIcon(FluentIcon.FILTER)
        self.only_problems.toggled.connect(lambda _checked: self.refresh_materials())
        self.filter = search_box(t("training_page.filter_by_filename_or_transcript"))
        self.filter.textChanged.connect(self.refresh_materials)
        self.material_error = error_label()
        body.addWidget(
            SplitRow(
                row(self.import_button, self.more_imports, (self.material_error, 1)),
                row(self.only_problems, self.filter),
            )
        )
        self.materials = Table(
            [
                ("play", "", 44),
                ("file_name", t("training_page.filename"), 170),
                ("seconds", t("generation_page.duration_115"), 90),
                ("text", t("training_page.transcript"), 0),
                ("state", t("app.status"), 130),
                ("actions", "", 44),
            ],
            fill=True,
            stretch=3,
            checks={"file_name": "included"},
            optional=("seconds",),
        )
        # Delete sits on every row, painted like the play icon, rather than appearing on the
        # selected row only.
        self.materials.icon_clicked.connect(self.material_icon_clicked)
        QShortcut(
            QKeySequence(QKeySequence.StandardKey.Delete),
            self.materials,
            activated=lambda: self.host.perform(self.delete_selected, self.material_error),
        ).setContext(Qt.ShortcutContext.WidgetShortcut)
        self.materials.selected.connect(self.select_material)
        self.materials.rows_model.toggled.connect(
            lambda key, checked: self.host.perform(
                lambda: self.toggle_included(key, checked), self.material_error
            )
        )
        body.addWidget(self.materials, 1)
        self.material_summary = elided_label()
        body.addWidget(row((self.material_summary, 1)))

        self.detail_card, detail = card()
        # The list gives up width first; the detail panel keeps at least DETAIL_MINIMUM.
        self.detail_card.setMinimumWidth(DETAIL_MINIMUM)
        self.detail_card.setMaximumWidth(340)
        columns.setStretch(0, 2)
        columns.addWidget(self.detail_card, 1)
        self.detail_stack = QStackedWidget()
        detail.addWidget(self.detail_stack)
        self.detail_stack.addWidget(empty_state(t("training_page.pick_a_recording")))
        panel = QWidget()
        self.detail_stack.addWidget(panel)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)
        # Everything but Save scrolls when the window is short, so Save stays in view.
        fields = QWidget()
        outer.addWidget(inner_scroll(fields), 1)
        layout = QVBoxLayout(fields)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.material_title = elided_label(strong=True)
        self.material_meta = elided_label()
        names = QVBoxLayout()
        names.setSpacing(2)
        names.addWidget(self.material_title)
        names.addWidget(self.material_meta)
        head = QHBoxLayout()
        head.addLayout(names, 1)
        head.addWidget(
            icon_button(
                FluentIcon.PAGE_LEFT,
                t("training_page.previous_recording"),
                lambda: self.step_material(-1),
            )
        )
        head.addWidget(
            icon_button(
                FluentIcon.PAGE_RIGHT,
                t("training_page.next_recording"),
                lambda: self.step_material(1),
            )
        )
        layout.addLayout(head)
        self.material_player = Player()
        layout.addWidget(self.material_player)
        self.material_text = smooth(TextEdit())
        # The saved fields the editor was filled with: the editor differs from them only by the
        # user's edits, also when the project changes underneath (a new transcription).
        self.loaded = None
        self.material_text.setMinimumHeight(90)
        bind(self.material_text, t("training_page.transcript_japanese"), "setAccessibleName")
        bind(
            self.material_text,
            t("training_page.input_the_transcript"),
            "setPlaceholderText",
        )
        self.material_hint = error_label(wrap=True)
        text = QVBoxLayout()
        text.setSpacing(6)
        text.addWidget(label(t("training_page.transcript_japanese"), wrap=False))
        text.addWidget(self.material_text, 1)
        text.addWidget(self.material_hint)
        layout.addLayout(text, 1)
        # What the cross-check model heard, next to the transcript it was compared with.
        self.review_box = QWidget()
        review = QVBoxLayout(self.review_box)
        review.setContentsMargins(0, 0, 0, 0)
        review.setSpacing(4)
        self.review_title = elided_label()
        self.review_text = label("")
        self.review_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.use_check = button(
            t("asr.use_check_text"),
            lambda: self.host.perform(self.use_check_text, self.detail_error),
        )
        self.mark_review = button(
            t("asr.mark_reviewed"),
            lambda: self.host.perform(self.save_material, self.detail_error),
            icon=FluentIcon.ACCEPT,
        )
        review.addWidget(self.review_title)
        review.addWidget(self.review_text)
        review.addWidget(SplitRow(row(None, self.use_check), self.mark_review))
        layout.addWidget(self.review_box)
        self.material_caption = smooth(LineEdit())
        bind(self.material_caption, t("generation_page.optional_for_example"), "setPlaceholderText")
        layout.addWidget(field(t("training_page.description"), self.material_caption))
        self.included = switch()
        layout.addWidget(
            row(label(t("training_page.include_in_training"), wrap=False), None, self.included)
        )
        self.detail_error = error_label()
        self.save_button = button(
            t("widgets.save"),
            lambda: self.host.perform(self.save_material, self.detail_error),
            primary=True,
        )
        # The shortcut is in the tooltip: as a label it left the two actions too little room.
        bind(self.save_button, t("training_page.save_tooltip"), "setToolTip")
        # The recording's own actions get a line of their own (beside Save their text was cut
        # off), and two lines when the panel is too narrow for both.
        layout.addWidget(
            SplitRow(
                button(
                    t("training_page.replace_recording"),
                    lambda: self.host.perform(self.replace_material, self.detail_error),
                    icon=FluentIcon.SYNC,
                ),
                row(
                    button(
                        t("asr.transcribe_again"),
                        lambda: self.host.perform(
                            lambda: self.retranscribe(
                                [self.material_id] if self.material_id else []
                            ),
                            self.detail_error,
                        ),
                        icon=FluentIcon.FONT,
                    ),
                    None,
                ),
                stretch=(0, 1),
            )
        )
        outer.addWidget(row((self.detail_error, 1), self.save_button))
        self.material_text.textChanged.connect(self.update_hint)
        # Any recording plays from its row; the detail panel's player is the one it uses.
        self.material_playback = RowPlayback(
            self.materials, self.material_player, lambda item: self.root / item["path"]
        )

    def material_icon_clicked(self, item, key):
        if key == "actions":
            self.host.perform(lambda: self.delete_materials([item["id"]]), self.material_error)

    def delete_selected(self):
        item = self.materials.item()
        if item:
            self.delete_materials([item["id"]])

    def clear_materials(self):
        self.delete_materials([item["id"] for item in (self.project or {}).get("materials", [])])

    def delete_materials(self, ids):
        """Take recordings out of the project after asking; their source files stay."""
        root = self.require()
        self.ensure_editable()
        items = [item for item in self.project["materials"] if item["id"] in set(ids)]
        if not items:
            return
        if len(items) == 1:
            title, message, action = (
                t("training_page.delete_recording_title"),
                t(
                    "training_page.delete_recording_message",
                    name=fmt(items[0]["file_name"], "", -1),
                ),
                t("training_page.delete_recording"),
            )
        else:
            title, message, action = (
                t("training_page.clear_recordings_title"),
                t("training_page.clear_recordings_message", count=len(items)),
                t("training_page.clear_recordings"),
            )
        if not self.host.confirm(title, message, action):
            return
        # Windows keeps a playing file open; the player lets go before the file is deleted.
        if self.material_id in {item["id"] for item in items}:
            self.material_player.clear()
        QTimer.singleShot(
            250,
            lambda: self.host.perform(
                lambda: self.finish_delete(root, [item["id"] for item in items]),
                self.material_error,
            ),
        )

    def finish_delete(self, root, ids):
        self.ensure_editable()
        remove_materials(root, ids)
        if self.material_id in ids:
            self.material_id = None
            self.loaded = None
        self.host.refresh()

    def update_hint(self):
        item = self.selected_material()
        if item is None:
            return
        problem = problem_of({**item, "text": self.material_text.toPlainText()})
        if problem is None:
            clear_text(self.material_hint)
        elif not self.material_text.toPlainText().strip():
            set_text(self.material_hint, t("training_page.transcript_missing_hint"))
        else:
            set_text(self.material_hint, item.get("error") or problem)

    def selected_material(self):
        if self.project and self.material_id:
            return next((i for i in self.project["materials"] if i["id"] == self.material_id), None)
        return None

    def refresh_materials(self):
        if not self.project:
            return
        needle = self.filter.text().lower()
        items = self.project["materials"]
        problems = [item for item in items if problem_of(item)]
        flagged = [item for item in items if problem_of(item) or needs_review(item)]
        bind(self.only_problems, t("training_page.only_problems", count=len(flagged)))
        only = self.only_problems.isChecked()
        rows = []
        for item in items:
            if needle not in (item["file_name"] + clean_field(item.get("text"))).lower():
                continue
            problem = problem_of(item)
            review = needs_review(item)
            if only and not (problem or review):
                continue
            rows.append(
                {
                    **item,
                    "seconds": t("app.seconds", p0=fmt(item["duration"], ".2f", -1)),
                    "state": problem
                    or (t("asr.needs_review") if review else t("training_page.material_ok")),
                    "tones": {"state": "bad" if item.get("error") else "warn"}
                    if problem or review
                    else {"state": "muted"},
                }
            )
        self.materials.update_rows(rows)
        self.material_playback.sync()
        tip = t("training_page.delete_recording")
        self.materials.set_icons(
            "actions",
            {row["id"]: FluentIcon.DELETE for row in rows},
            {row["id"]: tip for row in rows},
        )
        included = sum(item["included"] for item in items)
        # The footer counts what the filter counts: problems and recordings awaiting review.
        review = len(flagged) - len(problems)
        set_text(
            self.material_summary,
            t(
                "training_page.dataset_summary_review",
                included=included,
                total=len(items),
                problems=len(problems),
                review=review,
            )
            if review
            else t(
                "training_page.dataset_summary",
                included=included,
                total=len(items),
                problems=len(problems),
            ),
        )
        self.refresh_detail()

    def refresh_detail(self):
        rows = self.materials.rows_model.rows
        position = next((i for i, r in enumerate(rows) if r["id"] == self.material_id), None)
        item = self.selected_material()
        self.detail_stack.setCurrentIndex(1 if item and position is not None else 0)
        if item and self.fields(item) != self.loaded and self.editor_fields() == self.loaded:
            # The saved recording changed (a new transcription) and the user had not edited it
            # here: show what is saved now.
            self.load_fields(item)
        if item and position is not None:
            set_text(self.material_title, item["file_name"])
            set_text(
                self.material_meta,
                t(
                    "training_page.recording_position",
                    seconds=fmt(t("app.seconds", p0=fmt(item["duration"], ".2f", -1)), "", -1),
                    index=position + 1,
                    total=len(rows),
                ),
            )
            origin = item.get("segment")
            tip = (
                t(
                    "asr.segment_source",
                    file=fmt(origin["file"], "", -1),
                    start=fmt(origin["start"], ".2f"),
                    end=fmt(origin["end"], ".2f"),
                )
                if origin
                else ""
            )
            bind(self.material_meta, tip, "setToolTip")
            self.show_review(item)

    def use_check_text(self):
        """Take what the cross-check heard as the transcript and save it."""
        self.material_text.setPlainText(self.check_text())
        self.save_material()

    def check_text(self):
        item = self.selected_material() or {}
        return clean_field((item.get("asr") or {}).get("check_text"))

    def show_review(self, item):
        record = item.get("asr") or {}
        shown = bool(record.get("check_backend"))
        self.review_box.setVisible(shown)
        if not shown:
            return
        name = fmt(asr.backend_for(record["check_backend"]).title, "", -1)
        flagged = needs_review(item)
        heard = clean_field(record.get("check_text"))
        set_text(
            self.review_title,
            t("asr.check_text_flagged", name=name)
            if flagged
            else t("asr.check_text_title", name=name),
        )
        set_text(self.review_text, heard or t("asr.check_text_empty"))
        self.mark_review.setVisible(flagged)
        self.use_check.setVisible(bool(heard) and heard != clean_field(item.get("text")))

    def select_material(self, item):
        if item["id"] == self.material_id:
            # Also the selection restored at start, whose recording the player has not loaded.
            if self.root and self.material_player.path != str(self.root / item["path"]):
                self.material_player.load(self.root / item["path"])
            self.refresh_detail()
            return
        saved = self.autosave_draft()
        self.material_id = item["id"]
        item = self.selected_material() or item
        self.load_fields(item)
        if saved is not False:
            clear_text(self.detail_error)
        self.material_player.load(self.root / item["path"])
        self.update_hint()
        self.refresh_detail()

    @staticmethod
    def fields(item):
        return (clean_field(item.get("text")), clean_field(item.get("caption")), item["included"])

    def editor_fields(self):
        return (
            clean_field(self.material_text.toPlainText()),
            clean_field(self.material_caption.text()),
            self.included.isChecked(),
        )

    def load_fields(self, item):
        self.material_text.setPlainText(clean_field(item.get("text")))
        set_text(self.material_caption, clean_field(item.get("caption")))
        self.included.setChecked(item["included"])
        self.loaded = self.fields(item)

    def draft_changed(self):
        return bool(self.selected_material()) and self.editor_fields() != self.loaded

    def autosave_draft(self):
        """Keep the edit of the recording being left. Returns None when there was nothing to
        save or the draft is not valid yet, True when saved, False when saving failed."""
        if self.root is None or not self.draft_changed():
            return None
        text = clean_field(self.material_text.toPlainText())
        if not text or len(text) > MAX_TEXT_LENGTH:
            return None
        try:
            self.ensure_editable()
            edit_material(
                self.root,
                self.material_id,
                text=text,
                caption=self.material_caption.text(),
                included=self.included.isChecked(),
            )
        except ValueError as exc:
            set_text(self.detail_error, error_message(exc))
            return False
        self.project = load_project(self.root)
        # Rows and step markers follow once the selection change has finished.
        QTimer.singleShot(0, lambda root=self.root: root == self.root and self.set_project(root))
        return True

    def ensure_editable(self):
        """Recordings are read by a running task of this project; say which one."""
        task = self.host.tasks.current
        if self.root and self.host.project_busy(self.root):
            raise BusinessError(
                t(
                    "training_page.recordings_locked_by_task",
                    task=KIND_TEXT.get(task["kind"], task["kind"]),
                )
            )

    def step_material(self, delta):
        rows = self.materials.rows_model.rows
        position = next((i for i, r in enumerate(rows) if r["id"] == self.material_id), None)
        if position is not None and 0 <= position + delta < len(rows):
            self.materials.setCurrentIndex(self.materials.model().index(position + delta, 0))

    def toggle_included(self, key, checked):
        """The table checkbox changes only the flag; an unsaved text draft stays untouched."""
        root = self.require()
        try:
            self.ensure_editable()
            item = next(i for i in load_project(root)["materials"] if i["id"] == key)
            edit_material(
                root,
                key,
                text=clean_field(item.get("text")),
                caption=clean_field(item.get("caption")),
                included=checked,
            )
            if key == self.material_id:
                self.included.setChecked(checked)
                if self.loaded:
                    self.loaded = (*self.loaded[:2], checked)
        finally:
            # Rows always show the persisted state, also when the change was refused.
            self.set_project(root)

    def save_shortcut(self):
        if self.isVisible() and self.tabs.currentIndex() == 0 and self.save_button.isEnabled():
            self.save_button.click()

    def save_material(self, replacement=None):
        self.require()
        if not self.material_id:
            raise BusinessError(t("training_page.select_a_recording_first"))
        self.ensure_editable()
        edit_material(
            self.root,
            self.material_id,
            text=self.material_text.toPlainText(),
            caption=self.material_caption.text(),
            included=self.included.isChecked(),
            replacement=replacement,
            reviewed=True,
        )
        self.loaded = self.editor_fields()
        clear_text(self.detail_error)
        self.set_project(self.root)
        self.update_hint()

    def replace_material(self):
        self.require()
        self.ensure_editable()
        path, _ = QFileDialog.getOpenFileName(
            self,
            t("training_page.replace_selected_recording"),
            recordings_start(self.host.store.settings),
            t("training_page.audio_wav_flac_ogg_mp3"),
        )
        if path:
            self.save_material(Path(path))
            item = self.selected_material()
            if item:
                self.material_player.load(self.root / item["path"])

    def import_audio(self, with_csv):
        root = self.require()
        self.ensure_editable()
        folder = choose_recordings_folder(self, self.host.store.settings)
        if not folder:
            return
        csv_path = None
        if with_csv:
            path, _ = QFileDialog.getOpenFileName(
                self, t("training_page.choose_transcript_csv"), str(folder), t("files.csv")
            )
            if not path:
                return
            csv_path = Path(path)
        self.start_import(root, lambda: import_materials(root, folder, csv_path))

    def import_transcribed(self):
        root = self.require()
        self.ensure_editable()
        if self.host.tasks.busy:
            raise BusinessError(t("asr.wait_for_the_running_task"))
        choice = ask_transcription(self, self.host.store.settings)
        if choice:
            self.start_transcription(root, **choice)

    def retranscribe_listed(self):
        """Transcribe again what the list shows: all recordings, the flagged ones, or a search."""
        self.retranscribe([row["id"] for row in self.materials.requested_rows])

    def retranscribe(self, ids):
        """Replace the transcripts of these recordings with a new recognition pass."""
        root = self.require()
        self.ensure_editable()
        if not ids:
            raise BusinessError(t("training_page.select_a_recording_first"))
        if self.host.tasks.busy:
            raise BusinessError(t("asr.wait_for_the_running_task"))
        # An unsaved caption or inclusion change is kept; the text is what gets replaced.
        self.autosave_draft()
        choice = ask_transcription(self, self.host.store.settings, again=len(ids))
        if choice:
            self.start_transcription(
                root, None, choice["main"], choice["check"], False, materials=list(ids)
            )

    def start_transcription(self, root, folder, main, check, split, materials=None):
        """Queue the transcription of a folder, or of project recordings again. A backend whose
        add-on is not installed first queues the add-on's installation; the user agreed to both
        in the dialog."""
        self.ensure_editable()
        self.show_import_note(None)
        clear_text(self.material_error)
        backends = asr.selection(main, check)
        payload = {
            "kind": "transcribe",
            "project": str(root),
            "folder": str(folder) if folder else None,
            "materials": materials,
            "main": main,
            "check": check,
            "split": split,
            "addons": asr.required_addons(backends),
        }
        # Each step that is needed runs first and queues the next: add-on, models, transcription.
        if missing := [b.key for b in backends if asr.missing_models([b])]:
            payload = {"kind": "models", "project": str(root), "backends": missing, "then": payload}
        if missing := asr.missing_addons(backends):
            payload = {"kind": "addon", "project": str(root), "keys": missing, "then": payload}
        self.host.tasks.start(payload)

    def transcribed(self, task):
        """Say what a finished transcription did, beside the import menu that started it."""
        if task["kind"] not in {"transcribe", "addon", "models"} or task.get("project") != str(
            self.root
        ):
            return
        if task["status"] in {"failed", "interrupted"}:
            set_text(self.material_error, task["message"])
            return
        result = task.get("result") or {}
        if task["kind"] != "transcribe" or task["status"] != "completed" or not result:
            return
        if result.get("again"):
            note = t("asr.again_summary", count=result["imported"], review=result["needs_review"])
        else:
            note = t(
                "asr.import_summary",
                imported=result["imported"],
                segments=result["segments"],
                files=result["split_files"],
                review=result["needs_review"],
            )
        self.show_import_note(note)

    def show_import_note(self, note):
        # What an import did is its outcome, not the page's state: a toast, never a line
        # that stays and is cut short.
        if note is not None:
            self.host.inform(note)

    def import_example(self):
        root = self.require()
        self.ensure_editable()
        folder = self.host.repo / "examples/desktop"
        self.start_import(root, lambda: import_materials(root, folder, folder / "transcripts.csv"))

    def start_import(self, root, operation):
        self.show_import_note(None)
        clear_text(self.material_error)
        self.host.run_io(
            operation,
            lambda result: self.imported(root, result),
            self.material_error,
            busy=t("app.work_import_materials"),
        )

    def imported(self, root, items):
        """Refresh, then say what the import did, including what it left out and why."""
        self.host.refresh()
        if self.root != root:
            return
        excluded = [item for item in items if not item["included"]]
        note = t("training_page.import_summary", imported=len(items), excluded=len(excluded))
        if excluded:
            note = t(
                "training_page.import_summary_reasons",
                summary=note,
                reasons=problem_summary(excluded),
            )
        self.show_import_note(note)

    def csv_template(self):
        path = save_file(
            self,
            self.host.store.settings,
            t("training_page.save_transcript_template"),
            t("training_page.transcript_template_csv"),
            t("files.csv"),
        )
        if path:
            with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["file_name", "text", "caption"])
                writer.writerow(["example.wav", "こんにちは。今日はいい天気ですね。", "自然な声"])

    def rename(self):
        self.require()
        name, accepted = prompt_text(
            self,
            t("training_page.rename_project"),
            t("app.project_name"),
            value=self.project["name"],
        )
        if accepted and name.strip():
            with project_lock(self.root):
                project = load_project(self.root)
                project["name"] = name.strip()
                save_project(self.root, project)
            self.host.refresh()

    # ----- preprocessing -----

    def build_initialization(self):
        columns = QHBoxLayout()
        columns.setSpacing(14)
        self.init_page.body.addLayout(columns, 1)
        left, body = card()
        columns.addWidget(left, 27)

        self.init_icon = QLabel()
        self.init_icon.setFixedSize(32, 32)
        self.init_title = elided_label(strong=True)
        self.init_subtitle = elided_label()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(self.init_title)
        titles.addWidget(self.init_subtitle)
        head = QHBoxLayout()
        head.setSpacing(12)
        head.addWidget(self.init_icon, 0, Qt.AlignmentFlag.AlignTop)
        head.addLayout(titles, 1)
        body.addLayout(head)

        self.stats = {}
        stats = []
        for key, title in (
            ("included", t("training_page.include_in_training")),
            ("duration", t("app.total_duration")),
            ("excluded", t("training_page.excluded")),
            ("problems", t("training_page.stat_problems")),
        ):
            self.stats[key] = Metric(title)
            stats.append(self.stats[key])
        body.addWidget(grid(*stats, columns=4))

        self.step_check = StepRow(1, t("training_page.step_check"))
        self.step_encode = StepRow(2, t("training_page.step_encode"), progress=True)
        self.step_write = StepRow(
            3, t("training_page.step_write"), t("training_page.step_write_hint")
        )
        self.init_progress = self.step_encode.bar
        steps = QVBoxLayout()
        steps.setSpacing(16)
        for step in (self.step_check, self.step_encode, self.step_write):
            steps.addWidget(step)
        body.addSpacing(4)
        body.addLayout(steps)

        # How the last run ended when it did not finish: what went wrong, at which recording,
        # and the ways on from there.
        self.failure = QWidget()
        failure = QVBoxLayout(self.failure)
        failure.setContentsMargins(0, 4, 0, 0)
        failure.setSpacing(8)
        self.init_error = label("", "error")
        self.init_error.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        failure.addWidget(self.init_error)
        self.failure_open = button(
            t("training_page.open_failed_recording"),
            lambda: self.host.perform(self.open_failed),
            icon=FluentIcon.EDIT,
        )
        self.failure_exclude = button(
            t("training_page.exclude_failed_recording"),
            lambda: self.host.perform(self.exclude_failed),
            icon=FluentIcon.REMOVE_FROM,
        )
        self.failure_log = button(
            t("training_page.view_log"), lambda: self.host.perform(self.preprocess_log)
        )
        failure.addWidget(SplitRow(row(self.failure_open, self.failure_exclude), self.failure_log))
        self.failure.hide()
        body.addWidget(self.failure)
        body.addStretch(1)

        self.init_button = button(
            t("training_page.initialize_data"),
            lambda: self.host.perform(self.initialize, self.init_toast),
            primary=True,
        )
        self.init_toast = error_label()
        self.init_stop = button(t("training_page.stop_initialization"), self.host.tasks.stop)
        self.init_stop.setEnabled(False)
        self.goto_training = button(
            t("training_page.training_tab"),
            lambda: self.tabs.setCurrentIndex(2),
            icon=FluentIcon.RIGHT_ARROW,
        )
        self.review_note = label("", "muted")
        body.addWidget(self.review_note)
        self.init_notice = ModelNotice(lambda: self.host.show_guide())
        body.addWidget(self.init_notice)
        body.addWidget(
            row(self.init_stop, (self.init_toast, 1), self.goto_training, self.init_button)
        )

        right, issues = card()
        columns.addWidget(right, 20)
        self.exclude_all = button(
            t("training_page.exclude_all_problems"),
            lambda: self.host.perform(self.exclude_problems),
            icon=FluentIcon.REMOVE_FROM,
        )
        issues.addWidget(card_header(t("training_page.problem_recordings"), self.exclude_all))
        holder = QWidget()
        self.issue_list = QVBoxLayout(holder)
        self.issue_list.setContentsMargins(0, 0, 0, 0)
        self.issue_list.setSpacing(4)
        self.issue_list.setAlignment(Qt.AlignmentFlag.AlignTop)
        issues.addWidget(inner_scroll(holder), 1)
        self.issue_empty = label(t("training_page.no_issues"), "muted")
        self.issue_list.addWidget(self.issue_empty)
        # What preprocessing makes and what running it again changes: sentences, wrapped.
        self.data_note = label("", "muted")
        issues.addWidget(self.data_note)

    def last_attempt(self):
        """How the last preprocessing that did not finish ended, while it still applies."""
        attempt = (self.project or {}).get("last_attempt")
        return attempt if attempt and not self.ready else None

    def problem_items(self):
        """Included recordings preprocessing would refuse, with the reason: known problems and
        those the last check found."""
        items = (self.project or {}).get("materials", [])
        found = {item["id"]: (item, problem_of(item)) for item in items if problem_of(item)}
        attempt = self.last_attempt()
        if attempt and attempt.get("phase") == "check":
            by_id = {item["id"]: item for item in items}
            for entry in attempt.get("items", []):
                key = entry["id"] if isinstance(entry, dict) else entry
                reason = entry.get("reason") if isinstance(entry, dict) else None
                if key in by_id and key not in found:
                    found[key] = (by_id[key], reason or t("training_page.check_failed_reason"))
        return [entry for entry in found.values() if entry[0]["included"]]

    def refresh_issues(self):
        while self.issue_list.count() > 1:
            widget = self.issue_list.takeAt(1).widget()
            if widget:
                widget.deleteLater()
        problems = self.problem_items()
        self.issue_empty.setVisible(not problems)
        self.exclude_all.setEnabled(bool(problems) and not self.host.project_busy(self.root))
        for item, reason in problems:
            names = QVBoxLayout()
            names.setSpacing(0)
            names.addWidget(elided_label(item["file_name"]))
            names.addWidget(label(reason, "muted"))
            actions = [
                icon_button(
                    FluentIcon.SYNC if item.get("error") else FluentIcon.EDIT,
                    t("training_page.replace_short")
                    if item.get("error")
                    else t("training_page.edit"),
                    lambda _=False, i=item: self.fix(i, bool(i.get("error"))),
                ),
                icon_button(
                    FluentIcon.REMOVE_FROM,
                    t("training_page.exclude_recording"),
                    lambda _=False, i=item: self.host.perform(lambda: self.exclude([i["id"]])),
                ),
            ]
            line = QWidget()
            layout = QHBoxLayout(line)
            layout.setContentsMargins(0, 4, 0, 4)
            layout.setSpacing(4)
            layout.addLayout(names, 1)
            for action in actions:
                layout.addWidget(action, 0, Qt.AlignmentFlag.AlignTop)
            self.issue_list.addWidget(line)

    def fix(self, item, replace):
        """Open the dataset on this recording; for a damaged file go straight to replacing it."""
        self.tabs.setCurrentIndex(0)
        self.filter.clear()
        self.only_problems.setChecked(False)
        for position, candidate in enumerate(self.materials.rows_model.rows):
            if candidate["id"] == item["id"]:
                self.materials.setCurrentIndex(self.materials.model().index(position, 0))
        if replace:
            self.host.perform(self.replace_material, self.detail_error)

    def exclude(self, ids):
        """Leave these recordings out of training (they stay in the project)."""
        root = self.require()
        self.ensure_editable()
        with project_lock(root):
            project = load_project(root)
            changed = False
            for item in project["materials"]:
                if item["id"] in ids and item["included"]:
                    item["included"] = False
                    item.pop("auto_excluded", None)
                    changed = True
            if changed:
                project["revision"] += 1
                save_project(root, project)
        self.set_project(root)

    def exclude_problems(self):
        self.exclude([item["id"] for item, _reason in self.problem_items()])

    def failed_item(self):
        attempt = self.last_attempt() or {}
        key = attempt.get("item")
        return next((i for i in self.project["materials"] if i["id"] == key), None)

    def open_failed(self):
        if item := self.failed_item():
            self.fix(item, False)

    def exclude_failed(self):
        if item := self.failed_item():
            self.exclude([item["id"]])

    def preprocess_log(self):
        """The log of this project's last preprocessing task."""
        for task in self.host.tasks.tasks:
            if task["kind"] == "initialize" and task.get("project") == str(self.root):
                path = self.host.tasks.task_dir(task) / "worker.log"
                text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
                show_log(
                    self,
                    t("settings_page.task_diagnostics"),
                    text[-100000:] or t("settings_page.no_log_output_yet"),
                )
                return
        raise BusinessError(t("settings_page.no_log_output_yet"))

    def init_task(self):
        """This project's preprocessing task while it runs, else None."""
        task = self.host.tasks.current
        if (
            task
            and self.host.tasks.busy
            and task["kind"] == "initialize"
            and self.root
            and task.get("project") == str(self.root)
        ):
            return task
        return None

    def refresh_initialization(self):
        """State, statistics, steps and failure of the preprocessing tab."""
        task = self.init_task()
        init = (self.project or {}).get("initialization")
        items = (self.project or {}).get("materials", [])
        included = [i for i in items if i["included"]]
        problems = self.problem_items()
        attempt = self.last_attempt()
        self.stats["included"].set(t("training_page.recording_count", n=len(included)))
        self.stats["duration"].set(clock(sum(i["duration"] for i in included)))
        self.stats["excluded"].set(t("training_page.recording_count", n=len(items) - len(included)))
        self.stats["problems"].set(t("training_page.recording_count", n=len(problems)))
        unreviewed = sum(needs_review(i) for i in included)
        self.review_note.setVisible(bool(unreviewed) and not task)
        if unreviewed:
            set_text(self.review_note, t("asr.review_remaining", count=unreviewed))
        last = (
            t(
                "training_page.last_completed",
                n=init["count"],
                date=fmt(self.host.date(init["created"]), "", -1),
            )
            if init and init.get("status") == "completed"
            else ""
        )
        phase = task.get("phase") if task else None
        if task:
            state = "active"
            title = t("training_page.preprocessing_running")
            subtitle = (
                t(
                    "training_page.encoding_item",
                    step=task.get("step", 0),
                    total=task.get("total", 0),
                    file=fmt(task.get("item", ""), "", -1),
                )
                if phase == "encode" and task.get("total")
                else task["message"]
            )
        elif self.ready:
            state, title, subtitle = "done", t("training_page.initialized_and_ready_to_train"), last
        elif problems:
            state = "warn"
            title = t("training_page.problems_block", n=len(problems))
            subtitle = t("training_page.problems_block_hint")
        elif not included:
            state = "pending"
            title = t("training_page.nothing_included")
            subtitle = (
                t("training_page.include_some_hint")
                if items
                else t("training_page.import_recordings_to_begin")
            )
        elif attempt and attempt["status"] == "failed":
            state = "failed"
            title = t("training_page.preprocessing_failed")
            subtitle = t("training_page.failed_at", date=fmt(self.host.date(attempt["created"])))
        elif attempt and attempt["status"] == "stopped":
            state = "pending"
            title = t("training_page.preprocessing_stopped")
            subtitle = t("training_page.stopped_hint")
        elif init:
            state, title = "warn", t("training_page.recordings_changed_initialize_again")
            subtitle = last
        else:
            state = "pending"
            title = t("training_page.prepare_recordings_before_initialization")
            subtitle = t("training_page.initialize_hint")
        set_text(self.init_title, title)
        set_text(self.init_subtitle, subtitle)
        accent_drawn(
            self.init_icon,
            lambda: self.init_icon.setPixmap(step_icon(None, state, size=28).pixmap(28, 28)),
        )
        self.show_steps(task, attempt)
        failed = bool(attempt and attempt["status"] == "failed" and not task)
        self.failure.setVisible(failed)
        if failed:
            set_text(self.init_error, attempt["error"])
            item = self.failed_item()
            self.failure_open.setVisible(item is not None)
            self.failure_exclude.setVisible(item is not None and item["included"])
        size = (
            sum(
                path.stat().st_size
                for path in (self.root / "initialized").rglob("*")
                if path.is_file()
            )
            if self.root
            else 0
        )
        set_text(
            self.data_note,
            t(
                "training_page.preprocess_facts",
                size=fmt(t("common.megabytes", value=fmt(size / 1024**2, ".1f", -1)), "", -1),
            )
            if size
            else t("training_page.preprocess_facts_empty"),
        )
        speech = model_files.speech_ready(self.host.resource_root)
        self.init_notice.setVisible(not speech)
        busy = self.host.tasks.busy
        # Data that matches the recordings needs no second run: the way on is training.
        self.init_button.setVisible(task is None and not self.ready)
        self.init_stop.setVisible(task is not None)
        self.init_stop.setEnabled(task is not None)
        self.init_button.setEnabled(bool(self.root) and not busy and speech and bool(included))
        set_text(
            self.init_button,
            t("training_page.initialize_again")
            if init or attempt
            else t("training_page.initialize_data"),
        )
        self.goto_training.setVisible(self.ready and task is None)
        self.refresh_issues()

    def show_steps(self, task, attempt):
        """Markers driven by what the worker reports: the encoding count is the task's own
        step/total; checking and writing switch on as the worker reaches them."""
        steps = {"check": self.step_check, "encode": self.step_encode, "write": self.step_write}
        order = list(steps)
        details = {"check": "", "encode": "", "write": ""}
        count, total = 0, 0
        if task:
            phase = task.get("phase") or "check"
            reached = order.index(phase)
            states = {
                name: "done" if index < reached else "active" if index == reached else "pending"
                for index, name in enumerate(order)
            }
            total = task.get("total", 0) if phase != "check" else 0
            count = task.get("step", 0) if phase != "check" else 0
            if reached > 0:
                details["check"] = t("training_page.step_check_done", count=total)
        elif self.ready:
            states = dict.fromkeys(order, "done")
            total = count = self.project["initialization"]["count"]
            details["check"] = t("training_page.step_check_done", count=count)
            details["encode"] = t("training_page.step_encode_done")
        elif attempt and attempt["status"] == "failed":
            reached = order.index(attempt.get("phase") or "check")
            states = {
                name: "done" if index < reached else "failed" if index == reached else "pending"
                for index, name in enumerate(order)
            }
            details[order[reached]] = t("training_page.step_failed_here")
        else:
            states = dict.fromkeys(order, "pending")
        for name, step in steps.items():
            step.set_state(states[name])
            step.set_detail(details[name] or (step.hint if name == "write" else ""))
        set_text(self.step_encode.count, f"{count} / {total}" if total else "")
        self.init_progress.setRange(0, max(1, total))
        self.init_progress.setValue(count)

    def initialize(self, then=None):
        """Start preprocessing; then = a training run to start once it has finished."""
        self.require()
        if self.host.tasks.busy:
            raise BusinessError(t("asr.wait_for_the_running_task"))
        included = [item for item in self.project["materials"] if item["included"]]
        if not included:
            raise BusinessError(t("materials.import_and_include_at_least_one_recording"))
        # Known problems are listed beside the steps; a short line points there.
        if problems := self.problem_items():
            self.tabs.setCurrentIndex(1)
            first, reason = problems[0]
            raise BusinessError(
                t(
                    "materials.preprocess_blocked",
                    n=len(problems),
                    file=fmt(first["file_name"], "", -1),
                    reason=fmt(reason, "", -1),
                )
            )
        rebuilding = (
            self.project.get("initialization") is not None
            or (self.root / "initialized").exists()
            or (self.root / "temp/initialization").exists()
        )
        if (
            rebuilding
            and then is None
            and not self.host.confirm(
                t("training_page.initialize_data_again"),
                t("training_page.replace_existing_training_data_using_the_current"),
                t("training_page.initialize_again"),
            )
        ):
            return
        payload = {"kind": "initialize", "project": str(self.root), "rebuild": rebuilding}
        if then is not None:
            payload["train"] = then
        self.host.tasks.start(payload)

    # ----- LoRA training -----

    def build_training(self):
        # The runs are listed on top; under them, the selected run, or the form for a new one.
        self.new_run_button = button(t("training_page.new_run"), self.new_run, icon=FluentIcon.ADD)
        _records, record_body = card(self.train_page.body)
        record_body.addWidget(card_header(t("training_page.training_history"), self.new_run_button))
        self.runs = Table(
            [
                ("name", t("library_page.voice_name"), 0),
                ("steps", t("app.steps"), 130),
                ("state", t("app.status"), 190),
                ("created_text", t("training_page.created_at"), 160),
            ],
            fill=True,
            stretch=0,
            optional=("created_text",),
        )
        self.runs.selected.connect(self.select_run)
        # Right-click a record for the same actions as the selected run's More menu.
        self.runs.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.runs.customContextMenuRequested.connect(self.run_context_menu)
        record_body.addWidget(self.runs)
        _lower, lower = card(self.train_page.body, stretch=1)
        self.run_stack = QStackedWidget()
        lower.addWidget(self.run_stack)
        self.build_run_form()
        self.build_run_detail()
        for widget in self.parameters.values():
            widget.valueChanged.connect(self.update_config_summary)
        self.preset.currentIndexChanged.connect(self.apply_preset)
        self.update_config_summary()

    def build_run_form(self):
        panel = QWidget()
        self.run_stack.addWidget(panel)
        footer = QVBoxLayout(panel)
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(10)
        footer.addWidget(card_header(t("training_page.new_run")))
        # The parameters scroll when the window is short; the summary and Start stay in view.
        form = QWidget()
        footer.addWidget(inner_scroll(form), 1)
        body = QVBoxLayout(form)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(12)
        self.voice_name = smooth(LineEdit())
        # The base a new run trains on; the run keeps it for good.
        self.run_base = combo_box()
        self.preset = combo_box()
        add_item(self.preset, t("training_page.recommended"), userData={})
        for name, config in self.host.store.settings.get("presets", {}).items():
            add_item(self.preset, name, userData=config)
        body.addWidget(
            grid(
                field(t("library_page.voice_name"), self.voice_name),
                field(
                    t("training_page.training_preset"),
                    row(
                        self.preset,
                        icon_button(
                            FluentIcon.SAVE_AS,
                            t("training_page.save_custom_preset"),
                            self.save_preset,
                            size=32,
                        ),
                        spacing=4,
                    ),
                    buddy=self.preset,
                ),
                field(t("generation_page.base_model"), self.run_base),
            )
        )
        # The settings that depend on the card and the recordings start empty. While one is,
        # this line stands out and asks for the recommendation; then it says what it rests on.
        self.recommendation_note = label("")
        # Reserve the summary and review-note lines so completing the measurement does
        # not push the parameter controls down. Text still wraps at the available width.
        self.recommendation_note.setMinimumHeight(
            self.recommendation_note.fontMetrics().lineSpacing() * 5
        )
        self.recommendation_basis = None
        self.recommended_values = None
        self.recommendation_data = None
        self.recommendation_request = None
        self.recommend_button = button(
            t("training_page.generate_recommended_settings"),
            lambda: self.host.perform(self.generate_recommendation, self.start_error),
            icon=FluentIcon.SPEED_HIGH,
        )
        self.recommendation = Callout(
            row(
                (self.recommendation_note, 1),
                self.recommend_button,
                spacing=12,
            )
        )
        body.addWidget(self.recommendation)
        self.parameters = {}
        common, extra = [], []
        choices = {
            "gradient_checkpointing": t("training_page.gradient_checkpointing_hint"),
            "length_bucket_enabled": t("training_page.length_grouping_hint"),
        }
        for key, title, maximum, main in (
            ("max_steps", t("library_page.training_steps"), 1000000, True),
            ("learning_rate", t("training_page.learning_rate"), None, True),
            ("batch_size", t("training.batch"), 64, True),
            ("gradient_accumulation_steps", t("training_page.gradient_accumulation"), 128, True),
            ("lora_r", t("training.rank"), 256, True),
            ("lora_alpha", t("training.alpha"), 512, True),
            ("gradient_checkpointing", t("training_page.gradient_checkpointing"), None, True),
            ("length_bucket_enabled", t("training_page.length_grouping"), None, True),
            ("save_every", t("training_page.save_interval"), 1000000, False),
            ("valid_every", t("training_page.validation_interval"), 100000, False),
            ("warmup_steps", t("training_page.warmup_steps"), 100000, False),
            ("seed", t("generation_page.seed"), 2147483647, False),
        ):
            if key in choices:
                widget = ChoiceBox()
                add_item(widget, t("training_page.option_on"), userData=True)
                add_item(widget, t("training_page.option_off"), userData=False)
                bind(widget, choices[key], "setToolTip")
                if key in TUNED:
                    bind(widget, t("training_page.not_set_yet"), "setPlaceholderText")
            elif maximum is None:
                widget = smooth(LearningRateBox())
                widget.setRange(0.000001, 0.1)
                widget.setSingleStep(0.00001)
            elif key in TUNED:
                # 0 is "not set yet", shown empty.
                widget = smooth(OptionalSpinBox())
                widget.setRange(0, maximum)
                bind(widget.lineEdit(), t("training_page.not_set_yet"), "setPlaceholderText")
            else:
                widget = smooth(SpinBox())
                widget.setRange(0 if key in {"seed", "warmup_steps", "valid_every"} else 1, maximum)
            widget.setValue(self.defaults()[key])
            if key not in choices:
                widget.setSymbolVisible(False)
            self.parameters[key] = widget
            widget.valueChanged.connect(self.update_recommendation)
            (common if main else extra).append(field(title, widget))
        self.budget_steps = self.parameters["max_steps"].value()
        self.parameters["max_steps"].valueChanged.connect(self.update_budget_schedule)
        self.update_recommendation()
        body.addWidget(grid(*common, columns=3))
        advanced = QWidget()
        advanced_layout = QVBoxLayout(advanced)
        advanced_layout.setContentsMargins(0, 6, 0, 0)
        advanced_layout.addWidget(grid(*extra, columns=4))
        self.advanced = Expand(
            t("generation_page.advanced_settings"), advanced, self.host.reduced_motion
        )
        body.addWidget(self.advanced)
        body.addStretch(1)
        self.config_summary = label("", "muted")
        footer.addWidget(self.config_summary)
        # Recordings not preprocessed yet are preprocessed first, by the same button.
        self.prepare_note = label("", "muted")
        footer.addWidget(self.prepare_note)
        self.start_error = error_label()
        self.start_note = label("", "muted", wrap=False)
        self.start_button = button(
            t("training_page.start_training"),
            lambda: self.host.perform(self.start_training, self.start_error),
            primary=True,
            icon=FluentIcon.PLAY_SOLID,
        )
        self.start_button.setEnabled(False)
        self.train_notice = ModelNotice(lambda: self.host.show_guide())
        footer.addWidget(self.train_notice)
        footer.addWidget(row((self.start_error, 1), self.start_note, self.start_button))

    def build_run_detail(self):
        panel = QWidget()
        self.run_stack.addWidget(panel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.run_title = elided_label(strong=True)
        self.use_button = button(
            t("library_page.use_for_generation"),
            lambda: self.host.perform(self.use_run, self.train_error),
            primary=True,
            icon=AppIcon.INFERENCE,
        )
        self.stop_button = button(t("app.save_and_stop"), self.host.tasks.stop)
        self.stop_button.setEnabled(False)
        self.resume_button = button(
            t("training_page.resume_training"),
            lambda: self.host.perform(self.resume_training, self.train_error),
        )
        self.resume_button.setEnabled(False)
        # A finished run trains on from where it ended (a new total and save interval).
        self.extend_button = button(
            t("training_page.train_more"),
            lambda: self.host.perform(self.train_more, self.train_error),
        )
        self.extend_button.hide()
        self.more_button = icon_button(
            FluentIcon.MORE, t("library_page.more"), self.run_menu, size=32
        )
        layout.addWidget(
            row(
                (self.run_title, 1),
                self.stop_button,
                self.resume_button,
                self.extend_button,
                self.use_button,
                self.more_button,
            )
        )
        # Why the run cannot continue, or why it failed: sentences, on lines of their own.
        self.resume_note = label("", "muted")
        self.resume_note.hide()
        layout.addWidget(self.resume_note)
        self.train_error = error_label(wrap=True)
        layout.addWidget(self.train_error)
        self.metrics = {}
        for key, title in (
            ("step", t("app.steps")),
            ("loss", t("training_page.metric_loss")),
            ("lr", t("training_page.learning_rate")),
            ("elapsed", t("training_page.metric_elapsed")),
            ("eta", t("training_page.metric_eta")),
        ):
            self.metrics[key] = Metric(title)
        layout.addWidget(grid(*self.metrics.values(), columns=5))
        self.train_progress = ProgressBar()
        self.train_progress.setFixedHeight(6)
        layout.addWidget(self.train_progress)
        self.loss_chart = LossChart()
        self.loss_chart.chosen.connect(self.checkpoint_chosen)
        layout.addWidget(self.loss_chart, 1)
        # The checkpoint picked on the curve: what it is and what can be done with it, then
        # its sample's text and audio. The same height whether or not it has a sample.
        self.checkpoint_title = elided_label(strong=True)
        self.sample_button = button(
            t("library_page.generate_preview"),
            lambda: self.host.perform(self.make_sample, self.train_error),
            icon=FluentIcon.PLAY,
        )
        self.register_button = button(
            t("training_page.register_this_checkpoint"),
            # A checkpoint kept apart stays when training later removes older ones.
            lambda: self.host.perform(
                lambda: self.register_checkpoint(self.chosen_checkpoint()), self.train_error
            ),
            icon=FluentIcon.ADD_TO,
        )
        layout.addWidget(row((self.checkpoint_title, 1), self.sample_button, self.register_button))
        self.sample_text = elided_label()
        # Empty without a sample, but the line keeps its height: switching checkpoints must
        # not move the player.
        self.sample_text.setFixedHeight(self.sample_text.fontMetrics().height())
        layout.addWidget(self.sample_text)
        self.sample_player = Player()
        self.sample_empty = label("", "muted", wrap=False)
        self.sample_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sample_stack = QStackedWidget()
        self.sample_stack.addWidget(self.sample_player)
        self.sample_stack.addWidget(self.sample_empty)
        self.sample_stack.setFixedHeight(52)
        layout.addWidget(self.sample_stack)
        # Switching checkpoints while a sample plays continues at the same moment, so the
        # same words can be compared.
        self.pending_seek = 0
        self.sample_player.player.mediaStatusChanged.connect(self.seek_pending)
        self.checkpoint_run = None
        self.checkpoints_shown = {}
        self.checkpoint_samples = {}
        self.checkpoint_texts = {}
        self.sample_sources = {}
        # How to pick, how often checkpoints are saved and which are kept. One line (elided,
        # full text as tooltip): a wrapping label reserved several lines' height.
        self.checkpoint_note = elided_label()
        self.checkpoint_note.setTextColor(QColor("#5b6068"), QColor("#bfc5ce"))
        layout.addWidget(self.checkpoint_note)

    def new_run(self):
        """Show the form for a new run (no run selected)."""
        self.runs.setCurrentIndex(QModelIndex())
        self.runs.clearSelection()
        self.show_run()
        # A name no run of this project has yet, so the new run is told apart from the others.
        name = self.voice_name.text().strip() or self.project["name"]
        set_text(self.voice_name, unique_run_name(self.root, name))
        self.voice_name.setFocus()

    def values(self):
        return {key: widget.value() for key, widget in self.parameters.items()}

    def update_config_summary(self):
        count = sum(item["included"] for item in self.project["materials"]) if self.project else 0
        try:
            config = normalize_preset(self.values(), max(count, 1))
        except ValueError as exc:
            set_text(self.config_summary, error_message(exc))
            return
        per_step = config["batch_size"] * config["gradient_accumulation_steps"]
        training = training_count(count, config["valid_every"] > 0, config["valid_ratio"])
        epochs = config["max_steps"] * per_step / training if training else 0
        text = t(
            "training_page.epoch_hint",
            samples=fmt(per_step, "", -1),
            epochs=fmt(round(epochs), "", -1),
        )
        if count and training <= per_step:
            text = text + t("training_page.epoch_hint_few", count=count)
        set_text(self.config_summary, text if count else "")

    @staticmethod
    def defaults():
        """The recommendation with the settings that depend on the card and the recordings
        left empty: 0 for numbers, None for on/off."""
        return {
            **RECOMMENDED,
            **{key: None if key == "gradient_checkpointing" else 0 for key in TUNED},
        }

    def generate_recommendation(self):
        """Generate a metadata-only plan in the background worker."""
        self.require()
        if self.host.tasks.busy:
            return
        if not self.included_count():
            raise BusinessError(t("training_page.nothing_included"))
        settings = self.values()
        # Without an installed base the plan is the default base's.
        base = next((b for b in self.host.bases() if b.id == self.run_base.currentData()), None)
        clear_text(self.start_error)
        task = self.host.tasks.start(
            {
                "kind": "recommend",
                "project": str(self.root),
                "settings": settings,
                "base": base_models.base_record(base) if base else None,
            }
        )
        self.recommendation_request = (task["id"], self.root, settings, base and base.id)
        self.update_recommendation()

    def recommendation_finished(self, task):
        request = self.recommendation_request
        if task["kind"] != "recommend" or not request or task["id"] != request[0]:
            return
        self.recommendation_request = None
        if self.root != request[1]:
            return
        if task["status"] != "completed":
            self.update_recommendation()
            if task["status"] in {"failed", "interrupted"}:
                set_text(self.start_error, task["message"])
            return
        result = task["result"]
        if (
            self.values() != request[2]
            or self.run_base.currentData() != request[3]
            or materials_fingerprint(self.project) != result["fingerprint"]
        ):
            set_text(self.start_error, t("training_page.recommendation_changed"))
            self.update_recommendation()
            return
        values = result["values"]
        profile = result["profile"]
        for key, value in values.items():
            self.parameters[key].setValue(value)
        self.recommendation_data = result
        self.recommended_values = self.values()
        self.recommendation_basis = t(
            "training_page.quick_recommendation",
            card=result["gpu"],
            count=profile["count"],
            duration=human_duration(profile["total_seconds"]),
            median=fmt(profile["median_seconds"], ".1f", -1),
            p90=fmt(profile["p90_seconds"], ".1f", -1),
            batch=values["batch_size"],
            accumulation=values["gradient_accumulation_steps"],
            checkpointing=t("training_page.option_on")
            if values["gradient_checkpointing"]
            else t("training_page.option_off"),
            steps=values["max_steps"],
            memory=(
                fmt(result["memory_budget_bytes"] / 1024**3, ".1f", -1)
                if result["memory_budget_bytes"] is not None
                else "—"
            ),
            training_count=result["training_count"],
        )
        duplicates = profile["count"] - profile["distinct_texts"]
        if duplicates or result["review_count"]:
            self.recommendation_basis += t(
                "training_page.recommendation_material_notes",
                duplicates=duplicates,
                review=result["review_count"],
            )
        if result.get("memory_short"):
            self.recommendation_basis += t(
                "training_page.recommendation_memory_short",
                need=fmt(result["base_memory_gib"][0], ".1f", -1),
            )
        self.update_recommendation()

    def update_budget_schedule(self, steps):
        """Follow an edited budget while keeping independently edited schedule fields.

        Presets and drafts load their explicit schedule after max_steps, preserving saved
        values. Generating a recommendation explicitly replaces the whole schedule.
        """
        previous = budget_schedule(self.budget_steps)
        self.budget_steps = steps
        for key, value in budget_schedule(steps).items():
            if self.parameters[key].value() == previous[key]:
                self.parameters[key].setValue(value)

    def update_recommendation(self, *_):
        """Tinted, asking for the recommendation, while a TUNED setting is empty; plain after,
        with what the recommendation rests on while the settings are still the recommended
        ones."""
        # Called while the form is being constructed, before every parameter exists.
        values = {key: widget.value() for key, widget in self.parameters.items()}
        if any(key not in values for key in TUNED):
            return
        empty = values["gradient_checkpointing"] is None or not all(
            values[key] for key in TUNED if key != "gradient_checkpointing"
        )
        self.recommendation.set_tinted(empty)
        current = self.host.tasks.current
        if (
            self.host.tasks.busy
            and current
            and current["kind"] == "recommend"
            and current.get("project") == str(self.root)
        ):
            set_text(self.recommendation_note, t("training_page.recommendation_running"))
            return
        if empty:
            set_text(self.recommendation_note, t("training_page.recommendation_pending"))
            self.recommendation_note.setTextColor(QColor("#1b1b1b"), QColor("#ffffff"))
            return
        recommended = (
            values == self.recommended_values
            and self.recommendation_basis
            and self.recommendation_data
            and self.project
            and materials_fingerprint(self.project) == self.recommendation_data["fingerprint"]
        )
        set_text(
            self.recommendation_note,
            self.recommendation_basis if recommended else t("training_page.settings_filled"),
        )
        self.recommendation_note.setTextColor(QColor("#5b6068"), QColor("#bfc5ce"))

    def apply_preset(self):
        values = {**self.defaults(), **(self.preset.currentData() or {})}
        for key, widget in self.parameters.items():
            widget.setValue(values[key])

    def save_preset(self):
        name, accepted = prompt_text(
            self, t("training_page.save_training_preset"), t("training_page.preset_name")
        )
        name = name.strip()
        if not accepted or not name:
            return
        presets = self.host.store.settings.setdefault("presets", {})
        if name in presets and not self.host.confirm(
            t("training_page.overwrite_preset_title"),
            t("training_page.overwrite_preset_message", name=name),
            t("training_page.overwrite_preset_action"),
        ):
            return
        values = self.values()
        presets[name] = values
        self.host.store.save()
        # Index 0 is the built-in recommendation; saved presets follow by name.
        index = next(
            (i for i in range(1, self.preset.count()) if self.preset.itemText(i) == name), None
        )
        if index is None:
            add_item(self.preset, name, userData=values)
        else:
            self.preset.setItemData(index, values)

    def refresh(self):
        """Read the shown project again; the models page decides which project that is."""
        if self.root is not None:
            try:
                self.set_project(self.root)
            except (OSError, ValueError):
                self.root = self.project = None

    def set_project(self, root):
        changed = root != self.root
        if changed and self.root:
            self.host.project_states[self.project["id"]] = self.snapshot()
        self.root = root
        self.project = load_project(root)
        self.refresh_bases()
        if changed:
            self.material_id = None
            self.material_text.clear()
            self.loaded = None
            self.material_caption.clear()
            self.restore(self.host.project_states.get(self.project["id"], {}))
        set_text(self.title, self.project["name"])
        self.ready = initialization_current(self.project)
        self.refresh_materials()
        self.run_cache = self.run_rows()
        self.runs.update_rows(self.run_cache)
        if changed:
            state = self.host.project_states.get(self.project["id"], {})
            current = self.host.tasks.current
            active_id = (
                next(
                    (item["id"] for item in self.run_cache if item["run_root"] == current["run"]),
                    None,
                )
                if self.host.tasks.busy
                and current
                and current.get("run")
                and current.get("project") == str(root)
                else None
            )
            selected = active_id or state.get("run_id")
            if not selected and "run_id" not in state and self.run_cache:
                selected = self.run_cache[0]["id"]
            self.runs.setCurrentIndex(QModelIndex())
            self.runs.clearSelection()
            if selected:
                self.select_run_id(selected)
        self.size_runs()
        self.update_config_summary()
        self.refresh_initialization()
        self.refresh_steps()
        self.update_summary()
        self.refresh_controls()
        self.show_run()

    def run_rows(self):
        rows = []
        for path in sorted((self.root / "runs").glob("*/run.json"), reverse=True):
            run = read_json(path)
            total = run["config"]["train"]["max_steps"]
            rows.append(
                {
                    **run,
                    "total": total,
                    "steps": f"{run['step']} / {total}",
                    "created_text": self.host.date(run["created"]),
                    "run_root": str(path.parent),
                }
            )
        # Newest first (folder names are random IDs, not dates).
        rows.sort(key=lambda item: str(item.get("created", "")), reverse=True)
        for item in rows:
            self.describe_run(item)
        return rows

    def describe_run(self, item):
        status = item["status"]
        text = STATE_TEXT.get(status, status)
        if status in {"stopped", "interrupted", "failed"} and self.can_resume(item):
            text = t("training_page.state_resumable", state=fmt(text, "", -1))
        interlude = self.host.tasks.training_preview
        if interlude and interlude.training["run"] == item["run_root"]:
            text = t("tasks.preview_run_state")
        item["state"] = text
        # One meaning per color everywhere: done green, interrupted orange, failed red.
        item["tones"] = {"state": STATE_TONES.get(status, "muted")}

    def size_runs(self):
        # Up to four rows without scrolling; a short window keeps fewer, so the run or the
        # form under the list keeps the height it needs (the list scrolls inside).
        rows = len(self.run_cache)
        height = self.train_page.viewport().height()
        shown = 4 if height >= 640 else 2 if height >= 520 else 1
        self.runs.setMinimumHeight(44 + 40)
        self.runs.setMaximumHeight(44 + max(1, min(rows, shown)) * 40)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.root is not None:
            self.size_runs()

    def can_resume(self, item):
        init = (self.project or {}).get("initialization")
        return bool(
            item.get("checkpoint")
            and item["status"] != "completed"
            and self.ready
            and init
            and item.get("initialization_id") == init["id"]
        )

    def select_run(self, item):
        clear_text(self.train_error)
        set_text(self.train_error, item.get("error") or item.get("preview_error", ""))
        self.refresh_controls()
        self.show_run()

    def active_run(self):
        """The task, if it is a training task on the run selected in the list."""
        task = self.host.tasks.current
        item = self.runs.item()
        if (
            task
            and self.host.tasks.busy
            and task["kind"] == "train"
            and item
            and task.get("run") == item["run_root"]
        ):
            return task
        return None

    def show_run(self):
        item = self.runs.item()
        self.run_stack.setCurrentIndex(1 if item else 0)
        self.new_run_button.setEnabled(bool(item))
        if not item:
            self.refresh_controls()
            return
        base = base_models.normalize_record(item.get("base")) or {}
        name = base.get("name") or base.get("repo")
        set_text(
            self.run_title,
            t("common.task_state", name=fmt(item["name"], "", -1), state=fmt(name, "", -1))
            if name
            else item["name"],
        )
        task = self.active_run()
        step = task["step"] if task else item["step"]
        total = task.get("total") or item["total"] if task else item["total"]
        self.metrics["step"].set(f"{step} / {total}")
        self.show_curve(item, task)
        self.metrics["lr"].set(scientific(item["config"]["train"]["learning_rate"]))
        self.train_progress.setRange(0, max(1, total))
        self.train_progress.setValue(step)
        self.refresh_controls()
        self.refresh_checkpoints(item)

    def show_curve(self, item, task):
        """Loss, elapsed and remaining time of the selected run, from the task logs."""
        run_root = Path(item["run_root"])
        series = self.host.tasks.run_series(run_root)
        self.loss_chart.set_points(series, item["total"])
        last = task["loss"] if task and "loss" in task else series[-1][1] if series else None
        self.metrics["loss"].set(f"{last:.4f}" if last is not None else "—")
        elapsed = self.host.tasks.run_elapsed(run_root)
        self.metrics["elapsed"].set(clock(elapsed) if elapsed else "—")
        eta = self.host.tasks.eta_seconds(task) if task else None
        self.metrics["eta"].set(clock(eta) if eta is not None else "—")

    def samples_by_checkpoint(self, run_root):
        """({checkpoint: sample audio}, {checkpoint: the sample's text})."""
        found, texts = {}, {}
        for path in (run_root / "samples").glob("*/generation.json"):
            record = read_json(path)
            request = record.get("request", {})
            adapter = request.get("lora_adapter")
            # Records may name their audio relative to their own folder.
            audio = path.parent / record["audio"] if record.get("audio") else None
            if adapter and audio is not None and audio.is_file():
                found[str(Path(adapter))] = audio
                texts[str(Path(adapter))] = str(request.get("text", ""))
        return found, texts

    def refresh_checkpoints(self, item):
        run_root = Path(item["run_root"])
        samples, texts = self.samples_by_checkpoint(run_root)
        entries = []
        for path in (run_root / "checkpoints").glob("*"):
            # Atomic-save backups remain available for recovery, but are not
            # additional checkpoints for the user to compare or register.
            if not path.name.startswith(".") and path.is_dir() and complete_checkpoint(path):
                step = read_json(path / "complete.json").get("step")
                entries.append((step if step is not None else -1, path))
        # Keep a requested sample playable even after training retention removes
        # its original checkpoint. Its pinned adapter can also be registered.
        seen = {str(path) for _step, path in entries}
        sample_sources = {}
        for record_path in (run_root / "samples").glob("*/generation.json"):
            record = read_json(record_path)
            source = record.get("request", {}).get("lora_adapter")
            snapshot = record.get("checkpoint_snapshot")
            if source in seen or not snapshot or source not in samples:
                continue
            path = record_path.parent / snapshot
            if complete_checkpoint(path):
                step = read_json(path / "complete.json").get("step")
                entries.append((step if step is not None else -1, path))
                samples[str(path)] = samples[source]
                texts[str(path)] = texts.get(source, "")
                sample_sources[str(path)] = source
                seen.add(source)
        entries.sort(key=lambda entry: (entry[0], entry[1].name), reverse=True)
        # A checkpoint without a recorded step has no place on the curve.
        entries = [entry for entry in self.one_per_step(entries, samples) if entry[0] >= 0]
        keys = [str(path) for _step, path in entries]
        chosen = self.loss_chart.selected if run_root == self.checkpoint_run else None
        # The newest checkpoint until one is picked (training adds new ones without moving
        # the pick away from what is being listened to).
        chosen = chosen if chosen in keys else keys[0] if keys else None
        self.checkpoint_run = run_root
        self.checkpoints_shown = {str(path): (step, path) for step, path in entries}
        self.checkpoint_samples, self.checkpoint_texts = samples, texts
        self.sample_sources = sample_sources
        self.loss_chart.set_checkpoints([(step, str(path)) for step, path in entries], chosen)
        signature = (
            run_root,
            chosen,
            tuple(entries),
            tuple(sorted(samples.items())),
            tuple(sorted(texts.items())),
            self.host.tasks.busy,
            self.host.tasks.can_preview_training(run_root),
        )
        if getattr(self, "checkpoint_signature", None) == signature:
            return
        self.checkpoint_signature = signature
        self.show_checkpoint(chosen)

    def chosen_checkpoint(self):
        key = self.loss_chart.selected
        if key not in self.checkpoints_shown:
            raise BusinessError(t("training_page.select_a_complete_checkpoint"))
        return self.checkpoints_shown[key][1]

    def show_checkpoint(self, key):
        """The picked checkpoint's line, sample text and player."""
        run_root = self.checkpoint_run
        busy = self.host.tasks.busy
        interlude = run_root is not None and self.host.tasks.can_preview_training(run_root)
        shown = key in self.checkpoints_shown
        self.sample_button.setVisible(shown)
        self.register_button.setVisible(shown)
        if not shown:
            set_text(self.checkpoint_title, t("training_page.no_checkpoints"))
            set_text(self.sample_text, "")
            set_text(self.sample_empty, "")
            self.sample_stack.setCurrentWidget(self.sample_empty)
            return
        step, path = self.checkpoints_shown[key]
        loss = self.loss_chart.loss_at(step)
        set_text(
            self.checkpoint_title,
            t(
                "training_page.checkpoint_line",
                step=step,
                loss=f"{loss:.3f}" if loss is not None else "—",
                date=self.host.date(self.checkpoint_time(path)),
            ),
        )
        sample = self.checkpoint_samples.get(key)
        set_text(
            self.sample_button,
            t("training_page.regenerate_sample") if sample else t("library_page.generate_preview"),
        )
        self.sample_button.setIcon(FluentIcon.SYNC if sample else FluentIcon.PLAY)
        self.sample_button.setEnabled(not busy or interlude)
        self.sample_button.setProperty("sample_checkpoint", self.sample_sources.get(key, key))
        bind(
            self.sample_button,
            t("tasks.preview_training_hint") if interlude else "",
            "setToolTip",
        )
        self.register_button.setEnabled(not busy)
        text = self.checkpoint_texts.get(key, "")
        set_text(self.sample_text, text if sample else "")
        if sample:
            self.sample_stack.setCurrentWidget(self.sample_player)
            self.sample_player.load(sample)
        else:
            set_text(self.sample_empty, t("training_page.no_sample_yet"))
            self.sample_stack.setCurrentWidget(self.sample_empty)

    def checkpoint_chosen(self, key):
        player = self.sample_player.player
        playing = player.isPlaying()
        at, previous = player.position(), self.sample_player.path
        self.checkpoint_signature = None
        self.show_checkpoint(key)
        sample = self.checkpoint_samples.get(key)
        if playing and sample:
            if previous != str(sample):
                self.pending_seek = at
            self.sample_player.start()
        elif playing:
            self.sample_player.stop()

    def seek_pending(self, status):
        ready = (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia)
        if self.pending_seek and status in ready:
            at, self.pending_seek = self.pending_seek, 0
            # Not from inside the media callback (see Player._queue_source).
            QTimer.singleShot(0, lambda: self.sample_player.player.setPosition(at))

    def make_sample(self):
        self.preview(self.chosen_checkpoint())

    @staticmethod
    def one_per_step(entries, samples):
        """One row per step. When the last step falls on a save point the trainer writes the
        same weights twice (checkpoint_0001000 and checkpoint_final); the row keeps the copy
        that has a sample, else the final one, which the run itself points at."""
        chosen = {}
        for step, path in entries:
            if step < 0:
                chosen[(step, path.name)] = (step, path)
                continue
            kept = chosen.get(step)
            better = str(path) in samples or (
                kept is not None and str(kept[1]) not in samples and path.name == "checkpoint_final"
            )
            if kept is None or better:
                chosen[step] = (step, path)
        return [entry for entry in entries if entry in chosen.values()]

    @staticmethod
    def checkpoint_time(path):
        from datetime import datetime

        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()

    def refresh_controls(self):
        active = self.host.tasks.busy
        item = self.runs.item()
        here = self.active_run() is not None
        can_resume = bool(item and self.can_resume(item))
        speech = model_files.speech_ready(self.host.resource_root)
        self.train_notice.setVisible(not speech)
        count = self.included_count()
        problems = len(self.problem_items()) if self.project else 0
        self.recommend_button.setEnabled(not active and bool(count) and not problems)
        self.update_recommendation()
        self.start_button.setEnabled(not active and speech and bool(count))
        set_text(
            self.start_button,
            t("training_page.start_training")
            if self.ready
            else t("training_page.preprocess_and_train"),
        )
        set_text(self.start_note, t("training_page.task_running_note") if active else "")
        note = (
            ""
            if self.ready or not self.project
            else t("training_page.problems_block", n=problems)
            if problems
            else t("training_page.preprocess_first", n=count)
            if count
            else t("training_page.nothing_included")
        )
        set_text(self.prepare_note, note)
        self.prepare_note.setVisible(bool(note))
        self.resume_button.setVisible(not here and can_resume)
        self.resume_button.setEnabled(not active and can_resume)
        can_extend = bool(item and not here and self.can_extend(item))
        self.extend_button.setVisible(can_extend)
        self.extend_button.setEnabled(not active)
        self.stop_button.setVisible(here)
        self.use_button.setVisible(bool(item) and self.usable(item))
        self.use_button.setEnabled(not here)
        bind(self.use_button, t("training_page.use_latest_checkpoint"), "setToolTip")
        reason = self.resume_blocker(item) if item and not here and not can_resume else ""
        set_text(self.resume_note, reason)
        self.resume_note.setVisible(bool(reason))
        bind(self.resume_note, reason, "setToolTip")
        interlude = self.host.tasks.training_preview
        same = bool(item and interlude and interlude.training["run"] == item["run_root"])
        if item:
            train = item["config"]["train"]
            kept = (
                t("training_page.checkpoints_all_kept", steps=train["save_every"])
                if not train.get("checkpoint_best_n")
                else t(
                    "training_page.checkpoints_latest_and_best_kept",
                    steps=train["save_every"],
                    best=train["checkpoint_best_n"],
                )
                if train.get("valid_every") and train.get("valid_ratio")
                else t(
                    "training_page.checkpoints_latest_kept",
                    steps=train["save_every"],
                    count=train["checkpoint_best_n"] + 1,
                )
            )
            set_text(
                self.checkpoint_note,
                self.host.tasks.preview_message()
                if same
                else t("training_page.pick_on_curve_note", kept=fmt(kept, "", -1)),
            )
        self.refresh_steps()

    def resume_blocker(self, item):
        """Why an unfinished run with a saved checkpoint cannot resume, or ""."""
        if item["status"] == "completed" or not item.get("checkpoint"):
            return ""
        init = (self.project or {}).get("initialization") or {}
        if item.get("resume_blocked") or item.get("initialization_id") != init.get("id"):
            return item.get("resume_blocked") or t(
                "worker.data_was_reinitialized_start_a_new_training"
            )
        if not self.ready:
            return t("training_page.resume_needs_current_recordings")
        return ""

    def run_context_menu(self, position):
        index = self.runs.indexAt(position)
        if not index.isValid():
            return
        self.runs.setCurrentIndex(self.runs.model().index(index.row(), 0))
        self.run_menu(at=self.runs.viewport().mapToGlobal(position))

    def run_menu(self, at=None):
        item = self.runs.item()
        if not item:
            return
        show_menu(
            self.more_button,
            [
                (t("app.force_stop"), self.host.force_stop, self.active_run() is not None),
                (
                    t("library_page.open_location"),
                    lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(item["run_root"])),
                    True,
                ),
                (
                    t("library_page.export_voice_package"),
                    lambda: self.host.perform(lambda: self.export_run(item), self.train_error),
                    bool(item.get("checkpoint")),
                ),
                (t("training_page.view_log"), lambda: self.view_log(item), True),
                (
                    t("training_page.delete_training_record"),
                    lambda: self.host.perform(lambda: self.delete_run(item), self.train_error),
                    not self.run_in_use(item),
                ),
            ],
            at=at,
        )

    def run_in_use(self, item):
        """Active, waiting or paused for a sample: the record must stay."""
        tasks = self.host.tasks
        interlude = tasks.training_preview
        current = tasks.current
        return bool(
            (interlude and interlude.training["run"] == item["run_root"])
            or (tasks.busy and current and current.get("run") == item["run_root"])
        )

    def delete_run(self, item):
        root = self.require()
        if self.run_in_use(item):
            raise BusinessError(t("training_page.wait_for_the_current_task_before_deleting"))
        run_root = Path(item["run_root"])
        self.host.run_io(
            lambda: tree_size(run_root),
            lambda size: self.confirm_delete_run(root, item, run_root, size),
            self.train_error,
        )

    def confirm_delete_run(self, root, item, run_root, size):
        if self.run_in_use(item):
            raise BusinessError(t("training_page.wait_for_the_current_task_before_deleting"))
        message = t(
            "training_page.delete_training_record_message",
            name=item["name"],
            size=fmt(size / 1024**2, ".1f", -1),
        )
        if item.get("registered") and item.get("checkpoint"):
            # The library lists a model straight from this record's checkpoint.
            message = t("training_page.delete_training_record_model", message=message)
        if not self.host.confirm(
            t("training_page.delete_training_record_title"),
            message,
            t("training_page.delete_training_record_action"),
        ):
            return
        self.sample_player.clear()
        clear_text(self.train_error)

        def deleted(_result=None):
            self.runs.clearSelection()
            self.host.refresh()

        # A resident inference process may still hold this run's adapter open, and the
        # player its sample; the delete waits a moment for files still being let go of.
        self.host.tasks.release(
            lambda _released: self.host.run_io(
                lambda: delete_run(root, run_root),
                deleted,
                self.train_error,
                busy=t("app.work_deleting"),
            )
        )

    def view_log(self, item):
        logs = []
        for task in self.host.tasks.tasks:
            if task.get("run") == item["run_root"]:
                path = self.host.tasks.task_dir(task) / "worker.log"
                if path.exists():
                    logs.append(path.read_text(encoding="utf-8", errors="replace")[-100000:])
        show_log(
            self,
            t("settings_page.task_diagnostics"),
            "\n".join(reversed(logs)) or t("settings_page.no_log_output_yet"),
        )

    def start_training(self):
        self.require()
        if self.host.tasks.busy:
            return
        if not self.ready:
            # Preprocessing comes first, then this run starts by itself.
            if not clean_field(self.voice_name.text()):
                raise BusinessError(t("training.enter_a_voice_name"))
            normalize_preset(self.values(), max(1, self.included_count()))
            self.initialize(
                then={
                    "name": self.voice_name.text(),
                    "values": self.values(),
                    "base": self.run_base.currentData(),
                }
            )
            self.tabs.setCurrentIndex(1)
            return
        self.create_and_train(self.voice_name.text(), self.values(), self.chosen_base().id)

    def continue_training(self, task):
        """Preprocessing that was started by Start training has finished: start the run."""
        request = read_json(self.host.tasks.task_dir(task) / "request.json", {}) or {}
        then = request.get("train")
        if not then or task["status"] != "completed" or self.host.tasks.busy:
            return
        root = Path(task["project"])
        if root != self.root:
            self.host.models.select_project(root)
        self.tabs.setCurrentIndex(2)
        self.host.perform(
            lambda: self.create_and_train(then["name"], then["values"], then.get("base")), None
        )

    def included_count(self):
        return sum(item["included"] for item in (self.project or {}).get("materials", []))

    def checking_training(self, operation, completed, error):
        """Run a check of the training files, which hashes them, off the GUI thread."""
        if self.starting:
            return
        self.starting = True

        def settled():
            self.starting = False

        self.host.run_io(
            operation,
            completed,
            error,
            busy=t("app.work_check_training"),
            settled=settled,
        )

    def start_train_task(self, payload):
        """A task may have been started during the check, which would leave this one out."""
        if self.host.tasks.busy:
            raise BusinessError(t("asr.wait_for_the_running_task"))
        self.host.tasks.start(payload)

    def refresh_bases(self):
        """The installed bases a new run can train on; the generation base comes first."""
        bases = self.host.bases()
        old = self.run_base.currentData() or self.host.store.settings.get("generation_base")
        self.run_base.blockSignals(True)
        clear_items(self.run_base)
        for base in bases:
            add_item(self.run_base, base.name, userData=base.id)
        self.run_base.setCurrentIndex(max(0, self.run_base.findData(old)))
        self.run_base.blockSignals(False)

    def chosen_base(self, base_id=None):
        """The installed base of base_id, or the one the new-run form chooses."""
        bases = self.host.bases()
        wanted = base_id or self.run_base.currentData()
        base = next((b for b in bases if b.id == wanted), None) or (bases[0] if bases else None)
        if base is None:
            raise BusinessError(t("resources.resources_are_not_ready_choose_prepare_resources"))
        return base

    def create_and_train(self, name, values, base_id=None):
        root = self.require()
        resources = self.host.resource_root
        base = self.chosen_base(base_id)

        def create():
            with project_lock(root):
                run = create_run(root, name, values, base, resources)
                # Run folders use human-readable names; selection uses the stable record ID.
                return run, read_json(run / "run.json")["id"]

        def created(result):
            run, run_id = result
            self.start_train_task({"kind": "train", "project": str(root), "run": str(run)})
            if self.root == root:
                self.set_project(root)
                self.select_run_id(run_id)

        self.checking_training(create, created, self.start_error)

    def select_run_id(self, run_id):
        for position, item in enumerate(self.runs.rows_model.rows):
            if item["id"] == run_id:
                self.runs.setCurrentIndex(self.runs.model().index(position, 0))

    def can_extend(self, item):
        init = (self.project or {}).get("initialization")
        return bool(
            item["status"] == "completed"
            and item.get("checkpoint")
            and self.ready
            and init
            and item.get("initialization_id") == init["id"]
            and complete_checkpoint(self.root / item["checkpoint"], require_trainer=True)
        )

    def train_more(self):
        self.require()
        item = self.runs.item()
        if self.host.tasks.busy or not item:
            return
        train = item["config"]["train"]
        dialog = MoreSteps(self.window(), item["step"], train["max_steps"], train["save_every"])
        if not dialog.exec():
            return
        root, run_root = self.root, Path(item["run_root"])
        total, save_every = dialog.total.value(), dialog.save_every.value()
        bases = self.host.bases()

        def extend():
            with project_lock(root):
                extend_run(root, run_root, total, save_every, bases)

        def extended(_done):
            if self.root == root:
                self.set_project(root)
                self.select_run_id(item["id"])
                self.resume_training()

        self.checking_training(extend, extended, self.train_error)

    def resume_training(self):
        self.require()
        if self.host.tasks.busy:
            return
        item = self.runs.item()
        if not item:
            raise BusinessError(t("training_page.select_a_training_run_to_resume"))
        root = self.root
        bases = self.host.bases()
        clear_text(self.train_error)
        self.checking_training(
            lambda: resume_eligibility(root, item, bases),
            lambda _checked: self.start_train_task(
                {"kind": "train", "project": str(root), "run": item["run_root"], "resume": True}
            ),
            self.train_error,
        )

    def preview(self, checkpoint):
        item = self.runs.item()
        if not item:
            raise BusinessError(t("training_page.select_a_complete_checkpoint"))
        self.host.preview(self.root, Path(item["run_root"]), checkpoint=Path(checkpoint))

    def usable(self, item):
        """Whether the run has a complete checkpoint and an installed base to speak with."""
        return bool(
            self.root
            and item.get("checkpoint")
            and base_models.match_record(item.get("base"), self.host.bases()).base is not None
            and complete_checkpoint(self.root / item["checkpoint"])
        )

    def use_run(self):
        item = self.runs.item()
        if not item or not self.usable(item):
            raise BusinessError(t("training_page.select_a_complete_checkpoint"))
        self.host.use_model(item["id"])

    def export_run(self, item):
        path = save_file(
            self,
            self.host.store.settings,
            t("library_page.export_voice_package"),
            item["name"] + ".irovoice",
            t("library_page.voice_packages_irovoice"),
        )
        if path:
            root = self.require()
            run = read_json(Path(item["run_root"]) / "run.json")
            self.host.run_io(
                lambda: export_voice(root, run, Path(path)),
                lambda _result: self.host.inform(t("library_page.voice_package_exported")),
                self.train_error,
            )

    def project_menu(self):
        if not self.root:
            return
        show_menu(
            self.project_more,
            [
                (
                    t("library_page.export_project_package"),
                    lambda: self.host.perform(self.export_project),
                    True,
                ),
                (
                    t("models_page.forget_project"),
                    lambda: self.host.perform(self.forget_project),
                    not self.host.project_busy(self.root),
                ),
                (
                    t("models_page.delete_project"),
                    lambda: self.host.perform(self.delete_project),
                    not self.host.project_busy(self.root),
                ),
            ],
        )

    def export_project(self):
        root = self.require()
        if self.host.project_busy(root):
            raise BusinessError(t("library_page.a_project_task_is_running_wait_before"))
        path = save_file(
            self,
            self.host.store.settings,
            t("library_page.export_project_package"),
            self.project["name"] + ".iroproject",
            t("app.project_packages_iroproject"),
        )
        if path:
            self.host.run_io(
                lambda: export_project(root, Path(path)),
                lambda _result: self.host.inform(t("library_page.project_package_exported")),
                None,
                busy=t("app.work_export_project"),
            )

    def forget_project(self):
        """Take the project off the list; its folder and files stay where they are."""
        root = self.require()
        if self.host.project_busy(root):
            raise BusinessError(t("library_page.a_project_task_is_running_wait_before"))
        if not self.host.confirm(
            t("models_page.forget_project_title"),
            t("models_page.forget_project_message", name=self.project["name"], path=str(root)),
            t("models_page.forget_project"),
        ):
            return
        self.material_player.clear()
        self.sample_player.clear()
        self.host.store.forget(root)
        self.root = self.project = None
        self.host.refresh()

    def delete_project(self):
        """Delete the project's folder: recordings, preprocessing, every training record and the
        voices listed from them. Voices kept apart in the data folder stay."""
        root = self.require()
        if self.host.project_busy(root):
            raise BusinessError(t("library_page.a_project_task_is_running_wait_before"))
        self.host.run_io(
            lambda: tree_size(root),
            lambda size: self.confirm_delete_project(root, size),
            None,
        )

    def confirm_delete_project(self, root, size):
        if self.host.project_busy(root):
            raise BusinessError(t("library_page.a_project_task_is_running_wait_before"))
        if not self.host.confirm(
            t("models_page.delete_project_title"),
            t(
                "models_page.delete_project_message",
                name=self.project["name"],
                size=fmt(size / 1024**2, ".1f", -1),
                path=str(root),
            ),
            t("models_page.delete_project_action"),
        ):
            return
        self.material_player.clear()
        self.sample_player.clear()

        def released(_released):
            self.host.store.forget(root)
            if self.root == root:
                self.root = self.project = None
            self.host.refresh()
            self.host.run_io(
                lambda: remove_tree(root.parent, root),
                lambda _done: None,
                None,
                busy=t("app.work_deleting"),
            )

        # A resident inference process may still hold one of its adapters open.
        self.host.tasks.release(lambda result: self.host.perform(lambda: released(result)))

    def register_checkpoint(self, checkpoint):
        root = self.require()
        item = self.runs.item()
        if self.host.tasks.busy:
            raise BusinessError(t("training_page.wait_for_the_current_task_before_registering"))
        if not item:
            raise BusinessError(t("training_page.select_a_complete_checkpoint_305"))
        name, accepted = prompt_text(
            self,
            t("training_page.register_checkpoint_voice"),
            t("library_page.voice_name"),
            item["name"],
        )
        if accepted:
            self.host.run_io(
                lambda: register_checkpoint_voice(
                    root, item, Path(checkpoint), self.host.store.root / "voices", name
                ),
                lambda result: self.host.refresh(),
                self.train_error,
                busy=t("app.work_save_model"),
            )

    def task_changed(self, task):
        active = self.host.tasks.busy
        same = self.root and task.get("project") == str(self.root)
        state = (task["id"], task["status"])
        step, self.task_state = state == self.task_state, state
        # A step of another project's task or a download: nothing here reads its progress.
        if step and not same:
            return
        # These read checkpoints and samples from disk: a task's steps refresh them once a
        # second, and any change of state at once.
        listed = not step or time.monotonic() - self.task_listed >= 1.0
        if listed:
            self.task_listed = time.monotonic()
            self.refresh_controls()
            if item := self.runs.item():
                self.refresh_checkpoints(item)
        if not same:
            self.refresh_initialization()
            return
        if task["kind"] == "initialize":
            if listed:
                self.refresh_initialization()
        elif task["kind"] == "train":
            for item in self.run_cache:
                if item["run_root"] == task.get("run"):
                    item.update(
                        status=task["status"],
                        step=task["step"],
                        total=task["total"] or item["total"],
                    )
                    item["steps"] = f"{item['step']} / {item['total']}"
                    self.describe_run(item)
            self.runs.update_rows(self.run_cache)
            self.stop_button.setEnabled(
                active and task["status"] not in {"saving", "stop_requested"}
            )
            self.show_run_progress(task)
            if task["status"] in {"failed", "interrupted"}:
                set_text(self.train_error, task["message"])
            elif task.get("preview_error"):
                set_text(self.train_error, task["preview_error"])

    def show_run_progress(self, task):
        if self.active_run() is None:
            return
        total = task.get("total") or 1
        self.metrics["step"].set(f"{task['step']} / {total}")
        self.train_progress.setRange(0, max(1, total))
        self.train_progress.setValue(task["step"])
        self.show_curve(self.runs.item(), task)

    # ----- saved state -----

    def snapshot(self):
        return {
            "tab": self.tabs.currentIndex(),
            "name": self.voice_name.text(),
            "settings": self.values(),
            "advanced": self.advanced.toggle.isChecked(),
            "filter": self.filter.text(),
            "only_problems": self.only_problems.isChecked(),
            "material_id": self.material_id,
            "material_text": self.material_text.toPlainText(),
            "material_cursor": self.material_text.textCursor().position(),
            "material_caption": self.material_caption.text(),
            "included": self.included.isChecked(),
            "run_id": self.runs.item()["id"] if self.runs.item() else None,
            "scrolls": [
                page.verticalScrollBar().value()
                for page in (self.materials_page, self.init_page, self.train_page)
            ],
        }

    def restore(self, state):
        self.tabs.setCurrentIndex(state.get("tab", 0))
        set_text(self.voice_name, state.get("name", self.project["name"]))
        values = {**self.defaults(), **state.get("settings", {})}
        for key, widget in self.parameters.items():
            widget.setValue(values[key])
        self.advanced.set_open(state.get("advanced", False))
        self.only_problems.setChecked(state.get("only_problems", False))
        set_text(self.filter, state.get("filter", ""))
        self.material_id = state.get("material_id")
        self.material_text.setPlainText(state.get("material_text", ""))
        set_text(self.material_caption, state.get("material_caption", ""))
        self.included.setChecked(state.get("included", True))
        # A restored draft counts as edited against what is saved.
        saved = self.selected_material()
        self.loaded = self.fields(saved) if saved else None
        cursor = self.material_text.textCursor()
        cursor.setPosition(
            min(state.get("material_cursor", 0), len(self.material_text.toPlainText()))
        )
        self.material_text.setTextCursor(cursor)
        expected_project = self.project["id"]

        def restore_details():
            if not self.project or self.project["id"] != expected_project:
                return
            for table, key in ((self.materials, state.get("material_id")),):
                for index, candidate in enumerate(table.rows_model.rows):
                    if candidate["id"] == key:
                        table.setCurrentIndex(table.model().index(index, 0))
            for page, value in zip(
                (self.materials_page, self.init_page, self.train_page),
                state.get("scrolls", []),
                strict=False,
            ):
                page.verticalScrollBar().setValue(value)

        QTimer.singleShot(0, restore_details)
        if state.get("advanced", False) and not self.host.reduced_motion():
            self.advanced.animation.finished.connect(
                restore_details, Qt.ConnectionType.SingleShotConnection
            )
