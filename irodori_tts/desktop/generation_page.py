from __future__ import annotations

import shutil
from pathlib import Path

import soundfile as sf
from PySide6.QtCore import QMimeData, Qt, QTimer, QUrl, QVariantAnimation
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    DoubleSpinBox,
    FluentIcon,
    Flyout,
    FlyoutAnimationType,
    FlyoutViewBase,
    IndeterminateProgressBar,
    LineEdit,
    ProgressBar,
    PushButton,
    SpinBox,
    TextEdit,
    TogglePushButton,
)

from irodori_tts.desktop.messages import BusinessError, fmt, t

from . import base_models, model_files
from .asr_dialog import human_size
from .emoji_groups import emoji_name, grouped_emoji
from .exports import adapter_architecture, export_audio_file
from .i18n import add_item, bind, clear_items, clear_text, set_text
from .long_text import split_text
from .storage import digest, identity, read_json, stamp, write_json
from .widgets import (
    DETAIL_MINIMUM,
    AppIcon,
    Expand,
    ModelNotice,
    Page,
    Player,
    RowPlayback,
    Table,
    button,
    card,
    card_header,
    combo_box,
    elided_label,
    error_label,
    field,
    grid,
    icon_button,
    label,
    open_file,
    row,
    save_file,
    search_box,
    smooth,
    switch,
    toast,
)

REFERENCE_MIME = "application/x-irodori-reference"
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}
TEXT_LIMIT = 5000


def dropped_audio(mime):
    paths = [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]
    return [path for path in paths if Path(path).suffix.lower() in AUDIO_SUFFIXES]


def inspect_audio(path: Path):
    """(seconds, error) of a reference clip from its header only, so a long file is never
    decoded on the GUI thread (only its first 120 seconds are used anyway)."""
    try:
        info = sf.info(str(path))
        duration = info.frames / info.samplerate if info.samplerate else 0
    except Exception as exc:
        return 0, t("materials.cannot_read_the_recording_replace_it", p1=fmt(exc, "", -1))
    if duration < 0.3:
        return duration, t("materials.recording_is_shorter_than_0_3_seconds")
    return duration, ""


def reference_duration(path: Path) -> float:
    duration, error = inspect_audio(path)
    # A long clip is fine as a reference: only its first 120 seconds are used.
    if error and getattr(error, "key", None) != (
        "materials.recording_exceeds_12_seconds_split_the_audio"
    ):
        raise BusinessError(error)
    return duration


def reference_item(path: Path, name=None, duration=None) -> dict:
    return {
        "id": identity(),
        "path": str(path),
        "name": name or path.name,
        "duration": reference_duration(path) if duration is None else duration,
        "seconds": None,
    }


class References(Table):
    def __init__(self, page):
        super().__init__(
            [
                ("name", t("generation_page.reference_recording"), 0),
                ("seconds", t("generation_page.duration_115"), 86),
                ("actions", "", 64),
            ],
            height=90,
            header=False,
            stretch=0,
            # Clip names are short; the panel beside the history is narrow.
            main_minimum=100,
        )
        self.page = page
        self.manage_empty_height = False
        set_text(self.empty_label, t("generation_page.add_multiple_reference_recordings"))
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.animation = QVariantAnimation(self)
        self.animation.valueChanged.connect(lambda value: self.setFixedHeight(int(value)))
        self.set_row_actions(self.page.reference_actions, for_rows="all")

    def startDrag(self, supported):
        from PySide6.QtGui import QDrag

        if self.item():
            drag = QDrag(self)
            mime = QMimeData()
            mime.setData(REFERENCE_MIME, self.item()["id"].encode())
            drag.setMimeData(mime)
            drag.exec(Qt.DropAction.MoveAction)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(REFERENCE_MIME) or dropped_audio(event.mimeData()):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        if not event.mimeData().hasFormat(REFERENCE_MIME):
            self.page.drop_audio(event)
            return
        key = bytes(event.mimeData().data(REFERENCE_MIME)).decode()
        source = next(i for i, item in enumerate(self.page.references) if item["id"] == key)
        target = self.indexAt(event.position().toPoint()).row()
        self.page.references.insert(
            target if target >= 0 else len(self.page.references), self.page.references.pop(source)
        )
        self.page.refresh_references()
        event.acceptProposedAction()

    def resize_rows(self, rows):
        self.update_rows(rows)
        self.animation.stop()
        target = 4 + max(1, min(len(rows), 4)) * 40
        if self.page.host.reduced_motion():
            self.setFixedHeight(target)
        else:
            self.animation.setStartValue(self.height())
            self.animation.setEndValue(target)
            self.animation.setDuration(220)
            self.animation.start()


class EmojiPalette(FlyoutViewBase):
    """The upstream emoji in titled groups, each with its name; a click inserts and the palette
    stays open."""

    COLUMNS = 6

    def __init__(self, insert, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(4)
        self.buttons = []
        for title, emoji_hints in grouped_emoji():
            layout.addWidget(label(title, "muted", wrap=False))
            cells = QGridLayout()
            cells.setSpacing(2)
            cells.setColumnStretch(self.COLUMNS, 1)
            for position, (emoji, hint) in enumerate(emoji_hints):
                choice = PushButton(f"{emoji} {emoji_name(hint)}")
                choice.emoji = emoji
                choice.setFixedHeight(32)
                choice.setToolTip(str(hint))
                choice.setAccessibleName(str(hint))
                choice.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                choice.clicked.connect(lambda _checked=False, emoji=emoji: insert(emoji))
                cells.addWidget(choice, position // self.COLUMNS, position % self.COLUMNS)
                self.buttons.append(choice)
            layout.addLayout(cells)
        # One width, so the columns line up across the groups.
        width = max(choice.sizeHint().width() for choice in self.buttons)
        for choice in self.buttons:
            choice.setFixedWidth(width)


class Parameters(QWidget):
    """The right-hand column. It scrolls on its own when a short window cannot fit it."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if dropped_audio(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.page.drop_audio(event)


# Each model keeps the inference settings last used with it in its own folder (the base
# model in the data folder), so choosing it again brings them back.
SETTINGS_FILE = "inference_settings.json"
NUMBER_DEFAULTS = {
    "steps": 40,
    "seed": -1,
    "num_candidates": 1,
    "seconds": 0.0,
    "duration_scale": 1.0,
    "cfg_text": 3.0,
    "cfg_caption": 3.0,
    "cfg_speaker": 5.0,
    # 1 leaves the reference voice as the model hears it; above 1 pulls the speech toward it.
    "speaker_strength": 1.0,
}


# Painted on each history row: copy the text, generate again, fill the inputs back, delete.
HISTORY_ACTIONS = (
    ("copy", FluentIcon.COPY, t("generation_page.copy_text")),
    ("again", FluentIcon.SYNC, t("generation_page.generate_again")),
    ("fill", FluentIcon.RETURN, t("generation_page.fill_back")),
    ("delete", FluentIcon.DELETE, t("generation_page.delete_record")),
)


class GenerationPage(Page):
    def __init__(self, host):
        super().__init__(t("app.generate_speech"))
        self.host = host
        self.references = []
        self.voices = {}
        self.output_audio = None
        self.output_record = None
        self.output_index = 0
        # The model whose settings the page shows; None is the base model.
        self.voice_id = ...
        self.loading = False
        self.parameters_changed = False
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(400)
        self.save_timer.timeout.connect(self.save_parameters)
        self.voice = combo_box()
        # The base model the default voice speaks with; a trained voice uses its own base.
        self.base = combo_box()
        self.bases = []
        self.text = smooth(TextEdit())
        bind(self.text, t("generation_page.enter_japanese_text_here"), "setPlaceholderText")
        bind(self.text, t("generation_page.spoken_text_japanese"), "setAccessibleName")
        # Short windows give the editor less height before the history loses its rows.
        self.text.setMinimumHeight(60)
        self.text.setMaximumHeight(132)
        self.caption = smooth(LineEdit())
        bind(self.caption, t("generation_page.optional_for_example"), "setPlaceholderText")
        self.error = error_label()

        columns = QHBoxLayout()
        columns.setSpacing(14)
        self.body.addLayout(columns, 1)
        left = QVBoxLayout()
        left.setSpacing(14)
        columns.addLayout(left, 2)
        self.build_text(left)
        self.build_output(left)
        self.build_history(left)
        columns.addWidget(self.build_parameters(), 1)

        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.shortcut_generate).setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        QShortcut(QKeySequence("Ctrl+Enter"), self, activated=self.shortcut_generate).setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.text.textChanged.connect(self.update_count)
        # An error about the request is out of date once the request changes.
        self.text.textChanged.connect(self.input_changed)
        self.caption.textChanged.connect(self.input_changed)
        self.voice.currentIndexChanged.connect(self.input_changed)
        self.voice.currentIndexChanged.connect(lambda _index: self.voice_changed())
        self.base.currentIndexChanged.connect(lambda _index: self.base_changed())
        self.caption.textChanged.connect(self.parameter_changed)
        self.trim.checkedChanged.connect(self.parameter_changed)
        for name in NUMBER_DEFAULTS:
            getattr(self, name).valueChanged.connect(self.parameter_changed)
        self.update_count()
        self.refresh()

    def build_text(self, parent):
        _, body = card(parent)
        self.example_button = button(t("generation_page.fill_example"), self.fill_example)
        self.emoji_button = icon_button(
            AppIcon.EMOJI, t("generation_page.insert_emoji"), self.show_emoji
        )
        body.addWidget(
            card_header(
                t("generation_page.spoken_text_japanese"), self.emoji_button, self.example_button
            )
        )
        body.addWidget(self.text)
        self.count = label("", "muted", wrap=False)
        self.generate_button = button(
            t("app.generate"),
            lambda: self.host.perform(self.generate, self.error),
            primary=True,
            icon=FluentIcon.PLAY_SOLID,
        )
        self.cancel_button = button(t("generation_page.cancel_generation"), self.host.tasks.stop)
        self.cancel_button.setEnabled(False)
        footer = row(
            self.count,
            label(t("generation_page.shortcut_hint"), "muted", wrap=False),
            self.error,
            self.cancel_button,
            self.generate_button,
        )
        footer.layout().setStretchFactor(self.error, 1)
        self.model_notice = ModelNotice(lambda: self.host.show_guide())
        self.speech_ready = True
        body.addWidget(self.model_notice)
        body.addWidget(footer)

    def build_output(self, parent):
        _, body = card(parent)
        header = card_header(
            t("generation_page.output"),
            icon_button(FluentIcon.FOLDER, t("library_page.open_location"), self.open_output),
            button(
                t("generation_page.export_audio"),
                lambda: self.host.perform(self.export_audio, self.error),
                icon=FluentIcon.DOWNLOAD,
            ),
            meta=t("generation_page.generated_audio_will_appear_here"),
        )
        self.result_label = header.meta
        body.addWidget(header)
        self.player = Player()
        # While a generation runs, its progress takes the player's place: loading the model
        # on the first generation takes several seconds and must not look like nothing.
        progress = QWidget()
        lines = QVBoxLayout(progress)
        lines.setContentsMargins(10, 6, 10, 6)
        lines.setSpacing(8)
        self.progress_text = elided_label()
        self.progress_bar = ProgressBar()
        self.waiting_bar = IndeterminateProgressBar(start=False)
        lines.addWidget(self.progress_text)
        lines.addWidget(self.progress_bar)
        lines.addWidget(self.waiting_bar)
        self.output_stack = QStackedWidget()
        self.output_stack.addWidget(self.player)
        self.output_stack.addWidget(progress)
        self.output_stack.setFixedHeight(self.player.height())
        body.addWidget(self.output_stack)
        # One numbered choice per candidate of the shown generation; only there when it has
        # more than one.
        self.candidate_row = QWidget()
        self.candidate_layout = QHBoxLayout(self.candidate_row)
        self.candidate_layout.setContentsMargins(10, 0, 10, 0)
        self.candidate_layout.setSpacing(6)
        self.candidate_buttons = []
        self.candidate_row.hide()
        body.addWidget(self.candidate_row)

    def build_history(self, parent):
        _, body = card(parent, stretch=1)
        self.history_filter = search_box(t("generation_page.filter_history"))
        body.addWidget(card_header(t("generation_page.generation_history"), self.history_filter))
        self.history = Table(
            [
                ("play", "", 44),
                ("voice", t("app.voice"), 120),
                ("text", t("app.spoken_text"), 0),
                ("seconds", t("generation_page.duration_115"), 90),
                ("date", t("generation_page.generated_at"), 172),
                ("copy", "", 40),
                ("again", "", 40),
                ("fill", "", 40),
                ("delete", "", 44),
            ],
            fill=True,
            stretch=2,
            optional=("seconds", "voice"),
        )
        self.history_playback = RowPlayback(self.history, self.player, lambda row: row["audio"])
        self.history.selected.connect(self.select_history)
        # The row's actions are painted on every row, like its play icon, not only on the
        # selected one.
        self.history.icon_clicked.connect(self.history_icon_clicked)
        self.history_filter.textChanged.connect(self.refresh_history)
        body.addWidget(self.history, 1)

    def build_parameters(self):
        frame, outer = card()
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        panel = Parameters(self)
        body = QVBoxLayout(panel)
        body.setContentsMargins(16, 14, 16, 14)
        body.setSpacing(14)
        scroll.setWidget(panel)
        outer.addWidget(scroll)
        frame.setMinimumWidth(DETAIL_MINIMUM)
        frame.setMaximumWidth(360)

        body.addWidget(field(t("app.voice"), self.voice))
        self.base_field = field(t("generation_page.base_model"), self.base)
        body.addWidget(self.base_field)
        # The voice's base, a note on it, or why the chosen generation model is not used.
        self.base_note = label("", "muted")
        self.base_note.hide()
        body.addWidget(self.base_note)
        body.addWidget(field(t("generation_page.voice_description"), self.caption))

        self.add_button = button(
            t("generation_page.add_reference"),
            lambda: self.host.perform(self.add_references, self.error),
            icon=FluentIcon.ADD,
        )
        body.addWidget(card_header(t("generation_page.reference_audio"), self.add_button))
        self.reference_table = References(self)
        body.addWidget(self.reference_table)
        self.reference_summary = label(t("generation_page.drop_audio_hint"), "muted")
        body.addWidget(self.reference_summary)

        self.steps = smooth(SpinBox())
        self.steps.setRange(1, 200)
        self.steps.setValue(40)
        self.seed = smooth(SpinBox())
        self.seed.setRange(-1, 2147483647)
        # Qt sizes a spin box for its largest value (2,147,483,647), which pushed the narrow
        # parameter column wider than the window allows; a seed is rarely that long.
        self.seed.setMinimumWidth(120)
        self.steps.setMinimumWidth(80)
        self.seed.setValue(-1)
        self.num_candidates = smooth(SpinBox())
        self.num_candidates.setRange(1, 4)
        self.num_candidates.setMinimumWidth(80)
        self.num_candidates.setSymbolVisible(False)
        bind(self.seed, t("generation_page.random_seed"), "setToolTip")
        random_seed = icon_button(
            FluentIcon.SYNC,
            t("generation_page.random_seed"),
            lambda: self.seed.setValue(-1),
            size=32,
        )
        body.addWidget(
            grid(
                field(
                    t("generation_page.seed"),
                    row(self.seed, random_seed, spacing=4),
                    buddy=self.seed,
                ),
                field(t("generation_page.candidates"), self.num_candidates),
                field(t("generation_page.sampling_steps"), self.steps),
            )
        )

        self.seconds = smooth(DoubleSpinBox())
        self.seconds.setRange(0, 30)
        bind(self.seconds, t("generation_page.automatic_duration"), "setSpecialValueText")
        bind(self.seconds, t("generation_page.s"), "setSuffix")
        self.duration_scale = smooth(DoubleSpinBox())
        self.duration_scale.setRange(0.3, 3)
        self.duration_scale.setValue(1)
        self.cfg_text = smooth(DoubleSpinBox())
        self.cfg_caption = smooth(DoubleSpinBox())
        self.cfg_speaker = smooth(DoubleSpinBox())
        for widget, value in ((self.cfg_text, 3), (self.cfg_caption, 3), (self.cfg_speaker, 5)):
            widget.setRange(0, 15)
            widget.setValue(value)
        self.speaker_strength = smooth(DoubleSpinBox())
        # Measured: 1.5 still read the text right; 2 lost most of it.
        self.speaker_strength.setRange(1, 1.5)
        self.speaker_strength.setSingleStep(0.05)
        self.speaker_strength.setValue(1)
        bind(self.speaker_strength, t("generation_page.speaker_strength_hint"), "setToolTip")
        for widget in (
            self.seconds,
            self.duration_scale,
            self.cfg_text,
            self.cfg_caption,
            self.cfg_speaker,
            self.speaker_strength,
        ):
            widget.setSymbolVisible(False)
        self.steps.setSymbolVisible(False)
        self.seed.setSymbolVisible(False)
        self.trim = switch()
        self.trim.setChecked(True)
        advanced = QWidget()
        advanced_layout = QVBoxLayout(advanced)
        advanced_layout.setContentsMargins(0, 6, 0, 0)
        advanced_layout.setSpacing(10)
        advanced_layout.addWidget(
            grid(
                field(t("generation_page.duration"), self.seconds),
                field(t("generation_page.automatic_duration_scale"), self.duration_scale),
            )
        )
        advanced_layout.addWidget(
            grid(
                field(t("generation_page.text_guidance"), self.cfg_text),
                field(t("generation_page.description_guidance"), self.cfg_caption),
                field(t("generation_page.voice_guidance"), self.cfg_speaker),
                columns=3,
            )
        )
        advanced_layout.addWidget(
            grid(field(t("generation_page.speaker_strength"), self.speaker_strength))
        )
        # Shown while the fast model generates: it has its own steps and guidance.
        self.fast_note = label(t("generation_page.fast_model_note"), "muted")
        self.fast_note.setVisible(False)
        advanced_layout.addWidget(self.fast_note)
        advanced_layout.addWidget(
            row(label(t("generation_page.trim_trailing_silence"), wrap=False), None, self.trim)
        )
        self.advanced = Expand(
            t("generation_page.advanced_settings"), advanced, self.host.reduced_motion
        )
        body.addWidget(self.advanced)
        body.addStretch(1)
        return frame

    def update_count(self):
        text = self.text.toPlainText()
        segments = len(split_text(text))
        if segments > 1:
            message = t(
                "generation_page.character_count_segments",
                count=len(text),
                limit=TEXT_LIMIT,
                segments=segments,
            )
        else:
            message = t("generation_page.character_count", count=len(text), limit=TEXT_LIMIT)
        set_text(self.count, message)
        # Every segment is a single pass: candidates only exist for a text read in one.
        self.num_candidates.setEnabled(segments <= 1)
        bind(
            self.num_candidates,
            t("generation_page.candidates_single_pass") if segments > 1 else "",
            "setToolTip",
        )

    def input_changed(self, *_args):
        clear_text(self.error)

    def shortcut_generate(self):
        if self.generate_button.isEnabled():
            self.generate_button.click()

    def show_emoji(self):
        # Below the text box, so the text and its cursor stay in view while inserting.
        self.emoji_flyout = Flyout.make(
            EmojiPalette(self.insert_emoji),
            self.text,
            self.window(),
            FlyoutAnimationType.DROP_DOWN,
        )

    def insert_emoji(self, emoji):
        """Put an emoji at the text cursor and hand the focus back to the text box."""
        cursor = self.text.textCursor()
        cursor.insertText(emoji)
        self.text.setTextCursor(cursor)
        self.text.setFocus()

    def fill_example(self):
        self.text.setPlainText("おはようございます。今日はとてもいい天気ですね。")

    def add_references(self):
        paths = open_file(
            self,
            self.host.store.settings,
            t("generation_page.add_reference_audio"),
            t("generation_page.audio_wav_flac_mp3_ogg_aiff"),
            many=True,
        )
        self.add_reference_files(paths)

    def drop_audio(self, event):
        paths = dropped_audio(event.mimeData())
        if paths:
            event.acceptProposedAction()
            self.host.perform(lambda: self.add_reference_files(paths), self.error)

    def add_reference_files(self, paths):
        if not paths:
            return
        clear_text(self.error)
        directory = self.host.store.root / "references"

        def read():
            """What was added, and what stopped it: files before a bad one still count."""
            items = []
            try:
                for path in paths:
                    file = Path(path)
                    duration = reference_duration(file)
                    directory.mkdir(exist_ok=True)
                    destination = directory / (digest(file) + file.suffix.lower())
                    if not destination.exists():
                        shutil.copyfile(file, destination)
                    items.append(reference_item(destination, file.name, duration))
            except Exception as exc:
                return items, exc
            return items, None

        def added(result):
            items, problem = result
            self.references.extend(items)
            self.refresh_references()
            if problem:
                raise problem

        self.host.run_io(read, added, self.error, busy=t("app.work_add_references"))

    def reference_rows(self):
        rows = []
        for item in self.references:
            present = Path(item["path"]).is_file()
            rows.append(
                {
                    **item,
                    "icon": FluentIcon.MENU,
                    "seconds": t("app.seconds", p0=fmt(item.get("duration") or 0, ".2f", -1))
                    if present
                    else t("generation_page.reference_missing"),
                    "tooltip": item["path"],
                    "tones": {} if present else {"seconds": "bad", "name": "bad"},
                }
            )
        return rows

    def refresh_references(self):
        self.parameter_changed()
        self.reference_table.resize_rows(self.reference_rows())
        duration = sum(item.get("duration") or 0 for item in self.references)
        set_text(
            self.reference_summary,
            t(
                "generation_page.clips_s_first_120_seconds_used_in",
                n=len(self.references),
                p3=fmt(duration, ".2f", -1),
            )
            if self.references
            else t("generation_page.drop_audio_hint"),
        )

    def reference_actions(self, item):
        key = item["id"]
        box = QWidget()
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(
            icon_button(
                FluentIcon.PLAY,
                t("generation_page.preview_reference"),
                lambda: self.play_reference(key),
                size=28,
            )
        )
        layout.addWidget(
            icon_button(
                FluentIcon.CLOSE,
                t("generation_page.remove"),
                lambda: self.remove_reference(key),
                size=28,
            )
        )
        return box

    def remove_reference(self, key):
        self.references = [item for item in self.references if item["id"] != key]
        clear_text(self.error)
        self.refresh_references()

    def play_reference(self, key):
        for item in self.references:
            if item["id"] == key:
                self.player.load(item["path"], autoplay=True)

    def request(self):
        text = self.text.toPlainText().strip()
        if not text:
            self.text.setFocus()
            raise BusinessError(t("generation_page.enter_japanese_text_to_read"))
        if len(text) > TEXT_LIMIT:
            raise BusinessError(t("generation_page.maximum_5000_characters_per_generation"))
        if any(not Path(item["path"]).is_file() for item in self.references):
            raise BusinessError(t("generation_page.remove_missing_references"))
        voice_id = self.voice.currentData()
        voice = self.voices.get(voice_id)
        fast = self.generation_model() == model_files.FAST
        steps = self.fast_steps() if fast else self.steps.value()
        # 1 means off; it is then left out so the runtime skips the scaling entirely.
        strength = self.speaker_strength.value()
        return {
            "text": text,
            "caption": self.caption.text().strip() or None,
            "ref_wavs": [item["path"] for item in self.references] or None,
            "no_ref": not self.references,
            "num_steps": steps,
            "num_candidates": self.num_candidates.value(),
            "seed": self.seed.value() if self.seed.value() >= 0 else None,
            "seconds": self.seconds.value() or None,
            "duration_scale": self.duration_scale.value(),
            "cfg_scale_text": self.cfg_text.value(),
            "cfg_scale_caption": self.cfg_caption.value(),
            "cfg_scale_speaker": self.cfg_speaker.value(),
            "speaker_kv_scale": strength if strength > 1 and not fast else None,
            "trim_tail": self.trim.isChecked(),
            "lora_adapter": voice["adapter"] if voice else None,
        }

    def generate(self):
        if self.host.tasks.busy:
            return
        if not self.speech_ready:
            raise BusinessError(
                t(
                    "models.speech_needed",
                    size=human_size(model_files.download_for(model_files.SPEECH).size),
                )
            )
        voice = self.voices.get(self.voice.currentData())
        if voice and voice.get("base_id") is None:
            raise BusinessError(voice["base_problem"])
        clear_text(self.error)
        request = self.request()
        self.host.tasks.start(
            {
                "kind": "generate",
                "request": request,
                "voice": self.voice.currentText(),
                "voice_id": self.voice.currentData(),
                "base": self.generation_base(),
                "model": self.generation_model(),
                "output": str(self.host.store.root / "generations" / stamp()),
            }
        )

    def refresh(self):
        self.speech_ready = model_files.speech_ready(self.host.resource_root)
        self.model_notice.setVisible(not self.speech_ready)
        self.refresh_bases()
        self.generate_button.setEnabled(self.speech_ready and not self.host.tasks.busy)
        old = self.voice.currentData()
        self.voices = self.host.voice_entries()
        self.voice.blockSignals(True)
        clear_items(self.voice)
        add_item(self.voice, t("generation_page.default_voice"), userData=None)
        for key, voice in self.voices.items():
            add_item(self.voice, voice["name"], userData=key)
        self.voice.setCurrentIndex(max(0, self.voice.findData(old)))
        self.voice.blockSignals(False)
        # A model that is gone leaves the base model selected, with its own settings.
        self.voice_changed()
        self.refresh_history()

    def refresh_history(self):
        needle = self.history_filter.text().strip().lower()
        self.history.update_rows(
            [
                {
                    **item,
                    "text": item["request"]["text"],
                    "date": self.host.date(item["created"]),
                    "seconds": t("app.seconds", p0=fmt(item["duration"] or 0, ".2f", -1)),
                }
                for item in self.host.store.history()
                if not needle
                or needle in (str(item["request"]["text"]) + str(item["voice"])).lower()
            ]
        )
        self.history_playback.sync()
        rows = self.history.rows_model.rows
        for key, icon, tip in HISTORY_ACTIONS:
            self.history.set_icons(
                key, {row["id"]: icon for row in rows}, {row["id"]: tip for row in rows}
            )

    def history_icon_clicked(self, item, key):
        action = {
            "copy": self.copy_text,
            "again": self.regenerate,
            "fill": self.fill_back,
            "delete": self.delete_record,
        }
        if key in action:
            self.host.perform(lambda: action[key](item), self.error)

    def select_history(self, item):
        self.show_result(item)

    def show_new(self, record):
        """A generation that just finished: select its history row and play it at once,
        since that is what the user is waiting to hear."""
        rows = self.history.rows_model.rows
        position = next(
            (i for i, row in enumerate(rows) if Path(row["audio"]) == Path(record["audio"])), None
        )
        if position is not None:
            self.history.setCurrentIndex(self.history.model().index(position, 0))
        self.show_result(record, autoplay=True)

    def show_result(self, record, *, autoplay=False):
        """Put a finished or historical generation in the output card."""
        self.output_record = record
        candidates = record.get("candidates") or []
        self.fill_candidates(len(candidates) if len(candidates) > 1 else 0)
        self.choose_candidate(0, autoplay=autoplay)

    def fill_candidates(self, count):
        for widget in self.candidate_buttons:
            self.candidate_layout.removeWidget(widget)
            widget.deleteLater()
        self.candidate_buttons = []
        for index in range(count):
            choice = TogglePushButton(str(index + 1))
            choice.setFixedHeight(28)
            choice.setMinimumWidth(40)
            bind(choice, t("generation_page.candidate_choice", n=index + 1), "setAccessibleName")
            bind(choice, t("generation_page.candidate_choice", n=index + 1), "setToolTip")
            choice.clicked.connect(lambda _checked=False, i=index: self.choose_candidate(i, True))
            self.candidate_layout.addWidget(choice)
            self.candidate_buttons.append(choice)
        self.candidate_layout.addStretch(1)
        self.candidate_row.setVisible(count > 1)

    def choose_candidate(self, index, autoplay=False):
        """Load one candidate of the shown generation; export and the duration follow it."""
        record = self.output_record
        candidates = record.get("candidates") or [record["audio"]]
        self.output_index = index = min(index, len(candidates) - 1)
        self.output_audio = candidates[index]
        for position, choice in enumerate(self.candidate_buttons):
            choice.setChecked(position == index)
        self.player.load(self.output_audio, autoplay=autoplay)
        self.describe_output()

    def describe_output(self):
        record = self.output_record
        if record is None:
            set_text(self.result_label, t("generation_page.generated_audio_will_appear_here"))
            return
        seed = record.get("seed", record["request"].get("seed"))
        durations = record.get("durations") or []
        seconds = durations[self.output_index] if self.output_index < len(durations) else None
        set_text(
            self.result_label,
            t(
                "generation_page.output_meta",
                voice=fmt(record["voice"], "", -1),
                seed=fmt(seed, "", -1),
                steps=fmt(record["request"].get("num_steps", 0), "", -1),
                seconds=fmt(record["duration"] if seconds is None else seconds, ".2f", -1),
            ),
        )

    def copy_text(self, item):
        QApplication.clipboard().setText(item["request"]["text"])
        toast(self, t("generation_page.text_copied"), kind="success", duration=2000)

    def fill_back(self, item):
        """Write a history record's request back into every input."""
        request = item["request"]
        complete = True
        self.text.setPlainText(request["text"])
        set_text(self.caption, request.get("caption") or "")
        voice_id = item.get("voice_id") or self.voice_for_adapter(request.get("lora_adapter"))
        index = self.voice.findData(voice_id)
        if index >= 0:
            self.voice.setCurrentIndex(index)
        elif voice_id or request.get("lora_adapter"):
            complete = False
        else:
            self.voice.setCurrentIndex(0)
        if item.get("model") != model_files.FAST:
            # The fast model's four steps are its own, not a setting of the voice.
            self.steps.setValue(request.get("num_steps", 40))
        self.num_candidates.setValue(request.get("num_candidates", 1))
        # The seed actually used reproduces the audio, even when the request asked for a
        # random one.
        seed = item.get("seed", request.get("seed"))
        self.seed.setValue(-1 if seed is None else seed)
        self.seconds.setValue(request.get("seconds") or 0)
        self.duration_scale.setValue(request.get("duration_scale", 1))
        self.cfg_text.setValue(request.get("cfg_scale_text", 3))
        self.cfg_caption.setValue(request.get("cfg_scale_caption", 3))
        self.cfg_speaker.setValue(request.get("cfg_scale_speaker", 5))
        self.speaker_strength.setValue(request.get("speaker_kv_scale") or 1)
        self.trim.setChecked(request.get("trim_tail", True))
        self.references = []
        for path in request.get("ref_wavs") or []:
            file = Path(path)
            try:
                self.references.append(reference_item(file))
            except BusinessError:
                complete = False
        self.refresh_references()
        if complete:
            clear_text(self.error)
        else:
            raise BusinessError(t("generation_page.fill_back_incomplete"))

    def voice_for_adapter(self, adapter):
        """The voice whose adapter a record used (older records carry no voice ID)."""
        if not adapter:
            return None
        for key, voice in self.voices.items():
            if Path(voice["adapter"]) == Path(adapter):
                return key
        return None

    def delete_record(self, item):
        """Delete a history record and its audio, after letting go of the file if it is loaded."""
        if not self.host.confirm(
            t("generation_page.delete_record_title"),
            t("generation_page.delete_record_message"),
            t("generation_page.delete_record_action"),
        ):
            return
        # Any of the record's candidates may be the one loaded.
        if self.output_record and self.output_record["id"] == item["id"]:
            self.output_audio = None
            self.output_record = None
            self.fill_candidates(0)
            self.player.clear()
            self.describe_output()
        # The media backend releases the file on the next event-loop turns.
        QTimer.singleShot(
            250, lambda: self.host.perform(lambda: self.finish_delete(item), self.error)
        )

    def finish_delete(self, item):
        self.host.store.delete_history(item["id"])
        self.refresh_history()

    def regenerate(self, item=None):
        if self.host.tasks.busy:
            return
        if item := item or self.history.item():
            request = dict(item["request"])
            voice_id = item.get("voice_id") or self.voice_for_adapter(request.get("lora_adapter"))
            # The model may have moved with the data folder; use where it is now.
            voice = self.voices.get(voice_id)
            if voice:
                request["lora_adapter"] = voice["adapter"]
                if voice.get("base_id") is None:
                    raise BusinessError(voice["base_problem"])
                base = voice["base_id"]
            elif request.get("lora_adapter"):
                base = self.adapter_base(item, request["lora_adapter"])
            else:
                # The default voice again on the base the record used, when it is installed.
                match = base_models.match_record(item.get("base"), self.bases)
                base = match.base.id if match.base else self.generation_base()
            self.host.tasks.start(
                {
                    "kind": "generate",
                    "request": request,
                    "voice": item["voice"],
                    "voice_id": voice_id,
                    "base": base,
                    "output": str(self.host.store.root / "generations" / stamp()),
                }
            )

    def adapter_base(self, item, adapter):
        """The installed base for a record's adapter that is no listed voice (an older record,
        or a voice no longer registered): the base the record names, else the one its own
        files describe. Never another architecture's: such an adapter does not load."""
        match = base_models.match_record(item.get("base"), self.bases)
        if match.base is None:
            match = base_models.match_architecture(
                adapter_architecture(Path(adapter), self.bases), self.bases
            )
        if match.base is None:
            raise BusinessError(base_models.missing_message(match, item.get("base")))
        return match.base.id

    def open_output(self):
        if self.output_audio:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.output_audio).parent)))

    def export_audio(self):
        if not self.output_audio:
            raise BusinessError(t("generation_page.select_a_generated_result_first"))
        path = save_file(
            self,
            self.host.store.settings,
            t("generation_page.export_audio"),
            t("generation_page.speech_wav"),
            t("generation_page.wav_audio_wav"),
        )
        if path:
            audio = Path(self.output_audio)
            self.host.run_io(
                lambda: export_audio_file(audio, Path(path)),
                lambda _done: None,
                self.error,
                busy=t("app.work_export_audio"),
            )

    def task_changed(self, task):
        ours = task["kind"] == "generate"
        active = self.host.tasks.busy
        self.generate_button.setEnabled(not active and self.speech_ready)
        self.cancel_button.setEnabled(active and ours)
        self.show_progress(task if ours and active else None)
        if ours and task["status"] == "failed":
            set_text(self.error, task["message"])

    def show_progress(self, task):
        """The running generation in the output card: model loading, then sampling steps."""
        self.output_stack.setCurrentIndex(1 if task else 0)
        # The previous result's details would read as the one being made; they return if
        # this generation is stopped or fails.
        if task:
            set_text(self.result_label, t("generation_page.generating_new"))
        else:
            self.describe_output()
        total = (task or {}).get("total") or 0
        waiting = bool(task) and not total
        self.progress_bar.setVisible(not waiting)
        self.waiting_bar.setVisible(waiting)
        if waiting and not QApplication.instance().property("reduce_motion"):
            self.waiting_bar.start()
        else:
            self.waiting_bar.stop()
        if not task:
            return
        if total:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(task["step"])
            set_text(
                self.progress_text,
                t("generation_page.generating_steps", step=task["step"], total=total),
            )
        else:
            set_text(self.progress_text, task["message"])

    def settings_path(self, voice_id):
        if voice_id is None:
            return self.host.store.root / SETTINGS_FILE
        voice = self.voices.get(voice_id)
        return Path(voice["folder"]) / SETTINGS_FILE if voice and voice.get("folder") else None

    def parameters(self):
        return {
            "caption": self.caption.text(),
            "references": [
                {
                    "name": item.get("name"),
                    "path": self.host.store.portable(item["path"]),
                    "duration": item.get("duration") or 0,
                }
                for item in self.references
            ],
            "trim_tail": self.trim.isChecked(),
            **{name: getattr(self, name).value() for name in NUMBER_DEFAULTS},
        }

    def located_references(self, items):
        references = []
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict) and item.get("path"):
                # A data folder that moved still finds its reference copies.
                located = self.host.store.locate(item["path"])
                references.append(
                    {
                        **item,
                        "id": item.get("id") or identity(),
                        "path": str(located or item["path"]),
                        "name": item.get("name") or Path(item["path"]).name,
                        "duration": item.get("duration") or 0,
                    }
                )
        return references

    def apply_parameters(self, values):
        """Show these settings; anything missing or malformed takes its default."""
        values = values if isinstance(values, dict) else {}
        self.loading = True
        try:
            caption = values.get("caption")
            set_text(self.caption, caption if isinstance(caption, str) else "")
            self.trim.setChecked(values.get("trim_tail", True) is not False)
            self.references = self.located_references(values.get("references", []))
            self.refresh_references()
            for name, default in NUMBER_DEFAULTS.items():
                value = values.get(name, default)
                number = isinstance(value, int | float) and not isinstance(value, bool)
                getattr(self, name).setValue(value if number else default)
        finally:
            self.loading = False
        self.parameters_changed = False

    def saved_parameters(self, voice_id):
        path = self.settings_path(voice_id)
        if path is None or not path.is_file():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None

    def refresh_bases(self):
        """The base choice of the default voice: shown when there is a choice to make."""
        self.bases = self.host.bases()
        old = self.base.currentData() or self.host.store.settings.get("generation_base")
        self.base.blockSignals(True)
        clear_items(self.base)
        for base in self.bases:
            add_item(self.base, base.name, userData=base.id)
        self.base.setCurrentIndex(max(0, self.base.findData(old)))
        self.base.blockSignals(False)
        self.show_generation_model()

    def base_changed(self):
        self.host.store.settings["generation_base"] = self.base.currentData()
        self.host.store.save()
        self.show_generation_model()

    def installed_base(self, base_id):
        return next((base for base in self.bases if base.id == base_id), None)

    def generation_base(self):
        """The id of the base the chosen voice speaks with, None when it is not installed. A
        trained voice runs on the base it matches; the default voice on the chosen base."""
        voice = self.voices.get(self.voice.currentData())
        if voice:
            return voice.get("base_id")
        chosen = self.installed_base(self.base.currentData())
        chosen = chosen or (self.bases[0] if self.bases else None)
        return chosen.id if chosen else None

    def wanted_model(self):
        key = self.host.store.settings.get("generation_model")
        return key if key in model_files.GENERATION_MODELS else None

    def generation_model(self):
        """The variant the settings choose, when the voice's base has it installed; None is
        the base's standard checkpoint. Trained voices use it too: a LoRA trained on a base
        loads on that base's variants."""
        key = self.wanted_model()
        base = self.installed_base(self.generation_base())
        return key if key and base and key in base.variants else None

    def fast_steps(self):
        base = self.installed_base(self.generation_base())
        entry = base.catalog if base else None
        variant = entry.variant(model_files.FAST) if entry else None
        return variant.steps if variant and variant.steps else self.steps.value()

    def show_base(self):
        """The default voice's base choice, or the trained voice's base and remarks on it."""
        voice = self.voices.get(self.voice.currentData())
        self.base_field.setVisible(voice is None and len(self.bases) > 1)
        lines = []
        if voice:
            if voice.get("base_id") is None:
                lines.append(voice["base_problem"])
            else:
                lines.append(t("generation_page.voice_base", name=fmt(voice["base_name"], "", -1)))
                if voice.get("base_note"):
                    lines.append(voice["base_note"])
        base = self.installed_base(self.generation_base())
        key = self.wanted_model()
        if base and key and key not in base.variants:
            lines.append(
                t(
                    "generation_page.variant_not_installed",
                    name=fmt(base.name, "", -1),
                    model=t("settings_page.model_low_vram")
                    if key == model_files.LOW_VRAM
                    else t("settings_page.model_fast"),
                )
            )
        text = lines[0] if lines else ""
        for line in lines[1:]:
            text = t("common.lines", p0=text, p1=line)
        set_text(self.base_note, text)
        self.base_note.setVisible(bool(lines))

    def show_generation_model(self):
        """The fast model has its own steps and guidance; their fields rest while it is used."""
        self.show_base()
        fast = self.generation_model() == model_files.FAST
        for widget in (
            self.steps,
            self.cfg_text,
            self.cfg_caption,
            self.cfg_speaker,
            self.speaker_strength,
        ):
            widget.setEnabled(not fast)
        self.fast_note.setVisible(fast)

    def voice_changed(self):
        self.show_generation_model()
        voice_id = self.voice.currentData()
        if voice_id == self.voice_id:
            return
        # A change still waiting to be written belongs to the model being left.
        self.save_parameters()
        self.voice_id = voice_id
        self.apply_parameters(self.saved_parameters(voice_id))

    def parameter_changed(self, *_args):
        if not self.loading and self.voice_id is not ...:
            self.parameters_changed = True
            self.save_timer.start()

    def save_parameters(self):
        self.save_timer.stop()
        if not self.parameters_changed or self.voice_id is ...:
            return
        path = self.settings_path(self.voice_id)
        if path is not None and path.parent.is_dir():
            write_json(path, {"format_version": 1, **self.parameters()})
        self.parameters_changed = False

    def snapshot(self):
        self.save_parameters()
        return {
            "text": self.text.toPlainText(),
            "cursor": self.text.textCursor().position(),
            "caption": self.caption.text(),
            "references": self.references,
            "voice": self.voice.currentData(),
            "advanced": self.advanced.toggle.isChecked(),
            "trim_tail": self.trim.isChecked(),
            "history_filter": self.history_filter.text(),
            **{name: getattr(self, name).value() for name in NUMBER_DEFAULTS},
        }

    def restore(self, state):
        self.text.setPlainText(state.get("text", ""))
        cursor = self.text.textCursor()
        cursor.setPosition(min(state.get("cursor", 0), len(self.text.toPlainText())))
        self.text.setTextCursor(cursor)
        set_text(self.history_filter, state.get("history_filter", ""))
        self.voice.setCurrentIndex(max(0, self.voice.findData(state.get("voice"))))
        self.voice_changed()
        if self.saved_parameters(self.voice_id) is None:
            # Settings from before models kept their own: they become this model's.
            self.apply_parameters(state)
            self.parameters_changed = True
            self.save_parameters()
        self.advanced.set_open(state.get("advanced", False))
