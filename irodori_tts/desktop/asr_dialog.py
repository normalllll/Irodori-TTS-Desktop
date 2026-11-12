"""The dialog that imports a folder of recordings and transcribes them automatically."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QButtonGroup, QFileDialog, QVBoxLayout, QWidget
from qfluentwidgets import SubtitleLabel

from irodori_tts.desktop.messages import fmt, t

from . import asr
from .i18n import add_item, clear_items, set_text, translated_widget
from .widgets import (
    AccentRadio,
    Dialog,
    PathField,
    accent_line,
    combo_box,
    ghost_button,
    label,
    row,
    switch,
)

GB = 1000**3
MB = 1000**2


def human_size(size: int):
    """'4.38 GB' / '60 MB' as a message part."""
    return fmt(f"{size / GB:.2f} GB" if size >= GB else f"{size / MB:.0f} MB", "", -1)


def downloads_needed(main: str, check: str | None) -> tuple[list, list]:
    """(models, add-ons) a run still has to download, as (backend, size) and (key, size)."""
    backends = asr.selection(main, check)
    models = [
        (backend, sum(model.size for model in asr.missing_models([backend])))
        for backend in backends
        if asr.missing_models([backend])
    ]
    addons = [(key, asr.ADDONS[key].size) for key in asr.missing_addons(backends)]
    return models, addons


def recordings_start(settings: dict) -> str:
    """Where a recordings picker opens: the folder last chosen, else the user's documents
    (not the working directory, which is the application's own folder)."""
    last = settings.get("recordings_folder")
    if last and Path(last).is_dir():
        return str(last)
    return QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)


def choose_recordings_folder(parent, settings: dict) -> Path | None:
    path = QFileDialog.getExistingDirectory(
        parent, t("training_page.choose_recording_folder"), recordings_start(settings)
    )
    if not path:
        return None
    settings["recordings_folder"] = path
    return Path(path)


class TranscribeDialog(Dialog):
    def __init__(self, parent, settings: dict, folder: Path | None = None, *, again=None):
        """again: the number of project recordings to transcribe again, instead of a folder."""
        super().__init__(parent)
        self.folder = folder
        self.settings = settings
        self.again = again
        main, check = asr.choice(settings)
        self.main_key, self.check_key = main, check
        layout = self.viewLayout
        layout.setSpacing(10)
        title = t("asr.again_title", count=again) if again else t("asr.dialog_title")
        layout.addWidget(translated_widget(SubtitleLabel, title, self))

        self.path = PathField(folder or "")
        self.path.setText(folder or "")
        self.empty = label(t("asr.no_folder"), "muted", wrap=False)
        source = row(
            label(t("asr.folder"), wrap=False),
            (self.path, 1),
            (self.empty, 1),
            ghost_button(t("asr.choose_folder"), self.choose_folder),
        )
        layout.addWidget(source)
        # Visibility only once placed: a widget shown while it has no parent opens as a window
        # of its own at the top left of the screen for a moment.
        self.path.value.setVisible(folder is not None)
        self.empty.setVisible(folder is None)
        source.setVisible(not again)
        if again:
            layout.addWidget(label(t("asr.again_hint"), "muted"))

        layout.addWidget(label(t("asr.main_backend"), "card", wrap=False))
        self.group = QButtonGroup(self)
        self.radios = {}
        self.states = {}
        for backend in asr.BACKENDS:
            box = QWidget()
            lines = QVBoxLayout(box)
            lines.setContentsMargins(0, 0, 0, 0)
            lines.setSpacing(0)
            radio = AccentRadio(backend.title)
            radio.setProperty("backend", backend.key)
            self.group.addButton(radio)
            self.radios[backend.key] = radio
            state = label("", "muted", wrap=False)
            self.states[backend.key] = state
            head = [radio]
            if backend.recommended:
                badge = label(t("asr.recommended"), wrap=False)
                badge.setTextColor(accent_line(False), accent_line(True))
                head.append(badge)
            head += [None, state]
            lines.addWidget(row(*head, spacing=10))
            indent = row(
                label("", wrap=False),
                (label(backend.hint, "muted"), 1),
                spacing=0,
            )
            indent.layout().itemAt(0).widget().setFixedWidth(28)
            lines.addWidget(indent)
            layout.addWidget(box)
        self.group.buttonClicked.connect(lambda _button: self.selection_changed())

        self.check = combo_box()
        layout.addWidget(
            row(
                label(t("asr.check_backend"), "card", wrap=False),
                (self.check, 1),
                spacing=12,
            )
        )
        layout.addWidget(label(t("asr.check_hint"), "muted"))
        self.split = switch()
        self.split.setChecked(True)
        splitting = row(self.split, (label(t("asr.split")), 1), spacing=10)
        layout.addWidget(splitting)
        # Recordings in a project are already cut to length.
        splitting.setVisible(not again)
        self.summary = label("", "muted")
        layout.addWidget(self.summary)

        self.radios[main].setChecked(True)
        self.fill_check(check)
        self.check.currentIndexChanged.connect(lambda _index: self.selection_changed(False))
        set_text(self.cancelButton, t("app.cancel"))
        self.widget.setMinimumWidth(720)
        self.selection_changed(False)

    def choose_folder(self):
        path = choose_recordings_folder(self, self.settings)
        if path:
            self.folder = path
            self.path.setText(self.folder)
            self.path.value.setVisible(True)
            self.empty.setVisible(False)
            self.selection_changed(False)

    def main(self) -> str:
        return next(k for k, radio in self.radios.items() if radio.isChecked())

    def check_choice(self) -> str:
        return self.check.currentData() or asr.NO_CHECK

    def fill_check(self, wanted: str):
        """The cross-check choices: none, or any backend but the main one."""
        main = self.main()
        self.check.blockSignals(True)
        clear_items(self.check)
        add_item(self.check, t("asr.check_none"), userData=asr.NO_CHECK)
        for backend in asr.BACKENDS:
            if backend.key != main:
                add_item(self.check, backend.title, userData=backend.key)
        self.check.setCurrentIndex(max(0, self.check.findData(wanted)))
        self.check.blockSignals(False)

    def selection_changed(self, main_changed=True):
        if main_changed:
            self.fill_check(self.check_choice())
        for backend in asr.BACKENDS:
            ready = not asr.missing_models([backend])
            set_text(
                self.states[backend.key],
                t(
                    "asr.backend_meta",
                    size=human_size(backend.size),
                    vram=fmt(f"{backend.vram:.0f}"),
                    state=t("asr.state_ready") if ready else t("asr.state_missing"),
                ),
            )
        models, addons = downloads_needed(self.main(), self.check_choice())
        parts = [
            t("asr.download_item", name=fmt(backend.title, "", -1), size=human_size(size))
            for backend, size in models
        ] + [
            t(
                "asr.addon_item",
                name=fmt(", ".join(b.title for b in asr.BACKENDS if b.addon == key), "", -1),
                size=human_size(size),
            )
            for key, size in addons
        ]
        text = None
        for part in parts:
            text = part if text is None else t("asr.list_join", first=text, next=part)
        self.summary.setVisible(bool(parts))
        if parts:
            set_text(self.summary, t("asr.will_download", items=text))
        set_text(self.yesButton, t("asr.download_and_start") if parts else t("asr.start"))
        self.yesButton.setEnabled(bool(self.again) or self.folder is not None)

    def values(self) -> dict:
        check = self.check_choice()
        return {
            "folder": self.folder,
            "main": self.main(),
            "check": None if check == asr.NO_CHECK else check,
            "split": self.split.isChecked(),
        }


def ask_transcription(
    parent, settings: dict, folder: Path | None = None, *, again=None
) -> dict | None:
    dialog = TranscribeDialog(parent.window(), settings, folder, again=again)
    return dialog.values() if dialog.exec() else None
