from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import FluentIcon, InfoBar, InfoBarPosition, ToolTipFilter

from irodori_tts.desktop.messages import BusinessError, fmt, t

from . import asr, base_models, model_files
from .asr_dialog import human_size
from .exports import cuda_of, export_diagnostics, storage_entries
from .i18n import add_item, bind, clear_text, set_text
from .resources import (
    CODEC_REVISION,
    hf_home,
    remove_snapshot,
    resource_state,
    snapshot_size,
)
from .storage import clean_project, project_lock, removable, remove_tree
from .widgets import (
    KeyValues,
    Page,
    PathField,
    SplitRow,
    Table,
    button,
    card,
    card_header,
    choose_folder,
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
    show_log,
    show_menu,
    swatch,
    switch,
)


def settings_grid(lines, *, muted=False, spacing=8):
    """Caption / control lines. The caption column is as wide as its longest caption in
    the current language, so no caption is cut and no control loses width to padding."""
    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(24)
    grid.setVerticalSpacing(spacing)
    grid.setColumnStretch(1, 1)
    for position, (title, control, align_right) in enumerate(lines):
        grid.addWidget(label(title, "muted" if muted else "", wrap=False), position, 0)
        if align_right:
            grid.addWidget(control, position, 1, alignment=Qt.AlignmentFlag.AlignRight)
        else:
            grid.addWidget(control, position, 1)
    return grid


class SettingsPage(Page):
    def __init__(self, host):
        super().__init__(t("app.settings_and_diagnostics"))
        self.host = host
        settings = host.store.settings
        # Two columns, one under the other when the window is too narrow for both.
        halves = QWidget(), QWidget()
        left, right = QVBoxLayout(halves[0]), QVBoxLayout(halves[1])
        for column in (left, right):
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(14)
        self.body.addWidget(SplitRow(*halves, spacing=14, stretch=(1, 1)), 1)

        # General: every change is saved as soon as it is made.
        _, body = card(left)
        body.setSpacing(4)
        body.addWidget(
            card_header(t("settings_page.general"), meta=t("settings_page.autosave_note"))
        )
        self.theme = combo_box()
        for text, key in (
            (t("settings_page.follow_system"), "system"),
            (t("settings_page.light"), "light"),
            (t("settings_page.dark"), "dark"),
        ):
            add_item(self.theme, text, userData=key)
        self.theme.setCurrentIndex(max(0, self.theme.findData(settings.get("theme", "system"))))
        self.accent = combo_box()
        for text, key in (
            (t("settings_page.accent_aqua"), "aqua"),
            (t("settings_page.accent_pink"), "pink"),
        ):
            add_item(self.accent, text, icon=swatch(key), userData=key)
        self.accent.setCurrentIndex(max(0, self.accent.findData(settings.get("accent", "aqua"))))
        self.language = combo_box()
        self.language.setObjectName("settings.language")
        add_item(self.language, t("settings.language_en"), userData="en")
        add_item(self.language, t("settings.language_zh"), userData="zh_CN")
        self.language.setCurrentIndex(max(0, self.language.findData(host.locale.language)))
        self.reduce = switch()
        self.reduce.setChecked(settings.get("reduce_motion", False))
        self.generation_model = combo_box()
        for text, key in (
            (t("settings_page.model_standard"), "standard"),
            (t("settings_page.model_low_vram"), model_files.LOW_VRAM),
            (t("settings_page.model_fast"), model_files.FAST),
        ):
            add_item(self.generation_model, text, userData=key)
        self.generation_model.setCurrentIndex(
            max(0, self.generation_model.findData(settings.get("generation_model", "standard")))
        )
        bind(self.generation_model, t("settings_page.generation_model_hint"), "setToolTip")
        self.project_root = PathField(host.store.project_root(), change=self.choose_root)
        body.addSpacing(6)
        body.addLayout(
            settings_grid(
                [
                    (t("settings.language"), self.language, False),
                    (t("settings_page.appearance"), self.theme, False),
                    (t("settings_page.accent"), self.accent, False),
                    (t("settings_page.reduce_motion"), self.reduce, True),
                    (t("settings_page.generation_model"), self.generation_model, False),
                    (t("settings_page.new_project_root"), self.project_root, False),
                ]
            )
        )
        body.addSpacing(6)
        body.addWidget(
            row(
                button(t("tour.restart"), lambda: self.host.start_tour(), icon=FluentIcon.FLAG),
                None,
            )
        )

        # Model files
        _, body = card(left)
        self.resource_error = error_label()
        header = card_header(t("exports.model_resources"), self.resource_error)
        self.resource_status = header.meta
        body.addWidget(header)
        # The installed base models: each with its commit (or that it was imported), the
        # update its repository offers, and its removal.
        self.base_table = Table(
            [
                ("name", t("generation_page.base_model"), 0),
                ("version", t("settings_page.base_version"), 120),
                ("actions", "", 84),
            ],
            fill=True,
            header=False,
            stretch=0,
        )
        self.base_table.set_row_actions(self.base_actions, for_rows="all")
        self.base_table.setFixedHeight(120)
        body.addWidget(self.base_table)
        self.latest = {}
        self.latest_checked = False
        lines = []
        # The pinned revision is for support, not for reading: show its short form.
        value = elided_label(CODEC_REVISION[:12], emphasis=True)
        value.setToolTip(CODEC_REVISION)
        value.installEventFilter(ToolTipFilter(value, 300))
        lines.append((t("settings_page.codec_name"), value, False))
        lines.append(
            # The files live in the shared download cache, not in the data folder.
            (t("settings_page.cache_location"), PathField(hf_home() / "hub"), False)
        )
        body.addLayout(settings_grid(lines, muted=True, spacing=4))
        self.prepare_button = button(
            t("settings_page.prepare_resources"),
            self.prepare,
            primary=True,
            icon=FluentIcon.DOWNLOAD,
        )
        self.verify_button = button(
            t("settings_page.verify_integrity"), self.verify, icon=FluentIcon.ACCEPT
        )
        self.repair_button = button(
            t("settings_page.download_again"), self.repair, icon=FluentIcon.SYNC
        )
        body.addWidget(row(self.prepare_button, self.verify_button, self.repair_button, None))
        # Getting models without downloading them, and handing them to someone who cannot.
        self.import_models_button = button(
            t("guide.import_files"),
            lambda: self.host.guide.choose_import(self.import_models_button),
            icon=FluentIcon.FOLDER,
        )
        self.export_models_button = button(
            t("models.export_title"),
            lambda: self.host.perform(self.host.guide.export, self.resource_error),
            icon=FluentIcon.SHARE,
        )
        self.import_base_button = button(
            t("settings_page.import_base"),
            lambda: self.choose_base_import(),
            icon=FluentIcon.ADD,
        )
        body.addWidget(
            row(
                self.import_models_button,
                self.export_models_button,
                self.import_base_button,
                None,
            )
        )
        self.resources_ready = False
        self.resources_size = None

        # Automatic transcription: defaults, and what it has downloaded.
        _, body = card(left)
        self.asr_error = error_label()
        body.addWidget(card_header(t("asr.settings_title"), self.asr_error))
        main, check = asr.choice(settings)
        self.asr_main = combo_box()
        self.asr_check = combo_box()
        for backend in asr.BACKENDS:
            add_item(self.asr_main, backend.title, userData=backend.key)
        add_item(self.asr_check, t("asr.check_none"), userData=asr.NO_CHECK)
        for backend in asr.BACKENDS:
            add_item(self.asr_check, backend.title, userData=backend.key)
        self.asr_main.setCurrentIndex(max(0, self.asr_main.findData(main)))
        self.asr_check.setCurrentIndex(max(0, self.asr_check.findData(check)))
        body.addWidget(
            grid(
                field(t("asr.default_main"), self.asr_main),
                field(t("asr.default_check"), self.asr_check),
            )
        )
        self.asr_files = Table(
            [
                ("name", t("asr.installed_name"), 0),
                ("kind", t("settings_page.category"), 100),
                ("size", t("settings_page.size"), 90),
                ("actions", "", 48),
            ],
            fill=True,
            header=False,
            stretch=0,
        )
        self.asr_files.set_row_actions(self.asr_actions, for_rows="all")
        # Six rows (models and add-ons); more scroll inside, so the page itself never has to.
        self.asr_files.setFixedHeight(240)
        body.addWidget(self.asr_files)

        left.addStretch(1)

        # Runtime environment
        _, body = card(right)
        self.environment_error = error_label()
        body.addWidget(
            card_header(
                t("settings_page.runtime_environment"),
                self.environment_error,
                button(t("settings_page.redetect"), self.diagnose, icon=FluentIcon.SYNC),
                icon_button(
                    FluentIcon.COPY,
                    t("settings_page.copy_runtime_information"),
                    lambda: set_text(QApplication.clipboard(), self.environment.as_text()),
                ),
            )
        )
        self.environment = KeyValues(
            [
                ("gpu", t("settings_page.gpu")),
                ("driver", t("settings_page.driver_and_cuda")),
                ("torch", t("settings_page.pytorch")),
                ("python", t("settings_page.python")),
                ("system", t("settings_page.system")),
            ],
            key_width=110,
        )
        body.addWidget(self.environment)

        # Logs and diagnostics
        _, body = card(right)
        self.diagnostic_error = error_label()
        body.addWidget(card_header(t("settings_page.logs_and_diagnostics"), self.diagnostic_error))
        self.retry_button = button(
            t("settings_page.retry_last_failed"),
            lambda: host.perform(self.retry, self.diagnostic_error),
            icon=FluentIcon.SYNC,
        )
        self.retry_button.setEnabled(False)
        body.addWidget(
            row(
                button(t("settings_page.view_task_log"), self.show_log, icon=FluentIcon.DOCUMENT),
                button(
                    t("settings_page.export_diagnostics"),
                    self.export_diagnostics,
                    icon=FluentIcon.ZIP_FOLDER,
                ),
                self.retry_button,
            )
        )
        self.include_logs = switch()
        body.addWidget(
            row(
                self.include_logs,
                (label(t("settings_page.include_raw_task_logs_in_diagnostics_may"), "muted"), 1),
                spacing=10,
            )
        )
        # Storage
        _, body = card(right, stretch=1)
        self.error = error_label()
        header = card_header(
            t("settings_page.storage"),
            self.error,
            button(t("settings_page.scan"), self.scan_storage, icon=FluentIcon.SEARCH),
        )
        self.storage_total = header.meta
        body.addWidget(header)
        self.storage = Table(
            [
                ("kind", t("settings_page.category"), 0),
                ("project", t("library_page.project"), 150),
                ("size", t("settings_page.size"), 100),
                ("actions", "", 48),
            ],
            fill=True,
            stretch=0,
        )
        # Every row that can be cleaned shows its delete button, not only the selected one.
        self.storage.set_row_actions(self.storage_actions, for_rows="all")
        body.addWidget(self.storage, 1)
        body.addWidget(label(t("settings_page.storage_hint"), "muted"))

        self.language.currentIndexChanged.connect(self.change_language)
        self.theme.currentIndexChanged.connect(self.save)
        self.accent.currentIndexChanged.connect(self.save)
        self.reduce.checkedChanged.connect(self.save)
        self.generation_model.currentIndexChanged.connect(self.save)
        self.asr_main.currentIndexChanged.connect(self.save_asr)
        self.asr_check.currentIndexChanged.connect(self.save_asr)

    def show_asr_choice(self):
        """Show the transcription defaults saved elsewhere (the getting-started page)."""
        main, check = asr.choice(self.host.store.settings)
        for combo, value in ((self.asr_main, main), (self.asr_check, check)):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(value)))
            combo.blockSignals(False)

    def save_asr(self):
        main, check = self.asr_main.currentData(), self.asr_check.currentData()
        # The cross-check is another model than the one it checks.
        if check == main:
            check = asr.NO_CHECK
            self.asr_check.blockSignals(True)
            self.asr_check.setCurrentIndex(self.asr_check.findData(check))
            self.asr_check.blockSignals(False)
        self.host.store.settings["asr"] = {"main": main, "check": check}
        self.host.store.save()

    def asr_rows(self):
        """Downloaded ASR models and installed add-ons, with the space they take."""
        rows = []
        for backend in asr.BACKENDS:
            size = sum(snapshot_size(model) for model in backend.models)
            if size:
                rows.append(
                    {
                        "id": f"model:{backend.key}",
                        "name": backend.title,
                        "kind": t("asr.kind_model"),
                        "size": human_size(size),
                        "tones": {"size": "muted", "kind": "muted"},
                    }
                )
        for key in asr.ADDONS:
            state = asr.addon_state(key)
            if state["size"]:
                titles = ", ".join(b.title for b in asr.BACKENDS if b.addon == key)
                rows.append(
                    {
                        "id": f"addon:{key}",
                        "name": titles,
                        "kind": t("asr.kind_addon"),
                        "size": human_size(state["size"]),
                        "tones": {"size": "muted", "kind": "muted"},
                    }
                )
        return rows

    def base_rows(self):
        rows = []
        for base in self.host.bases():
            version = (
                base.revision[:12]
                if base.source == "catalog" and base.revision
                else t("settings_page.base_imported")
            )
            rows.append(
                {
                    "id": base.id,
                    "name": base.name,
                    "version": version,
                    "tooltip": str(base.checkpoint),
                    "tones": {"version": "muted"},
                }
            )
        return rows

    def check_updates(self):
        """Ask each installed catalog base's repository for its latest commit, once a session
        and only when Hugging Face answers: an update is offered when it differs."""
        bases = [base for base in self.host.bases() if base.source == "catalog"]
        if self.latest_checked or not bases:
            return
        self.latest_checked = True

        def latest():
            found = {}
            for base in bases:
                try:
                    found[base.id] = base_models.latest_revision(base.repo)
                except Exception:
                    continue
            return found

        def checked(found):
            self.latest = found
            self.base_table.update_rows(self.base_rows())

        self.host.run_io(latest, checked, None, blocking=False)

    def base_actions(self, item):
        box = QWidget()
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)
        base = next((b for b in self.host.bases() if b.id == item["id"]), None)
        latest = self.latest.get(item["id"])
        if base is not None and base.source == "catalog" and latest and latest != base.revision:
            layout.addWidget(
                icon_button(
                    FluentIcon.UPDATE,
                    t("settings_page.base_update", revision=fmt(latest[:12], "", -1)),
                    lambda: self.host.perform(lambda: self.update_base(item), self.resource_error),
                    size=28,
                )
            )
        layout.addWidget(
            icon_button(
                FluentIcon.DELETE,
                t("asr.remove"),
                lambda: self.host.perform(lambda: self.remove_base(item), self.resource_error),
                size=28,
            )
        )
        return box

    def update_base(self, item):
        if self.host.tasks.busy:
            raise BusinessError(t("settings_page.wait_for_compute_or_download_tasks_before"))
        clear_text(self.resource_error)
        self.host.tasks.start({"kind": "resources", "bases": [item["id"]], "update": True})

    def remove_base(self, item):
        if self.host.tasks.busy:
            raise BusinessError(t("settings_page.wait_for_compute_or_download_tasks_before"))
        base = next((b for b in self.host.bases() if b.id == item["id"]), None)
        if base is None:
            return
        if not self.host.confirm(
            t("asr.remove_title", name=fmt(base.name, "", -1)),
            t("settings_page.remove_base_message", name=fmt(base.name, "", -1)),
            t("asr.remove"),
        ):
            return
        root = self.host.resource_root

        def remove():
            if base.source == "imported":
                base_models.delete_imported(root, base)
            else:
                base_models.remove_catalog_files(root, base)

        self.host.run_io(remove, lambda _done: self.host.refresh(), self.resource_error)

    def choose_base_import(self):
        show_menu(
            self.import_base_button,
            [
                (t("settings_page.import_base_folder"), self.import_base_folder, True),
                (t("settings_page.import_base_file"), self.import_base_file, True),
            ],
        )

    def import_base_folder(self):
        path = choose_folder(self, self.host.store.settings, t("settings_page.import_base"))
        if path:
            self.host.perform(lambda: self.import_base(Path(path)), self.resource_error)

    def import_base_file(self):
        path = open_file(
            self,
            self.host.store.settings,
            t("settings_page.import_base"),
            t("settings_page.safetensors_filter"),
        )
        if path:
            self.host.perform(lambda: self.import_base(Path(path)), self.resource_error)

    def import_base(self, source):
        if self.host.tasks.busy:
            raise BusinessError(t("settings_page.wait_for_compute_or_download_tasks_before"))
        clear_text(self.resource_error)
        root = self.host.resource_root

        def imported(base):
            self.host.refresh()
            self.host.notify(t("settings_page.base_imported_as", name=fmt(base.name, "", -1)))

        self.host.run_io(
            lambda: base_models.import_base(source, root),
            imported,
            self.resource_error,
            busy=t("settings_page.work_import_base"),
        )

    def asr_actions(self, item):
        box = QWidget()
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            icon_button(
                FluentIcon.DELETE,
                t("asr.remove"),
                lambda: self.host.perform(lambda: self.remove_asr(item), self.asr_error),
                size=28,
            )
        )
        return box

    def remove_asr(self, item):
        if self.host.tasks.busy:
            raise BusinessError(t("settings_page.wait_for_compute_or_download_tasks_before"))
        kind, key = item["id"].split(":", 1)
        if not self.host.confirm(
            t("asr.remove_title", name=fmt(item["name"], "", -1)),
            t("asr.remove_message", name=fmt(item["name"], "", -1), size=fmt(item["size"], "", -1)),
            t("asr.remove"),
        ):
            return

        def remove():
            if kind == "addon":
                asr.remove_addon(key)
            else:
                for model in asr.backend_for(key).models:
                    remove_snapshot(model)

        self.host.run_io(remove, lambda _result: self.refresh(), self.asr_error)

    def retry(self):
        self.host.retry_task()
        clear_text(self.diagnostic_error)

    def change_language(self):
        language = self.language.currentData()
        self.host.store.settings["language"] = language
        self.host.store.save()
        self.host.locale.set_language(language)

    def choose_root(self):
        path = QFileDialog.getExistingDirectory(
            self, t("settings_page.choose_new_project_root"), self.project_root.text()
        )
        if path:
            self.project_root.setText(str(Path(path)))
            self.save()

    def save(self):
        settings = self.host.store.settings
        before = (settings.get("theme"), settings.get("reduce_motion"), settings.get("accent"))
        settings.update(
            theme=self.theme.currentData(),
            accent=self.accent.currentData(),
            reduce_motion=self.reduce.isChecked(),
            generation_model=self.generation_model.currentData(),
            project_root=self.host.store.portable(self.project_root.text()),
        )
        self.host.store.save()
        # The fast model's fields on Generate rest the moment it is chosen.
        self.host.generation.show_generation_model()
        # Only what changed: re-applying the theme restyles every widget, which a change of
        # colour alone does not need.
        if before[:2] != (settings["theme"], settings["reduce_motion"]):
            self.host.apply_theme()
        elif before[2] != settings["accent"]:
            self.host.apply_accent()

    def prepare(self):
        clear_text(self.resource_error)
        self.host.tasks.start({"kind": "resources"})

    def verify(self):
        clear_text(self.resource_error)
        self.host.tasks.start({"kind": "resources", "verify": True})

    def repair(self):
        if self.host.confirm(
            t("settings_page.download_pinned_resources_again"),
            t("settings_page.download_the_model_codec_and_watermark_resources"),
            t("settings_page.download_again"),
        ):
            self.host.tasks.start({"kind": "resources", "repair": True})

    def diagnose(self):
        clear_text(self.environment_error)
        self.host.detect_device(self.environment_error)

    def show_environment(self, report):
        gpu = report.get("gpu")
        torch = report.get("torch")
        self.environment.set(
            "gpu",
            t(
                "settings_page.gpu_memory",
                name=fmt(gpu["name"], "", -1),
                memory=fmt(f"{gpu['total_mib'] / 1024:.1f} GB", "", -1),
            )
            if gpu
            else report.get("gpu_error") or "—",
        )
        self.environment.set(
            "driver",
            t(
                "settings_page.driver_cuda_value",
                driver=fmt(gpu["driver"] if gpu else "—", "", -1),
                cuda=fmt(cuda_of(torch) or "—", "", -1),
            ),
        )
        capable = gpu.get("compute_cap") if gpu else None
        version = fmt(torch, "", -1)
        if not torch:
            summary = "—"
        elif capable is None:
            summary = torch
        elif capable >= 8:
            summary = t("settings_page.torch_bf16", version=version)
        else:
            summary = t("settings_page.torch_no_bf16", version=version)
        self.environment.set("torch", summary)
        self.environment.set("python", report["python"])
        memory = report.get("memory")
        self.environment.set(
            "system",
            t(
                "settings_page.system_value",
                system=fmt(report["system"], "", -1),
                cpu=fmt(report.get("cpu") or "—", "", -1),
                memory=fmt(f"{memory / 1024**3:.0f} GB" if memory else "—", "", -1),
            ),
        )

    def show_log(self):
        text = t("settings_page.no_log_output_yet")
        if self.host.tasks.current:
            path = self.host.tasks.task_dir() / "worker.log"
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="replace")[-100000:]
        show_log(self, t("settings_page.task_diagnostics"), text)

    def export_diagnostics(self):
        path = save_file(
            self,
            self.host.store.settings,
            t("settings_page.export_diagnostics"),
            t("settings_page.irodori_diagnostics_zip"),
            t("files.zip"),
        )
        if path:
            include_logs = self.include_logs.isChecked()
            self.host.run_io(
                lambda: export_diagnostics(self.host.store, Path(path), include_logs=include_logs),
                lambda result: InfoBar.success(
                    "",
                    str(t("settings_page.diagnostics_exported")),
                    duration=3000,
                    position=InfoBarPosition.TOP_RIGHT,
                    parent=self.window(),
                ),
                self.diagnostic_error,
                busy=t("app.work_export_diagnostics"),
            )

    def scan_storage(self):
        self.host.run_io(
            lambda: storage_entries(self.host.store),
            self.show_storage,
            self.error,
        )

    def show_storage(self, rows):
        self.storage.update_rows(
            [
                {
                    **item,
                    "size": f"{item['bytes'] / 1024**2:.1f} MB",
                    "tooltip": item["path"],
                    "tones": {"size": "muted"},
                }
                for item in rows
            ]
        )
        total = sum(item["bytes"] for item in rows)
        set_text(
            self.storage_total,
            t("settings_page.storage_total", size=fmt(f"{total / 1024**2:.1f} MB", "", -1)),
        )

    def storage_actions(self, item):
        box = QWidget()
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        if not item.get("referenced"):
            layout.addWidget(
                icon_button(
                    FluentIcon.DELETE,
                    t("settings_page.clean_selected_item"),
                    lambda: self.host.perform(lambda: self.clean(item), self.error),
                    size=28,
                )
            )
        return box

    def clean(self, row=None):
        row = row or self.storage.item()
        if not row:
            raise BusinessError(t("settings_page.scan_storage_and_select_an_item_first"))
        root = Path(row["root"])
        if row["kind_id"] == "partial_downloads" and self.host.tasks.busy:
            raise BusinessError(t("settings_page.wait_for_compute_or_download_tasks_before"))
        if self.host.project_busy(root):
            raise BusinessError(t("settings_page.a_project_task_is_running_wait_before"))
        if row.get("referenced"):
            raise BusinessError(t("settings_page.this_is_the_recovery_checkpoint_used_by"))
        if not self.host.confirm(
            t("settings_page.clean_selected_files"),
            t(
                "settings_page.this_cannot_be_undone",
                p0=fmt(row["kind"], "", -1),
                p2=fmt(row["project"], "", -1),
                p4=fmt(row["size"], "", -1),
                p6=fmt(row["path"], "", -1),
            ),
            t("settings_page.clean_files"),
        ):
            return

        def remove():
            with project_lock(root):
                path = removable(root, row["relative"])
                if row["kind_id"] == "partial_downloads" and path.name.endswith(".incomplete"):
                    path.unlink(missing_ok=True)
                elif row["kind_id"] == "temporary_files":
                    clean_project(root, row["relative"])
                elif (
                    row["kind_id"] == "historical_checkpoints"
                    and "/checkpoints/" in row["relative"]
                ):
                    remove_tree(root, path)
                else:
                    raise BusinessError(
                        t("settings_page.directory_is_outside_the_allowed_cleanup_scope")
                    )

        self.host.run_io(
            remove,
            lambda _done: self.scan_storage(),
            self.error,
            busy=t("app.work_cleaning"),
        )

    def showEvent(self, event):
        super().showEvent(event)
        self.check_updates()

    def refresh(self):
        ready, size = resource_state(self.host.resource_root)
        manifest = size is not None
        if manifest:
            set_text(
                self.resource_status,
                t(
                    "settings.resource_size",
                    state=t("settings_page.resources_ready")
                    if ready
                    else t("settings_page.resources_incomplete"),
                    size=size,
                ),
            )
        else:
            set_text(self.resource_status, t("settings_page.resources_not_downloaded"))
        self.asr_files.update_rows(self.asr_rows())
        self.base_table.update_rows(self.base_rows())
        self.resources_ready = ready
        self.resources_size = size if manifest else None
        self.resource_status.setTextColor(
            QColor("#107c10") if ready else QColor("#5b6068"),
            QColor("#6ccb5f") if ready else QColor("#bfc5ce"),
        )
        self.refresh_actions()

    def refresh_actions(self):
        busy = self.host.tasks.busy
        self.prepare_button.setVisible(not self.resources_ready)
        self.verify_button.setVisible(self.resources_ready)
        self.repair_button.setVisible(self.resources_ready)
        for widget in (
            self.prepare_button,
            self.verify_button,
            self.repair_button,
            self.import_models_button,
            self.export_models_button,
            self.import_base_button,
        ):
            widget.setEnabled(not busy)
        task = self.host.tasks.current
        self.retry_button.setEnabled(bool(task) and not busy and task["status"] != "completed")

    def task_changed(self, task):
        if task["kind"] == "resources":
            set_text(self.resource_status, task["message"])
            if task["status"] == "failed":
                set_text(self.resource_error, task["message"])
            elif task["status"] in {"completed", "stopped"}:
                clear_text(self.resource_error)
        self.refresh_actions()
