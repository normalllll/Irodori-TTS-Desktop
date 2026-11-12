"""The models page: every model in one list. A model trained here is a project, whose
recordings, preprocessing and training runs are its steps; a model imported from a package
(or saved from one checkpoint) has no steps, only its details."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import FluentIcon
from qfluentwidgets.common.style_sheet import setCustomStyleSheet

from irodori_tts.desktop.messages import BusinessError, fmt, t

from .exports import export_voice, import_voice, voice_candidates
from .i18n import set_text
from .storage import project_lock, read_json, removable, remove_tree, tree_size, write_json
from .training import initialization_current
from .training_page import TrainingPage
from .widgets import (
    AppIcon,
    KeyValues,
    ModelList,
    Player,
    button,
    card,
    card_header,
    elided_label,
    error_label,
    ghost_button,
    icon_button,
    label,
    open_file,
    open_folder,
    prompt_choice,
    prompt_text,
    row,
    save_file,
    show_menu,
    toast,
)


def parameters(metadata):
    """rank / alpha / learning rate / batch of the training behind a model, if recorded."""
    record = (read_json(Path(metadata)) if metadata else None) or {}
    train = record.get("config", {}).get("train", {}) if isinstance(record, dict) else {}
    parts = []
    if "lora_r" in train:
        parts.append(f"rank {train['lora_r']}")
    if "lora_alpha" in train:
        parts.append(f"alpha {train['lora_alpha']}")
    if "learning_rate" in train:
        from .widgets import scientific

        parts.append("lr " + scientific(train["learning_rate"]))
    if "batch_size" in train:
        accumulation = train.get("gradient_accumulation_steps", 1)
        parts.append(f"batch {train['batch_size']}×{accumulation}")
    return " · ".join(parts) or "—"


# Below this page width the model list gives its room to the model's own columns.
NARROW = 1100


class ImportedModel(QWidget):
    """A model without a training project: its facts, its sample, and what it is for."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.host = page.host
        self.item = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 20, 24, 20)
        outer.setSpacing(14)
        self.title = elided_label(heading=True)
        self.switch_button = icon_button(
            FluentIcon.MENU, t("models_page.all_models"), lambda: page.model_menu(self)
        )
        self.switch_button.hide()
        outer.addWidget(
            row(
                self.switch_button,
                (self.title, 1),
                icon_button(
                    FluentIcon.EDIT,
                    t("models_page.rename_model"),
                    lambda: self.host.perform(self.rename),
                ),
                icon_button(
                    FluentIcon.FOLDER,
                    t("library_page.open_location"),
                    lambda: self.item and open_folder(self.item["adapter"]),
                ),
                spacing=4,
            )
        )
        # As tall as its content; the page below it stays empty rather than the card.
        _frame, body = card(outer)
        outer.addStretch(1)
        body.addWidget(card_header(t("models_page.details")))
        self.facts = KeyValues(
            [
                ("project", t("library_page.source")),
                ("step", t("library_page.training_steps")),
                ("parameters", t("library_page.parameters")),
                ("file", t("library_page.file")),
            ],
            key_width=72,
            paths=("file",),
        )
        body.addWidget(self.facts)
        body.addSpacing(6)
        body.addWidget(
            card_header(
                t("library_page.sample"),
                button(
                    t("generation_page.generate_again"),
                    lambda: self.host.perform(self.preview),
                    icon=FluentIcon.SYNC,
                ),
            )
        )
        self.player = Player()
        body.addWidget(self.player)
        self.sample_error = error_label(wrap=True)
        body.addWidget(self.sample_error)
        body.addSpacing(6)
        self.delete_button = ghost_button(
            t("library_page.delete_files_ellipsis"),
            lambda: self.host.perform(self.delete_files),
            icon=FluentIcon.DELETE,
        )
        setCustomStyleSheet(
            self.delete_button,
            "TransparentPushButton{color: #c42b1c}",
            "TransparentPushButton{color: #ff99a4}",
        )
        self.use_button = button(
            t("library_page.use_for_generation"),
            lambda: self.host.perform(self.use),
            primary=True,
            icon=AppIcon.INFERENCE,
        )
        body.addWidget(
            row(
                self.delete_button,
                None,
                button(
                    t("library_page.export_model"),
                    lambda: self.host.perform(self.export),
                    icon=FluentIcon.SHARE,
                ),
                self.use_button,
            )
        )

    def set_inset(self, left):
        margins = self.layout().contentsMargins()
        self.layout().setContentsMargins(left, margins.top(), margins.right(), margins.bottom())

    def show_model(self, item):
        self.item = item
        set_text(self.title, item["name"])
        self.facts.set("project", item.get("project_name") or t("app.imported_voice"))
        self.facts.set("step", str(item.get("step", "—")))
        self.facts.set("parameters", parameters(item.get("metadata")))
        self.facts.set("file", item.get("adapter", "—"))
        if item.get("preview") and Path(item["preview"]).is_file():
            if self.player.path != item["preview"]:
                self.player.load(item["preview"])
        else:
            self.player.clear()

    def require(self):
        if not self.item:
            raise BusinessError(t("library_page.select_a_voice_first"))
        record = read_json(Path(self.item["metadata"])) if self.item.get("metadata") else None
        if not isinstance(record, dict):
            self.host.refresh()
            raise BusinessError(t("library_page.model_files_are_missing"))
        return self.item, record

    def use(self):
        item, _record = self.require()
        self.host.use_model(item["id"])

    def preview(self):
        item, _record = self.require()
        self.host.preview_adapter(item)

    def rename(self):
        item, record = self.require()
        name, accepted = prompt_text(
            self, t("library_page.rename_voice"), t("library_page.voice_name"), value=item["name"]
        )
        if accepted and name.strip():
            record["name"] = name.strip()
            write_json(Path(item["metadata"]), record)
            self.host.refresh()

    def export(self):
        item, record = self.require()
        path = save_file(
            self,
            self.host.store.settings,
            t("library_page.export_voice_package"),
            item["name"] + ".irovoice",
            t("library_page.voice_packages_irovoice"),
        )
        if path:
            folder = Path(item["adapter"])
            run = {**record, "checkpoint": folder.name}
            self.host.run_io(
                lambda: export_voice(folder.parent, run, Path(path)),
                lambda _result: self.host.inform(t("library_page.voice_package_exported")),
                None,
                busy=t("app.work_export_voice"),
            )

    def delete_files(self):
        item, _record = self.require()
        if self.host.tasks.busy:
            raise BusinessError(t("library_page.wait_for_the_current_task_before_deleting"))
        root = self.host.store.root
        directory = Path(item["adapter"])
        removable(root, directory.relative_to(root))
        self.host.run_io(
            lambda: tree_size(directory),
            lambda size: self.confirm_delete(root, directory, size),
            None,
        )

    def confirm_delete(self, root, directory, size):
        if self.host.tasks.busy:
            raise BusinessError(t("library_page.wait_for_the_current_task_before_deleting"))
        if not self.host.confirm(
            t("library_page.delete_result_files"),
            t(
                "library_page.delete_this_run_its_checkpoints_adapters_and",
                p1=fmt(size / 1024**2, ".1f", -1),
                p3=fmt(directory, "", -1),
            ),
            t("library_page.delete_result_files_139"),
        ):
            return
        # Let go of every file this window may hold open in the folder: the sample player and
        # the resident compute process (which keeps the adapter loaded).
        self.player.clear()
        self.host.tasks.release(
            lambda released: self.host.perform(lambda: self.released(root, directory, released))
        )

    def released(self, root, directory, released):
        if not released:
            raise BusinessError(t("library_page.compute_process_still_holds_the_files"))
        # The media backend releases the file on the next event-loop turns.
        QTimer.singleShot(250, lambda: self.finish(root, directory))

    def finish(self, root, directory):
        def remove():
            with project_lock(root):
                remove_tree(root, directory)

        def settled():
            self.item = None
            self.host.refresh()

        self.host.run_io(
            remove, lambda _done: None, None, busy=t("app.work_deleting"), settled=settled
        )


class ModelsPage(QWidget):
    def __init__(self, host):
        super().__init__()
        self.host = host
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        side = QWidget()
        self.side = side
        lines = QVBoxLayout(side)
        lines.setContentsMargins(24, 20, 0, 20)
        lines.setSpacing(12)
        self.more = icon_button(FluentIcon.MORE, t("models_page.more_ways"), self.more_menu)
        lines.addWidget(
            row(
                label(t("app.my_models"), "page", wrap=False),
                None,
                icon_button(FluentIcon.ADD, t("models_page.new_model"), host.new_project),
                self.more,
                spacing=4,
            )
        )
        _frame, body = card(lines, stretch=1)
        body.setContentsMargins(4, 6, 4, 6)
        self.list = ModelList()
        self.list.selected.connect(self.show_row)
        body.addWidget(self.list, 1)
        layout.addWidget(side)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.empty = self.build_empty()
        self.stack.addWidget(self.empty)
        self.training = TrainingPage(host)
        self.stack.addWidget(self.training)
        self.imported = ImportedModel(self)
        self.stack.addWidget(self.imported)
        # Where a restored selection is applied, once the list is filled.
        self.wanted = host.store.settings.get("ui", {}).get("model")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # The list takes about a fifth of the page, wider windows show longer names. Below
        # NARROW the model's own columns need the width: the list becomes a menu by the title.
        narrow = self.width() < NARROW
        self.side.setVisible(not narrow)
        self.side.setFixedWidth(24 + max(176, min(236, round(self.width() * 0.25) - 75)))
        for view in (self.training, self.imported):
            view.switch_button.setVisible(narrow)
            view.set_inset(24 if narrow else 14)

    def model_menu(self, anchor):
        """The list as a menu, for a window too narrow to show it."""
        current = self.selection()
        entries = [
            (
                t("models_page.current_model", name=fmt(row["name"], "", -1))
                if row["id"] == current
                else row["name"],
                lambda key=row["id"]: self.list.select_id(key),
                row["id"] != current,
            )
            for row in self.list.rows_model.rows
            if not row.get("section")
        ]
        entries += [
            (t("models_page.new_model"), lambda: self.host.perform(self.host.new_project), True),
            (t("app.open_project"), lambda: self.host.perform(self.host.open_project), True),
            (
                t("app.import_project_package"),
                lambda: self.host.perform(self.host.import_project),
                True,
            ),
            (
                t("library_page.import_voice_package"),
                lambda: self.host.perform(self.import_model),
                True,
            ),
        ]
        show_menu(anchor.switch_button, entries)

    def build_empty(self):
        """No model yet: what a model is made of, and the two ways to get one."""
        box = QWidget()
        outer = QVBoxLayout(box)
        outer.setContentsMargins(14, 20, 24, 20)
        _frame, body = card(outer, stretch=1)
        body.addStretch(1)
        for widget in (
            label(t("models_page.empty_title"), "section", wrap=False),
            label(t("models_page.empty_hint"), "muted"),
        ):
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            body.addWidget(widget)
        body.addSpacing(8)
        body.addWidget(
            row(
                None,
                button(
                    t("models_page.new_model"),
                    lambda: self.host.perform(self.host.new_project),
                    primary=True,
                    icon=FluentIcon.ADD,
                ),
                button(
                    t("library_page.import_voice_package"),
                    lambda: self.host.perform(self.import_model),
                    icon=FluentIcon.DOWNLOAD,
                ),
                None,
            )
        )
        body.addStretch(2)
        return box

    def more_menu(self):
        show_menu(
            self.more,
            [
                (t("app.open_project"), lambda: self.host.perform(self.host.open_project), True),
                (
                    t("app.import_project_package"),
                    lambda: self.host.perform(self.host.import_project),
                    True,
                ),
                (
                    t("library_page.import_voice_package"),
                    lambda: self.host.perform(self.import_model),
                    True,
                ),
            ],
        )

    def project_state(self, root, project):
        """The line under a trained model's name: recordings, then where training stands."""
        tasks = self.host.tasks
        task = tasks.current if tasks.busy else None
        count = len(project["materials"])
        if (
            task
            and task.get("project") == str(root)
            and task["kind"] in {"train", "initialize", "recommend"}
        ):
            total = task.get("total") or 0
            percent = round(100 * task["step"] / total) if total else 0
            tone = "accent"
            if task["kind"] == "train":
                state = t("models_page.state_training", percent=percent)
            elif task["kind"] == "recommend":
                state = t("tasks.training_recommendation")
            else:
                state = t("models_page.state_preprocessing")
        elif any(
            run.get("checkpoint")
            for path in (root / "runs").glob("*/run.json")
            if isinstance(run := read_json(path, {}), dict)
        ):
            state, tone = t("models_page.state_trained"), "ok"
        elif not count:
            state, tone = t("models_page.state_no_recordings"), None
        elif initialization_current(project):
            state, tone = t("models_page.state_ready"), None
        else:
            state, tone = t("models_page.state_untrained"), None
        detail = t("models_page.project_detail", n=count, state=fmt(state, "", -1))
        return detail, tone

    def refresh(self):
        trained = []
        for root, project in self.host.store.projects():
            detail, tone = self.project_state(root, project)
            trained.append(
                {
                    "id": f"project:{root}",
                    "kind": "project",
                    "root": str(root),
                    "name": project["name"],
                    "detail": detail,
                    "tone": tone,
                }
            )
        imported = [
            {
                "id": f"model:{key}",
                "kind": "model",
                "entry": entry,
                "name": entry["name"],
                "detail": t(
                    "models_page.model_detail",
                    source=fmt(entry["project_name"], "", -1),
                    step=fmt(entry["step"], "", -1),
                ),
            }
            for key, entry in self.host.imported_models().items()
        ]
        rows = []
        if trained and imported:
            rows.append({"section": True, "id": "", "name": t("models_page.trained_here")})
        rows += trained
        if imported:
            rows.append({"section": True, "id": "", "name": t("models_page.imported")})
        rows += imported
        self.list.update_rows(rows)
        current = self.list.item()
        if self.stack.currentWidget() is self.training and self.training.root:
            # The project on show stays on show, also when it was opened by another route.
            shown = f"project:{Path(self.training.root).resolve()}"
            if (current is None or current["id"] != shown) and self.list.select_id(
                shown, emit=False
            ):
                current = self.list.item()
        if current is None:
            wanted = self.wanted or (
                f"project:{self.training.root}" if self.training.root else None
            )
            self.wanted = None
            ids = [row["id"] for row in rows if not row.get("section")]
            target = wanted if wanted in ids else ids[0] if ids else None
            if target is None:
                self.training.root = self.training.project = None
                self.stack.setCurrentWidget(self.empty)
                return
            self.list.select_id(target, emit=False)
            current = self.list.item()
        self.show_row(current)

    def show_row(self, row):
        if row["kind"] == "project":
            self.stack.setCurrentWidget(self.training)
            self.training.set_project(Path(row["root"]))
        else:
            self.stack.setCurrentWidget(self.imported)
            self.imported.show_model(row["entry"])

    def selection(self):
        item = self.list.item()
        return item["id"] if item else None

    def select_project(self, root):
        """Show this project (a path) and its steps."""
        self.list.select_id(f"project:{Path(root).resolve()}")

    def update_states(self):
        """Refresh the state lines (a task moved on) without reloading the shown model."""
        rows = []
        for entry in self.list.rows_model.rows:
            if entry.get("kind") == "project":
                root = Path(entry["root"])
                try:
                    project = self.training.project if root == self.training.root else None
                    project = project or read_json(root / "project.json")
                    detail, tone = self.project_state(root, project)
                except (OSError, ValueError, KeyError, TypeError):
                    detail, tone = entry["detail"], entry.get("tone")
                entry = {**entry, "detail": detail, "tone": tone}
            rows.append(entry)
        self.list.update_rows(rows)

    def import_model(self):
        path = open_file(
            self,
            self.host.store.settings,
            t("library_page.import_voice_package"),
            t("library_page.voice_import_files"),
        )
        if path:
            source = Path(path)
            self.host.run_io(
                lambda: voice_candidates(source, self.host.store.root / "voices"),
                lambda found: self.import_found(source, found),
                None,
                busy=t("app.work_import_voice"),
            )

    def import_found(self, source, found):
        """Import the usable adapter of a package, asking which when there are several; say
        which were left out and why."""
        usable = [item for item in found if item["problem"] is None]
        skipped = [item for item in found if item["problem"] is not None]
        if not usable:
            toast(self, skipped[0]["problem"])
            return
        if len(usable) == 1:
            folder = usable[0]["folder"]
        else:
            folder = prompt_choice(
                self,
                t("library_page.choose_voice_title"),
                t("library_page.choose_voice_caption"),
                [(f"{item['name']}  ({item['folder'] or '/'})", item["folder"]) for item in usable],
            )
            if folder is None:
                return
        if skipped:
            toast(
                self,
                t(
                    "library_page.voices_skipped",
                    names=fmt(", ".join(item["name"] for item in skipped), "", -1),
                    reason=fmt(skipped[0]["problem"], "", -1),
                ),
                kind="warning",
            )
        self.host.run_io(
            lambda: import_voice(source, self.host.store.root / "voices", folder),
            self.imported_done,
            None,
            busy=t("app.work_import_voice"),
        )

    def imported_done(self, folder):
        self.host.refresh()
        self.list.select_id(f"model:{Path(folder).name}")
