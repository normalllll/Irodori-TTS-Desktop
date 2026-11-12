"""Business localization through real Qt widgets and the worker event protocol."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QCoreApplication, QEvent, QLocale, Qt
from PySide6.QtWidgets import QApplication

from irodori_tts.desktop.app import MainWindow
from irodori_tts.desktop.i18n import clear_text, locale_manager, set_text
from irodori_tts.desktop.materials import import_materials
from irodori_tts.desktop.messages import BusinessError, Message, catalog, fmt, t
from irodori_tts.desktop.storage import AppStore, create_project, read_json, write_json
from irodori_tts.desktop.tasks import TaskManager
from irodori_tts.desktop.widgets import label
from irodori_tts.task_control import TaskChannel
from scripts.translations import validate
from tests.qt_cleanup import cleanup_qt, stub_connection_check

REPO = Path(__file__).resolve().parents[1]


def setUpModule():
    unittest.addModuleCleanup(stub_connection_check().stop)


class InternationalizationTests(unittest.TestCase):
    def tearDown(self):
        cleanup_qt()

    @classmethod
    def setUpClass(cls):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        cls.app = QApplication.instance() or QApplication([])
        cls.locale = locale_manager()

    def setUp(self):
        self.locale.set_language("zh_CN")

    def settle(self):
        for _ in range(4):
            self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_catalog_coverage_and_native_plural_fallback(self):
        report = validate()
        self.assertGreater(report["messages_per_language"], 350)
        for language in ("en", "zh_CN"):
            self.locale.set_language(language)
            for key, translation in catalog(language).items():
                for n in (1, 2) if isinstance(translation, list) else (-1,):
                    expected = (
                        translation[0 if language == "zh_CN" or n == 1 else 1]
                        if isinstance(translation, list)
                        else translation
                    )
                    expected = expected.replace("%Ln", QLocale().toString(n)).replace("%n", str(n))
                    self.assertEqual(
                        QCoreApplication.translate("Desktop", key, None, n), expected, key
                    )
        self.locale.set_language("en")
        self.assertEqual(str(t("generation_page.characters", n=1)), "1 character")
        self.assertEqual(str(t("generation_page.characters", n=2)), "2 characters")
        self.locale.set_language("zh_CN")
        self.assertEqual(str(t("generation_page.characters", n=2)), "2 字符")
        # English translator remains installed beneath the selected catalog.
        selected = self.locale.translators[-1]
        self.app.removeTranslator(selected)
        self.assertEqual(str(t("app.home")), "Overview")
        self.app.installTranslator(selected)

    def test_switch_preserves_widgets_drafts_selection_focus_scroll_and_language(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AppStore(Path(directory))
            root = create_project(store.root / "projects", "User project 日本語")
            import_materials(
                root, REPO / "examples/desktop", REPO / "examples/desktop/transcripts.csv"
            )
            store.register(root)
            window = MainWindow(store)
            window.stack.setAnimationEnabled(False)
            window.resize(1200, 760)
            window.show()
            window.navigate(1)
            window.generation.text.setPlainText("日本語 draft that must stay")
            window.generation.caption.setText("User caption")
            window.generation.trim.setChecked(False)
            window.generation.advanced.set_open(True)
            window.generation.advanced.animation.setCurrentTime(220)
            window.training.voice_name.setText("User voice")
            window.training.materials.setCurrentIndex(window.training.materials.model().index(0, 0))
            self.settle()
            editor = window.generation.text
            cursor = editor.textCursor()
            cursor.setPosition(3)
            cursor.setPosition(8, cursor.MoveMode.KeepAnchor)
            editor.setTextCursor(cursor)
            editor.setFocus()
            window.generation.verticalScrollBar().setValue(70)
            self.settle()
            generation = window.generation.snapshot()
            training = window.training.snapshot()
            selected = window.training.materials.item()["id"]
            scroll = window.generation.verticalScrollBar().value()
            focus = self.app.focusWidget()
            widgets = tuple(window.pages), editor, window.training.materials
            window.settings_page.language.setCurrentIndex(
                window.settings_page.language.findData("en")
            )
            self.settle()
            self.assertEqual(window.generation.snapshot(), generation)
            self.assertEqual(window.training.snapshot(), training)
            self.assertEqual(window.training.materials.item()["id"], selected)
            self.assertEqual(window.generation.verticalScrollBar().value(), scroll)
            self.assertIs(self.app.focusWidget(), focus)
            self.assertEqual(
                (tuple(window.pages), window.generation.text, window.training.materials), widgets
            )
            self.assertEqual(editor.textCursor().selectionStart(), 3)
            self.assertEqual(editor.textCursor().selectionEnd(), 8)
            self.assertEqual(window.generation.generate_button.text(), "Generate")
            self.assertEqual(window.training.pivot.widget("0").text(), "Recordings")
            self.assertEqual(window.navigationInterface.widget("page_0").text(), "Getting started")
            model = window.training.materials.model()
            column = next(i for i, c in enumerate(model.columns) if c[0] == "file_name")
            self.assertEqual(model.headerData(column, Qt.Orientation.Horizontal), "Filename")
            self.assertEqual(window.generation.voice.currentData(), generation["voice"])
            self.assertFalse(window.generation.trim.isChecked())
            self.assertEqual(read_json(store.root / "settings.json")["language"], "en")
            window.save_ui()
            window.close()
            self.settle()
            window.deleteLater()
            self.settle()
            reopened = MainWindow(AppStore(store.root))
            self.settle()
            self.assertEqual(reopened.locale.language, "en")
            self.assertEqual(reopened.generation.text.toPlainText(), generation["text"])
            self.assertFalse(reopened.generation.trim.isChecked())
            self.assertEqual(reopened.generation.generate_button.text(), "Generate")
            reopened.close()
            self.settle()
            reopened.deleteLater()
            self.settle()

    def test_persistent_worker_event_translates_on_arrival_and_after_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AppStore(Path(directory))
            manager = TaskManager(store, REPO)
            manager.timer.stop()
            task = {
                "id": "locale-event",
                "kind": "resources",
                "status": "running",
                "sequence": 0,
                "created": "2026-10-27T00:00:00+00:00",
                "step": 0,
                "total": 0,
                "message": "",
            }
            manager.current = task
            manager.tasks = [task]
            target = manager.task_dir()
            target.mkdir(parents=True)
            raw_detail = "Qt/backend Error: code=7"
            message = t(
                "materials.recording_error",
                file="user.wav",
                error=t("widgets.playback_failed") + raw_detail,
            )
            with patch.dict(
                os.environ,
                IRODORI_TASK_ID=task["id"],
                IRODORI_TASK_EVENTS=str(target / "events.jsonl"),
            ):
                TaskChannel().emit("error", message=message)
            raw = json.loads((target / "events.jsonl").read_text("utf-8"))
            self.assertEqual(
                raw["_i18n"]["fields"]["message"]["$message"], "materials.recording_error"
            )
            self.locale.set_language("en")
            manager.poll()
            self.assertIn("Playback failed", str(task["message"]))
            self.assertIn(raw_detail, str(task["message"]))
            self.assertEqual(task["status"], "failed")
            saved = read_json(target / "task.json")
            self.assertIsInstance(saved["message"], Message)
            view = label(saved["message"])
            self.locale.set_language("zh_CN")
            self.assertIn("播放失败", view.text())
            self.assertIn(raw_detail, view.text())
            manager.finish_timer.stop()
            manager.deleteLater()
            view.deleteLater()
            self.settle()

    def test_worker_business_errors_keep_ids_and_external_errors_remain_raw(self):
        from irodori_tts.desktop.worker import execute

        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "events.jsonl"
            with patch.dict(
                os.environ, IRODORI_TASK_ID="error-translation", IRODORI_TASK_EVENTS=str(events)
            ):
                with patch(
                    "irodori_tts.desktop.worker.resolve_resources",
                    side_effect=BusinessError(
                        t(
                            "resources.resource_verification_failed_prepare_resources_again",
                            p1=fmt("weights.pth"),
                        )
                    ),
                ):
                    with contextlib.redirect_stderr(io.StringIO()):
                        execute({"kind": "resources", "resource_root": directory})
                error = json.loads(events.read_text("utf-8").splitlines()[-1])
                self.assertIn("_i18n", error)
                events.unlink()
                with patch(
                    "irodori_tts.desktop.worker.resolve_resources",
                    side_effect=RuntimeError("Backend raw diagnostic 123"),
                ):
                    with contextlib.redirect_stderr(io.StringIO()):
                        execute({"kind": "resources", "resource_root": directory})
                error = json.loads(events.read_text("utf-8").splitlines()[-1])
                self.assertEqual(error["message"], "Backend raw diagnostic 123")
                self.assertNotIn("_i18n", error)

    def test_empty_error_does_not_reappear_and_user_data_has_no_translation_metadata(self):
        widget = label("")
        set_text(widget, t("generation_page.enter_japanese_text_to_read"))
        clear_text(widget)
        self.locale.set_language("en")
        self.assertEqual(widget.text(), "")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            write_json(
                path,
                {
                    "name": t("training_page.natural_voice"),
                    "text": "日本語",
                    "caption": "natural",
                    "presets": {"_i18n": {"custom": 7}},
                    "error": t("tasks.complete_checkpoint_saved"),
                },
            )
            raw = json.loads(path.read_text("utf-8"))
            self.assertEqual(set(raw["_i18n"]["fields"]), {"error"})
            self.assertEqual(raw["text"], "日本語")
            self.assertEqual(read_json(path)["presets"], {"_i18n": {"custom": 7}})
            self.locale.set_language("zh_CN")
            self.assertEqual(str(read_json(path)["error"]), "完整检查点已保存")
        widget.deleteLater()
        self.settle()

    def test_coverage_rejects_missing_keys_and_placeholder_mismatch(self):
        import shutil
        import xml.etree.ElementTree as ET

        from scripts import translations

        with tempfile.TemporaryDirectory() as directory:
            locales = Path(directory)
            for language in ("en", "zh_CN"):
                shutil.copy2(translations.LOCALES / f"{language}.ts", locales / f"{language}.ts")
            path = locales / "zh_CN.ts"
            original = path.read_bytes()
            tree = ET.parse(path)
            tree.find("context").remove(tree.find("context/message"))
            tree.write(path, encoding="utf-8")
            with (
                patch.object(translations, "LOCALES", locales),
                self.assertRaisesRegex(ValueError, "key sets"),
            ):
                validate()
            path.write_bytes(original)
            tree = ET.parse(path)
            for message in tree.findall(".//message"):
                if message.findtext("source") == "materials.recording_error":
                    message.find("translation").text = "{wrong}: {error}"
            tree.write(path, encoding="utf-8")
            with (
                patch.object(translations, "LOCALES", locales),
                self.assertRaisesRegex(ValueError, "Placeholder mismatch"),
            ):
                validate()

    def test_reference_truncation_and_cleanup_use_stable_business_codes(self):
        from PySide6.QtCore import QThreadPool

        from irodori_tts.desktop.exports import storage_entries

        for language in ("zh_CN", "en"):
            with self.subTest(language=language), tempfile.TemporaryDirectory() as directory:
                store = AppStore(Path(directory))
                store.settings["language"] = language
                root = create_project(store.root / "projects", "User project")
                store.register(root)
                temporary = root / "temp" / "disposable"
                temporary.mkdir(parents=True)
                (temporary / "data").write_text("temporary", "utf-8")
                window = MainWindow(store)
                window.settings_page.storage.update_rows(
                    [
                        {**row, "size": str(row["bytes"])}
                        for row in storage_entries(store)
                        if row["id"] == str(temporary)
                    ]
                )
                window.settings_page.storage.row_animation.setCurrentTime(220)
                window.settings_page.storage.setCurrentIndex(
                    window.settings_page.storage.model().index(0, 0)
                )
                source = REPO / "examples/desktop/basic.wav"
                with (
                    patch(
                        "irodori_tts.desktop.widgets.QFileDialog.getOpenFileNames",
                        return_value=([str(source)], ""),
                    ),
                    patch(
                        "irodori_tts.desktop.generation_page.inspect_audio",
                        return_value=(
                            13.0,
                            t("materials.recording_exceeds_12_seconds_split_the_audio"),
                        ),
                    ),
                ):
                    window.generation.add_references()
                    self.assertTrue(QThreadPool.globalInstance().waitForDone(5000))
                    self.settle()
                self.assertEqual(len(window.generation.references), 1)
                self.assertEqual(window.generation.references[0]["duration"], 13.0)
                self.locale.set_language("en" if language == "zh_CN" else "zh_CN")
                with patch.object(window, "confirm", return_value=True):
                    window.settings_page.clean()
                self.assertTrue(QThreadPool.globalInstance().waitForDone(5000))
                self.assertFalse(temporary.exists())
                self.assertTrue(QThreadPool.globalInstance().waitForDone(5000))
                self.settle()
                window.close()
                self.settle()
                window.deleteLater()
                self.settle()

    def test_worker_protocol_has_no_qt_runtime_dependency(self):
        code = (
            "import sys,json; from irodori_tts.desktop.worker import execute; "
            "from irodori_tts.desktop.messages import t,persist; "
            "assert 'PySide6' not in sys.modules; "
            "print(json.dumps(persist({'message':t('worker.loading_voice_model')})))"
        )
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.assertIn("_i18n", json.loads(result.stdout))


if __name__ == "__main__":
    from tests.run import guarded_main

    raise SystemExit(guarded_main(lambda: unittest.main()))
