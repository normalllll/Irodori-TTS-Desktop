from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QCoreApplication, QEvent, QProcess, QThread, QThreadPool
from PySide6.QtNetwork import QLocalSocket
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
from qfluentwidgets import FluentIcon

from irodori_tts.desktop import asr
from irodori_tts.desktop.i18n import add_item, set_text
from irodori_tts.desktop.messages import BusinessError, t
from irodori_tts.desktop.resources import configure_environment
from irodori_tts.desktop.storage import (
    AppStore,
    create_project,
    load_project,
    read_json,
    write_json,
)
from irodori_tts.desktop.tasks import TaskManager
from irodori_tts.desktop.widgets import Expand, Table, apply_theme, label
from tests.base_fixtures import install_base
from tests.processes import managed_process
from tests.qt_cleanup import cleanup_qt, stop_manager, stub_connection_check


def setUpModule():
    unittest.addModuleCleanup(stub_connection_check().stop)


def delivered_recommendation(page, *, steps=950, batch=4, accumulation=2, deliver=True):
    """Simulate the asynchronous worker contract; this fixture isolates form delivery from planning."""
    from irodori_tts.desktop.materials import materials_fingerprint
    from irodori_tts.desktop.training import budget_schedule

    if page.root is None:
        root = create_project(page.host.store.root / "projects", "Recommendation fixture")
        page.host.store.register(root)
        page.set_project(root)
    settings = page.values()
    with patch.object(page.host.tasks, "start", return_value={"id": "measured"}) as start:
        with patch.object(page, "included_count", return_value=152):
            page.generate_recommendation()
    assert start.call_args.args[0]["kind"] == "recommend"
    assert page.values() == settings  # Nothing is filled before successful completion.
    result = {
        "values": {
            "max_steps": steps,
            "batch_size": batch,
            "gradient_accumulation_steps": accumulation,
            "gradient_checkpointing": False,
            **budget_schedule(steps),
        },
        "profile": {
            "count": 152,
            "total_seconds": 600,
            "median_seconds": 4,
            "p90_seconds": 7,
            "maximum_seconds": 10,
            "distinct_texts": 152,
        },
        "review_count": 0,
        "fingerprint": materials_fingerprint(page.project),
        "training_count": 152,
        "gpu": "Card",
        "memory_budget_bytes": 8 * 1024**3,
    }
    task = {"id": "measured", "kind": "recommend", "status": "completed", "result": result}
    if deliver:
        page.recommendation_finished(task)
    return task


class ProcessTests(unittest.TestCase):
    def tearDown(self):
        cleanup_qt()

    @classmethod
    def setUpClass(cls):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        cls.repo = Path(__file__).resolve().parents[1]
        configure_environment(cls.repo)
        cls.app = QApplication.instance() or QApplication([])

    def test_qprocess_real_resource_worker_and_duplicate_click(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = temporary.name
        manager = TaskManager(AppStore(Path(directory)), self.repo)
        self.addCleanup(stop_manager, manager)
        completed = QSignalSpy(manager.finished)
        task = manager.start({"kind": "resources"})
        duplicate = manager.start({"kind": "resources"})
        self.assertIs(task, duplicate)
        self.assertTrue(completed.wait(20000))
        self.assertEqual(task["status"], "completed", task)
        self.assertTrue(Path(task["result"]["codec"]).is_file())
        self.assertGreater(task["sequence"], 0)
        released = []
        manager.release(released.append)
        while not released:
            self.app.processEvents()
        self.assertEqual(released, [True])
        manager.timer.stop()
        manager.deleteLater()
        self.app.processEvents()

    def test_verify_reports_a_missing_model_file_instead_of_downloading_it(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        empty_cache = directory / "hf"
        manager = TaskManager(AppStore(directory / "data"), self.repo)
        self.addCleanup(stop_manager, manager)
        completed = QSignalSpy(manager.finished)
        with patch.dict(os.environ, {"HF_HOME": str(empty_cache), "HF_HUB_OFFLINE": "0"}):
            task = manager.start({"kind": "resources", "verify": True})
            self.assertTrue(completed.wait(30000))
        self.assertEqual(task["action"], "verify")
        self.assertEqual(task["status"], "failed", task)
        self.assertEqual(
            str(task["message"]),
            str(t("resources.verify_found_missing_file", file="weights.pth")),
        )
        self.assertEqual(list(empty_cache.rglob("*.safetensors")), [])
        released = []
        manager.release(released.append)
        while not released:
            self.app.processEvents()
        self.assertEqual(released, [True])
        manager.timer.stop()
        manager.deleteLater()
        self.app.processEvents()

    def test_process_owner_teardown_disconnects_callbacks(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = temporary.name
        manager = TaskManager(AppStore(Path(directory)), self.repo)
        self.addCleanup(stop_manager, manager)
        manager.start({"kind": "resources"})
        self.assertTrue(manager.process.waitForStarted(2000))
        errors = []
        with patch.object(sys, "excepthook", side_effect=lambda *error: errors.append(error)):
            process = manager.process
            stop_manager(manager)
            self.assertEqual(process.state(), process.ProcessState.NotRunning)
            manager.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.app.processEvents()
        self.assertEqual(errors, [])

    def test_fluent_rows_animate_insertion_removal_and_preserve_selection(self):
        apply_theme(self.app, "light")
        table = Table([("name", "名称", 200)])
        table.resize(400, 210)
        table.show()
        rows = [{"id": key, "name": key} for key in ("a", "b", "c")]
        table.update_rows(rows)
        table.row_animation.setCurrentTime(220)
        table.setCurrentIndex(table.model().index(1, 0))
        table.update_rows(rows[1:])
        table.row_animation.setCurrentTime(110)
        self.app.processEvents()
        self.assertTrue(1 < table.rowHeight(0) < 40)
        self.assertEqual(table.item()["id"], "b")
        table.row_animation.setCurrentTime(220)
        self.assertEqual([row["id"] for row in table.rows_model.rows], ["b", "c"])
        table.update_rows([rows[1], {"id": "d", "name": "d"}, rows[2]])
        table.row_animation.setCurrentTime(110)
        self.assertTrue(1 < table.rowHeight(1) < 40)
        table.row_animation.setCurrentTime(220)
        self.assertEqual(table.item()["id"], "b")
        self.app.setProperty("reduce_motion", True)
        table.update_rows(rows[1:])
        self.assertEqual([row["id"] for row in table.rows_model.rows], ["b", "c"])
        self.app.setProperty("reduce_motion", False)
        table.close()

    def test_expand_reverses_from_visible_height(self):
        apply_theme(self.app, "light")
        owner = QWidget()
        layout = QVBoxLayout(owner)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        for index in range(5):
            content_layout.addWidget(label(f"参数 {index}"))
        expanded = Expand("高级设置", content)
        layout.addWidget(expanded)
        owner.show()
        expanded.set_open(True)
        expanded.animation.setCurrentTime(110)
        self.app.processEvents()
        current = expanded.clip.height()
        self.assertGreater(current, 0)
        # The content keeps its final height while the viewport reveals it.
        self.assertGreater(content.height(), current)
        expanded.set_open(False)
        self.assertEqual(expanded.animation.startValue(), current)
        expanded.animation.setCurrentTime(220)
        self.assertFalse(content.isVisible())
        self.assertFalse(expanded.toggle.isChecked())
        # The header opens and closes on a click but never takes a checked look of its own,
        # which made it twitch when the pointer left it right after the click.
        expanded.toggle.click()
        self.assertTrue(expanded.toggle.isChecked())
        self.assertFalse(expanded.toggle.isCheckable())
        expanded.toggle.click()
        self.assertFalse(expanded.toggle.isChecked())
        owner.close()

    def test_async_import_keeps_requested_project_and_gui_thread(self):
        from irodori_tts.desktop.app import MainWindow

        with tempfile.TemporaryDirectory() as directory:
            store = AppStore(Path(directory))
            first = create_project(Path(directory) / "projects", "first")
            second = create_project(Path(directory) / "projects", "second")
            store.register(second)
            store.register(first)
            window = MainWindow(store)
            fixture = self.repo / "examples/desktop"
            with (
                patch(
                    "irodori_tts.desktop.training_page.QFileDialog.getExistingDirectory",
                    return_value=str(fixture),
                ),
                patch(
                    "irodori_tts.desktop.training_page.QFileDialog.getOpenFileName",
                    return_value=(str(fixture / "transcripts.csv"), ""),
                ),
            ):
                window.training.import_audio(True)
            window.training.set_project(second)
            self.assertTrue(QThreadPool.globalInstance().waitForDone(5000))
            self.app.processEvents()

            self.assertEqual(len(load_project(first)["materials"]), 2)
            self.assertEqual(len(load_project(second)["materials"]), 0)
            self.assertEqual(window.training.root, second)
            callback_threads = []
            window.run_io(
                lambda: 1,
                lambda value: callback_threads.append(QThread.currentThread() is self.app.thread()),
                None,
            )
            self.assertTrue(QThreadPool.globalInstance().waitForDone(5000))
            self.app.processEvents()
            self.assertEqual(callback_threads, [True])
            window.close()
            self.app.processEvents()

    def test_training_series_elapsed_and_remaining_time_come_from_task_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = TaskManager(AppStore(Path(directory)), self.repo)
            manager.timer.stop()
            run_root = Path(directory) / "projects" / "p" / "runs" / "r"
            for task_id, steps, created, ended in (
                ("b", (3, 4, 5), "2026-10-27T10:10:00+00:00", "2026-10-27T10:10:30+00:00"),
                ("a", (1, 2, 3), "2026-10-27T10:00:00+00:00", "2026-10-27T10:01:00+00:00"),
            ):
                task = {
                    "id": task_id,
                    "kind": "train",
                    "status": "completed",
                    "created": created,
                    "ended": ended,
                    "step": steps[-1],
                    "total": 10,
                    "run": str(run_root),
                    "message": "",
                }
                events = manager.task_dir(task)
                events.mkdir(parents=True)
                lines = [{"type": "progress", "step": s, "loss": 1 / s} for s in steps]
                lines.insert(1, {"type": "running"})
                (events / "events.jsonl").write_text(
                    "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
                )
                manager.tasks.append(task)
            self.assertEqual(manager.run_series(run_root), [(s, 1 / s) for s in (1, 2, 3, 4, 5)])
            self.assertEqual(manager.run_elapsed(run_root), 90.0)
            self.assertEqual(manager.run_series(run_root / "other"), [])
            live = {"id": "x", "status": "running", "step": 10, "total": 20}
            self.assertIsNone(manager.eta_seconds(live))
            manager.samples["x"] = deque([(0.0, 0), (10.0, 5), (20.0, 10)])
            self.assertEqual(manager.eta_seconds(live), 20.0)
            self.assertIsNone(manager.eta_seconds({**live, "status": "completed"}))
            manager.deleteLater()
            self.app.processEvents()

    def test_history_record_delete_removes_audio_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AppStore(Path(directory))
            record = store.root / "generations" / "g1"
            record.mkdir(parents=True)
            write_json(
                record / "generation.json",
                {"id": "g1", "created": "2026-10-27T10:00:00+00:00", "request": {"text": "a"}},
            )
            (record / "audio.wav").write_bytes(b"x")
            self.assertEqual([item["id"] for item in store.history()], ["g1"])
            store.delete_history("g1")
            self.assertFalse(record.exists())
            self.assertEqual(store.history(), [])
            with self.assertRaises(ValueError):
                store.delete_history("../settings")

    def test_muted_player_decodes_plays_seeks_and_keeps_same_source(self):
        # Multimedia must run in a real application loop, not QSignalSpy.wait's
        # nested loop. Every child has a hard deadline and verifies destruction.
        from tests.desktop_playback import run_probes

        results = run_probes(
            Path(os.environ.get("IRODORI_TEST_OUTPUT", str(self.repo / "outputs")))
            / "desktop-playback"
        )
        for result in results:
            self.assertEqual(result["exit_code"], 0, result)
            self.assertTrue(result["report"].get("passed"), result)

    def test_generation_complete_draft_survives_close_and_reopen(self):
        from irodori_tts.desktop.app import MainWindow

        with tempfile.TemporaryDirectory() as directory:
            store = AppStore(Path(directory))
            window = MainWindow(store)
            page = window.generation
            page.text.setPlainText("明日は晴れです。")
            page.caption.setText("落ち着いた声")
            page.trim.setChecked(False)
            page.seed.setValue(42)
            page.cfg_text.setValue(2.5)
            page.steps.setValue(20)
            page.advanced.set_open(True)
            expected = page.snapshot()
            window.close()
            self.app.processEvents()
            self.app.processEvents()
            reopened = MainWindow(AppStore(Path(directory)))
            self.assertEqual(reopened.generation.snapshot(), expected)
            self.assertFalse(reopened.generation.trim.isChecked())
            reopened.close()
            self.app.processEvents()
            self.app.processEvents()

    def test_selecting_voice_without_valid_preview_clears_player(self):
        from irodori_tts.desktop.app import MainWindow

        with tempfile.TemporaryDirectory() as directory:
            window = MainWindow(AppStore(Path(directory)))
            page = window.models.imported
            item = {
                "name": "有试听",
                "step": 4,
                "preview": str(self.repo / "examples/desktop/voice_design.wav"),
            }
            for preview in (None, str(Path(directory) / "missing.wav")):
                page.show_model(item)
                self.app.processEvents()
                self.assertFalse(page.player.player.source().isEmpty())
                page.show_model({**item, "name": "导入模型", "preview": preview})
                self.assertIsNone(page.player.path)
                self.assertFalse(page.player.isEnabled())
                self.app.processEvents()
                self.assertTrue(page.player.player.source().isEmpty())
                self.assertIn("导入模型", page.title.full_text)
            window.close()
            self.app.processEvents()
            self.app.processEvents()
            # The startup resource check writes into the data folder from a pool thread.
            self.assertTrue(QThreadPool.globalInstance().waitForDone(5000))

    def test_background_second_launch_uses_existing_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            # Keep only the background screenshot timer open long enough for a
            # second normal entry point to connect. No foreground path is used.
            code = """from unittest.mock import patch
from PySide6.QtCore import QTimer
from irodori_tts.desktop.app import main
original = QTimer.singleShot
with patch.object(QTimer, 'singleShot', side_effect=lambda delay, callback: original(7000 if delay == 800 else delay, callback)):
    raise SystemExit(main())
"""
            arguments = ["--background-check", "--data-dir", directory]
            with managed_process(
                [sys.executable, "-c", code, *arguments],
                cwd=self.repo,
                # An empty download cache: the launch does not depend on this machine's models.
                env={**os.environ, "HF_HOME": str(Path(directory) / "hf")},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            ) as first:
                server = (
                    "IrodoriDesktop-"
                    + hashlib.sha256(str(Path(directory).resolve()).encode()).hexdigest()[:20]
                )
                connected = False
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    socket = QLocalSocket()
                    socket.connectToServer(server)
                    if socket.waitForConnected(100):
                        socket.disconnectFromServer()
                        connected = True
                        break
                    time.sleep(0.05)
                self.assertTrue(connected, "First background instance did not create its server")
                second = subprocess.run(
                    [sys.executable, "desktop_app.py", *arguments],
                    cwd=self.repo,
                    capture_output=True,
                    timeout=6,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                self.assertEqual(second.returncode, 0, second.stderr.decode(errors="replace"))
                self.assertIsNone(
                    first.poll(), "Second launch must leave the first instance running"
                )
                self.assertEqual(first.wait(timeout=10), 0)
                self.assertTrue((Path(directory) / "screenshots/guide.png").is_file())


class ReviewFixTests(unittest.TestCase):
    """Window-level regressions: stale errors, new-project flow and model details."""

    @classmethod
    def setUpClass(cls):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        cls.repo = Path(__file__).resolve().parents[1]
        configure_environment(cls.repo)
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.windows = []
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = temporary.name

    def tearDown(self):
        for window in self.windows:
            window.close()
            self.app.processEvents()
            self.app.processEvents()
        cleanup_qt()

    def open_window(self, directory):
        from irodori_tts.desktop.app import MainWindow

        window = MainWindow(AppStore(Path(directory)))
        self.windows.append(window)
        return window

    def test_a_menu_entry_runs_after_the_click_that_chose_it(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        from irodori_tts.desktop.widgets import AcrylicMenu, show_menu

        anchor = QWidget()
        anchor.resize(200, 40)
        anchor.show()
        calls = []
        show_menu(anchor, [("Open", lambda: calls.append("ran"), True)])
        menu = anchor.findChildren(AcrylicMenu)[-1]
        view = menu.view
        item = next(view.item(i) for i in range(view.count()) if view.item(i).text() == "Open")
        QTest.mouseClick(
            view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            view.visualItemRect(item).center(),
        )
        # An entry that opens a modal dialog must not run inside the menu's own click: the
        # dialog's event loop deletes the closed menu while the click is still on its stack.
        self.assertEqual(calls, [])
        deadline = time.monotonic() + 5
        while not calls and time.monotonic() < deadline:
            self.app.processEvents()
        self.assertEqual(calls, ["ran"])

    def test_a_table_icon_click_is_announced_after_the_click(self):
        table = Table([("play", "", 44), ("name", "name", 100)])
        table.update_rows([{"id": "a", "name": "a", "icons": {"play": FluentIcon.PLAY}}])
        seen = []
        table.icon_clicked.connect(lambda row, key: seen.append(key))
        table.clicked.emit(table.rows_model.index(0, 0))
        self.assertEqual(seen, [])
        self.app.processEvents()
        self.assertEqual(seen, ["play"])

    def test_a_formatted_plural_count_renders_through_qt(self):
        from irodori_tts.desktop.i18n import locale_manager
        from irodori_tts.desktop.training_page import rough_duration

        locale_manager()
        # A count wrapped by fmt() reached QCoreApplication.translate and raised on every
        # training progress event, which stopped the step count at 1.
        self.assertIn("7", str(rough_duration(400)))
        self.assertIn("7", str(t("app.eta_left", duration=rough_duration(400))))

    def test_the_window_fits_the_screen_it_opens_on(self):
        from irodori_tts.desktop.app import MINIMUM_SIZE, PREFERRED_SIZE, window_size

        # A first start: the preferred size, at most 92% of the free area.
        self.assertEqual(window_size(None, 2560, 1392), PREFERRED_SIZE)
        self.assertEqual(window_size(None, 1280, 688), (1178, 640))
        # A saved size is kept, but never larger than the screen or below the minimum.
        self.assertEqual(window_size([1400, 900], 1920, 1032), (1400, 900))
        self.assertEqual(window_size([1900, 1000], 1280, 688), (1280, 688))
        self.assertEqual(window_size([400, 300], 1920, 1032), MINIMUM_SIZE)

    def test_saved_size_is_centered_on_every_launch_without_saving_position(self):
        from unittest.mock import Mock

        from PySide6.QtCore import QRect

        from irodori_tts.desktop.app import MainWindow

        area = QRect(1920, 40, 1920, 1040)
        store = AppStore(Path(self.directory))
        store.settings["ui"] = {"size": [1400, 900], "position": [-5000, 3000]}
        with patch(
            "irodori_tts.desktop.app.QApplication.primaryScreen",
            return_value=Mock(availableGeometry=lambda: area),
        ):
            window = MainWindow(store)
            self.windows.append(window)
        self.assertEqual((window.width(), window.height()), (1400, 900))
        self.assertEqual(window.pos(), area.center() - window.rect().center())
        window.move(10, 20)
        window.resize(1300, 800)
        window.save_ui()
        self.assertEqual(store.settings["ui"]["size"], [1300, 800])
        self.assertNotIn("position", store.settings["ui"])
        with patch(
            "irodori_tts.desktop.app.QApplication.primaryScreen",
            return_value=Mock(availableGeometry=lambda: area),
        ):
            reopened = MainWindow(AppStore(store.root))
            self.windows.append(reopened)
        self.assertEqual((reopened.width(), reopened.height()), (1300, 800))
        self.assertEqual(reopened.pos(), area.center() - reopened.rect().center())

    def test_first_guide_frame_has_themed_panel_and_distinct_card_fill(self):
        from PySide6.QtCore import QPoint
        from PySide6.QtTest import QTest
        from qfluentwidgets import SimpleCardWidget

        from irodori_tts.desktop.app import MainWindow

        for theme in ("light", "dark"):
            with self.subTest(theme=theme):
                store = AppStore(Path(self.directory) / theme)
                store.settings.update(
                    theme=theme, tour_done=True, reduce_motion=True, ui={"page": 0}
                )
                window = MainWindow(store)
                self.windows.append(window)
                panel = window.findChild(QWidget, "contentPanel")
                # The first application already has a registered background source;
                # no hover, theme toggle or second visit is needed to initialize it.
                self.assertIn("background-color:", panel.styleSheet())
                window.setMicaEffectEnabled(False)  # Offscreen has no OS backdrop compositor.
                window.show()
                QTest.qWait(250)
                self.assertIs(window.stack.currentWidget(), window.guide)
                card = window.guide.findChildren(SimpleCardWidget)[0]
                frame = window.grab().toImage()
                inside = frame.pixelColor(card.mapTo(window, QPoint(8, card.height() // 2)))
                outside = frame.pixelColor(card.mapTo(window, QPoint(-5, card.height() // 2)))
                self.assertGreater(inside.lightness(), outside.lightness())
                original = panel.styleSheet()
                store.settings["theme"] = "dark" if theme == "light" else "light"
                window.apply_theme()
                self.assertNotEqual(panel.styleSheet(), original)
                window.close()

    def test_the_navigation_folds_on_narrow_windows_and_unfolds_on_wide_ones(self):
        from PySide6.QtTest import QTest
        from qfluentwidgets import NavigationDisplayMode

        from irodori_tts.desktop.app import COMPACT_BELOW

        window = self.open_window(self.directory)
        window.show()
        panel = window.navigationInterface.panel
        window.resize(COMPACT_BELOW - 200, 720)
        QTest.qWait(500)
        self.assertEqual(panel.displayMode, NavigationDisplayMode.COMPACT)
        window.resize(COMPACT_BELOW + 100, 800)
        QTest.qWait(500)
        self.assertEqual(panel.displayMode, NavigationDisplayMode.EXPAND)

    def test_generation_error_clears_when_the_text_changes(self):
        directory = self.directory
        window = self.open_window(directory)
        page = window.generation
        page.text.setPlainText("")
        window.perform(page.generate, page.error)
        self.assertTrue(page.error.full_text)
        page.text.setPlainText("こんにちは。")
        self.assertEqual(page.error.full_text, "")

    def test_generation_shows_model_loading_then_steps_in_place_of_the_player(self):
        window = self.open_window(self.directory)
        page = window.generation
        loading = str(t("worker.loading_voice_model"))
        task = {"id": "g", "kind": "generate", "status": "running", "step": 0, "total": 0}
        task["message"] = t("worker.loading_voice_model")
        with (
            patch.object(type(window.tasks), "busy", True),
            patch.object(window.tasks, "current", task),
        ):
            window.task_changed(task)
            self.assertEqual(page.output_stack.currentIndex(), 1)
            self.assertFalse(page.waiting_bar.isHidden())
            self.assertTrue(page.progress_bar.isHidden())
            self.assertEqual(page.progress_text.full_text, loading)
            # The status bar says it as well, not only in a tooltip.
            self.assertEqual(window.task_detail.full_text, loading)
            task.update(step=12, total=40)
            window.task_changed(task)
            self.assertFalse(page.progress_bar.isHidden())
            self.assertEqual(page.progress_bar.value(), 12)
            self.assertIn("12", page.progress_text.full_text)
        task["status"] = "completed"
        window.task_changed(task)
        self.assertEqual(page.output_stack.currentIndex(), 0)

    def test_output_details_say_a_new_one_is_coming_and_return_if_it_does_not(self):
        window = self.open_window(self.directory)
        page = window.generation
        record = {
            "audio": "a.wav",
            "voice": "V",
            "seed": 1,
            "duration": 2.0,
            "request": {"num_steps": 40},
        }
        with patch.object(page.player, "load"):
            page.show_result(record)
        before = page.result_label.text()
        page.show_progress({"step": 0, "total": 0, "message": "…"})
        self.assertEqual(page.result_label.text(), str(t("generation_page.generating_new")))
        page.show_progress(None)
        self.assertEqual(page.result_label.text(), before)
        self.assertTrue(window.training.init_error.wordWrap())

    def test_a_finished_generation_plays_at_once(self):
        window = self.open_window(self.directory)
        result = {"audio": str(Path(self.directory) / "generations/x/audio.wav")}
        task = {"id": "g", "kind": "generate", "status": "completed", "result": result}
        with patch.object(window.generation, "show_result") as show:
            window.task_finished(task)
        show.assert_called_once_with(result, autoplay=True)

    def test_rows_play_and_pause_their_audio_from_an_icon(self):
        from PySide6.QtMultimedia import QMediaPlayer
        from qfluentwidgets import FluentIcon

        from irodori_tts.desktop.widgets import Player, RowPlayback

        window = self.open_window(self.directory)
        self.assertEqual(window.generation.history.rows_model.columns[0][0], "play")
        self.assertEqual(window.training.materials.rows_model.columns[0][0], "play")
        audio = str(Path(__file__).resolve().parents[1] / "examples/desktop/basic.wav")
        table = Table([("play", "", 44), ("name", "name", 100)])
        player = Player()
        player.player.audioOutput().setMuted(True)
        self.assertEqual(player.player.audioOutput().volume(), 1.0)
        playback = RowPlayback(table, player, lambda row: row["audio"])
        table.update_rows([{"id": "a", "name": "a", "audio": audio}])
        playback.sync()
        row = table.rows_model.rows[0]
        self.assertIs(row["icons"]["play"], FluentIcon.PLAY)
        spy = QSignalSpy(player.player.playbackStateChanged)
        table.icon_clicked.emit(row, "play")
        while player.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self.assertTrue(spy.wait(5000))
        self.assertIs(row["icons"]["play"], FluentIcon.PAUSE)
        table.icon_clicked.emit(row, "play")
        while player.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.assertTrue(spy.wait(5000))
        self.assertIs(row["icons"]["play"], FluentIcon.PLAY)
        player.stop()
        self.app.processEvents()

    def test_player_shows_the_waveform_and_seeks_from_it(self):
        from irodori_tts.desktop.widgets import Player

        audio = Path(__file__).resolve().parents[1] / "examples/desktop/basic.wav"
        player = Player()
        player.resize(600, 52)
        player.show()
        self.assertTrue(player.progressSlider.isHidden())
        self.assertIsNone(player.waveform.peaks)
        player.load(audio)
        # Decoded off the GUI thread; the waveform follows.
        self.assertTrue(QThreadPool.globalInstance().waitForDone(5000))
        self.app.processEvents()
        peaks = player.waveform.peaks
        self.assertGreater(len(peaks), 100)
        self.assertAlmostEqual(float(peaks.max()), 1.0)
        spy = QSignalSpy(player.player.durationChanged)
        while player.player.duration() <= 0:
            self.assertTrue(spy.wait(5000))
        with patch.object(player.player, "setPosition") as seek:
            player.waveform.seek.emit(0.5)
        seek.assert_called_once_with(player.player.duration() // 2)
        player._onPositionChanged(player.player.duration() // 4)
        self.assertAlmostEqual(player.waveform.position, 0.25, places=2)
        player.grab()  # paints without error
        player.clear()
        self.assertIsNone(player.waveform.peaks)
        player.stop()
        self.app.processEvents()

    def test_dialog_confirm_buttons_keep_white_text_on_the_accent(self):
        from qfluentwidgets.common.style_sheet import CustomStyleSheet

        from irodori_tts.desktop.asr_dialog import TranscribeDialog
        from irodori_tts.desktop.widgets import AccentRadio, TextPrompt

        window = self.open_window(self.directory)
        dialogs = [
            window.confirmation("title", "message", "OK"),
            TextPrompt(window, "title", "caption", "value"),
            TranscribeDialog(window, window.store.settings),
        ]
        for dialog in dialogs:
            self.assertIn("color: white", CustomStyleSheet(dialog.yesButton).darkStyleSheet())
        self.assertTrue(all(isinstance(r, AccentRadio) for r in dialogs[2].radios.values()))

    def test_model_file_tasks_are_named_for_what_they_do(self):
        window = self.open_window(self.directory)
        names = {
            action: window.kind_text({"id": action, "kind": "resources", "action": action})
            for action in ("download", "verify", "repair")
        }
        self.assertEqual(str(names["download"]), str(t("settings_page.prepare_resources")))
        self.assertEqual(str(names["verify"]), str(t("tasks.verify_resources")))
        self.assertEqual(str(names["repair"]), str(t("tasks.repair_resources")))
        # A task recorded before the action was stored is named from its request.
        old = {"id": "old", "kind": "resources"}
        write_json(
            window.tasks.task_dir(old) / "request.json", {"kind": "resources", "verify": True}
        )
        self.assertEqual(str(window.kind_text(old)), str(t("tasks.verify_resources")))

    def test_each_model_remembers_its_own_inference_settings(self):
        from irodori_tts.desktop.app import MainWindow

        voice = Path(self.directory) / "voices" / "hana"
        voice.mkdir(parents=True)
        write_json(voice / "voice.json", {"name": "Hana", "step": 1000, "base": {}})
        window = self.open_window(self.directory)
        page = window.generation
        self.assertIsNone(page.voice.currentData())
        page.steps.setValue(24)
        page.voice.setCurrentIndex(page.voice.findData("hana"))
        # A model with nothing saved starts from the defaults, not from the base model's edit.
        self.assertEqual(page.steps.value(), 40)
        page.duration_scale.setValue(0.85)
        set_text(page.caption, "落ち着いた声")
        page.voice.setCurrentIndex(page.voice.findData(None))
        self.assertEqual(page.steps.value(), 24)
        self.assertEqual(page.duration_scale.value(), 1.0)
        self.assertEqual(page.caption.text(), "")
        page.voice.setCurrentIndex(page.voice.findData("hana"))
        self.assertAlmostEqual(page.duration_scale.value(), 0.85)
        self.assertEqual(page.caption.text(), "落ち着いた声")
        # Each lives in the model's own folder; the base model's in the data folder.
        saved = read_json(voice / "inference_settings.json")
        self.assertAlmostEqual(saved["duration_scale"], 0.85)
        self.assertEqual(read_json(Path(self.directory) / "inference_settings.json")["steps"], 24)
        # The next start shows them again, without the window's own saved state.
        window.save_ui()
        window.close()
        self.app.processEvents()
        reopened = MainWindow(AppStore(Path(self.directory)))
        self.windows.append(reopened)
        generation = reopened.generation
        self.assertEqual(generation.voice.currentData(), "hana")
        self.assertAlmostEqual(generation.duration_scale.value(), 0.85)
        generation.voice.setCurrentIndex(generation.voice.findData(None))
        self.assertEqual(generation.steps.value(), 24)

    def test_damaged_inference_settings_fall_back_to_the_defaults(self):
        (Path(self.directory) / "inference_settings.json").write_text(
            '{"steps": "many", "seed": true, "duration_scale": 0.9, "references": 3}',
            encoding="utf-8",
        )
        page = self.open_window(self.directory).generation
        self.assertEqual(page.steps.value(), 40)
        self.assertEqual(page.seed.value(), -1)
        self.assertAlmostEqual(page.duration_scale.value(), 0.9)
        self.assertEqual(page.references, [])

    def test_a_new_data_folder_with_a_cached_base_opens_ready(self):
        from irodori_tts.desktop import model_files

        resources = Path(self.directory) / "resources"
        with (
            patch(
                "irodori_tts.desktop.base_models.adopt_cached_bases",
                side_effect=lambda root: [install_base(root)],
            ) as adopt,
            patch.object(model_files, "shared_ready", return_value=True),
        ):
            window = self.open_window(self.directory)
            # Before any event runs: the first frame already counts the cached base.
            adopt.assert_called_once_with(resources)
            self.assertIsNot(window.stack.currentWidget(), window.guide)
            self.assertTrue(window.generation.speech_ready)

    def test_without_the_speech_model_the_guide_opens_and_actions_wait(self):
        from irodori_tts.desktop import model_files

        speech = model_files.download_for(model_files.SPEECH)
        with (
            # No base comes from the machine's download cache either.
            patch("irodori_tts.desktop.base_models.adopt_cached_bases", return_value=[]),
            patch.object(model_files, "speech_ready", return_value=False),
            patch.object(
                model_files.Download,
                "installed",
                lambda item: item.key != model_files.SPEECH and item.backend is None,
            ),
            patch(
                "irodori_tts.desktop.guide_page.check_hugging_face",
                return_value={"reachable": True, "latency_ms": 1, "reason": None, "status": None},
            ),
        ):
            window = self.open_window(self.directory)
            self.app.processEvents()
            self.assertIs(window.stack.currentWidget(), window.guide)
            generation = window.generation
            self.assertFalse(generation.generate_button.isEnabled())
            self.assertFalse(generation.model_notice.isHidden())
            guide = window.guide
            fetch = guide.rows[model_files.SPEECH].action
            self.assertFalse(fetch.isHidden())
            # The main button fetches the required and the recommended models in one chain.
            with (
                patch.object(window.tasks, "start") as start,
                patch("irodori_tts.desktop.asr.missing_addons", return_value=[]),
            ):
                guide.download()
            payload = start.call_args.args[0]
            self.assertEqual(payload["kind"], "resources")
            self.assertEqual(payload["bases"], ["v4.1-small"])
            self.assertEqual(
                payload["then"],
                {"kind": "models", "backends": ["whisper-large-v3", "qwen3-asr"]},
            )
            # A row's own button fetches that model only.
            with (
                patch.object(window.tasks, "start") as start,
                patch("irodori_tts.desktop.asr.missing_addons", return_value=[]),
            ):
                guide.rows["anime-whisper"].action.click()
            self.assertEqual(
                start.call_args.args[0], {"kind": "models", "backends": ["anime-whisper"]}
            )
            self.assertEqual(guide.downloading, {"anime-whisper"})
            # The low video memory model is a download of its own, after its base.
            low_vram = model_files.variant_key("v4.1-small", model_files.LOW_VRAM)
            with (
                patch.object(window.tasks, "start") as start,
                patch.object(model_files.Download, "installed", return_value=False),
                patch("irodori_tts.desktop.asr.missing_addons", return_value=[]),
            ):
                guide.download([low_vram])
            self.assertEqual(
                start.call_args.args[0],
                {
                    "kind": "resources",
                    "bases": ["v4.1-small"],
                    "then": {"kind": "models", "backends": [low_vram]},
                },
            )
            # Each catalog base has a line of its own, and Large names its terms of use.
            large = guide.rows[model_files.base_key("v4-large")]
            self.assertIn("GEMMA_TERMS_OF_USE", large.detail.text())
            self.assertGreater(speech.size, 3 * 1000**3)
            # Getting started is a page of its own, restored like the others.
            window.save_ui()
            self.assertIs(window.pages[window.store.settings["ui"]["page"]], guide)

    def test_the_tour_dims_everything_but_the_control_it_explains(self):
        from PySide6.QtCore import QPoint
        from PySide6.QtTest import QTest

        window = self.open_window(self.directory)
        window.resize(1520, 920)
        window.show()
        window.start_tour()
        QTest.qWait(450)
        tour = window.tour
        target = window.guide.download_button
        centre = target.mapTo(window, target.rect().center())
        self.assertFalse(tour.spotlight.isHidden())
        self.assertIsNotNone(tour.tip)
        # The control is reachable through the hole; the rest of the window is not.
        self.assertFalse(tour.spotlight.mask().contains(centre))
        self.assertTrue(tour.spotlight.mask().contains(QPoint(5, window.height() - 5)))
        for _ in tour.steps:
            tour.next()
            QTest.qWait(350)
        self.assertFalse(tour.active)
        self.assertTrue(tour.spotlight.isHidden())
        self.assertTrue(window.store.settings["tour_done"])

    def test_a_project_taken_off_the_list_comes_back_from_its_folder(self):
        from irodori_tts.desktop.storage import create_project

        window = self.open_window(self.directory)
        parent = Path(self.directory) / "elsewhere"
        first = create_project(parent, "One")
        second = create_project(parent, "Two")
        listed = lambda: {root.resolve() for root, _project in window.store.projects()}  # noqa: E731
        # A project folder, or a folder of project folders.
        window.add_projects(first)
        self.assertIn(first.resolve(), listed())
        window.store.forget(first)
        window.add_projects(parent)
        self.assertTrue({first.resolve(), second.resolve()} <= listed())
        empty = Path(self.directory) / "empty"
        empty.mkdir()
        with self.assertRaises(BusinessError):
            window.add_projects(empty)

    def test_deleting_a_project_removes_its_folder(self):
        from irodori_tts.desktop.storage import create_project

        window = self.open_window(self.directory)
        root = create_project(window.store.project_root(), "Doomed")
        (root / "source" / "clip.wav").write_bytes(b"x" * 10)
        window.register_project(root)
        training = window.training

        def settle():
            for _pass in range(3):
                QThreadPool.globalInstance().waitForDone(5000)
                self.app.processEvents()

        with patch.object(window, "confirm", return_value=False):
            training.delete_project()
            settle()
        self.assertTrue(root.exists())
        with patch.object(window, "confirm", return_value=True):
            training.delete_project()
            settle()
        self.assertFalse(root.exists())
        self.assertNotIn(root.resolve(), {r.resolve() for r, _p in window.store.projects()})

    def test_the_guide_checks_the_connection_when_opened_from_the_navigation(self):
        window = self.open_window(self.directory)
        window.show()
        window.guide.connection_result = None
        window.navigate(0)
        QThreadPool.globalInstance().waitForDone(5000)
        self.app.processEvents()
        self.assertIsNotNone(window.guide.connection_result)

    def test_graphics_memory_is_judged_by_the_card_size(self):
        from irodori_tts.desktop import model_files

        self.assertEqual(model_files.card_memory_gb(7782), 8)
        self.assertEqual(model_files.card_memory_gb(16303), 16)
        window = self.open_window(self.directory)
        window.show()
        window.stack.setAnimationEnabled(False)
        window.show_guide()
        self.app.processEvents()
        guide = window.guide
        guide.device_known({"gpu": {"total_mib": 7782}})
        self.assertEqual(guide.recommended(), {"whisper-large-v3", "qwen3-asr"})
        # A 6 GB card cannot hold whisper-large-v3: the most accurate models that fit instead.
        with patch.object(model_files.Download, "installed", return_value=False):
            guide.device_known({"gpu": {"total_mib": 6144}})
            self.assertEqual(guide.recommended(), {"qwen3-asr", "anime-whisper"})
            self.assertIn("8 GB", guide.rows["whisper-large-v3"].value.text())

    def test_card_settings_start_empty_until_a_recommendation_fills_them(self):
        window = self.open_window(self.directory)
        training = window.training
        tuned = ("max_steps", "batch_size", "gradient_accumulation_steps")
        self.assertEqual([training.parameters[key].value() for key in tuned], [0, 0, 0])
        self.assertEqual([training.parameters[key].text() for key in tuned], ["", "", ""])
        self.assertIsNone(training.parameters["gradient_checkpointing"].value())
        self.assertIs(training.parameters["length_bucket_enabled"].value(), False)
        # Empty fields say so, and the line above them stands out and asks for the recommendation.
        placeholder = str(t("training_page.not_set_yet"))
        self.assertEqual(
            training.parameters["batch_size"].lineEdit().placeholderText(), placeholder
        )
        self.assertEqual(training.parameters["gradient_checkpointing"].text(), placeholder)
        self.assertTrue(training.recommendation.tinted)
        self.assertEqual(
            training.recommendation_note.text(), str(t("training_page.recommendation_pending"))
        )
        window.device = {"gpu": {"name": "Card", "total_mib": 7782, "compute_cap": 8.9}}
        delivered_recommendation(training)
        self.assertEqual([training.parameters[key].value() for key in tuned], [950, 4, 2])
        self.assertEqual(training.parameters["warmup_steps"].value(), 48)
        self.assertEqual(training.parameters["save_every"].value(), 190)
        self.assertIs(training.parameters["gradient_checkpointing"].value(), False)
        self.assertIn("Card", training.recommendation_note.text())
        self.assertFalse(training.recommendation_note.text().startswith("("))
        self.assertFalse(training.recommendation.tinted)
        self.assertEqual(
            training.parameters["length_bucket_enabled"].text(), str(t("training_page.option_off"))
        )
        # Changed by hand, the basis no longer describes the settings.
        training.parameters["batch_size"].setValue(2)
        self.assertEqual(
            training.recommendation_note.text(), str(t("training_page.settings_filled"))
        )
        training.parameters["batch_size"].setValue(4)
        self.assertIn("Card", training.recommendation_note.text())
        self.assertFalse(training.recommendation_note.text().startswith("("))
        self.assertEqual(training.snapshot()["settings"]["batch_size"], 4)
        training.preset.setCurrentIndex(0)
        training.apply_preset()
        self.assertEqual(training.parameters["batch_size"].text(), "")
        self.assertIsNone(training.parameters["gradient_checkpointing"].value())
        self.assertTrue(training.recommendation.tinted)

    def test_recommendation_does_not_overwrite_changed_settings_or_materials(self):
        training = self.open_window(self.directory).training
        task = delivered_recommendation(training, deliver=False)
        training.parameters["learning_rate"].setValue(0.0002)
        training.recommendation_finished(task)
        self.assertEqual(training.parameters["max_steps"].value(), 0)
        self.assertEqual(training.parameters["learning_rate"].value(), 0.0002)
        self.assertIn(
            str(t("training_page.recommendation_changed")), training.start_error.full_text
        )
        task = delivered_recommendation(training, deliver=False)
        training.project["materials"].append({"id": "added", "included": True, "text": "new"})
        training.recommendation_finished(task)
        self.assertEqual(training.parameters["max_steps"].value(), 0)
        self.assertFalse(training.recommendation_request)

    def test_recommendation_failure_and_project_switch_keep_drafts(self):
        training = self.open_window(self.directory).training
        task = delivered_recommendation(training, deliver=False)
        original = training.values()
        training.recommendation_finished({**task, "status": "failed", "message": "probe failed"})
        self.assertEqual(training.values(), original)
        self.assertEqual(training.start_error.full_text, "probe failed")
        task = delivered_recommendation(training, deliver=False)
        other = create_project(training.host.store.root / "projects", "Other")
        training.set_project(other)
        training.parameters["max_steps"].setValue(222)
        training.recommendation_finished(task)
        self.assertEqual(training.parameters["max_steps"].value(), 222)

    def test_training_schedule_follows_budget_and_preserves_manual_values(self):
        window = self.open_window(self.directory)
        training = window.training
        window.device = {"gpu": {"name": "Card", "total_mib": 16384, "compute_cap": 8.9}}
        delivered_recommendation(training, steps=13)
        parameters = training.parameters
        self.assertEqual(parameters["max_steps"].value(), 13)
        self.assertEqual(parameters["warmup_steps"].value(), 1)
        self.assertEqual(parameters["save_every"].value(), 3)

        parameters["max_steps"].setValue(950)
        self.assertEqual(parameters["warmup_steps"].value(), 48)
        self.assertEqual(parameters["save_every"].value(), 190)
        parameters["warmup_steps"].setValue(0)
        parameters["save_every"].setValue(17)
        parameters["max_steps"].setValue(400)
        self.assertEqual(parameters["warmup_steps"].value(), 0)
        self.assertEqual(parameters["save_every"].value(), 17)

        # Loading a preset preserves its explicit schedule, including disabled warmup.
        training.preset.addItem(
            "custom", userData={"max_steps": 123, "warmup_steps": 0, "save_every": 9}
        )
        training.preset.setCurrentIndex(training.preset.count() - 1)
        self.assertEqual(parameters["warmup_steps"].value(), 0)
        self.assertEqual(parameters["save_every"].value(), 9)

        # An explicit recommendation resets the schedule; editing it clears its basis.
        delivered_recommendation(training)
        self.assertIn("Card", training.recommendation_note.text())
        parameters["warmup_steps"].setValue(0)
        self.assertEqual(
            training.recommendation_note.text(), str(t("training_page.settings_filled"))
        )
        parameters["warmup_steps"].setValue(48)
        self.assertIn("Card", training.recommendation_note.text())
        parameters["save_every"].setValue(17)
        self.assertEqual(
            training.recommendation_note.text(), str(t("training_page.settings_filled"))
        )
        # The largest supported budget must not silently clamp the save interval.
        parameters["save_every"].setValue(190)
        parameters["max_steps"].setValue(1_000_000)
        self.assertEqual(parameters["warmup_steps"].value(), 50_000)
        self.assertEqual(parameters["save_every"].value(), 200_000)

    def test_a_long_confirmation_wraps_at_a_readable_width(self):
        window = self.open_window(self.directory)
        window.resize(1520, 920)
        box = window.confirmation("Title", "word " * 200, "OK")
        lines = box.contentLabel.text().splitlines()
        self.assertGreater(len(lines), 10)
        self.assertTrue(all(len(line) <= 64 for line in lines))
        self.assertLess(box.widget.width(), 700)

    def test_building_the_window_shows_no_other_window(self):
        from PySide6.QtCore import QEvent, QObject

        shown = []

        class Windows(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.Show and obj.isWidgetType() and obj.isWindow():
                    shown.append(type(obj).__name__)
                return False

        spy = Windows()
        self.app.installEventFilter(spy)
        try:
            self.open_window(self.directory)
            self.app.processEvents()
        finally:
            self.app.removeEventFilter(spy)
        # Waveforms shown before their player was laid out used to flash as windows.
        self.assertEqual(shown, [])

    def test_the_tour_ring_stays_inside_the_window(self):
        from irodori_tts.desktop.tour import RING, Spotlight

        window = self.open_window(self.directory)
        window.resize(1520, 920)
        window.show()
        self.app.processEvents()
        spotlight = Spotlight(window)
        spotlight.focus(window.navigationInterface.widget("page_0"))
        self.assertGreaterEqual(spotlight.hole.left(), RING)
        self.assertGreaterEqual(spotlight.hole.top(), RING)

    def test_cleared_error_does_not_return_after_a_language_switch(self):
        from irodori_tts.desktop.i18n import locale_manager, set_text
        from irodori_tts.desktop.messages import t
        from irodori_tts.desktop.widgets import error_label

        locale = locale_manager()
        original = locale.language
        self.addCleanup(locale.set_language, original)
        widget = error_label()
        set_text(widget, t("tasks.stopped"))
        widget.clear()
        locale.set_language("en" if original == "zh_CN" else "zh_CN")
        self.assertEqual(widget.full_text, "")

    def test_new_project_is_created_in_the_default_location_without_a_folder_dialog(self):
        directory = self.directory
        window = self.open_window(directory)
        with (
            patch("irodori_tts.desktop.app.prompt_text", return_value=("新项目", True)),
            patch(
                "irodori_tts.desktop.app.QFileDialog.getExistingDirectory",
                side_effect=AssertionError("no folder dialog"),
            ),
        ):
            window.new_project()
        (root, project), *rest = list(window.store.projects())
        self.assertEqual(rest, [])
        self.assertEqual(project["name"], "新项目")
        self.assertEqual(root.parent, window.store.project_root().resolve())
        self.assertEqual(window.training.root, root)
        self.assertIs(window.stack.currentWidget(), window.models)

    def test_model_details_follow_rename_and_fill_back_keeps_the_voice(self):
        directory = self.directory
        voice = Path(directory) / "voices" / "v1"
        voice.mkdir(parents=True)
        write_json(voice / "voice.json", {"name": "Before", "step": 3, "base": {}})
        window = self.open_window(directory)
        page = window.models.imported
        window.models.list.select_id("model:v1")
        self.assertIs(window.models.stack.currentWidget(), page)
        self.assertEqual(page.title.full_text, "Before")
        with patch("irodori_tts.desktop.library_page.prompt_text", return_value=("After", True)):
            page.rename()
        self.assertEqual(page.title.full_text, "After")
        generation = window.generation
        record = {
            "voice": "After",
            "request": {"text": "こんにちは。", "lora_adapter": str(voice)},
        }
        generation.fill_back(record)
        self.assertEqual(generation.voice.currentData(), "v1")

    def test_fill_back_restores_the_seed_that_was_used(self):
        generation = self.open_window(self.directory).generation
        self.assertEqual(generation.seed.value(), -1)
        # -1 is shown as a number, and a typed -1 means random like the button.
        self.assertEqual(generation.seed.text(), "-1")
        generation.seed.setValue(5)
        generation.seed.lineEdit().setText("-1")
        generation.seed.interpretText()
        self.assertEqual(generation.seed.value(), -1)
        generation.fill_back({"request": {"text": "a", "seed": None}, "seed": 1234})
        self.assertEqual(generation.seed.value(), 1234)
        generation.fill_back({"request": {"text": "a", "seed": 7}})
        self.assertEqual(generation.seed.value(), 7)
        generation.fill_back({"request": {"text": "a", "seed": None}})
        self.assertEqual(generation.seed.value(), -1)


class TrainingPageTests(unittest.TestCase):
    """Dataset editing rules of the training page, without compute work."""

    @classmethod
    def setUpClass(cls):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        cls.repo = Path(__file__).resolve().parents[1]
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from irodori_tts.desktop.app import MainWindow
        from irodori_tts.desktop.materials import import_materials

        self.temp = tempfile.TemporaryDirectory()
        self.store = AppStore(Path(self.temp.name))
        self.window = MainWindow(self.store)
        self.page = self.window.training
        models = self.window.models
        self.empty_state = models.stack.currentWidget() is models.empty
        self.root = create_project(self.store.root / "projects", "page")
        fixture = self.repo / "examples/desktop"
        import_materials(self.root, fixture, fixture / "transcripts.csv")
        self.store.register(self.root)
        self.window.refresh()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        cleanup_qt()
        self.temp.cleanup()

    def select(self, position):
        self.page.materials.setCurrentIndex(self.page.materials.model().index(position, 0))
        self.app.processEvents()

    def test_more_import_ways_open_their_menu(self):
        from PySide6.QtCore import QEventLoop, Qt, QTimer
        from PySide6.QtTest import QTest

        self.window.show()
        self.window.switchTo(self.window.models)
        self.page.tabs.setCurrentIndex(0)
        button = self.page.more_imports
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        loop = QEventLoop()
        QTimer.singleShot(300, loop.quit)
        loop.exec()
        menu = button.menu()
        self.assertTrue(menu.isVisible())
        self.assertIn(
            str(t("training_page.import_audio_and_csv")),
            [action.text() for action in menu.menuActions()],
        )
        menu.close()

    def test_empty_state_then_known_problems_block_initialization_before_any_launch(self):
        from irodori_tts.desktop.messages import BusinessError

        # Without a model the page says how to get one; with one it shows its steps.
        self.assertTrue(self.empty_state)
        self.assertIs(self.window.models.stack.currentWidget(), self.page)
        project = load_project(self.root)
        project["materials"][0]["text"] = ""
        write_json(self.root / "project.json", project)
        self.page.set_project(self.root)
        with patch.object(self.window.tasks, "start") as start:
            with self.assertRaises(BusinessError) as raised:
                self.page.initialize()
            start.assert_not_called()
        self.assertIn(project["materials"][0]["file_name"], str(raised.exception))

    def test_switching_recordings_saves_the_draft_and_refused_toggle_shows_saved_state(self):
        from PySide6.QtCore import Qt

        self.select(0)
        first = self.page.material_id
        self.page.material_text.setPlainText("新しい文です。")
        self.select(1)
        self.app.processEvents()
        saved = {item["id"]: item for item in load_project(self.root)["materials"]}
        self.assertEqual(saved[first]["text"], "新しい文です。")
        task = {"id": "t", "kind": "train", "project": str(self.root), "status": "running"}
        with (
            patch.object(self.window.tasks, "current", task),
            patch.object(type(self.window.tasks), "busy", True),
        ):
            model = self.page.materials.rows_model
            column = next(i for i, c in enumerate(model.columns) if c[0] == "file_name")
            model.setData(model.index(0, column), 0, Qt.ItemDataRole.CheckStateRole)
        self.assertTrue(load_project(self.root)["materials"][0]["included"])
        self.assertTrue(model.rows[0]["included"])
        self.assertTrue(self.page.material_error.full_text)

    def test_saving_a_preset_under_an_existing_name_replaces_it(self):
        self.page.parameters["max_steps"].setValue(300)
        with (
            patch("irodori_tts.desktop.training_page.prompt_text", return_value=("Mine", True)),
            patch.object(self.window, "confirm", return_value=True) as confirm,
        ):
            self.page.save_preset()
            count = self.page.preset.count()
            self.page.parameters["max_steps"].setValue(400)
            self.page.save_preset()
        confirm.assert_called_once()
        self.assertEqual(self.page.preset.count(), count)
        self.assertEqual(self.store.settings["presets"]["Mine"]["max_steps"], 400)
        self.assertEqual(self.page.preset.itemData(count - 1)["max_steps"], 400)


class ModelsAndSetupTests(unittest.TestCase):
    """The models page, how preprocessing stands and fails, and Getting started's checklist."""

    @classmethod
    def setUpClass(cls):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        cls.repo = Path(__file__).resolve().parents[1]
        configure_environment(cls.repo)
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from irodori_tts.desktop.app import MainWindow
        from irodori_tts.desktop.materials import import_materials

        self.temp = tempfile.TemporaryDirectory()
        self.store = AppStore(Path(self.temp.name))
        self.root = create_project(self.store.root / "projects", "first")
        fixture = self.repo / "examples/desktop"
        import_materials(self.root, fixture, fixture / "transcripts.csv")
        self.store.register(self.root)
        voice = self.store.root / "voices" / "v1"
        voice.mkdir(parents=True)
        write_json(voice / "voice.json", {"name": "Imported", "step": 7, "base": {}})
        self.window = MainWindow(self.store)
        self.page = self.window.training

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        cleanup_qt()
        self.temp.cleanup()

    def edit_project(self, change):
        project = load_project(self.root)
        change(project)
        write_json(self.root / "project.json", project)
        self.page.set_project(self.root)
        return project

    def test_start_training_selects_record_id_in_a_human_named_folder(self):
        from irodori_tts.desktop.training import RECOMMENDED

        page = self.page
        values = {**RECOMMENDED, "max_steps": 40, "batch_size": 1, "gradient_accumulation_steps": 2}
        resources = self.window.resource_root
        self.addCleanup(shutil.rmtree, resources / "bases", True)
        self.addCleanup((resources / "base-models.json").unlink, True)
        base = install_base(resources)
        page.refresh_bases()
        with (
            patch(
                "irodori_tts.desktop.training.validate_initialization",
                return_value={"id": "init", "revision": 1, "count": 2},
            ),
            patch("irodori_tts.desktop.training.training_config", return_value={"train": {}}),
            patch.object(self.window.tasks, "start") as start,
        ):
            for name in ("My new training", "My next training"):
                page.new_run()
                page.ready = True
                page.voice_name.setText(name)
                for key, widget in page.parameters.items():
                    widget.setValue(values[key])
                page.start_training()
                self.assertTrue(QThreadPool.globalInstance().waitForDone(5000))
                self.app.processEvents()
                run = Path(start.call_args.args[0]["run"])
                record = read_json(run / "run.json")
                self.assertNotEqual(run.name, record["id"])
                self.assertEqual(page.runs.item()["id"], record["id"])
                self.assertEqual(page.run_stack.currentIndex(), 1)
                # The run keeps the base it was created on, and its title names it.
                self.assertEqual(record["base"]["id"], base.id)
                self.assertEqual(page.run_title.full_text, f"{name} · {base.name}")
                self.app.processEvents()
                self.assertEqual(page.runs.item()["id"], record["id"])
                page.refresh()
                self.assertEqual(page.run_stack.currentIndex(), 1)
        # Reopening an active run also resolves its record ID from its path, not its name.
        page.new_run()
        other = create_project(self.store.root / "projects", "Other")
        self.store.register(other)
        self.window.refresh()
        self.window.models.select_project(other)
        self.window.tasks.current = {
            "id": "active",
            "kind": "train",
            "status": "running",
            "project": str(self.root),
            "run": str(run),
            "step": 0,
            "total": 40,
        }
        try:
            self.window.models.select_project(self.root)
            self.assertEqual(page.runs.item()["id"], record["id"])
            self.assertEqual(page.run_stack.currentIndex(), 1)
        finally:
            self.window.tasks.current = None

    def test_existing_run_opens_loss_and_sample_panel_and_preserves_explicit_new_draft(self):
        other = create_project(self.store.root / "projects", "With history")
        run_root = other / "runs" / "history"
        write_json(
            run_root / "run.json",
            {
                "id": "history",
                "name": "History",
                "created": "2026-10-30T10:00:00+00:00",
                "status": "completed",
                "step": 10,
                "checkpoint": None,
                "config": {"train": {"max_steps": 10, "save_every": 5, "learning_rate": 1e-4}},
            },
        )
        self.store.register(other)
        self.window.refresh()
        self.window.models.select_project(other)
        self.app.processEvents()
        self.assertEqual(self.page.runs.item()["id"], "history")
        self.assertEqual(self.page.run_stack.currentIndex(), 1)
        self.assertFalse(self.page.loss_chart.isHidden())
        self.assertFalse(self.page.sample_stack.isHidden())
        self.page.refresh()
        self.assertEqual(self.page.run_stack.currentIndex(), 1)
        self.page.new_run()
        self.page.parameters["max_steps"].setValue(123)
        self.window.models.select_project(self.root)
        self.window.models.select_project(other)
        self.app.processEvents()
        self.assertIsNone(self.page.runs.item())
        self.assertEqual(self.page.run_stack.currentIndex(), 0)
        self.assertEqual(self.page.parameters["max_steps"].value(), 123)
        self.page.select_run_id("history")
        self.window.models.select_project(self.root)
        self.window.models.select_project(other)
        self.app.processEvents()
        self.assertEqual(self.page.run_stack.currentIndex(), 1)
        self.assertEqual(self.page.runs.item()["id"], "history")

    def test_the_models_page_lists_trained_and_imported_models(self):
        from PySide6.QtTest import QTest

        models = self.window.models
        kinds = [row.get("kind", "section") for row in models.list.rows_model.rows]
        self.assertEqual(kinds, ["section", "project", "section", "model"])
        self.assertIs(models.stack.currentWidget(), self.page)
        self.assertEqual(self.page.root, self.root.resolve())
        models.list.select_id("model:v1")
        self.assertIs(models.stack.currentWidget(), models.imported)
        self.assertEqual(models.imported.title.full_text, "Imported")
        # A narrow window hides the list; a button by the title opens it as a menu.
        self.window.stack.setAnimationEnabled(False)
        self.window.show()
        self.window.navigate(2)
        self.window.resize(1024, 640)
        QTest.qWait(100)
        self.assertTrue(models.side.isHidden())
        self.assertFalse(models.imported.switch_button.isHidden())
        self.window.resize(1600, 900)
        QTest.qWait(100)
        self.assertFalse(models.side.isHidden())
        self.assertTrue(models.imported.switch_button.isHidden())

    def test_every_run_with_a_complete_checkpoint_is_a_model(self):
        from irodori_tts.desktop.base_models import base_record

        base = install_base(self.window.resource_root)
        self.addCleanup(shutil.rmtree, self.window.resource_root / "bases", True)
        self.addCleanup((self.window.resource_root / "base-models.json").unlink, True)
        run_root = self.root / "runs" / "r1"
        checkpoint = run_root / "checkpoints" / "checkpoint_0000100"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text("{}", "utf-8")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"x")
        write_json(
            checkpoint / "complete.json",
            {"step": 100, "files": {"adapter_config.json": 2, "adapter_model.safetensors": 1}},
        )
        write_json(
            run_root / "run.json",
            {
                "id": "r1",
                "name": "Run 1",
                "created": "2026-10-29T10:00:00+00:00",
                "status": "stopped",
                "step": 100,
                "checkpoint": "runs/r1/checkpoints/checkpoint_0000100",
                # Nothing needs adding to a list: a run with a checkpoint is a model.
                "registered": False,
                "base": base_record(base),
                "config": {"train": {"max_steps": 1000, "save_every": 100, "learning_rate": 1e-4}},
            },
        )
        entry = self.window.voice_entries()["r1"]
        self.assertEqual((entry["base_id"], entry["base_status"]), ("v4.1-small", "exact"))
        self.window.refresh()
        self.page.select_run_id("r1")
        self.assertFalse(self.page.use_button.isHidden())
        with patch.object(self.window, "navigate") as navigate:
            self.page.use_run()
        self.assertEqual(self.window.generation.voice.currentData(), "r1")
        navigate.assert_called_once_with(1)

    def test_a_failed_preprocessing_says_where_and_offers_the_ways_on(self):
        failed = {}

        def fail(project):
            failed.update(project["materials"][1])
            project["last_attempt"] = {
                "status": "failed",
                "error": "Encoding x.wav failed: boom",
                "phase": "encode",
                "item": failed["id"],
                "items": [],
                "created": "2026-10-29T10:00:00+00:00",
            }

        self.edit_project(fail)
        page = self.page
        self.assertEqual(page.init_title.full_text, str(t("training_page.preprocessing_failed")))
        self.assertFalse(page.failure.isHidden())
        self.assertEqual(page.init_error.text(), "Encoding x.wav failed: boom")
        states = [step.state for step in (page.step_check, page.step_encode, page.step_write)]
        self.assertEqual(states, ["done", "failed", "pending"])
        self.assertFalse(page.failure_exclude.isHidden())
        page.exclude_failed()
        saved = {item["id"]: item for item in load_project(self.root)["materials"]}
        self.assertFalse(saved[failed["id"]]["included"])

    def test_known_problems_are_all_listed_and_can_be_left_out_together(self):
        from irodori_tts.desktop.materials import MaterialProblems, validate_materials

        def blank(project):
            for item in project["materials"]:
                item["text"] = ""

        project = self.edit_project(blank)
        count = len(project["materials"])
        self.assertEqual(len(self.page.problem_items()), count)
        # One line per recording, beside the steps, not one problem per attempt.
        self.assertEqual(self.page.issue_list.count() - 1, count)
        with self.assertRaises(MaterialProblems) as raised:
            validate_materials(self.root, project)
        self.assertEqual(len(raised.exception.items), count)
        self.assertEqual(raised.exception.phase, "check")
        with patch.object(self.window.tasks, "start") as start:
            with self.assertRaises(BusinessError):
                self.page.initialize()
            start.assert_not_called()
        self.page.exclude_problems()
        self.assertFalse(any(item["included"] for item in load_project(self.root)["materials"]))
        self.assertEqual(self.page.problem_items(), [])

    def test_starting_training_preprocesses_first_and_then_trains(self):
        page = self.page
        self.assertEqual(page.start_button.text(), str(t("training_page.preprocess_and_train")))
        page.voice_name.setText("Run A")
        with patch.object(self.window.tasks, "start") as start:
            # The card's settings are empty until the recommendation fills them.
            with self.assertRaises(BusinessError):
                page.start_training()
            start.assert_not_called()
            delivered_recommendation(page, batch=2, accumulation=1)
            page.start_training()
        payload = start.call_args.args[0]
        self.assertEqual(payload["kind"], "initialize")
        self.assertEqual(payload["train"]["name"], "Run A")
        task = {"id": "i1", "kind": "initialize", "status": "completed", "project": str(self.root)}
        directory = self.window.tasks.task_dir(task)
        directory.mkdir(parents=True)
        write_json(directory / "request.json", payload)
        with patch.object(page, "create_and_train") as create:
            page.continue_training(task)
        create.assert_called_once_with(
            "Run A", payload["train"]["values"], payload["train"].get("base")
        )
        with patch.object(page, "create_and_train") as create:
            page.continue_training({**task, "status": "failed"})
        create.assert_not_called()

    def test_getting_started_shows_each_download_and_why_one_failed(self):
        from irodori_tts.desktop import model_files

        guide = self.window.guide
        with patch.object(model_files.Download, "installed", lambda item: False):
            guide.refresh(force=True)
            self.assertFalse(guide.complete())
            task = {
                "id": "d",
                "kind": "models",
                "status": "running",
                "step": 500,
                "total": 1000,
                "item": "anime-whisper",
                "message": "…",
            }
            tasks = self.window.tasks
            with patch.object(type(tasks), "busy", True), patch.object(tasks, "current", task):
                guide.downloading = {"anime-whisper", "qwen3-asr"}
                guide.task_changed(task)
                row = guide.rows["anime-whisper"]
                self.assertEqual(row.state, "active")
                self.assertFalse(row.bar.isHidden())
                self.assertEqual(row.bar.value(), 500)
                self.assertFalse(guide.pause_button.isHidden())
            guide.task_finished({**task, "status": "failed", "message": "connection reset"})
            row = guide.rows["anime-whisper"]
            self.assertEqual(row.state, "failed")
            self.assertEqual(row.problem.text(), "connection reset")
            self.assertEqual(row.action.text(), str(t("guide.retry")))
            self.assertTrue(row.bar.isHidden())

    def test_the_final_checkpoint_and_its_twin_are_one_row(self):
        final, twin = Path("c/checkpoint_final"), Path("c/checkpoint_0001000")
        earlier = Path("c/checkpoint_0000750")
        entries = [(1000, final), (1000, twin), (750, earlier)]
        self.assertEqual(self.page.one_per_step(entries, {}), [(1000, final), (750, earlier)])
        # The copy that has a sample stays, so its sample keeps playing.
        self.assertEqual(
            self.page.one_per_step(entries, {str(twin): Path("a.wav")}),
            [(1000, twin), (750, earlier)],
        )

    def test_the_theme_colour_is_chosen_in_settings_and_restyles_open_buttons(self):
        from irodori_tts.desktop.widgets import ACCENT, ACCENTS, button, current_accent

        primary = button("x", lambda: None, primary=True)
        settings = self.window.settings_page
        self.assertEqual(current_accent(), "aqua")
        settings.accent.setCurrentIndex(settings.accent.findData("pink"))
        try:
            self.assertEqual(self.store.settings["accent"], "pink")
            self.assertEqual(ACCENT.name(), ACCENTS["pink"]["color"])
            # The button made before the change is restyled with the new fill and white text.
            self.assertIn(ACCENTS["pink"]["color"], primary.styleSheet())
            self.assertIn("color: white", primary.styleSheet())
            # The current step's marker is a picture drawn with the accent; it is redrawn.
            current = self.page.tabs.currentIndex()
            marker = self.page.pivot.widget(str(current)).icon().pixmap(32, 32).toImage()
            edge = marker.pixelColor(marker.width() // 2, 2)
            self.assertLess(abs(edge.red() - ACCENT.red()) + abs(edge.blue() - ACCENT.blue()), 60)
        finally:
            settings.accent.setCurrentIndex(settings.accent.findData("aqua"))
        self.assertEqual(ACCENT.name(), ACCENTS["aqua"]["color"])
        primary.deleteLater()

    def test_checkpoints_are_picked_on_the_loss_curve(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QSignalSpy, QTest

        from irodori_tts.desktop.training_page import LossChart

        chart = LossChart()
        chart.resize(600, 160)
        chart.set_points([(step, 1.0 - step / 2000) for step in range(1, 1001)], 1000)
        chart.set_checkpoints([(500, "b"), (250, "a"), (1000, "c")], "c")
        self.assertEqual(chart.checkpoints, [(250, "a"), (500, "b"), (1000, "c")])
        frame = chart.frame()
        # Anywhere near a point picks it; the loss shown is the curve's at that step.
        x = chart.marker(frame, 500).x() + 20
        self.assertEqual(chart.nearest(x), (500, "b"))
        self.assertAlmostEqual(chart.loss_at(1000), chart.smoothed()[-1][1])
        spy = QSignalSpy(chart.chosen)
        QTest.keyClick(chart, Qt.Key.Key_Left)
        self.assertEqual((chart.selected, spy.count()), ("b", 1))
        QTest.keyClick(chart, Qt.Key.Key_Left)
        QTest.keyClick(chart, Qt.Key.Key_Left)
        self.assertEqual(chart.selected, "a")
        # Press, drag across the points, let go: the pick follows, and is announced once.
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QMouseEvent

        def send(kind, step, buttons):
            point = QPointF(chart.marker(chart.frame(), step).x(), 80)
            button = (
                Qt.MouseButton.NoButton
                if kind == QEvent.Type.MouseMove
                else Qt.MouseButton.LeftButton
            )
            chart.event(
                QMouseEvent(
                    kind,
                    point,
                    chart.mapToGlobal(point),
                    button,
                    buttons,
                    Qt.KeyboardModifier.NoModifier,
                )
            )

        left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
        send(QEvent.Type.MouseButtonPress, 250, left)
        send(QEvent.Type.MouseMove, 500, left)
        send(QEvent.Type.MouseMove, 1000, left)
        self.assertEqual((chart.selected, spy.count()), ("c", 2))
        send(QEvent.Type.MouseButtonRelease, 1000, none)
        self.assertEqual((chart.selected, spy.count(), spy.at(2)), ("c", 3, ["c"]))
        # A pick that is gone falls back to none; the page then takes the newest.
        chart.set_checkpoints([(750, "d")], "a")
        self.assertIsNone(chart.selected)
        chart.deleteLater()

    def test_dialogs_stay_inside_the_window_and_hold_the_keyboard(self):
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtTest import QTest

        self.window.show()
        seen = {}

        def answer():
            box = self.window.findChildren(QWidget, "centerWidget")[-1].parentWidget()
            seen["window"] = box.isWindow()
            seen["main_active"] = self.window.isActiveWindow() or not box.isWindow()
            # Keys aimed at the page under the layer never reach it.
            seen["blocked"] = box._in_window.eventFilter(
                self.window.generation.text,
                QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier),
            )
            QTest.keyClick(box.cancelButton, Qt.Key.Key_Escape)

        QTimer.singleShot(0, answer)
        self.assertFalse(self.window.confirm("t", "m", "ok"))
        self.assertEqual(seen, {"window": False, "main_active": True, "blocked": True})

    def test_row_icons_are_buttons_and_history_text_can_be_copied(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        from irodori_tts.desktop.widgets import ACTION_ROLE

        page = self.window.generation
        record = self.store.root / "generations" / "h1"
        record.mkdir(parents=True)
        (record / "audio.wav").write_bytes(b"")
        write_json(
            record / "generation.json",
            {"duration": 1.0, "created": "2026-10-29T00:00:00", "request": {"text": "こんにちは"}},
        )
        page.refresh_history()
        model = page.history.model()
        keys = [column[0] for column in model.columns]
        copy = model.index(0, keys.index("copy"))
        self.assertTrue(copy.data(ACTION_ROLE))
        self.assertFalse(model.index(0, keys.index("text")).data(ACTION_ROLE))
        self.window.show()
        self.window.navigate(1)
        self.app.processEvents()
        rect = page.history.visualRect(copy)
        point = QPointF(rect.center())
        move = QMouseEvent(
            QEvent.Type.MouseMove,
            point,
            page.history.viewport().mapToGlobal(point),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(page.history.viewport(), move)
        self.assertEqual(page.history.itemDelegate().hover_cell, (0, keys.index("copy")))
        self.assertEqual(
            page.history.viewport().cursor().shape(), Qt.CursorShape.PointingHandCursor
        )
        page.history_icon_clicked(model.rows[0], "copy")
        self.assertEqual(QApplication.clipboard().text(), "こんにちは")

    def test_choosing_a_candidate_loads_it_and_export_follows(self):
        page = self.window.generation
        record = self.store.root / "generations" / "c1"
        record.mkdir(parents=True)
        for name in ("audio.wav", "audio_2.wav", "audio_3.wav"):
            (record / name).write_bytes(b"")
        write_json(
            record / "generation.json",
            {
                "duration": 1.0,
                "durations": [1.0, 2.5, 3.0],
                "created": "2026-10-29T00:00:00",
                "audio": "audio.wav",
                "candidates": ["audio.wav", "audio_2.wav", "audio_3.wav"],
                "request": {"text": "こんにちは", "num_candidates": 3},
            },
        )
        single = self.store.root / "generations" / "c0"
        single.mkdir()
        (single / "audio.wav").write_bytes(b"")
        write_json(
            single / "generation.json",
            {"duration": 1.0, "created": "2026-10-28T00:00:00", "request": {"text": "a"}},
        )
        page.refresh_history()
        item = next(row for row in self.store.history() if row["id"] == "c1")
        self.assertEqual(item["candidates"][1], str(record / "audio_2.wav"))
        with patch.object(page.player, "load") as load:
            page.show_result(item)
            self.assertEqual(len(page.candidate_buttons), 3)
            self.assertFalse(page.candidate_row.isHidden())
            page.candidate_buttons[1].click()
            load.assert_called_with(str(record / "audio_2.wav"), autoplay=True)
        self.assertEqual(page.output_audio, str(record / "audio_2.wav"))
        self.assertIn("2.50", page.result_label.text())
        self.assertEqual(
            [button.isChecked() for button in page.candidate_buttons], [False, True, False]
        )
        with patch.object(page.player, "load"):
            page.show_result(next(row for row in self.store.history() if row["id"] == "c0"))
        self.assertTrue(page.candidate_row.isHidden())
        self.assertEqual(page.candidate_buttons, [])
        # Deleting a record removes every candidate, even when another one is loaded.
        with patch.object(page.player, "load"):
            page.show_result(item)
            page.candidate_buttons[2].click()
        self.window.confirm = lambda *_args: True
        with patch("irodori_tts.desktop.generation_page.QTimer.singleShot"):
            page.delete_record(item)
        self.assertIsNone(page.output_record)
        page.finish_delete(item)
        self.assertFalse(record.exists())

    def test_emoji_palette_inserts_at_the_cursor_and_stays_open(self):
        from irodori_tts.desktop.emoji_groups import EMOJI_GROUPS, emoji_name, grouped_emoji
        from irodori_tts.desktop.generation_page import EmojiPalette
        from irodori_tts.emoji_palette import EMOJI_PALETTE_ITEMS

        page = self.window.generation
        self.window.show()
        page.text.setPlainText("ab")
        cursor = page.text.textCursor()
        cursor.setPosition(1)
        page.text.setTextCursor(cursor)
        page.show_emoji()
        palette = page.emoji_flyout.view
        self.assertIsInstance(palette, EmojiPalette)
        self.assertEqual(len(palette.buttons), len(EMOJI_PALETTE_ITEMS))
        # Every upstream emoji has a group and a hint of ours, none falling through to Other.
        self.assertEqual(len(grouped_emoji()), len(EMOJI_GROUPS))
        self.assertEqual(
            {button.emoji for button in palette.buttons},
            {item.emoji for item in EMOJI_PALETTE_ITEMS},
        )
        hint = str(EMOJI_GROUPS[0][1]["😆"])
        self.assertEqual(palette.buttons[0].toolTip(), hint)
        # The name is on the button; the hint adds the detail.
        self.assertEqual(palette.buttons[0].text(), f"😆 {emoji_name(hint)}")
        self.assertEqual(emoji_name("Crying: sobbing"), "Crying")
        self.assertEqual(emoji_name("哭泣：呜咽"), "哭泣")
        self.assertEqual(emoji_name("Tender"), "Tender")
        for button in palette.buttons:
            self.assertGreaterEqual(button.width(), button.sizeHint().width())
        first, second = palette.buttons[0].emoji, palette.buttons[1].emoji
        palette.buttons[0].click()
        palette.buttons[1].click()
        self.assertEqual(page.text.toPlainText(), f"a{first}{second}b")
        # The popup holds the window activation; within the window the text box is focused.
        self.assertIs(self.window.focusWidget(), page.text)
        self.assertTrue(page.emoji_flyout.isVisible())
        page.emoji_flyout.close()

    def test_long_text_shows_its_segments_and_disables_candidates(self):
        page = self.window.generation
        page.text.setPlainText("あ。")
        self.assertTrue(page.num_candidates.isEnabled())
        plain = str(t("generation_page.character_count", count=2, limit=5000))
        self.assertEqual(page.count.text(), plain)
        page.text.setPlainText(("あ" * 99 + "。") * 3)
        parts = str(
            t("generation_page.character_count_segments", count=300, limit=5000, segments=3)
        )
        self.assertEqual(page.count.text(), parts)
        self.assertFalse(page.num_candidates.isEnabled())
        self.assertTrue(page.num_candidates.toolTip())
        self.assertEqual(page.request()["text"], ("あ" * 99 + "。") * 3)
        page.text.setPlainText("い。")
        self.assertTrue(page.num_candidates.isEnabled())
        self.assertEqual(page.num_candidates.toolTip(), "")

    def test_fill_back_restores_the_candidate_count(self):
        page = self.window.generation
        self.assertEqual(page.num_candidates.value(), 1)
        page.fill_back({"request": {"text": "a", "num_candidates": 3}})
        self.assertEqual(page.num_candidates.value(), 3)
        self.assertEqual(page.request()["num_candidates"], 3)

    def use_bases(self, *bases):
        """Install these bases (id or (id, kwargs)) for one test and show them on Generate."""
        root = self.window.resource_root
        self.addCleanup(shutil.rmtree, root / "bases", True)
        self.addCleanup((root / "base-models.json").unlink, True)
        installed = []
        for base in bases:
            base_id, options = base if isinstance(base, tuple) else (base, {})
            installed.append(install_base(root, base_id, **options))
        self.window.generation.refresh_bases()
        return installed

    def test_the_fast_model_takes_its_own_steps_for_every_voice(self):
        from irodori_tts.desktop import model_files

        page = self.window.generation
        settings = self.window.store.settings
        self.addCleanup(settings.pop, "generation_model", None)
        self.use_bases(("v4.1-small", {"variants": ("fast",)}))
        page.text.setPlainText("あ。")
        page.speaker_strength.setValue(1.5)
        self.assertEqual(page.request()["speaker_kv_scale"], 1.5)
        settings["generation_model"] = model_files.FAST
        page.show_generation_model()
        self.assertEqual(page.generation_model(), model_files.FAST)
        self.assertFalse(page.steps.isEnabled())
        self.assertFalse(page.cfg_speaker.isEnabled())
        self.assertFalse(page.fast_note.isHidden())
        request = page.request()
        self.assertEqual(request["num_steps"], 4)
        self.assertIsNone(request["speaker_kv_scale"])
        # A trained voice on that base uses it too, with its adapter.
        page.voices = {
            "v": {
                "name": "v",
                "adapter": "a",
                "base_id": "v4.1-small",
                "base_name": "Irodori-TTS v4.1 Small",
                "base_note": None,
                "base_problem": None,
            }
        }
        add_item(page.voice, "v", userData="v")
        page.voice.setCurrentIndex(page.voice.findData("v"))
        self.assertEqual(page.generation_model(), model_files.FAST)
        self.assertFalse(page.steps.isEnabled())
        self.assertEqual(page.request()["lora_adapter"], "a")
        self.assertEqual(page.request()["num_steps"], 4)

    def test_regenerating_an_unlisted_voice_uses_the_base_its_files_describe(self):
        from irodori_tts.desktop.base_models import SMALL_ARCHITECTURE

        page = self.window.generation
        settings = self.window.store.settings
        self.addCleanup(settings.pop, "generation_base", None)
        self.use_bases("v4.1-small", "v4-large")
        page.voice.setCurrentIndex(0)
        page.base.setCurrentIndex(page.base.findData("v4-large"))
        adapter = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, adapter, True)
        write_json(adapter / "config.json", {"model": SMALL_ARCHITECTURE, "train": {}})
        # An older record: no base, and a voice that is no longer listed.
        item = {
            "voice": "gone",
            "voice_id": "gone",
            "request": {"text": "あ。", "lora_adapter": str(adapter)},
        }
        with patch.object(self.window.tasks, "start") as start:
            page.regenerate(item)
        self.assertEqual(start.call_args.args[0]["base"], "v4.1-small")
        # Only another architecture installed: it says what is needed instead of generating.
        shutil.rmtree(self.window.resource_root / "bases")
        (self.window.resource_root / "base-models.json").unlink()
        self.use_bases("v4-large")
        with patch.object(self.window.tasks, "start") as start:
            with self.assertRaises(BusinessError) as raised:
                page.regenerate(item)
        start.assert_not_called()
        self.assertEqual(raised.exception.message.key, "base_models.needs_catalog_base")
        # Nothing tells its architecture: unknown, and refused as well.
        (adapter / "config.json").unlink()
        with patch.object(self.window.tasks, "start") as start:
            with self.assertRaises(BusinessError) as raised:
                page.regenerate(item)
        start.assert_not_called()
        self.assertEqual(raised.exception.message.key, "base_models.base_not_recorded")

    def test_a_base_without_the_chosen_variant_uses_its_standard_model_and_says_so(self):
        from irodori_tts.desktop import model_files

        page = self.window.generation
        settings = self.window.store.settings
        self.addCleanup(settings.pop, "generation_model", None)
        self.addCleanup(settings.pop, "generation_base", None)
        self.use_bases(("v4.1-small", {"variants": ("fast",)}), "v4-large")
        settings["generation_model"] = model_files.FAST
        page.voice.setCurrentIndex(0)
        page.base.setCurrentIndex(page.base.findData("v4-large"))
        self.assertEqual(settings["generation_base"], "v4-large")
        self.assertIsNone(page.generation_model())
        self.assertTrue(page.steps.isEnabled())
        self.assertFalse(page.base_note.isHidden())
        self.assertIn("Irodori-TTS v4 Large", page.base_note.text())
        page.base.setCurrentIndex(page.base.findData("v4.1-small"))
        self.assertEqual(page.generation_model(), model_files.FAST)
        self.assertTrue(page.base_note.isHidden())

    def test_generate_uses_the_chosen_base_or_the_voices_own(self):
        page = self.window.generation
        settings = self.window.store.settings
        self.addCleanup(settings.pop, "generation_base", None)
        self.use_bases("v4.1-small", "v4-large")
        page.text.setPlainText("あ。")
        page.speech_ready = True
        page.voice.setCurrentIndex(0)
        # With two bases, the default voice offers the choice.
        self.assertFalse(page.base_field.isHidden())
        page.base.setCurrentIndex(page.base.findData("v4-large"))
        with patch.object(self.window.tasks, "start") as start:
            page.generate()
        self.assertEqual(start.call_args.args[0]["base"], "v4-large")
        # A trained voice speaks on its own base; the choice gives way to its name.
        page.voices = {
            "v": {
                "name": "v",
                "adapter": "a",
                "base_id": "v4.1-small",
                "base_name": "Irodori-TTS v4.1 Small",
                "base_note": "note",
                "base_problem": None,
            },
            "gone": {
                "name": "gone",
                "adapter": "g",
                "base_id": None,
                "base_name": None,
                "base_note": None,
                "base_problem": "needs another base",
            },
        }
        for key in page.voices:
            add_item(page.voice, key, userData=key)
        page.voice.setCurrentIndex(page.voice.findData("v"))
        self.assertTrue(page.base_field.isHidden())
        self.assertIn("Irodori-TTS v4.1 Small", page.base_note.text())
        self.assertIn("note", page.base_note.text())
        with patch.object(self.window.tasks, "start") as start:
            page.generate()
        self.assertEqual(start.call_args.args[0]["base"], "v4.1-small")
        page.voice.setCurrentIndex(page.voice.findData("gone"))
        self.assertIn("needs another base", page.base_note.text())
        with patch.object(self.window.tasks, "start") as start, self.assertRaises(BusinessError):
            page.generate()
        start.assert_not_called()

    def test_importing_a_package_asks_only_between_the_adapters_that_fit(self):
        from irodori_tts.desktop import library_page
        from irodori_tts.desktop.messages import t

        models = self.window.models
        source = Path("pack.zip")
        bad = t("exports.voice_base_model_unknown")
        started = []
        with (
            patch.object(self.window, "run_io", lambda work, done, *a, **k: started.append(work)),
            patch.object(library_page, "import_voice", lambda *args: args),
            patch.object(library_page, "prompt_choice", return_value="b") as choose,
            patch.object(library_page, "toast") as toast,
        ):
            usable = [{"folder": f, "name": f, "problem": None} for f in ("a", "b")]
            models.import_found(source, [*usable, {"folder": "c", "name": "c", "problem": bad}])
            self.assertEqual([item[1] for item in choose.call_args.args[3]], ["a", "b"])
            self.assertEqual(started.pop()()[2], "b")
            self.assertEqual(toast.call_args.kwargs["kind"], "warning")

            choose.reset_mock()
            models.import_found(source, [usable[0]])
            choose.assert_not_called()
            self.assertEqual(started.pop()()[2], "a")

            models.import_found(source, [{"folder": "c", "name": "c", "problem": bad}])
            self.assertEqual(started, [])
            self.assertEqual(toast.call_args.args[1], bad)

    def test_the_generation_model_chosen_in_settings_is_saved_and_generates(self):
        from irodori_tts.desktop import model_files

        store = self.window.store
        self.addCleanup(store.settings.pop, "generation_model", None)
        self.use_bases(("v4.1-small", {"variants": ("low_vram", "fast")}))
        combo = self.window.settings_page.generation_model
        combo.setCurrentIndex(combo.findData(model_files.LOW_VRAM))
        self.assertEqual(AppStore(store.root).settings["generation_model"], model_files.LOW_VRAM)
        page = self.window.generation
        page.text.setPlainText("あ。")
        page.speech_ready = True
        with patch.object(self.window.tasks, "start") as start:
            page.generate()
            self.assertEqual(start.call_args.args[0]["model"], model_files.LOW_VRAM)
            self.assertEqual(start.call_args.args[0]["base"], "v4.1-small")
            self.assertTrue(page.steps.isEnabled())
            # Generate shows the fast model's fields as soon as Settings chooses it.
            combo.setCurrentIndex(combo.findData(model_files.FAST))
            self.assertFalse(page.steps.isEnabled())
            self.assertFalse(page.fast_note.isHidden())
            combo.setCurrentIndex(combo.findData("standard"))
            self.assertTrue(page.steps.isEnabled())

    def test_pasted_text_stays_plain_and_pickers_start_on_the_desktop(self):
        from PySide6.QtCore import QMimeData, QStandardPaths

        from irodori_tts.desktop.widgets import files_start

        text = self.window.generation.text
        data = QMimeData()
        data.setHtml('<span style="background:#000;font-family:serif">本日</span>')
        data.setText("本日")
        text.clear()
        text.insertFromMimeData(data)
        self.assertEqual(text.toHtml().count("background-color:#000000"), 0)
        self.assertEqual(text.toPlainText(), "本日")
        desktop = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DesktopLocation)
        self.assertEqual(files_start({}), Path(desktop))
        self.assertEqual(files_start({"files_folder": self.temp.name}), Path(self.temp.name))

    def test_table_titles_line_up_with_their_cells_and_rates_read_alike(self):
        from PySide6.QtCore import Qt

        from irodori_tts.desktop.widgets import scientific

        model = self.page.materials.model()
        alignment = model.headerData(
            1, Qt.Orientation.Horizontal, Qt.ItemDataRole.TextAlignmentRole
        )
        self.assertTrue(int(alignment) & Qt.AlignmentFlag.AlignLeft.value)
        self.assertEqual([scientific(v) for v in (1e-4, 2.5e-5, 3e-4)], ["1e-4", "2.5e-5", "3e-4"])
        self.assertEqual(self.page.parameters["learning_rate"].textFromValue(1e-4), "1e-4")


class TranscriptionPageTests(unittest.TestCase):
    """Automatic transcription in the window: dialog, queued steps, review and settings."""

    @classmethod
    def setUpClass(cls):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        cls.repo = Path(__file__).resolve().parents[1]
        configure_environment(cls.repo)
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from irodori_tts.desktop.app import MainWindow
        from irodori_tts.desktop.materials import import_materials

        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.addons = self.base / "addons"
        patcher = patch.object(asr, "addons_root", return_value=self.addons)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = AppStore(self.base / "data")
        self.window = MainWindow(self.store)
        self.page = self.window.training
        self.root = create_project(self.store.root / "projects", "page")
        fixture = self.repo / "examples/desktop"
        import_materials(self.root, fixture, fixture / "transcripts.csv")
        extra = self.base / "extra"
        extra.mkdir()
        shutil.copyfile(fixture / "basic.wav", extra / "third.wav")
        import_materials(self.root, extra)
        project = load_project(self.root)
        for index, item in enumerate(project["materials"][:2]):
            item["asr"] = {
                "backend": "qwen3-asr",
                "check_backend": "whisper-large-v3",
                "check_text": f"別の聞こえ方{index}",
                "needs_review": True,
            }
        write_json(self.root / "project.json", project)
        self.store.register(self.root)
        self.window.refresh()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        cleanup_qt()
        self.temp.cleanup()

    def select(self, position):
        self.page.materials.setCurrentIndex(self.page.materials.model().index(position, 0))
        self.app.processEvents()

    def test_dialog_starts_with_the_defaults_and_keeps_the_check_apart_from_the_main(self):
        from irodori_tts.desktop.asr_dialog import TranscribeDialog

        with (
            patch.object(asr, "missing_models", return_value=[]),
            patch.object(asr, "missing_addons", return_value=[]),
        ):
            dialog = TranscribeDialog(self.window, self.store.settings)
            self.assertEqual(list(dialog.radios), list(asr.BACKEND_KEYS))
            self.assertEqual(dialog.main(), "whisper-large-v3")
            self.assertEqual(dialog.check_choice(), "qwen3-asr")
            self.assertTrue(dialog.radios["whisper-large-v3"].isChecked())
            self.assertTrue(dialog.split.isChecked())
            self.assertFalse(dialog.yesButton.isEnabled())
            self.assertEqual(dialog.yesButton.text(), str(t("asr.start")))
            checks = [dialog.check.itemData(i) for i in range(dialog.check.count())]
            self.assertEqual(checks, ["none", "qwen3-asr", "anime-whisper", "fun-asr-nano"])
            dialog.radios["qwen3-asr"].click()
            checks = [dialog.check.itemData(i) for i in range(dialog.check.count())]
            self.assertNotIn("qwen3-asr", checks)
            self.assertEqual(dialog.check_choice(), "none")
            dialog.folder = self.base
            dialog.selection_changed(False)
            self.assertTrue(dialog.yesButton.isEnabled())
            self.assertEqual(
                dialog.values(),
                {"folder": self.base, "main": "qwen3-asr", "check": None, "split": True},
            )

    def test_dialog_offers_download_and_start_when_something_is_missing(self):
        from irodori_tts.desktop.asr_dialog import TranscribeDialog

        dialog = TranscribeDialog(self.window, self.store.settings, self.base)
        with (
            patch.object(asr, "missing_models", return_value=list(asr.BACKENDS[1].models)),
            patch.object(asr, "missing_addons", return_value=["qwen3-asr"]),
        ):
            dialog.selection_changed(False)
        self.assertEqual(dialog.yesButton.text(), str(t("asr.download_and_start")))
        self.assertFalse(dialog.summary.isHidden())
        self.assertIn("Qwen3-ASR-1.7B", dialog.summary.text())

    def test_missing_addon_and_models_are_queued_ahead_of_the_transcription(self):
        folder = self.base / "voices"
        folder.mkdir()
        with (
            patch.object(self.window.tasks, "start") as start,
            patch.object(
                asr, "missing_models", side_effect=lambda backends: list(backends[0].models)
            ),
        ):
            self.page.start_transcription(self.root, folder, "whisper-large-v3", "qwen3-asr", True)
        first = start.call_args.args[0]
        self.assertEqual((first["kind"], first["keys"]), ("addon", ["qwen3-asr"]))
        second = first["then"]
        self.assertEqual(
            (second["kind"], second["backends"]), ("models", ["whisper-large-v3", "qwen3-asr"])
        )
        third = second["then"]
        self.assertEqual(
            (third["kind"], third["main"], third["check"], third["split"], third["addons"]),
            ("transcribe", "whisper-large-v3", "qwen3-asr", True, ["qwen3-asr"]),
        )
        self.assertNotIn("then", third)
        # With everything in place the transcription starts directly, without add-on paths
        # for a backend that runs on the application's own packages.
        with (
            patch.object(self.window.tasks, "start") as start,
            patch.object(asr, "missing_models", return_value=[]),
        ):
            self.page.start_transcription(self.root, folder, "anime-whisper", None, False)
        direct = start.call_args.args[0]
        self.assertEqual(
            (direct["kind"], direct["addons"], direct["split"]), ("transcribe", [], False)
        )

    def test_worker_is_launched_with_its_addon_in_front_only_when_needed(self):
        manager = self.window.tasks
        environments = []

        class Process:
            ProcessChannelMode = QProcess.ProcessChannelMode

            def __init__(self, *_):
                self.arguments = None
                for name in ("readyReadStandardOutput", "finished", "errorOccurred", "started"):
                    setattr(self, name, type("Signal", (), {"connect": lambda *_: None})())

            def setProcessEnvironment(self, environment):
                environments.append(environment)

            def setArguments(self, arguments):
                self.arguments = arguments

            def __getattr__(self, name):
                return lambda *_: None

        with patch("irodori_tts.desktop.tasks.QProcess", Process):
            for addons in (["qwen3-asr", "funasr"], []):
                task = {"id": "t" + str(len(environments))}
                (manager.store.root / "tasks" / task["id"]).mkdir(parents=True)
                manager._launch(task, {"kind": "transcribe", "addons": addons})
        isolated, plain = environments
        self.assertEqual(
            isolated.value("PYTHONPATH").split(os.pathsep)[:2],
            [str(self.addons / "qwen3-asr"), str(self.addons / "funasr")],
        )
        self.assertEqual(isolated.value("PYTHONNOUSERSITE"), "1")
        self.assertNotIn(str(self.addons), plain.value("PYTHONPATH", ""))
        manager.process = None

    def test_stopping_a_queued_addon_install_ends_it_before_anything_is_installed(self):
        manager = self.window.tasks
        cache = self.base / "cache"
        spy = QSignalSpy(manager.finished)
        with patch.dict(os.environ, {"IRODORI_CACHE": str(cache)}):
            task = manager.start({"kind": "addon", "project": str(self.root), "keys": ["funasr"]})
            manager.stop()
            self.assertTrue(spy.wait(60000))
        self.assertEqual(task["status"], "stopped", task)
        self.assertFalse((cache / "addons" / "funasr").exists())
        self.assertFalse((cache / "addons" / "funasr.partial").exists())

    def test_review_rows_filter_detail_and_saving(self):
        page = self.page
        rows = page.materials.rows_model.rows
        flagged = [row for row in rows if row["state"] == str(t("asr.needs_review"))]
        self.assertEqual(len(flagged), 2)
        # The third recording has no transcript at all: a problem rather than a review.
        self.assertIn("(3)", page.only_problems.text())
        # The footer counts the same recordings the filter shows.
        self.assertEqual(
            page.material_summary.full_text,
            str(
                t(
                    "training_page.dataset_summary_review",
                    included=2,
                    total=3,
                    problems=1,
                    review=2,
                )
            ),
        )
        page.only_problems.setChecked(True)
        self.app.processEvents()
        self.assertEqual(len(page.materials.rows_model.rows), 3)
        page.only_problems.setChecked(False)
        self.select(0)
        self.assertFalse(page.review_box.isHidden())
        self.assertIn("別の聞こえ方0", page.review_text.text())
        self.assertFalse(page.mark_review.isHidden())
        # It does not hold up preprocessing, but the preprocessing page says what is left.
        self.assertFalse(page.review_note.isHidden())
        self.assertIn("2", page.review_note.text())
        # Taking the cross-check text saves it; no separate save is needed.
        page.use_check.click()
        self.app.processEvents()
        self.assertEqual(page.material_text.toPlainText(), "別の聞こえ方0")
        saved = load_project(self.root)["materials"]
        self.assertFalse(saved[0]["asr"]["needs_review"])
        self.assertEqual(saved[0]["text"], "別の聞こえ方0")
        self.assertTrue(saved[1]["asr"]["needs_review"])
        self.assertIn("(2)", page.only_problems.text())
        self.assertIn("1", page.review_note.text())
        self.select(1)
        page.mark_review.click()
        self.app.processEvents()
        self.assertFalse(
            any(i["asr"]["needs_review"] for i in load_project(self.root)["materials"][:2])
        )
        self.assertTrue(page.review_note.isHidden())
        # The cross-check text stays visible as a record once the flag is gone.
        self.assertFalse(page.review_box.isHidden())
        self.assertTrue(page.mark_review.isHidden())
        # A recording that was never cross-checked shows no such box.
        self.select(2)
        self.assertTrue(page.review_box.isHidden())

    def test_importing_is_transcribing_and_its_summary_is_a_toast(self):
        page = self.page
        with patch.object(page, "import_transcribed") as transcribe:
            page.import_button.click()
        transcribe.assert_called_once()
        others = [action.text() for action in page.more_imports.menu().actions()]
        self.assertIn(str(t("training_page.import_audio_and_csv")), others)
        self.assertNotIn(str(t("training_page.import_audio_and_transcribe")), others)
        with patch.object(self.window, "inform") as inform:
            page.transcribed(
                {
                    "kind": "transcribe",
                    "project": str(self.root),
                    "status": "completed",
                    "message": "",
                    "result": {
                        "imported": 152,
                        "segments": 40,
                        "split_files": 20,
                        "needs_review": 41,
                    },
                }
            )
        # What the import did is said once, not left on the page to be cut short.
        inform.assert_called_once()
        self.assertIn("152", str(inform.call_args.args[0]))
        self.assertFalse(hasattr(page, "import_note"))

    def test_an_action_error_is_a_toast_and_shows_again_on_a_second_click(self):
        page = self.page
        with patch("irodori_tts.desktop.widgets.toast") as shown:
            for _ in range(2):
                self.window.perform(
                    lambda: (_ for _ in ()).throw(BusinessError("broken")), page.material_error
                )
        self.assertEqual(shown.call_count, 2)
        self.assertEqual(page.material_error.full_text, "broken")
        self.assertEqual(page.material_error.text(), "")

    def test_every_recording_shows_its_delete_icon(self):
        page = self.page
        icons = [row.get("icons", {}).get("actions") for row in page.materials.rows_model.rows]
        self.assertTrue(icons)
        self.assertTrue(all(icon is FluentIcon.DELETE for icon in icons))

    def test_the_recording_selected_before_a_restart_can_be_played(self):
        from irodori_tts.desktop.app import MainWindow

        self.page.set_project(self.root)
        self.select(1)
        selected = self.page.selected_material()
        self.window.save_ui()
        reopened = MainWindow(AppStore(self.store.root))
        try:
            for _ in range(5):
                self.app.processEvents()
            page = reopened.training
            self.assertEqual(page.material_id, selected["id"])
            self.assertEqual(page.material_player.path, str(page.root / selected["path"]))
            self.assertTrue(page.material_player.isEnabled())
        finally:
            reopened.close()
            self.app.processEvents()

    def test_no_page_cuts_off_text_or_scrolls_sideways_at_the_smallest_window(self):
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QAbstractButton, QLabel

        from irodori_tts.desktop.app import MINIMUM_SIZE
        from irodori_tts.desktop.widgets import ElidedComboBox, ElidedLabel, ErrorLabel

        window = self.window
        window.stack.setAnimationEnabled(False)
        window.show()
        self.select(1)
        problems = []
        for language in ("zh_CN", "en"):
            combo = window.settings_page.language
            combo.setCurrentIndex(combo.findData(language))
            window.resize(*MINIMUM_SIZE)
            QTest.qWait(150)
            views = [window.show_guide] + [
                lambda i=i: window.navigate(i) for i in range(len(window.pages))
            ]
            views += [
                lambda i=i: (window.navigate(2), self.page.tabs.setCurrentIndex(i))
                for i in range(3)
            ]
            for show in views:
                show()
                QTest.qWait(150)
                page = window.stack.currentWidget()
                bar = getattr(page, "horizontalScrollBar", None)
                if bar is not None and bar().maximum() > 0:
                    problems.append((language, page.objectName(), "scrolls sideways"))
                for widget in page.findChildren(QAbstractButton) + page.findChildren(QLabel):
                    elided = ElidedLabel | ErrorLabel | ElidedComboBox
                    if not widget.isVisible() or isinstance(widget, elided):
                        continue
                    if isinstance(widget, QLabel) and (widget.wordWrap() or not widget.text()):
                        continue
                    if widget.text() and widget.width() < widget.sizeHint().width() - 1:
                        problems.append((language, page.objectName(), widget.text()))
        self.assertEqual(problems, [])

    def test_recordings_can_be_deleted_one_at_a_time_or_all_together(self):
        from PySide6.QtTest import QTest

        page = self.page
        project = load_project(self.root)
        first = project["materials"][0]
        copy = self.root / first["path"]
        source = Path(first["source"])
        revision = project["revision"]
        self.select(0)
        with patch.object(self.window, "confirm", return_value=True) as confirm:
            page.delete_selected()
            QTest.qWait(400)
        self.assertEqual(confirm.call_count, 1)
        left = load_project(self.root)
        self.assertNotIn(first["id"], [item["id"] for item in left["materials"]])
        self.assertEqual(len(left["materials"]), 2)
        self.assertGreater(left["revision"], revision)
        # The project's copy goes; the file it was imported from stays.
        self.assertFalse(copy.exists())
        self.assertTrue(source.exists())
        self.assertIsNone(page.material_id)
        # Declining keeps everything.
        with patch.object(self.window, "confirm", return_value=False):
            page.clear_materials()
            QTest.qWait(400)
        self.assertEqual(len(load_project(self.root)["materials"]), 2)
        with patch.object(self.window, "confirm", return_value=True):
            page.clear_materials()
            QTest.qWait(400)
        self.assertEqual(load_project(self.root)["materials"], [])
        self.assertEqual(page.materials.requested_rows, [])

    def test_the_drawn_icons_load_in_both_themes(self):
        from qfluentwidgets import Theme

        from irodori_tts.desktop.widgets import AppIcon

        for icon in AppIcon:
            for theme in (Theme.LIGHT, Theme.DARK):
                self.assertTrue(Path(icon.path(theme)).is_file(), icon)
                self.assertFalse(icon.icon(theme).pixmap(20, 20).isNull(), icon)

    def test_listed_recordings_can_be_transcribed_again(self):
        page = self.page
        page.only_problems.setChecked(True)
        self.app.processEvents()
        listed = [row["id"] for row in page.materials.requested_rows]
        self.assertEqual(len(listed), 3)
        choice = {"folder": None, "main": "qwen3-asr", "check": None, "split": True}
        with (
            patch(
                "irodori_tts.desktop.training_page.ask_transcription", return_value=choice
            ) as ask,
            patch.object(asr, "missing_models", return_value=[]),
            patch.object(asr, "missing_addons", return_value=[]),
            patch.object(self.window.tasks, "start") as start,
        ):
            page.retranscribe_listed()
        self.assertEqual(ask.call_args.kwargs["again"], 3)
        payload = start.call_args.args[0]
        self.assertEqual(payload["kind"], "transcribe")
        self.assertEqual(payload["materials"], listed)
        self.assertIsNone(payload["folder"])
        self.assertFalse(payload["split"])

    def test_a_transcript_changed_underneath_is_shown_and_never_overwritten(self):
        page = self.page
        self.select(0)
        first = page.material_id
        project = load_project(self.root)
        project["materials"][0]["text"] = "新しい文字起こし"
        write_json(self.root / "project.json", project)
        page.set_project(self.root)
        self.assertEqual(page.material_text.toPlainText(), "新しい文字起こし")
        self.select(1)
        saved = next(i for i in load_project(self.root)["materials"] if i["id"] == first)
        self.assertEqual(saved["text"], "新しい文字起こし")
        # An edit the user did make is still kept when leaving the recording.
        page.material_text.setPlainText("ユーザーの修正")
        self.select(0)
        edited = load_project(self.root)["materials"][1]
        self.assertEqual(edited["text"], "ユーザーの修正")

    def test_recording_pickers_open_in_the_last_folder_or_the_documents(self):
        from PySide6.QtCore import QStandardPaths

        from irodori_tts.desktop.asr_dialog import choose_recordings_folder, recordings_start

        documents = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        )
        self.assertEqual(recordings_start(self.store.settings), documents)
        chosen = self.base / "extra"
        with patch(
            "irodori_tts.desktop.asr_dialog.QFileDialog.getExistingDirectory",
            return_value=str(chosen),
        ) as dialog:
            self.assertEqual(choose_recordings_folder(self.window, self.store.settings), chosen)
            self.assertEqual(dialog.call_args.args[2], documents)
            choose_recordings_folder(self.window, self.store.settings)
            self.assertEqual(dialog.call_args.args[2], str(chosen))
        # A folder that has since gone falls back to the documents.
        self.store.settings["recordings_folder"] = str(self.base / "gone")
        self.assertEqual(recordings_start(self.store.settings), documents)

    def test_settings_choose_defaults_and_list_downloads_that_can_be_removed(self):
        page = self.window.settings_page
        self.assertEqual(page.asr_main.currentData(), "whisper-large-v3")
        self.assertEqual(page.asr_check.currentData(), "qwen3-asr")
        page.asr_main.setCurrentIndex(page.asr_main.findData("qwen3-asr"))
        self.assertEqual(self.store.settings["asr"], {"main": "qwen3-asr", "check": "none"})
        self.assertEqual(page.asr_check.currentData(), "none")
        folder = self.addons / "funasr"
        folder.mkdir(parents=True)
        (folder / "addon.json").write_text(
            json.dumps(
                {"packages": list(asr.ADDONS["funasr"].packages), "python": asr.python_tag()}
            )
        )
        (folder / "payload.bin").write_bytes(b"0" * 2048)
        page.refresh()
        ids = [row["id"] for row in page.asr_files.rows_model.rows]
        self.assertIn("addon:funasr", ids)
        row = next(r for r in page.asr_files.rows_model.rows if r["id"] == "addon:funasr")
        self.assertEqual(row["name"], "Fun-ASR-Nano")
        with patch.object(self.window, "confirm", return_value=True):
            page.remove_asr(row)
        deadline = time.monotonic() + 10
        while folder.exists() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.05)
        self.assertFalse(folder.exists())

    def test_settings_list_each_installed_base_with_its_version(self):
        page = self.window.settings_page
        root = self.window.resource_root
        self.addCleanup(shutil.rmtree, root / "bases", True)
        self.addCleanup((root / "base-models.json").unlink, True)
        install_base(root, "v4.1-small")
        install_base(root, "imported-aaaaaaaaaaaa", name="Mine")
        page.latest = {"v4.1-small": "f" * 40}
        page.refresh()
        rows = {row["id"]: row for row in page.base_table.rows_model.rows}
        self.assertEqual(str(rows["v4.1-small"]["version"]), "rev1")
        self.assertEqual(
            str(rows["imported-aaaaaaaaaaaa"]["version"]), str(t("settings_page.base_imported"))
        )
        # A newer commit upstream offers the update, and the update is a download task.
        with patch.object(self.window.tasks, "start") as start:
            page.update_base(rows["v4.1-small"])
        self.assertEqual(
            start.call_args.args[0], {"kind": "resources", "bases": ["v4.1-small"], "update": True}
        )

    def test_downloaded_models_are_listed_with_sizes_and_can_be_removed(self):
        page = self.window.settings_page
        whisper = asr.backend_for("whisper-large-v3").models[0]

        def size(model):
            return 3_091_519_764 if model.repo == whisper.repo else 0

        with patch("irodori_tts.desktop.settings_page.snapshot_size", side_effect=size):
            page.refresh()
        (row,) = [r for r in page.asr_files.requested_rows if r["id"].startswith("model:")]
        self.assertEqual((row["id"], row["name"]), ("model:whisper-large-v3", "whisper-large-v3"))
        self.assertEqual(str(row["size"]), "3.09 GB")
        removed = []
        with (
            patch.object(self.window, "confirm", return_value=True),
            patch("irodori_tts.desktop.settings_page.remove_snapshot", side_effect=removed.append),
            patch("irodori_tts.desktop.settings_page.snapshot_size", return_value=0),
        ):
            page.remove_asr(row)
            deadline = time.monotonic() + 10
            while not removed and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.05)
        self.assertEqual(removed, [whisper])


if __name__ == "__main__":
    from tests.run import guarded_main

    raise SystemExit(guarded_main(lambda: unittest.main()))
