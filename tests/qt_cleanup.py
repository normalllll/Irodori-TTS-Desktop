"""Release test-owned Qt resources while their QApplication still exists."""

from __future__ import annotations

import sys
import time


def stub_connection_check():
    """The guide checks Hugging Face over the network, which takes up to ~10 s, longer than
    cleanup_qt waits for background jobs: window tests get an instant answer instead. Returns
    the started patcher."""
    from unittest.mock import patch

    patcher = patch(
        "irodori_tts.desktop.guide_page.check_hugging_face",
        return_value={"reachable": True, "latency_ms": 1, "reason": None, "status": None},
    )
    patcher.start()
    return patcher


def stop_manager(manager):
    from shiboken6 import isValid

    if not isValid(manager):
        return
    manager.blockSignals(True)
    manager.training_preview = None
    manager.pending = None
    manager.timer.stop()
    manager.finish_timer.stop()
    process = manager.process
    if process is not None:
        if manager.busy:
            try:
                manager.stop()
            except OSError:
                # A failed test may already have removed its TemporaryDirectory.
                process.kill()
        else:
            process.write(b'{"command":"shutdown"}\n')
            process.closeWriteChannel()
        if not process.waitForFinished(5000):
            process.kill()
            if not process.waitForFinished(5000):
                raise RuntimeError("Test compute process did not exit")
        if manager.task_dir().exists():
            manager.poll()
    manager.finish_timer.stop()


def cleanup_qt():
    """Called after tests, before interpreter finalization; never opens a window."""
    widgets = sys.modules.get("PySide6.QtWidgets")
    if widgets is None or (app := widgets.QApplication.instance()) is None:
        return
    from PySide6.QtCore import QCoreApplication, QEvent, QThreadPool, QTimer
    from shiboken6 import isValid

    app.setQuitOnLastWindowClosed(False)
    main_module = sys.modules.get("irodori_tts.desktop.app")
    main_type = getattr(main_module, "MainWindow", ())
    windows = [window for window in app.topLevelWidgets() if isValid(window)]
    for window in windows:
        for timer in window.findChildren(QTimer):
            timer.stop()
        if isinstance(window, main_type) and hasattr(window, "tasks"):
            stop_manager(window.tasks)
    # Construction single-shots may enqueue jobs on the next event turn. Drain
    # both those submissions and their queued results before deleting their owners.
    pool = QThreadPool.globalInstance()
    deadline = time.monotonic() + 5
    while True:
        app.processEvents()
        pool.waitForDone(50)
        app.processEvents()
        if pool.activeThreadCount() == 0 and not any(
            getattr(window, "jobs", ()) for window in windows if isValid(window)
        ):
            break
        if time.monotonic() > deadline:
            raise RuntimeError("Test background jobs did not finish")
    player_type = getattr(sys.modules.get("irodori_tts.desktop.widgets"), "Player", ())
    waiting = set()
    for widget in app.allWidgets():
        if isValid(widget) and isinstance(widget, player_type):
            waiting.add(widget)
            widget.stopped.connect(lambda widget=widget: waiting.discard(widget))
            widget.destroyed.connect(lambda _=None, widget=widget: waiting.discard(widget))
            widget.stop()
    deadline = time.monotonic() + 5
    while waiting:
        app.processEvents()
        if time.monotonic() > deadline:
            raise RuntimeError("Test audio players did not stop")
    for window in windows:
        if isValid(window):
            window.hide()
            window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
