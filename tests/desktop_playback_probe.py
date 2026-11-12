"""Muted MainWindow lifecycle in a real Qt event loop, with a plain Python callback.

Run via tests.desktop_playback for an external hard process deadline.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QMetaObject, QObject, Qt, QTimer, Slot
from PySide6.QtMultimedia import QMediaDevices, QMediaPlayer
from PySide6.QtWidgets import QApplication

from irodori_tts.desktop.app import MainWindow
from irodori_tts.desktop.storage import AppStore, write_json

REPO = Path(__file__).resolve().parents[1]


class Probe(QObject):
    def __init__(self, app, args, directory):
        super().__init__()
        self.app, self.args = app, args
        self.window = MainWindow(AppStore(Path(directory)))
        self.window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.view = self.window.generation.player
        self.media = self.view.player
        for view in (
            self.view,
            self.window.models.imported.player,
            self.window.training.material_player,
        ):
            view.player.setMuted(True)
            view.player.setVolume(0)
        self.phase = "first"
        self.report = {
            "action": args.action,
            "default_device": QMediaDevices.defaultAudioOutput().description(),
            "muted": True,
            "stages": [],
            "passed": False,
        }
        self.media.positionChanged.connect(lambda value: self.position(value))
        self.media.durationChanged.connect(self.duration_ready)
        self.window.destroyed.connect(self.destroyed_window)
        self.window.show()  # offscreen platform only; no desktop surface or input
        self.first = REPO / "examples/desktop/voice_design.wav"
        self.second = REPO / "examples/desktop/basic.wav"
        self.view.load(self.first, autoplay=True)
        signature = self.view.source_signature
        self.view.load(self.first)
        assert self.view.source_signature == signature
        self.stage("source-requested")

    def stage(self, name):
        self.report["stages"].append(name)
        write_json(self.args.output, self.report)
        print(name, flush=True)

    def position(self, value):
        if value <= 100:
            return
        if self.phase == "second":
            self.phase = "closing"
            assert Path(self.media.source().toLocalFile()) == self.second
            self.stage("new-source-playing")
            self.view.stop()
            QMetaObject.invokeMethod(self, "close_window", Qt.ConnectionType.QueuedConnection)
            return
        if self.phase != "first":
            return
        self.phase = "acting"
        assert self.media.duration() > 0
        assert self.report.get("metadata_duration_visible")
        position = self.media.position()
        self.view.load(self.first)
        assert self.media.position() == position
        self.stage("same-source-preserved-playback")
        self.stage("position-advanced")
        self.media.pause()
        self.media.setPosition(1000)
        assert self.media.position() == 1000
        self.stage("seeked")
        if self.args.action == "stop":
            self.view.stop()
        elif self.args.action == "switch":
            self.view.load(self.second, autoplay=True)
        elif self.args.action == "clear":
            self.view.clear()
        else:
            self.window.close()
        self.stage("action-call-returned")
        if self.args.action != "close":
            QMetaObject.invokeMethod(self, "check_action", Qt.ConnectionType.QueuedConnection)

    @Slot(int)
    def duration_ready(self, duration):
        if duration > 0 and self.phase == "first":
            assert self.view.remainTimeLabel.text() != "0:00:00"
            self.report["metadata_duration_visible"] = True

    @Slot()
    def check_action(self):
        if self.args.action == "switch":
            assert Path(self.media.source().toLocalFile()) == self.second
            self.phase = "second"
            self.stage("source-switch-completed")
            return
        assert self.media.playbackState() == QMediaPlayer.PlaybackState.StoppedState
        assert self.media.position() == 0
        if self.args.action == "clear":
            assert (
                self.media.source().isEmpty()
                and not self.view.isEnabled()
                and self.view.path is None
            )
        self.stage("action-completed")
        self.close_window()

    @Slot()
    def close_window(self):
        self.window.close()
        self.stage("window-close-call-returned")

    @Slot()
    def destroyed_window(self):
        self.report["passed"] = True
        self.stage("window-and-media-owner-destroyed")
        self.app.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("stop", "switch", "clear", "close"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    with tempfile.TemporaryDirectory() as directory:
        probe = Probe(app, args, directory)
        QTimer.singleShot(12000, lambda: (probe.stage("timeout"), app.exit(1)))
        result = app.exec()
        from tests.qt_cleanup import cleanup_qt

        cleanup_qt()
        return result


if __name__ == "__main__":
    from tests.run import guarded_main

    sys.exit(guarded_main(main, timeout=18))
