"""Deliberate test failures, confined by tests.run and its process job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    args.state.mkdir(parents=True, exist_ok=True)
    if args.mode == "native-crash" and not os.environ.get("IRODORI_TEST_SUPERVISED"):
        raise RuntimeError("Native crash probe requires the process supervisor")
    process = psutil.Process()
    record = args.state / f"{process.pid}.json"
    temporary = record.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"pid": process.pid, "created": process.create_time(), "role": args.mode}),
        encoding="utf-8",
    )
    temporary.replace(record)
    child = None
    if args.mode != "grandchild":
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.process_probe",
                "--mode",
                "grandchild" if args.mode == "child" else "child",
                "--state",
                str(args.state),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    if args.mode in {"child", "grandchild"}:
        while not (args.state / "stop").exists():
            time.sleep(0.02)
        if child:
            child.wait(timeout=5)
        return
    deadline = time.monotonic() + 5
    while len(list(args.state.glob("*.json"))) < 3:
        if time.monotonic() >= deadline:
            raise RuntimeError("Descendant processes did not start")
        time.sleep(0.02)
    if args.mode == "normal":
        (args.state / "stop").touch()
        child.wait(timeout=5)
    elif args.mode == "failure":
        raise AssertionError("Deliberate assertion failure with live descendants")
    elif args.mode == "native-crash":
        from PySide6.QtWidgets import QApplication, QWidget

        app = QApplication([])
        window = QWidget()
        window.show()
        app.processEvents()
        os.abort()
    elif args.mode == "qt-failure":
        from PySide6.QtWidgets import QApplication

        from irodori_tts.desktop.app import MainWindow
        from irodori_tts.desktop.storage import AppStore

        app = QApplication([])
        app.setQuitOnLastWindowClosed(False)
        window = MainWindow(AppStore(args.state / "data"))
        window.show()
        window.tasks.start({"kind": "resources"})
        assert window.tasks.process.waitForStarted(2000)
        worker = psutil.Process(window.tasks.process.processId())
        (args.state / f"{worker.pid}.json").write_text(
            json.dumps({"pid": worker.pid, "created": worker.create_time(), "role": "Qt worker"}),
            encoding="utf-8",
        )
        app.processEvents()
        raise AssertionError("Deliberate failure with a live Qt window and worker")
    elif args.mode == "hang":
        while True:
            time.sleep(1)
    elif args.mode != "leak":
        raise ValueError(args.mode)


if __name__ == "__main__":
    from tests.run import guarded_main

    raise SystemExit(guarded_main(main))
