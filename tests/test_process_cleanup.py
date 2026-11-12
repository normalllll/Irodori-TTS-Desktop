"""Real process-tree cleanup under normal exit, failures, hangs and native abort."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import psutil

from tests.processes import WindowsJob, managed_process, run_process


@unittest.skipUnless(os.name == "nt", "Windows Job Object and native error-mode checks")
class TestProcessCleanup(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def assert_tree_exited(self, state):
        records = [json.loads(path.read_text("utf-8")) for path in state.glob("*.json")]
        self.assertGreaterEqual(len(records), 3)
        for record in records:
            try:
                process = psutil.Process(record["pid"])
                self.assertNotEqual(process.create_time(), record["created"], record)
            except psutil.NoSuchProcess:
                pass

    def run_probe(self, mode, timeout=15):
        report = self.root / "report.json"
        state = self.root / "state"
        with (self.root / "probe.log").open("w", encoding="utf-8") as log:
            result = run_process(
                [
                    sys.executable,
                    "-m",
                    "tests.run",
                    "--timeout",
                    str(timeout),
                    "--report",
                    str(report),
                    "tests.process_probe",
                    "--mode",
                    mode,
                    "--state",
                    str(state),
                ],
                timeout=timeout + 12,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        self.assert_tree_exited(state)
        data = json.loads(report.read_text("utf-8"))
        self.assertEqual(data["active_after_cleanup"], 0)
        if output := os.environ.get("IRODORI_TEST_OUTPUT"):
            destination = Path(output) / "process-probes" / mode
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(report, destination / "report.json")
            shutil.copyfile(self.root / "probe.log", destination / "probe.log")
            for path in state.glob("*.json"):
                shutil.copyfile(path, destination / path.name)
        return result, data, (self.root / "probe.log").read_text("utf-8")

    def test_normal_exit_leaves_no_children(self):
        result, report, _ = self.run_probe("normal")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(report["active_processes"], 0)

    def test_assertion_failure_reaps_grandchildren_and_retains_traceback(self):
        result, _, log = self.run_probe("failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Deliberate assertion failure", log)

    def test_hard_deadline_kills_entire_tree(self):
        result, report, _ = self.run_probe("hang", timeout=3)
        self.assertEqual(result.returncode, 124)
        self.assertTrue(report["timed_out"])

    def test_native_abort_returns_without_error_dialog(self):
        result, report, log = self.run_probe("native-crash")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotEqual(result.returncode, 124, log)
        self.assertFalse(report.get("timed_out", False))
        self.assertIn("Fatal Python error", log)

    def test_success_with_leaked_children_is_reported_as_failure(self):
        result, report, _ = self.run_probe("leak")
        self.assertEqual(report["returncode"], 0)
        self.assertEqual(result.returncode, 1)
        self.assertGreater(report["active_processes"], 0)

    def test_qt_failure_closes_window_and_resource_worker(self):
        result, _, log = self.run_probe("qt-failure", timeout=25)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotEqual(result.returncode, 124, log)
        self.assertIn("Deliberate failure with a live Qt window", log)
        self.assertNotIn("QProcess: Destroyed while process", log)

    def test_supervisor_crash_still_kills_its_job(self):
        report = self.root / "supervisor.json"
        state = self.root / "state"
        with managed_process(
            [
                sys.executable,
                "-m",
                "tests.run",
                "--report",
                str(report),
                "tests.process_probe",
                "--mode",
                "hang",
                "--state",
                str(state),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as outer:
            deadline = time.monotonic() + 10
            while len(list(state.glob("*.json"))) < 3:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.02)
            pid = json.loads(report.read_text("utf-8"))["supervisor_pid"]
            supervisor = psutil.Process(pid)
            supervisor.kill()
            supervisor.wait(timeout=5)
            outer.wait(timeout=5)
            deadline = time.monotonic() + 5
            while any(psutil.pid_exists(int(p.stem)) for p in state.glob("*.json")):
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.02)
            self.assert_tree_exited(state)

    def test_native_error_mode_remains_set_after_qt_initialization(self):
        code = (
            "import ctypes; import tests; from PySide6.QtWidgets import QApplication; "
            "app=QApplication([]); "
            "assert ctypes.windll.kernel32.GetErrorMode() & 0x8003 == 0x8003"
        )
        result = run_process(
            [sys.executable, "-c", code],
            timeout=10,
            capture_output=True,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_failed_job_assignment_never_runs_or_leaves_a_suspended_child(self):
        report = {}
        marker = self.root / "should-not-run"
        with patch.object(
            WindowsJob, "attach_and_resume", side_effect=RuntimeError("Job rejected")
        ):
            with self.assertRaisesRegex(RuntimeError, "Job rejected"):
                run_process(
                    [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
                        str(marker),
                    ],
                    timeout=5,
                    report=report,
                )
        self.assertFalse(marker.exists())
        self.assertFalse(psutil.pid_exists(report["pid"]))


if __name__ == "__main__":
    from tests.run import guarded_main

    raise SystemExit(guarded_main(lambda: unittest.main()))
