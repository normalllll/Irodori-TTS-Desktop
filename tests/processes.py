"""Test-owned process trees with bounded lifetime; no Qt or application imports."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from ctypes import wintypes


def configure_error_reporting():
    """A native test failure must return an error, never wait for a desktop dialog."""
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetErrorMode.restype = wintypes.UINT
        kernel.SetErrorMode.argtypes = [wintypes.UINT]
        kernel.SetErrorMode(kernel.GetErrorMode() | 0x8003)
        runtime = ctypes.CDLL("ucrtbase")
        runtime._set_abort_behavior.argtypes = [ctypes.c_uint, ctypes.c_uint]
        runtime._set_abort_behavior.restype = ctypes.c_uint
        runtime._set_abort_behavior(
            0, 3
        )  # CRT abort UI / ReportFault; stderr retains the traceback


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_int64),
        ("job_time", ctypes.c_int64),
        ("flags", wintypes.DWORD),
        ("minimum_working_set", ctypes.c_size_t),
        ("maximum_working_set", ctypes.c_size_t),
        ("active_process_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority", wintypes.DWORD),
        ("scheduling", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("read", "write", "other", "rb", "wb", "ob")]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _BasicLimits),
        ("io", _IoCounters),
        ("process_memory", ctypes.c_size_t),
        ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t),
    ]


class _Accounting(ctypes.Structure):
    _fields_ = [
        ("user", ctypes.c_int64),
        ("kernel", ctypes.c_int64),
        ("period_user", ctypes.c_int64),
        ("period_kernel", ctypes.c_int64),
        ("faults", wintypes.DWORD),
        ("total", wintypes.DWORD),
        ("active", wintypes.DWORD),
        ("terminated", wintypes.DWORD),
    ]


class WindowsJob:
    """Own exactly one test tree, including venv redirectors and native workers."""

    def __init__(self):
        self.kernel = k = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "SetInformationJobObject": (
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
                wintypes.BOOL,
            ),
            "QueryInformationJobObject": (
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p],
                wintypes.BOOL,
            ),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            "OpenThread": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            "ResumeThread": ([wintypes.HANDLE], wintypes.DWORD),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(k, name)
            function.argtypes, function.restype = arguments, result
        self.handle = k.CreateJobObjectW(None, None)  # non-inheritable, supervisor owns it
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def attach_and_resume(self, process):
        import psutil

        handle = self.kernel.OpenProcess(0x0101, False, process.pid)  # SET_QUOTA | TERMINATE
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.kernel.CloseHandle(handle)
        # The primary thread has not executed: no grandchildren can escape the job.
        threads = psutil.Process(process.pid).threads()
        if len(threads) != 1:
            raise RuntimeError("Suspended test process has an unexpected thread count")
        thread = self.kernel.OpenThread(0x0002, False, threads[0].id)
        if not thread:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if self.kernel.ResumeThread(thread) != 1:
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.kernel.CloseHandle(thread)

    def accounting(self):
        result = _Accounting()
        if not self.kernel.QueryInformationJobObject(
            self.handle, 1, ctypes.byref(result), ctypes.sizeof(result), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"total_processes": result.total, "active_processes": result.active}

    def reap(self):
        # Job accounting can lag the signalled process handle briefly.
        deadline = time.monotonic() + 0.2
        while self.accounting()["active_processes"] and time.monotonic() < deadline:
            time.sleep(0.01)
        before = self.accounting()
        if before["active_processes"]:
            if not self.kernel.TerminateJobObject(self.handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
        deadline = time.monotonic() + 5
        while self.accounting()["active_processes"]:
            if time.monotonic() >= deadline:
                raise RuntimeError("Test job still contains live processes after termination")
            time.sleep(0.02)
        return {**before, "active_after_cleanup": 0}

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


@contextmanager
def managed_process(arguments, *, report=None, **options):
    """Always reap the entire owned tree, even after an assertion or timeout."""
    configure_error_reporting()
    job = WindowsJob() if os.name == "nt" else None
    process = None
    report = report if report is not None else {}
    try:
        options["env"] = {**(options.get("env") or os.environ), "IRODORI_TEST_SUPERVISED": "1"}
        if job:
            options["creationflags"] = options.get("creationflags", 0) | 0x08000004
        else:
            options["start_new_session"] = True
        process = subprocess.Popen(arguments, **options)
        report["pid"] = process.pid
        if job:
            job.attach_and_resume(process)
        yield process
    finally:
        try:
            if job:
                report.update(job.reap())
            elif process:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
        finally:
            if job:
                job.close()


def run_process(
    arguments, *, timeout, check=False, capture_output=False, input=None, report=None, **options
):
    if capture_output:
        options.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if input is not None:
        options["stdin"] = subprocess.PIPE
    with managed_process(arguments, report=report, **options) as process:
        stdout, stderr = process.communicate(input=input, timeout=timeout)
        result = subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result
