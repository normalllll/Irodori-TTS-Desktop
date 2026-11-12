"""Bounded test supervisor: python -m tests.run --timeout 180 unittest ..."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import runpy
import subprocess
import sys
import traceback
from pathlib import Path

from .processes import run_process


def run_child(arguments):
    faulthandler.enable()
    from .qt_cleanup import cleanup_qt

    module, *arguments = arguments
    sys.argv = [module, *arguments]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        cleanup_qt()


def guarded_main(main, *, timeout=300):
    """Standalone acceptance scripts also get supervision when run directly."""
    if os.environ.get("IRODORI_TEST_SUPERVISED"):
        from .qt_cleanup import cleanup_qt

        try:
            return main()
        finally:
            cleanup_qt()
    module = sys.modules[main.__module__].__spec__.name
    return supervise([module, *sys.argv[1:]], timeout=timeout)


def supervise(arguments, *, timeout, output=None):
    report = {
        "module": arguments[0],
        "timeout_seconds": timeout,
        "supervisor_pid": os.getpid(),
        "state": "running",
    }

    def save_report():
        if output:
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
            temporary.replace(destination)

    save_report()
    env = {**os.environ, "IRODORI_TEST_SUPERVISED": "1", "QT_QPA_PLATFORM": "offscreen"}
    try:
        result = run_process(
            [sys.executable, "-X", "faulthandler", "-m", "tests.run", "--child", *arguments],
            timeout=timeout,
            env=env,
            report=report,
            stdout=sys.stdout,
            stderr=sys.stderr,
            stdin=subprocess.DEVNULL,
        )
        report["returncode"] = result.returncode
        code = result.returncode if 0 <= result.returncode <= 255 else 1
        if code == 0 and report.get("active_processes", 0):
            print(
                "Test returned success but left child processes; cleanup forced, marking failure.",
                file=sys.stderr,
            )
            code = 1
    except subprocess.TimeoutExpired:
        report["timed_out"] = True
        print(f"Test exceeded {timeout}s; its process tree was terminated.", file=sys.stderr)
        code = 124
    except KeyboardInterrupt:
        report["interrupted"] = True
        code = 130
    except Exception:
        traceback.print_exc()
        code = 1
    report["exit_code"] = code
    report["state"] = "finished"
    save_report()
    print(
        f"Test exit={code}; remaining owned processes={report.get('active_after_cleanup', 'unknown')}",
        flush=True,
    )
    return code


def main():
    if sys.argv[1:2] == ["--child"]:
        return run_child(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--report", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments or args.timeout <= 0:
        parser.error("Provide a positive timeout and the test module to run")
    return supervise(arguments, timeout=args.timeout, output=args.report)


if __name__ == "__main__":
    raise SystemExit(main())
