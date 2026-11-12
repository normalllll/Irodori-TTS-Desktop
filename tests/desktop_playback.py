"""Serial independent media probes, each with an external hard deadline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from irodori_tts.desktop.storage import read_json, write_json
from tests.processes import run_process


def run_probes(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for action in ("stop", "switch", "clear", "close"):
        output = root / (action + ".json")
        output.unlink(missing_ok=True)
        with (root / (action + ".log")).open("w", encoding="utf-8") as log:
            try:
                process = run_process(
                    [
                        sys.executable,
                        "-m",
                        "tests.desktop_playback_probe",
                        "--action",
                        action,
                        "--output",
                        str(output.resolve()),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=18,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = "hard-timeout"
        report = read_json(output, {})
        results.append({"action": action, "exit_code": code, "report": report})
        print(action, code, report.get("passed", False), flush=True)
    write_json(
        root / "playback-result.json",
        {
            "passed": all(
                result["exit_code"] == 0 and result["report"].get("passed") for result in results
            ),
            "results": results,
        },
    )
    return results


if __name__ == "__main__":
    records = run_probes(Path(__file__).resolve().parents[1] / "outputs/desktop-playback")
    sys.exit(
        0
        if all(record["exit_code"] == 0 and record["report"].get("passed") for record in records)
        else 1
    )
