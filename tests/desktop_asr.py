"""Opt-in offscreen Qt/GPU acceptance for automatic transcription.

Imports a folder of recordings through the application's own task path with each backend as
the main model, scores the unsplit clips against reference transcripts, then preprocesses and
trains a few steps on the default (main + cross-check) project.

    python -m tests.desktop_asr --audio <recordings> --data-dir outputs/asr-acceptance
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from irodori_tts.desktop import asr
from irodori_tts.desktop.app import MainWindow
from irodori_tts.desktop.resources import configure_environment
from irodori_tts.desktop.storage import (
    AppStore,
    create_project,
    load_project,
    read_json,
    write_json,
)
from tests.qt_cleanup import cleanup_qt
from tests.run import guarded_main

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "outputs/asr-bench/ref/reference.json"
# In-app acceptance numbers (whole pipeline: splitting, then segments joined per long file, all 103
# reference clips): text CER, reading CER in percent. The unsplit-clip comparison uses the
# benchmark transcripts in outputs/asr-bench/results for the same clips.
BENCHMARK = {
    "whisper-large-v3": (7.45, 2.98),
    "qwen3-asr": (8.01, 3.62),
    "anime-whisper": (16.80, 3.76),
    "fun-asr-nano": (10.29, 4.31),
}
RESULT_FILES = {
    "anime-whisper": "anime-whisper",
    "qwen3-asr": "qwen3-asr-1.7b",
    "whisper-large-v3": "whisper-large-v3",
    "fun-asr-nano": "fun-asr-nano",
}
TOLERANCE = 1.0


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def rate(pairs):
    errors = sum(distance(ref, hyp) for ref, hyp in pairs)
    return 100 * errors / max(1, sum(len(ref) for ref, _hyp in pairs))


def cer(pairs):
    """(text CER, reading CER) in percent for (reference, hypothesis) pairs."""
    return (
        rate([(asr.surface(ref), asr.surface(hyp)) for ref, hyp in pairs]),
        rate([(asr.reading(ref), asr.reading(hyp)) for ref, hyp in pairs]),
    )


def project_texts(items):
    """(unsplit transcripts by file, transcripts of long files joined from their segments)."""
    unsplit = {i["file_name"]: i["text"] for i in items if not i.get("segment")}
    parts = {}
    for item in items:
        if item.get("segment"):
            parts.setdefault(item["segment"]["file"], []).append(item)
    joined = {
        name: "".join(i["text"] for i in sorted(group, key=lambda i: i["segment"]["start"]))
        for name, group in parts.items()
    }
    return unsplit, joined


def benchmark_texts(key):
    path = REPO / "outputs/asr-bench/results" / f"{RESULT_FILES[key]}.json"
    return json.loads(path.read_text(encoding="utf-8"))["texts"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--audio", type=Path, required=True, help="folder of the reference recordings"
    )
    parser.add_argument("--only", nargs="*", help="run only these backends as the main model")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    if data.exists():
        raise ValueError("Use a new data directory to preserve earlier evidence")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    configure_environment(REPO)
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    store = AppStore(data)
    store.settings.update(language="en", theme="dark", reduce_motion=True)
    window = MainWindow(store)
    window.resize(1520, 920)
    window.show()
    window.navigate(2)
    manager = window.tasks
    finished = []
    manager.finished.connect(lambda task: finished.append(dict(task)))

    def wait_until(condition, timeout=3600):
        deadline = time.monotonic() + timeout
        while not condition():
            assert time.monotonic() < deadline, f"Timed out: {manager.current}"
            # QTest.qWait keeps the GIL, which starves the application's pool threads (the
            # run is created on one before training starts); a QEventLoop releases it.
            loop = QEventLoop()
            QTimer.singleShot(50, loop.quit)
            loop.exec()

    def run_import(main_key, check_key):
        project = create_project(data / "projects", f"ASR {main_key}")
        store.register(project)
        window.refresh()
        window.training.set_project(project)
        before = len(finished)
        started = time.monotonic()
        window.training.start_transcription(project, args.audio, main_key, check_key, True)
        wait_until(
            lambda: (
                not manager.busy
                and len(finished) > before
                and (finished[-1]["kind"] == "transcribe" or finished[-1]["status"] != "completed")
            )
        )
        task = finished[-1]
        assert task["kind"] == "transcribe" and task["status"] == "completed", task
        seconds = time.monotonic() - started
        return project, task, seconds

    order = [b.key for b in asr.BACKENDS if not args.only or b.key in args.only]
    report = {"backends": {}, "benchmark": BENCHMARK, "tolerance_points": TOLERANCE}
    failures = []
    default_project = None
    for key in order:
        check = asr.DEFAULT_CHECK if key == asr.DEFAULT_MAIN else None
        project, task, seconds = run_import(key, check)
        items = load_project(project)["materials"]
        unsplit, joined = project_texts(items)
        bench = benchmark_texts(key)
        # 103 reference clips: 85 stay whole, 18 are longer than 12 s and are cut into segments.
        whole = [name for name in reference if name in unsplit]
        both = [name for name in reference if name in unsplit or name in joined]
        mine = {**unsplit, **joined}
        results = {
            "unsplit_clips": (
                len(whole),
                cer([(reference[n], unsplit[n]) for n in whole]),
                cer([(reference[n], bench[n]) for n in whole]),
            ),
            "all_clips_segments_joined": (
                len(both),
                cer([(reference[n], mine[n]) for n in both]),
                BENCHMARK[key],
            ),
        }
        row = {
            "check": check,
            "seconds": round(seconds, 1),
            "recordings": len(items),
            "result": {k: v for k, v in task["result"].items() if k != "ids"},
        }
        for name, (count, got, expected) in results.items():
            row[name] = {
                "clips": count,
                "text_cer": round(got[0], 2),
                "reading_cer": round(got[1], 2),
                "benchmark_text_cer": round(expected[0], 2),
                "benchmark_reading_cer": round(expected[1], 2),
            }
            for label_, value, wanted in (
                ("text CER", got[0], expected[0]),
                ("reading CER", got[1], expected[1]),
            ):
                if abs(value - wanted) > TOLERANCE:
                    failures.append(f"{key} {name} {label_} {value:.2f} vs benchmark {wanted:.2f}")
        log = (manager.task_dir(task) / "worker.log").read_text(encoding="utf-8", errors="replace")
        row["peak_vram_gb"] = {
            name: float(gb) for name, gb in re.findall(r"\[asr\] (\S+) peak VRAM ([0-9.]+) GB", log)
        }
        report["backends"][key] = row
        print(json.dumps({key: row}, ensure_ascii=False), flush=True)
        if key == asr.DEFAULT_MAIN:
            default_project = project
            segments = [i for i in items if i.get("segment")]
            sources = {i["segment"]["file"] for i in segments}
            lengths = [i["duration"] for i in segments]
            report["segments"] = {
                "long_files": len(sources),
                "segments": len(segments),
                "min_seconds": round(min(lengths), 2),
                "max_seconds": round(max(lengths), 2),
                "all_at_most_12s": max(lengths) <= 12.0 + 1e-6,
            }
            flagged = [i for i in items if (i.get("asr") or {}).get("needs_review")]
            in_reference = [i for i in flagged if i["file_name"] in reference]
            report["needs_review"] = {
                "all_recordings": len(flagged),
                "reference_clips": len(in_reference),
                "recordings": len(items),
            }
            print(json.dumps({"segments": report["segments"]}), flush=True)
            print(json.dumps({"needs_review": report["needs_review"]}), flush=True)
    assert not failures, failures

    if default_project is not None and not args.skip_training:
        window.training.set_project(default_project)
        window.training.tabs.setCurrentIndex(1)
        before = len(finished)
        window.training.initialize()
        wait_until(lambda: not manager.busy and len(finished) > before)
        assert finished[-1]["kind"] == "initialize" and finished[-1]["status"] == "completed", (
            finished[-1]
        )
        window.training.tabs.setCurrentIndex(2)
        for name, value in {
            "max_steps": 4,
            "warmup_steps": 1,
            "save_every": 2,
            "valid_every": 0,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True,
        }.items():
            window.training.parameters[name].setValue(value)
        window.training.voice_name.setText("ASR acceptance")
        before = len(finished)
        window.training.start_training()
        # The run is created on a pool thread first; a refused start leaves the task idle.
        wait_until(lambda: manager.busy, timeout=60)
        wait_until(lambda: not manager.busy and len(finished) > before)
        training = next(t for t in finished[before:] if t["kind"] == "train")
        assert training["status"] == "completed" and training["result"]["step"] == 4, training
        report["preprocess_and_train"] = {
            "initialized": len(load_project(default_project)["materials"]),
            "training_steps": training["result"]["step"],
            "run": training["run"],
        }
        run = read_json(Path(training["run"]) / "run.json")
        assert run["status"] == "completed" and run["step"] == 4
        print("Default-pair project preprocessed and trained 4 steps", flush=True)

    write_json(data / "result.json", report)
    print("ASR ACCEPTANCE PASSED", flush=True)
    window.grab().save(str(data / "training-dataset.png"))
    cleanup_qt()


if __name__ == "__main__":
    raise SystemExit(guarded_main(main, timeout=7200))
