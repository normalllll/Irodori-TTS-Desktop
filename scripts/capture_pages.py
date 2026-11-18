"""Offscreen screenshots of every page, every step of a model and an imported model at one
or more window sizes, with layout checks.

Usage:
    python -m scripts.capture_pages OUT_DIR [--size 1520x920 ...] [--language zh_CN]
        [--theme light] [--long-names]

--size may be repeated; --size all takes the sizes in SIZES. Seeds a throwaway data
directory with the bundled example project, so it needs no GPU or downloaded model.
--long-names gives the project, models, runs and recordings long names.

Exits non-zero when a page scrolls sideways, scrolls vertically where it must not (see
SCROLLING), cuts off a button's or a single-line label's text, squeezes a table's main
column below its minimum, or leaves a primary button out of view.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The minimum window, a 1080p screen at 150% scaling, the old default and a full 1080p screen.
SIZES = ("1024x640", "1280x688", "1520x920", "1920x1080")
# Pages that read top to bottom (forms, checklists) may scroll at any size; the others only
# when the window is shorter than this. Lists, tables and editors always scroll inside.
SCROLLING = {"settings", "guide"}
SCROLL_BELOW = 680
NAMES = {
    "project": "Demo",
    "failed": "Second voice",
    "run": "my-lora",
    "imported": "studio-female",
    "clip": "sample_{:02d}.wav",
    "base": "my-base",
}
LONG_NAMES = {
    "project": "Hana 学校の放課後ボイス 2026 夏 完全版 (whisper 校対済み)",
    "failed": "もうひとつの声 — 録音し直し前の素材 (2026 秋, 未整理)",
    "run": "hana-summer-full-v3-lr1e4-rank32-final-really-final",
    "imported": "imported-studio-female-voice-from-another-computer-2026-10",
    "clip": "hana_after_school_line_{:02d}_take3_denoised_48k.wav",
    "base": "Irodori-TTS v4 Large community fine-tune for expressive narration (2026-11)",
}


def seed_bases(data, names):
    """Register the default catalog base and an imported base, as header-only checkpoints:
    the pages read a base's architecture from its header and never its weights. Returns
    their base records."""
    import json
    import struct

    from irodori_tts.desktop.base_models import (
        LARGE_ARCHITECTURE,
        InstalledBase,
        base_record,
        default_catalog_base,
        save_base,
    )

    root = data / "resources"
    small = default_catalog_base()
    records = []
    for base_id, name, source, architecture in (
        (small.id, small.name, "catalog", small.architecture),
        ("imported-0123456789ab", names["base"], "imported", LARGE_ARCHITECTURE),
    ):
        checkpoint = root / "bases" / base_id / "model.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        header = json.dumps({"__metadata__": {"config_json": json.dumps(architecture)}}).encode()
        checkpoint.write_bytes(struct.pack("<Q", len(header)) + header)
        base = InstalledBase(
            id=base_id,
            name=name,
            source=source,
            repo=small.repo if source == "catalog" else None,
            revision="0" * 40 if source == "catalog" else None,
            checkpoint=checkpoint,
            architecture=dict(architecture),
        )
        save_base(root, base)
        records.append(base_record(base))
    return records


def seed_training(root, names, base):
    """Mark the project preprocessed and add a finished and an interrupted run with checkpoints."""
    from irodori_tts.desktop.storage import identity, load_project, now, save_project, write_json

    project = load_project(root)
    init = {
        "id": identity(),
        "revision": project["revision"],
        "status": "completed",
        "count": len(project["materials"]),
        "created": now(),
    }
    project["initialization"] = init
    save_project(root, project)
    for name, status, step, saved in (
        ("first-try", "interrupted", 250, (250,)),
        (names["run"], "completed", 1000, (250, 500, 1000)),
    ):
        run_id = identity()
        run_root = root / "runs" / run_id
        for saved_step in saved:
            checkpoint = run_root / "checkpoints" / f"checkpoint_{saved_step:07d}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_config.json").write_text("{}", "utf-8")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"x")
            write_json(
                checkpoint / "complete.json",
                {
                    "step": saved_step,
                    "files": {"adapter_config.json": 2, "adapter_model.safetensors": 1},
                },
            )
        write_json(
            run_root / "run.json",
            {
                "id": run_id,
                "name": name,
                "created": now(),
                "initialization_id": init["id"],
                "revision": init["revision"],
                "status": status,
                "step": step,
                "checkpoint": f"runs/{run_id}/checkpoints/checkpoint_{saved[-1]:07d}",
                "registered": status == "completed",
                "base": base,
                "config": {"train": {"max_steps": 1000, "save_every": 250, "learning_rate": 1e-4}},
            },
        )
        # The finished run's last checkpoint has a sample: its text and player show.
        if status == "completed":
            sample = run_root / "samples" / identity()
            sample.mkdir(parents=True)
            shutil.copy(
                Path(__file__).resolve().parents[1] / "examples/desktop/basic.wav",
                sample / "audio.wav",
            )
            write_json(
                sample / "generation.json",
                {
                    "audio": "audio.wav",
                    "request": {
                        "text": "こんにちは。今日はとてもいい天気ですね。ゆっくりお話ししましょう。",
                        "lora_adapter": str(checkpoint),
                    },
                },
            )


def seed_transcripts(data, root, names):
    """Automatically transcribed recordings, some flagged for review, plus installed add-ons."""
    import json

    from irodori_tts.desktop import asr
    from irodori_tts.desktop.materials import import_transcribed

    clips = REPO / "examples/desktop"
    lines = (
        (
            "やっと着いたね。ここが噂の駅前のカフェ？",
            "やっとついたね。ここがうわさのえきまえのカフェ？",
            True,
        ),
        ("先生、今日はよろしくお願いします！", "先生、今日はよろしくお願いします。", False),
        ("ふふっ、びっくりした？", "ふふ、ビックリした？", False),
        ("それじゃあ、行こうか。", "それじゃあ、いこっか。", True),
        ("ありがとう、助かったよ。", "ありがとう、助かったよ", False),
        ("もう少しだけ、待ってて。", "もうすこしだけ、まってて。", False),
    )
    entries = []
    for number, (text, heard, _flag) in enumerate(lines, 1):
        entries.append(
            {
                "name": names["clip"].format(number),
                "audio": clips / ("basic.wav" if number % 2 else "voice_design.wav"),
                "move": False,
                "source": str(clips / "basic.wav"),
                "text": text,
                "asr": {
                    "backend": "qwen3-asr",
                    "check_backend": "whisper-large-v3",
                    "check_text": heard,
                    "needs_review": asr.readings_differ(text, heard),
                },
            }
        )
    import_transcribed(root, entries)
    cache = Path(os.environ["IRODORI_CACHE"]) / "addons"
    for key in ("qwen3-asr", "funasr"):
        folder = cache / key
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "addon.json").write_text(
            json.dumps({"packages": list(asr.ADDONS[key].packages), "python": asr.python_tag()}),
            encoding="utf-8",
        )
        with (folder / "payload.bin").open("wb") as stream:
            stream.truncate(asr.ADDONS[key].size)


def seed_failed(data, names):
    """A second project whose last preprocessing failed at one recording."""
    from irodori_tts.desktop.materials import import_materials
    from irodori_tts.desktop.storage import create_project, load_project, now, save_project

    root = create_project(data / "projects", names["failed"])
    examples = REPO / "examples/desktop"
    import_materials(root, examples, examples / "transcripts.csv")
    project = load_project(root)
    failed = project["materials"][0]
    project["last_attempt"] = {
        "status": "failed",
        "error": f"Encoding {failed['file_name']} failed: CUDA error: out of memory. Tried to "
        "allocate 2.00 GiB; 1.12 GiB free of 8.00 GiB.",
        "phase": "encode",
        "item": failed["id"],
        "items": [],
        "created": now(),
    }
    save_project(root, project)
    return root


def seed_tasks(data, root, names):
    """A few finished tasks, one of them failed, for the settings page and the status bar."""
    from irodori_tts.desktop.storage import identity, now, write_json

    for kind, status, message, extra, request in (
        ("resources", "completed", "Task complete", {}, {}),
        ("generate", "completed", "Task complete", {}, {"voice": names["run"]}),
        (
            "initialize",
            "failed",
            "Recording is empty, silent or contains invalid values.",
            {"project": str(root)},
            {},
        ),
        ("initialize", "completed", "Task complete", {"project": str(root)}, {}),
    ):
        task_id = identity()
        directory = data / "tasks" / task_id
        directory.mkdir(parents=True)
        write_json(
            directory / "task.json",
            {
                "id": task_id,
                "kind": kind,
                "status": status,
                "created": now(),
                "step": 0,
                "total": 0,
                "message": message,
                "sequence": 1,
                "project": None,
                "run": None,
                **extra,
            },
        )
        write_json(directory / "request.json", {"kind": kind, **request, **extra})


def layout_problems(page, name, height):
    """What is wrong with the layout of the page on screen, one line each."""
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtWidgets import QAbstractButton, QLabel, QScrollArea
    from qfluentwidgets import PrimaryPushButton

    from irodori_tts.desktop.widgets import ElidedComboBox, ElidedLabel, ErrorLabel, Page, Table

    problems = []
    shells = [shell for shell in (page, *page.findChildren(Page)) if isinstance(shell, Page)]
    for shell in shells:
        if not shell.isVisible():
            continue
        if overflow := shell.horizontalScrollBar().maximum():
            problems.append(f"{name}: scrolls sideways by {overflow}px")
        overflow = shell.verticalScrollBar().maximum()
        if overflow and name not in SCROLLING and height >= SCROLL_BELOW:
            problems.append(f"{name}: whole-page scroll of {overflow}px")

    def cut_off(widget):
        return widget.isVisible() and widget.sizeHint().width() > widget.width() + 1

    for widget in page.findChildren(QAbstractButton):
        # Elided drop-downs shorten their item on purpose and show it whole in the tooltip.
        if isinstance(widget, ElidedComboBox):
            continue
        if (text := widget.text().strip()) and cut_off(widget):
            problems.append(f"{name}: cut-off text in {type(widget).__name__}: {text!r}")
    for widget in page.findChildren(QLabel):
        if (
            (text := widget.text().strip())
            and not widget.wordWrap()
            and not isinstance(widget, (ElidedLabel, ErrorLabel))
            and cut_off(widget)
        ):
            problems.append(f"{name}: cut-off text in {type(widget).__name__}: {text!r}")
    for widget in page.findChildren(QLabel):
        if (
            widget.isVisible()
            and widget.wordWrap()
            and widget.text().strip()
            and widget.height() < widget.heightForWidth(widget.width()) - 1
        ):
            problems.append(f"{name}: wrapped text squeezed: {widget.text().strip()[:40]!r}")
    for area in page.findChildren(QScrollArea):
        content = area.widget()
        if (
            area.isVisible()
            and content is not None
            and area.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            and content.width() > area.viewport().width() + 1
        ):
            overflow = content.width() - area.viewport().width()
            problems.append(f"{name}: {type(content).__name__} cut off sideways by {overflow}px")
    for table in page.findChildren(Table):
        if table.isVisible() and (column := table.squeezed()):
            problems.append(f"{name}: table column {column!r} squeezed below its minimum")
    if height >= SCROLL_BELOW:
        for widget in page.findChildren(PrimaryPushButton):
            shell = next((s for s in shells if s.isAncestorOf(widget)), None)
            if not widget.isVisible() or shell is None:
                continue
            view = shell.viewport()
            seen = QRect(view.mapToGlobal(view.rect().topLeft()), view.size())
            spot = QRect(widget.mapToGlobal(widget.rect().topLeft()), widget.size())
            if not seen.contains(spot):
                problems.append(f"{name}: primary button {widget.text()!r} is out of view")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--size", action="append")
    parser.add_argument("--language", default="zh_CN", choices=("zh_CN", "en"))
    parser.add_argument("--theme", default="system", choices=("system", "light", "dark"))
    parser.add_argument("--long-names", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    sizes = args.size or ["1520x920"]
    if "all" in sizes:
        sizes = list(SIZES)
    sizes = [tuple(int(v) for v in size.split("x")) for size in sizes]
    names = LONG_NAMES if args.long_names else NAMES
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    # Add-ons for the settings page live under a throwaway cache, never the real one.
    os.environ["IRODORI_CACHE"] = str(args.out / "cache")

    from PySide6.QtCore import QThreadPool
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from irodori_tts.desktop.resources import configure_environment

    configure_environment(REPO)
    app = QApplication([])
    from irodori_tts.desktop.app import MainWindow
    from irodori_tts.desktop.asr_dialog import TranscribeDialog
    from irodori_tts.desktop.materials import import_materials
    from irodori_tts.desktop.storage import AppStore, create_project, identity, now, write_json

    args.out.mkdir(parents=True, exist_ok=True)
    data = args.out / "data"
    shutil.rmtree(data, ignore_errors=True)
    store = AppStore(data)
    store.settings.update(language=args.language, theme=args.theme, reduce_motion=True)
    root = create_project(data / "projects", names["project"])
    examples = REPO / "examples/desktop"
    import_materials(root, examples, examples / "transcripts.csv")
    seed_transcripts(data, root, names)
    bases = seed_bases(data, names)
    seed_training(root, names, bases[1])
    seed_tasks(data, root, names)
    store.register(seed_failed(data, names))
    store.register(root)
    base = "基础模型" if args.language == "zh_CN" else "Base model"
    for text, seconds in (
        ("おはようございます。今日はとてもいい天気ですね。少し散歩してから、駅前のカフェで…", 4.86),
        ("駅前のカフェで待ち合わせしましょう。", 2.4),
        ("本日はご来場いただき、誠にありがとうございます。", 3.62),
    ):
        generation = data / "generations" / identity()
        generation.mkdir(parents=True)
        shutil.copyfile(examples / "basic.wav", generation / "audio.wav")
        request = {"text": text, "num_steps": 40, "seed": 42, "no_ref": True}
        write_json(
            generation / "generation.json",
            {
                "id": generation.name,
                "created": now(),
                "request": request,
                "voice": names["run"] if args.long_names else base,
                "audio": str(generation / "audio.wav"),
                "duration": seconds,
                "seed": 42,
            },
        )

    imported = data / "voices" / identity()
    imported.mkdir(parents=True)
    write_json(
        imported / "voice.json",
        {
            "name": names["imported"],
            "step": 2000,
            "created": now(),
            "registered": True,
            # Trained on a base no one here has installed: the page names what it needs.
            "base": {
                **bases[1],
                "id": "imported-ffffffffffff",
                "architecture": {**bases[1]["architecture"], "num_layers": 30},
            },
        },
    )

    window = MainWindow(store)
    window.resize(*sizes[0])
    window.stack.setAnimationEnabled(False)
    window.show()
    window.generation.add_reference_files([examples / "basic.wav", examples / "voice_design.wav"])
    window.generation.history.setCurrentIndex(window.generation.history.model().index(0, 0))
    window.training.materials.setCurrentIndex(window.training.materials.model().index(0, 0))
    window.training.runs.setCurrentIndex(window.training.runs.model().index(0, 0))

    # Without the update check of the installed bases, which would reach the network.
    window.settings_page.latest_checked = True
    window.settings_page.scan_storage()
    window.settings_page.diagnose()
    QThreadPool.globalInstance().waitForDone(30000)

    def settle():
        # Page switches, row animations and the navigation's collapse all finish first, so a
        # screenshot never shows the previous page.
        for _ in range(3):
            QTest.qWait(100)
            app.processEvents()

    pages = ("guide", "generation", "models", "settings")
    project_row = f"project:{root.resolve()}"
    window.models.list.select_id(project_row)
    rows = window.training.materials.rows_model.rows
    flagged = next(i for i, row in enumerate(rows) if (row.get("asr") or {}).get("needs_review"))
    failures = []
    count = 0
    for width, height in sizes:
        window.resize(width, height)
        settle()
        size = f"{window.width()}x{window.height()}"
        if size != f"{width}x{height}":
            failures.append(f"window could not take {width}x{height}: it is {size}")
        shots = [(name, index, None) for index, name in enumerate(pages)]
        shots += [(f"model_step{tab}", 2, tab) for tab in range(window.training.tabs.count())]
        shots += [
            ("model_review", 2, "review"),
            ("model_new_run", 2, "new_run"),
            ("model_failed", 2, "failed"),
            ("model_imported", 2, "imported"),
        ]
        for name, index, tab in shots:
            window.models.list.select_id(project_row)
            if index == 0:
                # Without the connection check, which would reach the network.
                window.switchTo(window.guide)
                window.guide.refresh(force=True)
            else:
                window.navigate(index)
            page = window.pages[index]
            if tab == "new_run":
                window.training.tabs.setCurrentIndex(2)
                window.training.new_run()
            elif tab == "failed":
                failed = next(
                    row["id"]
                    for row in window.models.list.rows_model.rows
                    if row.get("kind") == "project" and row["id"] != project_row
                )
                window.models.list.select_id(failed)
                window.training.tabs.setCurrentIndex(1)
            elif tab == "imported":
                imported_row = next(
                    row["id"]
                    for row in window.models.list.rows_model.rows
                    if row.get("kind") == "model"
                )
                window.models.list.select_id(imported_row)
            elif tab == "review":
                window.training.tabs.setCurrentIndex(0)
                window.training.materials.setCurrentIndex(
                    window.training.materials.model().index(flagged, 0)
                )
            elif tab is not None:
                window.training.tabs.setCurrentIndex(tab)
                if tab == 2:
                    window.training.runs.setCurrentIndex(window.training.runs.model().index(0, 0))
            settle()
            failures += [f"{size} {line}" for line in layout_problems(page, name, height)]
            window.grab().save(str(args.out / f"{name}_{size}.png"))
            count += 1
        # The import dialog over the dataset.
        window.navigate(2)
        window.training.tabs.setCurrentIndex(0)
        dialog = TranscribeDialog(window, store.settings, examples)
        dialog.show()
        QTest.qWait(600)
        settle()
        outside = dialog.widget.geometry()
        if not window.rect().contains(outside):
            failures.append(f"{size} import dialog does not fit the window: {outside}")
        for control in (dialog.yesButton, dialog.cancelButton):
            if control.sizeHint().width() > control.width():
                failures.append(f"{size} import dialog button is truncated: {control.text()}")
        window.grab().save(str(args.out / f"import_dialog_{size}.png"))
        count += 1
        dialog.close()
        window.training.materials.setCurrentIndex(window.training.materials.model().index(0, 0))
    window.close()
    for line in failures:
        print(line)
    print(f"{count} screenshots in {args.out}; {len(failures)} layout problems")
    from tests.qt_cleanup import cleanup_qt

    cleanup_qt()
    return 1 if failures else 0


if __name__ == "__main__":
    from tests.run import guarded_main

    sys.exit(guarded_main(main, timeout=600))
