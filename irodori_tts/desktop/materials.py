from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import shutil
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf

from irodori_tts.desktop.messages import BusinessError, fmt, t

from .storage import digest, identity, load_project, project_lock, save_project, within

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}
# Transcripts come from many editors: Excel on Japanese Windows saves CSV as cp932,
# Chinese systems often use GB18030. UTF-8 (with or without BOM) is tried first.
TEXT_ENCODINGS = ("utf-8-sig", "cp932", "gb18030")
MAX_TEXT_LENGTH = 500


def inspect_audio(path: Path) -> tuple[float, str]:
    try:
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        duration = len(data) / rate
        if duration <= 0 or not np.isfinite(data).all() or np.max(np.abs(data)) < 1e-6:
            return duration, t("materials.recording_is_empty_silent_or_contains_invalid")
        if duration > 12:
            return duration, t("materials.recording_exceeds_12_seconds_split_the_audio")
        if duration < 0.3:
            return duration, t("materials.recording_is_shorter_than_0_3_seconds")
        return duration, ""
    except Exception as exc:
        return 0, t("materials.cannot_read_the_recording_replace_it", p1=fmt(exc, "", -1))


def clean_field(value) -> str:
    """Transcript and caption values as stored: stripped text, never None."""
    return value.strip() if isinstance(value, str) else ""


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise BusinessError(t("materials.text_encoding_unreadable", file=fmt(path.name, "", -1)))


def material_problem(item: dict):
    """Why a recording cannot be trained on as it is, or None."""
    if item.get("error"):
        return item["error"]
    text = clean_field(item.get("text"))
    if not text:
        return t("training_page.transcript_missing")
    if len(text) > MAX_TEXT_LENGTH:
        return t("materials.transcript_too_long")
    return None


def needs_review(item: dict) -> bool:
    """The automatic transcript disagreed with its cross-check and nobody has looked yet."""
    return bool((item.get("asr") or {}).get("needs_review"))


def materials_fingerprint(project: dict) -> str:
    """Identity of the training data: included recordings' content, transcripts and captions.
    Unlike the revision counter, undoing an edit restores the same fingerprint."""
    entries = sorted(
        (
            item["id"],
            item.get("sha256") or "",
            clean_field(item.get("text")),
            clean_field(item.get("caption")),
        )
        for item in project["materials"]
        if item.get("included")
    )
    return hashlib.sha256(json.dumps(entries, ensure_ascii=False).encode("utf-8")).hexdigest()


def _unique_name(relative: str, taken: set[str]) -> str:
    if relative not in taken:
        return relative
    path = PurePosixPath(relative)
    for number in itertools.count(2):
        candidate = path.with_name(f"{path.stem} ({number}){path.suffix}").as_posix()
        if candidate not in taken:
            return candidate
    raise AssertionError("unreachable")


def import_materials(root: Path, folder: Path, csv_path: Path | None = None) -> list[dict]:
    """Copy recordings into the project. Recordings that cannot be trained on are excluded;
    a re-import keeps transcripts and choices the new import does not provide."""
    with project_lock(root):
        project = load_project(root)
        mapping = {}
        if csv_path:
            reader = csv.DictReader(io.StringIO(read_text(csv_path), newline=""))
            if not {"file_name", "text"}.issubset(reader.fieldnames or []):
                raise BusinessError(t("materials.csv_requires_file_name_and_text_columns"))
            for row in reader:
                filename = clean_field(row.get("file_name")).replace("\\", "/")
                if not filename:
                    continue
                if filename in mapping:
                    raise BusinessError(
                        t(
                            "materials.duplicate_csv_association_keep_the_correct_entry",
                            p1=fmt(filename, "", -1),
                        )
                    )
                mapping[filename] = row
        imported = []
        changed = False
        taken = {item["file_name"] for item in project["materials"]}
        candidates = sorted(
            p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES and p.is_file()
        )
        for original in candidates:
            relative = original.relative_to(folder).as_posix()
            source = str(original.resolve())
            if csv_path:
                row = mapping.get(relative, {})
                text, caption = clean_field(row.get("text")), clean_field(row.get("caption"))
            else:
                text_path = original.with_suffix(".txt")
                text = clean_field(read_text(text_path)) if text_path.exists() else ""
                caption = ""
            checksum = digest(original)
            # A segment cut from a long file is a recording of its own, never this file.
            whole = [item for item in project["materials"] if not item.get("segment")]
            existing = next((item for item in whole if item.get("source") == source), None) or next(
                (
                    item
                    for item in whole
                    if item["file_name"] == relative and item.get("sha256") == checksum
                ),
                None,
            )
            if existing:
                item_id, file_name = existing["id"], existing["file_name"]
            else:
                # A same-named file from another folder is a different recording: keep both.
                item_id, file_name = identity(), _unique_name(relative, taken)
                taken.add(file_name)
            destination = within(root, f"source/{item_id}{original.suffix.lower()}")
            path = destination.relative_to(root).as_posix()
            # Fields this import does not decide (such as the source check cache) carry over.
            base = {key: value for key, value in (existing or {}).items() if key != "auto_excluded"}
            if (
                not existing
                or existing.get("sha256") != checksum
                or existing["path"] != path
                or not destination.is_file()
            ):
                base.pop("stat", None)
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                shutil.copyfile(original, temporary)
                temporary.replace(destination)
                checksum = digest(destination)
                if existing and existing["path"] != path:
                    previous = within(root, existing["path"])
                    if previous.is_file():
                        previous.unlink()
            duration, error = inspect_audio(destination)
            item = {
                **base,
                "id": item_id,
                "file_name": file_name,
                "path": path,
                "source": source,
                "text": text or clean_field((existing or {}).get("text")),
                "caption": caption or clean_field((existing or {}).get("caption")),
                "included": True,
                "duration": duration,
                "error": error,
                "sha256": checksum,
            }
            problem = material_problem(item)
            wanted = not existing or existing["included"] or existing.get("auto_excluded")
            item["included"] = bool(wanted) and problem is None
            if wanted and problem is not None:
                # Remember that the app, not the user, excluded it, so a later import
                # that fixes the problem includes it again.
                item["auto_excluded"] = True
            if existing:
                if existing != item:
                    project["materials"][project["materials"].index(existing)] = item
                    changed = True
            else:
                project["materials"].append(item)
                changed = True
            imported.append(item)
        if not imported:
            raise BusinessError(t("materials.no_supported_recordings_in_the_selected_folder"))
        if changed:
            project["revision"] += 1
            save_project(root, project)
        return imported


def import_transcribed(root: Path, entries: list[dict]) -> list[dict]:
    """Add recordings whose transcripts came from speech recognition. Each entry names the
    audio to place (`audio`, moved when `move` else copied), its `source` file and optional
    `segment` (time range in that file), the text and the `asr` record. A recording the project
    already has (`existing`) is updated in place. Recordings without text are excluded."""
    with project_lock(root):
        project = load_project(root)
        taken = {item["file_name"] for item in project["materials"]}
        imported = []
        for entry in entries:
            existing = next(
                (item for item in project["materials"] if item["id"] == entry.get("existing")),
                None,
            )
            audio = Path(entry["audio"])
            if existing:
                item_id, file_name = existing["id"], existing["file_name"]
            else:
                item_id, file_name = identity(), _unique_name(entry["name"], taken)
                taken.add(file_name)
            destination = within(root, f"source/{item_id}{audio.suffix.lower()}")
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(audio, temporary)
            temporary.replace(destination)
            if entry.get("move"):
                audio.unlink(missing_ok=True)
            path = destination.relative_to(root).as_posix()
            if existing and existing["path"] != path:
                previous = within(root, existing["path"])
                if previous.is_file():
                    previous.unlink()
            duration, error = inspect_audio(destination)
            item = {
                **{k: v for k, v in (existing or {}).items() if k not in {"auto_excluded", "stat"}},
                "id": item_id,
                "file_name": file_name,
                "path": path,
                "source": entry["source"],
                "text": clean_field(entry["text"]),
                "caption": clean_field((existing or {}).get("caption")),
                "included": True,
                "duration": duration,
                "error": error,
                "sha256": digest(destination),
                "asr": entry["asr"],
            }
            if entry.get("segment"):
                item["segment"] = entry["segment"]
            problem = material_problem(item)
            wanted = not existing or existing["included"] or existing.get("auto_excluded")
            item["included"] = bool(wanted) and problem is None
            if wanted and problem is not None:
                item["auto_excluded"] = True
            if existing:
                project["materials"][project["materials"].index(existing)] = item
            else:
                project["materials"].append(item)
            imported.append(item)
        if imported:
            project["revision"] += 1
            save_project(root, project)
        return imported


def remove_materials(root: Path, item_ids) -> int:
    """Take recordings out of the project, deleting the project's copies of their audio (the
    files they were imported from stay where they are). Returns how many were removed. The
    revision changes, so the training data must be prepared again."""
    wanted = set(item_ids)
    with project_lock(root):
        project = load_project(root)
        removed = [item for item in project["materials"] if item["id"] in wanted]
        if not removed:
            return 0
        project["materials"] = [item for item in project["materials"] if item["id"] not in wanted]
        project["revision"] += 1
        save_project(root, project)
    for item in removed:
        within(root, item["path"]).unlink(missing_ok=True)
    return len(removed)


def mark_reviewed(root: Path, item_ids: list[str]) -> int:
    """Clear the needs-review flag without changing anything else; returns how many it cleared."""
    with project_lock(root):
        project = load_project(root)
        cleared = 0
        for item in project["materials"]:
            if item["id"] in item_ids and needs_review(item):
                item["asr"] = {**item["asr"], "needs_review": False}
                cleared += 1
        if cleared:
            # Review state is not training data: the revision (and so the initialization) stays.
            save_project(root, project)
        return cleared


def edit_material(
    root: Path,
    item_id: str,
    *,
    text: str,
    caption: str,
    included: bool,
    replacement: Path | None = None,
    reviewed: bool = False,
) -> None:
    """Save a recording's fields. A transcript flagged "needs review" is cleared by changing the
    text, or by `reviewed` (the user pressed Save having read it)."""
    with project_lock(root):
        project = load_project(root)
        item = next(item for item in project["materials"] if item["id"] == item_id)
        text, caption = clean_field(text), clean_field(caption)
        changed = (
            clean_field(item.get("text")) != text
            or clean_field(item.get("caption")) != caption
            or item["included"] != included
        )
        if replacement and digest(replacement) != item.get("sha256"):
            destination = within(root, item["path"])
            temporary = destination.with_suffix(".tmp")
            shutil.copyfile(replacement, temporary)
            temporary.replace(destination)
            item["source"] = str(replacement.resolve())
            item["sha256"] = digest(destination)
            changed = True
        item["duration"], item["error"] = inspect_audio(within(root, item["path"]))
        cleared = needs_review(item) and (reviewed or clean_field(item.get("text")) != text)
        if cleared:
            item["asr"] = {**item["asr"], "needs_review": False}
        automatic = item.get("auto_excluded") and item["included"] == included
        if item["included"] != included:
            item.pop("auto_excluded", None)
        item.update(text=text, caption=caption, included=included)
        if automatic and material_problem(item) is None:
            # The app excluded it for a problem this edit fixed; the user's choice was never asked.
            item.pop("auto_excluded")
            item["included"] = True
            changed = True
        if changed:
            project["revision"] += 1
        if changed or cleared:
            save_project(root, project)


def material_issues(root: Path, items: list[dict]) -> list[tuple[dict, object]]:
    """Every included recording that cannot be preprocessed as it is, with the reason; the
    files are read again, since they may have changed since they were imported."""
    issues = []
    for item in items:
        _, error = inspect_audio(within(root, item["path"]))
        text = clean_field(item.get("text"))
        if error:
            issues.append((item, error))
        elif not text:
            issues.append((item, t("training_page.transcript_missing")))
        elif len(text) > MAX_TEXT_LENGTH:
            issues.append((item, t("materials.transcript_too_long")))
    return issues


class MaterialProblems(BusinessError):
    """Included recordings that stop preprocessing, all of them at once (not the first one)."""

    def __init__(self, issues):
        first, reason = issues[0]
        super().__init__(
            t(
                "materials.preprocess_blocked",
                n=len(issues),
                file=fmt(first["file_name"], "", -1),
                reason=fmt(reason, "", -1),
            )
        )
        self.phase = "check"
        self.items = [{"id": item["id"], "reason": reason} for item, reason in issues]


def validate_materials(root: Path, project: dict) -> list[dict]:
    items = [item for item in project["materials"] if item["included"]]
    if not items:
        raise BusinessError(t("materials.import_and_include_at_least_one_recording"))
    if issues := material_issues(root, items):
        raise MaterialProblems(issues)
    return items
