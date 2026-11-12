from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import stat
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from irodori_tts.desktop.messages import BusinessError, fmt, t

from . import ASR_DEFAULT_CHECK, ASR_DEFAULT_MAIN, PROJECT_VERSION
from .messages import persist, restore


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def identity() -> str:
    return uuid.uuid4().hex


def stamp() -> str:
    """A short, readable, sortable name for a folder made in passing (a generation, a sample,
    a task): the local time and four random characters, 20261030-153012-a3f9."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


# Names Windows keeps for devices, whatever the extension.
_DEVICES = {"con", "prn", "aux", "nul", *(f"com{n}" for n in range(1, 10))}
_DEVICES |= {f"lpt{n}" for n in range(1, 10)}


def folder_name(text: str, limit: int = 40) -> str:
    """text as a folder name Windows accepts: forbidden and control characters become _,
    no leading or trailing dots and spaces, at most limit characters (paths stay short)."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(text)).strip(" .")[:limit].strip(" .")
    if not name or name.split(".")[0].lower() in _DEVICES:
        name = f"_{name}"
    return name


def voice_folder(parent: Path, name: str) -> Path:
    """A folder for a voice kept apart, named after it and the time: its name is the voice's
    ID, so a voice made later under the same name never takes an earlier one's place."""
    return free_folder(parent, f"{folder_name(name, 30)}-{datetime.now():%Y%m%d-%H%M%S}")


def free_folder(parent: Path, name: str) -> Path:
    """parent/name, or the first of "name 2", "name 3"... that is not taken."""
    candidate, number = parent / name, 2
    while candidate.exists():
        candidate, number = parent / f"{name} {number}", number + 1
    return candidate


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return restore(json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{identity()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(persist(value), stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def within(root: Path, relative: str | Path) -> Path:
    root = root.absolute()
    path = root / relative
    if not path.absolute().is_relative_to(root) or not path.resolve().is_relative_to(
        root.resolve()
    ):
        raise BusinessError(t("storage.path_escapes_the_project", p1=fmt(relative, "", -1)))
    for part in (root, *path.absolute().relative_to(root).parents):
        candidate = part if part.is_absolute() else root / part
        attributes = (
            getattr(candidate.lstat(), "st_file_attributes", 0) if candidate.exists() else 0
        )
        if candidate.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise BusinessError(
                t(
                    "storage.symbolic_links_and_directory_junctions_are_not",
                    p1=fmt(candidate, "", -1),
                )
            )
    if path.exists():
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        if path.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise BusinessError(t("storage.links_are_not_allowed", p1=fmt(path, "", -1)))
    return path


def removable(root: Path, relative: str | Path) -> Path:
    """A path inside root that may be deleted: never root itself, never through a link."""
    path = within(root, relative)
    if path.absolute() == root.absolute() or path.resolve() == root.resolve():
        raise BusinessError(t("storage.refusing_to_delete_the_root", p1=fmt(root, "", -1)))
    return path


# How long a delete waits for files another part of the application is letting go of: a
# media player releases a sample a moment after it is cleared.
RELEASE_SECONDS = 3.0


def tree_size(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def remove_tree(root: Path, directory: Path) -> None:
    """Delete a directory inside root after checking every entry; files still held open are
    retried for RELEASE_SECONDS, then reported."""
    target = removable(root, directory.relative_to(root))
    if not target.exists():
        return
    for child in target.rglob("*"):
        within(root, child.relative_to(root))
    deadline = time.monotonic() + RELEASE_SECONDS
    failed = []
    while True:
        failed.clear()
        shutil.rmtree(target, onerror=lambda _function, path, _error: failed.append(path))
        if not failed or not target.exists() or time.monotonic() > deadline:
            break
        time.sleep(0.1)
    if failed and target.exists():
        raise BusinessError(
            t(
                "storage.some_files_could_not_be_deleted",
                count=fmt(len(failed), "", -1),
                p1=fmt(failed[0], "", -1),
            )
        )


def clean_project(root: Path, relative: str) -> None:
    if relative not in {"initialized", "temp/initialization"} and not relative.startswith("temp/"):
        raise BusinessError(t("storage.directory_is_not_in_the_cleanup_allowlist"))
    target = removable(root, relative)
    if target.is_file():
        target.unlink()
    elif target.exists():
        remove_tree(root, target)


@contextmanager
def project_lock(root: Path):
    """OS releases the lock on process death; lock metadata is never the authority."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".project.lock"
    with path.open("a+b") as stream:
        stream.seek(0)
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise BusinessError(t("storage.a_project_task_is_running_wait_for")) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def load_project(root: Path) -> dict:
    project = read_json(root / "project.json")
    if not project or project.get("format_version") != PROJECT_VERSION:
        raise BusinessError(t("storage.unsupported_project_format_open_it_with_a"))
    return project


def save_project(root: Path, project: dict) -> None:
    write_json(root / "project.json", project)


def create_project(parent: Path, name: str) -> Path:
    if not name.strip():
        raise BusinessError(t("storage.enter_a_project_name"))
    project_id = identity()
    # The folder carries the name it was created with; the ID stays in project.json.
    parent.mkdir(parents=True, exist_ok=True)
    root = free_folder(parent, folder_name(name))
    root.mkdir()
    for directory in ("source", "runs", "temp"):
        (root / directory).mkdir()
    save_project(
        root,
        {
            "format_version": PROJECT_VERSION,
            "id": project_id,
            "name": name.strip(),
            "created": now(),
            "revision": 0,
            "materials": [],
            "initialization": None,
        },
    )
    return root


def reconcile_sources(root: Path, project: dict) -> bool:
    """Detect recordings changed outside the app. A file whose size and modification time
    match the last check is not hashed again, so routine refreshes stay cheap."""
    changed = False
    touched = False
    for item in project["materials"]:
        path = within(root, item["path"])
        status = path.stat() if path.is_file() else None
        signature = [status.st_size, status.st_mtime_ns] if status else None
        if status and item.get("sha256") and item.get("stat") == signature:
            continue
        current = digest(path) if status else None
        if current != item.get("sha256"):
            item["sha256"] = current
            if current:
                from .materials import inspect_audio

                item["duration"], item["error"] = inspect_audio(path)
            else:
                item["error"] = t("storage.recording_copy_is_missing_replace_the_recording")
            changed = True
        if item.get("stat") != signature:
            item["stat"] = signature
            touched = True
    if changed:
        project["revision"] += 1
    if changed or touched:
        save_project(root, project)
    return changed


THEMES = ("system", "light", "dark")


def default_settings() -> dict:
    return {
        "projects": [],
        "theme": "system",
        # The theme colour: "aqua" or "pink" (widgets.ACCENTS).
        "accent": "aqua",
        "reduce_motion": False,
        "project_root": "projects",
        "ui": {},
        "presets": {},
        # The guided tour ran (or was skipped) once; Settings can start it again.
        "tour_done": False,
        # The folder recordings were last imported from, where the next picker opens.
        "recordings_folder": "",
        # The folder last exported to or imported from, where the next file picker opens.
        "files_folder": "",
        "asr": {"main": ASR_DEFAULT_MAIN, "check": ASR_DEFAULT_CHECK},
    }


def merge_defaults(value, defaults):
    """Fill keys an older settings file lacks; a value of the wrong type takes the default."""
    if not isinstance(value, dict):
        return copy.deepcopy(defaults)
    result = dict(value)
    for key, default in defaults.items():
        if key not in result or not isinstance(result[key], type(default)):
            result[key] = copy.deepcopy(default)
        elif isinstance(default, dict) and default:
            result[key] = merge_defaults(result[key], default)
    return result


class AppStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        # A notice for the user about this start (a settings file that could not be read).
        self.notice = None
        path = self.root / "settings.json"
        try:
            loaded = read_json(path, {})
            if not isinstance(loaded, dict):
                raise ValueError("settings.json is not an object")
        except (OSError, ValueError) as exc:
            loaded = {}
            backup = path.with_name(
                f"settings.json.corrupt-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            try:
                os.replace(path, backup)
            except OSError:
                backup = path
            self.notice = t(
                "storage.settings_could_not_be_read",
                p1=fmt(backup, "", -1),
                reason=fmt(exc, "", -1),
            )
        self.settings = merge_defaults(loaded, default_settings())
        if self.settings["theme"] not in THEMES:
            self.settings["theme"] = "system"
        self.settings["projects"] = [
            value for value in self.settings["projects"] if isinstance(value, str) and value
        ]
        self.saved = self._serialized() if loaded else None

    def _serialized(self) -> str:
        return json.dumps(persist(self.settings), ensure_ascii=False, sort_keys=True)

    def save(self) -> None:
        """Write settings.json, only when something changed since the last write."""
        current = self._serialized()
        if current == self.saved and (self.root / "settings.json").exists():
            return
        write_json(self.root / "settings.json", self.settings)
        self.saved = current

    def portable(self, path: Path | str) -> str:
        """How a location is stored: relative to the data root when inside it, so the whole
        folder can move; absolute otherwise."""
        path = Path(path).absolute()
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def locate(self, value: Path | str, *, must_exist=True) -> Path | None:
        """Resolve a stored location. Relative values are under the data root. An absolute
        value that no longer exists (the data folder moved) is re-based onto the current data
        root by its trailing components before it is given up."""
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
            return path if not must_exist or path.exists() else None
        if path.exists():
            return path
        parts = path.parts[1:]
        for count in range(min(len(parts), 3), 0, -1):
            candidate = self.root.joinpath(*parts[-count:])
            if candidate.exists():
                return candidate
        return None if must_exist else path

    def project_root(self) -> Path:
        """Where new and imported projects are created."""
        value = self.settings.get("project_root") or "projects"
        located = self.locate(value)
        if located is not None:
            return located
        path = self.locate(value, must_exist=False)
        # A drive that is gone (a removed USB disk) falls back to the default location.
        return path if Path(path.anchor).exists() else self.root / "projects"

    def register(self, root: Path) -> dict:
        project = load_project(root)
        resolved = root.resolve()
        kept = []
        # A migrated project retains its business ID and replaces its old location.
        for previous in self.settings["projects"]:
            location = self.locate(previous)
            if location is not None:
                if location.resolve() == resolved:
                    continue
                try:
                    if load_project(location)["id"] == project["id"]:
                        continue
                except (OSError, ValueError, KeyError):
                    pass
            kept.append(previous)
        self.settings["projects"] = [self.portable(resolved), *kept]
        self.save()
        return project

    def forget(self, root: Path) -> None:
        """Take a project off the list; its folder is left as it is."""
        resolved = root.resolve()
        self.settings["projects"] = [
            value
            for value in self.settings["projects"]
            if (location := self.locate(value, must_exist=False)) is None
            or location.resolve() != resolved
        ]
        self.save()

    def project_ids(self) -> set[str]:
        return {project["id"] for _root, project in self.projects()}

    def projects(self):
        entries = self.settings["projects"]
        for index, value in enumerate(list(entries)):
            location = self.locate(value)
            if location is None:
                continue
            try:
                project = load_project(location)
            except (OSError, ValueError, KeyError):
                continue
            portable = self.portable(location)
            if portable != value and index < len(entries) and entries[index] == value:
                entries[index] = portable
            yield location.resolve(), project

    def runs(self, only=None):
        for root, project in self.projects():
            if only and root != only:
                continue
            for path in sorted((root / "runs").glob("*/run.json"), reverse=True):
                try:
                    run = read_json(path)
                except (OSError, ValueError):
                    continue
                if isinstance(run, dict) and "id" in run:
                    yield root, project, path.parent, run

    def history(self):
        entries = []
        for path in (self.root / "generations").glob("*/generation.json"):
            try:
                entries.append(self.history_entry(path))
            except (OSError, ValueError, AttributeError):
                continue
        return sorted(entries, key=lambda value: str(value["created"]), reverse=True)

    def delete_history(self, entry_id: str) -> None:
        """Remove one generation (its record and audio) from disk."""
        generations = self.root / "generations"
        directory = removable(generations, entry_id)
        if directory.is_dir():
            remove_tree(generations, directory)

    @staticmethod
    def resolve_audio(record_path: Path, value) -> str:
        """A record's audio: relative to its folder, or absolute (older records); an absolute
        path from before the data folder moved is found again next to the record."""
        if not value:
            return str(record_path.parent / "audio.wav")
        path = Path(value)
        if not path.is_absolute():
            return str(record_path.parent / path)
        if not path.exists() and (record_path.parent / path.name).exists():
            return str(record_path.parent / path.name)
        return str(path)

    def history_entry(self, path):
        entry = read_json(path)
        if not isinstance(entry, dict):
            raise ValueError("generation record is not an object")
        # The folder name is the record's identity; deleting goes by it, never by content.
        entry["id"] = path.parent.name
        entry["audio"] = self.resolve_audio(path, entry.get("audio"))
        if isinstance(entry.get("candidates"), list):
            entry["candidates"] = [
                self.resolve_audio(path, name) for name in entry["candidates"] if name
            ]
        entry.setdefault("created", "")
        entry.setdefault("duration", 0)
        request = entry.get("request")
        if not isinstance(request, dict):
            request = entry["request"] = {}
        request.setdefault("text", "")
        if request.get("ref_wavs"):
            request["ref_wavs"] = [
                str(self.locate(item) or item) for item in request["ref_wavs"] if item
            ]
        if not entry.get("voice_id") and not request.get("lora_adapter"):
            entry["voice"] = t("generation_page.default_voice")
        entry.setdefault("voice", "")
        return entry
