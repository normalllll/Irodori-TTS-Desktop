"""Which models the application needs, and model files that arrive without a download.

The models are grouped for the onboarding page: required (no speech and no training without
them), recommended (automatic transcription, which users without transcripts depend on) and
optional. A user who cannot download can import files copied from someone else's computer;
they are accepted only when name, size and SHA-256 match the pinned revision, and are placed
in the download cache exactly as a download would place them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from irodori_tts.desktop.messages import BusinessError, Message, fmt, t
from irodori_tts.task_control import check_cancel

from . import ASR_DEFAULT_CHECK, ASR_DEFAULT_MAIN, asr
from .resources import (
    FAST_FILES,
    FAST_MODEL,
    FAST_REPO,
    FAST_REVISION,
    LOW_VRAM_FILES,
    LOW_VRAM_MODEL,
    LOW_VRAM_REPO,
    LOW_VRAM_REVISION,
    RESOURCE_SPEC,
    ModelSpec,
    cached_snapshot,
    hf_home,
)
from .storage import identity

PINS_PATH = Path(__file__).with_name("model_pins.json")
MANIFEST = "irodori-models.json"
CHUNK = 4 * 1024**2


@cache
def load_pins() -> dict:
    """{repo: {"revision", "files": {name: {"size", "sha256"}}}} for every model file."""
    if not PINS_PATH.is_file():
        return {}
    return json.loads(PINS_PATH.read_text(encoding="utf-8"))["repos"]


def pinned_models() -> list[ModelSpec]:
    """The speech resources (model, codec, watermark) as model specs."""
    pins = load_pins()
    specs = []
    for spec in RESOURCE_SPEC.values():
        files = tuple(spec["files"])
        pinned = pins.get(spec["repo"], {}).get("files", {})
        size = sum(pinned.get(name, {}).get("size", 0) for name in files)
        specs.append(ModelSpec(spec["repo"], spec["revision"], files, size))
    return specs


def _generation_model(repo: str, revision: str, files: tuple[str, ...]) -> ModelSpec:
    pinned = load_pins().get(repo, {}).get("files", {})
    size = sum(pinned.get(name, {}).get("size", 0) for name in files)
    return ModelSpec(repo, revision, files, size)


def low_vram_model() -> ModelSpec:
    return _generation_model(LOW_VRAM_REPO, LOW_VRAM_REVISION, LOW_VRAM_FILES)


def fast_model() -> ModelSpec:
    return _generation_model(FAST_REPO, FAST_REVISION, FAST_FILES)


@dataclass(frozen=True)
class Download:
    """One choice on the onboarding page."""

    key: str
    title: str | Message
    tier: str  # "required", "recommended" or "optional"
    reason: Message
    models: tuple[ModelSpec, ...]
    backend: str | None = None  # the transcription backend it provides

    @property
    def size(self) -> int:
        return sum(model.size for model in self.models)

    def installed(self) -> bool:
        return all(cached_snapshot(model) is not None for model in self.models)

    def missing_size(self) -> int:
        return sum(model.size for model in self.models if cached_snapshot(model) is None)


SPEECH = "speech"
LOW_VRAM = "low-vram"
FAST = "fast"
# The generation models a setting can choose instead of the speech model.
GENERATION_MODELS = (LOW_VRAM, FAST)


def downloads() -> list[Download]:
    items = [
        Download(
            SPEECH,
            t("models.speech_title"),
            "required",
            t("models.speech_reason"),
            tuple(pinned_models()),
        ),
        Download(
            LOW_VRAM,
            t("models.low_vram_title"),
            "optional",
            t("models.low_vram_reason"),
            (low_vram_model(),),
        ),
        Download(
            FAST,
            t("models.fast_title"),
            "optional",
            t("models.fast_reason"),
            (fast_model(),),
        ),
    ]
    for backend in asr.BACKENDS:
        if backend.key == ASR_DEFAULT_MAIN:
            tier, reason = "recommended", t("models.main_reason")
        elif backend.key == ASR_DEFAULT_CHECK:
            tier, reason = "recommended", t("models.check_reason")
        else:
            tier, reason = "optional", backend.hint
        items.append(
            Download(backend.key, backend.title, tier, reason, backend.models, backend.key)
        )
    return items


def download_for(key: str) -> Download:
    return next(item for item in downloads() if item.key == key)


def speech_ready() -> bool:
    return download_for(SPEECH).installed()


def low_vram_checkpoint() -> str:
    """The generation model of the low video memory setting, once it is downloaded."""
    return str(snapshot_path(LOW_VRAM_REPO, LOW_VRAM_REVISION, LOW_VRAM_MODEL))


def fast_checkpoint() -> str:
    """The generation model of the fast setting, once it is downloaded."""
    return str(snapshot_path(FAST_REPO, FAST_REVISION, FAST_MODEL))


def generation_checkpoint(key: str) -> str:
    return low_vram_checkpoint() if key == LOW_VRAM else fast_checkpoint()


def snapshot_path(repo: str, revision: str, filename: str) -> Path:
    return (
        hf_home()
        / "hub"
        / ("models--" + repo.replace("/", "--"))
        / "snapshots"
        / revision
        / filename
    )


# ----- import -----


SEVEN_ZIP_SIGNATURE = b"7z\xbc\xaf\x27\x1c"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def seven_zip() -> str | None:
    """7-Zip's console program: the one a package carries (runtime/7zip), else an installed
    one."""
    carried = Path(__file__).resolve().parents[2].parent / "runtime/7zip/7z.exe"
    installed = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "7-Zip/7z.exe"
    for candidate in (carried, shutil.which("7z"), installed):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


def _is_7z(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(len(SEVEN_ZIP_SIGNATURE)) == SEVEN_ZIP_SIGNATURE


class _SevenZipStream:
    """The files of a 7z archive, read in one pass.

    7-Zip writes every file to its output one after the other in archive order, so a solid
    archive is decompressed once rather than once per file. Each file is cut from that stream
    by its listed size and is only ever read in order; files nobody opens are read past. The
    SHA-256 check of every file read catches any misalignment, as it catches damage."""

    def __init__(self, source: Path, cancel=None):
        self.source = source
        self.cancel = cancel
        self.program = seven_zip()
        if self.program is None:
            raise BusinessError(t("models.import_needs_7zip"))
        listing = self._run(["l", "-slt", "-ba"])
        self.entries = []
        for block in listing.split("\n\n"):
            fields = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
            if "Path" in fields and fields.get("Folder") != "+":
                self.entries.append((fields["Path"], int(fields.get("Size") or 0)))
        self.process = None
        self.errors = None
        self.next = 0

    def _run(self, arguments: list[str]) -> str:
        result = subprocess.run(
            [self.program, *arguments, "-sccUTF-8", str(self.source)],
            capture_output=True,
            creationflags=NO_WINDOW,
        )
        if result.returncode:
            self._fail(result.stderr.decode("utf-8", "replace"))
        return result.stdout.decode("utf-8", "replace").replace("\r\n", "\n")

    def _fail(self, output: str):
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        raise BusinessError(
            t(
                "models.import_7zip_failed",
                name=fmt(self.source.name, "", -1),
                reason=fmt(lines[-1] if lines else "", "", -1),
            )
        )

    def files(self):
        return [
            (name, size, lambda index=index: self._open(index))
            for index, (name, size) in enumerate(self.entries)
        ]

    def _open(self, index: int):
        if index < self.next:
            raise RuntimeError("7z entries are read in archive order")
        if self.process is None:
            self.errors = tempfile.TemporaryFile()
            self.process = subprocess.Popen(
                [self.program, "x", "-so", "-y", "-bd", "-sccUTF-8", str(self.source)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=self.errors,
                creationflags=NO_WINDOW,
            )
        while self.next < index:
            with self._entry(self.entries[self.next][1]) as skipped:
                while skipped.read(CHUNK):
                    check_cancel(self.cancel)
        return self._entry(self.entries[index][1])

    @contextmanager
    def _entry(self, size: int):
        self.next += 1
        stream, remaining = self.process.stdout, size

        class Entry:
            @staticmethod
            def read(amount: int) -> bytes:
                nonlocal remaining
                data = stream.read(min(amount, remaining)) if remaining else b""
                if remaining and not data:
                    self.errors.seek(0)
                    self._fail(self.errors.read().decode("utf-8", "replace"))
                remaining -= len(data)
                return data

        yield Entry()

    def close(self):
        # The files after the last one needed are not read: stop 7-Zip rather than wait.
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            self.process.stdout.close()
            self.errors.close()


@contextmanager
def _source_files(source: Path, cancel=None):
    """[(display name, size, open())] for every file in a folder, a zip or a 7z archive, which
    stays open while they are read. A 7z archive's files open only in the listed order."""
    if source.is_dir():
        yield [
            (str(path.relative_to(source)), path.stat().st_size, lambda path=path: path.open("rb"))
            for path in sorted(source.rglob("*"))
            if path.is_file()
        ]
        return
    if source.is_file() and _is_7z(source):
        archive = _SevenZipStream(source, cancel)
        try:
            yield archive.files()
        finally:
            archive.close()
        return
    if not zipfile.is_zipfile(source):
        raise BusinessError(t("models.import_not_a_folder_or_zip", name=fmt(source.name, "", -1)))
    with zipfile.ZipFile(source) as archive:
        yield [
            (info.filename, info.file_size, lambda info=info: archive.open(info))
            for info in archive.infolist()
            if not info.is_dir()
        ]


def _copy(opener, target: Path, *, cancel, progress) -> str:
    """Copy into target while hashing; returns the SHA-256 of what was written."""
    sha = hashlib.sha256()
    target.parent.mkdir(parents=True, exist_ok=True)
    with opener() as stream, target.open("wb") as out:
        while chunk := stream.read(CHUNK):
            check_cancel(cancel)
            sha.update(chunk)
            out.write(chunk)
            if progress:
                progress(len(chunk))
    return sha.hexdigest()


def import_model_files(
    source: Path,
    *,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Place every file of source that is a pinned model file into the download cache.

    Files are recognised by name and size, then accepted only with the pinned SHA-256. A file
    the cache already has is left alone. Returns what was imported, already present, rejected,
    and which downloads are now complete."""
    pins = load_pins()
    by_name: dict[tuple[str, int], list[tuple[str, str, str, str]]] = {}
    for repo, entry in pins.items():
        for filename, info in entry["files"].items():
            key = (Path(filename).name, info["size"])
            by_name.setdefault(key, []).append((repo, entry["revision"], filename, info["sha256"]))
    with _source_files(source, cancel) as files:
        candidates = [c for c in files if (Path(c[0]).name, c[1]) in by_name]
        imported, present, rejected = _import(candidates, by_name, cancel, progress)
    complete = [item.key for item in downloads() if item.installed()]
    return {
        "imported": imported,
        "present": present,
        "rejected": [(name, str(reason)) for name, reason in rejected],
        "complete": complete,
    }


def _import(candidates, by_name, cancel, progress):
    total = sum(size for _, size, _ in candidates)
    done = 0
    imported, present, rejected = [], [], []
    for name, size, opener in candidates:
        check_cancel(cancel)
        matches = by_name[(Path(name).name, size)]
        # Prefer the pin whose path the source shares ("tokenizer/tokenizer.json").
        matches.sort(key=lambda m: not name.replace("\\", "/").endswith(m[2]))
        waiting = [m for m in matches if not snapshot_path(m[0], m[1], m[2]).is_file()]
        if not waiting:
            present.append(name)
            done += size
            continue
        staging = hf_home() / "hub" / f".import-{identity()}.partial"

        def moved(amount):
            nonlocal done
            done += amount
            if progress:
                progress(done, total)

        try:
            sha = _copy(opener, staging, cancel=cancel, progress=moved)
            # Models can share a file (whisper-large-v3 and anime-whisper share their
            # tokenizer files); every one still waiting for it gets a copy.
            targets = [m for m in waiting if m[3] == sha]
            if not targets:
                if any(m[3] == sha for m in matches):
                    present.append(name)
                elif any(name.replace("\\", "/").endswith(m[2]) for m in matches):
                    # It stands where a pinned file belongs: another version, or damaged.
                    rejected.append((name, t("models.import_hash_mismatch")))
                # Otherwise an unrelated file that only shares a name and size is not ours.
                continue
            for index, (repo, revision, filename, _) in enumerate(targets):
                target = snapshot_path(repo, revision, filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                if index == len(targets) - 1:
                    staging.replace(target)
                else:
                    shutil.copyfile(staging, target)
                imported.append(f"{repo}/{filename}")
        finally:
            staging.unlink(missing_ok=True)
    return imported, present, rejected


# ----- export -----


def export_model_files(
    keys: list[str],
    destination: Path,
    *,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """A zip of the downloaded files of these models, laid out like the download cache, with a
    manifest. Another user imports it with import_model_files."""
    pins = load_pins()
    entries = []
    for key in keys:
        for model in download_for(key).models:
            folder = cached_snapshot(model)
            if folder is None:
                raise BusinessError(t("models.export_not_downloaded", name=fmt(model.repo, "", -1)))
            for filename in model.files:
                entries.append((model, filename, folder / filename))
    total = sum(path.stat().st_size for _, _, path in entries)
    done = 0
    temporary = destination.with_name(f".{destination.name}.{identity()}.partial")
    try:
        # Model weights do not compress; storing them keeps export and import fast.
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
            manifest = {"format_version": 1, "models": {}}
            for model, filename, path in entries:
                arcname = (
                    f"models--{model.repo.replace('/', '--')}/snapshots/{model.revision}/{filename}"
                )
                with path.open("rb") as stream, archive.open(arcname, "w", force_zip64=True) as out:
                    while chunk := stream.read(CHUNK):
                        check_cancel(cancel)
                        out.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
                info = manifest["models"].setdefault(
                    model.repo, {"revision": model.revision, "files": {}}
                )
                info["files"][filename] = pins.get(model.repo, {}).get("files", {}).get(filename)
            archive.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=1))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"destination": str(destination), "bytes": total, "files": len(entries)}


# Memory sizes graphics cards are sold with, in GB.
CARD_SIZES = (2, 3, 4, 6, 8, 10, 11, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 141, 192)


def card_memory_gb(total_mib: float) -> int:
    """The card's nominal memory. The driver reports a little less than the size on the box
    (an 8 GB card shows about 7.6 to 7.9 GB), so the reading is rounded up to the nearest size
    cards are sold with; comparing the raw reading with "8 GB" would rule that card out."""
    reading = total_mib / 1024
    return next((size for size in CARD_SIZES if reading <= size * 1.02), round(reading))


def fits(item: Download, memory_gb: int | None) -> bool:
    """Whether a transcription model's peak memory fits the card (unknown card: assume yes)."""
    if memory_gb is None or not item.backend:
        return True
    return asr.backend_for(item.backend).vram <= memory_gb


def disk_free(path: Path) -> int:
    """Free bytes on the disk that holds path (or its nearest existing parent)."""
    path = Path(path)
    while not path.exists() and path.parent != path:
        path = path.parent
    return shutil.disk_usage(path).free
