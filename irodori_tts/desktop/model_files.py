"""Which models the application needs, and model files that arrive without a download.

The models are grouped for the onboarding page: required (no speech and no training without
them), recommended (automatic transcription, which users without transcripts depend on) and
optional. A user who cannot download can import files copied from someone else's computer.
Pinned files (codec, watermark, transcription models) are accepted only when name, size and
SHA-256 match the pinned revision; base model files follow their repository's latest commit
and are accepted when they match the record of the package that carries them. Both are
placed exactly as a download or an import would place them.
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

from . import ASR_DEFAULT_CHECK, ASR_DEFAULT_MAIN, asr, base_models
from .resources import (
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
    """{repo: {"revision", "files": {name: {"size", "sha256"}}}} for every pinned model file."""
    if not PINS_PATH.is_file():
        return {}
    return json.loads(PINS_PATH.read_text(encoding="utf-8"))["repos"]


def pinned_models() -> list[ModelSpec]:
    """The pinned resources every base model needs (codec, watermark) as model specs."""
    pins = load_pins()
    specs = []
    for spec in RESOURCE_SPEC.values():
        files = tuple(spec["files"])
        pinned = pins.get(spec["repo"], {}).get("files", {})
        size = sum(pinned.get(name, {}).get("size", 0) for name in files)
        specs.append(ModelSpec(spec["repo"], spec["revision"], files, size))
    return specs


@dataclass(frozen=True)
class Download:
    """One choice on the onboarding page: pinned models, a catalog base, or a variant."""

    key: str
    title: str | Message
    tier: str  # "required", "recommended" or "optional"
    reason: Message
    models: tuple[ModelSpec, ...]
    backend: str | None = None  # the transcription backend it provides
    base: str | None = None  # the catalog base it provides, or whose variant it provides
    variant: str | None = None
    base_size: int = 0  # the base or variant's bytes, which no pin records
    root: Path | None = None  # the resource folder whose base registry tells what is installed

    @property
    def size(self) -> int:
        return sum(model.size for model in self.models) + self.base_size

    def base_installed(self) -> bool:
        if self.base is None:
            return True
        installed = base_models.installed_base(self.root, self.base) if self.root else None
        if installed is None:
            return False
        return self.variant is None or self.variant in installed.variants

    def installed(self) -> bool:
        return self.base_installed() and all(
            cached_snapshot(model) is not None for model in self.models
        )

    def missing_size(self) -> int:
        pinned = sum(model.size for model in self.models if cached_snapshot(model) is None)
        return pinned + (0 if self.base_installed() else self.base_size)


SPEECH = "speech"
LOW_VRAM = "low_vram"
FAST = "fast"
# The generation variants a setting can choose instead of a base's standard checkpoint.
GENERATION_MODELS = (LOW_VRAM, FAST)


def base_key(base_id: str) -> str:
    """The download of a catalog base; the default base is the speech download."""
    entry = base_models.catalog_base(base_id)
    return SPEECH if entry is not None and entry.default else f"base:{base_id}"


def variant_key(base_id: str, variant: str) -> str:
    return f"variant:{base_id}:{variant}"


def _base_downloads(root: Path | None) -> list[Download]:
    items = []
    for entry in base_models.CATALOG:
        reason = (
            t("models.speech_reason")
            if entry.default
            else t("models.large_reason", url=fmt(entry.license_url or "", "", -1))
        )
        items.append(
            Download(
                base_key(entry.id),
                t("models.base_title", name=fmt(entry.name, "", -1)),
                "required" if entry.default else "optional",
                reason,
                tuple(pinned_models()) if entry.default else (),
                base=entry.id,
                base_size=entry.size,
                root=root,
            )
        )
        name = fmt(entry.name, "", -1)
        for variant in entry.variants:
            if variant.key == LOW_VRAM:
                title, why = t("models.low_vram_title", name=name), t("models.low_vram_reason")
            else:
                title, why = t("models.fast_title", name=name), t("models.fast_reason")
            items.append(
                Download(
                    variant_key(entry.id, variant.key),
                    title,
                    "optional",
                    why,
                    (),
                    base=entry.id,
                    variant=variant.key,
                    base_size=variant.size,
                    root=root,
                )
            )
    return items


def downloads(root: Path | None = None) -> list[Download]:
    """Every download, with root the resource folder whose base registry is consulted."""
    items = _base_downloads(root)
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


def download_for(key: str, root: Path | None = None) -> Download:
    return next(item for item in downloads(root) if item.key == key)


def shared_ready() -> bool:
    """Whether the pinned resources every base needs are in the cache."""
    return all(cached_snapshot(model) is not None for model in pinned_models())


def speech_ready(root: Path) -> bool:
    """At least one base model is installed, with the resources every base needs."""
    return bool(base_models.load_registry(root)) and shared_ready()


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

    def read(self, name: str) -> bytes | None:
        """One member's bytes, extracted on its own (the package manifest)."""
        if not any(entry == name for entry, _size in self.entries):
            return None
        result = subprocess.run(
            [self.program, "e", "-so", "-y", "-sccUTF-8", str(self.source), name],
            capture_output=True,
            creationflags=NO_WINDOW,
        )
        return result.stdout if not result.returncode else None

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


def _package_manifest(source: Path) -> dict:
    """The manifest a package written by export_model_files carries; {} for a plain folder or
    an archive without one."""
    raw = None
    try:
        if source.is_dir():
            path = source / MANIFEST
            raw = path.read_bytes() if path.is_file() else None
        elif source.is_file() and _is_7z(source):
            raw = _SevenZipStream(source).read(MANIFEST)
        elif zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                if MANIFEST in archive.namelist():
                    raw = archive.read(MANIFEST)
        manifest = json.loads(raw.decode("utf-8")) if raw else {}
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _package_target(root: Path, arcname: str) -> Path | None:
    """Where a base file of a package belongs: the download cache for a catalog base, the
    resource folder for an imported one."""
    parts = Path(arcname).parts
    if not parts or ".." in parts or Path(arcname).is_absolute():
        return None
    if parts[0].startswith("models--"):
        return hf_home() / "hub" / arcname
    if parts[0] == base_models.IMPORTED_FOLDER:
        return Path(root) / arcname
    return None


def import_model_files(
    source: Path,
    root: Path,
    *,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Place every file of source that is a known model file where it belongs, and register
    the base models the package carries.

    Pinned files are recognised by name and size, then accepted only with the pinned SHA-256.
    Base model files are accepted with the SHA-256 the package's own manifest records for
    them. A file already in place is left alone. Returns what was imported, already present,
    rejected, and which downloads are now complete."""
    by_name: dict[tuple[str, int], list[tuple[Path, str, str, str]]] = {}
    for repo, entry in load_pins().items():
        for filename, info in entry["files"].items():
            key = (Path(filename).name, info["size"])
            target = snapshot_path(repo, entry["revision"], filename)
            by_name.setdefault(key, []).append(
                (target, f"{repo}/{filename}", info["sha256"], filename)
            )
    bases = _package_manifest(source).get("bases")
    bases = bases if isinstance(bases, dict) else {}
    # An installed base stays as it is: a package must not move it to another commit (and
    # leave the newer one's files orphaned), nor replace an imported one.
    installed = {base.id for base in base_models.load_registry(root)}
    kept = sorted(base_id for base_id in bases if base_id in installed)
    bases = {base_id: record for base_id, record in bases.items() if base_id not in installed}
    for record in bases.values():
        files = record.get("files") if isinstance(record, dict) else None
        for arcname, info in (files if isinstance(files, dict) else {}).items():
            target = _package_target(root, arcname)
            if target is None or not isinstance(info, dict):
                continue
            key = (Path(arcname).name, info.get("size"))
            by_name.setdefault(key, []).append((target, arcname, info.get("sha256"), arcname))
    with _source_files(source, cancel) as files:
        candidates = [c for c in files if (Path(c[0]).name, c[1]) in by_name]
        imported, present, rejected = _import(candidates, by_name, cancel, progress)
    registered = _register_package_bases(root, bases)
    complete = [item.key for item in downloads(root) if item.installed()]
    complete += [f"imported:{base.id}" for base in registered if base.source == "imported"]
    return {
        "imported": imported,
        "present": present + [f"base:{base_id}" for base_id in kept],
        "kept": kept,
        "rejected": [(name, str(reason)) for name, reason in rejected],
        "complete": complete,
    }


def _register_package_bases(root: Path, bases: dict) -> list:
    """Register each base of a package whose files are now all in place."""
    registered = []
    for base_id, record in bases.items():
        try:
            targets = {arcname: _package_target(root, arcname) for arcname in record["files"]}
            if any(path is None or not path.is_file() for path in targets.values()):
                continue
            checkpoint = targets[record["checkpoint"]]
            architecture = base_models.read_architecture(checkpoint)
            if architecture is None:
                continue
            variants = {
                key: targets[arcname]
                for key, arcname in (record.get("variants") or {}).items()
                if arcname in targets
            }
            catalog = record.get("source") == "catalog" and base_models.catalog_base(base_id)
            base = base_models.InstalledBase(
                id=str(base_id),
                name=str(record.get("name") or base_id),
                source="catalog" if catalog else "imported",
                repo=record.get("repo"),
                revision=record.get("revision"),
                checkpoint=checkpoint,
                architecture=architecture,
                variants=variants,
            )
        except (KeyError, TypeError, AttributeError):
            continue
        base_models.save_base(root, base)
        registered.append(base)
    return registered


def _import(candidates, by_name, cancel, progress):
    total = sum(size for _, size, _ in candidates)
    done = 0
    imported, present, rejected = [], [], []
    for name, size, opener in candidates:
        check_cancel(cancel)
        matches = by_name[(Path(name).name, size)]
        path = name.replace("\\", "/")
        # Prefer the file whose path the source shares ("tokenizer/tokenizer.json").
        matches.sort(key=lambda m: not path.endswith(m[3]))
        waiting = [m for m in matches if not m[0].is_file()]
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
            targets = [m for m in waiting if m[2] == sha]
            if not targets:
                if any(m[2] == sha for m in matches):
                    present.append(name)
                elif any(path.endswith(m[3]) for m in matches):
                    # It stands where a known file belongs: another version, or damaged.
                    rejected.append((name, t("models.import_hash_mismatch")))
                # Otherwise an unrelated file that only shares a name and size is not ours.
                continue
            for index, (target, label, _sha, _suffix) in enumerate(targets):
                target.parent.mkdir(parents=True, exist_ok=True)
                if index == len(targets) - 1:
                    staging.replace(target)
                else:
                    shutil.copyfile(staging, target)
                imported.append(label)
        finally:
            staging.unlink(missing_ok=True)
    return imported, present, rejected


# ----- export -----


def _arcname(repo: str, revision: str, filename: str) -> str:
    return f"models--{repo.replace('/', '--')}/snapshots/{revision}/{filename}"


def _base_entries(base: base_models.InstalledBase, variants) -> tuple[list, dict]:
    """[(arcname, path)] of a base and the chosen variants, and its package record without
    the file hashes, which are computed while the files are written."""
    record = {**base_models.base_record(base), "source": base.source, "files": {}, "variants": {}}
    folder = Path(base.checkpoint).parent
    entry = base.catalog
    if entry is None:
        names = ["model.safetensors"] + [
            path.relative_to(folder).as_posix()
            for path in sorted((folder / "tokenizer").rglob("*"))
            if path.is_file()
        ]
        prefix = f"{base_models.IMPORTED_FOLDER}/{base.id}"
        record["checkpoint"] = f"{prefix}/model.safetensors"
        return [(f"{prefix}/{name}", folder / name) for name in names], record
    entries = [(_arcname(entry.repo, base.revision, name), folder / name) for name in entry.files]
    record["checkpoint"] = _arcname(entry.repo, base.revision, "model.safetensors")
    for key in variants:
        variant = entry.variant(key)
        path = base.variants.get(key)
        if variant is None or path is None:
            continue
        # The cache keeps the file at snapshots/<commit>/<model_file>.
        commit = Path(path).parents[len(Path(variant.model_file).parts) - 1]
        entries += [
            (_arcname(variant.repo, commit.name, name), commit / name) for name in variant.files
        ]
        record["variants"][key] = _arcname(variant.repo, commit.name, variant.model_file)
    return entries, record


def export_model_files(
    keys: list[str],
    destination: Path,
    root: Path,
    *,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """A zip of the downloaded files of these models, laid out like the download cache, with a
    manifest. Another user imports it with import_model_files. Keys are download keys, or
    "imported:<id>" for an imported base."""
    pins = load_pins()
    entries = []  # (arcname, path, pinned model or None, base id or None)
    bases: dict[str, tuple] = {}
    for key in keys:
        if key.startswith("imported:"):
            base = base_models.installed_base(root, key.split(":", 1)[1])
            if base is not None:
                bases.setdefault(base.id, (base, set()))
            continue
        item = download_for(key, root)
        for model in item.models:
            folder = cached_snapshot(model)
            if folder is None:
                raise BusinessError(t("models.export_not_downloaded", name=fmt(model.repo, "", -1)))
            entries += [
                (_arcname(model.repo, model.revision, name), folder / name, model, None)
                for name in model.files
            ]
        if item.base is not None:
            base = base_models.installed_base(root, item.base)
            if base is None or (item.variant and item.variant not in base.variants):
                raise BusinessError(t("models.export_not_downloaded", name=fmt(item.title, "", -1)))
            chosen = bases.setdefault(base.id, (base, set()))[1]
            if item.variant:
                chosen.add(item.variant)
    if bases:
        # No base speaks without the codec and the watermark: a pack carrying one carries them.
        for model in pinned_models():
            folder = cached_snapshot(model)
            if folder is None:
                raise BusinessError(t("models.export_not_downloaded", name=fmt(model.repo, "", -1)))
            entries += [
                (_arcname(model.repo, model.revision, name), folder / name, model, None)
                for name in model.files
            ]
    records = {}
    for base, variants in bases.values():
        files, records[base.id] = _base_entries(base, sorted(variants))
        entries += [(arcname, path, None, base.id) for arcname, path in files]
    unique = {}
    for entry in entries:
        unique.setdefault(entry[0], entry)
    total = sum(entry[1].stat().st_size for entry in unique.values())
    done = 0
    temporary = destination.with_name(f".{destination.name}.{identity()}.partial")
    try:
        # Model weights do not compress; storing them keeps export and import fast.
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
            manifest = {"format_version": 1, "models": {}, "bases": records}
            for arcname, path, model, base_id in unique.values():
                sha = hashlib.sha256()
                with path.open("rb") as stream, archive.open(arcname, "w", force_zip64=True) as out:
                    while chunk := stream.read(CHUNK):
                        check_cancel(cancel)
                        sha.update(chunk)
                        out.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
                if model is not None:
                    filename = arcname.split("/", 3)[3]
                    info = manifest["models"].setdefault(
                        model.repo, {"revision": model.revision, "files": {}}
                    )
                    info["files"][filename] = (
                        pins.get(model.repo, {}).get("files", {}).get(filename)
                    )
                else:
                    records[base_id]["files"][arcname] = {
                        "size": path.stat().st_size,
                        "sha256": sha.hexdigest(),
                    }
            archive.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=1))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"destination": str(destination), "bytes": total, "files": len(unique)}


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
