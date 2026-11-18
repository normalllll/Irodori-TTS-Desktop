"""The base speech models the application can generate with and train on.

A base is identified by its architecture: the model shape its checkpoint's safetensors header
records in config_json. The runtime builds the model from that metadata, and a LoRA voice loads
onto any base whose shape is the one it was trained on, so voices are matched to bases by
architecture rather than by repository, revision or file hash. A newer commit of a catalog base
keeps the shape and keeps every trained voice working.

Installed bases are listed in the base registry, base-models.json in the resource folder. A
catalog base follows the latest commit of its Hugging Face repository and is downloaded into
the Hugging Face cache; an imported base is copied into bases/<id>/ of the resource folder.
Every base keeps its tokenizer/ folder beside its model file: with it, generation and training
need no network access (without it, upstream fetches the text tokenizer, and some are gated).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path

from irodori_tts.desktop.messages import BusinessError, fmt, t
from irodori_tts.task_control import check_cancel

from .storage import identity, read_json, write_json

ARCHITECTURE_KEYS = (
    "model_dim",
    "num_layers",
    "num_heads",
    "text_tokenizer_repo",
    "text_dim",
    "speaker_dim",
    "speaker_layers",
    "duration_hidden_dim",
    "duration_layers",
)
REGISTRY = "base-models.json"
IMPORTED_FOLDER = "bases"
TOKENIZER_FILES = ("tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")
# A safetensors header is a few hundred kilobytes; anything far larger is not one.
MAX_HEADER = 100 * 1024**2


@dataclass(frozen=True)
class Variant:
    """A generation-only alternative checkpoint of a catalog base, in a repository of its own
    with its own tokenizer folder. A LoRA voice never runs on a variant of another base."""

    key: str  # "low_vram" | "fast"
    repo: str
    model_file: str
    files: tuple[str, ...]
    steps: int | None  # sampling steps the variant is distilled for, None for the usual steps
    size: int = 0  # bytes to download, shown before the revision is resolved


@dataclass(frozen=True)
class CatalogBase:
    id: str
    name: str
    repo: str
    files: tuple[str, ...]
    architecture: dict
    variants: tuple[Variant, ...]
    training_memory: tuple[float, float]  # (fixed GiB, GiB per recording in a batch)
    default: bool
    size: int = 0  # bytes to download, shown before the revision is resolved
    license_url: str | None = None  # terms the base is used under, when they are not open

    def variant(self, key: str) -> Variant | None:
        return next((variant for variant in self.variants if variant.key == key), None)


SMALL_ARCHITECTURE = {
    "model_dim": 1280,
    "num_layers": 12,
    "num_heads": 20,
    "text_tokenizer_repo": "sbintuitions/modernbert-ja-310m",
    "text_dim": 512,
    "speaker_dim": 768,
    "speaker_layers": 8,
    "duration_hidden_dim": 1024,
    "duration_layers": 3,
}
LARGE_ARCHITECTURE = {
    "model_dim": 2048,
    "num_layers": 24,
    "num_heads": 32,
    "text_tokenizer_repo": "google/t5gemma-2-1b-1b",
    "text_dim": 1280,
    "speaker_dim": 1280,
    "speaker_layers": 14,
    "duration_hidden_dim": 1280,
    "duration_layers": 4,
}
SMALL_TOKENIZER = 6_718_495 + 668
LARGE_TOKENIZER = 33_384_387 + 802
QUANTIZED_MODEL = "int8-weight-only/model.safetensors"

CATALOG = (
    CatalogBase(
        id="v4.1-small",
        name="Irodori-TTS v4.1 Small",
        repo="Aratako/Irodori-TTS-v4.1-Small",
        files=("model.safetensors", *TOKENIZER_FILES),
        architecture=SMALL_ARCHITECTURE,
        variants=(
            # About a quarter less video memory, and the same speech at the same seed.
            Variant(
                "low_vram",
                "Aratako/Irodori-TTS-v4.1-Small-Quantized",
                QUANTIZED_MODEL,
                (QUANTIZED_MODEL, *TOKENIZER_FILES),
                None,
                914_316_834 + SMALL_TOKENIZER,
            ),
            # The MeanFlow distillation: four steps instead of forty, guidance trained in.
            Variant(
                "fast",
                "Aratako/Irodori-TTS-v4.1-Small-MF",
                "model.safetensors",
                ("model.safetensors", *TOKENIZER_FILES),
                4,
                3_093_131_788 + SMALL_TOKENIZER,
            ),
        ),
        training_memory=(4.0, 1.0),
        default=True,
        size=3_064_295_596 + SMALL_TOKENIZER,
    ),
    CatalogBase(
        id="v4-large",
        name="Irodori-TTS v4 Large",
        repo="Aratako/Irodori-TTS-v4-Large",
        files=("model.safetensors", *TOKENIZER_FILES, "NOTICE"),
        architecture=LARGE_ARCHITECTURE,
        variants=(
            Variant(
                "low_vram",
                "Aratako/Irodori-TTS-v4-Large-Quantized",
                QUANTIZED_MODEL,
                (QUANTIZED_MODEL, *TOKENIZER_FILES),
                None,
                3_839_580_074 + LARGE_TOKENIZER,
            ),
        ),
        # LoRA training peaks at about 13.1 GiB with batch one and checkpointing; without
        # checkpointing it does not fit in 13.5 GiB of free memory even at batch one.
        training_memory=(13.5, 3.0),
        default=False,
        size=13_152_893_348 + LARGE_TOKENIZER + 96,
        license_url=(
            "https://huggingface.co/Aratako/Irodori-TTS-v4-Large/blob/main/GEMMA_TERMS_OF_USE.md"
        ),
    ),
)
VARIANT_KEYS = ("low_vram", "fast")


def catalog_base(base_id: str) -> CatalogBase | None:
    return next((entry for entry in CATALOG if entry.id == base_id), None)


def default_catalog_base() -> CatalogBase:
    return next(entry for entry in CATALOG if entry.default)


def _catalog_by_repo(repo) -> CatalogBase | None:
    return next((entry for entry in CATALOG if entry.repo == repo), None)


def _catalog_by_architecture(architecture: dict) -> CatalogBase | None:
    return next((entry for entry in CATALOG if entry.architecture == architecture), None)


@dataclass(frozen=True)
class InstalledBase:
    id: str
    name: str
    source: str  # "catalog" | "imported"
    repo: str | None
    revision: str | None
    checkpoint: Path
    architecture: dict
    variants: dict = field(default_factory=dict)  # key -> Path

    @property
    def catalog(self) -> CatalogBase | None:
        return catalog_base(self.id) if self.source == "catalog" else None


# ----- architecture -----


def _header(path: Path) -> bytes | None:
    """The JSON header of a safetensors file, without the length prefix; None otherwise."""
    try:
        with Path(path).open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                return None
            (length,) = struct.unpack("<Q", prefix)
            if not 2 <= length <= MAX_HEADER:
                return None
            raw = stream.read(length)
    except OSError:
        return None
    return raw if len(raw) == length and raw[:1] == b"{" else None


def _metadata(raw: bytes | None) -> dict | None:
    if raw is None:
        return None
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    metadata = header.get("__metadata__") if isinstance(header, dict) else None
    return metadata if isinstance(metadata, dict) else None


def architecture_of_config(model_config: dict) -> dict | None:
    if not isinstance(model_config, dict) or any(
        key not in model_config for key in ARCHITECTURE_KEYS
    ):
        return None
    return {key: model_config[key] for key in ARCHITECTURE_KEYS}


def read_architecture(path: Path) -> dict | None:
    """The architecture from the header only; the weights are never read."""
    metadata = _metadata(_header(path))
    if metadata is None:
        return None
    try:
        config = json.loads(metadata.get("config_json") or "")
    except (TypeError, ValueError):
        return None
    return architecture_of_config(config)


# ----- registry -----


def _registry(root: Path) -> dict:
    data = read_json(Path(root) / REGISTRY, None)
    if not isinstance(data, dict) or not isinstance(data.get("bases"), dict):
        return {"format_version": 1, "bases": {}}
    return data


# Registry paths under the resource folder are written relative to it with this prefix, so a
# moved data folder keeps its imported bases; others are relative to the download cache.
IN_ROOT = "./"


def _store(root: Path, path: Path) -> str:
    from .resources import stored_path

    try:
        return IN_ROOT + Path(os.path.abspath(path)).relative_to(os.path.abspath(root)).as_posix()
    except ValueError:
        return stored_path(Path(path))


def _load(root: Path, value: str) -> Path:
    from .resources import manifest_file

    if value.startswith(IN_ROOT):
        return Path(root) / value[len(IN_ROOT) :]
    return manifest_file(value)


def _from_entry(root: Path, entry: dict) -> InstalledBase | None:
    try:
        checkpoint = _load(root, entry["checkpoint"])
        variants = {key: _load(root, value) for key, value in (entry.get("variants") or {}).items()}
        base = InstalledBase(
            id=str(entry["id"]),
            name=str(entry.get("name") or entry["id"]),
            source=entry.get("source") or "imported",
            repo=entry.get("repo"),
            revision=entry.get("revision"),
            checkpoint=checkpoint,
            architecture=dict(entry["architecture"]),
            variants={key: path for key, path in variants.items() if path.is_file()},
        )
    except (KeyError, TypeError, ValueError):
        return None
    return base if checkpoint.is_file() else None


def load_registry(root: Path) -> list[InstalledBase]:
    """The installed bases, catalog ones first in catalog order, then imported ones."""
    bases = [_from_entry(root, entry) for entry in _registry(root)["bases"].values()]
    bases = [base for base in bases if base is not None]
    order = {entry.id: index for index, entry in enumerate(CATALOG)}
    return sorted(
        bases, key=lambda base: (base.source != "catalog", order.get(base.id, len(order)))
    )


def save_base(root: Path, base: InstalledBase) -> None:
    data = _registry(root)
    data["bases"][base.id] = {
        "id": base.id,
        "name": base.name,
        "source": base.source,
        "repo": base.repo,
        "revision": base.revision,
        "checkpoint": _store(root, Path(base.checkpoint)),
        "architecture": dict(base.architecture),
        "variants": {key: _store(root, Path(path)) for key, path in base.variants.items()},
    }
    write_json(Path(root) / REGISTRY, data)


def remove_base(root: Path, base_id: str) -> None:
    data = _registry(root)
    if data["bases"].pop(base_id, None) is not None:
        write_json(Path(root) / REGISTRY, data)


def installed_base(root: Path, base_id: str | None) -> InstalledBase | None:
    return next((base for base in load_registry(root) if base.id == base_id), None)


def bases_of(resources: dict) -> list[InstalledBase]:
    """The installed bases of a resolve_resources result (which crosses process boundaries
    as JSON)."""
    return [
        InstalledBase(
            id=entry["id"],
            name=entry["name"],
            source=entry.get("source") or "imported",
            repo=entry.get("repo"),
            revision=entry.get("revision"),
            checkpoint=Path(entry["checkpoint"]),
            architecture=dict(entry["architecture"]),
            variants={key: Path(path) for key, path in (entry.get("variants") or {}).items()},
        )
        for entry in (resources.get("bases") or {}).values()
    ]


# ----- records -----


def base_record(base: InstalledBase) -> dict:
    """What a run, a voice, an export or a generation keeps of the base it used."""
    return {
        "id": base.id,
        "name": base.name,
        "repo": base.repo,
        "revision": base.revision,
        "architecture": dict(base.architecture),
    }


def normalize_record(record) -> dict | None:
    """A base record in the current shape. A record of only repo and revision, from a catalog
    repository, takes that catalog base's id, name and architecture; one from another
    repository stays without them (unresolved)."""
    if not isinstance(record, dict):
        return None
    record = dict(record)
    if record.get("architecture"):
        return record
    entry = catalog_base(record.get("id")) or _catalog_by_repo(record.get("repo"))
    if entry is None:
        return record
    return {
        "id": entry.id,
        "name": record.get("name") or entry.name,
        "repo": record.get("repo") or entry.repo,
        "revision": record.get("revision"),
        "architecture": dict(entry.architecture),
    }


@dataclass(frozen=True)
class Match:
    status: str  # "exact" | "compatible" | "missing" | "unknown"
    base: InstalledBase | None
    needs: CatalogBase | None  # for "missing": the catalog base with that architecture, if any


def _compatible(architecture: dict, bases: list[InstalledBase]) -> InstalledBase | None:
    same = [base for base in bases if base.architecture == architecture]
    same.sort(key=lambda base: base.source != "catalog")
    return same[0] if same else None


def match_architecture(architecture: dict | None, bases: list[InstalledBase]) -> Match:
    if not architecture:
        return Match("unknown", None, None)
    found = _compatible(architecture, bases)
    if found is None:
        return Match("missing", None, _catalog_by_architecture(architecture))
    return Match("compatible", found, None)


def match_record(record, bases: list[InstalledBase]) -> Match:
    """The installed base a voice or run uses: the very base it was made with when it is
    installed, else another base of the same architecture (catalog before imported)."""
    record = normalize_record(record)
    if record is None or not record.get("architecture"):
        return Match("unknown", None, None)
    architecture = record["architecture"]
    exact = next(
        (
            base
            for base in bases
            if base.architecture == architecture
            and base.id == record.get("id")
            and base.revision == record.get("revision")
        ),
        None,
    )
    if exact is not None:
        return Match("exact", exact, None)
    return match_architecture(architecture, bases)


def describe(record) -> str:
    """A base record's name, or its shape when it has none (a voice trained elsewhere)."""
    record = normalize_record(record) or {}
    architecture = record.get("architecture") or {}
    if record.get("name"):
        return str(record["name"])
    if architecture:
        return f"model_dim {architecture['model_dim']}, {architecture['num_layers']} layers"
    return str(record.get("repo") or "?")


def missing_message(match: Match, record=None):
    """Why a voice or run cannot use any installed base, naming what to do."""
    if match.needs is not None:
        return t("base_models.needs_catalog_base", name=fmt(match.needs.name, "", -1))
    if match.status == "unknown":
        return t("base_models.base_not_recorded")
    return t("base_models.needs_unknown_base", name=fmt(describe(record), "", -1))


def voice_base(record, bases) -> dict:
    """Which installed base a voice made on the base of record runs on: {"base_id",
    "base_name", "base_status", "base_note" (a one-line remark for a voice made on another
    commit of its base), "base_problem" (why it cannot run)}."""
    match = match_record(record, bases)
    normalized = normalize_record(record) or {}
    note = None
    if match.status == "compatible" and normalized.get("revision"):
        note = t("base_models.trained_on_other_version")
    problem = None
    if match.base is None:
        problem = missing_message(match, record)
    return {
        "base_id": match.base.id if match.base else None,
        "base_name": match.base.name if match.base else normalized.get("name"),
        "base_status": match.status,
        "base_note": note,
        "base_problem": problem,
    }


# ----- import -----


def header_digest(raw: bytes) -> str:
    """SHA-256 of a safetensors header in canonical JSON. The writer stores its metadata map
    in no fixed order, so the same checkpoint written twice can differ in its header bytes;
    sorted keys give both the same digest."""
    canonical = json.dumps(
        json.loads(raw.decode("utf-8")), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _link_or_copy(source: Path, target: Path, *, cancel=None, progress=None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{identity()}.partial")
    try:
        try:
            os.link(source, partial)
            if progress:
                progress(source.stat().st_size)
        except OSError:
            with source.open("rb") as stream, partial.open("wb") as out:
                while chunk := stream.read(4 * 1024**2):
                    check_cancel(cancel)
                    out.write(chunk)
                    if progress:
                        progress(len(chunk))
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def import_base(source: Path, root: Path, *, cancel=None, progress=None) -> InstalledBase:
    """Register a base model from a folder holding model.safetensors and tokenizer/, or from
    a safetensors file with tokenizer/ beside it. The files are copied (hard linked on the
    same disk) into the resource folder, so the source can be deleted afterwards.
    progress(n) receives the bytes just copied."""
    source = Path(source)
    if source.is_dir():
        folder, model, name = source, source / "model.safetensors", source.name
        if not model.is_file():
            found = sorted(source.glob("*.safetensors"))
            model = found[0] if len(found) == 1 else model
    else:
        folder, model = source.parent, source
        name = source.stem if source.stem.lower() != "model" else source.parent.name
    raw = _header(model) if model.is_file() else None
    architecture = read_architecture(model) if raw is not None else None
    if architecture is None:
        raise BusinessError(t("base_models.import_no_architecture", file=fmt(model.name, "", -1)))
    tokenizer = folder / "tokenizer"
    if not (tokenizer / "tokenizer_config.json").is_file():
        raise BusinessError(t("base_models.import_no_tokenizer", file=fmt(model.name, "", -1)))
    base_id = "imported-" + header_digest(raw)[:12]
    existing = installed_base(root, base_id)
    if existing is not None:
        return existing
    target = Path(root) / IMPORTED_FOLDER / base_id
    stage = target.with_name(f".{base_id}.{identity()}.partial")
    try:
        _link_or_copy(model, stage / "model.safetensors", cancel=cancel, progress=progress)
        for path in sorted(tokenizer.rglob("*")):
            if path.is_file():
                check_cancel(cancel)
                _link_or_copy(
                    path,
                    stage / "tokenizer" / path.relative_to(tokenizer),
                    cancel=cancel,
                    progress=progress,
                )
        if target.exists():
            shutil.rmtree(target)
        stage.replace(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    base = InstalledBase(
        id=base_id,
        name=name or base_id,
        source="imported",
        repo=None,
        revision=None,
        checkpoint=target / "model.safetensors",
        architecture=architecture,
        variants={},
    )
    save_base(root, base)
    return base


def delete_imported(root: Path, base: InstalledBase) -> None:
    """Unregister an imported base and delete its copied files."""
    remove_base(root, base.id)
    folder = Path(root) / IMPORTED_FOLDER / base.id
    if base.source == "imported" and folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)


# ----- catalog downloads -----


def latest_revision(repo: str) -> str:
    """The commit a catalog repository's main branch points to now (network)."""
    from huggingface_hub import HfApi

    return HfApi().model_info(repo).sha


def _spec(repo: str, revision: str, files: tuple[str, ...], size: int = 0):
    from .resources import ModelSpec

    return ModelSpec(repo, revision, tuple(files), size)


def cached_revision(repo: str, files: tuple[str, ...]) -> str | None:
    """A commit of the repository whose files are all in the cache, without network access:
    the one main points to when there is one, else the newest."""
    from .resources import cached_snapshot, hf_home

    folder = hf_home() / "hub" / ("models--" + repo.replace("/", "--"))
    candidates = [path for path in (folder / "snapshots").glob("*") if path.is_dir()]
    main = folder / "refs" / "main"
    preferred = main.read_text(encoding="utf-8").strip() if main.is_file() else None
    candidates.sort(key=lambda path: (path.name != preferred, -path.stat().st_mtime))
    for path in candidates:
        if cached_snapshot(_spec(repo, path.name, files)) is not None:
            return path.name
    return None


def _register_catalog(root: Path, entry: CatalogBase, revision: str, folder: Path) -> InstalledBase:
    checkpoint = folder / "model.safetensors"
    architecture = read_architecture(checkpoint)
    if architecture is None:
        raise BusinessError(
            t("base_models.import_no_architecture", file=fmt(checkpoint.name, "", -1))
        )
    previous = installed_base(root, entry.id)
    base = InstalledBase(
        id=entry.id,
        name=entry.name,
        source="catalog",
        repo=entry.repo,
        revision=revision,
        checkpoint=checkpoint,
        architecture=architecture,
        variants=dict(previous.variants) if previous else {},
    )
    save_base(root, base)
    return base


def adopt_cached_bases(root: Path) -> list[InstalledBase]:
    """Register the catalog bases and variants the cache already holds in full. No network."""
    from .resources import cached_snapshot

    adopted = []
    for entry in CATALOG:
        base = installed_base(root, entry.id)
        if base is None:
            revision = cached_revision(entry.repo, entry.files)
            if revision is None:
                continue
            folder = cached_snapshot(_spec(entry.repo, revision, entry.files))
            try:
                base = _register_catalog(root, entry, revision, folder)
            except BusinessError:
                continue
            adopted.append(base)
        for variant in entry.variants:
            if variant.key in base.variants:
                continue
            revision = cached_revision(variant.repo, variant.files)
            if revision is not None:
                folder = cached_snapshot(_spec(variant.repo, revision, variant.files))
                add_variant(root, base, variant.key, folder / variant.model_file)
                base = installed_base(root, entry.id)
    return adopted


def add_variant(root: Path, base: InstalledBase, key: str, checkpoint: Path) -> InstalledBase:
    base = InstalledBase(**{**base.__dict__, "variants": {**base.variants, key: Path(checkpoint)}})
    save_base(root, base)
    return base


def download_catalog_base(
    root: Path, base_id: str, *, update=False, cancel=None, progress=None
) -> InstalledBase:
    """Download a catalog base at its repository's latest commit and register it. With
    update, an installed base moves to the latest commit and the old commit's files are
    removed; without it, an installed base is left as it is. progress(n) receives bytes."""
    from .resources import fetch_snapshot, remove_snapshot

    entry = catalog_base(base_id)
    if entry is None:
        raise BusinessError(t("base_models.unknown_catalog_base", name=fmt(base_id, "", -1)))
    previous = installed_base(root, base_id)
    if previous is not None and not update:
        return previous
    revision = latest_revision(entry.repo)
    if previous is not None and previous.revision == revision:
        return previous
    folder = fetch_snapshot(
        _spec(entry.repo, revision, entry.files), cancel=cancel, progress=progress
    )
    base = _register_catalog(root, entry, revision, folder)
    if previous is not None and previous.revision and previous.revision != revision:
        remove_snapshot(_spec(entry.repo, previous.revision, entry.files))
    return base


def download_variant(
    root: Path, base_id: str, key: str, *, cancel=None, progress=None
) -> InstalledBase:
    """Download a generation variant of an installed catalog base at its latest commit."""
    from .resources import fetch_snapshot

    entry = catalog_base(base_id)
    variant = entry.variant(key) if entry else None
    base = installed_base(root, base_id)
    if variant is None or base is None:
        raise BusinessError(t("base_models.variant_needs_base", name=fmt(base_id, "", -1)))
    revision = latest_revision(variant.repo)
    folder = fetch_snapshot(
        _spec(variant.repo, revision, variant.files), cancel=cancel, progress=progress
    )
    return add_variant(root, base, key, folder / variant.model_file)


def remove_catalog_files(root: Path, base: InstalledBase) -> None:
    """Unregister a catalog base and remove its commit and its variants from the cache."""
    from .resources import remove_snapshot

    entry = base.catalog
    remove_base(root, base.id)
    if entry is None:
        return
    if base.revision:
        remove_snapshot(_spec(entry.repo, base.revision, entry.files))
    for variant in entry.variants:
        path = base.variants.get(variant.key)
        if path is not None:
            # The cache keeps a file at snapshots/<commit>/<model_file>.
            commit = Path(path).parents[len(Path(variant.model_file).parts) - 1].name
            remove_snapshot(_spec(variant.repo, commit, variant.files))
