from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from irodori_tts.desktop.messages import BusinessError, error_message, fmt, t
from irodori_tts.task_control import check_cancel

from .storage import digest, read_json, write_json

MODEL_REPO = "Aratako/Irodori-TTS-v4.1-Small"
MODEL_REVISION = "2b28324dc263ed5e6638b3cf3dd94c82ead07b4b"
# The speech model's shape, from its config_json metadata. An adapter trained for another model
# (another size, another text encoder) records other values and cannot load onto this one.
MODEL_ARCHITECTURE = {
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
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
CODEC_REVISION = "47376ee24834d7a05a48ebabfe3cde29b3c5e214"
WATERMARK_REVISION = "a1c4d021905e0dc5b24be5f68db5fc4dba410ee1"
# Generation only: about a quarter less video memory, and the same speech at the same seed.
LOW_VRAM_REPO = "Aratako/Irodori-TTS-v4.1-Small-Quantized"
LOW_VRAM_REVISION = "ef04e6c3ba56138ae23e86a2eabc004f76990e37"
LOW_VRAM_MODEL = "int8-weight-only/model.safetensors"
LOW_VRAM_FILES = (
    LOW_VRAM_MODEL,
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)
# Generation only: the MeanFlow distillation of the speech model, four steps instead of forty.
# Its guidance is trained in, so the guidance settings do not apply.
FAST_REPO = "Aratako/Irodori-TTS-v4.1-Small-MF"
FAST_REVISION = "ccc78f5d480b6e51b69b2d5042a14c4da04fea6e"
FAST_MODEL = "model.safetensors"
FAST_FILES = (
    FAST_MODEL,
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)
FAST_STEPS = 4
RESOURCE_SPEC = {
    "model": {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "files": [
            "model.safetensors",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        ],
    },
    "codec": {"repo": CODEC_REPO, "revision": CODEC_REVISION, "files": ["weights.pth"]},
    "watermark": {
        "repo": "sony/silentcipher",
        "revision": WATERMARK_REVISION,
        "files": [
            f"44_1_khz/73999_iteration/{name}"
            for name in ("enc_c.ckpt", "dec_c.ckpt", "dec_m_0.ckpt", "hparams.yaml")
        ],
    },
}


# Files at least this large are downloaded resumably (see resumable_blob).
LARGE_FILE = 64 * 1024**2
CHUNK = 1024**2


def resumable_blob(repo, filename, revision, *, cancel=None, progress=None) -> None:
    """Download a large file into the Hugging Face cache so that a stop or a dropped connection
    continues where it ended. huggingface_hub deletes a partial download on any failure, which
    would restart gigabytes from zero; the finished blob is where hf_hub_download looks first, and
    it then only links it into the snapshot. progress(n) receives bytes, including those a
    previous attempt already has."""
    from huggingface_hub import constants, get_hf_file_metadata, hf_hub_url, try_to_load_from_cache
    from huggingface_hub.file_download import repo_folder_name
    from huggingface_hub.utils import build_hf_headers, get_session

    # A file already in the cache needs no network at all (and works offline).
    cached = try_to_load_from_cache(repo, filename, revision=revision, cache_dir=hf_home() / "hub")
    if isinstance(cached, str) and Path(cached).is_file():
        return
    url = hf_hub_url(repo, filename, revision=revision)
    metadata = get_hf_file_metadata(url)
    size, etag = metadata.size, metadata.etag
    if not size or size < LARGE_FILE or not etag:
        return
    blobs = hf_home() / "hub" / repo_folder_name(repo_id=repo, repo_type="model") / "blobs"
    blob = blobs / etag
    if blob.exists():
        return
    blobs.mkdir(parents=True, exist_ok=True)
    partial = blobs / f"{etag}.partial"
    done = partial.stat().st_size if partial.exists() else 0
    if done > size:
        partial.unlink()
        done = 0
    if progress and done:
        progress(done)
    if done < size:
        headers = build_hf_headers()
        if done:
            headers["Range"] = f"bytes={done}-"
        with get_session().stream(
            "GET",
            url,
            headers=headers,
            follow_redirects=True,
            timeout=constants.HF_HUB_DOWNLOAD_TIMEOUT,
        ) as response:
            response.raise_for_status()
            # A server that ignores the range sends the whole file again.
            append = done and response.status_code == 206
            with partial.open("ab" if append else "wb") as stream:
                for chunk in response.iter_bytes(CHUNK):
                    check_cancel(cancel)
                    stream.write(chunk)
                    if progress:
                        progress(len(chunk))
    if partial.stat().st_size != size:
        raise BusinessError(t("resources.download_incomplete", file=fmt(filename, "", -1)))
    # Large files are stored by their SHA-256, which is their ETag.
    if re.fullmatch(r"[0-9a-f]{64}", etag) and digest(partial) != etag:
        partial.unlink()
        raise BusinessError(t("resources.download_corrupt", file=fmt(filename, "", -1)))
    partial.replace(blob)


def resolve_resources(
    root: Path,
    *,
    download=False,
    verify=False,
    progress=None,
    cancel=None,
    repair=False,
    received=None,
) -> dict:
    """The speech resources' local paths, downloading missing files when asked. progress gets
    a message per file; received(n) the bytes a download just added."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from tqdm.auto import tqdm

    class DownloadProgress(tqdm):
        def update(self, amount=1):
            check_cancel(cancel)
            if received and amount:
                received(int(amount))
            return super().update(amount)

    result = {}
    old = read_json(root / "resource-manifest.json", {})
    manifest = {"resources": RESOURCE_SPEC, "files": {}}
    for name, spec in RESOURCE_SPEC.items():
        for filename in spec["files"]:
            check_cancel(cancel)
            if progress:
                progress(t("resources.checking", p1=fmt(name, "", -1), p3=fmt(filename, "", -1)))
            local = root / name / filename
            if not local.exists() or repair:
                try:
                    if download and not repair:
                        resumable_blob(
                            spec["repo"],
                            filename,
                            spec["revision"],
                            cancel=cancel,
                            progress=received,
                        )
                    local = Path(
                        hf_hub_download(
                            spec["repo"],
                            filename,
                            revision=spec["revision"],
                            local_files_only=not download,
                            force_download=repair,
                            tqdm_class=DownloadProgress,
                        )
                    )
                except LocalEntryNotFoundError as exc:
                    if verify:
                        raise BusinessError(
                            t("resources.verify_found_missing_file", file=fmt(filename, "", -1))
                        ) from exc
                    raise BusinessError(
                        t("resources.resources_are_not_ready_choose_prepare_resources")
                        + error_message(exc)
                    ) from exc
            key = f"{name}/{filename}"
            known = (old.get("files", {}).get(key) or {}).get("sha256")
            checksum = digest(local) if verify or repair or not known else known
            if verify and not repair and known and checksum != known:
                raise BusinessError(
                    t(
                        "resources.resource_verification_failed_prepare_resources_again",
                        p1=fmt(filename, "", -1),
                    )
                )
            manifest["files"][key] = {
                "path": stored_path(local),
                "size": local.stat().st_size,
                "sha256": checksum,
            }
            if filename in {"model.safetensors", "weights.pth"}:
                result[name] = str(local)
    write_json(root / "resource-manifest.json", manifest)
    pin_watermark_revision()
    return result


def pin_watermark_revision() -> None:
    """Point the cache's main at the pinned revision. silentcipher asks for main with no way to
    name a revision, and offline it finds a snapshot only through this file, which downloads by
    commit do not write."""
    from huggingface_hub.file_download import repo_folder_name

    folder = repo_folder_name(repo_id=RESOURCE_SPEC["watermark"]["repo"], repo_type="model")
    ref = hf_home() / "hub" / folder / "refs/main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(WATERMARK_REVISION, encoding="utf-8")


def hf_home() -> Path:
    """The Hugging Face cache the resources are downloaded into."""
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")


def stored_path(path: Path) -> str:
    """Manifest paths are relative to the cache, so a moved application folder still finds
    them; a file outside the cache keeps its absolute path."""
    try:
        return Path(path).resolve().relative_to(hf_home().resolve()).as_posix()
    except ValueError:
        return str(path)


def manifest_file(value: str) -> Path:
    """Resolve a manifest path, including absolute paths written before the cache moved."""
    path = Path(value)
    if not path.is_absolute():
        return hf_home() / path
    if not path.exists() and "hub" in path.parts:
        return hf_home().joinpath(*path.parts[path.parts.index("hub") :])
    return path


def resource_state(root: Path) -> tuple[bool, float | None]:
    """(ready, size in GB or None when nothing is recorded) from the manifest, by size only."""
    manifest = read_json(root / "resource-manifest.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        return False, None
    entries = manifest["files"]
    expected = {
        f"{name}/{filename}" for name, spec in RESOURCE_SPEC.items() for filename in spec["files"]
    }
    ready = expected <= set(entries)
    size = 0
    for entry in entries.values():
        try:
            path = manifest_file(entry["path"])
            size += entry["size"]
            ready = ready and path.is_file() and path.stat().st_size == entry["size"]
        except (KeyError, TypeError, OSError):
            ready = False
    # Decimal gigabytes, like every other size the application shows.
    return ready, size / 1000**3


def adopt_cached_resources(root: Path) -> bool:
    """Record resources that the cache already holds, without network access or hashing, so
    a new data folder does not ask for a download the cache has already satisfied."""
    from huggingface_hub import try_to_load_from_cache

    old = read_json(root / "resource-manifest.json", {}) or {}
    manifest = {"resources": RESOURCE_SPEC, "files": {}}
    for name, spec in RESOURCE_SPEC.items():
        for filename in spec["files"]:
            found = try_to_load_from_cache(
                spec["repo"], filename, revision=spec["revision"], cache_dir=hf_home() / "hub"
            )
            if not isinstance(found, str) or not Path(found).is_file():
                return False
            key = f"{name}/{filename}"
            local = Path(found)
            manifest["files"][key] = {
                "path": stored_path(local),
                "size": local.stat().st_size,
                # Hashing happens in the resource task (worker process), not here.
                "sha256": (old.get("files", {}).get(key) or {}).get("sha256"),
            }
    write_json(root / "resource-manifest.json", manifest)
    return True


def cache_root(repo: Path | None = None) -> Path:
    """Where downloaded resources and add-ons live: the resource pack of a packaged launcher,
    else the checkout's .cache."""
    repo = repo or Path(__file__).resolve().parents[2]
    return Path(os.environ.get("IRODORI_CACHE", str(repo / ".cache")))


def configure_environment(repo: Path) -> None:
    # Packaged launchers set IRODORI_CACHE to the independently updatable resource pack.
    cache = cache_root(repo)
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    # cuDNN attention compiles an execution plan for every new sequence shape. Recordings of
    # varying length make that compilation dominate: training ran ~6x slower on 150 clips.
    os.environ.setdefault("IRODORI_SDPA_CUDNN", "0")
    # Xet transfers ignore a stop until the whole file is in, and keep a second copy of every
    # file in their chunk cache (6 GB on disk for a 3 GB model).
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"


def hide_child_consoles() -> None:
    """On Windows, start every child process without a console window unless the caller asks
    for flags of its own. The window-less worker (pythonw) has no console to share, so each
    command a library runs (ffmpeg, nvidia-smi, cmd /c ver) flashed a black window of its own."""
    if os.name != "nt":
        return
    import subprocess

    if getattr(subprocess.Popen, "_hides_consoles", False):
        return
    start = subprocess.Popen.__init__

    def __init__(self, *args, creationflags=0, **kwargs):
        if not creationflags:
            creationflags = subprocess.CREATE_NO_WINDOW
        start(self, *args, creationflags=creationflags, **kwargs)

    subprocess.Popen.__init__ = __init__
    subprocess.Popen._hides_consoles = True


# Windows paths stop at 259 characters unless long paths are enabled for the system. Measured
# deepest files, plus a margin: under the data folder a checkpoint being published, 165
# (projects/<id>/runs/<id>/checkpoints/.ckpt-<id>.partial/irodori_lora_metadata.json); under
# the cache a speech recognition model's snapshot file, 139.
MAX_PATH = 259
DATA_DEPTH = 175
CACHE_DEPTH = 150


def long_paths_enabled() -> bool:
    if os.name != "nt":
        return True
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem"
        ) as key:
            return bool(winreg.QueryValueEx(key, "LongPathsEnabled")[0])
    except OSError:
        return False


def paths_too_long(data_root: Path, cache: Path) -> list[Path]:
    """The folders whose path leaves too little room for the files the application writes."""
    if long_paths_enabled():
        return []
    folders = ((Path(data_root).resolve(), DATA_DEPTH), (Path(cache).resolve(), CACHE_DEPTH))
    return [folder for folder, depth in folders if len(str(folder)) + depth > MAX_PATH]


@dataclass(frozen=True)
class ModelSpec:
    """Files of one Hugging Face repository at an exact revision, with their total size."""

    repo: str
    revision: str
    files: tuple[str, ...]
    size: int


def _snapshot_of(path: Path, filename: str) -> Path:
    folder = Path(path)
    for _ in PurePosixPath(filename).parts:
        folder = folder.parent
    return folder


def cached_snapshot(spec: ModelSpec) -> Path | None:
    """The revision's folder in the cache when every file is there, else None. No network."""
    from huggingface_hub import try_to_load_from_cache

    folder = None
    for filename in spec.files:
        found = try_to_load_from_cache(
            spec.repo, filename, revision=spec.revision, cache_dir=hf_home() / "hub"
        )
        if not isinstance(found, str) or not Path(found).is_file():
            return None
        folder = _snapshot_of(Path(found), filename)
    return folder


def fetch_snapshot(spec: ModelSpec, *, download=True, cancel=None, progress=None) -> Path:
    """Download the pinned files into the cache. Partial files resume on the next call, and a
    cancel stops at the next chunk. progress(n) receives the bytes just received."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from tqdm.auto import tqdm

    class Meter(tqdm):
        def update(self, amount=1):
            check_cancel(cancel)
            result = super().update(amount)
            if progress and amount and self.unit == "B":
                progress(int(amount))
            return result

    folder = None
    for filename in spec.files:
        check_cancel(cancel)
        try:
            if download:
                resumable_blob(spec.repo, filename, spec.revision, cancel=cancel, progress=progress)
            local = Path(
                hf_hub_download(
                    spec.repo,
                    filename,
                    revision=spec.revision,
                    local_files_only=not download,
                    tqdm_class=Meter,
                )
            )
        except LocalEntryNotFoundError as exc:
            raise BusinessError(
                t("resources.resources_are_not_ready_choose_prepare_resources") + error_message(exc)
            ) from exc
        folder = _snapshot_of(local, filename)
    return folder


def snapshot_size(spec: ModelSpec) -> int:
    """Bytes the cached files occupy, 0 when the revision is not (fully) there."""
    folder = cached_snapshot(spec)
    if folder is None:
        return 0
    return sum((folder / name).stat().st_size for name in spec.files)


def remove_snapshot(spec: ModelSpec) -> None:
    """Delete the revision from the cache; blobs another revision shares are kept."""
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir(cache_dir=hf_home() / "hub")
    hashes = [
        revision.commit_hash
        for repo in info.repos
        if repo.repo_id == spec.repo and repo.repo_type == "model"
        for revision in repo.revisions
        if revision.commit_hash == spec.revision
    ]
    if hashes:
        info.delete_revisions(*hashes).execute()
