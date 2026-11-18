from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

from irodori_tts.desktop.messages import BusinessError, fmt, t

from . import APP_VERSION
from .base_models import CATALOG, architecture_of_config, normalize_record
from .resources import RESOURCE_SPEC, hf_home
from .storage import (
    folder_name,
    free_folder,
    identity,
    load_project,
    now,
    project_lock,
    read_json,
    save_project,
    voice_folder,
    within,
    write_json,
)


def export_audio_file(source: Path, destination: Path):
    temporary = destination.with_name(f".{destination.name}.{identity()}.partial")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_bases(root: Path) -> list[dict]:
    """The base records of a project's runs, each once."""
    bases = []
    for path in sorted((root / "runs").glob("*/run.json")):
        record = normalize_record((read_json(path, {}) or {}).get("base"))
        if record is not None and record not in bases:
            bases.append(record)
    return bases


def export_project(root: Path, destination: Path):
    with project_lock(root):
        load_project(root)
        bases = run_bases(root)
        temporary = destination.with_name(f".{destination.name}.{identity()}.partial")
        try:
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for path in root.rglob("*"):
                    relative = path.relative_to(root)
                    within(root, relative)
                    # A package saved into the project folder must not include itself.
                    if path in {temporary, destination}:
                        continue
                    if path.is_file() and relative.parts[0] not in {"temp", ".project.lock"}:
                        archive.write(path, relative.as_posix())
                archive.writestr(
                    "package.json",
                    json.dumps(
                        {
                            "kind": "project",
                            "format_version": 1,
                            "base": bases,
                            "created": now(),
                        },
                        ensure_ascii=False,
                    ),
                )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def import_project(archive_path: Path, parent: Path, *, taken_projects=(), taken_runs=()):
    """Unpack a project package into a new folder under parent. A package whose project or
    run IDs are already in use here (a backup imported next to its original) gets new IDs,
    so the copy never replaces the original's registration or voices. The folder is named
    after the project."""
    with zipfile.ZipFile(archive_path) as archive:
        try:
            name = json.loads(archive.read("project.json")).get("name")
        except (KeyError, ValueError, AttributeError):
            name = None
    parent.mkdir(parents=True, exist_ok=True)
    root = free_folder(parent, folder_name(name or archive_path.stem))
    root.mkdir()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            package = json.loads(archive.read("package.json"))
            if package["kind"] != "project" or package["format_version"] != 1:
                raise BusinessError(t("exports.unsupported_project_package_format"))
            for info in archive.infolist():
                target = within(root, info.filename)
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise BusinessError(t("exports.project_package_contains_symbolic_links"))
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
        project = load_project(root)
        if project["id"] in set(taken_projects):
            project["id"] = identity()
            save_project(root, project)
        taken_runs = set(taken_runs)
        for record in sorted((root / "runs").glob("*/run.json")):
            run = read_json(record)
            if run.get("id") in taken_runs:
                renumber_run(root, record.parent, run)
        return root
    except BaseException:
        shutil.rmtree(root)
        raise


def renumber_run(root: Path, run_root: Path, run: dict) -> Path:
    """Give an imported run a new ID; its folder keeps its name."""
    run["id"] = identity()
    write_json(run_root / "run.json", run)
    return run_root


def resolved_base(run: dict) -> dict:
    """The run's base record with its architecture, which a voice needs to find its base."""
    base = normalize_record(run.get("base"))
    if base is None or not base.get("architecture"):
        raise BusinessError(t("exports.unable_to_resolve_the_base_model"))
    return base


def export_voice(root: Path, run: dict, destination: Path):
    with project_lock(root):
        checkpoint = within(root, run["checkpoint"])
        base = resolved_base(run)
        temporary = destination.with_name(f".{destination.name}.{identity()}.partial")
        try:
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
                for name in (
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "adapter_model.bin",
                    "config.json",
                    "irodori_lora_metadata.json",
                ):
                    path = checkpoint / name
                    if path.is_file():
                        if name == "irodori_lora_metadata.json":
                            archive.writestr(name, json.dumps({"base_resource": base}))
                        else:
                            archive.write(path, name)
                # The inference settings tuned for this model travel with it; reference
                # recordings are files of this computer and stay behind.
                settings = checkpoint.parents[1] / "inference_settings.json"
                if settings.is_file():
                    values = read_json(settings)
                    if isinstance(values, dict):
                        values.pop("references", None)
                        archive.writestr(
                            "inference_settings.json", json.dumps(values, ensure_ascii=False)
                        )
                archive.writestr(
                    "voice.json",
                    json.dumps(
                        {
                            "format_version": 1,
                            "id": run["id"],
                            "name": run["name"],
                            "base": base,
                            "step": run["step"],
                        },
                        ensure_ascii=False,
                    ),
                )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


# ----- voice import -----
#
# A voice arrives as this application's package (a flat zip with voice.json) or as what the
# upstream trainer saves: a checkpoint folder (adapter_config.json, adapter weights, config.json,
# trainer_state.pt), usually shared zipped, 7z'd or rar'd under a folder or two, beside READMEs
# and other files. Every folder holding an adapter is a candidate; only the files inference
# needs are taken, and anything else in the package is left behind.

ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
# What a voice keeps besides the adapter, when the package has it.
VOICE_EXTRAS = ("config.json", "inference_settings.json")


class _Package:
    """The files of a voice package, by their path inside it (posix): a folder, a zip, or
    anything 7-Zip opens (7z, rar)."""

    def __init__(self, source: Path, stage: Path | None = None):
        self.source, self.stage = source, stage
        if source.is_dir():
            self.kind = "folder"
            self.names = [
                path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()
            ]
        elif zipfile.is_zipfile(source):
            self.kind = "zip"
            with zipfile.ZipFile(source) as archive:
                self.names = [info.filename for info in archive.infolist() if not info.is_dir()]
        else:
            self.kind = "7z"
            self.names = [
                name for name, _size in _seven_zip_listing(source) if not name.endswith("/")
            ]
        self.names = [name.replace("\\", "/") for name in self.names]

    def copy_tree(self, names: list[str], destination: Path):
        """Copy these files into destination at their paths inside the package."""
        if self.kind == "7z":
            _seven_zip_extract(self.source, names, destination)
            return
        for name in names:
            target = within(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            self.copy([name], target.parent)

    def copy(self, names: list[str], destination: Path):
        """Copy these files into destination, by their own names (no folders)."""
        destination.mkdir(parents=True, exist_ok=True)
        if self.kind == "folder":
            for name in names:
                shutil.copyfile(within(self.source, name), destination / Path(name).name)
        elif self.kind == "zip":
            with zipfile.ZipFile(self.source) as archive:
                for name in names:
                    with (
                        archive.open(name) as stream,
                        (destination / Path(name).name).open("wb") as out,
                    ):
                        shutil.copyfileobj(stream, out)
        else:
            unpacked = self.stage / "unpacked"
            _seven_zip_extract(self.source, names, unpacked)
            for name in names:
                within(unpacked, name).replace(destination / Path(name).name)


def _seven_zip_run(source: Path, arguments: list[str], names: tuple[str, ...] = ()) -> str:
    from .model_files import NO_WINDOW, seven_zip

    program = seven_zip()
    if program is None:
        raise BusinessError(t("models.import_needs_7zip"))
    result = subprocess.run(
        [program, *arguments, "-sccUTF-8", "--", str(source), *names],
        capture_output=True,
        creationflags=NO_WINDOW,
    )
    if result.returncode:
        lines = [
            line.strip()
            for line in result.stderr.decode("utf-8", "replace").splitlines()
            if line.strip()
        ]
        raise BusinessError(
            t(
                "exports.voice_package_unreadable",
                name=fmt(source.name, "", -1),
                reason=fmt(lines[-1] if lines else "", "", -1),
            )
        )
    return result.stdout.decode("utf-8", "replace").replace("\r\n", "\n")


def _seven_zip_listing(source: Path) -> list[tuple[str, int]]:
    entries = []
    for block in _seven_zip_run(source, ["l", "-slt", "-ba"]).split("\n\n"):
        fields = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
        if "Path" in fields and fields.get("Folder") != "+":
            entries.append((fields["Path"], int(fields.get("Size") or 0)))
    return entries


def _seven_zip_extract(source: Path, names: list[str], destination: Path):
    # -spd: the names are paths, not wildcards (a voice folder may be called "[v2]").
    _seven_zip_run(source, ["x", "-y", "-bd", "-spd", f"-o{destination}"], tuple(names))


def _adapter_folders(names: list[str]) -> list[str]:
    """The folders of the package that hold an adapter ("" for its top)."""
    present = set(names)
    folders = []
    for name in names:
        path = PurePosixPath(name)
        if path.name != "adapter_config.json":
            continue
        folder = "" if str(path.parent) == "." else str(path.parent)
        if any(_inside(folder, weights) in present for weights in ADAPTER_WEIGHTS):
            folders.append(folder)
    return sorted(set(folders))


def _inside(folder: str, name: str) -> str:
    return f"{folder}/{name}" if folder else name


def _package_source(source: Path) -> Path:
    # The import dialog also accepts an adapter_config.json: its folder is the voice.
    return source.parent if source.name == "adapter_config.json" else source


def voice_candidates(source: Path, parent: Path, bases=()) -> list[dict]:
    """The adapters of a voice package: {"folder", "name", "problem"} for each folder holding
    one, problem being why it cannot be used here (None when it can). The user chooses between
    the usable ones when there are several (several checkpoints, or several sizes of a voice);
    the metadata of each is read, its weights are not."""
    source = _package_source(source)
    parent.mkdir(parents=True, exist_ok=True)
    stage = within(parent, f".import-{identity()}.partial")
    stage.mkdir()
    try:
        package = _Package(source, stage)
        folders = _package_folders(package, source)
        present = set(package.names)
        metadata = ("voice.json", "config.json", "base_model.json")
        package.copy_tree(
            [_inside(f, n) for f in folders for n in metadata if _inside(f, n) in present],
            stage / "meta",
        )
        return [
            {
                "folder": folder,
                "name": _voice_name(source, folder, len(folders) > 1),
                "problem": _incompatibility(
                    within(stage / "meta", folder or "."), bases, final=False
                ),
            }
            for folder in folders
        ]
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _package_folders(package: _Package, source: Path) -> list[str]:
    folders = _adapter_folders(package.names)
    if not folders:
        raise BusinessError(t("exports.no_voice_adapter_in_package", name=fmt(source.name, "", -1)))
    return folders


def _incompatibility(folder: Path, bases, *, final: bool):
    """Why the adapter whose files are in folder cannot be imported, or None. A voice is kept
    when its architecture is known, even when no installed base has it yet: Generate then
    names the base to download or import. Without the weights at hand (final=False) an
    adapter nothing describes passes."""
    if final and _voice_base(folder, bases) is None:
        return t("exports.voice_base_model_unknown")
    return None


def _meaningful(name: str) -> bool:
    return bool(name) and not re.match(r"checkpoint|final$|best", name, re.IGNORECASE)


def _voice_name(source: Path, folder: str, several: bool) -> str:
    """A name for a package that carries none: the package's, or the folder above a generic
    "checkpoint_final"; with several adapters, the part of the path telling them apart."""
    name = source.stem if source.is_file() else source.name
    if not _meaningful(name):
        name = next((parent.name for parent in source.parents if _meaningful(parent.name)), name)
    if several:
        part = next((p for p in reversed(PurePosixPath(folder).parts) if _meaningful(p)), None)
        if part and part != name:
            return f"{name} {part}"
    return name


def adapter_architecture(folder: Path, bases=()) -> dict | None:
    """The architecture of the base the adapter in folder was trained on, from its own files
    (voice.json, the trainer's config.json, else its tensor shapes); None when they do not
    tell. Only metadata and the safetensors header are read."""
    record = _voice_base(Path(folder), bases) if Path(folder).is_dir() else None
    return record["architecture"] if record else None


def _voice_base(folder: Path, bases) -> dict | None:
    """The base record of the adapter in folder, or None when nothing tells its architecture.
    This application's packages record their base; the trainer's config.json records the
    model; failing both, the adapter's tensor shapes identify a known base by width and depth.
    """
    voice = read_json(folder / "voice.json") if (folder / "voice.json").is_file() else None
    if isinstance(voice, dict):
        record = normalize_record(voice.get("base"))
        if record is not None and record.get("architecture"):
            return record
    config = read_json(folder / "config.json") if (folder / "config.json").is_file() else None
    model = config.get("model") if isinstance(config, dict) else None
    architecture = architecture_of_config(model) if isinstance(model, dict) else None
    weights = folder / "adapter_model.safetensors"
    if architecture is None and weights.is_file():
        found = _safetensors_blocks(weights)
        known = [entry.architecture for entry in CATALOG] + [b.architecture for b in bases]
        shaped = [
            candidate
            for candidate in known
            if found
            and candidate["model_dim"] == found["model_dim"]
            and candidate["num_layers"] == found["num_layers"]
        ]
        if shaped and all(candidate == shaped[0] for candidate in shaped):
            architecture = shaped[0]
    if architecture is None:
        return None
    entry = next((entry for entry in CATALOG if entry.architecture == architecture), None)
    # Who trained it and on which commit is not known: the record names only the shape (and
    # the catalog base of that shape, for the download hint).
    return {
        "id": entry.id if entry else None,
        "name": entry.name if entry else None,
        "repo": entry.repo if entry else None,
        "revision": None,
        "architecture": architecture,
    }


def _safetensors_blocks(path: Path) -> dict | None:
    """The width and depth of the diffusion blocks an adapter adapts, from its header."""
    try:
        with path.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if not 0 < size < 64 * 1024**2:
                return None
            header = json.loads(stream.read(size))
    except (OSError, ValueError):
        return None
    widths, layers = set(), set()
    for key, info in header.items():
        match = re.search(r"(?:^|\.)blocks\.(\d+)\.attention\.(?:wq|wk|wv)\.lora_A\.", key)
        if match and isinstance(info, dict) and len(info.get("shape", ())) == 2:
            layers.add(int(match[1]))
            widths.add(info["shape"][1])
    if not widths:
        return None
    return {"model_dim": max(widths), "num_layers": max(layers) + 1}


def import_voice(source: Path, parent: Path, folder: str | None = None, bases=()):
    """Import a voice from a package; folder is the one of voice_candidates to take, needed
    when the package holds several adapters. bases are the installed bases, which also
    identify an adapter by its shape."""
    source = _package_source(source)
    parent.mkdir(parents=True, exist_ok=True)
    stage = within(parent, f".import-{identity()}.partial")
    stage.mkdir()
    try:
        package = _Package(source, stage)
        folders = _package_folders(package, source)
        if folder is None:
            if len(folders) > 1:
                raise BusinessError(t("exports.choose_one_voice_in_package"))
            folder = folders[0]
        if folder not in folders:
            raise BusinessError(
                t("exports.no_voice_adapter_in_package", name=fmt(source.name, "", -1))
            )
        present = set(package.names)
        weights = next(w for w in ADAPTER_WEIGHTS if _inside(folder, w) in present)
        wanted = ["adapter_config.json", weights, *VOICE_EXTRAS, "voice.json", "base_model.json"]
        voice_root = stage / "voice"
        taken = [_inside(folder, name) for name in wanted if _inside(folder, name) in present]
        package.copy(taken, voice_root)

        base = _voice_base(voice_root, bases)
        if base is None:
            raise BusinessError(t("exports.voice_base_model_unknown"))
        voice = (
            read_json(voice_root / "voice.json") if (voice_root / "voice.json").is_file() else None
        )
        if not isinstance(voice, dict):
            voice = {
                "format_version": 1,
                "name": _voice_name(source, folder, len(folders) > 1),
            }
            match = re.search(r"checkpoint_(?:best_val_loss_)?(\d+)", folder)
            if match:
                voice["step"] = int(match[1])
        voice["base"] = base
        (voice_root / "base_model.json").unlink(missing_ok=True)
        root = voice_folder(parent, voice.get("name") or source.stem)
        voice.setdefault("id", root.name)
        voice.setdefault("created", now())
        voice["registered"] = True
        write_json(voice_root / "voice.json", voice)
        # The trainer's metadata names a base checkpoint by a path of its own computer.
        write_json(
            voice_root / "irodori_lora_metadata.json",
            {"base_resource": base},
        )
        voice_root.replace(root)
        return root
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def register_checkpoint_voice(root: Path, run: dict, checkpoint: Path, parent: Path, name: str):
    """Keep an inference-only snapshot independently of trainer checkpoint retention."""
    with project_lock(root):
        from .training import complete_checkpoint

        relative = checkpoint.relative_to(root)
        within(root, relative)
        if not name.strip() or not complete_checkpoint(checkpoint):
            raise BusinessError(t("exports.select_a_complete_checkpoint_and_enter_a"))
        base = normalize_record(run.get("base"))
        if base is None or not base.get("architecture"):
            raise BusinessError(t("exports.unable_to_resolve_the_checkpoint_base_model"))
        parent.mkdir(parents=True, exist_ok=True)
        voice_id = voice_folder(parent, name.strip()).name
        stage = within(parent, f".{voice_id}.partial")
        stage.mkdir()
        marker = read_json(checkpoint / "complete.json")
        step = marker.get("step")
        if step is None:
            match = re.search(r"checkpoint_(?:best_val_loss_)?(\d+)", checkpoint.name)
            step = int(match[1]) if match else run["step"]
        try:
            for filename in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "adapter_model.bin",
                "config.json",
            ):
                source = within(root, relative / filename)
                if source.is_file():
                    shutil.copyfile(source, stage / filename)
            write_json(stage / "irodori_lora_metadata.json", {"base_resource": base})
            write_json(
                stage / "voice.json",
                {
                    "format_version": 1,
                    "id": voice_id,
                    "name": name.strip(),
                    "base": base,
                    "step": step,
                    "created": now(),
                    "registered": True,
                    "project_name": load_project(root)["name"],
                    "source_run_id": run["id"],
                    "source_checkpoint": relative.as_posix(),
                },
            )
            destination = parent / voice_id
            stage.replace(destination)
            return destination
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def redact(value):
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if re.search(
                    r"token|password|secret|proxy|caption|text|request|source|audio", key, re.I
                )
                else redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", value)
        value = re.sub(r"\bhf_[a-zA-Z0-9]+\b", "[REDACTED]", value)
        value = re.sub(
            r"(?i)(bearer\s+|token[=: ]+|password[=: ]+)[^\s,;]+", r"\1[REDACTED]", value
        )
        return value
    return value


def cuda_of(torch_version):
    """The CUDA version a PyTorch wheel was built for, from its local tag (2.7.1+cu128)."""
    tag = str(torch_version).partition("+cu")[2]
    return f"{tag[:-1]}.{tag[-1]}" if tag.isdigit() and len(tag) > 1 else None


def cpu_name():
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (ImportError, OSError):
        return platform.processor()


def system_name():
    if platform.system() == "Windows":
        build = int(platform.version().rpartition(".")[2] or 0)
        return "Windows 11" if build >= 22000 else f"Windows {platform.release()}"
    return f"{platform.system()} {platform.release()}"


def device_report(*, timeout=15):
    """What the app runs on: GPU and driver from nvidia-smi, versions, system. Never raises, and
    never imports torch (the GUI process does not carry it)."""
    report = {
        "gpu": None,
        "gpu_error": None,
        "torch": None,
        "python": platform.python_version(),
        "system": system_name(),
        "cpu": cpu_name(),
        "memory": None,
    }
    try:
        report["torch"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        import psutil

        report["memory"] = psutil.virtual_memory().total
    except ImportError:
        pass
    fields = "name,driver_version,memory.total,memory.used"
    try:
        for query in (fields + ",compute_cap", fields):
            probe = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                parts = [part.strip() for part in probe.stdout.splitlines()[0].split(",")]
                report["gpu"] = {
                    "name": parts[0],
                    "driver": parts[1],
                    "total_mib": int(float(parts[2])),
                    "used_mib": int(float(parts[3])),
                    "compute_cap": float(parts[4]) if len(parts) > 4 else None,
                }
                break
        else:
            report["gpu_error"] = t(
                "settings_page.gpu_driver_unavailable_install_a_compatible_nvidia"
            )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        report["gpu_error"] = t("settings_page.nvidia_driver_diagnostic_tool_not_found_check")
    return report


def environment_report():
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    return {
        "app_version": APP_VERSION,
        "python": platform.python_version(),
        "system": platform.platform(),
        "packages": packages,
        "resources": RESOURCE_SPEC,
    }


def export_diagnostics(store, destination: Path, *, include_logs=False):
    # No recordings, transcripts, weights or raw request files are included.
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "environment.json", json.dumps(environment_report(), ensure_ascii=False, indent=2)
        )
        for path in (store.root / "tasks").glob("*/task.json"):
            task = redact(read_json(path))
            archive.writestr(
                f"tasks/{path.parent.name}.json", json.dumps(task, ensure_ascii=False, indent=2)
            )
            if include_logs and (path.parent / "worker.log").exists():
                text = (path.parent / "worker.log").read_text(encoding="utf-8", errors="replace")
                archive.writestr(f"logs/{path.parent.name}.log", redact(text))


def storage_entries(store):
    entries = []
    cache = hf_home() / "hub"
    if cache.exists():
        for path in cache.glob("models--*/blobs/*.incomplete"):
            within(cache, path.relative_to(cache))
            entries.append(
                {
                    "id": str(path),
                    "path": str(path),
                    "kind_id": "partial_downloads",
                    "kind": t("exports.partial_downloads"),
                    "project": t("exports.model_resources"),
                    "bytes": path.stat().st_size,
                    "root": str(cache),
                    "relative": path.relative_to(cache).as_posix(),
                }
            )
    for root, project in store.projects():
        for directory in (root / "temp").glob("*"):
            entries.append(
                {
                    "id": str(directory),
                    "path": str(directory),
                    "kind_id": "temporary_files",
                    "kind": t("exports.application_temporary_files"),
                    "project": project["name"],
                    "bytes": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
                    "root": str(root),
                    "relative": directory.relative_to(root).as_posix(),
                }
            )
        for path in (root / "runs").glob("*/checkpoints/*"):
            if path.is_dir():
                run = read_json(path.parent.parent / "run.json", {})
                referenced = run.get("checkpoint") == path.relative_to(root).as_posix()
                if run.get("checkpoint"):
                    current = Path(run["checkpoint"])
                    referenced = referenced or path.relative_to(root) == current.with_name(
                        f".{current.name}.previous"
                    )
                entries.append(
                    {
                        "id": str(path),
                        "path": str(path),
                        "kind_id": "current_checkpoint" if referenced else "historical_checkpoints",
                        "kind": t("exports.current_recovery_checkpoint")
                        if referenced
                        else t("exports.historical_checkpoints"),
                        "project": project["name"],
                        "bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
                        "root": str(root),
                        "relative": path.relative_to(root).as_posix(),
                        "referenced": referenced,
                    }
                )
    return entries
