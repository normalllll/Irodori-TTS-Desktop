from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from irodori_tts.desktop.messages import BusinessError, fmt, t
from irodori_tts.upstream import upstream_file

from . import INITIALIZATION_RULES_VERSION
from .base_models import (
    InstalledBase,
    architecture_of_config,
    base_record,
    describe,
    load_registry,
    match_record,
    missing_message,
    normalize_record,
)
from .materials import materials_fingerprint
from .resources import CODEC_REVISION
from .storage import (
    digest,
    folder_name,
    free_folder,
    identity,
    load_project,
    now,
    project_lock,
    read_json,
    remove_tree,
    within,
    write_json,
)

# The training page leaves these empty until "Generate recommended settings" fills them from
# the card and the recordings; a run does not start while one is empty.
# Automatic planning lives in training_recommendation; normalization only validates a plan.
TUNED = ("max_steps", "batch_size", "gradient_accumulation_steps", "gradient_checkpointing")
RECOMMENDED = {
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    # Random batches were more reliable in our small single-speaker comparisons. Grouping
    # and physical batch changes also change timestep sampling; neither is quality-neutral.
    "length_bucket_enabled": False,
    "num_workers": 0,
    "dataloader_cuda_prefetch": False,
    "precision": "bf16",
    "optimizer": "adamw",
    "learning_rate": 0.0001,
    "lr_scheduler": "cosine",
    "warmup_steps": 0,
    "max_steps": 0,
    "save_every": 1,
    # Off by default: validation holds recordings out of training (one of two, seven of 150)
    # for a loss the application neither shows nor uses to pick checkpoints; listening to the
    # samples is how a checkpoint is judged. A validation interval in the advanced settings
    # turns it on with this ratio.
    "valid_every": 0,
    "valid_ratio": 0.05,
    "seed": 0,
    "lora_enabled": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_target_modules": "diffusion_attn",
    "lora_modules_to_save": "auto",
    "ref_max_seconds": 10.0,
    "max_latent_steps": 300,
    "log_every": 1,
    # 0 keeps every periodic save so they can be compared after training; the worker
    # removes the optimizer state from all but the newest one (see compact_checkpoints).
    "checkpoint_best_n": 0,
    "progress": False,
}


def training_count(count: int, validation: bool, ratio: float = RECOMMENDED["valid_ratio"]) -> int:
    """Recordings a run trains on: validation holds int(count * ratio), at least one, out."""
    if not validation or count < 2:
        return count
    return count - min(count - 1, max(1, int(count * ratio)))


def budget_schedule(steps: int) -> dict[str, int]:
    """Warm up for about 5% of the budget and save about five times, plus the final save.

    Round only to whole optimizer steps. Even a one-step run must have an update after
    warmup; zero is the unfilled form's budget and uses the same initial field values.
    """
    return {
        "warmup_steps": min(max(0, steps - 1), math.ceil(steps / 20)),
        "save_every": max(1, math.ceil(steps / 5)),
    }


def normalize_preset(values: dict, count: int) -> dict:
    config = {**RECOMMENDED, **values}
    if config["gradient_checkpointing"] is None or any(
        not config[key] for key in TUNED if key != "gradient_checkpointing"
    ):
        raise BusinessError(t("training.generate_recommended_settings_first"))
    for key in (
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "lora_r",
        "lora_alpha",
        "save_every",
    ):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise BusinessError(t("training.must_be_a_positive_integer", p0=fmt(key, "", -1)))
    if config["learning_rate"] <= 0 or config["learning_rate"] > 0.1:
        raise BusinessError(t("training.learning_rate_must_be_between_0_and"))
    steps = config["max_steps"]
    for key, value in budget_schedule(steps).items():
        if key not in values:
            config[key] = value
    config["warmup_steps"] = min(config["warmup_steps"], max(0, steps - 1))
    config["save_every"] = min(config["save_every"], steps)
    config["valid_every"] = min(config["valid_every"], steps) if count > 1 else 0
    config["valid_ratio"] = config["valid_ratio"] if count > 1 and config["valid_every"] else 0.0
    if config["batch_size"] > max(
        1, training_count(count, config["valid_ratio"] > 0, config["valid_ratio"])
    ):
        raise BusinessError(t("training.batch_exceeds_the_number_of_available_training"))
    return config


_digests: dict[tuple[str, int, int], str] = {}


def cached_digest(path: Path) -> str:
    """SHA-256 of a file, reused while its size and modification time are unchanged."""
    status = path.stat()
    key = (str(path.absolute()), status.st_size, status.st_mtime_ns)
    if key not in _digests:
        _digests[key] = digest(path)
    return _digests[key]


def initialization_current(project: dict) -> bool:
    """Whether the published initialization was built from the recordings as they are now."""
    init = project.get("initialization")
    if not init or init.get("status") != "completed":
        return False
    if init.get("fingerprint"):
        return init["fingerprint"] == materials_fingerprint(project)
    # Initializations made before fingerprints existed are bound to the revision counter.
    return init.get("revision") == project["revision"]


def validate_initialization(root: Path) -> dict:
    project = load_project(root)
    init = project.get("initialization")
    if not initialization_current(project):
        raise BusinessError(t("training.recordings_need_initialization_go_to_data_initialization"))
    record = read_json(within(root, "initialized/initialization.json"))
    if not record or any(
        record.get(key) != init.get(key)
        for key in ("id", "revision", "codec_revision", "rules_version", "count", "fingerprint")
    ):
        raise BusinessError(t("training.initialization_record_is_missing_or_damaged_initialize"))
    if (
        record.get("status") != "completed"
        or record.get("codec_revision") != CODEC_REVISION
        or record.get("rules_version") != INITIALIZATION_RULES_VERSION
    ):
        raise BusinessError(t("training.initialization_does_not_match_current_recordings_or"))
    checksums = record.get("checksums")
    if not isinstance(checksums, dict) or "manifest.jsonl" not in checksums:
        raise BusinessError(
            t("training.initialization_integrity_record_is_missing_initialize_again")
        )
    for filename, checksum in checksums.items():
        path = within(root / "initialized", filename)
        if not path.is_file() or cached_digest(path) != checksum:
            raise BusinessError(
                t(
                    "training.initialization_output_is_damaged_initialize_again",
                    p1=fmt(filename, "", -1),
                )
            )
    try:
        rows = [
            json.loads(line)
            for line in (root / "initialized/manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if not rows or len(rows) != record["count"]:
            raise BusinessError(t("training.entry_count_mismatch"))
        for row in rows:
            latent = within(root / "initialized", row["latent_path"])
            if (
                latent.relative_to(root / "initialized").as_posix() not in checksums
                or not row["text"].strip()
                or int(row["num_frames"]) <= 0
            ):
                raise BusinessError(t("training.entry_is_incomplete"))
    except (KeyError, TypeError, ValueError) as exc:
        raise BusinessError(
            t("training.initialization_manifest_is_damaged_initialize_again")
        ) from exc
    return record


def validate_run_binding(root: Path, run: dict) -> dict:
    record = validate_initialization(root)
    if run.get("initialization_id") != record["id"] or run.get("revision") != record["revision"]:
        raise BusinessError(t("training.the_data_version_bound_to_this_run"))
    # The run's adapter has the shape of the base it records; its model settings must too.
    base = normalize_record(run.get("base"))
    model = architecture_of_config((run.get("config") or {}).get("model") or {})
    if base is None or not base.get("architecture") or model != base["architecture"]:
        raise BusinessError(t("training.base_does_not_match_run"))
    return record


def run_base(run: dict, bases: list[InstalledBase]) -> InstalledBase:
    """The installed base a run trains on: the one it was created with, else one of the same
    architecture. Fails naming the base when none is installed."""
    match = match_record(run.get("base"), bases)
    if match.base is None:
        raise BusinessError(
            t(
                "training.base_not_installed",
                name=fmt(describe(run.get("base")), "", -1),
                hint=missing_message(match, run.get("base")),
            )
        )
    return match.base


def run_names(root: Path) -> set[str]:
    names = set()
    for path in (root / "runs").glob("*/run.json"):
        try:
            names.add(str(read_json(path).get("name", "")).strip())
        except (OSError, ValueError, AttributeError):
            continue
    return names


def unique_run_name(root: Path, name: str) -> str:
    """name, or "name 2", "name 3"... when another run of the project already has it: runs
    of one project are told apart by name in the list and in Generate's model menu."""
    name, taken = name.strip(), run_names(root)
    if name not in taken:
        return name
    number = 2
    while f"{name} {number}" in taken:
        number += 1
    return f"{name} {number}"


def training_config(checkpoint) -> dict:
    """Use the same published model definition for resource probes and real training."""
    from dataclasses import fields

    import yaml
    from safetensors import safe_open

    from irodori_tts.config import ModelConfig

    config = yaml.safe_load(
        upstream_file("configs/train_v4_small_lora.yaml").read_text(encoding="utf-8")
    )
    # Only the metadata is read; framework="pt" would import torch into the window's process.
    with safe_open(str(checkpoint), framework="np") as stream:
        published = json.loads(stream.metadata()["config_json"])
    model_fields = {field.name for field in fields(ModelConfig)}
    config["model"] = {key: value for key, value in published.items() if key in model_fields}
    return config


def create_run(
    root: Path, name: str, values: dict, base: InstalledBase, resource_root: Path | None = None
) -> Path:
    """A new run on base. The train section is the published LoRA configuration for every
    base (its settings do not depend on the model size); the model section is the base's."""
    record = validate_initialization(root)
    if not name.strip():
        raise BusinessError(t("training.enter_a_voice_name"))
    if resource_root is not None and not any(b.id == base.id for b in load_registry(resource_root)):
        raise BusinessError(t("training.base_not_installed", name=fmt(base.name, "", -1), hint=""))
    config = training_config(base.checkpoint)
    config["train"].update(normalize_preset(values, record["count"]))
    run_id = identity()
    run_name = unique_run_name(root, name)
    (root / "runs").mkdir(exist_ok=True)
    run_root = free_folder(root / "runs", folder_name(run_name))
    run_root.mkdir()
    write_json(run_root / "config.json", config)
    write_json(
        run_root / "run.json",
        {
            "id": run_id,
            "name": run_name,
            "created": now(),
            "initialization_id": record["id"],
            "revision": record["revision"],
            "status": "preparing",
            "config": config,
            "step": 0,
            "checkpoint": None,
            "registered": False,
            "preview_status": "pending",
            "base": base_record(base),
        },
    )
    return run_root


def complete_checkpoint(path: Path, *, require_trainer=False) -> bool:
    if path.name.endswith(".partial"):
        return False
    marker = read_json(path / "complete.json")
    if not marker:
        return False
    return (
        all(
            (path / filename).is_file() and (path / filename).stat().st_size == size
            for filename, size in marker["files"].items()
            if require_trainer or filename != "trainer_state.pt"
        )
        and (path / "adapter_config.json").is_file()
        and any(
            (path / filename).is_file()
            for filename in ("adapter_model.safetensors", "adapter_model.bin")
        )
        and (not require_trainer or (path / "trainer_state.pt").is_file())
    )


def snapshot_preview_checkpoint(root: Path, checkpoint: Path, destination: Path) -> Path:
    """Keep the selected adapter before the next training save can prune it."""
    checkpoint = within(root, checkpoint)
    destination = within(root, destination)
    if not complete_checkpoint(checkpoint):
        raise BusinessError(t("training_page.select_a_complete_checkpoint"))
    marker = read_json(checkpoint / "complete.json")
    stage = within(root, destination.with_name(".adapter.partial"))
    stage.mkdir(parents=True)
    try:
        files = {}
        for name, size in marker["files"].items():
            if name == "trainer_state.pt":
                continue
            source = within(checkpoint, name)
            target = within(stage, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if target.stat().st_size != size:
                raise BusinessError(t("training_page.select_a_complete_checkpoint"))
            files[name] = size
        write_json(stage / "complete.json", {"step": marker.get("step"), "files": files})
        if not (
            (stage / "adapter_config.json").is_file()
            and any(
                (stage / name).is_file()
                for name in ("adapter_model.safetensors", "adapter_model.bin")
            )
        ):
            raise BusinessError(t("training_page.select_a_complete_checkpoint"))
        stage.replace(destination)
        return destination
    finally:
        if stage.exists():
            within(root, stage.relative_to(root))
            shutil.rmtree(stage)


def discard_preview_snapshot(root: Path, snapshot: Path) -> None:
    """Remove a sample's pinned adapter when the sample itself was never produced."""
    output = Path(snapshot).parent
    if not output.exists() or (output / "generation.json").exists():
        return
    within(root, output.relative_to(root))
    shutil.rmtree(output, ignore_errors=True)


def compact_checkpoints(run_root: Path, keep: Path | None) -> None:
    """Drop the optimizer state from every checkpoint except the recovery one.

    The adapters stay complete for samples and registration (complete_checkpoint without
    require_trainer); only the newest checkpoint can resume training."""
    for path in (run_root / "checkpoints").glob("*"):
        if (
            path.name.startswith(".")
            or not path.is_dir()
            or (keep is not None and path == keep)
            or not complete_checkpoint(path)
        ):
            continue
        state = path / "trainer_state.pt"
        if state.is_file():
            state.unlink()


def delete_run(root: Path, run_root: Path) -> None:
    """Delete a training record with its checkpoints and samples. The record itself goes
    last: a run whose files cannot all be deleted stays listed, to be deleted again."""
    with project_lock(root):
        run_root = within(root, run_root.relative_to(root))
        record = run_root / "run.json"
        if run_root.parent != within(root, "runs") or not record.is_file():
            raise BusinessError(t("training.select_a_training_record"))
        for path in run_root.rglob("*"):
            within(root, path.relative_to(root))
        for child in run_root.iterdir():
            if child.is_dir():
                remove_tree(root, child)
            elif child != record:
                child.unlink()
        record.unlink()
        run_root.rmdir()


def latest_checkpoint(run_root: Path) -> tuple[Path | None, int]:
    candidates = []
    for path in (run_root / "checkpoints").glob("*"):
        if path.is_dir() and complete_checkpoint(path, require_trainer=True):
            step = read_json(path / "complete.json").get("step")
            if step is None:
                import torch

                try:
                    state = torch.load(
                        path / "trainer_state.pt", map_location="cpu", weights_only=True
                    )
                    step = int(state["step"])
                except (OSError, ValueError, RuntimeError, EOFError):
                    continue
            candidates.append((step, path))
    if not candidates:
        return None, 0
    step, path = max(
        candidates,
        key=lambda value: (
            value[0],
            value[1].name == "checkpoint_final",
            not value[1].name.startswith("."),
        ),
    )
    return path, step


def extend_run(
    root: Path, run_root: Path, total: int, save_every: int, bases: list | None = None
) -> None:
    """Let a finished run train on from where it ended to total steps, saving every
    save_every steps; Resume then continues it (the trainer checks only the LoRA settings).

    The trainer writes the new end to checkpoint_final, which would replace the weights the
    run ended with while its samples still named that folder. The ended checkpoint therefore
    takes its step's name first (its identical twin, which kept no trainer state, gives way),
    and the samples made from it follow it."""
    run = read_json(run_root / "run.json")
    validate_run_binding(root, run)
    if bases is not None:
        run_base(run, bases)
    if total <= run["step"]:
        raise BusinessError(t("training.more_steps_than_done", step=run["step"]))
    checkpoint = within(root, run["checkpoint"]) if run.get("checkpoint") else None
    if checkpoint is None or not complete_checkpoint(checkpoint, require_trainer=True):
        raise BusinessError(t("training.training_recovery_state_is_incomplete_choose_another"))
    if checkpoint.name == "checkpoint_final":
        named = checkpoint.with_name(f"checkpoint_{run['step']:07d}")
        if named.exists():
            shutil.rmtree(named)
        checkpoint.rename(named)
        for record_path in (run_root / "samples").glob("*/generation.json"):
            record = read_json(record_path)
            request = record.get("request") or {}
            adapter = request.get("lora_adapter")
            if adapter and Path(adapter).resolve() == checkpoint.resolve():
                request["lora_adapter"] = str(named)
                write_json(record_path, record)
        run["checkpoint"] = named.relative_to(root).as_posix()
    config = read_json(run_root / "config.json")
    config["train"].update(max_steps=int(total), save_every=int(save_every))
    write_json(run_root / "config.json", config)
    run.update(config=config, status="stopped")
    write_json(run_root / "run.json", run)


def resume_eligibility(root: Path, run: dict, bases: list | None = None) -> None:
    validate_run_binding(root, run)
    if bases is not None:
        run_base(run, bases)
    _validate_resume_checkpoint(root, run)


def _validate_resume_checkpoint(root: Path, run: dict) -> None:
    if run["step"] >= run["config"]["train"]["max_steps"]:
        raise BusinessError(t("training.this_training_run_is_complete_create_a"))
    if not run.get("checkpoint"):
        raise BusinessError(t("training.no_complete_recovery_checkpoint_start_a_new"))
    checkpoint = within(root, run["checkpoint"])
    if not complete_checkpoint(checkpoint, require_trainer=True):
        raise BusinessError(t("training.training_recovery_state_is_incomplete_choose_another"))


def write_trainer_config(run_root: Path) -> Path:
    """The run's configuration as YAML for the trainer, which reads JSON's 1e-05 as a string."""
    import yaml

    path = run_root / "trainer_config.yaml"
    path.write_text(
        yaml.safe_dump(read_json(run_root / "config.json"), allow_unicode=True), encoding="utf-8"
    )
    return path


def run_command(
    root: Path, run_root: Path, bases: list[InstalledBase], *, resume=False
) -> list[str]:
    run = read_json(run_root / "run.json")
    validate_run_binding(root, run)
    base = run_base(run, bases)
    command = [
        str(upstream_file("train.py")),
        "--config",
        str(write_trainer_config(run_root)),
        "--manifest",
        str(root / "initialized/manifest.jsonl"),
        "--output-dir",
        str(run_root / "checkpoints"),
        "--init-checkpoint",
        str(base.checkpoint),
        "--device",
        "cuda",
    ]
    if resume:
        _validate_resume_checkpoint(root, run)
        command += ["--resume", str(within(root, run["checkpoint"]))]
    return command
