"""Fast, conservative training plans from metadata, without loading a model."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass

from irodori_tts.task_control import check_cancel

from .base_models import catalog_base, default_catalog_base, normalize_record
from .materials import materials_fingerprint, needs_review
from .messages import BusinessError, t
from .storage import load_project
from .training import RECOMMENDED, budget_schedule, training_count


@dataclass(frozen=True)
class PlanningPolicy:
    # First-run engineering defaults, not quality or convergence thresholds. Keep the
    # effective batch independent of device speed: faster is not evidence of better audio.
    exposures: int = 100
    effective_batch: int = 8
    usable_memory_fraction: float = 0.75


POLICY = PlanningPolicy()


def base_memory(base) -> tuple[float, float]:
    """(fixed GiB, GiB per recording) of LoRA training on a base: the catalog's deliberately
    rounded-up envelope, not an exact peak prediction. A base outside the catalog scales the
    default base's envelope by its size (model width times depth)."""
    record = normalize_record(base) or {}
    entry = catalog_base(record.get("id"))
    if entry is not None and entry.architecture == record.get("architecture"):
        return entry.training_memory
    reference = default_catalog_base()
    architecture = record.get("architecture")
    if not architecture:
        return reference.training_memory
    scale = (architecture["model_dim"] * architecture["num_layers"]) / (
        reference.architecture["model_dim"] * reference.architecture["num_layers"]
    )
    fixed, per_recording = reference.training_memory
    return fixed * scale, per_recording * scale


@dataclass(frozen=True)
class MaterialProfile:
    count: int
    total_seconds: float
    median_seconds: float
    p90_seconds: float
    maximum_seconds: float
    distinct_texts: int


def material_profile(rows: list[dict]) -> MaterialProfile:
    if not rows:
        raise BusinessError(t("training_page.nothing_included"))
    durations = sorted(float(row.get("duration", 0)) for row in rows)
    if durations[0] <= 0 or not all(math.isfinite(value) for value in durations):
        raise BusinessError(t("training.entry_is_incomplete"))
    return MaterialProfile(
        count=len(rows),
        total_seconds=sum(durations),
        median_seconds=statistics.median(durations),
        p90_seconds=durations[math.ceil(0.9 * len(durations)) - 1],
        maximum_seconds=durations[-1],
        distinct_texts=len({" ".join(row["text"].split()) for row in rows}),
    )


def plan_training(project: dict, settings: dict, gpu: dict | None, base=None) -> dict:
    """A plan for a run on base (a base record; None is the default base)."""
    options = {**RECOMMENDED, **settings}
    fixed_gib, activation_gib = base_memory(base)
    rows = [item for item in project["materials"] if item["included"]]
    summary = material_profile(rows)
    count = training_count(summary.count, bool(options["valid_every"]), options["valid_ratio"])
    effective = min(POLICY.effective_batch, 2 ** (count.bit_length() - 1))
    # Statistics describe all included recordings; validation's exact random split is
    # left to the trainer. Including the held-out tail makes the memory estimate safer.
    free_gib = max(0, (gpu["total_mib"] - gpu["used_mib"]) / 1024) if gpu else None
    budget = free_gib * POLICY.usable_memory_fraction if free_gib is not None else None
    bf16 = bool(gpu and (gpu.get("compute_cap") or 0) >= 8)
    precision_scale = 1 if bf16 else 2
    # Target and reference sequences contribute activation memory; the quadratic term
    # conservatively accounts for attention. The maximum protects rare long clips.
    target_seconds = min(summary.maximum_seconds, options["max_latent_steps"] / 25)
    baseline_seconds = RECOMMENDED["max_latent_steps"] / 25 + RECOMMENDED["ref_max_seconds"]
    length_scale = (target_seconds + options["ref_max_seconds"]) / baseline_seconds
    activation = activation_gib * max(length_scale, length_scale**2)
    fixed = fixed_gib * precision_scale
    activation *= precision_scale
    standard = (
        options["lora_enabled"]
        and options["lora_r"] <= RECOMMENDED["lora_r"]
        and options["lora_target_modules"] == RECOMMENDED["lora_target_modules"]
        and options["lora_modules_to_save"] == RECOMMENDED["lora_modules_to_save"]
        and options["optimizer"] == RECOMMENDED["optimizer"]
    )
    batch, checkpointing = 1, True
    if budget is not None and standard:
        # Do not squeeze the last batch into memory. If the ordinary envelope does not
        # fit, use batch one and checkpointing rather than guessing its exact savings.
        for candidate in (8, 4, 2, 1):
            if candidate <= effective and fixed + candidate * activation <= budget:
                batch, checkpointing = candidate, False
                break
    steps = math.ceil(POLICY.exposures * count / effective)
    values = {
        "batch_size": batch,
        "gradient_accumulation_steps": effective // batch,
        "gradient_checkpointing": checkpointing,
        "max_steps": steps,
        **budget_schedule(steps),
    }
    return {
        "values": values,
        "profile": asdict(summary),
        "training_count": count,
        "review_count": sum(needs_review(item) for item in rows),
        "fingerprint": materials_fingerprint(project),
        "settings": settings,
        "gpu": gpu["name"] if gpu else t("training.recommendation_unknown_device"),
        "memory_budget_bytes": int(budget * 1024**3) if budget is not None else None,
        "policy": asdict(POLICY),
        "base_memory_gib": [fixed_gib, activation_gib],
        # Batch one with checkpointing needs about the fixed part; with less free memory than
        # that, training will likely run out of memory, which the page says up front.
        "memory_short": free_gib is not None and free_gib < fixed,
    }


def recommend_training(payload: dict, channel) -> dict:
    from pathlib import Path

    from .exports import device_report

    check_cancel(channel.stopped)
    project = load_project(Path(payload["project"]))
    # A bounded read-only driver query. No encoding, model files, torch import or GPU work.
    gpu = device_report(timeout=1)["gpu"]
    check_cancel(channel.stopped)
    return plan_training(project, payload["settings"], gpu, payload.get("base"))
