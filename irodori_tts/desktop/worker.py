"""Background compute entry point. No Qt imports, windows, or desktop input."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from irodori_tts.desktop.messages import BusinessError, error_message, fmt, t
from irodori_tts.task_control import TaskCancelled, TaskChannel, check_cancel
from irodori_tts.upstream import runtime_sources

from . import INITIALIZATION_RULES_VERSION, asr, model_files
from .long_text import split_text
from .materials import clean_field, materials_fingerprint, validate_materials
from .resources import (
    CODEC_REVISION,
    adopt_cached_resources,
    configure_environment,
    hide_child_consoles,
    resolve_resources,
)
from .storage import (
    clean_project,
    digest,
    identity,
    load_project,
    now,
    project_lock,
    read_json,
    reconcile_sources,
    save_project,
    within,
    write_json,
)
from .training import (
    compact_checkpoints,
    complete_checkpoint,
    latest_checkpoint,
    run_command,
    write_trainer_config,
)

REPO = Path(__file__).resolve().parents[2]
PREVIEW_TEXT = "こんにちは。今日はとてもいい天気ですね。ゆっくりお話ししましょう。"


def initialize(payload: dict, channel: TaskChannel, resources: dict) -> dict:
    from .manifest import prepare_local_manifest

    channel.emit("sources", sources=runtime_sources())

    root = Path(payload["project"])
    with project_lock(root):
        project = load_project(root)
        channel.emit("stage", phase="check", message=t("worker.checking_recordings"))
        reconcile_sources(root, project)
        try:
            items = validate_materials(root, project)
        except BaseException as exc:
            remember_attempt(root, project, exc, "check")
            raise
        # The previous initialization stays in place until the new one is published.
        required = sum(within(root, item["path"]).stat().st_size for item in items) + 128 * 1024**2
        if shutil.disk_usage(root).free < required:
            raise BusinessError(
                t(
                    "worker.insufficient_space_keep_at_least_mb_free",
                    p1=fmt(required / 1024**2, ".0f", -1),
                )
            )
        previous = project["initialization"]
        if previous is not None and not payload.get("rebuild"):
            raise BusinessError(
                t("worker.initialization_data_already_exists_confirm_reinitialization")
            )
        # The recording being encoded, for a failure to name.
        encoding = {"index": 0}
        check_cancel(channel.stopped)
        init = {
            "id": identity(),
            "revision": project["revision"],
            "fingerprint": materials_fingerprint(project),
            "status": "running",
            "codec_revision": CODEC_REVISION,
            "rules_version": INITIALIZATION_RULES_VERSION,
            "count": len(items),
            "duration": sum(item["duration"] for item in items),
            "created": now(),
        }
        published = root / "initialized"
        backup = within(root, "temp/initialization.previous")
        try:
            clean_project(root, "temp/initialization")
            clean_project(root, "temp/initialization.previous")
            stage = within(root, "temp/initialization")
            stage.mkdir(parents=True)
            # Managed copies are encoded in place: they cannot change while the lock is held.
            local_items = [
                {
                    **item,
                    "text": clean_field(item.get("text")),
                    "caption": clean_field(item.get("caption")),
                    "audio": str(within(root, item["path"])),
                    "speaker_id": project["id"],
                }
                for item in items
            ]
            channel.emit(
                "progress",
                step=0,
                total=len(items),
                phase="encode",
                item=items[0]["file_name"],
                message=t("worker.preparing_training_data"),
            )

            def encoded(current, total):
                encoding["index"] = current
                channel.emit(
                    "progress",
                    step=current,
                    total=total,
                    phase="encode",
                    item=items[min(current, total - 1)]["file_name"],
                )

            try:
                prepare_local_manifest(
                    local_items,
                    output_manifest=stage / "manifest.jsonl",
                    latent_dir=stage / "latents",
                    codec_path=resources["codec"],
                    cancel=channel.stopped,
                    progress=encoded,
                )
            except TaskCancelled:
                raise
            except Exception as exc:
                # The same failure, telling which recording it stopped at.
                failed = items[min(encoding["index"], len(items) - 1)]
                reason = error_message(exc)
                exc.phase, exc.item = "encode", failed["id"]
                exc.message = t(
                    "worker.encoding_failed",
                    file=fmt(failed["file_name"], "", -1),
                    reason=fmt(reason, "", -1),
                )
                raise
            channel.emit("stage", phase="write", message=t("worker.writing_manifest"))
            init["checksums"] = {
                str(p.relative_to(stage)).replace("\\", "/"): digest(p)
                for p in stage.rglob("*")
                if p.is_file()
            }
            init["status"] = "completed"
            write_json(stage / "initialization.json", init)
            check_cancel(channel.stopped)
            # Publish: only now is the previous data replaced and are older runs bound to it
            # no longer resumable.
            if published.exists():
                published.replace(backup)
            stage.replace(published)
            project["initialization"] = init
            project.pop("last_attempt", None)
            save_project(root, project)
            for path in (root / "runs").glob("*/run.json"):
                run = read_json(path)
                if run.get("initialization_id") != init["id"]:
                    run["resume_blocked"] = t("worker.data_was_reinitialized_start_a_new_training")
                    write_json(path, run)
            clean_project(root, "temp/initialization.previous")
            return init
        except BaseException as exc:
            if project["initialization"] is not init:
                # Not published: restore the previous data exactly as it was.
                try:
                    if backup.exists():
                        if published.exists():
                            clean_project(root, "initialized")
                        backup.replace(published)
                    clean_project(root, "temp/initialization")
                except Exception:
                    traceback.print_exc()
                if previous is None or previous.get("status") != "completed":
                    # Without usable data the outcome stays visible; a rebuild keeps the
                    # previous complete initialization usable instead.
                    init["status"] = "stopped" if isinstance(exc, TaskCancelled) else "failed"
                    init["error"] = error_message(exc)
                    init.pop("checksums", None)
                    project["initialization"] = init
                remember_attempt(root, project, exc, getattr(exc, "phase", "encode"))
            raise


def remember_attempt(root: Path, project: dict, exc: BaseException, phase: str) -> None:
    """Keep how the last preprocessing ended, so the page still says it after a restart."""
    project["last_attempt"] = {
        "status": "stopped" if isinstance(exc, TaskCancelled) else "failed",
        "error": error_message(exc),
        "phase": getattr(exc, "phase", phase),
        "item": getattr(exc, "item", None),
        "items": getattr(exc, "items", []),
        "created": now(),
    }
    save_project(root, project)


def train_run(payload: dict, channel: TaskChannel, resources: dict) -> dict:
    root = Path(payload["project"])
    run_root = Path(payload["run"])
    with project_lock(root):
        run = read_json(run_root / "run.json")
        try:
            project = load_project(root)
            reconcile_sources(root, project)
            # This gate runs under the project lock for new and resumed runs.
            command = run_command(root, run_root, resources, resume=payload.get("resume", False))
        except Exception as exc:
            run.update(status="failed", error=error_message(exc))
            write_json(run_root / "run.json", run)
            raise

        import torch
        import train

        channel.emit("sources", sources=runtime_sources())

        if run["config"]["train"]["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            if payload.get("resume"):
                raise BusinessError(t("worker.this_device_does_not_support_this_run"))
            run["config"]["train"]["precision"] = "fp32"
            run["precision_reason"] = t("worker.bf16_is_unsupported_on_this_device_using")
            write_json(run_root / "config.json", run["config"])
            write_trainer_config(run_root)
        run["status"] = "running"
        run.pop("error", None)
        write_json(run_root / "run.json", run)
        try:
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            sys.argv = command + (["--stop-file", channel.stop_file] if channel.stop_file else [])
            train.main(event_callback=lambda kind, values: channel.emit(kind, **values))
            checkpoint, step = latest_checkpoint(run_root)
            if checkpoint is None:
                raise BusinessError(t("worker.training_did_not_publish_a_complete_checkpoint"))
            compact_checkpoints(run_root, checkpoint)
            run.update(
                status="stopped" if step < run["config"]["train"]["max_steps"] else "completed",
                step=step,
                checkpoint=checkpoint.relative_to(root).as_posix(),
                registered=True,
            )
            run["device_metrics"] = {
                "gpu": torch.cuda.get_device_name(),
                "cuda": torch.version.cuda,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "elapsed_seconds": time.monotonic() - started,
            }
            write_json(run_root / "run.json", run)
            return run
        except BaseException as exc:
            checkpoint, step = latest_checkpoint(run_root)
            run.update(status="failed", error=error_message(exc), step=step)
            if checkpoint:
                run.update(checkpoint=checkpoint.relative_to(root).as_posix(), registered=True)
                try:
                    compact_checkpoints(run_root, checkpoint)
                except OSError:
                    traceback.print_exc()
            write_json(run_root / "run.json", run)
            raise


def install_addons(payload: dict, channel: TaskChannel) -> dict:
    for key in payload["keys"]:
        channel.emit("stage", message=t("asr.installing_addon", name=fmt(key, "", -1)))
        asr.install_addon(
            key,
            cancel=channel.stopped,
            progress=lambda line, key=key: channel.emit(
                "stage",
                message=t("asr.installing_addon_line", name=fmt(key, "", -1), line=fmt(line)),
            ),
        )
    return {"status": "completed", "installed": list(payload["keys"])}


def generation_model_download(key):
    """A generation model (low video memory or fast), in the shape of a backend for the shared
    download loop."""
    from types import SimpleNamespace

    item = model_files.download_for(key)
    return SimpleNamespace(key=item.key, title=item.title, size=item.size, models=item.models)


def download_asr_models(payload: dict, channel: TaskChannel) -> dict:
    backends = [
        generation_model_download(key)
        if key in model_files.GENERATION_MODELS
        else asr.backend_for(key)
        for key in payload["backends"]
    ]
    due = progress_pacer()

    def report(backend, done, total):
        # The start and end of each model always show: they change which line is active.
        if due(not done or done >= total):
            channel.emit(
                "progress",
                step=round(done / 1000**2),
                total=round(total / 1000**2),
                item=backend.key,
                message=t(
                    "asr.downloading",
                    name=fmt(backend.title, "", -1),
                    done=fmt(done / 1000**3, ".2f"),
                    total=fmt(total / 1000**3, ".2f"),
                ),
            )

    asr.download_models(backends, cancel=channel.stopped, progress=report)
    return {"status": "completed", "backends": list(payload["backends"])}


SEGMENT_PAUSE = 0.3
# Segments sampled together in one pass: each keeps its own seed and length, so the audio
# matches one by one up to rounding (waveform correlation about 0.9 on an RTX 5080), and it is
# 1.1 to 1.3 times faster there. The low video memory setting samples them one at a time.
SEGMENT_BATCH = 4


def synthesize_segments(
    runtime, request: dict, segments: list[str], channel: TaskChannel, *, batch=SEGMENT_BATCH
):
    """One clip from a long text: each segment is a pass of its own, with the request's
    settings, joined with a short pause. The first segment takes the request's seed (a random
    one when it has none), the next ones follow it, so the seed of the result reproduces the
    whole. Up to batch segments share a pass; a pass that runs out of video memory is
    repeated one segment at a time."""
    import secrets
    from types import SimpleNamespace

    import torch

    from irodori_tts.inference_runtime import SamplingRequest

    count = len(segments)
    seed = request.get("seed")
    first_seed = int(secrets.randbits(31)) if seed is None else int(seed)
    pieces = [
        SamplingRequest(
            **{
                **request,
                "text": segment,
                "seconds": None,
                "num_candidates": 1,
                "seed": first_seed + index,
            }
        )
        for index, segment in enumerate(segments)
    ]
    audios = []
    timings = []
    sample_rate = None
    steps = 0
    passes = math.ceil(count / batch)
    done = 0

    def run(group, position):
        def report_step(current, total):
            nonlocal steps
            check_cancel(channel.stopped)
            steps = steps or total
            channel.emit("progress", step=position * steps + current, total=passes * steps)

        return runtime.synthesize_batch(
            group, step_callback=report_step, log_fn=lambda message: print(message, flush=True)
        )

    for start in range(0, count, batch):
        group = pieces[start : start + batch]
        try:
            results = run(group, done)
        except torch.cuda.OutOfMemoryError:
            if len(group) == 1:
                raise
            torch.cuda.empty_cache()
            # One pass per segment from here on; the progress total grows to match.
            passes += len(group) - 1
            results = [run([piece], done + i)[0] for i, piece in enumerate(group)]
            done += len(group) - 1
        done += 1
        check_cancel(channel.stopped)
        for result in results:
            sample_rate = result.sample_rate
            audios.append(result.audio)
            timings.extend(result.stage_timings)
    pause = audios[0].new_zeros((audios[0].shape[0], round(SEGMENT_PAUSE * sample_rate)))
    parts = [audios[0]]
    for audio in audios[1:]:
        parts.extend([pause, audio])
    joined = torch.cat(parts, dim=-1)
    return SimpleNamespace(
        audio=joined,
        audios=[joined],
        sample_rate=sample_rate,
        stage_timings=timings,
        used_seed=first_seed,
    )


def synthesize(payload: dict, channel: TaskChannel, resources: dict, runtime):
    from irodori_tts.inference_runtime import (
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
        save_wav,
    )

    channel.emit("sources", sources=runtime_sources())

    if payload["kind"] == "preview":
        root = Path(payload.get("project") or Path(payload["run"]).parents[1])
        checkpoint = Path(payload.get("checkpoint_snapshot") or payload["request"]["lora_adapter"])
        within(root, checkpoint.relative_to(root))
        if not complete_checkpoint(checkpoint):
            raise BusinessError(t("training_page.select_a_complete_checkpoint"))

    model = payload.get("model")
    checkpoint = (
        model_files.generation_checkpoint(model)
        if model in model_files.GENERATION_MODELS
        else resources["model"]
    )
    if runtime is not None and runtime.key.checkpoint != checkpoint:
        runtime.unload()
        runtime = None
    if runtime is None:
        channel.emit("stage", message=t("worker.loading_voice_model"))
        import torch

        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
        runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=checkpoint,
                model_device="cuda",
                model_precision=precision,
                codec_repo=resources["codec"],
                codec_device="cuda",
                codec_precision="fp32",
            )
        )
        # Upstream warns and carries on without the mark; a package must not.
        if not runtime.watermarker.ready:
            runtime.unload()
            raise BusinessError(t("resources.resources_are_not_ready_choose_prepare_resources"))
        channel.emit("loaded", process_id=os.getpid())
    request = dict(payload["request"])
    if payload.get("checkpoint_snapshot"):
        request["lora_adapter"] = payload["checkpoint_snapshot"]
    generating = payload["kind"] == "generate"
    # A training preview is always a single clip.
    request["num_candidates"] = int(request.get("num_candidates") or 1) if generating else 1

    segments = split_text(request["text"]) if generating else []
    if len(segments) > 1:
        batch = 1 if model == model_files.LOW_VRAM else SEGMENT_BATCH
        result = synthesize_segments(runtime, request, segments, channel, batch=batch)
    else:

        def report_step(current, total):
            check_cancel(channel.stopped)
            channel.emit("progress", step=current, total=total)

        result = runtime.synthesize(
            SamplingRequest(**request),
            step_callback=report_step,
            log_fn=lambda message: print(message, flush=True),
        )
    check_cancel(channel.stopped)
    audios = list(result.audios) or [result.audio]
    output_root = Path(payload["output"])
    output_root.mkdir(parents=True, exist_ok=True)
    names = ["audio.wav" if i == 0 else f"audio_{i + 1}.wav" for i in range(len(audios))]
    # Every candidate is written before any takes its final name, so a stop leaves no result.
    temporaries = []
    for name, audio in zip(names, audios, strict=True):
        temporary = output_root / f"{Path(name).stem}.partial.wav"
        save_wav(temporary, audio, result.sample_rate)
        temporaries.append(temporary)
        check_cancel(channel.stopped)
    for temporary, name in zip(temporaries, names, strict=True):
        temporary.replace(output_root / name)
    record = {
        "id": output_root.name,
        "created": now(),
        "request": dict(payload["request"]),
        "voice": payload.get("voice", t("generation_page.default_voice")),
        "voice_id": payload.get("voice_id"),
        "audio": str(output_root / "audio.wav"),
        "sample_rate": result.sample_rate,
        "duration": result.audio.shape[-1] / result.sample_rate,
        "seed": result.used_seed,
        "timings": result.stage_timings,
        "base_revision": payload.get("base_revision"),
        "model": model,
    }
    if len(segments) > 1:
        record["segments"] = len(segments)
    if generating:
        record["candidates"] = [str(output_root / name) for name in names]
        record["durations"] = [audio.shape[-1] / result.sample_rate for audio in audios]
    if payload.get("checkpoint_snapshot"):
        record["checkpoint_snapshot"] = payload["checkpoint_snapshot"]
    # A generation record names its audio relative to its own folder, so the data folder can
    # move; the task result keeps the absolute path. Training samples stay absolute until
    # their reader (the training page) resolves relative paths too.
    stored = {**record, "audio": "audio.wav", "candidates": names} if generating else record
    write_json(output_root / "generation.json", stored)
    if payload.get("run"):
        run_root = Path(payload["run"])
        run = read_json(run_root / "run.json")
        run.update(
            preview_status="completed", preview=str(output_root.relative_to(run_root) / "audio.wav")
        )
        run.pop("preview_error", None)
        write_json(run_root / "run.json", run)
    return runtime, record


PROGRESS_INTERVAL = 0.25

# Tasks that only move files: no speech model and no GPU.
FILE_TASKS = {"transcribe", "addon", "models", "import_models", "export_models", "recommend"}


def progress_pacer():
    """due(finished): whether a byte counter should report now, a few times a second."""
    last = 0.0

    def due(finished):
        nonlocal last
        current = time.monotonic()
        if not finished and current - last < PROGRESS_INTERVAL:
            return False
        last = current
        return True

    return due


def copy_progress(channel, message):
    """progress(done, total) in bytes as MB steps; message(done GB, total GB) describes it."""
    due = progress_pacer()

    def report(done, total):
        if not due(done >= total):
            return
        channel.emit(
            "progress",
            step=round(done / 1000**2),
            total=round(total / 1000**2),
            message=message(fmt(done / 1000**3, ".2f"), fmt(total / 1000**3, ".2f")),
        )

    return report


def speech_progress(channel):
    """received(n) for the speech model's download: progress over what is still missing."""
    item = model_files.download_for(model_files.SPEECH)
    total = max(1, item.missing_size())
    done = 0
    due = progress_pacer()

    def received(amount):
        nonlocal done
        done = min(total, done + amount)
        if not due(done >= total):
            return
        channel.emit(
            "progress",
            step=round(done / 1000**2),
            total=round(total / 1000**2),
            item=model_files.SPEECH,
            message=t(
                "asr.downloading",
                name=fmt(item.title, "", -1),
                done=fmt(done / 1000**3, ".2f"),
                total=fmt(total / 1000**3, ".2f"),
            ),
        )

    return received


def import_models(payload: dict, channel: TaskChannel) -> dict:
    result = model_files.import_model_files(
        Path(payload["source"]),
        cancel=channel.stopped,
        progress=copy_progress(
            channel, lambda done, total: t("models.importing", done=done, total=total)
        ),
    )
    if model_files.SPEECH in result["complete"]:
        # The speech resources are read from their manifest; record what the cache now has.
        adopt_cached_resources(Path(payload["resource_root"]))
    return {"status": "completed", **result}


def export_models(payload: dict, channel: TaskChannel) -> dict:
    result = model_files.export_model_files(
        list(payload["keys"]),
        Path(payload["destination"]),
        cancel=channel.stopped,
        progress=copy_progress(
            channel, lambda done, total: t("models.exporting", done=done, total=total)
        ),
    )
    return {"status": "completed", **result}


def execute(payload: dict, runtime=None):
    channel = TaskChannel()
    channel.emit("running", process_id=os.getpid())
    try:
        speech = payload["kind"] not in FILE_TASKS
        # Verifying reports a missing file instead of fetching it unasked.
        download = payload["kind"] == "resources" and not payload.get("verify")
        # Transcription uses its own models; installing an add-on or downloading those models
        # needs none of the speech model's.
        resources = (
            resolve_resources(
                Path(payload["resource_root"]),
                download=download,
                verify=payload.get("verify", False),
                cancel=channel.stopped,
                repair=payload.get("repair", False),
                progress=lambda message: channel.emit("stage", message=message, item="speech"),
                received=speech_progress(channel) if download else None,
            )
            if speech
            else {}
        )
        check_cancel(channel.stopped)
        if payload["kind"] not in FILE_TASKS | {"resources"}:
            import torch

            if not torch.cuda.is_available():
                raise BusinessError(t("worker.no_nvidia_cuda_device_detected_check_the"))
        if payload["kind"] == "transcribe":
            result = asr.run_import(payload, channel)
        elif payload["kind"] == "addon":
            result = install_addons(payload, channel)
        elif payload["kind"] == "models":
            result = download_asr_models(payload, channel)
        elif payload["kind"] == "import_models":
            result = import_models(payload, channel)
        elif payload["kind"] == "export_models":
            result = export_models(payload, channel)
        elif payload["kind"] == "initialize":
            result = initialize(payload, channel, resources)
        elif payload["kind"] == "train":
            result = train_run(payload, channel, resources)
        elif payload["kind"] == "recommend":
            from .training_recommendation import recommend_training

            result = recommend_training(payload, channel)
        elif payload["kind"] == "resources":
            result = resources
        else:
            runtime, result = synthesize(payload, channel, resources, runtime)
        channel.emit("result", result=result)
    except TaskCancelled:
        if payload.get("run") and payload["kind"] == "preview":
            path = Path(payload["run"]) / "run.json"
            run = read_json(path)
            run.update(preview_status="stopped")
            write_json(path, run)
        channel.emit("cancelled", message=t("worker.task_stopped_at_a_safe_boundary"))
    except BaseException as exc:
        traceback.print_exc()
        message = error_message(exc)
        if "out of memory" in str(message).lower():
            message = t("worker.the_current_task_ran_out_of_gpu") + message
        if payload.get("run") and payload["kind"] == "preview":
            path = Path(payload["run"]) / "run.json"
            run = read_json(path)
            run.update(preview_status="failed", preview_error=message)
            write_json(path, run)
        # Where it failed, for the page to point at: the step, the recording, the recordings.
        where = {key: getattr(exc, key) for key in ("phase", "item", "items") if hasattr(exc, key)}
        channel.emit("error", message=message, **where)
    return runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=Path)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    configure_environment(REPO)
    hide_child_consoles()
    runtime = None
    payload = read_json(args.task / "request.json")
    runtime = execute(payload, runtime)
    if args.serve:
        for line in sys.stdin:
            message = json.loads(line)
            if message.get("command") == "shutdown":
                if runtime:
                    runtime.unload()
                return
            task = Path(message["task"])
            os.environ.update(
                IRODORI_TASK_ID=task.name,
                IRODORI_TASK_EVENTS=str(task / "events.jsonl"),
                IRODORI_TASK_STOP=str(task / "stop"),
            )
            runtime = execute(read_json(task / "request.json"), runtime)


if __name__ == "__main__":
    main()
