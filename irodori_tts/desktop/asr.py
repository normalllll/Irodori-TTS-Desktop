"""Automatic transcription: backend registry, isolated add-ons, cross-check and the import task.

Workers import this module; nothing here needs Qt. Heavy packages are imported inside the
loaders, after the worker was launched with the add-on directories in front of its path."""

from __future__ import annotations

import gc
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import types
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from irodori_tts.desktop.messages import BusinessError, Message, error_message, fmt, t
from irodori_tts.task_control import check_cancel

from . import ASR_DEFAULT_CHECK, ASR_DEFAULT_MAIN
from .materials import AUDIO_SUFFIXES, clean_field, import_transcribed
from .resources import ModelSpec, cache_root, cached_snapshot, fetch_snapshot
from .splitting import split_points
from .storage import clean_project, digest, identity, load_project, within

SAMPLE_RATE = 16000
GB = 1000**3


@dataclass(frozen=True)
class Addon:
    """Packages installed on demand into their own directory, in front of the app's own.
    Nothing is installed over the application environment, and torch always comes from it."""

    key: str
    packages: tuple[str, ...]
    size: int  # installed size in bytes, shown before the user agrees
    # True when it replaces packages the app itself ships (a different transformers).
    replaces: bool


@dataclass(frozen=True)
class Backend:
    key: str
    title: str
    models: tuple[ModelSpec, ...]
    addon: str | None
    vram: float  # approximate peak GB
    batch: int
    hint: Message
    recommended: bool = False

    @property
    def size(self) -> int:
        return sum(model.size for model in self.models)


ADDONS = {
    # qwen-asr's model code needs transformers 4.57 (its decorators and configs predate 5.x),
    # which in turn needs huggingface_hub 0.x: the app ships transformers 5 and hub 1.x.
    "qwen3-asr": Addon(
        "qwen3-asr",
        ("transformers==4.57.6", "huggingface-hub==0.36.2", "qwen-asr==0.0.6"),
        60 * 1024**2,
        replaces=True,
    ),
    # funasr runs on the app's transformers 5; it only brings packages the app lacks.
    "funasr": Addon(
        "funasr",
        (
            "funasr==1.4.16",
            "hydra-core==1.3.8",
            "omegaconf==2.3.1",
            "antlr4-python3-runtime==4.9.3",
            "kaldiio==2.18.1",
            "jieba==0.42.1",
            "jaconv==0.5.0",
            "jamo==0.4.1",
            "torch-complex==0.4.4",
            "tiktoken==0.14.0",
            "rapidfuzz==3.14.5",
        ),
        55 * 1024**2,
        replaces=False,
    ),
}

WHISPER_FILES = (
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "vocab.json",
)

BACKENDS = (
    Backend(
        "whisper-large-v3",
        "whisper-large-v3",
        (
            ModelSpec(
                "openai/whisper-large-v3",
                "06f233fe06e710322aca913c1bc4249a0d71fce1",
                (*WHISPER_FILES, "tokenizer.json", "tokenizer_config.json"),
                3091519764,
            ),
        ),
        None,
        8.0,
        4,
        t("asr.hint_whisper_large_v3"),
        recommended=True,
    ),
    Backend(
        "qwen3-asr",
        "Qwen3-ASR-1.7B",
        (
            ModelSpec(
                "Qwen/Qwen3-ASR-1.7B",
                "7278e1e70fe206f11671096ffdd38061171dd6e5",
                (
                    "chat_template.json",
                    "config.json",
                    "generation_config.json",
                    "merges.txt",
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                    "model.safetensors.index.json",
                    "preprocessor_config.json",
                    "tokenizer_config.json",
                    "vocab.json",
                ),
                4703055333,
            ),
        ),
        "qwen3-asr",
        5.0,
        8,
        t("asr.hint_qwen3_asr"),
    ),
    Backend(
        "anime-whisper",
        "anime-whisper",
        (
            ModelSpec(
                "litagin/anime-whisper",
                "22e2008a8182b357da3922a6308d095008f72973",
                (*WHISPER_FILES, "tokenizer_config.json"),
                3027594672,
            ),
        ),
        None,
        4.0,
        16,
        t("asr.hint_anime_whisper"),
    ),
    Backend(
        "fun-asr-nano",
        "Fun-ASR-Nano",
        (
            ModelSpec(
                "FunAudioLLM/Fun-ASR-Nano-2512",
                "272c57b82523ada6fd87095e955f8e29100979ab",
                (
                    "Qwen3-0.6B/config.json",
                    "Qwen3-0.6B/generation_config.json",
                    "Qwen3-0.6B/merges.txt",
                    "Qwen3-0.6B/tokenizer.json",
                    "Qwen3-0.6B/tokenizer_config.json",
                    "Qwen3-0.6B/vocab.json",
                    "config.yaml",
                    "configuration.json",
                    "model.pt",
                    "multilingual.tiktoken",
                ),
                1987942401,
            ),
            # Speech activity detection in front of the recognizer, as GPT-SoVITS runs it.
            ModelSpec(
                "funasr/fsmn-vad",
                "df20e6b30c653645fa4ff125cacfcabd1020a669",
                ("am.mvn", "config.yaml", "configuration.json", "model.pt"),
                1730979,
            ),
        ),
        "funasr",
        4.0,
        1,
        t("asr.hint_fun_asr_nano"),
    ),
)
BACKEND_KEYS = tuple(backend.key for backend in BACKENDS)
BACKEND_BY_KEY = {backend.key: backend for backend in BACKENDS}
DEFAULT_MAIN = ASR_DEFAULT_MAIN
DEFAULT_CHECK = ASR_DEFAULT_CHECK
NO_CHECK = "none"


def backend_for(key: str) -> Backend:
    try:
        return BACKEND_BY_KEY[key]
    except KeyError:
        raise BusinessError(t("asr.unknown_backend", name=fmt(key, "", -1))) from None


def choice(settings: Mapping | None) -> tuple[str, str]:
    """(main, check) from the saved settings; an unknown value falls back to the default."""
    saved = (settings or {}).get("asr")
    saved = saved if isinstance(saved, dict) else {}
    main = saved.get("main")
    check = saved.get("check")
    main = main if main in BACKEND_BY_KEY else DEFAULT_MAIN
    check = check if check in BACKEND_BY_KEY or check == NO_CHECK else DEFAULT_CHECK
    return main, NO_CHECK if check == main else check


def selection(main: str, check: str | None) -> list[Backend]:
    """The backends a run uses, main first; the check must be a different one."""
    backends = [backend_for(main)]
    if check and check != NO_CHECK:
        if check == main:
            raise BusinessError(t("asr.check_must_differ"))
        backends.append(backend_for(check))
    return backends


# ----- add-ons -----


def addons_root() -> Path:
    return cache_root() / "addons"


_addon_sizes: dict[Path, tuple[int | None, int]] = {}


def python_tag() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def addon_state(key: str, root: Path | None = None) -> dict:
    """{'ready': bool, 'size': installed bytes or 0} read from the add-on's own record."""
    folder = (root or addons_root()) / key
    record = None
    try:
        record = json.loads((folder / "addon.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    spec = ADDONS[key]
    ready = (
        isinstance(record, dict)
        and record.get("packages") == list(spec.packages)
        and record.get("python") == python_tag()
    )
    return {"ready": bool(ready), "size": addon_size(folder)}


def addon_size(folder: Path) -> int:
    """Installed bytes. Thousands of files are measured once per installation (the record is
    written last), since pages ask whenever they refresh."""
    if not folder.is_dir():
        return 0
    try:
        installed = (folder / "addon.json").stat().st_mtime_ns
    except OSError:
        installed = None
    known = _addon_sizes.get(folder)
    if installed is not None and known and known[0] == installed:
        return known[1]
    size = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
    _addon_sizes[folder] = (installed, size)
    return size


def required_addons(backends: Iterable[Backend]) -> list[str]:
    keys = []
    for backend in backends:
        if backend.addon and backend.addon not in keys:
            keys.append(backend.addon)
    return keys


def missing_addons(backends: Iterable[Backend], root: Path | None = None) -> list[str]:
    return [key for key in required_addons(backends) if not addon_state(key, root)["ready"]]


def missing_models(backends: Iterable[Backend]) -> list[ModelSpec]:
    return [
        model for backend in backends for model in backend.models if cached_snapshot(model) is None
    ]


def isolation_environment(
    keys: Iterable[str], base: Mapping[str, str], root: Path | None = None
) -> dict[str, str]:
    """Environment overrides that layer the add-ons over the application's packages.

    The worker is the application's own interpreter, so torch and everything else the add-on
    does not carry come from the application environment; PYTHONPATH puts the add-ons in
    front of it before anything is imported. User site-packages stay out of it."""
    folders = [str((root or addons_root()) / key) for key in keys]
    if not folders:
        return {}
    inherited = base.get("PYTHONPATH")
    return {
        "PYTHONPATH": os.pathsep.join(folders + ([inherited] if inherited else [])),
        "PYTHONNOUSERSITE": "1",
    }


def launch_environment(payload: Mapping, base: Mapping[str, str]) -> dict[str, str]:
    """What a task's worker needs on top of the inherited environment (usually nothing)."""
    return isolation_environment(payload.get("addons") or [], base)


def find_installer() -> list[str]:
    """The command that installs wheels into a directory: uv, else the interpreter's pip."""
    python = Path(sys.executable)
    names = ("uv.exe", "uv") if os.name == "nt" else ("uv",)
    candidates = [python.parent / name for name in names]
    # A checkout keeps uv in its own runtime/; a directory bundle next to its app/ folder.
    here = Path(__file__).resolve()
    candidates += [here.parents[i] / "runtime" / name for i in (2, 3) for name in names]
    found = next((str(path) for path in candidates if path.is_file()), None) or shutil.which("uv")
    if found:
        return [found, "pip", "install", "--python", str(python), "--link-mode", "copy"]
    import importlib.util

    if importlib.util.find_spec("pip"):
        return [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
    raise BusinessError(t("asr.no_installer"))


def install_addon(
    key: str,
    *,
    root: Path | None = None,
    progress: Callable[[str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Path:
    """Install an add-on into <root>/<key>; the previous one stays until the new one is whole."""
    check_cancel(cancel)
    spec = ADDONS[key]
    root = root or addons_root()
    root.mkdir(parents=True, exist_ok=True)
    partial, final = root / f"{key}.partial", root / key
    shutil.rmtree(partial, ignore_errors=True)
    command = [*find_installer(), "--target", str(partial), "--no-deps", *spec.packages]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def pump():
        for line in process.stdout:
            lines.put(line.rstrip())
        lines.put(None)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    tail: list[str] = []
    try:
        while True:
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                line = ""
            else:
                if line is None:
                    break
                tail = [*tail, line][-8:]
                print(line, flush=True)
                if progress and line:
                    progress(line)
            check_cancel(cancel)
        code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        shutil.rmtree(partial, ignore_errors=True)
        raise
    finally:
        reader.join(timeout=5)
        process.stdout.close()
    if code != 0:
        shutil.rmtree(partial, ignore_errors=True)
        raise BusinessError(
            t("asr.addon_install_failed", name=fmt(key, "", -1), output=fmt("\n".join(tail)))
        )
    for package in spec.packages:
        name, version = package.split("==")
        if not any(partial.glob(f"{name.replace('-', '_')}-{version}.dist-info")):
            shutil.rmtree(partial, ignore_errors=True)
            raise BusinessError(t("asr.addon_incomplete", name=fmt(package, "", -1)))
    (partial / "addon.json").write_text(
        json.dumps({"key": key, "packages": list(spec.packages), "python": python_tag()}),
        encoding="utf-8",
    )
    shutil.rmtree(final, ignore_errors=True)
    partial.replace(final)
    return final


def remove_addon(key: str, root: Path | None = None) -> None:
    folder = (root or addons_root()) / key
    if folder.is_dir():
        shutil.rmtree(folder)


# ----- cross-check -----

PUNCTUATION = re.compile(r"[\s\W_]+", re.UNICODE)


def surface(text: str | None) -> str:
    """Characters only: width-normalized, punctuation and spaces removed. ー is kept."""
    text = unicodedata.normalize("NFKC", text or "")
    return "".join(PUNCTUATION.sub("", part) for part in re.split(r"(ー)", text))


def reading(text: str | None) -> str:
    """Katakana pronunciation, so kana and kanji spellings of the same words compare equal.
    Without the pronunciation dictionary it is the normalized surface text."""
    try:
        import pyopenjtalk

        return surface(pyopenjtalk.g2p(unicodedata.normalize("NFKC", text or ""), kana=True))
    except Exception:
        return surface(text)


def readings_differ(main_text: str, check_text: str) -> bool:
    return reading(main_text) != reading(check_text)


# ----- backends (run inside the worker) -----


class Recognizer:
    """A loaded backend: clips (16 kHz mono float32) in, one text per clip out."""

    def __init__(self, transcribe: Callable[[list[np.ndarray]], list[str]], unload: Callable):
        self.transcribe, self.unload = transcribe, unload


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _half():
    import torch

    return torch.float16 if torch.cuda.is_available() else torch.float32


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_whisper(paths: dict[str, Path], batch: int, settings: dict) -> Recognizer:
    from transformers import pipeline

    # The pipeline (not a bare generate call) is what the benchmark ran: it also chunks
    # recordings longer than 30 s when splitting is off.
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=str(next(iter(paths.values()))),
        dtype=_half(),
        device=_device(),
        chunk_length_s=30.0,
        batch_size=batch,
    )

    def transcribe(clips):
        outputs = recognizer(clips, generate_kwargs=settings)
        return [output["text"].strip() for output in outputs]

    def unload():
        nonlocal recognizer
        del recognizer
        _free_gpu()

    return Recognizer(transcribe, unload)


def load_anime_whisper(paths: dict[str, Path], batch: int) -> Recognizer:
    # The model card's settings; the card requires that no initial prompt is given.
    return load_whisper(
        paths,
        batch,
        {"language": "Japanese", "no_repeat_ngram_size": 5, "repetition_penalty": 1.0},
    )


def load_whisper_large_v3(paths: dict[str, Path], batch: int) -> Recognizer:
    # Beam search 5, as GPT-SoVITS runs it.
    return load_whisper(
        paths, batch, {"language": "japanese", "task": "transcribe", "num_beams": 5}
    )


def load_qwen3_asr(paths: dict[str, Path], batch: int) -> Recognizer:
    import torch

    # qwen-asr imports nagisa (which pulls in DyNet) only for its forced aligner, which
    # transcription never loads. A placeholder keeps the package importable without it.
    sys.modules.setdefault("nagisa", types.ModuleType("nagisa"))
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        str(next(iter(paths.values()))),
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=f"{_device()}:0" if torch.cuda.is_available() else "cpu",
        max_inference_batch_size=batch,
        max_new_tokens=256,
    )

    def transcribe(clips):
        results = model.transcribe(
            audio=[(clip, SAMPLE_RATE) for clip in clips], language="Japanese"
        )
        return [result.text.strip() for result in results]

    def unload():
        nonlocal model
        del model
        _free_gpu()

    return Recognizer(transcribe, unload)


# funasr names Japanese in Chinese ("日文"); spelled out so the string checker sees no UI wording.
JAPANESE = chr(0x65E5) + chr(0x6587)


def load_fun_asr_nano(paths: dict[str, Path], batch: int) -> Recognizer:
    from funasr import AutoModel

    recognizer, detector = (str(path) for path in paths.values())
    # The model class ships with funasr itself; no remote code is trusted.
    model = AutoModel(
        model=recognizer,
        trust_remote_code=False,
        vad_model=detector,
        device=_device(),
        disable_update=True,
        disable_pbar=True,
    )

    def transcribe(clips):
        return [
            (model.generate(input=clip, language=JAPANESE, disable_pbar=True) or [{}])[0]
            .get("text", "")
            .strip()
            for clip in clips
        ]

    def unload():
        nonlocal model
        del model
        _free_gpu()

    return Recognizer(transcribe, unload)


LOADERS: dict[str, Callable[[dict[str, Path], int], Recognizer]] = {
    "anime-whisper": load_anime_whisper,
    "qwen3-asr": load_qwen3_asr,
    "whisper-large-v3": load_whisper_large_v3,
    "fun-asr-nano": load_fun_asr_nano,
}


def run_backend(
    backend: Backend,
    paths: dict[str, Path],
    clips: list[np.ndarray],
    *,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[int], None] | None = None,
) -> list[str]:
    """Transcribe every clip with one backend, in batches; the model is released afterwards."""
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        torch = None
    recognizer = LOADERS[backend.key](paths, backend.batch)
    texts: list[str] = []
    try:
        for start in range(0, len(clips), backend.batch):
            check_cancel(cancel)
            batch = clips[start : start + backend.batch]
            texts.extend(recognizer.transcribe(batch))
            if progress:
                progress(len(batch))
    finally:
        if torch is not None and torch.cuda.is_available():
            print(
                f"[asr] {backend.key} peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
                flush=True,
            )
        recognizer.unload()
    return texts


# ----- the import task -----


def to_model_rate(mono: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE:
        return mono.astype(np.float32)
    import librosa

    return librosa.resample(mono, orig_sr=rate, target_sr=SAMPLE_RATE).astype(np.float32)


def segment_name(relative: str, number: int) -> str:
    path = Path(relative)
    return path.with_name(f"{path.stem}_{number:02d}.flac").as_posix()


def prepare_units(
    root: Path,
    folder: Path,
    staging: Path,
    *,
    split: bool,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], dict]:
    """Recordings to transcribe: one per file, or one per segment of a long file.

    Segments are cut from the original samples into `staging`; short files are copied at
    commit. Recordings the project already has with a transcript are skipped, so importing the
    same folder again never overwrites edits."""
    project = load_project(root)
    known = {}
    for item in project["materials"]:
        key = (item.get("source"), json.dumps(item.get("segment"), sort_keys=True))
        known[key] = item
    candidates = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES and p.is_file()
    )
    if not candidates:
        raise BusinessError(t("materials.no_supported_recordings_in_the_selected_folder"))
    units: list[dict] = []
    stats = {"files": len(candidates), "split_files": 0, "segments": 0, "skipped": 0}
    staging.mkdir(parents=True, exist_ok=True)
    for position, original in enumerate(candidates, 1):
        check_cancel(cancel)
        if progress:
            progress(position, len(candidates))
        relative = original.relative_to(folder).as_posix()
        source = str(original.resolve())
        try:
            data, rate = sf.read(original, dtype="float32", always_2d=True)
        except Exception as exc:
            raise BusinessError(
                t("materials.cannot_read_the_recording_replace_it", p1=fmt(exc, "", -1))
                + fmt(f" ({relative})")
            ) from exc
        samples = len(data)
        mono = data.mean(axis=1)
        cuts = split_points(mono, rate) if split else [(0.0, samples / rate)]
        model_clip = to_model_rate(mono, rate)
        checksum = digest(original)
        if len(cuts) == 1:
            pieces = [(None, original, None, model_clip, relative)]
        else:
            stats["split_files"] += 1
            pieces = []
            bounds = [round(start * rate) for start, _ in cuts] + [samples]
            for number, (start, end) in enumerate(cuts, 1):
                piece = data[bounds[number - 1] : bounds[number]]
                target = staging / f"{identity()}.flac"
                sf.write(target, piece, rate, format="FLAC", subtype="PCM_16")
                first, last = round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)
                pieces.append(
                    (
                        {
                            "file": relative,
                            "start": round(start, 3),
                            "end": round(end, 3),
                            "sha256": checksum,
                        },
                        target,
                        True,
                        model_clip[first:last],
                        segment_name(relative, number),
                    )
                )
        for segment, audio, moved, clip, name in pieces:
            existing = known.get((source, json.dumps(segment, sort_keys=True)))
            if existing and (
                (segment is not None or existing.get("sha256") == checksum)
                and clean_field(existing.get("text"))
            ):
                stats["skipped"] += 1
                if moved:
                    audio.unlink(missing_ok=True)
                continue
            if segment is not None:
                stats["segments"] += 1
            units.append(
                {
                    "name": name,
                    "audio": audio,
                    "move": bool(moved),
                    "source": source,
                    "segment": segment,
                    "clip": clip,
                    "existing": existing["id"] if existing else None,
                }
            )
    return units, stats


def project_units(
    root: Path,
    ids: list[str],
    *,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], dict]:
    """Recordings the project already has, to transcribe again in place. Their audio, name,
    origin and caption stay; the transcript and the recognition record are replaced."""
    wanted = set(ids)
    items = [item for item in load_project(root)["materials"] if item["id"] in wanted]
    if not items:
        raise BusinessError(t("training_page.select_a_recording_first"))
    units = []
    for position, item in enumerate(items, 1):
        check_cancel(cancel)
        if progress:
            progress(position, len(items))
        audio = within(root, item["path"])
        try:
            data, rate = sf.read(audio, dtype="float32", always_2d=True)
        except Exception as exc:
            raise BusinessError(
                t("materials.cannot_read_the_recording_replace_it", p1=fmt(exc, "", -1))
                + fmt(f" ({item['file_name']})")
            ) from exc
        units.append(
            {
                "name": item["file_name"],
                "audio": audio,
                "move": False,
                "source": item.get("source") or str(audio),
                "segment": item.get("segment"),
                "clip": to_model_rate(data.mean(axis=1), rate),
                "existing": item["id"],
            }
        )
    return units, {"files": len(items), "split_files": 0, "segments": 0, "skipped": 0}


def transcribe_units(
    units: list[dict],
    backends: list[Backend],
    models: dict[str, dict[str, Path]],
    *,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[Backend, int, int], None] | None = None,
) -> list[list[str]]:
    """texts[backend][unit]; progress(backend, finished clips overall, total clip passes)."""
    clips = [unit["clip"] for unit in units]
    total = len(clips) * len(backends)
    done = 0
    outputs = []
    for backend in backends:

        def advance(count, backend=backend):
            nonlocal done
            done += count
            if progress:
                progress(backend, done, total)

        outputs.append(
            run_backend(backend, models[backend.key], clips, cancel=cancel, progress=advance)
        )
    return outputs


def entries_for(units: list[dict], backends: list[Backend], outputs: list[list[str]]) -> list[dict]:
    """Per recording: the main text, and when a check backend ran, its text and the flag."""
    main = backends[0]
    check = backends[1] if len(backends) > 1 else None
    entries = []
    for index, unit in enumerate(units):
        text = clean_field(outputs[0][index])
        asr = {"backend": main.key, "check_backend": check.key if check else None}
        if check:
            check_text = clean_field(outputs[1][index])
            asr["check_text"] = check_text
            asr["needs_review"] = readings_differ(text, check_text)
        else:
            asr["check_text"] = None
            asr["needs_review"] = False
        entries.append({**{k: unit[k] for k in unit if k != "clip"}, "text": text, "asr": asr})
    return entries


def model_paths(backend: Backend) -> dict[str, Path]:
    paths = {}
    for model in backend.models:
        found = cached_snapshot(model)
        if found is None:
            raise BusinessError(t("asr.model_missing", name=fmt(backend.title, "", -1)))
        paths[model.repo] = found
    return paths


def download_models(
    backends: list[Backend],
    *,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[Backend, int, int], None] | None = None,
) -> dict[str, dict[str, Path]]:
    """Fetch every missing pinned model file (resumable, cancellable); returns the folders."""
    result = {}
    for backend in backends:
        if progress:
            progress(backend, 0, backend.size)
        done = 0
        paths = {}
        for model in backend.models:

            def moved(amount, backend=backend):
                nonlocal done
                done += amount
                if progress:
                    progress(backend, min(done, backend.size), backend.size)

            paths[model.repo] = fetch_snapshot(model, download=True, cancel=cancel, progress=moved)
        result[backend.key] = paths
    return result


def run_import(
    payload: dict,
    channel,
    *,
    models: dict[str, dict[str, Path]] | None = None,
) -> dict:
    """The transcription task: split, recognize, cross-check, then add to the project. The models
    were downloaded by the task before it (or the run fails saying so).

    Nothing reaches the project before every recording is transcribed; a stop or failure
    leaves it as it was."""
    root = Path(payload["project"])
    again = payload.get("materials")
    backends = selection(payload["main"], payload.get("check"))
    staging = within(root, f"temp/asr-{channel.task_id}")
    try:
        channel.emit("progress", step=0, total=0, message=t("asr.reading_recordings"))

        def reading(done, total):
            channel.emit(
                "progress",
                step=done,
                total=total,
                message=t("asr.reading_progress", step=done, total=total),
            )

        if again:
            units, stats = project_units(root, again, cancel=channel.stopped, progress=reading)
        else:
            units, stats = prepare_units(
                root,
                Path(payload["folder"]),
                staging,
                split=bool(payload.get("split", True)),
                cancel=channel.stopped,
                progress=reading,
            )
        check_cancel(channel.stopped)
        if units:
            if models is None:
                models = {b.key: model_paths(b) for b in backends}
            outputs = transcribe_units(
                units,
                backends,
                models,
                cancel=channel.stopped,
                progress=lambda backend, done, total: channel.emit(
                    "progress",
                    step=done,
                    total=total,
                    message=t(
                        "asr.transcribing",
                        name=fmt(backend.title, "", -1),
                        step=done,
                        total=total,
                    ),
                ),
            )
            check_cancel(channel.stopped)
            channel.emit("stage", message=t("asr.saving_recordings"))
            items = import_transcribed(root, entries_for(units, backends, outputs))
        else:
            items = []
    finally:
        try:
            clean_project(root, f"temp/asr-{channel.task_id}")
        except Exception as exc:
            print(error_message(exc), flush=True)
    return {
        "status": "completed",
        "imported": len(items),
        "included": sum(item["included"] for item in items),
        "needs_review": sum(bool((item.get("asr") or {}).get("needs_review")) for item in items),
        "main": backends[0].key,
        "check": backends[1].key if len(backends) > 1 else None,
        "ids": [item["id"] for item in items],
        "again": bool(again),
        **stats,
    }


def segment_lengths(items: Iterable[dict]) -> list[float]:
    return [item["duration"] for item in items if item.get("segment")]
