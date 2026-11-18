"""Write irodori_tts/desktop/model_pins.json: the size and SHA-256 of every file of every pinned
model the application downloads (codec, watermark, transcription), at its pinned revision.
Base speech models are not pinned: they follow their repository's latest commit.

Imported model files (from another user's copy) are accepted only when they match these values.
The files are hashed from the local download cache; large files are also checked against the
SHA-256 that Hugging Face records for them, so a damaged local copy cannot become the pin.

    .venv/Scripts/python.exe -m scripts.pin_models            # write the file
    .venv/Scripts/python.exe -m scripts.pin_models --check    # fail if it is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from irodori_tts.desktop.resources import configure_environment  # noqa: E402

configure_environment(REPO)

from irodori_tts.desktop import asr  # noqa: E402
from irodori_tts.desktop.model_files import (  # noqa: E402
    PINS_PATH,
    pinned_models,
)
from irodori_tts.desktop.resources import ModelSpec, cached_snapshot  # noqa: E402
from irodori_tts.desktop.storage import digest  # noqa: E402


def recorded_hashes(spec: ModelSpec) -> dict[str, str]:
    """SHA-256 values Hugging Face records for the repository's large (LFS) files."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(spec.repo, revision=spec.revision, files_metadata=True)
    return {s.rfilename: s.lfs.sha256 for s in info.siblings if getattr(s, "lfs", None)}


def pin(spec: ModelSpec, *, online: bool) -> dict:
    folder = cached_snapshot(spec)
    if folder is None:
        raise SystemExit(f"{spec.repo}@{spec.revision[:8]} is not in the local cache; download it")
    recorded = recorded_hashes(spec) if online else {}
    files = {}
    for name in spec.files:
        path = folder / name
        sha = digest(path)
        if name in recorded and recorded[name] != sha:
            raise SystemExit(f"{spec.repo}/{name}: local copy differs from Hugging Face; repair it")
        files[name] = {"size": path.stat().st_size, "sha256": sha}
    return {"revision": spec.revision, "files": files}


def build(*, online: bool) -> dict:
    specs = [
        *pinned_models(),
        *(model for backend in asr.BACKENDS for model in backend.models),
    ]
    repos = {}
    for spec in specs:
        if spec.repo not in repos:
            print(f"pinning {spec.repo}", flush=True)
            repos[spec.repo] = pin(spec, online=online)
    return {"format_version": 1, "repos": repos}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--offline", action="store_true", help="skip the Hugging Face cross-check")
    args = parser.parse_args()
    pins = build(online=not args.offline)
    text = json.dumps(pins, indent=1, sort_keys=True) + "\n"
    if args.check:
        if not PINS_PATH.is_file() or PINS_PATH.read_text(encoding="utf-8") != text:
            raise SystemExit("model_pins.json is out of date; run python -m scripts.pin_models")
        print("model_pins.json is current")
        return
    PINS_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {PINS_PATH.relative_to(REPO)}: {len(pins['repos'])} repositories")


if __name__ == "__main__":
    main()
