"""The desktop application on top of the pinned Irodori-TTS fork in upstream/.

No upstream files are modified at import time.
"""

import os
import sys
from pathlib import Path


def _carried_ffmpeg(root: Path) -> Path | None:
    """The FFmpeg of a checkout (root/runtime) or a package (beside its app folder)."""
    for folder in (root / "runtime/ffmpeg", root.parent / "runtime/ffmpeg"):
        if folder.is_dir():
            return folder
    return None


# torchaudio reads and writes audio through FFmpeg's libraries, which it finds on PATH, and
# add-ons run its program. Ours goes first: users install nothing, and another version on the
# computer cannot get in the way. Worker processes inherit it.
_ffmpeg = _carried_ffmpeg(Path(__file__).resolve().parents[1])
if _ffmpeg and not os.environ.get("PATH", "").startswith(str(_ffmpeg) + os.pathsep):
    os.environ["PATH"] = str(_ffmpeg) + os.pathsep + os.environ.get("PATH", "")

_upstream_package = Path(__file__).resolve().parents[1] / "upstream/Irodori-TTS/irodori_tts"
if (_upstream_package / "model.py").is_file():
    __path__.append(str(_upstream_package))
    sys.path.append(str(_upstream_package.parent))
elif not (Path(__file__).resolve().parent / "model.py").is_file():
    raise ImportError("Pinned upstream is missing; run git submodule update --init --recursive")

# Public exports load on first use: they pull in torch, which the desktop window does not
# need to start (about a second of its start-up).
_EXPORTS = {
    "LORA_TARGET_PRESETS": "lora",
    "ModelConfig": "config",
    "PretrainedTextTokenizer": "tokenizer",
    "TextToLatentRFDiT": "model",
    "TrainConfig": "config",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
