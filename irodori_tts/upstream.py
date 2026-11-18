"""Read-only upstream locations for source checkouts and assembled directory apps."""

import sys
from pathlib import Path

UPSTREAM_URL = "https://github.com/normalllll/Irodori-TTS.git"
UPSTREAM_COMMIT = "18b55648051e7494ee3501852e631aadc91c581c"
SOURCE_ROOT = Path(__file__).resolve().parents[1]
_checkout = SOURCE_ROOT / "upstream/Irodori-TTS"
UPSTREAM_ROOT = _checkout if (_checkout / "irodori_tts/model.py").is_file() else SOURCE_ROOT


def upstream_file(relative: str) -> Path:
    """Assembled apps already contain baseline files underneath their app root."""
    return UPSTREAM_ROOT / relative


def runtime_sources() -> dict:
    """Record actual loaded modules in worker diagnostics, without loading models."""
    names = (
        "irodori_tts",
        "irodori_tts.model",
        "irodori_tts.codec",
        "irodori_tts.config",
        "irodori_tts.inference_runtime",
        "irodori_tts.rf",
        "irodori_tts.meanflow",
        "irodori_tts.desktop.worker",
        "train",
        "prepare_manifest",
    )
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "source_root": str(SOURCE_ROOT),
        "upstream_root": str(UPSTREAM_ROOT),
        "loaded_modules": {
            name: str(sys.modules[name].__file__) for name in names if name in sys.modules
        },
    }
