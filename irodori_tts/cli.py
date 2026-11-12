"""Normal module entry point for the command line tools of the pinned fork."""

from __future__ import annotations

import argparse
import runpy
import sys

from .upstream import upstream_file


def main():
    commands = {
        "infer": upstream_file("infer.py"),
        "train": upstream_file("train.py"),
        "prepare-manifest": upstream_file("prepare_manifest.py"),
        "gradio": upstream_file("gradio_app.py"),
        "gradio-voice-design": upstream_file("gradio_app_voicedesign.py"),
        "convert-checkpoint": upstream_file("convert_checkpoint_to_safetensors.py"),
        "quantize-checkpoint": upstream_file("quantize_checkpoint.py"),
    }
    parser = argparse.ArgumentParser(description="Irodori-TTS-Desktop command line")
    parser.add_argument("command", choices=commands)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    entry = commands[args.command]
    sys.argv = [str(entry), *args.arguments]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
