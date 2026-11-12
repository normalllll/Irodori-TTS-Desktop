"""Encode a project's recordings into the manifest format the trainer reads."""

from __future__ import annotations

import json
import os
from pathlib import Path

import soundfile as sf
import torch

from irodori_tts.codec import DACVAECodec
from irodori_tts.desktop.messages import BusinessError, fmt, t
from irodori_tts.task_control import check_cancel
from irodori_tts.text_normalization import normalize_text


def prepare_local_manifest(
    items,
    *,
    output_manifest: Path,
    latent_dir: Path,
    codec_path: str,
    device: str = "cuda",
    cancel=None,
    progress=None,
):
    """Encode managed local audio with the same codec and manifest format as the CLI."""
    codec = DACVAECodec.load(repo_id=codec_path, device=device, dtype=torch.float32)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with output_manifest.open("w", encoding="utf-8") as stream:
        for index, item in enumerate(items):
            check_cancel(cancel)
            audio, rate = sf.read(item["audio"], dtype="float32", always_2d=True)
            waveform = torch.from_numpy(audio.T.copy())
            with torch.inference_mode():
                latent = codec.encode_waveform(waveform, sample_rate=rate)[0].cpu()
            if latent.ndim != 2 or not torch.isfinite(latent).all():
                raise BusinessError(
                    t("prepare_manifest.invalid_encoding_result", p1=fmt(item["audio"], "", -1))
                )
            filename = latent_dir / f"{item['id']}.pt"
            torch.save(latent, filename)
            row = {
                "text": normalize_text(item["text"]),
                "caption": item.get("caption", ""),
                "speaker_id": item["speaker_id"],
                "latent_path": os.path.relpath(filename, output_manifest.parent),
                "num_frames": int(latent.shape[0]),
            }
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if progress:
                progress(index + 1, len(items))
        check_cancel(cancel)
    return rows
