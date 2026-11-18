"""Installed base models for window tests."""

from __future__ import annotations

import json
from pathlib import Path


def install_base(root, base_id="v4.1-small", *, name=None, architecture=None, variants=()):
    """Register a base in the resource folder root, as a header-only checkpoint (the pages
    read a base's architecture from its header, never its weights)."""
    import struct

    from irodori_tts.desktop.base_models import (
        SMALL_ARCHITECTURE,
        InstalledBase,
        catalog_base,
        save_base,
    )

    entry = catalog_base(base_id)
    architecture = dict(architecture or (entry.architecture if entry else SMALL_ARCHITECTURE))
    folder = Path(root) / "bases" / base_id
    paths = {}
    for key in ("model", *variants):
        path = folder / f"{key}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        header = json.dumps({"__metadata__": {"config_json": json.dumps(architecture)}}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header)
        paths[key] = path
    base = InstalledBase(
        id=base_id,
        name=name or (entry.name if entry else base_id),
        source="catalog" if entry else "imported",
        repo=entry.repo if entry else None,
        revision="rev1" if entry else None,
        checkpoint=paths.pop("model"),
        architecture=architecture,
        variants=paths,
    )
    save_base(root, base)
    return base
