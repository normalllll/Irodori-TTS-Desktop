"""Verify an independent directory package without launching its UI or GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from irodori_tts.desktop.storage import digest, now, read_json, write_json
from irodori_tts.upstream import UPSTREAM_COMMIT


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=repo / "dist/Irodori-TTS-Desktop")
    parser.add_argument(
        "--output", type=Path, default=repo / "outputs/desktop-checks/package-verification.json"
    )
    args = parser.parse_args()
    root = args.package.resolve()
    manifest = read_json(root / "app/package-manifest.json")
    failures = []
    if manifest.get("upstream", {}).get("commit") != UPSTREAM_COMMIT:
        failures.append({"path": "app/package-manifest.json", "reason": "wrong upstream commit"})
    total = 0
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            failures.append({"path": relative, "reason": "outside package"})
        elif not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
        elif path.stat().st_size != expected["size"] or digest(path) != expected["sha256"]:
            failures.append({"path": relative, "reason": "size or SHA256 mismatch"})
        total += expected["size"]
    report = {
        "created": now(),
        "package": str(root),
        "files_checked": len(manifest["files"]),
        "total_bytes": total,
        "upstream": manifest.get("upstream"),
        "desktop": manifest.get("desktop"),
        "failures": failures,
        "passed": not failures,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
