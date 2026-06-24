# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Package the built bundle into GitHub-release assets.

Run AFTER `scripts/build_exe.py` has produced `dist/waruka/`. From that
bundle this writes:

    dist/waruka-<version>.zip                  reassembled bundle zip
    dist/release_v<version>/
        waruka-<version>.zip.001               raw byte-split parts,
        waruka-<version>.zip.002               <= 1900 MiB each (under
        ...                                    GitHub's 2 GB asset cap).
        SHA256SUMS.txt                         hashes of the parts, the
                                               weights zip (if present),
                                               and the reassembled zip.

Rejoin the parts with:
    Windows (cmd):  copy /b waruka-<v>.zip.001 + waruka-<v>.zip.002 waruka-<v>.zip
    macOS / Linux:  cat waruka-<v>.zip.001 waruka-<v>.zip.002 > waruka-<v>.zip

The bundle zip mirrors `build_exe.py`'s layout (entries under `waruka/`)
but excludes the runtime `logs/` dir. The weights zip
(`waruka-weights-<version>.zip`) is NOT rebuilt here -- the model weights
are independent of the code, so an existing one is carried over and
re-hashed; a warning is printed if it's missing.

Usage:
    python scripts/make_release_zips.py
"""
from __future__ import annotations

import hashlib
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"
APP_NAME = "waruka"
PART_SIZE = 1900 * 1024 * 1024  # 1900 MiB -- matches the v1.0.0 split


def _version() -> str:
    init = REPO_ROOT / "waruka" / "__init__.py"
    for line in init.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("__version__"):
            return line.split("=", 1)[1].strip().strip("\"'")
    return "0.0.0"


def _zip_bundle(bundle: Path, zip_path: Path) -> None:
    """Zip `bundle/` -> zip_path with entries under 'waruka/', skipping
    the runtime `logs/` dir. Writes to a temp file first, then renames."""
    print(f"  zipping {bundle.name}/ -> {zip_path.name} (excluding logs/) ...")
    t0 = time.time()
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(bundle.rglob("*")):
            if f.is_dir():
                continue
            rel = f.relative_to(bundle.parent)          # waruka/...
            if len(rel.parts) > 1 and rel.parts[1] == "logs":
                continue
            zf.write(f, str(rel))
    tmp.replace(zip_path)
    print(f"  wrote {zip_path.name} "
          f"({zip_path.stat().st_size / 1e9:.2f} GB) in {time.time() - t0:.1f}s")


def _split(zip_path: Path, out_dir: Path) -> list[Path]:
    """Raw byte-split into <out_dir>/<zipname>.NNN parts of PART_SIZE."""
    for old in out_dir.glob(f"{zip_path.name}.[0-9][0-9][0-9]"):
        old.unlink()
    parts: list[Path] = []
    with open(zip_path, "rb") as f:
        idx = 1
        while True:
            chunk = f.read(PART_SIZE)
            if not chunk:
                break
            part = out_dir / f"{zip_path.name}.{idx:03d}"
            part.write_bytes(chunk)
            parts.append(part)
            print(f"  wrote {part.name} ({len(chunk) / 1e9:.2f} GB)")
            idx += 1
    return parts


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    version = _version()
    bundle = DIST_DIR / APP_NAME
    if not bundle.is_dir():
        print(f"ABORT: {bundle} not found. Run scripts/build_exe.py first.",
              file=sys.stderr)
        return 1

    zip_path = DIST_DIR / f"{APP_NAME}-{version}.zip"
    rel_dir = DIST_DIR / f"release_v{version}"
    rel_dir.mkdir(parents=True, exist_ok=True)

    print(f"Packaging release v{version} -> {rel_dir}")
    t0 = time.time()
    _zip_bundle(bundle, zip_path)

    print("  splitting into <= 1900 MiB parts ...")
    parts = _split(zip_path, rel_dir)

    print("  hashing ...")
    lines = [f"{_sha256(p)} *{p.name}" for p in parts]
    weights = rel_dir / f"{APP_NAME}-weights-{version}.zip"
    if weights.is_file():
        lines.append(f"{_sha256(weights)} *{weights.name}")
    else:
        print(f"  WARN: {weights.name} not found -- it is independent of the "
              f"code; carry it over from the previous release. Omitted from "
              f"SHA256SUMS.", file=sys.stderr)

    body = "\n".join(lines) + "\n\n"
    body += f"# Expected SHA-256 of the reassembled {zip_path.name}:\n"
    body += f"# {_sha256(zip_path)}  {zip_path.name}\n"
    sums = rel_dir / "SHA256SUMS.txt"
    sums.write_text(body, encoding="utf-8")
    print(f"  wrote {sums.name}")
    print(f"\nDone in {time.time() - t0:.1f}s. Release assets in {rel_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
