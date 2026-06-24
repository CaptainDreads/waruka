# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Produce a clean, source-only release distribution.

Reads from `<repo>/` and writes to `<parent>/Waruka_<version>/`. The
output is the minimal set of files needed to run Waruka from source:
the `waruka/` package, vendored ML code + weights under `third_party/`,
the build scripts, the docs, and the legal files.

Excluded: dev artefacts (tracks_*.json, players_*.json, broadcast_*.mp4,
`_handover_*.md`, scratch `_*` dirs, dist/build directories, the
`.claude` Code metadata, any generated `*.spec` files), every
`__pycache__`, and the `.git` directory if present.

The result is what you'd commit to a fresh OSS repo: nothing the user
doesn't need, nothing identifying as session-private.

Usage:
    python scripts/make_release_dist.py                 # source-only
    python scripts/make_release_dist.py --include-build # source + dist/waruka/ + zip
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"

# Read the version from the package -- the release dir is named after it.
def _version() -> str:
    init = REPO_ROOT / "waruka" / "__init__.py"
    for line in init.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("__version__"):
            return line.split("=", 1)[1].strip().strip("\"'")
    return "0.0.0"


# Files to copy at the release-dist root. Dev env and release dist
# share the same clean filenames (README.md, docs/cli_reference.md etc),
# so most entries are identity copies. Listed explicitly rather than
# wildcard-scanning to make every shipped file an intentional choice.
COPY_FILES = [
    "README.md",
    "LICENSE",
    "NOTICE.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "BACKLOG.md",
    "BUILDING.md",
    "HARDWARE.md",
    "requirements.txt",
    "build.bat",
    "yolo11n.pt",
    "docs/cli_reference.md",
    "docs/cli_reference.pdf",
    "docs/gui_walkthrough.md",
    "docs/gui_walkthrough.pdf",
]
COPY_DIRS = [
    "waruka",
    "scripts",
    "third_party",
    "docs/screenshots",
]
# Items inside copied dirs that we explicitly drop.
SKIP_INSIDE = {"__pycache__", ".git", ".claude"}


def _copytree_clean(src: Path, dst: Path) -> tuple[int, int]:
    """Copy a directory tree, skipping __pycache__ and friends."""
    def ignore(d, names):
        return {n for n in names if n in SKIP_INSIDE}
    shutil.copytree(src, dst, ignore=ignore)
    files = [f for f in dst.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def _copy_build_artifacts(out_root: Path, version: str) -> tuple[int, int]:
    """Optionally copy the existing PyInstaller bundle + zip across.

    Run after the normal `python scripts/build_exe.py --zip` so the
    release dir is one place that has BOTH the source you'd commit AND
    the binaries you'd attach to a GitHub release. The bundle is
    content-identical regardless of which source root produced it
    (everything PyInstaller bundles -- weights, third_party, vendored
    Python -- comes from the same files), so it's safe to just copy.
    """
    bundle_src = DIST_DIR / "waruka"
    zip_src = DIST_DIR / f"waruka-{version}.zip"
    if not bundle_src.is_dir():
        print(f"  WARN: {bundle_src} not found; build the project first "
              f"(python scripts/build_exe.py --zip)")
        return (0, 0)
    dst_dist = out_root / "dist"
    dst_dist.mkdir(parents=True, exist_ok=True)
    print("  copying dist/waruka/ ...")
    # Skip the runtime `logs/` dir (created on first run -- not part of a
    # release, and may be open/locked while a job is running) and any
    # stray __pycache__.
    def _ignore_runtime(_d, names):
        return {n for n in names if n in {"logs", "__pycache__"}}
    shutil.copytree(bundle_src, dst_dist / "waruka", ignore=_ignore_runtime)
    n_files = sum(1 for f in (dst_dist / "waruka").rglob("*") if f.is_file())
    n_bytes = sum(f.stat().st_size for f in (dst_dist / "waruka").rglob("*") if f.is_file())
    if zip_src.is_file():
        shutil.copy2(zip_src, dst_dist / zip_src.name)
        n_files += 1
        n_bytes += zip_src.stat().st_size
        print(f"  copied {zip_src.name}")
    else:
        print(f"  (no {zip_src.name} found; build with --zip to produce one)")
    return (n_files, n_bytes)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--include-build", action="store_true",
                   help="also copy dist/waruka/ + waruka-<version>.zip into "
                        "the release dist. Run after scripts/build_exe.py.")
    a = p.parse_args()

    version = _version()
    out_root = REPO_ROOT.parent / f"Waruka_{version}"
    if out_root.exists():
        print(f"ABORT: {out_root} already exists. Delete or move it first.")
        return 1

    print(f"Building source distribution at {out_root} ...")
    t0 = time.time()
    out_root.mkdir(parents=True)

    n_files = 0
    n_bytes = 0
    for rel in COPY_FILES:
        src = REPO_ROOT / rel
        if not src.is_file():
            print(f"  skip (not found): {rel}")
            continue
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n_files += 1
        n_bytes += src.stat().st_size

    for rel in COPY_DIRS:
        src = REPO_ROOT / rel
        if not src.is_dir():
            print(f"  skip (not found): {rel}")
            continue
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        nf, nb = _copytree_clean(src, dst)
        print(f"  {rel:<22} {nf} files, {nb / 1024 / 1024:.1f} MB")
        n_files += nf
        n_bytes += nb

    if a.include_build:
        print("\nIncluding build artifacts ...")
        bf, bb = _copy_build_artifacts(out_root, version)
        n_files += bf
        n_bytes += bb

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s -- {n_files} files, "
          f"{n_bytes / 1024 / 1024:.1f} MB")
    print(f"Path: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
