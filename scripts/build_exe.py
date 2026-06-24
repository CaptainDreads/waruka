# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Build a portable Windows bundle of Waruka via PyInstaller.

Usage:
    python scripts/build_exe.py            # build, no smoke test
    python scripts/build_exe.py --smoke    # build + launch dist/waruka/waruka.exe briefly
    python scripts/build_exe.py --zip      # build + produce dist/waruka-<version>.zip

Output layout (after a successful run):
    dist/waruka/waruka.exe          <-- double-click to start the GUI (no console)
    dist/waruka/waruka-cli.exe      <-- run from PowerShell for CLI subcommands
    dist/waruka/README.md          <-- user-facing overview
    dist/waruka/docs/              <-- CLI reference + GUI walkthrough + screenshots
    dist/waruka/_internal/...      <-- bundled Python, deps, weights
    dist/waruka/logs/...           <-- created at first run; one file per launch
    dist/waruka-<version>.zip      <-- only when --zip is passed

The two exes share `_internal/` bit-for-bit and only differ in their PE
subsystem flag (windowed vs console). Both write a log file in
`<exe_dir>/logs/waruka-<timestamp>-<pid>.log` so the GUI-mode launch
(which has no terminal) still produces a diagnostic trail.

Prerequisites are documented in BUILDING.md. The short version: a working
Waruka dev env (Python 3.13, `pip install`'d deps, weights present), plus
PyInstaller. NVIDIA hardware is NOT required to build -- contributors can
build the bundle on any Windows machine and ship it to NVIDIA users.

The script aborts early if anything required is missing rather than producing
a half-broken bundle. Each error message points at the fix.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"
BUILD_DIR = REPO_ROOT / "build"
APP_NAME = "waruka"
CLI_NAME = "waruka-cli"
CLI_STAGING = DIST_DIR / "_cli_staging"


# ---------------------------------------------------------------------------
# Required runtime assets. Each entry is checked before invoking PyInstaller
# so contributors get a fast, clear error rather than a broken bundle at the
# end of a 5-minute build.
# ---------------------------------------------------------------------------
REQUIRED_FILES = [
    REPO_ROOT / "yolo11n.pt",
    REPO_ROOT / "third_party" / "film" / "film_net_fp16.pt",
    REPO_ROOT / "third_party" / "rife" / "train_log" / "flownet.pkl",
    REPO_ROOT / "third_party" / "realesrgan" / "weights" / "RealESRGAN_x2plus.pth",
    REPO_ROOT / "third_party" / "realesrgan" / "realesrgan" / "archs" / "rrdbnet_arch.py",
    REPO_ROOT / "waruka" / "__main__.py",
]

# Bundled as raw data (PyInstaller doesn't statically analyse the runtime
# importlib / chdir+sys.path loads inside these dirs, so they must ship as
# files rather than be discovered as modules).
DATA_DIRS = [
    "third_party/film",
    "third_party/rife",
    "third_party/realesrgan",
]
DATA_FILES = [
    ("yolo11n.pt", "."),
    # Both icons ship in _internal/icons/. waruka.ico is the GUI icon
    # (loaded at runtime by MainWindow.setWindowIcon for title bar /
    # taskbar). waruka-cli.ico is only useful as the .exe PE icon
    # baked at build time, but ship it anyway so contributors who
    # poke around the bundle see both.
    ("icons/waruka.ico", "icons"),
    ("icons/waruka-cli.ico", "icons"),
]


# User-facing documentation that ships AT THE BUNDLE ROOT (alongside waruka.exe),
# not inside `_internal/`. Each entry is (src_relpath, dst_relpath_in_bundle).
# Copied post-build because PyInstaller's --add-data targets `_internal/`.
USER_DOCS = [
    ("README.md",                  "README.md"),
    ("LICENSE",                    "LICENSE"),
    ("NOTICE.md",                  "NOTICE.md"),
    ("HARDWARE.md",                "docs/HARDWARE.md"),
    ("docs/cli_reference.md",      "docs/cli_reference.md"),
    ("docs/cli_reference.pdf",     "docs/cli_reference.pdf"),
    ("docs/gui_walkthrough.md",    "docs/gui_walkthrough.md"),
    ("docs/gui_walkthrough.pdf",   "docs/gui_walkthrough.pdf"),
]
USER_DOC_DIRS = [
    ("docs/screenshots", "docs/screenshots"),
]


# ---------------------------------------------------------------------------
def _read_version() -> str:
    init = REPO_ROOT / "waruka" / "__init__.py"
    for line in init.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("__version__"):
            return line.split("=", 1)[1].strip().strip("\"'")
    return "0.0.0"


def _check_prereqs() -> None:
    missing = [str(p.relative_to(REPO_ROOT)) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        print("ERROR: required asset(s) not found:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print("\nCheck BUILDING.md for how to fetch each asset.", file=sys.stderr)
        sys.exit(1)
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("ERROR: PyInstaller is not installed.\n"
              "  pip install pyinstaller", file=sys.stderr)
        sys.exit(1)


def _clean() -> None:
    for d in (DIST_DIR / APP_NAME, BUILD_DIR / APP_NAME,
              CLI_STAGING, BUILD_DIR / CLI_NAME):
        if d.exists():
            print(f"  removing {d.relative_to(REPO_ROOT)}/")
            shutil.rmtree(d, ignore_errors=True)
    for spec_name in (APP_NAME, CLI_NAME):
        spec = REPO_ROOT / f"{spec_name}.spec"
        if spec.exists():
            spec.unlink()


def _enumerate_waruka_submodules() -> list[str]:
    """Every .py file under waruka/ becomes waruka.<name>.

    We pass these as --hidden-import rather than --collect-submodules
    because PyInstaller's pkgutil-based submodule collector silently
    skips packages that aren't importable at arg-parse time (waruka
    isn't installed -- it lives under repo root, which only gets added
    to sys.path during the Analysis pass).
    """
    pkg = REPO_ROOT / "waruka"
    mods = []
    for p in sorted(pkg.glob("*.py")):
        name = p.stem
        if name.startswith("_"):
            continue  # __init__, __main__ are picked up by static analysis
        mods.append(f"waruka.{name}")
    return mods


def _build_pyinstaller_args(name: str, *, windowed: bool,
                              workpath: Path | None = None,
                              distpath: Path | None = None) -> list[str]:
    """Build the PyInstaller CLI argv for one of the two exes.

    `windowed=True`  -> Windows GUI subsystem, no console allocated.
                        Used for `waruka.exe` (double-click friendly).
    `windowed=False` -> console subsystem. Used for `waruka-cli.exe`
                        (PowerShell-friendly: stdout/stderr visible,
                        exit codes returned to the shell).

    Both share the same dep graph and data; the only differences are
    the subsystem flag, the output name, and (for the CLI build) the
    workpath/distpath redirected to a staging area so we don't trash
    the GUI build's _internal/.
    """
    sep = os.pathsep  # ";" on Windows, ":" elsewhere

    # Pick the right icon per build pass: waruka.ico for the GUI exe,
    # waruka-cli.ico for the console exe. Falls back to waruka.ico if
    # the CLI icon isn't present.
    gui_icon = REPO_ROOT / "icons" / "waruka.ico"
    cli_icon = REPO_ROOT / "icons" / "waruka-cli.ico"
    icon_path = gui_icon if windowed else (
        cli_icon if cli_icon.is_file() else gui_icon)
    args = [
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed" if windowed else "--console",
        "--name", name,
        "--paths", str(REPO_ROOT),
    ]
    if icon_path.is_file():
        # PE icon -- shows in Explorer + Task Manager + the EXE itself.
        # MainWindow.setWindowIcon() handles the title-bar and taskbar
        # icons at runtime (GUI exe only); the AppUserModelID call in
        # the launcher handles Windows' taskbar grouping.
        args += ["--icon", str(icon_path)]
    args += [
        # Critical: pull in everything PyInstaller's static analysis tends to miss.
        "--collect-all", "PySide6",
        # ultralytics has a sprawling optional-dep graph (tensorflow, jax,
        # comet, mlflow, ...). Collect data + binaries (model configs) but
        # let static analysis decide what Python modules are needed -- this
        # alone trims ~1-2 GB versus --collect-all.
        "--collect-data", "ultralytics",
        "--collect-binaries", "ultralytics",
        "--collect-all", "imageio_ffmpeg",
        "--collect-binaries", "PyNvVideoCodec",
        "--collect-data", "PyNvVideoCodec",
        # nvidia.cuda_runtime ships cudart64_12.dll that waruka.nvdecode
        # loads explicitly. The other nvidia.* packages (cublas, cudnn,
        # cufft, ...) are pulled in via torch's analysis as needed.
        "--collect-all", "nvidia.cuda_runtime",
        # Hidden imports: pulled in at runtime by chdir+sys.path or by
        # importlib.util.spec_from_file_location, so static analysis misses them.
        "--hidden-import", "torchvision",
        "--hidden-import", "cv2",
        # Trim ultralytics' optional ML-backend dependency graph.
        "--exclude-module", "tensorflow",
        "--exclude-module", "keras",
        "--exclude-module", "jax",
        "--exclude-module", "jaxlib",
        "--exclude-module", "comet_ml",
        "--exclude-module", "mlflow",
        "--exclude-module", "wandb",
        "--exclude-module", "boto3",
        "--exclude-module", "botocore",
        "--exclude-module", "roboflow",
        "--exclude-module", "h5py",
        # NOTE: matplotlib and pandas are imported unconditionally inside
        # ultralytics.utils -- excluding them crashes `waruka track` on
        # first model load. Keep them in the bundle.
    ]

    for mod in _enumerate_waruka_submodules():
        args += ["--hidden-import", mod]

    for rel in DATA_DIRS:
        src = REPO_ROOT / rel
        if not src.is_dir():
            print(f"  skipping (not found): {rel}")
            continue
        args += ["--add-data", f"{src}{sep}{rel}"]

    for src_rel, dst_rel in DATA_FILES:
        src = REPO_ROOT / src_rel
        if not src.is_file():
            print(f"  skipping (not found): {src_rel}")
            continue
        args += ["--add-data", f"{src}{sep}{dst_rel}"]

    if workpath is not None:
        args += ["--workpath", str(workpath)]
    if distpath is not None:
        args += ["--distpath", str(distpath)]

    args.append(str(REPO_ROOT / "scripts" / "waruka_launcher.py"))
    return args


def _run_pyinstaller(args: list[str], label: str = "PyInstaller") -> None:
    import PyInstaller.__main__ as pim

    print(f"Running {label}...")
    t0 = time.time()
    pim.run(args)
    print(f"\n{label} finished in {time.time() - t0:.1f} s")


def _copy_user_docs() -> None:
    """Place user-facing docs at the bundle root so end users find them.

    PyInstaller's --add-data targets `_internal/` (the contents dir), which
    is fine for runtime resources (weights, model code) but wrong for things
    the user reads (README, CLI reference PDF, GUI walkthrough). Those want
    to sit next to waruka.exe.
    """
    bundle = DIST_DIR / APP_NAME
    if not bundle.is_dir():
        return
    print("\nCopying user docs to bundle root...")
    n_files = 0
    n_bytes = 0
    for src_rel, dst_rel in USER_DOCS:
        src = REPO_ROOT / src_rel
        if not src.is_file():
            print(f"  skipping (not found): {src_rel}")
            continue
        dst = bundle / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n_files += 1
        n_bytes += src.stat().st_size
    for src_rel, dst_rel in USER_DOC_DIRS:
        src = REPO_ROOT / src_rel
        if not src.is_dir():
            print(f"  skipping (not found): {src_rel}")
            continue
        dst = bundle / dst_rel
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        for f in dst.rglob("*"):
            if f.is_file():
                n_files += 1
                n_bytes += f.stat().st_size
    print(f"  copied {n_files} files ({n_bytes / 1024:.1f} KB)")


def _build_cli_companion() -> None:
    """Build the console-subsystem `waruka-cli.exe` and slot it next to
    `waruka.exe` in the GUI bundle. Reuses the same dep graph; only the
    PE subsystem flag differs.

    We can't share the analysis pass between the two builds (PyInstaller
    doesn't expose that across separate invocations without a custom
    .spec file), so this is a second full ~8 min build. The trade is
    simpler code -- a contributor reading build_exe.py sees two
    PyInstaller calls instead of a hand-written spec.
    """
    print("\n--- Second build: console companion (waruka-cli.exe) ---")
    args = _build_pyinstaller_args(
        CLI_NAME,
        windowed=False,
        workpath=BUILD_DIR / CLI_NAME,
        distpath=CLI_STAGING,
    )
    _run_pyinstaller(args, label=f"PyInstaller ({CLI_NAME})")

    src_exe = CLI_STAGING / CLI_NAME / f"{CLI_NAME}.exe"
    dst_exe = DIST_DIR / APP_NAME / f"{CLI_NAME}.exe"
    if not src_exe.exists():
        print(f"ERROR: expected {src_exe} after CLI build; not found.",
              file=sys.stderr)
        sys.exit(1)
    shutil.copy2(src_exe, dst_exe)
    print(f"  copied {dst_exe.relative_to(REPO_ROOT)}")
    # Drop the staging dir -- its _internal/ duplicates the GUI bundle's,
    # and the CLI exe needs the GUI bundle's _internal/ to run anyway
    # (they share dependencies bit-for-bit).
    shutil.rmtree(CLI_STAGING, ignore_errors=True)


def _summarise() -> Path:
    dist = DIST_DIR / APP_NAME
    if not (dist / f"{APP_NAME}.exe").exists():
        print(f"ERROR: expected {dist / (APP_NAME + '.exe')} after build, "
              f"but it's not there. Inspect PyInstaller output above.",
              file=sys.stderr)
        sys.exit(1)
    total_bytes = sum(p.stat().st_size for p in dist.rglob("*") if p.is_file())
    print(f"\nBundle: {dist.relative_to(REPO_ROOT)}/")
    print(f"  waruka.exe at: {(dist / (APP_NAME + '.exe')).relative_to(REPO_ROOT)}")
    print(f"  total size : {total_bytes / 1e9:.2f} GB")
    return dist


def _smoke_launch(dist: Path) -> None:
    """Briefly launch waruka.exe to confirm it doesn't crash on startup.

    We can't fully test the GUI in batch -- but we can prove the bundle
    is at least loadable. Launch detached, wait 6 s, then terminate.
    """
    exe = dist / f"{APP_NAME}.exe"
    print(f"\nSmoke: launching {exe.name} for 6 s...")
    proc = subprocess.Popen([str(exe)], cwd=str(dist))
    try:
        time.sleep(6.0)
        if proc.poll() is not None:
            print(f"  FAILED: process exited with code {proc.returncode}")
            sys.exit(1)
        print("  OK: process stayed up for 6 s")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _zip_bundle(dist: Path, version: str) -> Path:
    archive_base = DIST_DIR / f"{APP_NAME}-{version}"
    print(f"\nZipping bundle -> {archive_base.name}.zip ...")
    t0 = time.time()
    archive = shutil.make_archive(str(archive_base), "zip",
                                   root_dir=str(DIST_DIR),
                                   base_dir=APP_NAME)
    print(f"  wrote {Path(archive).name} "
          f"({Path(archive).stat().st_size / 1e9:.2f} GB) "
          f"in {time.time() - t0:.1f} s")
    return Path(archive)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true",
                   help="briefly launch waruka.exe after build to verify it starts")
    p.add_argument("--zip", action="store_true",
                   help="zip dist/waruka/ into dist/waruka-<version>.zip")
    p.add_argument("--no-prune", action="store_true",
                   help="skip the post-build prune pass (scripts/prune_bundle.py)")
    p.add_argument("--keep-build", action="store_true",
                   help="don't remove the build/waruka/ scratch dir on success")
    a = p.parse_args()

    version = _read_version()
    print(f"Waruka build: version {version}, repo at {REPO_ROOT}")

    print("\nChecking prerequisites...")
    _check_prereqs()
    print("  OK")

    print("\nCleaning previous build...")
    _clean()

    print("\n--- First build: GUI (waruka.exe, windowed) ---")
    gui_args = _build_pyinstaller_args(APP_NAME, windowed=True)
    _run_pyinstaller(gui_args, label=f"PyInstaller ({APP_NAME})")
    dist = _summarise()

    _build_cli_companion()
    _copy_user_docs()

    if not a.no_prune:
        print("\nRunning post-build prune...")
        prune_script = REPO_ROOT / "scripts" / "prune_bundle.py"
        if prune_script.exists():
            subprocess.run([sys.executable, str(prune_script)], check=True)
        else:
            print("  (scripts/prune_bundle.py not found; skipping)")

    if a.smoke:
        _smoke_launch(dist)
    if a.zip:
        _zip_bundle(dist, version)

    if not a.keep_build:
        for scratch in (BUILD_DIR / APP_NAME, BUILD_DIR / CLI_NAME):
            if scratch.exists():
                shutil.rmtree(scratch, ignore_errors=True)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
