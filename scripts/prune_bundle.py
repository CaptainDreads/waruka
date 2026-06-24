# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Post-build prune pass for `dist/waruka/`.

PyInstaller collects PySide6 and ultralytics aggressively because both
have sprawling import graphs. Waruka only uses a tiny slice -- gui.py
imports `QtCore, QtGui, QtWidgets` and nothing else; ultralytics drags in
matplotlib + pandas at import time but never touches them past that.

This script removes Qt subsystems we don't ship (WebEngine, Quick/QML,
3D, Charts, Bluetooth, Sensors, Pdf, Multimedia, ...), drops the unused
RIFE/FILM/Real-ESRGAN demo + docs + asset dirs, and clears __pycache__
trees. Run after `python scripts/build_exe.py`.

Each prune is conservative: only files/dirs that grep confirms are not
referenced from any waruka/*.py go on the denylist. New PyInstaller
versions may collect slightly different files; this script logs every
delete so you can verify nothing critical went.

Usage:
    python scripts/prune_bundle.py             # prune dist/waruka/
    python scripts/prune_bundle.py --dry-run   # show what would go
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = REPO_ROOT / "dist" / "waruka"
INTERNAL = BUNDLE_DIR / "_internal"


# Qt6 module *stems* we don't ship. Matches:
#   _internal/PySide6/Qt6<STEM>.dll        (libraries)
#   _internal/PySide6/<STEM>.pyd            (python bindings)
#   _internal/PySide6/<STEM>/               (resource subdir, sometimes)
#   _internal/PySide6/translations/qt<stem>_*.qm  (UI translations)
# waruka.gui uses only QtCore, QtGui, QtWidgets -- everything else is dead weight.
PYSIDE6_DROP_STEMS = [
    "WebEngineCore", "WebEngineQuick", "WebEngineWidgets",
    "WebChannel", "WebChannelQuick", "WebSockets", "WebView",
    "Quick", "Quick3D", "Quick3DAssetImport", "Quick3DAssetUtils",
    "Quick3DEffects", "Quick3DGlslParser", "Quick3DHelpers",
    "Quick3DHelpersImpl", "Quick3DIblBaker", "Quick3DParticleEffects",
    "Quick3DParticles", "Quick3DRuntimeRender", "Quick3DSpatialAudio",
    "Quick3DUtils", "Quick3DXr",
    "QuickControls2", "QuickControls2Basic", "QuickControls2Fusion",
    "QuickControls2Imagine", "QuickControls2Material",
    "QuickControls2Universal", "QuickControls2BasicStyleImpl",
    "QuickControls2FusionStyleImpl", "QuickControls2ImagineStyleImpl",
    "QuickControls2MaterialStyleImpl", "QuickControls2UniversalStyleImpl",
    "QuickDialogs2", "QuickDialogs2QuickImpl", "QuickDialogs2Utils",
    "QuickEffects", "QuickLayouts", "QuickParticles", "QuickShapes",
    "QuickTemplates2", "QuickTest", "QuickTimeline", "QuickVectorImage",
    "QuickVectorImageGenerator", "QuickWidgets",
    "Qml", "QmlCompiler", "QmlCore", "QmlIntegration", "QmlLocalStorage",
    "QmlMeta", "QmlModels", "QmlNetwork", "QmlWorkerScript", "QmlXmlListModel",
    "3DAnimation", "3DCore", "3DExtras", "3DInput", "3DLogic",
    "3DQuick", "3DQuickAnimation", "3DQuickExtras", "3DQuickInput",
    "3DQuickRender", "3DQuickScene2D", "3DRender",
    "Charts", "ChartsQml",
    "DataVisualization", "DataVisualizationQml",
    "Graphs", "GraphsWidgets",
    "Bluetooth", "Nfc", "SerialBus", "SerialPort",
    "Positioning", "PositioningQuick", "Location",
    "Sensors", "SensorsQuick",
    "RemoteObjects", "RemoteObjectsQml",
    "Scxml", "StateMachine", "StateMachineQml",
    "TextToSpeech",
    "Pdf", "PdfQuick", "PdfWidgets",
    "Multimedia", "MultimediaQuick", "MultimediaWidgets",
    "Designer", "DesignerComponents",
    "Help",
    "Test",
    "UiTools",
    "AxContainer",
    "HttpServer",
    "Sql",
    "SpatialAudio",
    "OpcUa",
]

# Bundled FFmpeg DLLs that ship with PySide6 specifically for QtMultimedia.
# We drop QtMultimedia, so these are dead weight (waruka uses imageio_ffmpeg's
# separate ffmpeg.exe binary).
PYSIDE6_DROP_FILES = [
    "avcodec-61.dll", "avformat-61.dll", "avutil-59.dll",
    "swresample-5.dll", "swscale-8.dll",
    "qmlls.exe", "qmlformat.exe", "qmllint.exe", "qmlplugindump.exe",
    "qmlprofiler.exe", "qmlscene.exe", "qmltestrunner.exe",
    "designer.exe", "linguist.exe", "lrelease.exe", "lupdate.exe",
    "qhelpgenerator.exe", "qmltyperegistrar.exe",
]

# Whole PySide6 resource subdirs we don't need.
PYSIDE6_DROP_SUBDIRS = [
    "resources",            # 70+ MB of webengine devtools .pak files
    "translations",         # UI translations for all Qt modules (we don't translate)
    "qml",                  # QML runtime modules
    "Assistant",
]

# third_party housekeeping: demo images, training docs, test fixtures.
THIRD_PARTY_DROP_DIRS = [
    "film/photos",
    "film/__pycache__",
    "rife/demo",
    "rife/model/__pycache__",
    "rife/train_log/__pycache__",
    "realesrgan/.github",
    "realesrgan/.vscode",
    "realesrgan/assets",
    "realesrgan/docs",
    "realesrgan/inputs",
    "realesrgan/options",
    "realesrgan/scripts",
    "realesrgan/tests",
    "realesrgan/experiments",
    "realesrgan/realesrgan/__pycache__",
]

THIRD_PARTY_DROP_FILES = [
    "film/film_net_l1_fp16.pt",     # L1 variant; waruka uses the style model only
    "film/film_net_fp16.pt.txt",    # if present
    "film/.gitignore",
    "film/README.md",
    "film/requirements.txt",
    "film/LICENSE",
    "film/export.py",
    "film/inference.py",
    "rife/.gitignore",
    "rife/README.md",
    "rife/requirements.txt",
    "rife/LICENSE",
    "rife/inference_img.py",
    "rife/inference_img_SR.py",
    "rife/inference_video.py",
    "rife/inference_video_enhance.py",
    "rife/rife_4_25.zip",
    "realesrgan/.gitignore",
    "realesrgan/.pre-commit-config.yaml",
    "realesrgan/CODE_OF_CONDUCT.md",
    "realesrgan/cog.yaml",
    "realesrgan/cog_predict.py",
    "realesrgan/inference_realesrgan.py",
    "realesrgan/inference_realesrgan_video.py",
    "realesrgan/MANIFEST.in",
    "realesrgan/README.md",
    "realesrgan/README_CN.md",
    "realesrgan/requirements.txt",
    "realesrgan/setup.cfg",
    "realesrgan/setup.py",
    "realesrgan/VERSION",
    "realesrgan/LICENSE",
    "realesrgan/weights/README.md",
]


def _human(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def _delete_path(p: Path, dry_run: bool, log: list[tuple[Path, int]]) -> int:
    if not p.exists():
        return 0
    if p.is_file() or p.is_symlink():
        size = p.stat().st_size
        log.append((p, size))
        if not dry_run:
            try:
                p.unlink()
            except OSError as e:
                print(f"  WARN: could not delete {p}: {e}", file=sys.stderr)
                return 0
        return size
    # Directory.
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    log.append((p, total))
    if not dry_run:
        shutil.rmtree(p, ignore_errors=True)
    return total


def _prune_pyside6(dry_run: bool, log: list) -> int:
    pyside = INTERNAL / "PySide6"
    if not pyside.is_dir():
        return 0
    saved = 0

    # DLL + pyd files matching the drop stems.
    for stem in PYSIDE6_DROP_STEMS:
        for p in pyside.glob(f"Qt6{stem}.dll"):
            saved += _delete_path(p, dry_run, log)
        for p in pyside.glob(f"Qt{stem}.pyd"):
            saved += _delete_path(p, dry_run, log)
        # The pure-stem .pyd variant (newer PySide6).
        for p in pyside.glob(f"{stem}.pyd"):
            saved += _delete_path(p, dry_run, log)
        # Some modules also have a subdir (e.g. PySide6/Qt6/qml/QtCharts).
        for p in pyside.glob(f"Qt{stem}"):
            if p.is_dir():
                saved += _delete_path(p, dry_run, log)

    for name in PYSIDE6_DROP_FILES:
        saved += _delete_path(pyside / name, dry_run, log)

    for sub in PYSIDE6_DROP_SUBDIRS:
        saved += _delete_path(pyside / sub, dry_run, log)

    # Plugin subdirs: drop anything inside plugins/ that's clearly for a
    # subsystem we don't use. Conservative -- keep platforms/, styles/,
    # imageformats/ (loaded dynamically by QtGui).
    plugins = pyside / "plugins"
    if plugins.is_dir():
        keep = {"platforms", "styles", "imageformats", "iconengines",
                "tls", "generic"}
        for child in plugins.iterdir():
            if child.is_dir() and child.name not in keep:
                saved += _delete_path(child, dry_run, log)

    return saved


def _prune_third_party(dry_run: bool, log: list) -> int:
    base = INTERNAL / "third_party"
    if not base.is_dir():
        return 0
    saved = 0
    for rel in THIRD_PARTY_DROP_DIRS:
        saved += _delete_path(base / rel, dry_run, log)
    for rel in THIRD_PARTY_DROP_FILES:
        saved += _delete_path(base / rel, dry_run, log)
    return saved


def _prune_pycache(dry_run: bool, log: list) -> int:
    saved = 0
    for p in BUNDLE_DIR.rglob("__pycache__"):
        if p.is_dir():
            saved += _delete_path(p, dry_run, log)
    return saved


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="show what would be deleted; don't touch the bundle")
    p.add_argument("--verbose", action="store_true",
                   help="list every deleted path")
    a = p.parse_args()

    if not BUNDLE_DIR.is_dir():
        print(f"ERROR: bundle dir not found: {BUNDLE_DIR.relative_to(REPO_ROOT)}",
              file=sys.stderr)
        return 1

    before = sum(f.stat().st_size for f in BUNDLE_DIR.rglob("*") if f.is_file())
    print(f"Bundle: {BUNDLE_DIR.relative_to(REPO_ROOT)}  ({_human(before)} before)")
    if a.dry_run:
        print("[dry-run mode -- no files will be touched]")

    log: list[tuple[Path, int]] = []
    pyside_saved = _prune_pyside6(a.dry_run, log)
    print(f"  PySide6 prune    : {_human(pyside_saved)}")
    tp_saved = _prune_third_party(a.dry_run, log)
    print(f"  third_party prune: {_human(tp_saved)}")
    pyc_saved = _prune_pycache(a.dry_run, log)
    print(f"  __pycache__ prune: {_human(pyc_saved)}")

    if a.verbose:
        print("\nDeleted paths (largest first):")
        for path, size in sorted(log, key=lambda x: -x[1])[:50]:
            print(f"  {_human(size):>10}  {path.relative_to(BUNDLE_DIR)}")

    total_saved = pyside_saved + tp_saved + pyc_saved
    print(f"\nTotal: {_human(total_saved)} saved")
    if not a.dry_run:
        after = sum(f.stat().st_size for f in BUNDLE_DIR.rglob("*") if f.is_file())
        print(f"Bundle now: {_human(after)} ({_human(before - after)} smaller)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
