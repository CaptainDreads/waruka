# Building Waruka

This document explains how to produce a portable Windows bundle of Waruka
(`waruka.exe` + dependencies) from a clean clone. Everything below is
Windows-only.

The build does **not** require an NVIDIA GPU. You can build on any modern
Windows machine and ship the resulting zip to users who have NVIDIA
hardware -- Waruka's runtime checks for CUDA, so the bundle gracefully
degrades on machines without it (core pipeline works; the
GPU-only `interpolate`, `upscale`, and `--sr` paths fail with a clear
message).

## TL;DR

```powershell
pip install -r requirements.txt    # one-time
.\build.bat                        # one-click build + zip
```

## Overview

Waruka is packaged with [PyInstaller](https://pyinstaller.org/) into a
portable Windows `--onedir` bundle. There's no installer, no registry
writes, no admin rights -- the user unzips and runs. Two executables
share the bundle: a windowed `waruka.exe` for double-click GUI launch
(no console window) and a console `waruka-cli.exe` for PowerShell-driven
CLI use.

The build pipeline runs in this order:

```
build.bat                              one-click wrapper
  -> python scripts/build_exe.py
       1. _check_prereqs()             verify weights + PyInstaller present, fail fast
       2. _clean()                     wipe dist/waruka/, build/, leftover .spec files
       3. PyInstaller pass 1 (GUI)     waruka.exe (--windowed) -> dist/waruka/
       4. PyInstaller pass 2 (CLI)     waruka-cli.exe (--console) -> dist/_cli_staging/
       5. copy waruka-cli.exe           into dist/waruka/, drop staging dir
       6. _copy_user_docs()            README.md + docs/ at bundle root
       7. scripts/prune_bundle.py      drop unused Qt + third-party assets (~600 MB)
       8. shutil.make_archive          dist/waruka-<version>.zip (only with --zip)
```

Each script in `scripts/` is single-purpose and has a docstring header
explaining what it does and why:

| Script | Purpose |
|---|---|
| [scripts/build_exe.py](scripts/build_exe.py) | Orchestrates the full build. Read this if a build step is failing or you want to add a new bundled asset. |
| [scripts/waruka_launcher.py](scripts/waruka_launcher.py) | Entry point baked into both exes. Sets up file logging + DLL search paths before dispatching to `waruka.__main__`. |
| [scripts/prune_bundle.py](scripts/prune_bundle.py) | Post-build prune of unused Qt subsystems (WebEngine, 3D, Quick/QML, Charts, Multimedia, etc.) and third-party demo/docs/inputs. Has a `--dry-run` flag for inspecting what would be cut. |

The whole pipeline takes 15-20 minutes on a modern dev box -- two
PyInstaller passes dominate (analysis is single-threaded and the import
graph is large). There is no incremental mode; a fresh build is always
from scratch. This is intentional: every shipped bundle is reproducible
from the same `requirements.txt`.

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Windows | 10 1909+ or 11 | Tested on Windows 10 Pro 22H2 |
| Python | 3.13.x | Older Python versions are not tested |
| pip | recent | `python -m pip install --upgrade pip` |
| Visual C++ Runtime | 2015-2022 | Almost always present already; `vc_redist.x64.exe` from microsoft.com if not |

Verify each:

```powershell
python --version    # Python 3.13.x
pip --version
```

## One-time setup

From a clean clone of the repo:

```powershell
# 1. Install Python deps (uses the CUDA 11.8 PyTorch wheel)
pip install -r requirements.txt

# 2. Confirm runtime assets are present. yolo11n.pt is committed to the
#    repo; the three optional-stage model weights are NOT (each exceeds
#    GitHub's file-size limit), so a fresh clone always needs to fetch
#    them before a full build -- see "Fetching weights" below.
python -c "from pathlib import Path; r = Path('.'); needed = [r/'yolo11n.pt', r/'third_party/film/film_net_fp16.pt', r/'third_party/rife/train_log/flownet.pkl', r/'third_party/realesrgan/weights/RealESRGAN_x2plus.pth']; [print('OK' if p.exists() else f'MISSING: {p}') for p in needed]"
```

If anything is missing, see [Fetching weights](#fetching-weights) below.

## Build

```powershell
python scripts/build_exe.py
```

Typical wall time: 15-20 minutes (two PyInstaller passes -- one for the
windowed GUI exe, one for the console CLI exe -- plus a post-build
prune pass). Output:

```
dist/waruka/
  waruka.exe                <-- double-click for GUI (no console window)
  waruka-cli.exe            <-- run from PowerShell for CLI subcommands
  _internal/...            <-- bundled Python, deps, model weights, CUDA DLLs
  logs/                    <-- created at first run; one file per launch
```

Bundle size is ~5 GB (model weights, torch's bundled CUDA libs, and the
nvidia-cu12 runtime are the bulk; PyInstaller can't make them much
smaller). Zipped distributable is ~3 GB.

**Two exes, shared `_internal/`.** `waruka.exe` is a windowed-subsystem
binary (no console flashes on double-click; GUI launches directly).
`waruka-cli.exe` is a console-subsystem binary with the same dispatch
behaviour but visible stdout/stderr -- preferred for PowerShell use,
piping, and exit-code chaining (`waruka-cli.exe track ... && waruka-cli.exe render ...`).
Both share `_internal/` bit-for-bit; the only difference is the PE
subsystem flag.

**Logging.** Every launch writes a log file at
`<exe_dir>/logs/waruka-<timestamp>-<pid>.log`. For the GUI exe this is
the only way to see what happened past startup, since there's no
terminal attached. For the CLI exe, output is tee'd to both the
terminal and the log file. If you hit an issue, open the most recent
log file -- the launch banner records argv + frozen state + exe dir at
the top.

### Useful flags

| Flag | Effect |
|---|---|
| `--smoke` | Launch `waruka.exe` for 6s after build; fail if it crashes on startup. Needs an NVIDIA GPU to be meaningful past Qt init. |
| `--zip`   | Also produce `dist/waruka-<version>.zip` for distribution. |
| `--no-prune` | Skip the post-build prune pass (`scripts/prune_bundle.py`). The prune cuts ~600 MB of unused Qt + third-party assets; pass this to debug a hook issue or to inspect the raw PyInstaller output. |
| `--keep-build` | Keep the `build/waruka/` PyInstaller scratch dir (useful when debugging hook issues). |

## Test the bundle

After a successful build:

```powershell
# GUI smoke test -- should open the main window within ~3 s, no terminal
dist\waruka\waruka.exe

# CLI smoke test -- prints help and exits, terminal output visible
dist\waruka\waruka-cli.exe --help

# Full pipeline smoke test on a short clip
dist\waruka\waruka-cli.exe track --project project.json --t0 0 --t1 5 --out _smoke_tracks.json
```

Anything that runs as `python -m waruka <subcommand>` in dev runs as
`waruka-cli.exe <subcommand>` in the bundle. `waruka.exe <subcommand>`
also works but has no visible terminal output -- check the log file.

## Ship it

```powershell
python scripts/build_exe.py --zip
# -> dist/waruka-<version>.zip
```

The zip is self-contained. Users unzip anywhere and run `waruka.exe`. No
install step, no admin rights, no registry writes -- Waruka writes its
working files to the directory it's run from.

## Fetching weights

The three optional-stage model weights are **not** committed to the repo
(each exceeds GitHub's 100 MB file limit), so every fresh clone needs to
fetch them before a full build. `yolo11n.pt` *is* committed, so you only
need it if you've pruned it. Two ways to get the weights:

**Easiest -- the release bundle.** Download `waruka-weights-1.0.0.zip`
from the [latest GitHub release](https://github.com/CaptainDreads/waruka/releases/latest)
and extract it at the repo root; the three files land at the correct
paths. (Same guidance as [`third_party/README.md`](third_party/README.md).)

**Or fetch each from upstream:**

| File | Source |
|---|---|
| `yolo11n.pt` | Committed to the repo. If pruned: auto-downloaded by ultralytics on first detection run, or `python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"` then copy from the ultralytics cache. |
| `third_party/film/film_net_fp16.pt` | <https://github.com/dajes/frame-interpolation-pytorch/releases> (pre-built FILM-Style TorchScript) |
| `third_party/rife/train_log/flownet.pkl` | <https://github.com/hzwer/Practical-RIFE> -- RIFE 4.25 model release |
| `third_party/realesrgan/weights/RealESRGAN_x2plus.pth` | <https://github.com/xinntao/Real-ESRGAN/releases> |

## Common build failures

**`ModuleNotFoundError: No module named 'PyInstaller'`**
The build script asks you to `pip install pyinstaller`.

**`ERROR: required asset(s) not found`**
See "Fetching weights" above. The script aborts before invoking
PyInstaller so you don't waste 5 minutes building a broken bundle.

**Bundle launches but `waruka.exe track ...` says `cudart64_12.dll not found`**
The `nvidia-cuda-runtime-cu12` wheel didn't get picked up. Confirm:
```powershell
python -c "import site, os; print([p for r in site.getsitepackages() for p in os.listdir(r) if 'cuda_runtime' in p.lower()])"
```
If empty, `pip install nvidia-cuda-runtime-cu12` and rebuild.

**Bundle launches but no GUI window appears**
PySide6 plugins may not have been collected. Confirm
`dist/waruka/_internal/PySide6/plugins/platforms/` exists; if not,
rebuild with `--clean` and report a hook bug.

## Code signing (deferred, FYI)

The bundle is unsigned; Windows SmartScreen will warn first-time users
("Windows protected your PC"). To sign:

1. Get an Authenticode certificate. For an OSS project the right path is
   [SignPath Foundation](https://signpath.org/foundation) -- free EV
   signing for qualifying open-source projects.
2. Install the Windows SDK to get `signtool.exe`.
3. After build: `signtool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 dist\waruka\waruka.exe`

Signing is not currently wired into the build script.
