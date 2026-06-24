# Changelog

All notable changes to Waruka.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> The project was originally developed under the name **Vemos** and
> renamed to **Waruka** in v1.0.0 to avoid potential trademark
> conflict with an unrelated commercial company. The codebase and
> history are otherwise continuous; older entries below refer to
> "Vemos" / `vemos/` where appropriate.

For in-flight / planned work see [BACKLOG.md](BACKLOG.md).

---

## [1.0.0] -- 2026-06-15

First open-source release. Project renamed from Vemos to Waruka and
released under the GNU Affero General Public License v3.0.

### Added

- **#24 + #25** -- single-exe Windows build via PyInstaller.
  Dual-exe: `waruka.exe` (windowed subsystem) for double-click GUI
  launch, `waruka-cli.exe` (console subsystem) for PowerShell-driven
  CLI use, sharing one `_internal/` between them.
- File logging in the launcher: three-layer redirect (Python
  `sys.stdout` / `sys.stderr` + OS fd 1 / fd 2 via `dup2` + Windows
  `STD_OUTPUT_HANDLE` / `STD_ERROR_HANDLE` via `SetStdHandle`) so
  subprocess stderr from ffmpeg, NVENC, etc. lands in the log even
  when launched as the windowed exe.
- PE-subsystem detection in the launcher: reads the exe's PE header
  Subsystem field to decide console vs windowed mode reliably,
  regardless of how PowerShell or the harness is configured.
- `scripts/build_exe.py` -- the build orchestrator. Two PyInstaller
  passes, post-build prune, optional zip. ~17 min wall time.
- `scripts/prune_bundle.py` -- post-build prune of unused Qt
  subsystems (WebEngine, Quick/QML, 3D, Charts, etc.) and
  third-party demo/docs/inputs. Saves ~600 MB.
- `scripts/make_release_dist.py` -- produces a clean
  `Waruka_<version>/` source release dist. `--include-build` flag
  bundles the PyInstaller artefacts alongside source.
- `build.bat` -- one-click build wrapper at repo root.
- `BUILDING.md` -- one-screen contributor doc with build-pipeline
  overview + step-by-step.
- `HARDWARE.md` -- supported GPU matrix, VRAM budget per stage,
  per-stage speed expectations.
- `CONTRIBUTING.md` -- DCO (Developer Certificate of Origin)
  sign-off instructions for OSS contributions.
- `LICENSE` -- canonical AGPL-3.0 text from gnu.org.
- `NOTICE.md` -- third-party attribution table (PyTorch, ultralytics,
  PySide6, NVIDIA, RIFE, FILM, Real-ESRGAN, PyInstaller, etc.) plus
  redistribution guidance.
- `BACKLOG.md` -- structured post-1.0 roadmap with architectural
  notes and explicit rejection reasoning.
- `dev/README.md` -- index for the development-scratch subtree
  (test clips, run outputs, scratch dirs, session handovers).
- SPDX `AGPL-3.0-or-later` headers + copyright notices on all 26
  source files under `waruka/` and `scripts/`.

### Changed

- **Renamed from Vemos to Waruka.** 1041 substitutions across 146
  files. Package dir `vemos/` -> `waruka/`. Launcher
  `vemos_launcher.py` -> `waruka_launcher.py`. All comments, docs,
  and code references updated.
- Version 0.16.0 -> 1.0.0.
- Filename normalisation: `_dist_readme.md` -> `README.md`,
  `_cli_documentation.{md,pdf}` -> `docs/cli_reference.{md,pdf}`,
  `_gui_documentation.{md,pdf}` -> `docs/gui_walkthrough.{md,pdf}`,
  `_doc_screenshots/` -> `docs/screenshots/`. Updated build script
  `USER_DOCS` accordingly.
- Working-dir reorg: 1,355 dev artefacts (10.9 GB) moved out of
  repo root into `dev/{scratch, clips, runs, projects, alt_models,
  handovers, progress, misc}/`. Repo root is now 10 files + 8 dirs.
- Queue add-job dialog brought in line with the Concat tab: separate
  **folder** + **concat filename** fields (filename prefilled `YYYYMMDD `
  from the first clip's recording date), **Preserve audio in
  concatenated output** and **Also write a silent copy of the concat**
  options, and a live preview of the artefacts produced. The user names
  the *concat* file; the tracked output is that name + `_broadcast`
  (concat, optional silent concat, broadcast, optional silent broadcast).

### Fixed

- **Calibrate-launch bug** affecting GUI subprocess calls. Two-part
  fix:
  - The launcher now strips a leading `-m waruka` from `argv` so
    legacy callers that build subprocess invocations as
    `python -m waruka <cmd>` don't trip argparse with exit code 2
    in the frozen bundle (where `sys.executable` is `waruka.exe`,
    not `python.exe`).
  - `StepRunner.start()` in `waruka/gui.py` switches the subprocess
    program to `waruka-cli.exe` (console subsystem) when frozen,
    so `QProcess` can read stderr from the merged-channel pipe
    instead of getting empty output via the windowed exe's
    `SetStdHandle` redirect.
- Post-rename oversight: `realesrgan/inputs/video/onepiece_demo.mp4`
  was leaking into the bundle; the prune script now removes it
  reliably.
- Concat output-safety hardening: a queue/concat output can never be
  written onto one of its own input clips (a colliding name would let
  `ffmpeg -y` truncate a source mid-concat). A shared `assert_output_safe`
  guard hard-fails the stage (`jobqueue.concat_cmd`,
  `pipeline._concat_mp4s`, Concat-tab save), and both dialogs block output
  names that already exist or collide with an input -- failing closed
  rather than risking source data.

---

## [0.16.0] -- early June 2026

### Added

- **#43** -- `vemos interpolate` rewritten around NVDEC + GPU-resident
  `a_t` cache + three-thread pipeline (decoder / model / encoder) +
  batched dts. Loop wall 19.9 s -> 10.9 s on the 5 s 1440p bench.
  100-min match estimate 6.7 h -> 3.6 h. New flags: `--no-nvdec`,
  `--no-pipeline`, `--no-batch-dts`, `--cq`.
- `vemos upscale` -- new CLI command for standalone 2x super-
  resolution via Real-ESRGAN x2plus on any video. Mirrors
  interpolate's NVDEC -> GPU -> NVENC architecture.
- **#35** -- Queue tab + Post-process tab in the GUI. Persistent
  JSON-backed batch processing with crash recovery, pause / resume,
  retry, per-job logs. Add-job dialog mirrors the Track tab.
- New GUI documentation (`_gui_documentation.md`/`.pdf`) with
  screenshots in `_doc_screenshots/`.

### Fixed

- **#42** -- "RIFE shimmer" diagnosed as NVENC bitrate starvation,
  not RIFE itself. Fixed by switching NVENC to constant-quality mode
  (`-rc vbr -cq 23`). Tunable via the new `--cq` flag.

### Changed

- CLI documentation (`_cli_documentation.md`/`.pdf`) refreshed for
  the new commands + flags.
- Minimal distribution at `Vemos_1.0/` refreshed with new modules +
  docs.

---

## [0.15.0] -- late May 2026

### Added

- **#41** -- source-crop super-resolution via Real-ESRGAN x2plus.
  `GpuRenderer` accepts an `sr_model` arg; per frame crops the
  source-pano bbox of the sampling grid, runs Real-ESRGAN on the
  crop (fp16 on CUDA), and samples the upscaled crop. CLI: `--sr`
  flag on `vemos render`. GUI: checkbox in the tracking-params
  dialog.
- **#18 main work** -- frame interpolation via RIFE 4.25.
  `vemos interpolate` reads a rendered broadcast and produces a
  higher-fps copy. RIFE 4.25 default (~250 ms/pair end-to-end at
  1440p, ~16 h for a 100-min match at 3x). FILM-Style opt-in via
  `--backend film` with a startup warning for the ~4x cost.

### Changed

- `vemos polish` -> `vemos interpolate` rename. The subcommand was
  always doing frame interp; renamed to be honest about scope.

### Investigated -- no fix this release

- cuDNN warmup attempted in `vemos/interpolate.py` (three discarded
  calls per dt). Did not fix the "RIFE shimmer" issue. Root cause
  diagnosed and fixed in v0.16 as NVENC bitrate starvation.

---

## Foundation (pre-0.15)

The core pipeline established before tracked-version snapshots
began. Not exhaustive -- the foundational pieces that everything
else depends on:

### Calibration + field geometry

- Dewarp model with plumb-line cost + radial k1 fit. Interactive
  slider tool in `vemos calibrate` for k1 / hfov / vfov / pitch0 /
  roll0.
- `markfield` with LSQ homography from 4 corners + N sideline
  points. MLE per-mark auto-balancing + `near_trust` confidence
  multiplier for the near sideline.

### Detection + tracking (Stage 1)

- YOLO11n detection on 3 single-row tiles (1920x1680 each,
  vfov ~62°). Tile coverage densely sampled from the field via the
  homography, not from 4 corners.
- Per-tile BoTSORT tracking, global ground-coordinate fusion +
  Kalman tracker. Native coast cap, stationary suppression,
  min-hits birth threshold.
- NVDEC hardware decode + GPU-resident tile remap (avoids
  CPU<->GPU frame copies).
- Hybrid foot/head ground projection by bbox aspect ratio for
  occluded vs visible feet.

### Classifier (Stage 2)

- Three-gate architecture:
  1. Per-track lifetime label: `player` / `sideline` / `foreign`.
     `foreign` requires median-off-field AND `frac_on < 0.4` AND
     `max_deep_run_s < min_deep_run_s`. Asymmetric static-spread
     threshold for far-sideline tracks (wobble + small angular
     size inflate spread).
  2. Per-frame Schmitt activation with asymmetric margins (strict
     inward at sidelines, lenient outward at endzones).
  3. Per-frame probation gate for newly-active-while-isolated
     tracks, using cumulative time in a `well_inside` zone.

### Camera-path planning + rendering

- `campath`: bounded sliding-window optimiser with critically-
  damped follow + 2.5s lookahead. Min/max azimuth bracketing so
  all framing-pool players are kept on screen. Fixed pitch from
  field-centre direction.
- `render`: cylindrical/Panini blend (`projection_blend = 0.3`).
  GPU `grid_sample` projection (`projection.py:GpuRenderer`) +
  NVENC encode (h264_nvenc with libx264 fallback).
- Optional `--debug-pano` mode draws the full panorama with the
  crop box (yellow) + field polygon (cyan) + tracked-player dots.

### GUI base

- PySide6 GUI with the Track tab (calibrate -> markfield -> params
  -> process flow), Concat tab (multi-clip concatenation with
  trim), persistent project state across sessions.
- Live progress monitor (`vemos/monitor.py`): step / progress bar /
  ETA / fps / per-tile counts. Backed by atomic `_progress.json`
  files written by `vemos/progress.py`.

### Pipeline orchestration

- `vemos pipeline` for cross-stage chunked processing
  (track + classify + campath + render in parallel chunks).
- Chunked processing makes the full match feasible despite
  per-stage latencies.
