# Waruka 1.0

Veo-style automated broadcast camera for ultimate frisbee.

Takes a single panoramic video of a match (Reolink Duo 2 stitched
output, 4608×1728 @ 20fps) and produces a 2560×1440 (1440p) MP4 of a
virtual broadcast camera that pans, zooms and tracks the play
automatically. Built for tactics review.

Optional post-processing:
* **Source-crop super-resolution** (Real-ESRGAN x2plus) for sharper
  detail at tight zooms, applied during render.
* **Standalone 2x upscale** (Real-ESRGAN) on any existing video --
  `waruka upscale`, new in v0.16.
* **Frame interpolation to 40 / 60 / 80 fps** (RIFE 4.25 default;
  FILM-Style opt-in for special renders), applied after render.
* **Overnight batch queue** (Queue tab in the GUI, new in v0.16) for
  processing many matches end-to-end while you sleep.

All are off by default; pick them in the GUI's Tracking parameters
dialog or via `waruka render --sr`, `waruka interpolate`, and
`waruka upscale` from the CLI.

## Contents

```
waruka_1.0/
├── README.md                                  this file
├── _cli_documentation.md                      full CLI reference (text)
├── _cli_documentation.pdf                     full CLI reference (PDF)
├── _gui_documentation.md                      GUI walkthrough (text)
├── _gui_documentation.pdf                     GUI walkthrough (PDF)
├── _doc_screenshots/                          tab + dialog screenshots
├── yolo11n.pt                                 YOLO11n weights (auto-loaded)
├── waruka/                                     Python package source
│   └── (gui, calibrate, markfield, track, classify, campath, render,
│        interpolate, upscale, sr, jobqueue, ...)
├── third_party/                               vendored ML model code + weights
│   ├── film/      film_net_fp16.pt (FILM-Style for interpolation, opt-in)
│   ├── rife/      train_log/flownet.pkl (RIFE 4.25, default interp backend)
│   └── realesrgan/ weights/RealESRGAN_x2plus.pth + arch (source-crop SR)
├── input_video_short_1.mp4                    sample clip 1 (62 MB)
├── input_video_short_2.mp4                    sample clip 2 (98 MB)
├── input_video_short_3.mp4                    sample clip 3 (142 MB)
├── input_video_short_4.mp4                    sample clip 4 (36 MB)
├── project.json                               clip 2 calibration + marks
├── project_input_video_short_1.json           clip 1 calibration + marks
├── project_input_video_short_3.json           clip 3 calibration + marks
└── project_input_video_short_4.json           clip 4 calibration + marks
```

The model **weights** for the optional SR / interpolation stages are
not committed to this repo (they exceed GitHub's file-size limit). The
base `track → classify → campath → render` pipeline runs without them.
To enable interpolation or super-resolution, fetch the weights as
described in [`third_party/README.md`](third_party/README.md) — either
the `waruka-weights-1.0.0.zip` release bundle or the upstream sources.

The four `input_video_short_*.mp4` files are 120-second test clips
representing the four kinds of footage Waruka has been validated
against (different mount geometries, jerseys, lighting, sideline
character).

## Requirements

- **Python 3.13** (or 3.11+ with adjustment to `pyproject.toml`).
- **NVIDIA GPU** with CUDA 12.x runtime (tested on RTX 2080 Ti). CPU
  fallback exists for decode (`--decoder opencv`) but YOLO inference
  + render rely on a CUDA-capable GPU for the documented throughput.
- ~4 GB free disk per match (intermediate JSONs + output MP4).

### Python deps

```
pip install ultralytics torch opencv-python imageio-ffmpeg \
            numpy scipy PyNvVideoCodec nvidia-cuda-runtime-cu12 \
            PySide6
```

`PyNvVideoCodec` + `nvidia-cuda-runtime-cu12` are only needed if you
want NVDEC GPU video decode (the default; significantly faster than
the OpenCV CPU fallback for full-match runs). `PySide6` is required
for the GUI; the CLI works without it.

## Quick start — GUI (recommended)

```bash
python -m waruka gui
```

PySide6 tabbed shell with **four** tabs as of v0.16:

- **Track tab** — drives calibrate → markfield → params → process for
  a single source video. Drag-drop a video onto the source-video box
  (or click `Open video...`). Calibrate / markfield auto-skipped when
  `project.json` already has them. Artefacts at
  `<source_dir>/waruka_tracking/<basename>/`; final output at
  `<source_dir>/<basename>_tracked.mp4`.
- **Concat tab** — multi-clip concatenation + trim for matches
  recorded as 5-min Reolink chunks. Drag-drop file list, audio/codec
  consistency checks, ffmpeg `-c copy` concat with live progress,
  cv2-based scrubber for trim selection, date-prefilled output name,
  auto-handover to the Track tab on Save.
- **Post-process tab** (v0.16) — run frame interpolation and/or
  2x super-resolution against any existing video. When both are
  ticked the source is upscaled first then interpolated (faster
  per-frame math). Cancel-safe; cleans up temp intermediates.
- **Queue tab** (v0.16) — overnight batch processor. Set up many
  games end-to-end via an Add-job dialog that mirrors the Track tab
  (calibrate + markfield inline, "reuse project from previous job"
  for back-to-back same-mount matches, full ParamsDialog for stride
  / mode / SR / audio handling). Persistent across crashes; pause /
  resume at stage boundaries; retry from failed stage; per-job
  archived logs.

See `_gui_documentation.md` (or `.pdf`) for a screenshot walkthrough
of each tab.

## Quick start — CLI (run end-to-end on one of the sample clips)

```bash
N=2  # which sample clip to process
PROJ=$([ "$N" = "2" ] && echo "project.json" || echo "project_input_video_short_${N}.json")

# 1. Track (~3 min on RTX 2080 Ti)
# Production defaults are baked in -- no need to pass --stride/--conf/
# --min-hits/--phantom-*/--rows explicitly.
python -m waruka track --project ${PROJ} \
    --t0 0 --t1 120 --out tracks_${N}.json

# 2. Classify (~10 sec) -- writes players_${N}.json (framing pool)
#                         and players_${N}_labeled.json (overlay)
python -m waruka classify tracks_${N}.json \
    --project ${PROJ} --out players_${N}.json

# 3. Camera path (~10 sec)
python -m waruka campath players_${N}.json \
    --project ${PROJ} --out campath_${N}.json

# 4. Render the broadcast (~90 sec)
python -m waruka render campath_${N}.json --project ${PROJ} \
    --out broadcast_${N}.mp4
```

Total: ~5 minutes for a 120-second clip.

Output: `broadcast_${N}.mp4` (2560×1440 @ 20 fps, ~28 MB).

### For long matches (5+ min): chunked pipeline

The four sequential commands can be replaced by a single chunked
pipeline run that processes track/classify/campath/render as
concurrent worker threads, saving ~33-50% wall time:

```bash
python -m waruka pipeline --project ${PROJ} \
    --t0 0 --t1 6000 --chunk 30 --out broadcast.mp4
```

For short clips (< 2 min) prefer the sequential chain above -- the
chunked pipeline has a small known divergence in the first ~20 s that
matters less when it's a tiny fraction of a long match.

## Run on your own video

Two interactive setup steps are needed for a new camera mount:

```bash
# 1. Dewarp calibration (one-time per camera setup)
python -m waruka calibrate your_video.mp4 --project your_project.json

# 2. Field marking (one-time per recording session)
python -m waruka markfield your_video.mp4 --project your_project.json
```

Both tools open OpenCV windows with live previews. Keybindings and
mouse interactions are documented in `_cli_documentation.md`
(sections 1 and 2).

Once you have a `your_project.json` saved, run the same `track →
classify → campath → render` chain pointing at your video and project
file.

## Pipeline outputs

Every stage writes a JSON file you can inspect:

- `tracks_N.json` — raw per-frame detections, tracked across frames
  in metric ground coordinates.
- `players_N.json` — the **framing pool**: stable-active
  player-labelled tracks the camera should follow.
- `players_N_labeled.json` — every track every frame with a label
  (`player` / `probation` / `sideline` / `foreign`). Used for the
  debug-pano overlay only.
- `campath_N.json` — smoothed per-frame `yaw / pitch / hfov` for the
  virtual broadcast camera, plus projection mode and Panini `d`
  parameter.
- `broadcast_N.mp4` — final virtual broadcast video.

## Debug rendering

To see what the classifier and tracker are doing, render in
debug-pano mode with the labelled overlay:

```bash
python -m waruka render campath_${N}.json --project ${PROJ} \
    --out broadcast_${N}_debug.mp4 \
    --debug-pano --overlay-tracks players_${N}_labeled.json
```

Dot colours:
- **Green** — stable active (in the framing pool, drives the camera).
- **Yellow** — probation (active but isolated/unconfirmed — waiting
  to be promoted to stable).
- **Red** — sideline, foreign, or player currently off-field.

The yellow polygon is the virtual camera crop. The cyan polygon is
the field perimeter.

## CLI reference

Full reference for every command and parameter in
`_cli_documentation.md` (or `.pdf` for print).

## Project file (`project.json`)

Persisted between runs. Holds:

- Dewarp parameters (`pano`: `k1`, `hfov_deg`, etc.)
- Field corner + sideline marks
- Homography (ground ↔ ray)
- Projection look (`projection_blend`, `panini_d`)
- Output size (`out_w`, `out_h`, default 2560×1440)

The shipped sample project files (`project.json`,
`project_input_video_short_*.json`) are pre-calibrated and marked
against the four sample clips so you can run the pipeline immediately
without going through `calibrate` / `markfield`.

## Version

This snapshot is `waruka_1.0`, based on the `Waruka_v0.16`
development snapshot. The development tree (with diagnostic
visualisations, A/B comparison renders, classifier-iteration history,
and the full snapshot lineage v0.1–v0.16) lives outside this
distribution.

### What changed since the v0.15 distribution

- **Interpolation perf (#43)** — NVDEC source decode + GPU-resident
  cached `a_t` + three-stage thread pipeline + batched dts. Loop wall
  19.9 s → 10.9 s on a 5-s 1440p bench; 100-min match estimate
  6.7 h → 3.6 h. New flags: `--no-nvdec`, `--no-pipeline`,
  `--no-batch-dts`, `--cq`.
- **#42 'RIFE shimmer' fix** — diagnosed as NVENC bitrate starvation
  (default rate-control gave 60 fps interp output only 40% of plain's
  per-frame bit budget). Fixed by switching to constant-quality VBR
  (`-rc vbr -cq 23`). Sharpness on the perftest clip went from 38%
  below source to within 7%.
- **`waruka upscale`** — new standalone 2x super-resolution command.
- **Queue tab + Post-process tab** in the GUI (see above).
- **`waruka pipeline` chunked render mode** is still available but
  the default flow in both Track tab and Queue tab is sequential
  (`track → classify → campath → render`) to avoid the known
  chunk-0 ~5° framing residual (#21).
