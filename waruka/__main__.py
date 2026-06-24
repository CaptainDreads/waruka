# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Waruka CLI.

    python -m waruka calibrate VIDEO [--project P] [--time T]
    python -m waruka preview   VIDEO [--project P] [--time T] [--out O]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .config import ProjectConfig
from .preview import grab_frame, _label


def _load_model(video: str, project: str):
    pp = Path(project)
    if pp.exists():
        return ProjectConfig.load(pp).pano
    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return ProjectConfig.for_video(video, w, h).pano


def cmd_preview(args):
    model = _load_model(args.video, args.project)
    frame = grab_frame(args.video, args.time)
    yaws = [float(y) for y in args.yaws.split(",")]
    tiles = [
        _label(
            model.render(frame, y, 0.0, args.vfov, 1280, 720),
            f"yaw={y:g} vfov={args.vfov:g} k1={model.k1:+.3f} "
            f"pitch0={model.pitch0_deg:+.1f}",
        )
        for y in yaws
    ]
    out = np.vstack(tiles)
    cv2.imwrite(args.out, out)
    print(f"wrote {args.out}  ({out.shape[1]}x{out.shape[0]})")


def cmd_calibrate(args):
    from .calibrate import run_calibrator

    run_calibrator(args.video, args.project, args.time)


def cmd_markfield(args):
    from .markfield import run_markfield

    cw = None
    if args.corner_weights:
        cw = [float(s.strip()) for s in args.corner_weights.split(",")]
    run_markfield(args.video, args.project, args.time,
                  args.length, args.width,
                  cam_height_m=args.cam_height_m,
                  corner_weights=cw,
                  auto_balance_marks=args.auto_balance_marks,
                  near_trust=args.near_trust)


def main(argv=None):
    p = argparse.ArgumentParser(prog="waruka")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="interactive plumb-line calibration")
    c.add_argument("video")
    c.add_argument("--project", default="project.json")
    c.add_argument("--time", type=float, default=2.0)
    c.set_defaults(func=cmd_calibrate)

    m = sub.add_parser("markfield", help="mark field corners -> ground homography")
    m.add_argument("video")
    m.add_argument("--project", default="project.json")
    m.add_argument("--time", type=float, default=2.0)
    m.add_argument("--length", type=float, default=None,
                   help="field length incl. end zones, m (default WFDF 100)")
    m.add_argument("--width", type=float, default=None,
                   help="field width, m (default WFDF 37)")
    # Opt-in refinement controls (added 2026-05-29). Both default to None
    # = baseline behaviour. See waruka.ground.refine_homography docstring
    # for semantics and the dewarp-ceiling memory file for when to use
    # them (specifically: when corner clicks are known noisy/missing).
    m.add_argument("--cam-height-m", type=float, default=None,
                   dest="cam_height_m",
                   help="known camera mount height (m). Anchors the "
                        "decomposed camera Y in the LSQ. Use when corner "
                        "clicks are unreliable (no visible markers at the "
                        "back corners). OFF by default.")
    m.add_argument("--corner-weights", default=None,
                   dest="corner_weights",
                   help="comma-separated per-corner LSQ weights in "
                        "C0,C1,C2,C3 order. Use to downweight noisy "
                        "back corners, e.g. '0.5,0.5,2.0,2.0'. "
                        "OFF by default (uniform weight 2.0).")
    m.add_argument("--no-auto-balance", action="store_false",
                   dest="auto_balance_marks", default=None,
                   help="Disable MLE per-mark weighting (added 2026-05-30). "
                        "By default markfield weights each mark by the "
                        "inverse of its local click-error amplification: "
                        "extreme-longitude marks (far sideline ends, back "
                        "corners) get low weight; middle marks get high "
                        "weight. Pass this flag to use uniform weights "
                        "instead.")
    m.add_argument("--near-trust", type=float, default=None,
                   dest="near_trust",
                   help="Near-sideline trust multiplier (default 3.0). "
                        "Multiplies near-sideline LSQ weights so the "
                        "near sideline (closest to camera, visually "
                        "easiest to verify) dominates the fit. 1.0 = "
                        "MLE only, no extra boost.")
    m.set_defaults(func=cmd_markfield)

    t = sub.add_parser("track", help="detect+track players -> tracks.json")
    t.add_argument("--project", default="project.json")
    t.add_argument("--video", default=None)
    t.add_argument("--stride", type=int, default=3,
                   help="run detection on every Nth source frame; output is "
                        "densified to every frame by track interpolation")
    t.add_argument("--t0", type=float, default=0.0)
    t.add_argument("--t1", type=float, default=None)
    t.add_argument("--out", default="tracks.json")
    t.add_argument("--conf", type=float, default=0.50,
                   help="YOLO detection confidence threshold (production "
                        "default 0.50; lower to 0.20 to catch borderline "
                        "detections at the cost of more false positives)")
    t.add_argument("--iou", type=float, default=0.5,
                   help="YOLO NMS IoU; lower suppresses same-player double "
                        "boxes within a tile")
    t.add_argument("--fuse-lat-m", type=float, default=0.6,
                   help="per-frame fusion lateral (cross-bearing) tolerance, m")
    t.add_argument("--fuse-rad-m", type=float, default=2.5,
                   help="per-frame fusion radial (depth) tolerance, m")
    t.add_argument("--max-coast-s", type=float, default=0.3,
                   help="max time a track may coast past its last real "
                        "detection before output dots are suppressed")
    t.add_argument("--min-hits", type=int, default=5,
                   help="real-detection hits required before a track is "
                        "emitted (kills single-frame YOLO blips). "
                        "Production default 5; pairs with --conf 0.50.")
    t.add_argument("--stationary-pos-spread-m", type=float, default=0.5,
                   help="drop tracks whose median position spread is below "
                        "this over --stationary-min-duration-s (set 0 to "
                        "disable; kills fixed-object false positives)")
    t.add_argument("--stationary-min-duration-s", type=float, default=5.0,
                   help="duration before stationary-track filter applies")
    t.add_argument("--phantom-window-s", type=float, default=2.5,
                   help="per-frame phantom-segment filter window (sec). "
                        "Production default 2.5; set 0 to disable. At "
                        "each emit frame, look at real hits within +/- "
                        "this window: if all fit within --phantom-max-"
                        "tiles AND positionally tight "
                        "(--phantom-max-spread-m), the dot is suppressed. "
                        "Catches id-hijacked phantom segments the "
                        "whole-track stationary filter misses.")
    t.add_argument("--phantom-max-spread-m", type=float, default=0.1,
                   help="phantom-segment filter spread threshold, m "
                        "(production default 0.1)")
    t.add_argument("--phantom-max-tiles", type=int, default=8,
                   help="phantom-segment filter: max unique tiles per hit "
                        "in the window for the segment to count as phantom "
                        "(production default 8; 1 = single-tile only)")
    t.add_argument("--down-pad-deg", type=float, default=20.0,
                   help="how far below the closest field ground point the "
                        "NEAR-row tiles extend (degrees). Raise for mounts "
                        "where sideline-bench players sit close to the "
                        "camera (Z<0) and you want their feet inside a tile "
                        "rather than relying on head-projection")
    t.add_argument("--tile-h-near", type=int, default=960,
                   help="NEAR-row tile pixel height (FAR row stays at the "
                        "default 720). Bigger NEAR tile lets us extend "
                        "vfov for close-player coverage without losing "
                        "px/deg. Set 720 to make NEAR and FAR same size")
    t.add_argument("--rows", type=int, default=1, choices=[1, 2],
                   help="number of tile rows. 1 (production default since "
                        "v0.6) uses a single tall row covering the whole "
                        "field in one tile per yaw column -- avoids "
                        "NEAR/FAR cross-row fusion ambiguity (one player "
                        "== one detection). 2 splits NEAR+FAR (legacy).")
    t.add_argument("--tile-h-single", type=int, default=None,
                   help="single-row mode tile pixel height (only used with "
                        "--rows 1; default = tile_h_near + tile_h_far = 1680)")
    t.add_argument("--decoder", default="auto",
                   choices=["auto", "nvdec", "opencv", "cpu"],
                   help="video decoder: auto (NVDEC if available), nvdec "
                        "(require GPU decode), opencv/cpu (force CPU decode)")
    t.add_argument("--batched-predict", dest="batched_predict",
                   action="store_true", default=True,
                   help="single batched YOLO forward pass across all tiles "
                        "(default; ~1.9x faster than per-tile)")
    t.add_argument("--no-batched-predict", dest="batched_predict",
                   action="store_false",
                   help="revert to model.predict(list) -- N forward passes")
    t.set_defaults(func=lambda a: __import__(
        "waruka.perception", fromlist=["run_perception"]).run_perception(
        a.project, a.video, a.stride, a.t0, a.t1, a.out,
        conf=a.conf, iou=a.iou,
        fuse_lat_m=a.fuse_lat_m, fuse_rad_m=a.fuse_rad_m,
        max_coast_s=a.max_coast_s, min_hits=a.min_hits,
        stationary_pos_spread_m=a.stationary_pos_spread_m,
        stationary_min_duration_s=a.stationary_min_duration_s,
        phantom_window_s=a.phantom_window_s,
        phantom_max_spread_m=a.phantom_max_spread_m,
        phantom_max_tiles=a.phantom_max_tiles,
        down_pad_deg=a.down_pad_deg,
        tile_h_near=a.tile_h_near,
        tile_h_single=a.tile_h_single,
        rows=a.rows,
        decoder=("opencv" if a.decoder == "cpu" else a.decoder),
        batched_predict=a.batched_predict))

    dt = sub.add_parser("tiles",
                        help="dump detection tiles at a time, boxes+confidence")
    dt.add_argument("--project", default="project.json")
    dt.add_argument("--video", default=None)
    dt.add_argument("--time", type=float, default=2.0)
    dt.add_argument("--out", default="_tiles")
    dt.add_argument("--conf", type=float, default=0.20)
    dt.add_argument("--iou", type=float, default=0.5)
    dt.set_defaults(func=lambda a: __import__(
        "waruka.perception", fromlist=["dump_detection_tiles"]
    ).dump_detection_tiles(a.project, a.time, a.out, a.video,
                           conf=a.conf, iou=a.iou))

    dp = sub.add_parser("detectpano",
                        help="single-frame pano with YOLO boxes + tracked dots")
    dp.add_argument("--project", default="project.json")
    dp.add_argument("--video", default=None)
    dp.add_argument("--time", type=float, required=True)
    dp.add_argument("--tracks", default=None,
                    help="tracks.json to overlay feet dots + IDs")
    dp.add_argument("--out", default="_detpano.png")
    dp.add_argument("--conf", type=float, default=0.20)
    dp.add_argument("--iou", type=float, default=0.5)
    dp.set_defaults(func=lambda a: __import__(
        "waruka.perception", fromlist=["dump_detection_pano"]
    ).dump_detection_pano(a.project, a.time, a.tracks, a.out, a.video,
                          conf=a.conf, iou=a.iou))

    tp = sub.add_parser("trackpreview", help="overlay tracks on raw panorama")
    tp.add_argument("tracks")
    tp.add_argument("--project", default="project.json")
    tp.add_argument("--time", type=float, default=2.0)
    tp.add_argument("--out", default="track_overlay.png")
    tp.set_defaults(func=lambda a: __import__(
        "waruka.perception", fromlist=["render_track_overlay"]
    ).render_track_overlay(a.project, a.tracks, a.time, a.out))

    cl = sub.add_parser("classify",
                        help="on-field vs sideline -> players.json")
    cl.add_argument("tracks")
    cl.add_argument("--project", default="project.json")
    cl.add_argument("--out", default="players.json")
    cl.add_argument("--buffer", type=float, default=1.0)
    cl.add_argument("--overlay-times", default=None,
                    help="comma-separated seconds to dump class overlays")

    def _run_classify(a):
        from .classify import classify_tracks, render_class_overlay
        res = classify_tracks(a.tracks, a.project, a.buffer, out_path=a.out)
        if a.overlay_times:
            for ts in a.overlay_times.split(","):
                render_class_overlay(a.project, a.tracks, res, float(ts),
                                     f"_inspect/cls_{ts}s.png")
    cl.set_defaults(func=_run_classify)

    cp = sub.add_parser("campath", help="plan smoothed camera path")
    cp.add_argument("players")
    cp.add_argument("--project", default="project.json")
    cp.add_argument("--margin", type=float, default=None,
                    help="angular margin per side in degrees; default is "
                         "the preset's choice")
    cp.add_argument("--hfov-min", dest="hfov_min", type=float, default=None,
                    help="floor on output hfov in degrees; default is "
                         "the preset's choice")
    cp.add_argument("--view-mode", dest="view_mode", default="default",
                    choices=["default", "wide"],
                    help="named framing preset (default | wide)")
    cp.add_argument("--panini-preset", dest="panini_preset",
                    default="rectilinear",
                    choices=["rectilinear", "panini"],
                    help="Panini d preset: rectilinear (d=0.0, straightest) | "
                         "panini (d=1.0, classic stereographic)")
    cp.add_argument("--panini-d", dest="panini_d", type=float, default=None,
                    help="explicit Panini d; overrides --panini-preset")
    # Adaptive d controls (v0.12). Defaults pulled from project.json.
    cp.add_argument("--panini-d-adaptive", dest="panini_d_adaptive",
                    action="store_true", default=None,
                    help="enable per-frame adaptive Panini d (default: project setting)")
    cp.add_argument("--no-panini-d-adaptive", dest="panini_d_adaptive",
                    action="store_false",
                    help="disable per-frame adaptive Panini d (use static d)")
    cp.add_argument("--panini-d-cap", dest="panini_d_cap", type=float,
                    default=None,
                    help="cap on adaptive Panini d (default 1.5)")
    cp.add_argument("--panini-d-safety", dest="panini_d_safety_deg",
                    type=float, default=None,
                    help="safety margin (deg) from pano vfov edge "
                         "when solving for d (default 2.0)")
    cp.add_argument("--panini-d-black-tolerance",
                    dest="panini_d_black_tolerance_deg",
                    type=float, default=None,
                    help="deg of phi overflow tolerated before d engages "
                         "(default 0.0 = strict no-black; raising it keeps "
                         "d=0 over a wider HFOV range at the cost of a "
                         "black sliver at intermediate framings)")
    cp.add_argument("--panini-d-min-threshold",
                    dest="panini_d_min_threshold",
                    type=float, default=None,
                    help="snap smoothed d to 0 when below this (default 0.0 "
                         "= disabled; >0 snaps small smoothed-d values to 0)")
    cp.add_argument("--out", default="campath.json")
    cp.set_defaults(func=lambda a: __import__(
        "waruka.campath", fromlist=["plan_campath"]).plan_campath(
        a.players, a.project, margin_deg=a.margin, hfov_min=a.hfov_min,
        view_mode=a.view_mode,
        panini_preset=a.panini_preset, panini_d=a.panini_d,
        panini_d_adaptive=a.panini_d_adaptive,
        panini_d_cap=a.panini_d_cap,
        panini_d_safety_deg=a.panini_d_safety_deg,
        panini_d_black_tolerance_deg=a.panini_d_black_tolerance_deg,
        panini_d_min_threshold=a.panini_d_min_threshold,
        out_path=a.out))

    rd = sub.add_parser("render", help="render camera path to MP4")
    rd.add_argument("campath")
    rd.add_argument("--project", default="project.json")
    rd.add_argument("--video", default=None)
    rd.add_argument("--out", default="broadcast.mp4")
    rd.add_argument("--overlay-tracks", default=None,
                    help="labeled tracks json -> burn green/red foot dots")
    rd.add_argument("--t0", type=float, default=None)
    rd.add_argument("--t1", type=float, default=None)
    rd.add_argument("--debug-pano", action="store_true",
                    help="output the full panorama with the crop region drawn "
                         "as a yellow polygon (debug; ignores the actual crop)")
    rd.add_argument("--debug-pano-width", type=int, default=2560,
                    help="output width for --debug-pano (height auto from pano aspect)")
    rd.add_argument("--plain-dots", action="store_true",
                    help="draw all overlay dots one colour (no on/off-field "
                         "split); use with raw tracks.json to inspect tracking")
    rd.add_argument("--show-raw-yolo", action="store_true",
                    help="(debug-pano only) per frame, run YOLO across all "
                         "tiles and draw every person box on the pano with "
                         "tile-coded colour + conf label, plus the tile "
                         "outlines themselves. Lets you scrub the raw "
                         "detector output and see fusion/tracker effects")
    rd.add_argument("--det-conf", type=float, default=0.20,
                    help="conf threshold for --show-raw-yolo (default 0.20 "
                         "low, so all noise is visible)")
    rd.add_argument("--det-iou", type=float, default=0.5,
                    help="NMS IoU for --show-raw-yolo")
    rd.add_argument("--rows", type=int, default=1, choices=[1, 2],
                    help="(--show-raw-yolo) tile rows to draw -- must match "
                         "what `track` used or the tile outlines won't "
                         "match the dot positions")
    rd.add_argument("--down-pad-deg", type=float, default=20.0,
                    help="(--show-raw-yolo) NEAR-row down-pad in degrees")
    rd.add_argument("--tile-h-near", type=int, default=960,
                    help="(--show-raw-yolo) NEAR tile pixel height")
    rd.add_argument("--tile-h-single", type=int, default=None,
                    help="(--show-raw-yolo) single-row tile pixel height "
                         "(default = tile_h_near + 720 = 1680)")
    rd.add_argument("--pano-edge-fill",
                    dest="pano_edge_fill_mode", default=None,
                    choices=["zeros", "border", "blur"],
                    help="what the renderer does where the output rectangle "
                         "asks for rays beyond the pano vfov. zeros=black "
                         "(legacy); border=clamp to edge row; "
                         "blur=pre-pad with horizontally-blurred edge rows "
                         "(default in cfg)")
    rd.add_argument("--pano-edge-fill-blur-deg",
                    dest="pano_edge_fill_blur_deg",
                    type=float, default=None,
                    help="(blur mode) vfov extension per side in degrees")
    rd.add_argument("--pano-edge-fill-blur-sigma",
                    dest="pano_edge_fill_blur_sigma_px",
                    type=float, default=None,
                    help="(blur mode) horizontal gaussian-blur sigma in pixels")
    rd.add_argument("--sr", dest="sr_enable", action="store_true",
                    help="enable source-crop super-resolution (#41) via "
                         "Real-ESRGAN x2plus. Per frame, the GpuRenderer "
                         "crops the source pano to the grid bbox, upscales "
                         "x2, and samples from the SR'd crop. SR runs "
                         "every frame by default (no per-frame bypass) so "
                         "perceived sharpness is constant across zooms. "
                         "Adds ~150-1200 ms per frame depending on hfov. "
                         "Batch-only; not real-time.")
    rd.add_argument("--sr-min-upscale", dest="sr_min_upscale",
                    type=float, default=0.0,
                    help="minimum natural upscale ratio (output_dim / "
                         "source_crop_dim, max across axes) before SR "
                         "runs. Default 0.0: SR runs every frame. Set to "
                         ">1.0 to bypass SR at wider framings where the "
                         "natural upscale ratio drops below the threshold "
                         "-- saves compute but the transition can be "
                         "visually noticeable.")
    rd.set_defaults(func=lambda a: __import__(
        "waruka.render", fromlist=["render_path"]).render_path(
        a.campath, a.project, a.video, a.out, a.overlay_tracks,
        t0=a.t0, t1=a.t1, debug_pano=a.debug_pano,
        debug_pano_width=a.debug_pano_width, plain_dots=a.plain_dots,
        show_raw_yolo=a.show_raw_yolo, det_conf=a.det_conf,
        det_iou=a.det_iou, rows=a.rows, down_pad_deg=a.down_pad_deg,
        tile_h_near=a.tile_h_near, tile_h_single=a.tile_h_single,
        pano_edge_fill_mode=a.pano_edge_fill_mode,
        pano_edge_fill_blur_deg=a.pano_edge_fill_blur_deg,
        pano_edge_fill_blur_sigma_px=a.pano_edge_fill_blur_sigma_px,
        sr_enable=a.sr_enable, sr_min_upscale=a.sr_min_upscale))

    pl = sub.add_parser("pipeline",
                         help="cross-stage chunked pipeline: track + "
                              "classify + campath + render in parallel")
    pl.add_argument("--project", default="project.json")
    pl.add_argument("--video", default=None)
    pl.add_argument("--out", default="broadcast.mp4")
    pl.add_argument("--t0", type=float, default=0.0)
    pl.add_argument("--t1", type=float, default=None)
    pl.add_argument("--chunk", type=float, default=30.0,
                    help="chunk size in seconds (default 30)")
    pl.add_argument("--pre-overlap", dest="pre_overlap", type=float,
                    default=0.0,
                    help="pre-overlap seconds (default 0 because "
                         "cross-chunk state continuity replaces it)")
    pl.add_argument("--post-overlap", dest="post_overlap", type=float,
                    default=0.0,
                    help="post-overlap seconds (default 0 because "
                         "cross-chunk state continuity replaces it)")
    pl.add_argument("--no-cross-chunk-state", dest="cross_chunk_state",
                    action="store_false", default=True,
                    help="disable cross-chunk tracker + classifier state "
                         "continuity (legacy / A-B comparison)")
    pl.add_argument("--work-dir", default="_pipeline_chunks",
                    help="directory for per-chunk intermediates")
    pl.add_argument("--keep-chunks", action="store_true", default=False,
                    help="keep per-chunk intermediate files at end")
    pl.set_defaults(func=lambda a: __import__(
        "waruka.pipeline", fromlist=["run_pipeline"]).run_pipeline(
        a.project, video=a.video, t0=a.t0, t1=a.t1,
        chunk_seconds=a.chunk,
        pre_overlap_seconds=a.pre_overlap,
        post_overlap_seconds=a.post_overlap,
        cross_chunk_state=a.cross_chunk_state,
        out_path=a.out, work_dir=a.work_dir,
        cleanup_chunks=not a.keep_chunks))

    mon = sub.add_parser("monitor",
                          help="live tkinter progress monitor for waruka track")
    mon.add_argument("--path", default="_progress.json",
                     help="progress file written by run_perception")
    mon.set_defaults(func=lambda a: __import__(
        "waruka.monitor", fromlist=["run_monitor"]).run_monitor(a.path))

    gui = sub.add_parser("gui",
                         help="end-to-end GUI (PySide6) for the full pipeline")
    gui.set_defaults(func=lambda a: __import__(
        "waruka.gui", fromlist=["run_gui"]).run_gui())

    ip = sub.add_parser("interpolate",
                        help="post-render frame interpolation (#18)")
    ip.add_argument("input", help="input mp4 (e.g. broadcast.mp4)")
    ip.add_argument("--out", required=True, help="output mp4")
    ip.add_argument("--fps", type=float, default=60.0,
                    help="target output fps; must be an integer "
                         ">=2x multiple of source fps. Default 60 "
                         "(3x from a 20-fps render).")
    ip.add_argument("--backend", choices=["rife", "film"], default="rife",
                    help="interpolation backend. 'rife' (default, "
                         "recommended) uses RIFE 4.25 at ~250 ms/pair "
                         "end-to-end at 1440p. 'film' uses FILM-Style; "
                         "slightly cleaner on huge motion but ~4x slower "
                         "(see WARNING below).")
    ip.add_argument("--model", default=None,
                    help="override the model file/dir. For rife: path "
                         "to the train_log dir's parent. For film: path "
                         "to the .pt TorchScript file.")
    ip.add_argument("--fp32", action="store_true",
                    help="use float32 instead of float16 (slower; "
                         "debugging only)")
    ip.add_argument("--no-tile", dest="tile", action="store_false",
                    default=None,
                    help="FILM-only: disable tile-stitch (default auto-"
                         "tiles when input width >= 1920 to avoid the "
                         "1440p cuDNN kernel-cliff). RIFE never tiles.")
    ip.add_argument("--t0", type=float, default=None)
    ip.add_argument("--t1", type=float, default=None)
    ip.add_argument("--no-nvdec", dest="force_cv2", action="store_true",
                    help="force cv2 H264 decode for source frames instead "
                         "of NVDEC. Slower (~40%% wall-time penalty at "
                         "1440p) but uses cv2's YUV->BGR matrix; pick this "
                         "if you care about exact colour parity with v0.14 "
                         "output.")
    ip.add_argument("--no-pipeline", dest="use_pipeline",
                    action="store_false", default=True,
                    help="disable the three-stage (decoder / model / "
                         "encoder) thread pipeline and run the loop "
                         "synchronously. Slower; for debugging or comparing "
                         "against the v0.15.1 sequential baseline.")
    ip.add_argument("--no-batch-dts", dest="batch_dts",
                    action="store_false", default=True,
                    help="run one model call per timestep instead of "
                         "batching all timesteps of a pair into a single "
                         "call. The batched call is ~14%% faster on the "
                         "model itself (quick win #4 in #43). Disable for "
                         "debugging or to match v0.15.2 numerics exactly.")
    ip.add_argument("--cq", type=int, default=23,
                    help="NVENC constant-quality target (0-51, lower="
                         "better). Default 23 keeps interp sharpness close "
                         "to the source. Drop to 26-28 to halve file size "
                         "with marginal quality loss; 30 is the model-floor "
                         "point. Fixes the 'shimmer' / first-2s blur from "
                         "#42 vs the pre-#42 default NVENC rate-control.")
    def _run_interpolate(a):
        import json, sys as _sys
        from .interpolate import interpolate_video
        if a.backend == "film":
            print("============================================================",
                  file=_sys.stderr)
            print("WARNING: FILM backend is ~4x slower end-to-end than RIFE.",
                  file=_sys.stderr)
            print("  At 1440p, FILM tile-stitch ~1000 ms/pair vs RIFE ~250 ms.",
                  file=_sys.stderr)
            print("  100-min match: ~33h @ 2x, ~66h @ 3x. Only worth picking",
                  file=_sys.stderr)
            print("  for special cases where you want FILM's slightly softer",
                  file=_sys.stderr)
            print("  in-betweens on very large motion. Default 'rife' is",
                  file=_sys.stderr)
            print("  recommended for routine batch use.",
                  file=_sys.stderr)
            print("============================================================",
                  file=_sys.stderr, flush=True)
        report = interpolate_video(
            input_path=a.input, output_path=a.out, target_fps=a.fps,
            backend=a.backend, model_path=a.model,
            fp16=not a.fp32, tile=a.tile, t0=a.t0, t1=a.t1,
            force_cv2=a.force_cv2, use_pipeline=a.use_pipeline,
            batch_dts=a.batch_dts, cq=a.cq)
        print(json.dumps(report, indent=2))
    ip.set_defaults(func=_run_interpolate)

    up = sub.add_parser("upscale",
                         help="2x super-resolution on a video (Real-ESRGAN x2plus)")
    up.add_argument("input", help="input video (any codec OpenCV/NVDEC reads)")
    up.add_argument("--out", required=True, help="output mp4")
    up.add_argument("--weights", default=None,
                    help="override Real-ESRGAN weights path. Default is "
                         "third_party/realesrgan/weights/RealESRGAN_x2plus.pth")
    up.add_argument("--fp32", action="store_true",
                    help="use float32 instead of float16 (slower; debugging)")
    up.add_argument("--t0", type=float, default=None,
                    help="start time in seconds (default 0)")
    up.add_argument("--t1", type=float, default=None,
                    help="end time in seconds (default = end of clip). "
                         "At ~1.2 s/frame on a 2080 Ti the full 100-min "
                         "match is ~40 hours, so prefer short windows.")
    up.add_argument("--no-nvdec", dest="force_cv2", action="store_true",
                    help="force cv2 H264 decode for source frames instead "
                         "of NVDEC. SR dominates wall time so the source-"
                         "decode choice is mostly irrelevant.")
    up.add_argument("--cq", type=int, default=23,
                    help="NVENC constant-quality target (0-51, lower=better). "
                         "Default 23 keeps output close to source quality.")
    def _run_upscale(a):
        import json
        from .upscale import upscale_video
        report = upscale_video(
            input_path=a.input, output_path=a.out,
            weights_path=a.weights, fp16=not a.fp32,
            t0=a.t0, t1=a.t1, force_cv2=a.force_cv2, cq=a.cq)
        print(json.dumps(report, indent=2))
    up.set_defaults(func=_run_upscale)

    v = sub.add_parser("preview", help="headless dewarp preview to an image")
    v.add_argument("video")
    v.add_argument("--project", default="project.json")
    v.add_argument("--time", type=float, default=2.0)
    v.add_argument("--yaws", default="-45,0,45")
    v.add_argument("--vfov", type=float, default=75.0)
    v.add_argument("--out", default="preview.png")
    v.set_defaults(func=cmd_preview)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
