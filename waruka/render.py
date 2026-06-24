# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Render the smoothed camera path to a 2560x1440 MP4.

The path is sampled at the perception stride; here we interpolate it to every
source frame and do a single resample from the panorama straight to the
output (no intermediate dewarp), then encode H.264 via imageio-ffmpeg.
No audio in phase 1.

If debug_overlay is on, per-person foot dots are burned in: green = active
on-field player, red = detected person classified off-field/not playing.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from .config import ProjectConfig
from .ground import GroundModel
from .progress import Progress


def _crop_outline_on_pano(pano, yaw, pitch, vf, vw, vh, projection, blend,
                           samples_per_side=128):
    """Crop view perimeter as pano src pixel coords (perimeter-only; cheap)."""
    return pano.view_outline(yaw, pitch, vf, vw, vh,
                             samples_per_side=samples_per_side,
                             projection=projection, blend=blend)


def _tile_rect_to_pano(T, x1, y1, x2, y2, n=12):
    """Sample the 4 sides of a tile-space bbox at N points, map each to pano
    source coords via the tile's precomputed map. Returns (4N, 2) polyline."""
    H, W = T.map_x.shape
    x1c = int(np.clip(x1, 0, W - 1)); x2c = int(np.clip(x2, 0, W - 1))
    y1c = int(np.clip(y1, 0, H - 1)); y2c = int(np.clip(y2, 0, H - 1))
    top_x = np.linspace(x1c, x2c, n).astype(int); top_y = np.full(n, y1c)
    rgt_y = np.linspace(y1c, y2c, n).astype(int); rgt_x = np.full(n, x2c)
    bot_x = np.linspace(x2c, x1c, n).astype(int); bot_y = np.full(n, y2c)
    lft_y = np.linspace(y2c, y1c, n).astype(int); lft_x = np.full(n, x1c)
    xs = np.concatenate([top_x, rgt_x, bot_x, lft_x])
    ys = np.concatenate([top_y, rgt_y, bot_y, lft_y])
    return np.column_stack([T.map_x[ys, xs], T.map_y[ys, xs]])


class _RawYoloOverlay:
    """Per-frame raw-detection overlay for the debug-pano video.

    Precomputes the tile outlines on the pano (constant), loads YOLO and the
    GPU tile remapper once, and on each frame: remaps the source frame into
    the tiles, runs YOLO across all tiles in one batched predict, projects
    every person box back to pano src coords as a curved polygon (curved
    because the tile is a virtual perspective view; straight box edges in
    tile space become curves in the equirectangular source), and draws it
    in the tile's colour with a confidence label. Tile outlines are drawn
    in the same per-tile colour so you can see which tile produced which
    box.

    Use this with `--debug-pano --show-raw-yolo` on the render CLI to scrub
    through the raw detector behaviour over time (intermittent FPs, tile-
    seam misses, conf distribution).
    """

    def __init__(self, pano, cfg, det_conf: float = 0.20,
                 det_iou: float = 0.5, det_model: str = "yolo11n.pt",
                 rows: int = 1, down_pad_deg: float = 20.0,
                 tile_h_near: int = 960, tile_h_single: int | None = None):
        from .perception import build_tiles, _GpuTileRemapper
        from ultralytics import YOLO
        gm = GroundModel(cfg.homography, cfg.field_length_m, cfg.field_width_m)
        self.tiles = build_tiles(pano, gm, rows=rows,
                                   down_pad_deg=down_pad_deg,
                                   tile_h_near=tile_h_near,
                                   tile_h_single=tile_h_single)
        self.model = YOLO(det_model)
        self.imgsz = max(self.tiles[0].out_w, self.tiles[0].out_h)
        self.conf = det_conf
        self.iou = det_iou
        try:
            import torch
            self.gpu_remap = (_GpuTileRemapper(pano.src_w, pano.src_h, self.tiles)
                              if torch.cuda.is_available() else None)
        except Exception:
            self.gpu_remap = None
        # Precompute tile outlines on pano src coords + per-tile colour.
        self.tile_perims = []
        self.tile_colors = []
        for ti, T in enumerate(self.tiles):
            H, W = T.map_x.shape
            self.tile_perims.append(_tile_rect_to_pano(T, 0, 0, W - 1, H - 1,
                                                       n=64))
            hue = int(ti * 180 / max(1, len(self.tiles))) % 180
            bgr = cv2.cvtColor(np.array([[[hue, 220, 230]]], np.uint8),
                                cv2.COLOR_HSV2BGR)[0, 0]
            self.tile_colors.append(tuple(int(c) for c in bgr))

    def draw(self, view: np.ndarray, frame: np.ndarray,
             sx_scale: float, sy_scale: float):
        # Tile outlines (thin, faint to stay out of the way of boxes).
        for perim, col in zip(self.tile_perims, self.tile_colors):
            poly = np.column_stack([perim[:, 0] * sx_scale,
                                    perim[:, 1] * sy_scale]).astype(np.int32)
            cv2.polylines(view, [poly], True, col, 1, cv2.LINE_AA)
        # Remap into tiles and detect.
        if self.gpu_remap is not None:
            imgs = self.gpu_remap.remap(frame)
        else:
            imgs = [cv2.remap(frame, T.map_x, T.map_y, cv2.INTER_LINEAR)
                    for T in self.tiles]
        res = self.model.predict(imgs, conf=self.conf, iou=self.iou,
                                  imgsz=self.imgsz, half=True, verbose=False)
        edge_px = 6
        for ti, r in enumerate(res):
            T = self.tiles[ti]
            col = self.tile_colors[ti]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy()
            for k in range(len(xyxy)):
                if int(cl[k]) != 0:                  # persons only
                    continue
                x1, y1, x2, y2 = [float(v) for v in xyxy[k]]
                top_cut = y1 <= edge_px
                bot_cut = y2 >= T.out_h - edge_px
                tag = "TT" if (top_cut and bot_cut) else (
                    "T" if top_cut else ("B" if bot_cut else ""))
                perim = _tile_rect_to_pano(T, x1, y1, x2, y2, n=12)
                poly = np.column_stack([perim[:, 0] * sx_scale,
                                        perim[:, 1] * sy_scale]
                                       ).astype(np.int32)
                cv2.polylines(view, [poly], True, col, 1, cv2.LINE_AA)
                # Conf label at top-left of the projected box.
                lx = int(T.map_x[int(np.clip(y1, 0, T.out_h - 1)),
                                  int(np.clip(x1, 0, T.out_w - 1))] * sx_scale)
                ly = int(T.map_y[int(np.clip(y1, 0, T.out_h - 1)),
                                  int(np.clip(x1, 0, T.out_w - 1))] * sy_scale)
                txt = f"t{ti}:{cf[k]:.2f}{tag}"
                cv2.putText(view, txt, (lx, max(10, ly - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2,
                            cv2.LINE_AA)
                cv2.putText(view, txt, (lx, max(10, ly - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)


def render_path(campath_json: str, project: str = "project.json",
                video: str | None = None, out_path: str = "broadcast.mp4",
                overlay_tracks: str | None = None, crf: int = 18,
                t0: float | None = None, t1: float | None = None,
                debug_pano: bool = False, debug_pano_width: int = 2560,
                plain_dots: bool = False, show_raw_yolo: bool = False,
                det_conf: float = 0.20, det_iou: float = 0.5,
                rows: int = 1, down_pad_deg: float = 20.0,
                tile_h_near: int = 960, tile_h_single: int | None = None,
                # Edge-fill overrides (added v0.12). When None, the
                # project config value is used. Set explicitly to
                # A/B test edge-fill modes without editing project.json.
                pano_edge_fill_mode: str | None = None,
                pano_edge_fill_blur_deg: float | None = None,
                pano_edge_fill_blur_sigma_px: float | None = None,
                # Source-crop super-resolution (#41). When True, the
                # GpuRenderer crops the source pano to the per-frame grid
                # bbox, runs Real-ESRGAN x2 on it, and samples from the
                # upscaled crop. Bypassed per-frame when the source crop
                # is already at least sr_min_upscale times the output.
                sr_enable: bool = False,
                sr_min_upscale: float = 0.0):
    """Render the camera path to MP4.

    By default, renders the cropped/zoomed virtual broadcast view. With
    `debug_pano=True`, instead outputs the full panorama (downscaled to
    `debug_pano_width` keeping pano aspect) with a yellow polygon drawn
    where the crop would be, plus the overlay dots projected to pano
    coordinates. This is for inspecting framing + detection quality
    without losing context to the crop.
    """
    import time as _time
    import imageio_ffmpeg

    prog = Progress("render", out_path=out_path)
    prog.set_step("load_inputs", detail=f"campath {campath_json}")
    cp = json.load(open(campath_json))
    cfg = ProjectConfig.load(project)
    pano = cfg.pano
    src = video or cp["video"]
    prog.state["source"] = src
    fps = cp["fps"]
    ow, oh = cp["out_w"], cp["out_h"]   # virtual crop size from campath
    projection = cp.get("projection", "rectilinear")
    # The "blend" slot in projection.py carries different things for
    # different projection modes: it's the rect/cyl interpolation factor
    # for "cylindrical" (cfg.projection_blend), the Panini-General `d`
    # parameter for "panini". Per-run panini_d may also be written into
    # the campath JSON (overrides cfg.panini_d) so users can A/B test
    # without editing the project file.
    if projection == "panini":
        blend = cp.get("panini_d", getattr(cfg, "panini_d", 0.0))
    else:
        blend = cfg.projection_blend

    if debug_pano:
        out_w = debug_pano_width if debug_pano_width % 2 == 0 else debug_pano_width + 1
        out_h = int(round(out_w * pano.src_h / pano.src_w))
        if out_h % 2:
            out_h += 1
    else:
        out_w, out_h = ow, oh

    # Both "cylindrical" and "panini" branches interpret fov as HFOV;
    # only the legacy "rectilinear" branch wants VFOV.
    fov_key = "vfov" if projection == "rectilinear" else "hfov"
    sf = np.array([p["frame"] for p in cp["path"]])
    yaw = np.array([p["yaw"] for p in cp["path"]])
    pitch = np.array([p["pitch"] for p in cp["path"]])
    fov = np.array([p[fov_key] for p in cp["path"]])
    # Per-frame Panini d, if the campath has it (v0.12+ adaptive d).
    # Older campaths don't have this key; fall back to the static blend
    # for every frame.
    if projection == "panini" and "d" in cp["path"][0]:
        d_arr = np.array([p["d"] for p in cp["path"]])
        adaptive_d = True
    else:
        d_arr = None
        adaptive_d = False
    f0, f1 = int(sf[0]), int(sf[-1])
    if t0 is not None:
        f0 = max(f0, int(t0 * fps))
    if t1 is not None:
        f1 = min(f1, int(t1 * fps))

    overlay = None
    if overlay_tracks and (cfg.debug_overlay or overlay_tracks):
        od = json.load(open(overlay_tracks))
        gm = GroundModel(cfg.homography, cfg.field_length_m, cfg.field_width_m)
        ofr = np.array([fr["frame"] for fr in od["frames"]])
        overlay = (gm, ofr, {fr["frame"]: fr["players"]
                             for fr in od["frames"]})

    raw_yolo = None
    if show_raw_yolo:
        if not debug_pano:
            print("warning: --show-raw-yolo only meaningful with --debug-pano; "
                  "ignoring", flush=True)
        else:
            raw_yolo = _RawYoloOverlay(pano, cfg, det_conf=det_conf,
                                        det_iou=det_iou, rows=rows,
                                        down_pad_deg=down_pad_deg,
                                        tile_h_near=tile_h_near,
                                        tile_h_single=tile_h_single)
            print(f"raw-yolo overlay: {len(raw_yolo.tiles)} tiles "
                  f"(rows={rows}), conf>={det_conf} iou={det_iou}",
                  flush=True)

    # GPU renderer for the crop path (the CPU pano.render is ~1 s/frame).
    # Debug-pano uses cheap cv2.resize so it doesn't need the GPU path.
    edge_mode = (pano_edge_fill_mode
                 if pano_edge_fill_mode is not None
                 else getattr(cfg, "pano_edge_fill_mode", "zeros"))
    # edge_blur_deg: None -> auto-compute from calibration + campath.
    # CLI override beats project config; both None -> auto. Any
    # explicit numeric value disables auto.
    if pano_edge_fill_blur_deg is not None:
        edge_blur_deg = float(pano_edge_fill_blur_deg)
    else:
        cfg_val = getattr(cfg, "pano_edge_fill_blur_deg", None)
        edge_blur_deg = (float(cfg_val) if cfg_val is not None else None)
    edge_blur_sigma = (pano_edge_fill_blur_sigma_px
                      if pano_edge_fill_blur_sigma_px is not None
                      else getattr(cfg, "pano_edge_fill_blur_sigma_px", 40.0))
    edge_blur_boundary_sigma = getattr(
        cfg, "pano_edge_fill_blur_boundary_sigma_px", 8.0)
    edge_blur_fade_deg = getattr(cfg, "pano_edge_fill_blur_fade_deg", 2.0)
    from .projection import (make_renderer, pad_source_for_blur,
                              compute_required_pad_deg)
    # Auto-compute pad_deg when configured to None (the default). Uses
    # the calibration + the per-frame hfov/pitch/d from the campath so
    # the blur extension is sized for the actual demand of this clip's
    # mount + framing. Border-clamp still handles any residual.
    if edge_mode == "blur" and edge_blur_deg is None:
        d_per_frame = (np.array([p.get("d", cp.get("panini_d", 0.0))
                                  for p in cp["path"]])
                       if "d" in cp["path"][0]
                       else np.full(len(cp["path"]),
                                     cp.get("panini_d", 0.0)))
        edge_blur_deg = compute_required_pad_deg(
            pano,
            hfovs=[p["hfov"] for p in cp["path"]],
            pitches=[p["pitch"] for p in cp["path"]],
            ds=d_per_frame,
            aspect=oh / ow,
            safety_deg=2.0)
        print(f"auto pano_edge_fill_blur_deg = {edge_blur_deg:.1f} "
              f"(vfov_pano={pano.vfov_deg:.1f}, "
              f"pitch0={pano.pitch0_deg:.1f})", flush=True)
    elif edge_blur_deg is None:
        edge_blur_deg = 10.0  # fallback for non-blur modes (unused)
    # Optional source-crop SR (#41). Built once and threaded into the
    # renderer; bypass logic per-frame lives inside GpuRenderer.
    sr_model = None
    if sr_enable and not debug_pano:
        from .sr import make_sr_model
        sr_model = make_sr_model(enable=True)
        if sr_model is None:
            print("[render] sr_enable was True but the SR model couldn't "
                  "load; continuing without SR.", flush=True)
        else:
            print(f"[render] source-crop SR active "
                  f"(min upscale ratio {sr_min_upscale:g}; sr off when "
                  f"crop is already that close to output size)",
                  flush=True)
    gpu = None if debug_pano else make_renderer(
        pano, out_w, out_h, projection, blend,
        edge_fill_mode=edge_mode,
        edge_fill_blur_deg=edge_blur_deg,
        edge_fill_blur_sigma_px=edge_blur_sigma,
        edge_fill_blur_boundary_sigma_px=edge_blur_boundary_sigma,
        edge_fill_blur_fade_deg=edge_blur_fade_deg,
        sr_model=sr_model, sr_min_upscale=sr_min_upscale)
    # For "blur" mode the renderer was initialised against a virtual
    # taller pano; per-frame we pad the source before render().
    blur_pad_rows = (int(round(edge_blur_deg * pano.src_h / pano.vfov_deg))
                     if edge_mode == "blur" else 0)
    blur_fade_rows = (int(round(edge_blur_fade_deg * pano.src_h
                                 / pano.vfov_deg))
                      if edge_mode == "blur" else 0)
    prog.update(detail=f"gpu_render={'yes' if gpu else 'no(cpu)'} "
                       f"edge_fill={edge_mode}"
                       f"{f' pad_deg={edge_blur_deg:.1f}' if edge_mode == 'blur' else ''}")

    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    def _open_writer(codec, params):
        w = imageio_ffmpeg.write_frames(
            out_path, (out_w, out_h), pix_fmt_in="bgr24", fps=fps,
            codec=codec, quality=None, macro_block_size=1,
            output_params=params)
        w.send(None)
        return w
    # Prefer GPU encode (NVENC); fall back to libx264 veryfast.
    try:
        writer = _open_writer("h264_nvenc",
                              ["-preset", "p4", "-pix_fmt", "yuv420p"])
        encoder = "h264_nvenc"
    except Exception:
        writer = _open_writer("libx264",
                              ["-crf", str(crf), "-preset", "veryfast",
                               "-pix_fmt", "yuv420p"])
        encoder = "libx264-veryfast"

    sx_scale = out_w / pano.src_w if debug_pano else 1.0
    sy_scale = out_h / pano.src_h if debug_pano else 1.0

    n = f1 - f0 + 1
    mode = "debug-pano" if debug_pano else "crop"
    prog.set_step("render_frames", progress=0.0,
                  detail=f"{mode} {out_w}x{out_h}, {n} frames",
                  f_start=f0, f_end=f1, current_frame=f0,
                  fps_observed=0.0, eta_s=None)
    render_started = _time.time()
    # Pre-fetch frames in a background thread so cap.read() (CPU
    # decode) overlaps with gpu.render() / writer.send() (added v0.12
    # for #20). queue_size=4 buffers ~120 MB on a 2560x1440 input;
    # increase for slower main loops, decrease if memory tight.
    import queue as _queue, threading as _threading
    _frame_q: _queue.Queue = _queue.Queue(maxsize=4)
    _stop_evt = _threading.Event()
    _frame_sentinel = object()
    def _prefetch():
        try:
            for _ in range(n):
                if _stop_evt.is_set():
                    break
                ok_, frame_ = cap.read()
                if not ok_:
                    break
                _frame_q.put(frame_)
        finally:
            _frame_q.put(_frame_sentinel)
    _prefetch_thread = _threading.Thread(target=_prefetch, daemon=True)
    _prefetch_thread.start()
    for k, f in enumerate(range(f0, f1 + 1)):
        frame = _frame_q.get()
        if frame is _frame_sentinel:
            break
        y = float(np.interp(f, sf, yaw))
        p = float(np.interp(f, sf, pitch))
        vf = float(np.interp(f, sf, fov))
        # Per-frame blend for adaptive Panini d. Falls back to the
        # static `blend` when the campath doesn't carry per-frame d.
        b_f = (float(np.interp(f, sf, d_arr)) if adaptive_d else blend)

        if debug_pano:
            view = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            # thin cyan field-boundary outline (helps tell whether a marker
            # is inside the polygon or just past the sideline -- distinguishes
            # classifier issues from homography issues)
            if overlay is not None:
                gm_, _, _ = overlay
                fb = gm_.boundary_xz(1.0)
                fbx, fby = pano.directions_to_src(gm_.ray_from_ground(fb))
                fb_px = np.column_stack([fbx * sx_scale,
                                          fby * sy_scale]).astype(np.int32)
                cv2.polylines(view, [fb_px], True, (255, 200, 0), 1,
                              cv2.LINE_AA)
            # crop region outline (thicker yellow, drawn after so it stays
            # on top of the field boundary)
            perim = _crop_outline_on_pano(pano, y, p, vf, ow, oh,
                                          projection, b_f)
            perim[:, 0] *= sx_scale
            perim[:, 1] *= sy_scale
            cv2.polylines(view, [perim.astype(np.int32)], True,
                          (0, 255, 255), 3, cv2.LINE_AA)
            # Raw-detection overlay (per-tile YOLO boxes + tile outlines +
            # conf labels). Drawn AFTER the crop outline so labels are on
            # top; the box colours are tile-coded so seam duplicates stand
            # out as overlapping boxes in different colours.
            if raw_yolo is not None:
                raw_yolo.draw(view, frame, sx_scale, sy_scale)
        elif gpu is not None:
            frame_for_gpu = (pad_source_for_blur(
                frame, blur_pad_rows,
                sigma_max_px=edge_blur_sigma,
                sigma_boundary_px=edge_blur_boundary_sigma,
                fade_rows=blur_fade_rows)
                if blur_pad_rows > 0 else frame)
            view = gpu.render(frame_for_gpu, y, p, vf, blend=b_f)
        else:
            view = pano.render(frame, y, p, vf, ow, oh,
                               interp=cv2.INTER_CUBIC,
                               projection=projection, blend=b_f)

        if overlay is not None:
            gm, ofr, byframe = overlay
            s = int(ofr[np.argmin(np.abs(ofr - f))])
            pls = byframe.get(s, [])
            if pls:
                P = np.array([[q["X"], q["Z"]] for q in pls], float)
                on_field = gm.in_field(P, margin_near=0.0,
                                       margin_far=0.0, margin_ends=2.0)
                if debug_pano:
                    rays = gm.ray_from_ground(P)
                    sxa, sya = pano.directions_to_src(rays)
                    uv = np.column_stack([sxa * sx_scale, sya * sy_scale])
                else:
                    uv = pano.world_to_view(
                        gm.ray_from_ground(P), y, p, vf, ow, oh,
                        projection=projection, blend=b_f)
                radius = 6 if debug_pano else 9
                for (u, v), q, on in zip(uv, pls, on_field):
                    if not np.isfinite(u) or not np.isfinite(v):
                        continue
                    if not (0 <= u < out_w and 0 <= v < out_h):
                        continue
                    if plain_dots:
                        c = (0, 255, 255)        # one colour, no on/off split
                    elif "label" in q:
                        if q["label"] == "player":
                            c = (0, 200, 0)      # green = stable active
                        elif q["label"] == "probation":
                            c = (0, 255, 255)    # yellow = in probation
                        else:
                            c = (0, 0, 255)      # red = sideline / foreign
                    else:
                        c = (0, 200, 0) if on else (0, 0, 255)
                    cv2.circle(view, (int(u), int(v)), radius, c, -1)
                    cv2.circle(view, (int(u), int(v)), radius, (255, 255, 255), 1)
                    # If tracks.json carries per-player conf, draw it next
                    # to the dot. Black halo + white text for legibility
                    # against grass/treeline backgrounds. Only in debug
                    # modes (plain_dots or labelled classify).
                    if plain_dots and "conf" in q:
                        txt = f"{q['conf']:.2f}"
                        tx, ty = int(u) + radius + 2, int(v) + 4
                        cv2.putText(view, txt, (tx, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(view, txt, (tx, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 255), 1, cv2.LINE_AA)
        writer.send(np.ascontiguousarray(view))
        done = k + 1
        rel = _time.time() - render_started
        rfps = done / max(rel, 1e-6)
        eta = (n - done) / max(rfps, 1e-6)
        prog.update(progress=done / max(1, n),
                    detail=f"frame {done}/{n}  {rfps:.1f} fps  "
                           f"eta {eta/60:.1f}m",
                    current_frame=f, f_start=f0, f_end=f1,
                    fps_observed=rfps, eta_s=eta)
        if k % 100 == 0:
            print(f"render {k}/{n}  {rfps:.1f}fps  eta {eta/60:.1f}m",
                  flush=True)
    _stop_evt.set()
    # Drain queue so the producer thread can exit if it's blocked on put().
    while True:
        try:
            item = _frame_q.get_nowait()
            if item is _frame_sentinel:
                break
        except _queue.Empty:
            break
    _prefetch_thread.join(timeout=5.0)
    cap.release()
    prog.set_step("finalize_video", detail="closing encoder")
    writer.close()
    print(f"wrote {out_path}  ({n} frames @ {fps:.2f}fps, mode={mode}, "
          f"size={out_w}x{out_h})", flush=True)
    prog.done(out_path=out_path, frames=n)
    return out_path
