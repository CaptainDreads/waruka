# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Perception pass: detect + globally track players in metric ground space.

Pipeline (per perception-stride frame):
    NVDEC decode -> GPU tile remap -> batched YOLO predict ->
    project every box to (X, Z) via the homography (head/foot anchor,
    plausibility-gated) -> single global ground-space tracker
    (waruka.track.Tracker) does per-frame fusion + Hungarian-associated
    Kalman update -> emit per-source-frame JSON.

Why this shape: the old per-tile BoTSORT + cross-tile merge architecture
hit a ~1.4x duplication floor (4 m far-field projection error vs <2 m
player spacing) and needed a post-process coast cap to stop dots parking
on stale positions during multi-frame YOLO misses. Moving tracking into a
single global ground-space Kalman makes coast cap, stationary-FP
suppression, and ID continuity all first-class -- no merge phase, no
post-process script. See waruka/track.py for the tracker.

What stayed: tiles (8 wide rectilinear views; small-object recall requires
this), batched YOLO across all tiles in one GPU call, hybrid head/foot
projection with the edge-anchor fix, the plausibility gate, NVDEC + GPU
batch remap. What's gone: per-tile BoTSORT, cross-tile merge, post-process
coast cap script, _DetSnapshot adapter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ProjectConfig
from .ground import GroundModel
from .progress import Progress as _Progress
from .track import Tracker

L_ANKLE, R_ANKLE = 15, 16  # COCO-17 keypoint indices (legacy, pose-only)


def _resolve_weights_path(name: str = "yolo11n.pt") -> str:
    """Return the absolute path to a YOLO weights file if it lives in
    the project root, otherwise the bare name.

    Why: Ultralytics' ``YOLO("yolo11n.pt")`` searches the subprocess's
    CWD first, falls back to the Ultralytics cache, then downloads
    from the CDN. The GUI + pipeline launch every ``waruka track``
    subprocess with ``cwd=<artefact_dir>`` (for `_progress.json`
    isolation), so each new match's empty artefact dir misses the
    CWD copy and triggers a fresh ~5MB download. Resolving to the
    absolute path under the project root bypasses the CWD search
    and reuses the existing file. Returning the bare name on miss
    preserves the old behaviour for installs that don't ship the
    weights co-located with the package. [#37]
    """
    candidate = Path(__file__).resolve().parent.parent / name
    return str(candidate) if candidate.exists() else name


@dataclass
class Tile:
    yaw_deg: float
    pitch_deg: float
    vfov_deg: float
    out_w: int
    out_h: int
    map_x: np.ndarray  # (out_h,out_w) -> source x
    map_y: np.ndarray


def _corner_dirs(pano, corners):
    c = np.asarray(corners, float)
    return pano.src_to_direction(c[:, 0], c[:, 1])


def build_tiles(pano, gm, tile_w=1920, tile_h=720, overlap=0.15,
                margin_deg=6.0, down_pad_deg=20.0, rows: int = 2,
                row_overlap_deg: float = 16.0,
                far_head_pad_deg: float = 6.0,
                tile_h_near: int | None = 960,
                tile_h_single: int | None = None) -> list[Tile]:
    """Cover the whole field ground with overlapping rectilinear tiles.

    Tiles are sized from a dense sampling of the field ground via the
    homography (corner-only sizing clips near-sideline feet — closest ground
    point is below any corner).

    With `rows=2`, the lat range is split asymmetrically: the NEAR row
    inherits the full head_pad needed by close players (their heads
    subtend many degrees above their feet), while the FAR row gets only
    `far_head_pad_deg` (far players subtend few degrees). The old code
    used a symmetric vfov for both rows, which left ~60% of the far tile
    pointed at sky/treeline above the horizon and starved far-player
    pixel density. Per-row sizing roughly halves far-row vfov, doubling
    px/deg for the far half of the field. Set rows=1 to restore the
    single-row layout.

    `down_pad_deg` extends NEAR-row coverage *below* the closest field
    ground point so off-field / sideline-standing players (Z<0, sitting
    in the bench / camera-side region) still have their feet inside a
    tile. Without this padding (or with too little), close off-field
    players are seen only by the FAR row -- which has their feet below
    its bottom edge -- and have to be located via `head_to_ground` from
    just the head pixel. That works when calibration is solid, but fails
    badly when the homography has weak Y-scale (4-corner-only fits),
    landing the dot at mid-body. Bumping to ~12 deg restores direct foot
    visibility for the typical sideline-bench region.
    """
    L, W = gm.field_length_m, gm.field_width_m
    xs = np.linspace(-L / 2 - 1, L / 2 + 1, 70)
    zs = np.linspace(-1.5, W + 1.5, 28)
    gx, gz = np.meshgrid(xs, zs)
    rays = gm.ray_from_ground(np.column_stack([gx.ravel(), gz.ravel()]))
    lon = np.degrees(np.arctan2(rays[:, 0], rays[:, 2]))
    lat = np.degrees(np.arcsin(np.clip(rays[:, 1], -1, 1)))
    g_lo, g_hi = float(lat.min()), float(lat.max())  # ground-feet lat range
    head_pad_near = float(np.clip(0.85 * abs(g_lo), 14.0, 38.0))
    lon_lo, lon_hi = lon.min() - margin_deg, lon.max() + margin_deg

    rows = max(1, int(rows))
    # Each band is (pitch, vfov, this_band_tile_h). Allowing tile_h to vary
    # per row lets the NEAR row be taller in pixels so it can extend further
    # downward (more vfov) without losing px/deg -- catches very-close
    # off-field/sideline players whose feet would otherwise fall below the
    # NEAR row's bottom edge.
    #
    # pitch_deg sign convention (verified empirically 2026-05-27): the
    # view_maps `pitch_deg` parameter is the angle by which the camera
    # rotates around its X axis, and per `_rot_x` in projection.py, a
    # positive pitch_deg sends the camera's optical axis to a NEGATIVE
    # world latitude (looks DOWN below horizon). So to point a tile at a
    # target world latitude L, pass pitch_deg = -L. NEAR row should look
    # DOWN at close (negative-lat) ground, so its pitch_deg must be
    # POSITIVE. Earlier versions of this function passed the midpoint of
    # the world lat range directly as pitch_deg without negation, which
    # silently aimed the NEAR row at the sky/treeline (+lat) -- it
    # contributed almost nothing useful, and all the field detection was
    # being done by the FAR row alone.
    th_near = tile_h_near if tile_h_near is not None else tile_h
    th_single = (tile_h_single if tile_h_single is not None
                  else th_near + tile_h)
    if rows == 1:
        lat_lo = g_lo - down_pad_deg
        lat_hi = g_hi + head_pad_near
        bands = [(-0.5 * (lat_lo + lat_hi), min(lat_hi - lat_lo, 88.0),
                  th_single)]
    else:
        split = 0.5 * (g_lo + g_hi)
        near_lo = g_lo - down_pad_deg
        near_hi = max(split + row_overlap_deg / 2, g_lo + head_pad_near)
        near_vfov = min(near_hi - near_lo, 88.0)
        near_pitch = -0.5 * (near_lo + near_hi)
        far_lo = split - row_overlap_deg / 2
        far_hi = g_hi + far_head_pad_deg
        far_vfov = min(far_hi - far_lo, 88.0)
        far_pitch = -0.5 * (far_lo + far_hi)
        bands = [(near_pitch, near_vfov, th_near),
                 (far_pitch, far_vfov, tile_h)]

    tiles: list[Tile] = []
    for pitch, vfov, this_tile_h in bands:
        hfov_tile = vfov * tile_w / this_tile_h
        step = hfov_tile * (1.0 - overlap)
        n = max(1, int(np.ceil((lon_hi - lon_lo) / step)))
        yaws = ([0.5 * (lon_lo + lon_hi)] if n == 1 else
                list(np.linspace(lon_lo + hfov_tile / 2,
                                 lon_hi - hfov_tile / 2, n)))
        for y in yaws:
            mx, my = pano.view_maps(y, pitch, vfov, tile_w, this_tile_h)
            tiles.append(Tile(y, pitch, vfov, tile_w, this_tile_h, mx, my))
    return tiles


def _foot_xy(box, kpts, kconf, kp_thr=0.5):
    """Bbox-bottom foot point (legacy, used as fallback when head-based
    projection is degenerate). Ankles can override only if they extend
    *below* the box (truncated-at-feet case).
    """
    x1, y1, x2, y2 = box
    fx, fy = float((x1 + x2) / 2), float(y2)
    if kconf is not None:
        ank = [kpts[i] for i in (L_ANKLE, R_ANKLE) if kconf[i] >= kp_thr]
        if ank:
            a = np.mean(ank, axis=0)
            if a[1] > y2:
                fx, fy = float(a[0]), float(a[1])
    return fx, fy


def _head_xy(box):
    """Bbox-top-centre — used as the head pixel for head-based ground
    projection. Heads are far more often visible than feet (which get
    occluded by other players, benches, frame edges), so this is the
    primary anchor when computing on-ground (X, Z)."""
    x1, y1, x2, y2 = box
    return float((x1 + x2) / 2), float(y1)


def _plausible_xz(xz, gm, z_lo=-12.0, z_hi_pad=15.0, x_pad=25.0):
    """Reject ground points that are obviously a projection blow-up.

    Neither the head- nor foot-projection is bounded as the viewing ray
    grazes the horizon: a short/occluded box's anchor pixel can ray nearly
    parallel to the assumed plane and fling the ground point hundreds of
    metres out. The window is deliberately generous (deep cuts, end-zone
    catches, near-camera benches and a band of sideline context all survive);
    it only kills the |Z|=50..600 m garbage that was ~40% of all markers.
    """
    X, Z = float(xz[0]), float(xz[1])
    return (z_lo <= Z <= gm.field_width_m + z_hi_pad
            and abs(X) <= gm.field_length_m / 2 + x_pad)


def _project_box(box, tile, pano, gm, height_m, top_cut=False, bot_cut=False):
    """Project a detection box to a metric ground (X, Z), robustly.

    Tries the aspect-preferred anchor first (head when the box is short/wide,
    i.e. feet likely occluded; foot otherwise), validates it against the
    plausibility window, and falls back to the other anchor if the primary
    blows up. Returns None when both are implausible (drop the detection).

    When the box is truncated by a tile edge, the anchor at that edge is
    invalid (a clipped head/foot sits at the tile border, not the player's
    real head/foot), so we use only the *visible* end: bottom-cut -> head,
    top-cut -> foot. This recovers players straddling the row-split, who
    used to be rejected outright -- the visible end still projects accurately
    via head_to_ground (head pixel + Y=height_m), and the plausibility gate
    drops it if the homography blows it up.

    NB on close-camera + bad-calibration cases: when the decomposed pose has
    a wrong Y-scale (typically 4-corner-only fits with no sideline anchors),
    head_to_ground for bot_cut boxes produces an incorrect (X, Z) that draws
    the dot at the player's mid-body instead of feet. A 2026-05-27
    experiment ("option B'") tried foot-anchor in this case and got the dot
    at the visible cut-off line -- but lost fusion consistency between
    NEAR-row and FAR-row detections of the same player (each tile's
    cut-off line is at a different ground point, so the same player got
    two un-fused dots). Reverted to head_to_ground here; the proper fix is
    correct sideline calibration (see memory:
    project_waruka_markfield_sideline_workflow).
    """
    x1, y1, x2, y2 = box
    cx = float((x1 + x2) / 2)
    aspect = (y2 - y1) / max(x2 - x1, 1.0)
    if bot_cut and not top_cut:
        order = ("head",)
    elif top_cut and not bot_cut:
        order = ("foot",)
    else:
        order = ("head", "foot") if aspect < 2.0 else ("foot", "head")
    for which in order:
        py = float(y1) if which == "head" else float(y2)
        ix = int(np.clip(cx, 0, tile.out_w - 1))
        iy = int(np.clip(py, 0, tile.out_h - 1))
        sx = tile.map_x[iy, ix]
        sy = tile.map_y[iy, ix]
        if not (0 <= sx < pano.src_w and 0 <= sy < pano.src_h):
            continue
        ray = pano.src_to_direction(np.array([sx]), np.array([sy]))
        if which == "head":
            xz = gm.head_to_ground(ray, height_m=height_m)[0]
        else:
            xz = gm.ground_from_ray(ray)[0]
        if np.isfinite(xz).all() and _plausible_xz(xz, gm):
            return float(xz[0]), float(xz[1])
    return None


class _GpuTileRemapper:
    """GPU batch remap of the source frame into all tiles via grid_sample.

    The per-frame CPU `cv2.remap` over 8 tiles is ~80 ms (25% of the
    detection loop). The tile maps are constant, so we precompute normalized
    sampling grids on the GPU once and resample all tiles in one batched
    grid_sample. Returns numpy tiles (downloaded) so the downstream YOLO
    predict path is unchanged. Bilinear (matches cv2 INTER_LINEAR).

    Tiles can have different (out_w, out_h) per row (e.g. taller NEAR tiles
    for close-player coverage). We group tiles by shape and run one
    grid_sample per shape group, then re-assemble back to the original
    tile order.
    """

    def __init__(self, src_w, src_h, tiles, device="cuda"):
        import torch
        self.torch = torch
        self.device = device
        self.n = len(tiles)
        # Group tile indices by (H, W).
        self._groups: list[tuple[list[int], "torch.Tensor"]] = []
        from collections import defaultdict
        by_shape: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, T in enumerate(tiles):
            H, W = T.map_x.shape
            by_shape[(H, W)].append(i)
        for (H, W), idxs in by_shape.items():
            grids = []
            for i in idxs:
                T = tiles[i]
                gx = 2.0 * T.map_x / (src_w - 1) - 1.0
                gy = 2.0 * T.map_y / (src_h - 1) - 1.0
                grids.append(np.stack([gx, gy], axis=-1))
            g = torch.from_numpy(np.stack(grids)).float().to(device)
            self._groups.append((idxs, g))

    def remap(self, frame):
        f = self.torch.from_numpy(np.ascontiguousarray(frame)).to(self.device)
        return self._remap_tensor(f)

    def remap_gpu(self, frame_t):
        return self._remap_tensor(frame_t)

    def _remap_tensor(self, f):
        torch = self.torch
        f = f.permute(2, 0, 1).unsqueeze(0).float()              # (1,3,Hs,Ws)
        out: list[np.ndarray | None] = [None] * self.n
        for idxs, grid in self._groups:
            n_g = grid.shape[0]
            fb = f.expand(n_g, -1, -1, -1)
            o = torch.nn.functional.grid_sample(
                fb, grid, mode="bilinear", align_corners=True,
                padding_mode="zeros")
            o = (o.permute(0, 2, 3, 1).round().clamp(0, 255).byte()
                 .cpu().numpy())
            for k, i in enumerate(idxs):
                out[i] = o[k]
        return out


def _predict_batched(model, imgs, conf, iou, imgsz, device, half=True):
    """Manually-batched YOLO predict: GPU letterbox + single forward
    pass + NMS (added v0.12 for #19). Replaces ultralytics'
    `model.predict(list-of-imgs)` which serialises into N forward
    passes despite accepting a list.

    Returns: list of (xyxy, confs, clses) ndarrays per image, in
    the original image's pixel coordinates. Same downstream shape as
    `r.boxes.xyxy.cpu().numpy()` / `r.boxes.conf` / `r.boxes.cls`
    from the predict() path.

    Letterbox preserves aspect ratio (matches what predict() does
    internally) so detections numerically agree with the legacy path
    to within NMS sort-order noise. Forward pass is FP16 via autocast
    for parity with `half=True`.
    """
    import torch
    import torch.nn.functional as F
    from ultralytics.utils.ops import non_max_suppression
    # Letterbox each tile to (imgsz, imgsz) on the GPU.
    scales = []
    letterboxed = []
    for img in imgs:
        h, w = img.shape[:2]
        t = (torch.from_numpy(np.ascontiguousarray(img))
             .to(device).float() / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, h, w)
        scale = min(imgsz / h, imgsz / w)
        if scale < 0.999 or scale > 1.001:
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            t = F.interpolate(t, size=(new_h, new_w), mode="bilinear",
                              align_corners=False)
        else:
            new_h, new_w = h, w
        pad_y = (imgsz - new_h) // 2
        pad_x = (imgsz - new_w) // 2
        # F.pad order: (left, right, top, bottom)
        t = F.pad(t, (pad_x, imgsz - new_w - pad_x,
                       pad_y, imgsz - new_h - pad_y),
                  value=114.0 / 255.0)
        letterboxed.append(t)
        scales.append((scale, pad_x, pad_y))
    batch = torch.cat(letterboxed, dim=0)  # (N, 3, imgsz, imgsz)
    with torch.no_grad():
        if half and device != "cpu":
            with torch.amp.autocast("cuda", enabled=True):
                preds = model.model(batch)
        else:
            preds = model.model(batch)
    if isinstance(preds, tuple):
        preds = preds[0]
    dets_list = non_max_suppression(preds, conf_thres=conf, iou_thres=iou,
                                     classes=[0])
    # Convert back to original-image coordinates.
    results = []
    for i, dets in enumerate(dets_list):
        if dets is None or len(dets) == 0:
            results.append((np.zeros((0, 4), float),
                            np.zeros(0, float),
                            np.zeros(0, int)))
            continue
        xyxy = dets[:, :4].cpu().numpy().astype(float)
        confs = dets[:, 4].cpu().numpy().astype(float)
        clses = dets[:, 5].cpu().numpy().astype(int)
        scale, pad_x, pad_y = scales[i]
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
        results.append((xyxy, confs, clses))
    return results


def run_perception(project: str, video: str | None = None,
                    stride: int = 3, t0: float = 0.0, t1: float | None = None,
                    out_path: str = "tracks.json", model_name="yolo11n.pt",
                    # NOTE: defaults synced to the CLI's production
                    # values (v0.11 update; previously the function
                    # had looser defaults that diverged from the CLI).
                    # In-process callers (pipeline, tests) now get
                    # production behaviour without having to pass
                    # every value explicitly.
                    conf: float = 0.50, iou: float = 0.5,
                    fuse_lat_m: float = 0.6, fuse_rad_m: float = 2.5,
                    max_coast_s: float = 0.3, min_hits: int = 5,
                    stationary_pos_spread_m: float = 0.5,
                    stationary_min_duration_s: float = 5.0,
                    phantom_window_s: float = 2.5,
                    phantom_max_spread_m: float = 0.1,
                    phantom_max_tiles: int = 8,
                    gate_mahal: float = 4.0,
                    q_accel: float = 3.0, r_pos: float = 0.8,
                    down_pad_deg: float = 20.0,
                    tile_h_near: int = 960,
                    tile_h_single: int | None = None,
                    rows: int = 1,
                    device=0, imgsz: int | None = None,
                    player_height_m: float = 1.75,
                    progress_every: int = 200,
                    half: bool = True, decoder: str = "auto",
                    # Batched predict (added v0.12, #19). Single forward
                    # pass over all tiles per frame instead of N. ~1.9x
                    # speedup on a 3-tile production setup. Default on.
                    # Set to False to revert to model.predict(list) for
                    # A/B comparison. Detections are numerically
                    # equivalent up to NMS sort-order noise.
                    batched_predict: bool = True,
                    # Cross-chunk handoff (#20b). When set, tracker
                    # resumes from the previous chunk's final state
                    # (active tracks + ID counter). String -> path to
                    # state JSON. Dict -> already-loaded state.
                    initial_tracker_state: str | dict | None = None,
                    # When set, the final tracker state is written to
                    # this path before returning -- next chunk passes
                    # this path as initial_tracker_state. The state
                    # is small (active tracks only, not history).
                    tracker_state_out: str | None = None,
                    # Back-emit: after the main emit, also re-emit
                    # this frame range using the current tracker
                    # state. The current tracker has hits from both
                    # this chunk and (via resumed state) the previous
                    # chunk, so the back-emit's interpolation for the
                    # previous chunk's last frames matches single-pass.
                    # Pipeline merges the back-emit into the
                    # cumulative tracks file to patch the prior chunk.
                    back_emit_range: tuple[int, int] | None = None,
                    back_emit_out: str | None = None):
    """Run detection + global ground-space tracking on a clip.

    Returns the output path. Writes a tracks.json with shape
        {video, fps, stride=1, perception_stride, field_length_m,
         field_width_m, frames: [{frame, t, players: [{id, X, Z, boxes?}]}]}
    -- compatible with the existing campath/render/detectpano consumers.

    decoder: 'auto' (NVDEC if available, else OpenCV), 'nvdec' (require),
    or 'opencv'/'cpu' (force CPU decode).
    """
    import time as _time

    prog = _Progress("track", source=video, out_path=out_path)
    try:
        prog.set_step("load_config", detail=f"reading {project}")
        cfg = ProjectConfig.load(project)
        src = video or cfg.source_video
        prog.state["source"] = src
        pano = cfg.pano
        if cfg.homography is None:
            prog.fail("no homography in project.json - run `waruka markfield`")
            raise SystemExit(
                "no homography in project.json - run `waruka markfield`")

        prog.set_step("build_tiles", detail="sampling field ground")
        gm = GroundModel(cfg.homography, cfg.field_length_m, cfg.field_width_m)
        tiles = build_tiles(pano, gm, down_pad_deg=down_pad_deg,
                             tile_h_near=tile_h_near,
                             tile_h_single=tile_h_single, rows=rows)
        if imgsz is None:
            imgsz = max(tiles[0].out_w, tiles[0].out_h)
        prog.update(detail=f"{len(tiles)} tiles, imgsz={imgsz}",
                    n_tiles=len(tiles), imgsz=imgsz)
        print(f"{len(tiles)} tiles, {tiles[0].out_w}x{tiles[0].out_h} "
              f"vfov={tiles[0].vfov_deg:.1f}  imgsz={imgsz}  "
              f"yaws={[round(t.yaw_deg,1) for t in tiles]}", flush=True)

        prog.set_step("load_model", detail=f"loading {model_name}")
        from ultralytics import YOLO
        model = YOLO(_resolve_weights_path(model_name))
        # batched_predict calls model.model() directly (bypassing the
        # ultralytics predict wrapper), so the underlying nn.Module
        # has to be on the target device with the right dtype. predict()
        # handles this automatically; the direct path needs an explicit
        # setup. eval() disables train-time dropout/BN updates.
        if batched_predict:
            _dev = device if isinstance(device, str) else f"cuda:{device}"
            model.model.to(_dev).eval()

        gpu_remap = None
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                gpu_remap = _GpuTileRemapper(pano.src_w, pano.src_h, tiles)
        except Exception:
            gpu_remap = None

        prog.set_step("open_video", detail=src)
        from . import nvdecode
        nv = None
        cap = None
        want_nv = decoder in ("auto", "nvdec") and gpu_remap is not None
        if want_nv:
            try:
                if nvdecode.is_available():
                    nv = nvdecode.NvVideoDecoder(src)
                elif decoder == "nvdec":
                    raise RuntimeError(
                        "decoder='nvdec' requested but NVDEC unavailable")
            except Exception:
                if decoder == "nvdec":
                    raise
                nv = None
        if nv is not None:
            fps = nv.fps or 20.0
            nfr = nv.num_frames
            prog.update(detail=f"NVDEC {nv.width}x{nv.height} {nfr}f")
        else:
            cap = cv2.VideoCapture(src)
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            prog.update(detail="OpenCV CPU decode (NVDEC unavailable)")

        dt = stride / fps
        f_start = int(t0 * fps)
        f_end = nfr if t1 is None else min(nfr, int(t1 * fps))

        # Camera ground position from the homography (per clip; the mount is
        # not assumed to be at midfield). The tracker uses this to split
        # per-frame fusion displacements into reliable lateral vs noisy
        # radial (depth) components. Fall back to the field origin only if
        # the decomposition is degenerate.
        try:
            r1, r2, r3, tcam = gm.decompose_pose()
            Cw = -np.linalg.solve(np.column_stack([r1, r2, r3]), tcam)
            cam_xz = (float(Cw[0]), float(Cw[2]))
        except Exception:
            cam_xz = (0.0, 0.0)
        print(f"camera ground pos: X={cam_xz[0]:+.1f} Z={cam_xz[1]:+.1f}",
              flush=True)

        # NEAR row tiles have pitch_deg > 0 (look down at close field);
        # FAR row has pitch_deg < 0 (look up at far/treeline). build_tiles
        # emits NEAR row first, so a single count is enough to split.
        near_tile_count = sum(1 for T in tiles if T.pitch_deg > 0)
        tracker = Tracker(
            dt=dt, fps=fps, cam_xz=cam_xz,
            fuse_lat_m=fuse_lat_m, fuse_rad_m=fuse_rad_m,
            gate_mahal=gate_mahal,
            max_coast_s=max_coast_s, min_hits=min_hits,
            stationary_pos_spread_m=stationary_pos_spread_m,
            stationary_min_duration_s=stationary_min_duration_s,
            phantom_window_s=phantom_window_s,
            phantom_max_spread_m=phantom_max_spread_m,
            phantom_max_tiles=phantom_max_tiles,
            near_tile_count=near_tile_count,
            q_accel=q_accel, r_pos=r_pos)
        # Cross-chunk handoff: restore tracker state from previous chunk
        # so active tracks (with their Kalman states and IDs) continue
        # uninterrupted across the boundary (#20b cross-chunk continuity).
        if initial_tracker_state is not None:
            import json as _json
            if isinstance(initial_tracker_state, str):
                with open(initial_tracker_state) as _f:
                    state = _json.load(_f)
            else:
                state = initial_tracker_state
            tracker.load_state(state)
            print(f"resumed tracker with {len(tracker.tracks)} active "
                  f"tracks, next_id={tracker._next_id}", flush=True)

        prog.set_step("detect_and_track",
                      progress=0.0,
                      f_start=f_start, f_end=f_end, current_frame=f_start,
                      fps_observed=0.0, eta_s=None,
                      per_tile_track_counts=[0] * len(tiles))

        def _frame_source():
            if nv is not None:
                for fi_, gframe in nv.frames(f_start, f_end):
                    if (fi_ - f_start) % stride != 0:
                        continue
                    yield fi_, gpu_remap.remap_gpu(gframe)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
                fi_ = f_start
                while fi_ < f_end:
                    if (fi_ - f_start) % stride != 0:
                        if not cap.grab():
                            break
                        fi_ += 1
                        continue
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if gpu_remap is not None:
                        imgs_ = gpu_remap.remap(frame)
                    else:
                        imgs_ = [cv2.remap(frame, t.map_x, t.map_y,
                                           cv2.INTER_LINEAR) for t in tiles]
                    yield fi_, imgs_
                    fi_ += 1

        edge_px = 6
        loop_started = _time.time()
        fi = f_start
        per_tile_hit_counts = [0] * len(tiles)
        # Worker-thread prefetching was tried for #20 but NVDEC's CUDA
        # context is per-thread (driver throws CUDA_ERROR_INVALID_CONTEXT
        # when accessed cross-thread). Plus the producer's GPU remap
        # serializes with the consumer's predict on the default stream,
        # so the actual wall-clock win was modest (~5-10%). Revisit
        # with explicit CUDA streams + per-thread context if track
        # becomes the bottleneck again.
        for fi, imgs in _frame_source():
            if batched_predict:
                dets_per_tile = _predict_batched(
                    model, imgs, conf=conf, iou=iou, imgsz=imgsz,
                    device=device, half=half)
            else:
                res = model.predict(imgs, conf=conf, iou=iou, device=device,
                                    imgsz=imgsz, half=half, verbose=False)
                dets_per_tile = []
                for r in res:
                    if r.boxes is None or len(r.boxes) == 0:
                        dets_per_tile.append((np.zeros((0, 4)),
                                              np.zeros(0),
                                              np.zeros(0, int)))
                    else:
                        dets_per_tile.append((
                            r.boxes.xyxy.cpu().numpy(),
                            r.boxes.conf.cpu().numpy(),
                            r.boxes.cls.cpu().numpy().astype(int),
                        ))
            raw_dets = []
            for ti, (xyxy, confs, clses) in enumerate(dets_per_tile):
                T = tiles[ti]
                if len(xyxy) == 0:
                    continue
                n_here = 0
                for k in range(len(xyxy)):
                    if int(clses[k]) != 0:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in xyxy[k]]
                    top_cut = y1 <= edge_px
                    bot_cut = y2 >= T.out_h - edge_px
                    if top_cut and bot_cut:
                        continue
                    xz = _project_box((x1, y1, x2, y2), T, pano, gm,
                                      player_height_m,
                                      top_cut=top_cut, bot_cut=bot_cut)
                    if xz is None:
                        continue
                    raw_dets.append((xz[0], xz[1], float(confs[k]),
                                     int(ti),
                                     (int(x1), int(y1), int(x2), int(y2))))
                    n_here += 1
                per_tile_hit_counts[ti] = n_here

            tracker.step(int(fi), raw_dets)

            done = max(1, fi - f_start)
            loop_elapsed = _time.time() - loop_started
            fps_obs = done / max(loop_elapsed, 1e-6)
            remaining = max(0, f_end - fi)
            eta = remaining / max(fps_obs, 1e-6)
            span = max(1, f_end - f_start)
            n_live = sum(1 for t in tracker.tracks if t.confirmed)
            prog.update(
                progress=(fi - f_start) / span,
                detail=(f"frame {fi}/{f_end} fps {fps_obs:.2f} "
                        f"eta {eta/60:.1f}m  live={n_live}"),
                current_frame=fi,
                fps_observed=fps_obs,
                eta_s=eta,
                per_tile_track_counts=list(per_tile_hit_counts),
            )
            if (fi - f_start) % progress_every == 0:
                print(f"frame {fi}/{f_end}  fps={fps_obs:.2f}  "
                      f"eta={eta/60:.1f}m  live={n_live}", flush=True)

        if cap is not None:
            cap.release()
        if nv is not None:
            nv.close()

        prog.set_step("emit_per_frame", detail=f"{len(tracker.history)} tracks")
        # Densify all confirmed, non-stationary tracks to every source frame.
        # Stride=1 in output (every frame); perception_stride stored for
        # provenance. Coast cap is native -- frames more than max_coast_s
        # away from any real-detection hit are dropped per-track.
        perframe = tracker.emit_per_frame(f_start, max(f_start, f_end - 1))
        frames_out = [{"frame": f, "t": round(f / fps, 3), "players": pls}
                      for f, pls in sorted(perframe.items())]
        stats = getattr(tracker, "_emit_stats", {})
        print(f"tracks: history={stats.get('n_history', 0)} "
              f"emitted={stats.get('n_emitted', 0)} "
              f"dropped_short={stats.get('n_dropped_short', 0)} "
              f"dropped_stationary={stats.get('n_dropped_stationary', 0)} "
              f"phantom_frames_suppressed="
              f"{stats.get('n_suppressed_phantom_frames', 0)}",
              flush=True)

        prog.set_step("write_output", detail=out_path)
        json.dump({"video": src, "fps": fps, "stride": 1,
                   "perception_stride": stride,
                   "field_length_m": cfg.field_length_m,
                   "field_width_m": cfg.field_width_m,
                   "frames": frames_out}, open(out_path, "w"))
        print(f"wrote {out_path}  ({len(frames_out)} frames)", flush=True)

        # Cross-chunk handoff: emit tracker state for the next chunk.
        if tracker_state_out is not None:
            state = tracker.get_state()
            with open(tracker_state_out, "w") as _f:
                json.dump(state, _f)
            print(f"wrote {tracker_state_out} "
                  f"({len(state['tracks'])} active tracks)", flush=True)

        # Back-emit: re-emit PREVIOUS chunk's frame range using THIS
        # chunk's tracker. The current tracker has hits from both
        # chunks (because it resumed from previous chunk's state and
        # added new hits), so interpolation across the boundary is as
        # good as single-pass. Pipeline merges this with cumulative to
        # patch the previous chunk's last-frame positions.
        if back_emit_range is not None and back_emit_out is not None:
            bf0, bf1 = back_emit_range
            back_perframe = tracker.emit_per_frame(int(bf0), int(bf1))
            back_frames = [{"frame": f, "t": round(f / fps, 3),
                            "players": pls}
                           for f, pls in sorted(back_perframe.items())]
            with open(back_emit_out, "w") as _f:
                json.dump({"video": src, "fps": fps, "stride": 1,
                           "perception_stride": stride,
                           "field_length_m": cfg.field_length_m,
                           "field_width_m": cfg.field_width_m,
                           "frames": back_frames}, _f)
            print(f"wrote {back_emit_out} (back-emit {bf0}-{bf1}, "
                  f"{len(back_frames)} frames)", flush=True)

        prog.done(final_tracks=stats.get("n_emitted", 0),
                  out_path=out_path)
        return out_path
    except SystemExit:
        raise
    except Exception as e:
        prog.fail(f"{type(e).__name__}: {e}")
        raise


def render_track_overlay(project: str, tracks_json: str, t_seconds: float,
                         out_path: str):
    """Reproject tracked ground positions onto the raw panorama to eyeball
    whether detections land on real players."""
    cfg = ProjectConfig.load(project)
    pano = cfg.pano
    gm = GroundModel(cfg.homography, cfg.field_length_m, cfg.field_width_m)
    data = json.load(open(tracks_json))
    fps = data["fps"]
    fidx = int(t_seconds * fps)
    rec = min(data["frames"], key=lambda f: abs(f["frame"] - fidx))

    cap = cv2.VideoCapture(data["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, rec["frame"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("cannot read frame for overlay")

    b = gm.boundary_xz(2.0)
    bpx, bpy = pano.directions_to_src(gm.ray_from_ground(b))
    cv2.polylines(frame, [np.column_stack([bpx, bpy]).astype(np.int32)],
                  True, (0, 200, 255), 2, cv2.LINE_AA)
    for p in rec["players"]:
        d = gm.ray_from_ground(np.array([[p["X"], p["Z"]]]))
        sx, sy = pano.directions_to_src(d)
        c = (int(sx[0]), int(sy[0]))
        cv2.circle(frame, c, 9, (0, 0, 255), 2)
        cv2.putText(frame, str(p["id"]), (c[0] + 8, c[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    h, w = frame.shape[:2]
    cv2.imwrite(out_path, cv2.resize(frame, (1700, int(1700 * h / w))))
    print(f"{out_path}: frame {rec['frame']} t={rec['t']}s "
          f"players={len(rec['players'])}")


def dump_detection_tiles(project: str, t_seconds: float, out_dir: str = "_tiles",
                         video: str | None = None, conf: float = 0.20,
                         iou: float = 0.5, imgsz: int | None = None,
                         model_name: str = "yolo11n.pt",
                         player_height_m: float = 1.75):
    """Debug aid: dump every detection tile at time `t_seconds` with each
    YOLO box drawn and annotated.

    Per box: class+confidence, whether it's edge-truncated (CUT-top/bot), and
    where it projects on the ground (X,Z) or DROP if the plausibility gate
    rejects it. Same detection settings (conf/iou/imgsz) as `track`, so it
    reflects the real pipeline.
    """
    from pathlib import Path
    from ultralytics import YOLO
    cfg = ProjectConfig.load(project)
    src = video or cfg.source_video
    pano = cfg.pano
    gm = GroundModel(cfg.homography, cfg.field_length_m, cfg.field_width_m)
    tiles = build_tiles(pano, gm)
    if imgsz is None:
        imgsz = max(tiles[0].out_w, tiles[0].out_h)

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    fidx = int(t_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame at t={t_seconds}s")
    imgs = [cv2.remap(frame, t.map_x, t.map_y, cv2.INTER_LINEAR) for t in tiles]

    model = YOLO(_resolve_weights_path(model_name))
    res = model.predict(imgs, conf=conf, iou=iou, imgsz=imgsz, half=True,
                        verbose=False)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    edge_px = 6
    n_person = 0
    for ti, (T, img, r) in enumerate(zip(tiles, imgs, res)):
        canvas = img.copy()
        boxes = r.boxes
        for k in range(len(boxes)):
            cls = int(boxes.cls[k].item())
            cf = float(boxes.conf[k].item())
            xy = boxes.xyxy[k].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xy]
            person = cls == 0
            top_cut = y1 <= edge_px
            bot_cut = y2 >= T.out_h - edge_px
            if person:
                n_person += 1
                if top_cut and bot_cut:
                    proj = "DROP(both-cut)"
                else:
                    xz = _project_box(tuple(xy), T, pano, gm, player_height_m,
                                      top_cut=top_cut, bot_cut=bot_cut)
                    proj = (f"X{xz[0]:+.0f} Z{xz[1]:.0f}" if xz is not None
                            else "DROP(implausible)")
                cut = ("Tcut" if top_cut else "") + ("Bcut" if bot_cut else "")
                color = ((0, 0, 255) if proj.startswith("DROP")
                         else (0, 220, 0))
                txt = f"p {cf:.2f} {proj} {cut}".rstrip()
            else:
                color = (0, 165, 255)
                txt = f"c{cls} {cf:.2f}"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, txt, (x1, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        head = (f"tile {ti}  yaw {T.yaw_deg:.0f} pitch {T.pitch_deg:.0f} "
                f"vfov {T.vfov_deg:.0f}  conf>={conf} iou={iou}")
        cv2.putText(canvas, head, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, head, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"tile{ti:02d}_yaw{T.yaw_deg:+04.0f}.png"), canvas)
    print(f"wrote {len(tiles)} tiles to {out}/  frame {fidx} t={t_seconds}s  "
          f"{n_person} person boxes (conf>={conf}, iou={iou})")
    return str(out)


def dump_detection_pano(project: str, t_seconds: float,
                        tracks_json: str | None = None,
                        out_path: str = "_detpano.png",
                        video: str | None = None,
                        conf: float = 0.20, iou: float = 0.5,
                        imgsz: int | None = None,
                        model_name: str = "yolo11n.pt",
                        player_height_m: float = 1.75,
                        max_w: int = 4608):
    """Single-frame diagnostic: detection boxes + tracked feet dots on pano.

    If `tracks_json` carries stored per-frame bboxes, each tracked player's
    actual contributing box(es) are drawn 1:1 next to their dot/ID (no YOLO
    re-run, ~5 s). Otherwise YOLO is re-run on this frame (~45 s).
    """
    cfg = ProjectConfig.load(project)
    src = video or cfg.source_video
    pano = cfg.pano
    gm = GroundModel(cfg.homography, cfg.field_length_m, cfg.field_width_m)
    tiles = build_tiles(pano, gm)
    if imgsz is None:
        imgsz = max(tiles[0].out_w, tiles[0].out_h)

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    fidx = int(t_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame at t={t_seconds}s")

    edge_px = 6
    canvas = frame.copy()
    b = gm.boundary_xz(1.0)
    bx, by = pano.directions_to_src(gm.ray_from_ground(b))
    cv2.polylines(canvas, [np.column_stack([bx, by]).astype(np.int32)], True,
                  (0, 200, 255), 2, cv2.LINE_AA)

    use_stored = False
    rec = None
    if tracks_json:
        td = json.load(open(tracks_json))
        rec = min(td["frames"], key=lambda f: abs(f["frame"] - fidx))
        if any("boxes" in p for p in rec["players"]):
            use_stored = True

    def _edge_to_pano(T, x1, y1, x2, y2, n=12):
        H, W = T.map_x.shape
        x1c = int(np.clip(x1, 0, W - 1)); x2c = int(np.clip(x2, 0, W - 1))
        y1c = int(np.clip(y1, 0, H - 1)); y2c = int(np.clip(y2, 0, H - 1))
        top = np.linspace(x1c, x2c, n).astype(int); top_y = np.full(n, y1c)
        right_y = np.linspace(y1c, y2c, n).astype(int); right_x = np.full(n, x2c)
        bot = np.linspace(x2c, x1c, n).astype(int); bot_y = np.full(n, y2c)
        left_y = np.linspace(y2c, y1c, n).astype(int); left_x = np.full(n, x1c)
        xs = np.concatenate([top, right_x, bot, left_x])
        ys = np.concatenate([top_y, right_y, bot_y, left_y])
        px = T.map_x[ys, xs]; py = T.map_y[ys, xs]
        return np.column_stack([px, py]).astype(np.int32)

    n_box = 0
    n_tracked = 0
    if use_stored:
        for p in rec["players"]:
            for b_ in p.get("boxes", []):
                T = tiles[int(b_["tile"])]
                x1, y1, x2, y2 = b_["xyxy"]
                poly = _edge_to_pano(T, x1, y1, x2, y2)
                cv2.polylines(canvas, [poly], True, (0, 220, 0), 1,
                              cv2.LINE_AA)
                lx, ly = int(T.map_x[int(np.clip(y1, 0, T.out_h - 1)),
                                      int(np.clip(x1, 0, T.out_w - 1))]), \
                         int(T.map_y[int(np.clip(y1, 0, T.out_h - 1)),
                                      int(np.clip(x1, 0, T.out_w - 1))])
                cv2.putText(canvas, f"#{p['id']}", (lx, max(12, ly - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1,
                            cv2.LINE_AA)
                n_box += 1
    else:
        from ultralytics import YOLO
        imgs = [cv2.remap(frame, t.map_x, t.map_y, cv2.INTER_LINEAR)
                for t in tiles]
        model = YOLO(_resolve_weights_path(model_name))
        res = model.predict(imgs, conf=conf, iou=iou, imgsz=imgsz, half=True,
                            verbose=False)
        for ti, (T, r) in enumerate(zip(tiles, res)):
            for k in range(len(r.boxes)):
                if int(r.boxes.cls[k].item()) != 0:
                    continue
                xy = r.boxes.xyxy[k].cpu().numpy()
                cf = float(r.boxes.conf[k].item())
                x1, y1, x2, y2 = xy
                top_cut = y1 <= edge_px
                bot_cut = y2 >= T.out_h - edge_px
                if top_cut and bot_cut:
                    color = (0, 0, 255)
                else:
                    xz = _project_box(tuple(xy), T, pano, gm, player_height_m,
                                      top_cut=top_cut, bot_cut=bot_cut)
                    color = (0, 0, 255) if xz is None else (0, 220, 0)
                poly = _edge_to_pano(T, x1, y1, x2, y2)
                cv2.polylines(canvas, [poly], True, color, 1, cv2.LINE_AA)
                lx, ly = int(T.map_x[int(np.clip(y1, 0, T.out_h - 1)),
                                      int(np.clip(x1, 0, T.out_w - 1))]), \
                         int(T.map_y[int(np.clip(y1, 0, T.out_h - 1)),
                                      int(np.clip(x1, 0, T.out_w - 1))])
                cv2.putText(canvas, f"{cf:.2f}", (lx, max(12, ly - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                            cv2.LINE_AA)
                n_box += 1

    if rec is not None:
        for p in rec["players"]:
            d = gm.ray_from_ground(np.array([[p["X"], p["Z"]]]))
            sx, sy = pano.directions_to_src(d)
            c = (int(sx[0]), int(sy[0]))
            cv2.circle(canvas, c, 7, (0, 255, 255), 2)
            cv2.putText(canvas, str(p["id"]), (c[0] + 6, c[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                        cv2.LINE_AA)
            n_tracked += 1

    h, w = canvas.shape[:2]
    if w > max_w:
        canvas = cv2.resize(canvas, (max_w, int(max_w * h / w)))
    cv2.imwrite(out_path, canvas)
    src_tag = "stored" if use_stored else "live-YOLO"
    print(f"wrote {out_path}  frame {fidx} t={t_seconds}s  "
          f"boxes drawn={n_box} ({src_tag})  tracked dots={n_tracked}")
    return out_path
