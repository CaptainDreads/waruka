# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Ground-space global multi-object tracker.

One Kalman filter per track in metric (X, Z) coordinates. Replaces the old
per-tile BoTSORT + cross-tile merge architecture: per-frame dedup of raw
per-tile detections happens upstream in ground space (anisotropic fusion),
then this tracker associates the fused detections to persistent track IDs
via Hungarian assignment on Mahalanobis distance. One ID per real player,
in world coordinates -- no cross-tile reassignment phase.

Constant-velocity 4-state model (X, Z, Vx, Vz). Tracks coast through brief
detection misses via Kalman prediction; coast cap and stationary-track
suppression are first-class knobs:
  - max_coast_s     -> the tracker's max emission gap (native, no post-proc)
  - stationary_pos_spread_m / stationary_min_duration_s
                     -> drop tracks that hardly move (fixed-object FPs)
  - min_hits         -> birth threshold (kills single-frame YOLO blips)

Output densification (one entry per source frame) is done at emit time by
linear interpolation between the track's real-detection hits, capped to
`max_coast_frames` distance from any real-detection frame. This preserves
the existing tracks.json shape that downstream campath/render consume.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class _Hit:
    frame: int
    x: float
    z: float
    conf: float                # max YOLO confidence across contributing boxes
    boxes: list                # [(tile_idx, (x1,y1,x2,y2), conf), ...]


def fuse_detections(dets, cam_xz, lat_tol_m=0.6, rad_tol_m=2.5,
                    near_tile_count=0, cross_row_factor=2.5):
    """Per-frame anisotropic-distance clustering of raw per-tile detections.

    `dets`: list of (X, Z, conf, tile_idx, (x1,y1,x2,y2)).

    Distance is split relative to the bearing from `cam_xz` (the camera's
    ground position, derived per-clip from the homography):
        score = hypot(lateral / lat_tol_m, radial / rad_tol_m)
    Pairs with score < 1 are merged. This catches depth-divergent duplicates
    (same player seen from two tiles, where projection puts them at noticeably
    different range but the same bearing) while keeping two laterally-
    separated players distinct.

    **Cross-row tolerance widening.** When `near_tile_count > 0`, tile
    indices [0, near_tile_count) are NEAR row and [near_tile_count, N) are
    FAR row. A NEAR-row detection of a close player uses foot-projection
    (full body visible), while a FAR-row detection of the same player is
    typically bot_cut and uses `head_to_ground` -- which depends on the
    homography's Y-scale being correctly calibrated. On under-calibrated
    clips (no sideline marks) the two projections can differ by 3-5 m in
    ground space, well beyond the same-row fusion tolerance. Multiplying
    tolerances by `cross_row_factor` (~2.5) when the pair spans rows lets
    these cross-row duplicates merge. A correctly-calibrated clip has head
    and foot agreeing to ~0.5 m so the looser tolerance changes nothing
    there. Risk: two distinct same-yaw players in different rows whose
    spacing is within the looser tolerance get over-merged; uncommon at
    typical 7v7 player spacing.

    Returns: list of (X_avg, Z_avg, conf_max, [(tile, box), ...]). Position is
    confidence-weighted-averaged across the cluster; every contributing
    (tile, box) is preserved so downstream diagnostics can draw them.
    """
    n = len(dets)
    if n == 0:
        return []
    xs = np.array([d[0] for d in dets], float)
    zs = np.array([d[1] for d in dets], float)
    cs = np.array([d[2] for d in dets], float)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cx, cz = cam_xz
    tiles = np.array([d[3] for d in dets], int)
    is_near = (tiles < near_tile_count) if near_tile_count > 0 else None
    for i in range(n):
        for j in range(i + 1, n):
            mx = 0.5 * (xs[i] + xs[j]) - cx
            mz = 0.5 * (zs[i] + zs[j]) - cz
            r = float(np.hypot(mx, mz))
            if r < 1e-6:
                continue
            ux, uz = mx / r, mz / r
            dx, dz = xs[i] - xs[j], zs[i] - zs[j]
            radial = dx * ux + dz * uz
            lateral = float(np.sqrt(max(dx * dx + dz * dz - radial * radial,
                                        0.0)))
            # Cross-row pairs (one NEAR, one FAR) get a wider tolerance to
            # absorb the head_to_ground vs foot-projection mismatch caused
            # by imperfect Y-scale calibration. See docstring.
            scale = (cross_row_factor if (is_near is not None
                                           and is_near[i] != is_near[j])
                     else 1.0)
            score = float(np.hypot(lateral / (lat_tol_m * scale),
                                   abs(radial) / (rad_tol_m * scale)))
            if score < 1.0:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out = []
    for members in groups.values():
        m = np.asarray(members)
        w = cs[m]
        # If this is a cross-row cluster (members in both NEAR and FAR rows),
        # one of the (X, Z) estimates is foot-projected (typically the
        # full-body NEAR detection) and the other is head-projected through
        # a possibly mis-scaled pose (typically the bot_cut FAR detection).
        # Averaging them puts the dot at their midpoint, which jitters
        # frame-to-frame as each tile's projection wobbles independently --
        # the tracker then fragments the track. Instead, pick the SINGLE
        # most-reliable detection (the one with the largest box height ->
        # most of the body visible -> most accurate foot anchor) and use
        # ITS (X, Z) for the cluster. Same-row clusters still use
        # confidence-weighted average (geometric jitter there is
        # negligible).
        cross_row = False
        if near_tile_count > 0:
            near_flags = [int(tiles[i] < near_tile_count) for i in members]
            cross_row = (0 in near_flags) and (1 in near_flags)
        if cross_row:
            heights = np.array([
                dets[i][4][3] - dets[i][4][1] for i in members], float)
            best = int(members[int(np.argmax(heights))])
            x_avg = float(dets[best][0])
            z_avg = float(dets[best][1])
        else:
            wsum = float(w.sum()) if w.sum() > 1e-9 else 1.0
            x_avg = float((xs[m] * w).sum() / wsum)
            z_avg = float((zs[m] * w).sum() / wsum)
        # Keep conf per contributing box so the JSON / overlay can label it.
        boxes = [(int(dets[i][3]), tuple(dets[i][4]), float(dets[i][2]))
                 for i in members]
        out.append((x_avg, z_avg, float(w.max()), boxes))
    return out


class Track:
    """A single tracked player. 4-state constant-velocity Kalman in (X, Z)."""

    def __init__(self, tid: int, x: float, z: float, dt: float,
                 q_accel: float = 3.0, r_pos: float = 0.8,
                 min_hits: int = 2):
        self.id = tid
        self.dt = dt
        self.min_hits = min_hits
        # state: [X, Z, Vx, Vz]
        self.state = np.array([x, z, 0.0, 0.0], float)
        # initial covariance: position confident, velocity unknown
        self.P = np.diag([r_pos ** 2, r_pos ** 2, 5.0 ** 2, 5.0 ** 2])
        self.F = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], float)
        # process noise (continuous white-noise acceleration)
        q = q_accel ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        self.Q = np.array([
            [dt4 / 4 * q, 0,            dt3 / 2 * q, 0],
            [0,           dt4 / 4 * q,  0,           dt3 / 2 * q],
            [dt3 / 2 * q, 0,            dt2 * q,     0],
            [0,           dt3 / 2 * q,  0,           dt2 * q],
        ])
        self.R = np.diag([r_pos ** 2, r_pos ** 2])
        self.hits: list[_Hit] = []
        self.hit_count = 0
        self.misses_since_hit = 0
        self.age = 0
        self.first_hit_frame = -1
        self.last_hit_frame = -1

    # ---- Kalman ------------------------------------------------------------
    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.misses_since_hit += 1

    def gating_distance(self, z_meas: np.ndarray) -> float:
        """Mahalanobis distance from predicted measurement to z_meas (2-vec)."""
        y = z_meas - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return float("inf")
        return float(np.sqrt(y @ S_inv @ y))

    def update(self, frame: int, x: float, z: float, conf: float,
               boxes: list):
        z_meas = np.array([x, z], float)
        y = z_meas - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.hits.append(_Hit(frame, float(x), float(z), float(conf), boxes))
        self.hit_count += 1
        self.misses_since_hit = 0
        self.last_hit_frame = frame
        if self.first_hit_frame < 0:
            self.first_hit_frame = frame

    # ---- introspection -----------------------------------------------------
    @property
    def position(self) -> tuple[float, float]:
        return float(self.state[0]), float(self.state[1])

    @property
    def confirmed(self) -> bool:
        return self.hit_count >= self.min_hits

    def pos_spread_m(self) -> float:
        """Median 2D distance from the median (X, Z) of all hits."""
        if len(self.hits) < 2:
            return float("inf")
        xs = np.array([h.x for h in self.hits])
        zs = np.array([h.z for h in self.hits])
        mx, mz = float(np.median(xs)), float(np.median(zs))
        return float(np.median(np.hypot(xs - mx, zs - mz)))

    def duration_s(self, fps: float) -> float:
        if self.first_hit_frame < 0:
            return 0.0
        return (self.last_hit_frame - self.first_hit_frame) / fps

    def is_stationary(self, max_spread_m: float, min_dur_s: float,
                      fps: float) -> bool:
        if max_spread_m <= 0.0 or min_dur_s <= 0.0:
            return False
        if self.duration_s(fps) < min_dur_s:
            return False
        return self.pos_spread_m() < max_spread_m

    # ---- chunk handoff (v0.12 #20b cross-chunk state) -----------------
    def to_dict(self) -> dict:
        """Serialize for cross-chunk handoff. dt/min_hits/q_accel/r_pos
        are passed at reconstruction time (they're Tracker-level
        constants); F/H/Q/R matrices are rebuilt from dt+constants too,
        so we only serialize the time-varying state."""
        return {
            "id": int(self.id),
            "state": [float(v) for v in self.state],
            "P": self.P.tolist(),
            "hits": [{"frame": h.frame, "x": h.x, "z": h.z,
                      "conf": h.conf,
                      "boxes": [list(b) if isinstance(b, tuple) else b
                                for b in h.boxes]}
                     for h in self.hits],
            "hit_count": int(self.hit_count),
            "misses_since_hit": int(self.misses_since_hit),
            "age": int(self.age),
            "first_hit_frame": int(self.first_hit_frame),
            "last_hit_frame": int(self.last_hit_frame),
        }

    @classmethod
    def from_dict(cls, d: dict, dt: float, q_accel: float, r_pos: float,
                   min_hits: int) -> "Track":
        # Bypass __init__ -- rebuild the time-invariant pieces ourselves
        # and load the time-varying state from d.
        t = cls.__new__(cls)
        t.id = int(d["id"])
        t.dt = float(dt)
        t.min_hits = int(min_hits)
        t.state = np.array(d["state"], float)
        t.P = np.array(d["P"], float)
        t.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], float)
        t.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], float)
        q = q_accel ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        t.Q = np.array([
            [dt4 / 4 * q, 0,            dt3 / 2 * q, 0],
            [0,           dt4 / 4 * q,  0,           dt3 / 2 * q],
            [dt3 / 2 * q, 0,            dt2 * q,     0],
            [0,           dt3 / 2 * q,  0,           dt2 * q],
        ])
        t.R = np.diag([r_pos ** 2, r_pos ** 2])
        t.hits = [_Hit(frame=h["frame"], x=h["x"], z=h["z"],
                       conf=h["conf"],
                       boxes=[tuple(b) if isinstance(b, list) else b
                              for b in h["boxes"]])
                  for h in d["hits"]]
        t.hit_count = int(d["hit_count"])
        t.misses_since_hit = int(d["misses_since_hit"])
        t.age = int(d["age"])
        t.first_hit_frame = int(d["first_hit_frame"])
        t.last_hit_frame = int(d["last_hit_frame"])
        return t


class Tracker:
    """Manages a list of Tracks, running one detect-fuse-associate-update step
    per perception frame.

    `dt` is the time between perception frames (stride / fps), not 1/fps.
    """

    def __init__(self, *, dt: float, fps: float,
                 cam_xz: tuple[float, float] = (0.0, 0.0),
                 fuse_lat_m: float = 0.6, fuse_rad_m: float = 2.5,
                 gate_mahal: float = 4.0,
                 max_coast_s: float = 0.3, min_hits: int = 2,
                 stationary_pos_spread_m: float = 0.5,
                 stationary_min_duration_s: float = 5.0,
                 phantom_window_s: float = 0.0,
                 phantom_max_spread_m: float = 0.5,
                 phantom_max_tiles: int = 1,
                 near_tile_count: int = 0,
                 cross_row_factor: float = 2.0,
                 q_accel: float = 3.0, r_pos: float = 0.8):
        self.dt = dt
        self.fps = fps
        self.cam_xz = cam_xz
        self.fuse_lat_m = fuse_lat_m
        self.fuse_rad_m = fuse_rad_m
        self.gate_mahal = gate_mahal
        # max_coast_frames is in *source* frames (output is densified to fps).
        self.max_coast_frames = max(0, int(round(max_coast_s * fps)))
        self.min_hits = min_hits
        self.stat_spread_m = stationary_pos_spread_m
        self.stat_min_dur_s = stationary_min_duration_s
        # Per-frame phantom-segment filter (opt-in via phantom_window_s > 0).
        # At each emit frame, look at real-detection hits within ±window
        # seconds. If they are all single-tile (max unique tiles per hit <=
        # phantom_max_tiles) AND positionally tight (median radial spread <
        # phantom_max_spread_m), suppress the dot at this frame. Catches
        # stationary YOLO false positives that survive on a fixed pano
        # pixel without multi-tile corroboration, including the
        # ID-hijacked-into-phantom segments of otherwise-real tracks (which
        # the whole-track stationary filter misses because the lifetime
        # max-tiles or lifetime spread looks legitimate).
        self.phantom_window_s = phantom_window_s
        self.phantom_max_spread_m = phantom_max_spread_m
        self.phantom_max_tiles = phantom_max_tiles
        self.near_tile_count = near_tile_count
        self.cross_row_factor = cross_row_factor
        self.q_accel = q_accel
        self.r_pos = r_pos
        # A track dies when it has been missing for longer than the coast cap
        # plus a small grace, so a track that's about to be re-acquired still
        # lives across the gap.
        self.max_age_frames = max(
            int(round(2 * max_coast_s * fps / max(1, dt * fps))),
            int(round(1.0 * fps / max(1, dt * fps))))
        self.tracks: list[Track] = []
        # All tracks that have ever existed (for emit; live + culled).
        self.history: list[Track] = []
        self._next_id = 0

    def step(self, frame: int, raw_dets):
        """One perception step. `raw_dets` = list of (X, Z, conf, tile, box).

        Order: fuse -> predict all tracks -> Hungarian-associate -> update
        matched / spawn unmatched -> cull tracks coasting past max_age.
        """
        fused = fuse_detections(raw_dets, self.cam_xz,
                                self.fuse_lat_m, self.fuse_rad_m,
                                near_tile_count=self.near_tile_count,
                                cross_row_factor=self.cross_row_factor)

        for t in self.tracks:
            t.predict()

        matched_t: set[int] = set()
        matched_d: set[int] = set()
        if self.tracks and fused:
            M, N = len(self.tracks), len(fused)
            cost = np.full((M, N), 1e6, float)
            for i, t in enumerate(self.tracks):
                for j, (x, z, _c, _b) in enumerate(fused):
                    d = t.gating_distance(np.array([x, z], float))
                    if d <= self.gate_mahal:
                        cost[i, j] = d
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < 1e6:
                    x, z, c, boxes = fused[j]
                    self.tracks[i].update(frame, x, z, c, boxes)
                    matched_t.add(int(i))
                    matched_d.add(int(j))

        for j, (x, z, c, boxes) in enumerate(fused):
            if j in matched_d:
                continue
            t = Track(self._next_id, x, z, self.dt,
                      self.q_accel, self.r_pos, self.min_hits)
            t.update(frame, x, z, c, boxes)
            self.tracks.append(t)
            self.history.append(t)
            self._next_id += 1

        # Cull tracks that have coasted too long; they stay in history for
        # potential emit (their hits remain valid; only future updates die).
        kept = []
        for t in self.tracks:
            if t.misses_since_hit <= self.max_age_frames:
                kept.append(t)
        self.tracks = kept

    # ---- chunk handoff (v0.12 #20b cross-chunk state) -----------------
    def get_state(self) -> dict:
        """Snapshot of currently-active tracks + ID counter, suitable
        for hand-off to the next chunk. History (dead tracks) is NOT
        serialized -- those have already been emitted by this chunk's
        output and don't need to continue."""
        return {
            "tracks": [t.to_dict() for t in self.tracks],
            "next_id": int(self._next_id),
            "dt": float(self.dt),
        }

    def load_state(self, state: dict):
        """Restore active tracks + ID counter from get_state() output.
        Restored tracks go into BOTH `self.tracks` (for continuing
        step() updates) AND `self.history` (for emit_per_frame to
        include any NEW hits added during this chunk -- old hits from
        previous chunks are filtered out by emit's f0/f1 range)."""
        self.tracks = [
            Track.from_dict(t, self.dt, self.q_accel, self.r_pos,
                            self.min_hits)
            for t in state.get("tracks", [])
        ]
        self.history.extend(self.tracks)
        self._next_id = int(state.get("next_id", 0))

    def emit_per_frame(self, f0: int, f1: int) -> dict[int, list[dict]]:
        """Densify all confirmed, non-stationary tracks to every source frame.

        Linear interpolation between the track's real-detection hits; output
        frames whose distance to the nearest real-detection hit exceeds
        `max_coast_frames` are dropped (native coast cap).
        """
        out: dict[int, list[dict]] = {f: [] for f in range(f0, f1 + 1)}
        n_dropped_stationary = 0
        n_dropped_short = 0
        n_suppressed_phantom_frames = 0
        # Phantom-segment filter pre-compute: only active when window > 0.
        phantom_on = (self.phantom_window_s > 0.0
                      and self.phantom_max_spread_m > 0.0)
        phantom_win_frames = int(round(self.phantom_window_s * self.fps))
        for t in self.history:
            if t.hit_count < self.min_hits:
                n_dropped_short += 1
                continue
            if t.is_stationary(self.stat_spread_m, self.stat_min_dur_s,
                               self.fps):
                n_dropped_stationary += 1
                continue
            fr = np.array([h.frame for h in t.hits])
            xs = np.array([h.x for h in t.hits])
            zs = np.array([h.z for h in t.hits])
            cs = np.array([h.conf for h in t.hits])
            # Unique tiles per real-detection hit, for the phantom segment
            # check below. 0 = no boxes stored on this hit.
            tiles_per_hit = (np.array([len({b[0] for b in h.boxes})
                                        if h.boxes else 0 for h in t.hits])
                              if phantom_on else None)
            boxes_by_f = {int(h.frame): h.boxes for h in t.hits if h.boxes}
            a, b = int(fr[0]), int(fr[-1])
            if a == b:
                gf = np.array([a])
                gx = xs.copy()
                gz = zs.copy()
                gc = cs.copy()
            else:
                gf = np.arange(a, b + 1)
                gx = np.interp(gf, fr, xs)
                gz = np.interp(gf, fr, zs)
                gc = np.interp(gf, fr, cs)
            if self.max_coast_frames > 0 and len(fr) > 0:
                # Distance from each gf to nearest value in fr (fr sorted).
                idx = np.searchsorted(fr, gf)
                left = np.clip(idx - 1, 0, len(fr) - 1)
                right = np.clip(idx, 0, len(fr) - 1)
                dist = np.minimum(np.abs(gf - fr[left]),
                                  np.abs(gf - fr[right]))
                keep = dist <= self.max_coast_frames
            else:
                keep = np.ones(len(gf), dtype=bool)
            for f, x, z, c, k in zip(gf, gx, gz, gc, keep):
                if not k:
                    continue
                fi = int(f)
                if fi < f0 or fi > f1:
                    continue
                # Phantom-segment filter: look at real-detection hits within
                # ±phantom_window of this emit frame. If all those hits
                # were single-tile (<= phantom_max_tiles) AND positionally
                # tight (< phantom_max_spread_m), this frame is in a
                # phantom segment -- skip the dot.
                if phantom_on:
                    win = np.abs(fr - fi) <= phantom_win_frames
                    if win.sum() >= 2:
                        win_tiles_max = int(tiles_per_hit[win].max())
                        if 0 < win_tiles_max <= self.phantom_max_tiles:
                            wx = xs[win]; wz = zs[win]
                            wmx, wmz = float(np.median(wx)), float(np.median(wz))
                            win_spread = float(np.median(
                                np.hypot(wx - wmx, wz - wmz)))
                            if win_spread < self.phantom_max_spread_m:
                                n_suppressed_phantom_frames += 1
                                continue
                entry = {"id": int(t.id),
                         "X": round(float(x), 3),
                         "Z": round(float(z), 3),
                         "conf": round(float(c), 3)}
                bx = boxes_by_f.get(fi)
                if bx:
                    entry["boxes"] = [
                        {"tile": int(ti),
                         "xyxy": [int(v) for v in box],
                         "conf": round(float(bc), 3)}
                        for ti, box, bc in bx]
                out[fi].append(entry)
        self._emit_stats = {
            "n_history": len(self.history),
            "n_dropped_short": n_dropped_short,
            "n_dropped_stationary": n_dropped_stationary,
            "n_suppressed_phantom_frames": n_suppressed_phantom_frames,
            "n_emitted": len(self.history) - n_dropped_short
                         - n_dropped_stationary,
        }
        return out
