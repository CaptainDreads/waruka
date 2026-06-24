# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""On-field vs sideline classification (per-track, behavioural, offline).

Sideline loiterers often stand literally on the line, so a spatial test
cannot separate them from real players. Instead we judge each track over its
whole history: a real player has a sustained on-field presence and/or fast
play episodes (a pre-point lineup player stands still, then sprints the pull
-- their track penetrates the field, so they classify as a player). A
loiterer stays in the near-sideline band, moves slowly, and drifts in and
out without ever committing to the field.

Bias is toward inclusion (capture every actively-playing player plus space);
the strongest filter is adjacency exclusion of other games.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np

from .config import ProjectConfig
from .ground import GroundModel


@dataclass
class TrackFeatures:
    track_id: int
    n: int
    dur_s: float
    p85_speed: float
    z_med: float
    z_max: float
    x_span: float
    frac_on: float          # fraction of life inside field (+buffer)
    frac_deep: float        # fraction with Z in [deep, W-deep]
    max_deep_run_s: float   # longest CONSECUTIVE stretch in the deep zone
    total_deep_s: float     # CUMULATIVE time spent in the deep zone (Z in
                            # [deep_m, W-deep_m]); used by the Schmitt
                            # committed-grace gate. Cumulative rather than
                            # consecutive: a player who passes through the
                            # middle of the field repeatedly during a point
                            # accumulates eligibility for the sideline grace.
    pos_spread: float       # ROBUST spread (median dist from median pos), m;
                            # ~0 = fixed object. Robust to ID-switch teleports
    pos_std: float          # std of position (m); captures real excursions
                            # (a near-line player who cuts deep) for commitment
    med_in_field: bool
    label: str = "player"   # or "sideline" / "foreign"


def _tracks_by_id(frames, fps, stride):
    series: dict[int, list] = {}
    for fr in frames:
        for p in fr["players"]:
            series.setdefault(p["id"], []).append(
                (fr["frame"], p["X"], p["Z"]))
    dt = stride / fps
    return series, dt


def _smooth(a, k=5):
    if len(a) < k:
        return a
    ker = np.ones(k) / k
    return np.convolve(a, ker, mode="same")


def classify_tracks(
    tracks_json: str,
    project: str = "project.json",
    buffer_m: float = 1.0,
    deep_m: float = 5.0,
    min_deep_run_s: float = 2.0,
    sideline_band_m: float = 5.0,
    min_deep_run_sideline_s: float = 7.0,
    static_min_s: float = 6.0,
    min_move_m: float = 0.5,
    far_static_spread_m: float = 2.0,
    commit_move_m: float = 2.5,
    active_margin_near: float = -0.5,
    active_margin_far: float = -2.0,
    active_margin_ends: float = 2.0,
    active_hyst_band_m: float = 1.5,
    # Per-direction off-hold times. Sideline exits flip fast (subs and
    # spectators walk in/out constantly). Endzone exits get a long grace
    # because real off-field traffic almost never goes past the back lines
    # -- a player past the back endzone is most likely retrieving a disc
    # or about to come back. If a track exits past multiple boundaries
    # simultaneously (e.g. a corner exit), the LONGEST applicable hold is
    # used.
    active_off_hold_near_s: float = 0.3,
    active_off_hold_far_s: float = 0.3,
    active_off_hold_ends_s: float = 10.0,
    # Extra grace at the sidelines for COMMITTED tracks -- those whose
    # cumulative time spent in the deep zone (Z in [deep_m, W-deep_m])
    # exceeds committed_min_deep_s. Sustained presence in the middle of
    # the field is the "real player" signal -- a sideline walker who
    # briefly steps in for a sub touch won't accumulate enough deep-zone
    # time; a real player who plays the point through will. Field marks
    # are imperfect (especially at the far sideline due to projection
    # wobble + small angular size of far players); a clearly-committed
    # track shouldn't flip inactive on a brief boundary slip. Composes
    # with the per-direction hold via MAX, like the corner-exit case.
    # Does NOT apply at endzones (already 10 s there).
    # Master toggle for the committed-grace gate (#27). When True
    # (default), tracks that have spent committed_min_deep_s of the last
    # committed_recent_window_s in the deep zone get extended sideline
    # off-hold (committed_off_hold_s). When False, sideline boundaries
    # use only the per-direction holds from #26 -- equivalent to the
    # pre-#27 behaviour. Useful for A/B comparing or when the grace
    # isn't worth the small ID-switch bleed risk.
    committed_grace_enabled: bool = True,
    committed_off_hold_s: float = 7.0,
    # Must be <= committed_recent_window_s or the criterion is impossible
    # to satisfy. Default ~60% of the 5 s window.
    committed_min_deep_s: float = 3.0,
    # Rolling window over which deep-zone time is accumulated for the
    # committed check. Using lifetime total_deep_s caused tracker
    # ID-switches to bleed credentials onto sideline players: a real
    # player's track ID can transfer to a nearby (sideline) detection,
    # and the post-switch dot would inherit the pre-switch deep history
    # and get committed-grace. With a rolling window, the committed
    # status decays after the switch because the sideline dot's recent
    # position doesn't accumulate deep time. 5 s is aggressive: a real
    # walk-off still gets a few seconds of grace; a sideline dot loses
    # committed status within ~2-3 s of an ID-switch.
    committed_recent_window_s: float = 5.0,
    probation_s: float = 3.0,
    isolated_dist_m: float = 8.0,
    well_inside_margin_near: float = -1.0,
    well_inside_margin_far: float = -3.0,
    well_inside_margin_ends: float = 2.0,
    out_path: str = "players.json",
) -> dict:
    data = json.load(open(tracks_json))
    cfg = ProjectConfig.load(project)
    gm = GroundModel(cfg.homography, cfg.field_length_m, cfg.field_width_m)
    L, W = cfg.field_length_m, cfg.field_width_m
    series, dt = _tracks_by_id(data["frames"], data["fps"], data["stride"])

    feats: dict[int, TrackFeatures] = {}
    for tid, pts in series.items():
        arr = np.array(pts, float)  # (n, [frame,X,Z])
        n = len(arr)
        X, Z = _smooth(arr[:, 1]), _smooth(arr[:, 2])
        inb = gm.in_field(np.column_stack([X, Z]), margin_m=buffer_m)
        deep = (Z >= deep_m) & (Z <= W - deep_m) & (np.abs(X) <= L / 2)
        # Longest *consecutive* deep stretch (seconds). Distinguishes a real
        # player who lived inside the field for a sustained window from a
        # sideline stander whose noisy Z occasionally spikes deep -- on long
        # tracks frac_deep*dur (the old "commit" feature) accumulates linearly
        # with track length and trips the threshold on pure noise.
        max_run = 0
        run = 0
        for d in deep:
            if d:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        max_deep_run_s = max_run * dt
        # Cumulative time in deep zone -- a track that passes through the
        # middle of the field multiple times during the point accumulates
        # eligibility for the committed-grace gate even if no single pass
        # is long enough on its own.
        total_deep_s = float(deep.sum()) * dt

        if n >= 2:
            sp = np.hypot(np.diff(X), np.diff(Z)) / max(dt, 1e-6)
            p85 = float(np.percentile(sp, 85))
        else:
            p85 = 0.0
        med_in = bool(gm.in_field(np.array([[np.median(X), np.median(Z)]]),
                                  margin_m=buffer_m)[0])
        # Robust spatial spread: median distance from the track's median
        # position. Unlike std, this ignores the occasional ID-switch
        # "teleport" (a phantom pinned to a fixed spot but momentarily jumping
        # to a far detection and back), so a genuinely stationary track still
        # reads as stationary even when a few outlier frames fling its std up.
        medX, medZ = float(np.median(X)), float(np.median(Z))
        pos_spread = float(np.median(np.hypot(X - medX, Z - medZ)))
        pos_std = float(np.hypot(X.std(), Z.std()))
        f = TrackFeatures(
            track_id=tid, n=n, dur_s=round(n * dt, 2), p85_speed=round(p85, 2),
            z_med=round(float(np.median(Z)), 2), z_max=round(float(Z.max()), 2),
            x_span=round(float(X.max() - X.min()), 2),
            frac_on=round(float(inb.mean()), 2),
            frac_deep=round(float(deep.mean()), 2),
            max_deep_run_s=round(max_deep_run_s, 2),
            total_deep_s=round(total_deep_s, 2),
            pos_spread=round(pos_spread, 2),
            pos_std=round(pos_std, 2),
            med_in_field=med_in)

        # A track anchored near either sideline (z_med close to 0 or W)
        # needs a *much longer* sustained deep run to be a "player" -- it's
        # most likely a sub/spectator who briefly stepped onto the playing
        # area. A track whose median position is well inside the field can
        # qualify on the shorter (default 2 s) threshold. This is symmetric
        # near/far and protects against far-sideline false-positives where
        # the camera can't easily resolve whether someone is on the line
        # or playing.
        z_med_val = f.z_med
        near_a_sideline = (z_med_val < sideline_band_m or
                           z_med_val > W - sideline_band_m)
        # The strict (long) deep-run requirement targets loiterers anchored
        # near a line whose noisy Z occasionally reads deep. But a real
        # handler who plays *near* the line still moves -- cuts, shuffles,
        # resets -- so it has a high pos_rms. Only apply the strict
        # requirement to near-sideline tracks that also barely move; a moving
        # near-line track is a player on the normal threshold. (Without this,
        # a short, fast track whose *median* sits just inside the band -- e.g.
        # a reset handler who then cuts deep -- is wrongly dropped.)
        committed = pos_std >= commit_move_m
        required = (min_deep_run_sideline_s
                    if (near_a_sideline and not committed)
                    else min_deep_run_s)
        # Stationarity veto: a track pinned to a fixed spot for a sustained
        # window is not actively playing -- it's a fixed-object false
        # positive (kit/cone), a sitting spectator, or a motionless stander
        # whose noisy Z happens to read "deep". The deep-run test alone can't
        # tell them apart (it checks position, never motion), so a static
        # blob in the deep zone trivially earns a long deep run.
        # Far-sideline tracks use a looser spread threshold: camera wobble +
        # the small angular size of distant players means even genuinely
        # stationary far-side objects read with 1-2 m of pos_spread. Using
        # the global 1.0 m threshold there lets too many fixed phantoms
        # through; the far_static_spread_m (default 2.0 m) catches them.
        in_far_band = z_med_val > W - sideline_band_m
        static_thr = far_static_spread_m if in_far_band else min_move_m
        is_static = f.dur_s > static_min_s and pos_spread < static_thr
        if (not med_in and f.frac_on < 0.4
                and max_deep_run_s < min_deep_run_s):
            f.label = "foreign"          # other field / spectator
        elif is_static:
            f.label = "sideline"         # fixed object / motionless stander
        elif max_deep_run_s >= required:
            f.label = "player"           # sustained on-field presence
        else:
            f.label = "sideline"         # never committed long enough
        feats[tid] = f

    players = {tid for tid, f in feats.items() if f.label == "player"}
    lab = {tid: f.label for tid, f in feats.items()}

    # Per-frame "active" gate as a Schmitt trigger (hysteresis band): a player
    # ACTIVATES when clearly in-field (inner margins, tolerating a small line
    # overrun) and only DEACTIVATES after sitting clearly OUTSIDE a larger
    # boundary (inner margin + hysteresis band) for a sustained hold time.
    # Between the two boundaries the current state is held. This stops a player
    # parked right on a line -- whose noisy projected position jitters across
    # the boundary -- from flickering active/inactive every few frames, while
    # still dropping someone who genuinely walks well off and stays there.
    def _to_frames(s: float) -> int:
        return max(0, int(round(s / max(dt, 1e-6))))
    hold_near = _to_frames(active_off_hold_near_s)
    hold_far  = _to_frames(active_off_hold_far_s)
    hold_ends = _to_frames(active_off_hold_ends_s)
    hold_committed = _to_frames(committed_off_hold_s)
    active_set = set()
    well_inside_set = set()  # used by the probation gate (stricter zone)
    win_frames = max(1, int(round(committed_recent_window_s / dt)))
    for tid in players:
        pts = sorted(series[tid])                  # (frame, X, Z)
        P = np.array([[x, z] for _, x, z in pts], float)
        n_frames = len(P)
        if committed_grace_enabled:
            # Per-frame "committed" flag based on cumulative deep time
            # within the last committed_recent_window_s seconds. A track
            # is committed at frame i if it has spent at least
            # committed_min_deep_s in the deep zone (middle of the field
            # width) within that window. Sliding-window cumsum trick:
            # O(n). The status decays automatically after an ID-switch
            # because the post-switch dot doesn't accumulate deep time
            # at its new (sideline) position.
            Z_arr = P[:, 1]
            X_arr = P[:, 0]
            deep_pf = ((Z_arr >= deep_m) & (Z_arr <= W - deep_m)
                       & (np.abs(X_arr) <= L / 2))
            cumdeep = np.concatenate([[0],
                                       np.cumsum(deep_pf.astype(int))])
            los = np.maximum(0, np.arange(n_frames) + 1 - win_frames)
            recent_deep_s = (cumdeep[1:] - cumdeep[los]) * dt
            committed_pf = recent_deep_s >= committed_min_deep_s
        else:
            committed_pf = np.zeros(n_frames, dtype=bool)
        act_in = gm.in_field(P, margin_near=active_margin_near,
                             margin_far=active_margin_far,
                             margin_ends=active_margin_ends)
        clear_off = ~gm.in_field(
            P, margin_near=active_margin_near + active_hyst_band_m,
            margin_far=active_margin_far + active_hyst_band_m,
            margin_ends=active_margin_ends + active_hyst_band_m)
        well_in = gm.in_field(P, margin_near=well_inside_margin_near,
                              margin_far=well_inside_margin_far,
                              margin_ends=well_inside_margin_ends)
        active = False
        off_run = 0
        for idx, ((frm, x, z), ai, co, wi) in enumerate(
                zip(pts, act_in, clear_off, well_in)):
            if ai:
                active = True
                off_run = 0
            elif co:
                off_run += 1
                # Pick the longest applicable hold based on which boundary(s)
                # the dot is past at this frame. The activation-boundary
                # checks (no hyst added) are the right test here: clear_off
                # being True already implies the dot is past at least one
                # of these. The MAX semantics mean a corner exit gets the
                # endzone grace, and a player wandering from endzone-exit
                # into past-sideline mid-grace flips fast (they're now
                # acting like a sideline player).
                past_near = z < -active_margin_near
                past_far  = z > W + active_margin_far
                past_ends = abs(x) > L / 2 + active_margin_ends
                applicable = []
                if past_near: applicable.append(hold_near)
                if past_far:  applicable.append(hold_far)
                if past_ends: applicable.append(hold_ends)
                # Committed-player extended grace applies only at sidelines
                # (endzones already get the longer hold_ends). Per-FRAME
                # committed: the rolling-window recent_deep_s decays after
                # an ID-switch, so a sideline dot that inherited a real
                # player's track ID loses grace within ~2-3 s.
                if committed_pf[idx] and (past_near or past_far):
                    applicable.append(hold_committed)
                hold = max(applicable) if applicable else hold_near
                if off_run >= hold:
                    active = False
            else:
                off_run = 0                        # in hysteresis band: hold
            if active:
                active_set.add((tid, frm))
                if wi:
                    well_inside_set.add((tid, frm))

    # Probation gate (on top of Schmitt active flag): a "fresh" activation
    # (no active state in the previous frame for this track) that's ALSO
    # ISOLATED (far from the existing stable-active cluster) must stay
    # continuously active for probation_s before being promoted to the
    # framing pool. Catches sideline players briefly stepping onto the field
    # (their dot flashes inside but they're far from the action and disappear
    # again before probation elapses). Mid-cluster ID-switches don't trigger
    # probation -- the new track appears next to teammates and is promoted
    # immediately. A continuously-tracked sprinter is never "fresh" because
    # they're active every frame, so probation does NOT interfere with real
    # play. The "stable_set" is the framing-pool subset of active_set.
    prob_frames = max(0, int(round(probation_s / max(dt, 1e-6))))

    # Build per-frame position lookup for player-labelled tracks only.
    frame_player_xz: dict[int, list[tuple[int, float, float]]] = {}
    for tid in players:
        for frm, x, z in series[tid]:
            frame_player_xz.setdefault(frm, []).append((tid, x, z))
    sorted_frames = sorted(frame_player_xz)

    # Per-track walk state for the probation gate.
    track_state: dict[int, dict] = {tid: {
        "prev_active": False,
        "prev_xz": None,
        "run_start": None,    # frame index when current active run began
        "promoted": False,    # True once this run cleared probation
        "well_inside_count": 0,  # cumulative well-inside frames this run
    } for tid in players}

    stable_set: set = set()

    for frm in sorted_frames:
        # Two passes per frame so isolation is checked against tracks already
        # confirmed stable in PRIOR frames (avoids new-isolated-tracks
        # reinforcing each other into immediate promotion).
        candidates = []
        for tid, x, z in frame_player_xz[frm]:
            s = track_state[tid]
            is_act = (tid, frm) in active_set
            if not is_act:
                s["prev_active"] = False
                s["prev_xz"] = (x, z)
                s["run_start"] = None
                s["promoted"] = False
                s["well_inside_count"] = 0
                continue
            fresh = not s["prev_active"]
            candidates.append((tid, x, z, fresh, s))

        # Cluster reference = stable-active tracks this frame that were
        # already promoted in a prior frame (NOT freshly activated this
        # frame). This is what a sideline-flash sees when checking isolation.
        stable_xz = [(x, z) for tid, x, z, fresh, s in candidates
                     if (not fresh) and s["promoted"]]

        for tid, x, z, fresh, s in candidates:
            # Accumulate well-inside time for this active run (used as the
            # probation-promotion criterion: a track must spend the
            # probation_s budget in the strict well_inside zone, not just
            # the Schmitt activation zone, before being promoted).
            if (tid, frm) in well_inside_set:
                s["well_inside_count"] += 1
            if fresh:
                if stable_xz:
                    dmin = min(np.hypot(x - sx, z - sz)
                               for sx, sz in stable_xz)
                else:
                    dmin = float("inf")
                s["run_start"] = frm
                # If activation point is near the cluster AND already well
                # inside, promote immediately. Otherwise probationary.
                s["promoted"] = (dmin <= isolated_dist_m
                                 and (tid, frm) in well_inside_set)
            else:
                # Continuing run: check if probation has now elapsed.
                if not s["promoted"] and s["well_inside_count"] >= prob_frames:
                    s["promoted"] = True
            s["prev_active"] = True
            s["prev_xz"] = (x, z)
            if s["promoted"]:
                stable_set.add((tid, frm))

    out_frames, lab_frames = [], []
    for fr in data["frames"]:
        f_no = fr["frame"]
        out_frames.append({"frame": f_no, "t": fr["t"],
                           "players": [p for p in fr["players"]
                                       if p["id"] in players
                                       and (p["id"], f_no) in stable_set]})
        lp = []
        for p in fr["players"]:
            base = lab.get(p["id"], "player")
            if base != "player":
                label = base
            else:
                in_active = (p["id"], f_no) in active_set
                in_stable = (p["id"], f_no) in stable_set
                if in_stable:
                    label = "player"
                elif in_active:
                    label = "probation"  # active per Schmitt but probationary
                else:
                    label = "sideline"   # active player, currently off-field
            lp.append({**p, "label": label})
        lab_frames.append({"frame": f_no, "t": fr["t"], "players": lp})
    meta = {k: data[k] for k in ("video", "fps", "stride",
                                 "field_length_m", "field_width_m")}
    json.dump({**meta, "frames": out_frames}, open(out_path, "w"))
    labeled_path = out_path.replace(".json", "_labeled.json")
    json.dump({**meta, "frames": lab_frames}, open(labeled_path, "w"))

    counts = {}
    for f in feats.values():
        counts[f.label] = counts.get(f.label, 0) + 1
    print(f"tracks={len(feats)} -> {counts}  wrote {out_path}")
    return {"features": {k: asdict(v) for k, v in feats.items()},
            "counts": counts, "out": out_path}


def render_class_overlay(project: str, tracks_json: str, classification: dict,
                         t_seconds: float, out_path: str):
    import cv2
    cfg = ProjectConfig.load(project)
    pano = cfg.pano
    gm = GroundModel(cfg.homography, cfg.field_length_m, cfg.field_width_m)
    data = json.load(open(tracks_json))
    feats = classification["features"]
    fidx = int(t_seconds * data["fps"])
    rec = min(data["frames"], key=lambda f: abs(f["frame"] - fidx))
    cap = cv2.VideoCapture(data["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, rec["frame"])
    ok, frame = cap.read()
    cap.release()
    b = gm.boundary_xz(1.0)
    bx, by = pano.directions_to_src(gm.ray_from_ground(b))
    cv2.polylines(frame, [np.column_stack([bx, by]).astype(np.int32)],
                  True, (0, 200, 255), 2, cv2.LINE_AA)
    col = {"player": (0, 220, 0), "sideline": (0, 0, 255),
           "foreign": (255, 0, 255)}
    def _lab(pid):
        return feats.get(str(pid), feats.get(pid, {})).get("label", "player")

    for p in rec["players"]:
        lab = _lab(p["id"])
        d = gm.ray_from_ground(np.array([[p["X"], p["Z"]]]))
        sx, sy = pano.directions_to_src(d)
        c = (int(sx[0]), int(sy[0]))
        cv2.circle(frame, c, 8, col[lab], 2)
        cv2.putText(frame, f'{p["id"]}{lab[0]}', (c[0] + 6, c[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col[lab], 2, cv2.LINE_AA)
    h, w = frame.shape[:2]
    cor = np.array(cfg.field_marks["corners"])
    x0, y0 = int(cor[:, 0].min() - 80), int(cor[:, 1].min() - 80)
    x1, y1 = int(cor[:, 0].max() + 80), int(cor[:, 1].max() + 90)
    crop = frame[max(0, y0):y1, max(0, x0):x1]
    ch, cw = crop.shape[:2]
    segs = [crop[:, i * cw // 3:(i + 1) * cw // 3] for i in range(3)]
    mw = max(s.shape[1] for s in segs)
    segs = [cv2.copyMakeBorder(s, 0, 0, 0, mw - s.shape[1],
                               cv2.BORDER_CONSTANT) for s in segs]
    cv2.imwrite(out_path, cv2.resize(np.vstack(segs), None, fx=1.5, fy=1.5))
    kept = sum(1 for p in rec["players"] if _lab(p["id"]) == "player")
    print(f"{out_path}: t={rec['t']}s players_kept={kept}/{len(rec['players'])}")
