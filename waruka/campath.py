# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Virtual camera-path planner (cylindrical follow-the-action framing).

A sideline-mid camera cannot enclose every on-field player in one natural
rectilinear frame (near players sit at ~+/-90deg). So the output is rendered
*cylindrically* (linear azimuth, straight verticals) and the path just needs:

  * center yaw  = robust centre bearing of the on-field players
  * hfov        = robust horizontal angular span (5..95 pct, ignores a stray
                  near-camera player) + angular margin, clamped to a sane band
  * pitch       = players' vertical centre

then a bounded look-ahead + critically-damped smoothing (jitter/lag control,
live-ready since the window is bounded, not whole-video).
"""

from __future__ import annotations

import json

import numpy as np

from .config import ProjectConfig
from .ground import GroundModel


def _targets(rays: np.ndarray, margin_deg: float):
    """(center_yaw, pitch, hfov) deg enclosing ALL given player rays.

    Framing must contain every classifier-labeled player on the field -- a
    cut into deep space or an end-zone stander on the far X must not be
    trimmed out of the broadcast frame. Previously this used a 18-82 pct
    azimuth band that robustly tracked the bulk but would silently drop
    edge players. With duplicates / classifier noise present upstream the
    output may need to be wide (and letterboxed) -- that's an acceptable
    tradeoff against losing a real player.
    """
    az = np.degrees(np.arctan2(rays[:, 0], rays[:, 2]))
    el = np.degrees(np.arcsin(np.clip(-rays[:, 1], -1, 1)))
    lo, hi = float(az.min()), float(az.max())
    center = 0.5 * (lo + hi)
    hfov = (hi - lo) + 2.0 * margin_deg
    pitch = float(np.median(el))
    return center, pitch, hfov


def _smooth_signal(target, dt, smooth_t, v_max, a_max, deadzone,
                    soft_deadzone: bool = True,
                    init_pos: float | None = None,
                    init_vel: float = 0.0):
    """Critically-damped follow with velocity/accel caps and a deadzone.

    The deadzone is the per-axis range around the current smoothed
    position within which response is suppressed -- intended to reject
    per-frame jitter without moving the camera. Two modes:

    * soft (default, recommended): the spring force is scaled
      quadratically from 0 at diff=0 to full at diff=deadzone, then
      stays full beyond. Response is continuous -- the camera always
      moves toward the target, just slower for small diffs. A constant
      drift in the target (e.g. a smooth pan during sustained play)
      produces a smooth constant-velocity camera response.
    * hard (legacy, soft_deadzone=False): if the target is within
      `deadzone` of current position, the target is snapped to current
      (spring force killed). Outside the deadzone, full spring force.
      Discontinuous response at the boundary causes visible
      "stop-start" stair-step motion during slow continuous pans:
      camera coasts to a stop inside the deadzone, then snaps when the
      target drifts outside, then stops, then snaps again. Useful as
      a sanity-check / A-B comparison default; soft_deadzone=True is
      the production default.

    `init_pos` / `init_vel` initialize the smoother. Used by the
    chunked pipeline (#20b) to hand off state across chunk boundaries
    so the camera bridges smoothly between adjacent campaths. When
    None (default), starts at target[0] with zero velocity -- the
    legacy single-pass behaviour. Returns (out, final_pos, final_vel)
    so the caller can chain.
    """
    n = len(target)
    out = np.empty(n)
    pos = float(target[0]) if init_pos is None else float(init_pos)
    vel = float(init_vel)
    w = 2.0 / max(smooth_t, 1e-3)
    for i in range(n):
        tgt = target[i]
        diff = tgt - pos
        if deadzone > 0 and abs(diff) < deadzone:
            if soft_deadzone:
                # Quadratic taper: scale spring force by (|diff|/deadzone)^2
                scale = (abs(diff) / deadzone) ** 2
                # Equivalent to using a scaled effective diff in the spring
                # force calculation.
                spring = w * w * diff * scale
            else:
                spring = 0.0  # hard deadzone
        else:
            spring = w * w * diff
        a = np.clip(spring - 2.0 * w * vel, -a_max, a_max)
        vel = np.clip(vel + a * dt, -v_max, v_max)
        pos += vel * dt
        out[i] = pos
    return out, pos, vel


_VIEW_PRESETS = {
    # Default: no extra floor, no extra margin. Camera hugs the
    # natural cluster width. (Reverted from #15-era hfov_min=50 in a
    # later iteration once Panini reduced the perceived "too tight"
    # feel on small clusters.)
    "default": {"hfov_min": 26.0, "margin_deg": 8.0},
    # Wide (#15 option B): same low floor, but bump the per-side
    # margin to 15 deg so every moment has ~14 deg more breathing
    # room around the cluster than default. Useful when the user
    # prefers a more generous broadcast frame.
    "wide": {"hfov_min": 26.0, "margin_deg": 15.0},
}

# Panini-General `d` parameter named presets. Default "rectilinear"
# (d=0.0) gives pure pinhole math via the Panini formula: straightest
# possible lines, mild edge stretch on very wide framings. "panini"
# (d=1.0) is the classic stereographic Panini: less line-straight but
# less edge stretch. plan_campath writes the resolved d to the campath
# JSON so the renderer can use it independently of the project file
# (allows per-run A/B testing without editing the project).
_PANINI_PRESETS = {
    "rectilinear": 0.0,
    "panini": 1.0,
}


def _d_for_no_black(hfov_deg: float, pv_deg: float,
                    vfov_pano_deg: float, pitch0_deg: float,
                    aspect: float, safety_deg: float = 2.0,
                    black_tolerance_deg: float = 0.0) -> float:
    """Closed-form minimum Panini-General d such that no output-frame
    pixel samples a ray beyond the pano's vfov.

    The constraint is tightest at the top-center / bottom-center of the
    output frame (where the ray-pitch demand is largest). For Panini-
    General that demand is:

        phi_top(d) = atan( (d+1) * a * sin(hfov/2) / (d + cos(hfov/2)) )

    where a = out_h / out_w. The available budget on each side is the
    pano half-vfov minus the source mounting pitch0 and the per-frame
    virtual cam pitch, less a safety margin to avoid sampling artefacts
    at the very edge of the pano:

        avail_top = vfov_pano/2 - pitch0 - pv - safety
        avail_bot = vfov_pano/2 + pitch0 + pv - safety
        phi_avail = min(avail_top, avail_bot)

    Setting phi_top(d) = phi_avail and solving:

        tan(phi_avail) * (d + cos(hfov/2))
            = (d+1) * a * sin(hfov/2)
        d * (tan(phi_avail) - a*sin(hfov/2))
            = a*sin(hfov/2) - tan(phi_avail)*cos(hfov/2)
        d = (a*sin - tan*cos) / (tan - a*sin)

    Returned value clamped to >= 0. Note: when phi_avail is very small
    (highly asymmetric pitch + extreme hfov) the formula approaches the
    asymptotic limit of the Panini family and d explodes; the caller is
    expected to clip via panini_d_cap. Inf returned when phi_avail <= 0
    (pano cannot cover even d=inf -- shouldn't happen on sane mounts).
    """
    half_v = vfov_pano_deg / 2.0 - safety_deg
    avail_top = half_v - pitch0_deg - pv_deg
    avail_bot = half_v + pitch0_deg + pv_deg
    phi_avail = min(avail_top, avail_bot)
    # Black tolerance: relax the target by this many degrees of phi.
    # The d-formula will return a smaller value (possibly 0) than the
    # strict-no-black solve.
    phi_avail = phi_avail + float(black_tolerance_deg)
    if phi_avail <= 0:
        return float("inf")
    tp = np.tan(np.radians(phi_avail))
    hf = np.radians(hfov_deg)
    s = np.sin(hf / 2.0)
    c = np.cos(hf / 2.0)
    # Two threshold checks:
    # (1) phi(d=0) = atan(a * tan(hf/2)). If phi_avail >= that, d=0 fits
    #     (rectilinear already demands less than the budget). The check
    #     is tan(phi_avail) >= a * tan(hf/2), equivalent to tp*c >= a*s.
    # (2) phi(d=inf) = atan(a * sin(hf/2)). If phi_avail < that, NO
    #     finite d works -- we're past the Panini family asymptote.
    #     Return infinity; caller's d_cap will clamp.
    # The denominator (tp - a*s) is positive iff between these two
    # cases (the formula has a real solution for d > 0).
    if tp * c >= aspect * s:
        return 0.0  # rectilinear already fits
    den = tp - aspect * s
    if den <= 1e-9:
        return float("inf")  # past the Panini asymptote
    num = aspect * s - tp * c
    return max(0.0, float(num / den))


def plan_campath(players_json: str, project: str = "project.json",
                 margin_deg: float | None = None,
                 hfov_min: float | None = None,
                 hfov_max: float = 180.0, lookahead_s: float = 2.5,
                 smooth_t: float = 0.7,
                 # Named framing preset. Each preset sets defaults for
                 # margin_deg and hfov_min; explicit values for those
                 # parameters override the preset's choice. "default" is
                 # the production setting from #15; "wide" matches the
                 # B-style "more breathing room everywhere" comparison.
                 view_mode: str = "default",
                 # Detection-dropout protection: when a chunk of framing-pool
                 # players briefly drops out (e.g. YOLO miss + Schmitt
                 # deactivation), the natural per-frame hfov collapses
                 # because the remaining dots cluster on one side. Without
                 # protection the lookahead-MEAN yaw target gets pulled
                 # toward that cluster and the camera swings. We detect
                 # the collapse (natural hfov drops to < dropout_hfov_frac
                 # of last valid frame's hfov) and treat the affected
                 # frames as invalid -- the existing carry-last-valid
                 # logic then reuses the previous target. After
                 # max_dropout_hold_s of sustained collapse we accept
                 # the new reality (it wasn't transient after all).
                 dropout_hfov_frac: float = 0.5,
                 max_dropout_hold_s: float = 4.0,
                 # Smoothing-deadzones: per-axis "do nothing" range. If
                 # the target moves less than the deadzone from current
                 # position, the smoother holds. Bigger deadzones reduce
                 # micro-pans / jitter from per-frame detection noise but
                 # increase responsiveness lag for small genuine shifts.
                 # Pre-#13 defaults were yaw 0.4, pitch 0.3, hfov 0.6 --
                 # too twitchy. Bumped to let normal cluster wiggle pass
                 # without moving the camera; bigger framing changes still
                 # get through normally (v_max/a_max not affected).
                 yaw_deadzone_deg: float = 2.0,
                 pitch_deadzone_deg: float = 1.5,
                 hfov_deadzone_deg: float = 3.0,
                 # Soft deadzone: quadratic spring-force taper inside the
                 # deadzone band, so a slowly-drifting target produces
                 # smooth continuous camera motion rather than the
                 # stair-step "hold + snap" of a hard deadzone. Default
                 # ON (production). False reverts to the hard cut-off
                 # (rejects jitter more aggressively at the cost of
                 # visible stair-stepping during slow pans -- useful
                 # only as an A/B comparison.
                 soft_deadzone: bool = True,
                 # Lookahead aggregation: how to summarise the lookahead
                 # window's per-frame targets into the value the smoother
                 # tracks. MEAN is pulled by brief excursions (a single-
                 # frame yaw spike contributes proportionally to its
                 # duration). MEDIAN requires >50% of the window to
                 # have shifted before the target moves -- the user's
                 # "commit only if it persists" wish (#14). hfov is
                 # ALWAYS aggregated via MAX regardless (we want the
                 # framing wide enough to include brief cluster
                 # expansions; tighter is worse than wider for the same
                 # smoothness cost).
                 lookahead_aggregator: str = "median",
                 # Projection mode written to the campath file; the
                 # renderer respects this. "panini" = true Panini-General
                 # (uses `projection_blend` from project file as the
                 # Panini `d` parameter, d=1.0 = classic stereographic
                 # Panini, default). "cylindrical" = legacy hybrid
                 # (rect-x + cyl-y with `projection_blend` as blend).
                 # "rectilinear" = pure pinhole (FOV interpreted as VFOV).
                 projection_mode: str = "panini",
                 # Panini d-parameter preset, picked at campath time so
                 # per-run A/B testing doesn't require editing the
                 # project file. "rectilinear" (default) = d=0.0, pure
                 # pinhole math via the Panini formula: straightest
                 # possible lines. "panini" = d=1.0, classic stereographic
                 # Panini. Set explicit panini_d to override the preset.
                 panini_preset: str = "rectilinear",
                 panini_d: float | None = None,
                 # Adaptive Panini d (added v0.12). When True, the
                 # per-frame d is computed from each smoothed frame's
                 # (hfov, pitch) to be just large enough to keep no
                 # rays sampled past the pano's vfov. Below the
                 # critical HFOV (clip-specific, depends on calibration)
                 # d stays at 0 (pure rectilinear). The result is
                 # smoothed with a small additional pass so it doesn't
                 # twitch as hfov crosses the threshold. Per-frame d
                 # is stored in path entries; the top-level panini_d
                 # field is used only when adaptive is off OR by older
                 # renderers reading new campath files.
                 # Default values mirror cfg.panini_d_* but can be
                 # overridden per campath run for A/B testing.
                 panini_d_adaptive: bool | None = None,
                 panini_d_cap: float | None = None,
                 panini_d_safety_deg: float | None = None,
                 # Black tolerance (added 2026-06-01): degrees of ray-
                 # pitch overflow allowed before d engages. Bigger =
                 # d stays at 0 over a wider HFOV range, at the cost
                 # of a visible black sliver. 0 = strict no-black,
                 # 5 = small sliver tolerated (default), 10 =
                 # significantly aggressive d=0 bias.
                 panini_d_black_tolerance_deg: float | None = None,
                 # Snap-to-zero threshold on smoothed d. The smoother
                 # asymptotes toward 0 but never reaches it; below this
                 # value the renderer is told d=0 exactly.
                 panini_d_min_threshold: float | None = None,
                 # Smoothing time constant for the per-frame d signal.
                 # Smaller than yaw/hfov's smooth_t because d is derived
                 # from already-smoothed hfov and only needs to soften
                 # the C0 kink at the d=0 threshold. Per-axis caps are
                 # generous because d-changes are visually cheap.
                 d_smooth_t: float = 0.4,
                 d_deadzone: float = 0.05,
                 # Chunked-pipeline state handoff (#20b, v0.12). When
                 # set, the smoothers initialise at these positions
                 # and velocities instead of from target[0] with zero
                 # velocity. Each chunk's plan_campath returns its
                 # final state (written to the campath JSON under
                 # `smoother_final_state`); the pipeline passes it
                 # forward so chunk N+1's camera bridges smoothly from
                 # chunk N's final position. Keys: yaw_pos, yaw_vel,
                 # pitch_pos, pitch_vel, hfov_pos, hfov_vel, d_pos,
                 # d_vel. Any missing key falls back to the legacy
                 # single-pass behaviour for that axis.
                 initial_smoother_state: dict | None = None,
                 out_path: str = "campath.json") -> str:
    # Pass the classifier's framing pool (players_*.json), not raw tracks.
    # Raw-tracks input still works (the gm.in_field filter below is a safety
    # net) but framing will chase sideline subs and between-point lineup
    # people. As of v0.9 the per-track + per-frame classifier (waruka/
    # classify.py) is validated on 4 clips and writes a clean stable-active
    # framing pool that the campath should be driven from.
    if view_mode not in _VIEW_PRESETS:
        raise ValueError(
            f"view_mode must be one of {list(_VIEW_PRESETS)}, "
            f"got {view_mode!r}")
    preset = _VIEW_PRESETS[view_mode]
    if panini_preset not in _PANINI_PRESETS:
        raise ValueError(
            f"panini_preset must be one of {list(_PANINI_PRESETS)}, "
            f"got {panini_preset!r}")
    if panini_d is None:
        panini_d = _PANINI_PRESETS[panini_preset]
    if margin_deg is None:
        margin_deg = preset["margin_deg"]
    if hfov_min is None:
        hfov_min = preset["hfov_min"]
    data = json.load(open(players_json))
    cfg = ProjectConfig.load(project)
    # Adaptive-d defaults pull from the project file unless overridden
    # at the CLI. The project file holds the canonical per-clip
    # calibration; CLI override is for A/B testing.
    if panini_d_adaptive is None:
        panini_d_adaptive = getattr(cfg, "panini_d_adaptive", True)
    if panini_d_cap is None:
        panini_d_cap = getattr(cfg, "panini_d_cap", 1.5)
    if panini_d_safety_deg is None:
        panini_d_safety_deg = getattr(cfg, "panini_d_safety_deg", 2.0)
    if panini_d_black_tolerance_deg is None:
        panini_d_black_tolerance_deg = getattr(
            cfg, "panini_d_black_tolerance_deg", 0.0)
    if panini_d_min_threshold is None:
        panini_d_min_threshold = getattr(
            cfg, "panini_d_min_threshold", 0.0)
    gm = GroundModel(cfg.homography, cfg.field_length_m, cfg.field_width_m)
    fps, stride = data["fps"], data["stride"]
    dt = stride / fps

    # Stable pitch aimed at the field interior (broadcast cameras pan/zoom,
    # pitch ~ constant). Per-frame foot-elevation is the wrong source: with
    # accurate geometry the near players' feet rays are steeply downward and
    # the camera over-tilts (black void below the source). Aim ~mid-depth
    # with headroom instead, clamped.
    cdir = gm.ray_from_ground(np.array([[0.0, cfg.field_width_m * 0.5]]))[0]
    el_c = float(np.degrees(np.arcsin(np.clip(-cdir[1], -1, 1))))
    pitch_fixed = float(np.clip(el_c - 6.0, -2.0, 10.0))

    frames = data["frames"]
    n = len(frames)
    yaw = np.zeros(n)
    pitch = np.full(n, pitch_fixed)
    hfov = np.zeros(n)
    valid = np.zeros(n, bool)
    last_valid_hfov_nat = None     # most recent NOT-dropout natural hfov
    dropout_start_i = None         # frame index where current dropout began
    max_dropout_frames = max(0, int(round(max_dropout_hold_s
                                          / max(dt, 1e-6))))
    n_dropouts_skipped = 0
    for i, fr in enumerate(frames):
        pl = fr["players"]
        if len(pl) < 1:
            continue
        P = np.array([[p["X"], p["Z"]] for p in pl], float)
        # No spatial in_field filter: the classifier (waruka/classify.py)
        # already owns framing-pool membership and applies asymmetric
        # in_field margins plus per-direction off_hold grace (endzone
        # walk-outs stay in the pool for 10s). Re-applying gm.in_field
        # here silently overrode those decisions -- in particular it
        # threw out endzone-grace players the moment they crossed
        # |X|=52, causing the camera to pan AWAY from a player walking
        # out to retrieve a disc. Trust classifier output.
        rays = gm.ray_from_ground(P)
        y, _pt, hf = _targets(rays, margin_deg)

        # Dropout test: natural hfov collapsed relative to last accepted
        # frame. If yes and we're still within the hold window, mark
        # invalid so carry-last-valid takes over.
        is_dropout = False
        if (last_valid_hfov_nat is not None
                and hf < dropout_hfov_frac * last_valid_hfov_nat):
            if dropout_start_i is None:
                dropout_start_i = i
            if (i - dropout_start_i) < max_dropout_frames:
                is_dropout = True
            # else: dropout has lasted too long, accept the new reality

        if is_dropout:
            n_dropouts_skipped += 1
            continue  # valid[i] stays False; carry-last-valid fills it

        # Accept this frame as the new reference.
        yaw[i] = y
        hfov[i] = np.clip(hf, hfov_min, hfov_max)
        valid[i] = True
        last_valid_hfov_nat = hf
        dropout_start_i = None

    last = None
    for i in range(n):
        if valid[i]:
            last = (yaw[i], pitch[i], hfov[i])
        elif last is not None:
            yaw[i], pitch[i], hfov[i] = last
    if last is None:
        raise SystemExit("no on-field players in any frame")
    first = int(np.argmax(valid))
    yaw[:first], pitch[:first], hfov[:first] = (
        yaw[first], pitch[first], hfov[first])

    la = max(1, int(round(lookahead_s / dt)))
    if lookahead_aggregator == "median":
        _agg = np.median
    elif lookahead_aggregator == "mean":
        _agg = np.mean
    else:
        raise ValueError(
            f"lookahead_aggregator must be 'median' or 'mean', "
            f"got {lookahead_aggregator!r}")
    yaw_t = np.array([_agg(yaw[i:i + la + 1]) for i in range(n)])
    pitch_t = np.array([_agg(pitch[i:i + la + 1]) for i in range(n)])
    hfov_t = np.array([hfov[i:i + la + 1].max() for i in range(n)])

    # Initial smoother state for chunked-pipeline state handoff (#20b).
    # Each axis can be initialised independently; missing keys fall
    # back to single-pass behaviour (start at target[0] with v=0).
    iss = initial_smoother_state or {}
    yaw_s, yaw_pos_f, yaw_vel_f = _smooth_signal(
        yaw_t, dt, smooth_t, 35.0, 120.0,
        yaw_deadzone_deg, soft_deadzone=soft_deadzone,
        init_pos=iss.get("yaw_pos"), init_vel=iss.get("yaw_vel", 0.0))
    pitch_s, pitch_pos_f, pitch_vel_f = _smooth_signal(
        pitch_t, dt, smooth_t, 12.0, 50.0,
        pitch_deadzone_deg, soft_deadzone=soft_deadzone,
        init_pos=iss.get("pitch_pos"),
        init_vel=iss.get("pitch_vel", 0.0))
    hfov_raw, hfov_pos_f, hfov_vel_f = _smooth_signal(
        hfov_t, dt, smooth_t * 1.3, 30.0, 90.0,
        hfov_deadzone_deg, soft_deadzone=soft_deadzone,
        init_pos=iss.get("hfov_pos"),
        init_vel=iss.get("hfov_vel", 0.0))
    hfov_s = np.clip(hfov_raw, hfov_min, hfov_max)

    # Per-frame Panini d. When adaptive, derive from the smoothed
    # (hfov, pitch) trajectory and the project calibration, capped to
    # panini_d_cap, then smoothed (small additional pass) to soften the
    # C0 kink at the no-black threshold. When non-adaptive, every
    # frame gets the static panini_d.
    aspect = cfg.out_h / cfg.out_w
    vfov_p = cfg.pano.vfov_deg
    pitch0 = cfg.pano.pitch0_deg
    if panini_d_adaptive and projection_mode == "panini":
        d_raw = np.array([
            min(float(panini_d_cap),
                _d_for_no_black(float(hfov_s[i]), float(pitch_s[i]),
                                vfov_p, pitch0, aspect,
                                float(panini_d_safety_deg),
                                float(panini_d_black_tolerance_deg)))
            for i in range(n)
        ])
        d_s, d_pos_f, d_vel_f = _smooth_signal(
            d_raw, dt, d_smooth_t, 5.0, 20.0,
            d_deadzone, soft_deadzone=soft_deadzone,
            init_pos=iss.get("d_pos"), init_vel=iss.get("d_vel", 0.0))
        d_s = np.clip(d_s, 0.0, float(panini_d_cap))
        # Snap tiny smoothed-d to exactly 0. The smoother asymptotes
        # toward its target but never quite reaches it; without this
        # the renderer sees d=0.001 or d=0.02 forever after a wide
        # framing ends, which is conceptually noisy even if visually
        # indistinguishable from 0.
        if panini_d_min_threshold > 0:
            d_s = np.where(d_s < float(panini_d_min_threshold), 0.0, d_s)
    else:
        d_s = np.full(n, float(panini_d))
        d_pos_f, d_vel_f = float(panini_d), 0.0

    path = [{"frame": frames[i]["frame"],
             "yaw": round(float(yaw_s[i]), 4),
             "pitch": round(float(pitch_s[i]), 4),
             "hfov": round(float(hfov_s[i]), 4),
             "d": round(float(d_s[i]), 4)} for i in range(n)]
    out_payload = {
        "video": data["video"], "fps": fps,
        "out_w": cfg.out_w, "out_h": cfg.out_h,
        "projection": projection_mode,
        "panini_d": float(panini_d),  # back-compat: legacy renderers
        "panini_d_adaptive": bool(panini_d_adaptive),
        "panini_d_cap": float(panini_d_cap),
        "panini_d_safety_deg": float(panini_d_safety_deg),
        "panini_d_black_tolerance_deg": float(panini_d_black_tolerance_deg),
        "panini_d_min_threshold": float(panini_d_min_threshold),
        # Smoother state at the END of this chunk, ready to be passed
        # to the next chunk's plan_campath as initial_smoother_state.
        # Pipeline (#20b) chains these across chunks.
        "smoother_final_state": {
            "yaw_pos": float(yaw_pos_f), "yaw_vel": float(yaw_vel_f),
            "pitch_pos": float(pitch_pos_f), "pitch_vel": float(pitch_vel_f),
            "hfov_pos": float(hfov_pos_f), "hfov_vel": float(hfov_vel_f),
            "d_pos": float(d_pos_f), "d_vel": float(d_vel_f),
        },
        "path": path,
    }
    json.dump(out_payload, open(out_path, "w"))
    if panini_d_adaptive and projection_mode == "panini":
        n_active = int(np.sum(d_s > 0.005))
        d_max_used = float(d_s.max())
        d_mean_when_active = (float(d_s[d_s > 0.005].mean())
                              if n_active > 0 else 0.0)
        d_msg = (f", adaptive d: {n_active}/{n} frames active "
                 f"({100*n_active/n:.1f}%), max d={d_max_used:.2f}, "
                 f"mean-when-active={d_mean_when_active:.2f}")
    else:
        d_msg = f", d={panini_d:.2f} (static)"
    print(f"wrote {out_path}  ({n} samples, hfov "
          f"{hfov_s.min():.1f}-{hfov_s.max():.1f}, "
          f"yaw {yaw_s.min():.1f}..{yaw_s.max():.1f}, "
          f"pitch {pitch_s.min():.1f}..{pitch_s.max():.1f}{d_msg})")
    return out_path
