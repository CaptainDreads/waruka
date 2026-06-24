# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Project file: per-video calibration + output settings, persisted as JSON.

The camera is fixed, so calibration is done once per recording and reused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .projection import PanoModel


@dataclass
class ProjectConfig:
    source_video: str
    pano: PanoModel
    out_w: int = 2560
    out_h: int = 1440
    # Straight-in-reality references the user marked, in source pixels:
    # list of polylines, each a list of [x, y]. Kept so calibration can be
    # re-fitted or audited later.
    calib_lines: list[list[list[float]]] = field(default_factory=list)
    # Field boundary polygon in source pixels (display / quick in-out test).
    field_polygon: list[list[float]] = field(default_factory=list)
    # Ground-plane calibration (phase 2). homography is 9 floats row-major
    # for d ~ H[X,Z,1]; None until markfield is run. field_marks holds the
    # raw-panorama source pixels that were clicked.
    field_length_m: float = 100.0
    field_width_m: float = 37.0
    homography: list[float] | None = None
    field_marks: dict = field(default_factory=dict)
    # Output look. projection_blend: 1=pure cylindrical (most line curve),
    # 0=rectilinear azimuth (straightest lines, mild player stretch).
    # Used when the campath JSON specifies projection="cylindrical".
    projection_blend: float = 0.6
    # Panini-General `d` parameter. Used when the campath JSON specifies
    # projection="panini" AND no panini_d is set in the campath JSON.
    # d=0 -> pure rectilinear (straightest lines, mild edge stretch on
    # wide framings; recommended default per user 2026-05-31). d=1 ->
    # classic stereographic Panini (less line-straight but less edge
    # stretch). d > 1 -> progressively more cylindrical-like.
    panini_d: float = 0.0
    # Adaptive Panini d (added v0.12). When True (default), campath
    # computes a per-frame Panini d that's just large enough to keep the
    # output rectangle from sampling rays beyond the pano's vfov. The
    # critical HFOV depends on the calibration (pano vfov, mounting
    # pitch0) and the per-frame virtual-camera pitch, so the threshold is
    # derived per clip rather than hard-coded. Below the critical HFOV
    # d=0 (pure rectilinear). Above it d ramps to fill the available
    # vfov, capped at panini_d_cap. See campath._d_for_no_black for the
    # closed-form solution. Set False to use a constant panini_d for the
    # whole clip (the legacy v0.11 behaviour).
    panini_d_adaptive: bool = True
    # Upper bound on adaptive d. d > ~1.5 starts to look visibly
    # cylindrical (curved horizon); below that it's hard to tell from
    # rectilinear at typical play distances. At the most extreme HFOVs
    # the analytical d explodes near the asymptote (the Panini family
    # has a hard limit on how small it can make the top-edge ray pitch),
    # so capping is essential. Residual black at the d_cap ceiling is
    # tiny and best handled by edge-fill (TODO #7 / #30).
    panini_d_cap: float = 1.5
    # Safety margin from pano vfov edge when computing adaptive d. The
    # analytical zero-black condition is exact at safety=0, but the
    # sampling kernel (grid_sample bilinear) needs a couple of degrees
    # of slack to avoid edge artefacts at the pano's top/bottom rows.
    panini_d_safety_deg: float = 2.0
    # Black tolerance: degrees of top-center ray pitch we'll let exceed
    # the pano vfov budget before engaging d. Bigger = d stays at 0
    # over a wider HFOV range, at the cost of a visible black strip at
    # the top/bottom center of the frame for those intermediate
    # framings. Default 0 (strict no-black) restores the v0.12 original
    # behaviour. Trial 2026-06-01 showed that with tolerance > 0 the
    # camera lurches more between "no d" and "high d" because the
    # smooth ramp through intermediate d-values gets compressed into
    # a smaller HFOV range -- the visible projection-change is more
    # abrupt at higher tolerance, not less. User preference is for
    # the strict-no-black behaviour with its gentle d ramp. Knob
    # kept for experimentation.
    panini_d_black_tolerance_deg: float = 0.0
    # Snap-to-zero threshold for the smoothed per-frame d. Default 0
    # (disabled). When > 0, values below this snap to exactly 0. This
    # was added to kill the smoother's asymptotic trailing-tail (d
    # drifting at ~0.02 for seconds after a wide framing ends), but
    # in practice 0 vs 0.02 is visually indistinguishable AND if the
    # smoothed-d oscillates around the threshold (entirely plausible
    # given the smoother's deadzone of 0.05) the snap produces
    # visible flicker -- worse than the trail it was meant to fix.
    # Knob kept configurable; default off.
    panini_d_min_threshold: float = 0.0
    # Pano edge-fill mode (added v0.12). What the renderer does when a
    # ray asks for a pano lat/lon outside the source image's coverage:
    #   "zeros"  - return black (the v0.12 original behaviour). At
    #              wide framings where adaptive-d hits the cap a thin
    #              black sliver remains; this is what you see.
    #   "border" - clamp to the nearest edge pixel, preserving
    #              longitude. The top row of the pano (sky/treeline)
    #              extends upward, the bottom row (close ground)
    #              extends downward. Visually natural, zero compute
    #              cost. Recommended default per user 2026-06-01.
    #   "blur"   - pre-pad the source image with progressively-blurred
    #              extensions of the top/bottom rows, with an optional
    #              fade band inside the original that hides the seam.
    #              Higher quality than "border" (kills vertical stripe
    #              artefacts from cloud edges etc.) at a small per-
    #              frame compute cost (2 Gaussian blurs + a couple of
    #              numpy blends). Recommended default per user
    #              preference 2026-06-01.
    pano_edge_fill_mode: str = "blur"
    # Vfov extension (deg) per side when pano_edge_fill_mode = "blur".
    # When None (default): computed automatically per render from the
    # calibration + campath so the blur extension always covers the
    # worst-case ray demand for the current mount. Generic across
    # mounts -- a future match with stronger pitch0 / off-centre
    # camera placement will get more padding automatically.
    # When set to a number: explicit override (useful for A/B testing
    # different fade depths). Border-clamp fallback handles any
    # residual that exceeds this either way.
    # Ignored unless mode = "blur".
    pano_edge_fill_blur_deg: float | None = None
    # Horizontal gaussian-blur sigma (pixels) at the OUTERMOST padded
    # row (top of image / bottom of image). Used as the max value in
    # the progressive blur ramp. Higher = smoother distant sky/ground
    # but loses any longitude-varying texture.
    pano_edge_fill_blur_sigma_px: float = 40.0
    # Blur sigma at the SEAM between original content and the padded
    # extension. Smaller than sigma_max_px. Sets the floor of the blur
    # ramp: padding rows interpolate from this at the seam up to
    # sigma_max_px at the outermost row, and the fade band inside the
    # original interpolates from 0 (interior) up to this at the seam.
    # Setting both sigmas equal disables the progressive ramp (uniform
    # blur). Setting this to 0 yields a sharp boundary -- the padding
    # blends from heavy back down to nothing at the seam (gives a soft
    # "halo" where the original's top row visibly fades into a smear).
    pano_edge_fill_blur_boundary_sigma_px: float = 8.0
    # Fade-band depth (deg of pano vfov) -- how many degrees of the
    # original's top/bottom edge get smoothly blurred into the seam.
    # 0 = sharp seam (still at sigma_boundary_px, but no in-original
    # ramp). 2 = a couple of degrees of fade. Hides the "now-I'm-in-
    # blur-mode" edge by ramping in the blur before the actual
    # boundary -- per user preference 2026-06-01.
    pano_edge_fill_blur_fade_deg: float = 2.0
    debug_overlay: bool = False
    # Opt-in refinement controls (added 2026-05-29). Default = None means
    # baseline behaviour (uniform corner_weight=2.0, no camera-height
    # anchor). See waruka.ground.refine_homography for semantics and the
    # 2026-05-29 dewarp-ceiling investigation for when to use them.
    # cam_height_m: known mount height in metres. When set, anchors the
    #   decomposed camera Y via a third pose residual. Useful when corner
    #   clicks are unreliable (e.g. no visible corner markers in the pano).
    # corner_weights: per-corner weights (length must match field_marks
    #   "corners"). Useful to downweight extreme-longitude corners that
    #   carry inherent click noise.
    cam_height_m: float | None = None
    corner_weights: list[float] | None = None
    # Auto MLE per-mark weighting (added 2026-05-30). When True (default),
    # markfield's fit weights each mark by 1/(local click-error
    # amplification) computed from the initial 4-corner DLT homography.
    # Marks at extreme pano longitudes (far ends of sidelines, back
    # corners) get low weight; well-conditioned middle marks get high
    # weight. Pass --no-auto-balance to disable.
    auto_balance_marks: bool = True
    # Near-sideline trust multiplier (added 2026-05-30). Multiplies all
    # near-sideline LSQ weights by this factor before the refit. The near
    # sideline is closest to the camera and visually easiest to verify
    # by eye, so trusting it more than the far sideline reflects reality.
    # 1.0 = no extra boost (MLE only); 3.0 (default) makes the LSQ
    # essentially lock the near sideline onto the marks at the cost of a
    # small (<0.1 m) increase in far_rms on a well-marked clip.
    near_trust: float = 3.0
    # Markfield UI persistence (added 2026-05-30). Toggle keys G/H in
    # markfield set these. When False, the corresponding overlay is
    # hidden so the user can click against the raw image without
    # interference. Marks themselves are never hidden.
    show_guides: bool = True
    show_fitbox: bool = True
    # Default scrub time when (re)opening markfield or calibrate. Lets
    # the user resume at the frame they last worked on rather than t=2.0.
    last_scrub_t: float | None = None
    # Calibrate-preview UI persistence (added 2026-05-30). Toggle keys
    # L/O in calibrate set these. show_level_line: translucent perfectly
    # horizontal reference line through the preview, useful to check the
    # dewarped horizon is level. show_calib_overlay: reproject the marked
    # calibration lines into the preview so you can see how straight they
    # look under the current k1/k2 fit.
    show_level_line: bool = True
    show_calib_overlay: bool = True
    # Vertical position of the level reference line as a fraction of
    # preview height, signed from the centre. 0 = centred; +0.25 = a
    # quarter of the height below centre; -0.4 = near the top. Stored
    # as a fraction (not pixels) so it survives a future change to the
    # preview output size. Right-drag in the preview window updates it
    # interactively; key 0 resets to centred.
    level_line_y_frac: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pano"] = self.pano.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectConfig":
        d = dict(d)
        d["pano"] = PanoModel.from_dict(d["pano"])
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def for_video(cls, source_video: str, src_w: int, src_h: int) -> "ProjectConfig":
        return cls(source_video=source_video, pano=PanoModel(src_w=src_w, src_h=src_h))
