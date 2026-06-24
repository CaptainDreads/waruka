# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Ground-plane geometry.

For a planar pitch the relation between a de-warped camera ray d and the
metric ground point [X, Z] is an exact linear 3x3 homography:

    d  ~  H . [X, Z, 1]^T          (proportional, any ray angle)

This holds for arbitrarily wide rays (no pinhole degeneracy), so the whole
field can be solved from corners marked across the raw panorama.

Ground frame: X = along the long axis (endzone to endzone), centred at the
field centre so X in [-L/2, +L/2]; Z = across, Z=0 at the near sideline
(camera side), Z = W at the far sideline.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def solve_homography(rays: np.ndarray, xz: np.ndarray) -> np.ndarray:
    """DLT for d ~ H[X,Z,1]. rays:(N,3) ground:(N,2), N>=4. Returns 3x3."""
    rays = np.asarray(rays, float)
    xz = np.asarray(xz, float)
    if len(rays) < 4:
        raise ValueError("need >=4 ray/ground correspondences")
    g = np.column_stack([xz, np.ones(len(xz))])  # (N,3) [X,Z,1]
    A = []
    for d, gi in zip(rays, g):
        z3 = np.zeros(3)
        # d x (H g) = 0 -> two independent rows
        A.append(np.concatenate([z3, -d[2] * gi, d[1] * gi]))
        A.append(np.concatenate([d[2] * gi, z3, -d[0] * gi]))
    _, _, vt = np.linalg.svd(np.asarray(A))
    H = vt[-1].reshape(3, 3)
    # Fix global sign so ground points sit in front (positive scale).
    scale = (np.linalg.inv(H) @ rays.T)[2]
    if np.median(scale) < 0:
        H = -H
    return H


def _compute_cam_y(H: np.ndarray) -> float:
    """Camera world Y coordinate from H, mirroring GroundModel.decompose_pose.

    Used by `refine_homography` to enforce a camera-height anchor. Inlined
    here (rather than constructing a GroundModel) so it stays cheap inside
    the LSQ residual function.
    """
    a, b, c = H[:, 0], H[:, 1], H[:, 2]
    s = 2.0 / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-9)
    r1, r3, t = a * s, b * s, c * s
    M = np.column_stack([r1, r3])
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R13 = U @ Vt
    if np.dot(R13[:, 0], r1) < 0:
        R13[:, 0] *= -1
    if np.dot(R13[:, 1], r3) < 0:
        R13[:, 1] *= -1
    r1, r3 = R13[:, 0], R13[:, 1]
    r2 = np.cross(r3, r1)
    return float(-np.dot(r2, t))


def compute_mle_weights(pano, H, corner_px=None, near_px=None, far_px=None,
                        pixel_step: float = 1.0):
    """Per-mark MLE weights based on local click-error amplification.

    For each marked source pixel, perturb by `pixel_step` in each axis and
    measure how much the projected ground point moves. The weight is the
    inverse of that motion's magnitude.

    Marks at extreme pano longitudes (e.g. back corners at lon≈±88°, or
    far ends of a sideline as seen from a midfield mount) produce metres
    of ground motion per pixel of click error -- they get LOW weight.
    Marks near the camera's forward direction (middle of the near
    sideline) produce ~cm of ground motion per pixel -- they get HIGH
    weight. This implements maximum-likelihood weighting under the
    assumption that click error is roughly constant in pano pixels.

    Weights are normalised within each group (corner / near / far) so
    the median weight per group is 1.0. This keeps the LSQ residual
    scale comparable to uniform-weight fits.

    Returns a dict like ``{'corner': np.ndarray, 'near': ..., 'far': ...}``;
    keys are present only for groups with marks.
    """
    Hinv = np.linalg.inv(np.asarray(H, float))

    def _project_xz(px_arr):
        d = pano.src_to_direction(px_arr[:, 0], px_arr[:, 1])
        q = (Hinv @ d.T).T
        return q[:, :2] / q[:, 2:3]

    def _amp(px_arr, only_z: bool):
        base = _project_xz(px_arr)
        dx = _project_xz(px_arr + np.array([pixel_step, 0.0])) - base
        dy = _project_xz(px_arr + np.array([0.0, pixel_step])) - base
        if only_z:
            return np.sqrt(dx[:, 1] ** 2 + dy[:, 1] ** 2)
        # Magnitude of the full (X,Z) motion.
        return np.sqrt(np.sum(dx ** 2 + dy ** 2, axis=1))

    def _weights(px, only_z):
        if px is None or len(px) == 0:
            return None
        amp = _amp(np.asarray(px, float).reshape(-1, 2), only_z)
        amp = np.maximum(amp, 1e-6)
        w = 1.0 / amp
        # Normalise so median weight = 1.0 -> overall LSQ scale comparable
        # to uniform weighting.
        med = float(np.median(w))
        return w / med if med > 0 else w

    out = {}
    if corner_px is not None and len(corner_px):
        out["corner"] = _weights(corner_px, only_z=False)
    if near_px is not None and len(near_px):
        out["near"] = _weights(near_px, only_z=True)
    if far_px is not None and len(far_px):
        out["far"] = _weights(far_px, only_z=True)
    return out


def refine_homography(corner_rays, corner_xz, near_rays=None, far_rays=None,
                      W: float = 37.0, corner_weight: float = 1.0,
                      sideline_weight: float = 1.0,
                      enforce_pose: bool = False,
                      pose_weight: float = 0.0,
                      corner_weights=None,
                      near_weights=None,
                      far_weights=None,
                      cam_height_m: float | None = None,
                      cam_height_weight: float | None = None):
    """Least-squares homography refit in METRIC ground units.

    Init from 4-corner exact DLT, then refine 8 free DOF against:
      - corners (X, Z both known): 2 residuals per corner, each the metric
        (X, Z) error of `ground_from_ray(d) - [X, Z]` in metres.
      - near-sideline points: 1 residual per point, the metric Z error
        `ground_from_ray(d)[1] - 0` in metres.
      - far-sideline points: 1 residual per point, the metric Z error
        `ground_from_ray(d)[1] - W` in metres.

    All data residuals are in metres so weights have direct physical
    meaning. The LSQ minimises mean-squared mark error: the answer hugs
    your clicked dots.

    --- Default behaviour change (2026-05-29 → today) ---

    Previously the residuals were in dimensionless `det` units and the
    fit also enforced a "valid camera pose" constraint at `pose_weight=50`
    (||h0||=||h1||, h0⊥h1). On a sideline-mid mount the BACK corners sit
    at lon ≈ ±88° where ½-px click error becomes metres of ground error;
    when those noisy corners conflicted with the (clean) sideline marks
    AND with the pose constraint, the LSQ had to compromise -- and it
    preserved the pose constraint, letting the marks drift several metres
    away from the fitted Z=0/Z=W lines (visible as the yellow LSQ outline
    drifting off the user's sideline dots in markfield).

    The current defaults drop the pose constraint and rebalance everything
    to metric units. The fitted H follows the marks. The tradeoff: H no
    longer decomposes to a strictly orthonormal camera pose, so
    `head_to_ground` (foot-from-head projection) is marginally less
    self-consistent. For Waruka's current pipeline (foot-based ground
    tracking + crowd-IQR framing) this doesn't materially affect output;
    `head_to_ground` accuracy was already deferred.

    Pass `enforce_pose=True, pose_weight=50.0` to recover the old
    pose-constrained behaviour.

    --- Opt-in extensions (added 2026-05-29) ---

    **`corner_weights`** (optional, list/array of N floats matching
    `corner_rays` length): per-corner weight overriding the scalar
    `corner_weight`. Useful when some corner clicks are inherently noisier
    than others -- e.g. typical weighting that trusts forward corners
    more: `[0.5, 0.5, 2.0, 2.0]`. When None (default), the scalar
    `corner_weight` is used uniformly.

    **`near_weights` / `far_weights`** (optional, per-mark weight arrays
    same length as `near_rays` / `far_rays`): per-mark weight overriding
    the scalar `sideline_weight`. Use with `compute_mle_weights()` to
    automatically downweight extreme-longitude marks where click error
    causes huge ground error, and upweight middle-of-the-image marks
    where the geometry is well-conditioned. When None, `sideline_weight`
    is used uniformly.

    **`cam_height_m`** (optional, float metres): only active when
    `enforce_pose=True`. Anchors the decomposed camera Y coordinate
    (height above ground) to this value via a metric (in metres)
    residual `cam_height_weight * (Y_cam - cam_height_m)`. Defaults
    to `cam_height_weight = pose_weight`; use a value of ~10-50 if you
    want a hard Y anchor with the new metric data residuals.
    """
    H0 = solve_homography(corner_rays, corner_xz)
    flat = H0.reshape(-1)
    fix_i = int(np.argmax(np.abs(flat)))  # fix the largest-magnitude entry
    fix_v = float(flat[fix_i])
    free_idx = [i for i in range(9) if i != fix_i]
    x0 = flat[free_idx]

    cr = np.asarray(corner_rays, float)
    cxz = np.asarray(corner_xz, float)
    nr = (np.asarray(near_rays, float) if near_rays is not None
          and len(near_rays) else np.empty((0, 3)))
    fr = (np.asarray(far_rays, float) if far_rays is not None
          and len(far_rays) else np.empty((0, 3)))

    # Per-corner weights: array of N floats, same length as corner_rays.
    # Default = scalar corner_weight applied uniformly.
    if corner_weights is None:
        cw_arr = np.full(len(cr), float(corner_weight), dtype=float)
    else:
        cw_arr = np.asarray(corner_weights, dtype=float).reshape(-1)
        if len(cw_arr) != len(cr):
            raise ValueError(
                f"corner_weights has length {len(cw_arr)} but corner_rays "
                f"has length {len(cr)}")

    # Per-near-sideline weights: array of N floats, same length as near_rays.
    if near_weights is None:
        nw_arr = np.full(len(nr), float(sideline_weight), dtype=float)
    else:
        nw_arr = np.asarray(near_weights, dtype=float).reshape(-1)
        if len(nw_arr) != len(nr):
            raise ValueError(
                f"near_weights has length {len(nw_arr)} but near_rays "
                f"has length {len(nr)}")

    # Per-far-sideline weights: array of N floats, same length as far_rays.
    if far_weights is None:
        fw_arr = np.full(len(fr), float(sideline_weight), dtype=float)
    else:
        fw_arr = np.asarray(far_weights, dtype=float).reshape(-1)
        if len(fw_arr) != len(fr):
            raise ValueError(
                f"far_weights has length {len(fw_arr)} but far_rays "
                f"has length {len(fr)}")

    chw = float(pose_weight if cam_height_weight is None
                else cam_height_weight)

    def make_H(x):
        flat = np.empty(9); flat[fix_i] = fix_v; flat[free_idx] = x
        return flat.reshape(3, 3)

    def res(x):
        H = make_H(x)
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            # Degenerate H during optimisation -- return a large penalty
            # so the LM step is rejected. Should be rare given the DLT seed.
            return np.full(2 * len(cr) + len(nr) + len(fr) + 3, 1e6)
        out = []
        # Corners: metric (X, Z) error per corner.
        if len(cr):
            q = (Hinv @ cr.T).T            # (Nc, 3) ~ [X, Z, scale]
            xz_pred = q[:, :2] / q[:, 2:3] # (Nc, 2) metric
            err = xz_pred - cxz            # (Nc, 2) metric error
            for i in range(len(cr)):
                w = cw_arr[i]
                if w == 0.0:
                    continue
                out.extend(w * err[i])
        # Near sideline: metric Z error (target Z=0).
        if len(nr):
            q = (Hinv @ nr.T).T
            Z_pred = q[:, 1] / q[:, 2]
            out.extend(nw_arr * Z_pred)
        # Far sideline: metric Z error (target Z=W).
        if len(fr):
            q = (Hinv @ fr.T).T
            Z_pred = q[:, 1] / q[:, 2]
            out.extend(fw_arr * (Z_pred - W))
        if enforce_pose and pose_weight > 0:
            h0, h1 = H[:, 0], H[:, 1]
            s_sq = 0.5 * (h0 @ h0 + h1 @ h1)
            out.append(pose_weight * (h0 @ h0 - h1 @ h1) / max(s_sq, 1e-9))
            out.append(pose_weight * (h0 @ h1) / max(s_sq, 1e-9))
            # Optional third pose residual: camera height anchor. Metric
            # (in metres) so it competes evenly with the data residuals.
            if cam_height_m is not None:
                cy = _compute_cam_y(H)
                out.append(chw * (cy - cam_height_m))
        return np.asarray(out)

    sol = least_squares(res, x0, method="lm", max_nfev=400)
    H = make_H(sol.x)
    scale = (np.linalg.inv(H) @ cr.T)[2]
    if np.median(scale) < 0:
        H = -H
    return H


# Tight physical bounds for refittable pano parameters. The joint refit
# is partially redundant with the homography (the LSQ can find many
# (pano, H) pairs that give similar marked-point residuals); without
# tight bounds the optimiser drifts to unphysical local minima.
# Mount tilt is typically <5 deg; lens distortion is a fixed property
# of the camera optics so k1 shouldn't drift far either.
_PANO_BOUNDS = {
    "k1": (-0.1, 0.3),
    "k2": (-0.2, 0.2),
    "pitch0_deg": (-5.0, 5.0),
    "roll0_deg": (-5.0, 5.0),
    "hfov_deg": (170.0, 210.0),
    "vfov_deg": (70.0, 90.0),
}

# Per-param "natural scale" for the prior regularisation. The
# regularisation residual is (p - p_initial) / scale, so a scale of 0.05
# for k1 means "1.0 of residual per 0.05 of drift", and the optimiser
# only spends that residual budget if the data strongly demand it.
_PANO_PRIOR_SCALE = {
    "k1": 0.05,
    "k2": 0.05,
    "pitch0_deg": 2.0,
    "roll0_deg": 2.0,
    "hfov_deg": 5.0,
    "vfov_deg": 3.0,
}


def refine_homography_with_pano(pano, corner_px, corner_xz, near_px=None,
                                 far_px=None, W: float = 37.0,
                                 corner_weight: float = 2.0,
                                 enforce_pose: bool = True,
                                 pose_weight: float = 50.0,
                                 free_pano_params=("k1", "pitch0_deg",
                                                   "roll0_deg"),
                                 pano_prior_weight: float = 2.0):
    """Joint LSQ fit of homography + selected PanoModel parameters.

    Takes raw source pixel coordinates (not pre-derived rays) so the pano
    model can be re-evaluated inside the optimiser as its parameters
    change. Refits a selected subset of pano params -- k1, pitch0_deg,
    roll0_deg by default -- alongside the 8 homography DOFs, anchored by:

        - 4 corners (known X, Z)
        - near-sideline points (known Z=0)
        - far-sideline points (known Z=W)

    The 23-ish field correspondences a user provides during markfield are
    far stronger geometric constraints than the single plumb-line that
    `waruka calibrate` uses. They pin the camera tilt and lens distortion
    well enough to drive a per-mount refit of the dewarp model, which
    `calibrate` alone struggles to do reliably (k1 drifts between mounts
    because it's absorbing un-modelled tilt and stitching error).

    Returns (refined_pano, H). The input pano is NOT mutated -- the
    returned pano is a deep copy with updated params.
    """
    pano = deepcopy(pano)
    free_pano_params = tuple(free_pano_params)
    n_pano = len(free_pano_params)

    corner_px = np.asarray(corner_px, float).reshape(-1, 2)
    corner_xz = np.asarray(corner_xz, float).reshape(-1, 2)
    if len(corner_px) < 4:
        raise ValueError("need >=4 corner correspondences for joint refit")
    near_px = (np.asarray(near_px, float).reshape(-1, 2)
               if near_px is not None and len(near_px) else np.empty((0, 2)))
    far_px = (np.asarray(far_px, float).reshape(-1, 2)
              if far_px is not None and len(far_px) else np.empty((0, 2)))

    # Initial H = 4-corner DLT using the *current* pano's rays.
    crays0 = pano.src_to_direction(corner_px[:, 0], corner_px[:, 1])
    H0 = solve_homography(crays0, corner_xz)

    flat = H0.reshape(-1)
    fix_i = int(np.argmax(np.abs(flat)))
    fix_v = float(flat[fix_i])
    free_H_idx = [i for i in range(9) if i != fix_i]

    x0_H = flat[free_H_idx]
    x0_pano = np.array([float(getattr(pano, p)) for p in free_pano_params])
    x0 = np.concatenate([x0_H, x0_pano])

    # Prior scales: regularisation pulls each pano param toward its initial
    # value with strength inversely proportional to its natural scale.
    prior_scales = np.array([_PANO_PRIOR_SCALE.get(p, 1.0)
                              for p in free_pano_params])

    lb_pano = [_PANO_BOUNDS.get(p, (-np.inf, np.inf))[0]
               for p in free_pano_params]
    ub_pano = [_PANO_BOUNDS.get(p, (-np.inf, np.inf))[1]
               for p in free_pano_params]
    lb = np.concatenate([np.full(8, -np.inf), lb_pano])
    ub = np.concatenate([np.full(8, np.inf), ub_pano])

    cg = np.column_stack([corner_xz, np.ones(len(corner_xz))])

    def make_H(x):
        flat_H = np.empty(9)
        flat_H[fix_i] = fix_v
        flat_H[free_H_idx] = x[:8]
        return flat_H.reshape(3, 3)

    def apply_pano(x):
        for i, p in enumerate(free_pano_params):
            setattr(pano, p, float(x[8 + i]))

    def res(x):
        apply_pano(x)
        H = make_H(x)
        crays = pano.src_to_direction(corner_px[:, 0], corner_px[:, 1])
        nrays = (pano.src_to_direction(near_px[:, 0], near_px[:, 1])
                 if len(near_px) else np.empty((0, 3)))
        frays = (pano.src_to_direction(far_px[:, 0], far_px[:, 1])
                 if len(far_px) else np.empty((0, 3)))
        h0, h1, h2 = H[:, 0], H[:, 1], H[:, 2]
        out = []
        for d, g in zip(crays, cg):
            c = np.cross(d, H @ g)
            out.extend(corner_weight * c[:2])
        if len(nrays):
            c = np.cross(h2[None, :], nrays)
            out.extend(c @ h0)
        if len(frays):
            c = np.cross((W * h1 + h2)[None, :], frays)
            out.extend(c @ h0)
        if enforce_pose:
            s_sq = 0.5 * (h0 @ h0 + h1 @ h1)
            out.append(pose_weight * (h0 @ h0 - h1 @ h1) / max(s_sq, 1e-9))
            out.append(pose_weight * (h0 @ h1) / max(s_sq, 1e-9))
        # Tikhonov prior on pano params: keep them close to initial unless
        # the data really demand otherwise.
        if pano_prior_weight > 0 and n_pano > 0:
            drift = (x[8:] - x0_pano) / prior_scales
            out.extend(pano_prior_weight * drift)
        return np.asarray(out)

    sol = least_squares(res, x0, bounds=(lb, ub), method="trf", max_nfev=400)
    apply_pano(sol.x)
    H = make_H(sol.x)
    final_crays = pano.src_to_direction(corner_px[:, 0], corner_px[:, 1])
    scale = (np.linalg.inv(H) @ final_crays.T)[2]
    if np.median(scale) < 0:
        H = -H
    return pano, H


@dataclass
class GroundModel:
    H: list  # 3x3, row-major
    field_length_m: float = 100.0
    field_width_m: float = 37.0

    def _H(self) -> np.ndarray:
        return np.asarray(self.H, float).reshape(3, 3)

    @classmethod
    def fit(cls, rays, xz, length_m=100.0, width_m=37.0) -> "GroundModel":
        H = solve_homography(rays, xz)
        return cls(H=H.reshape(-1).tolist(),
                   field_length_m=length_m, field_width_m=width_m)

    def ground_from_ray(self, dirs: np.ndarray) -> np.ndarray:
        """(N,3) world rays -> (N,2) metric [X, Z]."""
        d = np.asarray(dirs, float).reshape(-1, 3)
        q = (np.linalg.inv(self._H()) @ d.T).T  # (N,3) ~ [X,Z,1]
        return q[:, :2] / q[:, 2:3]

    def ray_from_ground(self, xz: np.ndarray) -> np.ndarray:
        """(N,2) metric [X,Z] -> (N,3) unit world rays."""
        g = np.column_stack([np.asarray(xz, float).reshape(-1, 2),
                              np.ones(len(np.atleast_2d(xz)))])
        d = (self._H() @ g.T).T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def in_field(self, xz: np.ndarray, margin_m: float = 0.0,
                 margin_near: float | None = None,
                 margin_far: float | None = None,
                 margin_ends: float | None = None) -> np.ndarray:
        """Inside-field test with optional per-edge margins.

        margin_m sets the same slack on all 4 edges (legacy behaviour).
        margin_near / margin_far / margin_ends override per-edge if given:
            near = camera-side sideline (Z=0); be strict to avoid sweeping
                in subs/benches standing just camera-side of the line.
            far  = opposite sideline (Z=W); usually a small positive slack.
            ends = endzone X edges (|X|=L/2); generous slack so deep cuts
                / endzone catches aren't dropped.
        """
        mn = margin_m if margin_near is None else margin_near
        mf = margin_m if margin_far is None else margin_far
        me = margin_m if margin_ends is None else margin_ends
        xz = np.asarray(xz, float).reshape(-1, 2)
        hx, w = self.field_length_m / 2.0, self.field_width_m
        return (
            (xz[:, 0] >= -hx - me) & (xz[:, 0] <= hx + me)
            & (xz[:, 1] >= -mn) & (xz[:, 1] <= w + mf)
        )

    def decompose_pose(self):
        """Decompose H into (r1, r2, r3, t) of the camera's 3x4 projection.

        Our H has columns [P_col0, P_col2, P_col3] = [r1, r3, t]; the missing
        Y column r2 = r3 x r1 once r1, r3 are scaled to unit length and made
        orthonormal. Enables head/ground projection at any world height.
        """
        H = self._H()
        a, b, c = H[:, 0], H[:, 1], H[:, 2]
        s = 2.0 / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-9)
        r1, r3, t = a * s, b * s, c * s
        M = np.column_stack([r1, r3])
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        R13 = U @ Vt
        if np.dot(R13[:, 0], r1) < 0:
            R13[:, 0] *= -1
        if np.dot(R13[:, 1], r3) < 0:
            R13[:, 1] *= -1
        r1, r3 = R13[:, 0], R13[:, 1]
        r2 = np.cross(r3, r1)
        return r1, r2, r3, t

    def head_to_ground(self, head_rays: np.ndarray,
                        height_m: float = 1.75) -> np.ndarray:
        """Project head rays to metric ground (X, Z), assuming the head sits
        at world Y=height_m above the foot point.

        Solves the 3x3 linear system d*λ = r1*X + r3*Z + r2*h + t for
        (X, Z, λ) per ray. Robust to feet occlusion (the typical sideline-
        marked-as-on-field error) because heads are far more often visible.
        """
        r1, r2, r3, t = self.decompose_pose()
        d = np.asarray(head_rays, float).reshape(-1, 3)
        rhs = -(r2 * height_m + t)
        out = np.full((len(d), 2), np.nan)
        for i, di in enumerate(d):
            A = np.column_stack([r1, r3, -di])
            try:
                sol = np.linalg.solve(A, rhs)
                if sol[2] > 0:  # positive λ -> point in front of camera
                    out[i] = sol[:2]
            except np.linalg.LinAlgError:
                pass
        return out

    def boundary_xz(self, step_m: float = 1.0) -> np.ndarray:
        """Field-rectangle perimeter, for overlay sanity checks."""
        hx, w = self.field_length_m / 2.0, self.field_width_m
        xs = np.arange(-hx, hx + step_m, step_m)
        zs = np.arange(0, w + step_m, step_m)
        top = np.column_stack([xs, np.zeros_like(xs)])
        right = np.column_stack([np.full_like(zs, hx), zs])
        bot = np.column_stack([xs[::-1], np.full_like(xs, w)])
        left = np.column_stack([np.full_like(zs, -hx), zs[::-1]])
        return np.vstack([top, right, bot, left])


def sideline_residual_m(gm: GroundModel, near_xz: np.ndarray,
                        far_xz: np.ndarray) -> dict:
    """RMS deviation of marked sideline points from their true Z.

    Near sideline should map to Z=0, far sideline to Z=field_width. A large
    value flags residual de-warp the homography could not absorb.
    """
    out = {}
    if len(near_xz):
        out["near_rms_m"] = float(np.sqrt(np.mean(
            (np.asarray(near_xz)[:, 1] - 0.0) ** 2)))
    if len(far_xz):
        out["far_rms_m"] = float(np.sqrt(np.mean(
            (np.asarray(far_xz)[:, 1] - gm.field_width_m) ** 2)))
    return out
