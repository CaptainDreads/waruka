# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Plumb-line calibration.

A straight line in the real world projects, through any single-viewpoint
camera, to a bundle of rays that are *coplanar* (they share the plane through
the camera centre and the line). So for each reference the user marked as
straight-in-reality, the model is correct when the back-projected rays lie on
a common plane through the origin. Coplanarity is exact for arbitrarily wide
references (e.g. a treeline spanning most of the panorama), unlike a single
synthetic pinhole, which distorts very wide lines.

Only the genuine non-linear distortion (k1, k2) is fitted. The angular extent
(hfov/vfov) and mounting tilt/roll are *not* recoverable from straightness:
hfov/vfov scale the equirect mapping and, if left free, collapse the cost
trivially (shrinking vfov pushes every latitude to ~0 so all rays become
coplanar regardless of straightness); pitch0/roll0 are a rigid rotation of
every ray and cannot bend lines. FOV stays user-set; horizon level is solved
separately.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from .projection import PanoModel


def _plane_residuals(dirs: np.ndarray) -> np.ndarray:
    """Signed distance of each unit ray to its best-fit plane through origin.

    Zero for every point iff the rays are coplanar, i.e. the world line is
    straight under the current model.
    """
    _, _, vt = np.linalg.svd(dirs, full_matrices=False)
    normal = vt[-1]  # smallest singular vector = plane normal
    return dirs @ normal


def straightness_rms(model: PanoModel, lines) -> float:
    res = [
        _plane_residuals(model.src_to_direction(np.asarray(l)[:, 0], np.asarray(l)[:, 1]))
        for l in lines
        if len(l) >= 3
    ]
    return float(np.sqrt(np.mean(np.concatenate(res) ** 2)))


def fit_distortion(
    model: PanoModel,
    lines: list[list[tuple[float, float]]],
    fit_k2: bool = False,
    k2_reg: float = 0.02,
) -> tuple[PanoModel, float]:
    """Fit radial distortion from marked straight references.

    Needs >=2 references with >=3 points each, differing in orientation
    (e.g. a horizontal treeline plus the vertical pylon) so the fit is
    well-posed.

    By default only k1 is fitted: the k1/k2 pair is ill-conditioned, so a
    couple of pixels of marking noise can throw k2 to extreme values. A
    single-parameter radial model is robust and sufficient for this use.
    Pass fit_k2=True to also fit k2, lightly regularised toward 0 (k2_reg)
    to keep it from running away when the data does not truly support it.

    Returns the updated model and the RMS ray-coplanarity residual (radians).
    """
    usable = [np.asarray(ln, dtype=np.float64) for ln in lines if len(ln) >= 3]
    if len(usable) < 2:
        raise ValueError("need at least 2 reference lines with >=3 points each")

    def coplan(m: PanoModel) -> np.ndarray:
        return np.concatenate(
            [_plane_residuals(m.src_to_direction(p[:, 0], p[:, 1])) for p in usable]
        )

    if fit_k2:
        x0 = np.array([model.k1, model.k2])
        bounds = (np.array([-1.0, -0.5]), np.array([0.5, 0.5]))

        def residuals(x):
            m = replace(model, k1=x[0], k2=x[1])
            return np.concatenate([coplan(m), [k2_reg * x[1]]])
    else:
        x0 = np.array([model.k1])
        bounds = (np.array([-1.0]), np.array([0.5]))

        def residuals(x):
            return coplan(replace(model, k1=x[0]))

    sol = least_squares(residuals, x0, bounds=bounds, method="trf")
    if fit_k2:
        fitted = replace(model, k1=float(sol.x[0]), k2=float(sol.x[1]))
    else:
        fitted = replace(model, k1=float(sol.x[0]))
    rms = float(np.sqrt(np.mean(coplan(fitted) ** 2)))
    return fitted, rms


def level_horizon(
    model: PanoModel, horizon_line: list[tuple[float, float]]
) -> PanoModel:
    """Set pitch0/roll0 so a marked horizontal reference is level & centred.

    Rotates the source frame so the reference's ray-plane normal aligns with
    world up (+Y): the line then renders horizontal through the view centre.
    """
    flat = replace(model, pitch0_deg=0.0, roll0_deg=0.0)
    pts = np.asarray(horizon_line, dtype=np.float64)
    d = flat.src_to_direction(pts[:, 0], pts[:, 1])
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    n = vt[-1]
    if n[1] < 0:
        n = -n  # point "up"
    roll = np.arctan2(n[0], n[1])
    pitch = -np.arctan2(n[2], np.hypot(n[0], n[1]))
    return replace(
        model, pitch0_deg=float(np.degrees(pitch)), roll0_deg=float(np.degrees(roll))
    )
