# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Non-interactive dewarp previews for eyeballing the projection model."""

from __future__ import annotations

import cv2
import numpy as np

from .projection import PanoModel


def grab_frame(video_path: str, t_seconds: float) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_seconds * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame at {t_seconds}s from {video_path}")
    return frame


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA
    )
    return out


def sweep(
    frame: np.ndarray,
    pitch0_values,
    vfov_out_values,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    tile_w: int = 760,
    tile_h: int = 428,
) -> np.ndarray:
    """Grid of dewarps: rows = pitch0 (source tilt), cols = virtual zoom."""
    src_h, src_w = frame.shape[:2]
    rows = []
    for p0 in pitch0_values:
        cols = []
        for vf in vfov_out_values:
            m = PanoModel(src_w=src_w, src_h=src_h, pitch0_deg=p0)
            view = m.render(frame, yaw_deg, pitch_deg, vf, tile_w, tile_h)
            cols.append(_label(view, f"pitch0={p0:g}  vfov={vf:g}  yaw={yaw_deg:g}"))
        rows.append(np.hstack(cols))
    return np.vstack(rows)


def pan_strip(
    frame: np.ndarray,
    pitch0_deg: float,
    vfov_out_deg: float,
    yaws,
    pitch_deg: float = 0.0,
    tile_w: int = 760,
    tile_h: int = 428,
) -> np.ndarray:
    """Horizontal strip showing the virtual camera panned across the panorama."""
    src_h, src_w = frame.shape[:2]
    m = PanoModel(src_w=src_w, src_h=src_h, pitch0_deg=pitch0_deg)
    tiles = [
        _label(
            m.render(frame, y, pitch_deg, vfov_out_deg, tile_w, tile_h),
            f"yaw={y:g} pitch={pitch_deg:g} vfov={vfov_out_deg:g} p0={pitch0_deg:g}",
        )
        for y in yaws
    ]
    return np.vstack(tiles)
