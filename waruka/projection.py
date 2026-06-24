# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Panorama -> rectilinear virtual camera.

The Reolink Duo 2 produces a single stitched dual-lens panorama (~180 deg
combined horizontal FOV, 4608x1728). We model it as an equirectangular-style
panorama with tunable angular extent and a fixed source orientation (the
camera's mounting tilt/roll), then render a virtual pinhole ("broadcast")
camera that can be panned (yaw), tilted (pitch) and zoomed (fov).

Reprojecting equirectangular -> pinhole straightens the panoramic curvature
and keeps verticals vertical, which is the natural broadcast look we want.
Exact lens intrinsics are unknown, so the model parameters are fitted later
from straight-line references the user marks during calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import cv2


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _panini_image_to_rays(uu, vv, out_w, fov_out_deg, panini_d):
    """Inverse Panini-General projection (image pixels -> unit world rays).

    Panini-General forward:
        u = (d+1) * sin(lambda) / (d + cos(lambda))   * scale
        v = (d+1) * tan(phi)    / (d + cos(lambda))   * scale

    For d=0 reduces to pure rectilinear (pinhole). For d=1 is classic
    stereographic Panini -- straight verticals AND much less horizontal
    bow than cylindrical, at the cost of mild edge curvature on lines
    far above/below the horizon. The 'blend' slot in the projection API
    is reused to carry the d parameter for the panini branch.
    """
    d = float(panini_d)
    hf = np.radians(fov_out_deg)
    # Pick scale so that pixel us = +/- out_w/2 maps to the FOV edges.
    u_max = (d + 1.0) * np.sin(hf / 2.0) / (d + np.cos(hf / 2.0))
    scale = (out_w / 2.0) / u_max
    s = (uu / scale) / (d + 1.0)
    # lambda = atan(s) + asin(s*d / sqrt(1+s^2))
    asin_arg = np.clip(s * d / np.sqrt(1.0 + s * s), -1.0, 1.0)
    lam = np.arctan(s) + np.arcsin(asin_arg)
    # phi from vertical: tan(phi) = (v_norm) * (d + cos(lam)) where
    # v_norm = (vv/scale) / (d+1)
    t = (vv / scale) / (d + 1.0)
    phi = np.arctan(t * (d + np.cos(lam)))
    cp = np.cos(phi)
    rays = np.stack([np.sin(lam) * cp, -np.sin(phi),
                     np.cos(lam) * cp], axis=-1)
    return rays


@dataclass
class PanoModel:
    """Equirectangular-style source model.

    Angles in degrees for ergonomic config files; converted internally.

    hfov_deg / vfov_deg: angular extent the full source image spans.
    pitch0_deg: source mounting tilt (positive = camera aimed downward, so the
        horizon sits above image centre).
    roll0_deg: source mounting roll.
    """

    src_w: int
    src_h: int
    hfov_deg: float = 190.0
    vfov_deg: float = 80.0
    pitch0_deg: float = 0.0
    roll0_deg: float = 0.0
    # Residual barrel distortion of the stitched dual-lens image. The
    # equirectangular mapping alone does not straighten this camera's lines;
    # k1/k2 are fitted from user-marked straight references (plumb-line
    # calibration). cx/cy default to image centre (set <0 to mean "centre").
    k1: float = 0.0
    k2: float = 0.0
    cx: float = -1.0
    cy: float = -1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PanoModel":
        return cls(**d)

    # --- source orientation -------------------------------------------------
    def _source_rotation(self) -> np.ndarray:
        return _rot_x(np.radians(self.pitch0_deg)) @ _rot_z(np.radians(self.roll0_deg))

    def _center(self) -> tuple[float, float]:
        cx = self.cx if self.cx >= 0 else (self.src_w - 1) / 2.0
        cy = self.cy if self.cy >= 0 else (self.src_h - 1) / 2.0
        return cx, cy

    def _norm_radius(self) -> float:
        return float(np.hypot(self.src_w / 2.0, self.src_h / 2.0))

    def _apply_distortion(self, sx: np.ndarray, sy: np.ndarray):
        """Map ideal (undistorted-equirect) pixels to the real distorted image.

        The captured frame still carries barrel distortion, so to sample the
        correct pixel we push ideal coordinates back through the forward
        radial model centred at (cx, cy).
        """
        if self.k1 == 0.0 and self.k2 == 0.0:
            return sx, sy
        cx, cy = self._center()
        rn = self._norm_radius()
        nx = (sx - cx) / rn
        ny = (sy - cy) / rn
        r2 = nx * nx + ny * ny
        fac = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        return cx + nx * fac * rn, cy + ny * fac * rn

    # --- world direction -> source pixel -----------------------------------
    def directions_to_src(self, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """dirs: (...,3) unit-ish world directions -> (map_x, map_y) float32."""
        rs = self._source_rotation()
        d = dirs @ rs.T  # apply source orientation
        dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
        lon = np.arctan2(dx, dz)
        lat = np.arcsin(np.clip(dy, -1.0, 1.0))
        hfov = np.radians(self.hfov_deg)
        vfov = np.radians(self.vfov_deg)
        sx = (lon / hfov + 0.5) * (self.src_w - 1)
        sy = (0.5 - lat / vfov) * (self.src_h - 1)
        sx, sy = self._apply_distortion(sx, sy)
        return sx.astype(np.float32), sy.astype(np.float32)

    def _undistort(self, sx: np.ndarray, sy: np.ndarray):
        """Inverse of _apply_distortion: real distorted pixels -> ideal pixels."""
        if self.k1 == 0.0 and self.k2 == 0.0:
            return sx, sy
        cx, cy = self._center()
        rn = self._norm_radius()
        nx = (sx - cx) / rn
        ny = (sy - cy) / rn
        rd = np.hypot(nx, ny)
        r = rd.copy()
        for _ in range(12):  # fixed-point inversion of the radial polynomial
            r2 = r * r
            r = rd / (1.0 + self.k1 * r2 + self.k2 * r2 * r2)
        scale = np.where(rd > 1e-9, r / np.maximum(rd, 1e-9), 1.0)
        return cx + nx * scale * rn, cy + ny * scale * rn

    def src_to_direction(self, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
        """Real source pixels -> world unit directions (inverse of view path)."""
        sx = np.asarray(sx, dtype=np.float64)
        sy = np.asarray(sy, dtype=np.float64)
        ix, iy = self._undistort(sx, sy)
        hfov = np.radians(self.hfov_deg)
        vfov = np.radians(self.vfov_deg)
        lon = (ix / (self.src_w - 1) - 0.5) * hfov
        lat = (0.5 - iy / (self.src_h - 1)) * vfov
        d_src = np.stack(
            [np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat)],
            axis=-1,
        )
        rs = self._source_rotation()
        return d_src @ rs  # world = Rs^T @ d_src  (rs orthonormal)

    # --- virtual camera -----------------------------------------------------
    def view_maps(
        self,
        yaw_deg: float,
        pitch_deg: float,
        fov_out_deg: float,
        out_w: int,
        out_h: int,
        roll_deg: float = 0.0,
        projection: str = "rectilinear",
        blend: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remap coordinates for the virtual broadcast camera.

        projection="rectilinear": pinhole, fov_out_deg is the *vertical* FOV;
        natural perspective but diverges past ~90deg of content.

        projection="cylindrical": fov_out_deg is the *horizontal* FOV and
        verticals always stay straight (good for players). `blend` controls
        how azimuth maps across the width: 1.0 = pure cylindrical (linear
        azimuth, widest, most ground/horizon curvature); 0.0 = rectilinear
        azimuth distribution (straightest horizontal lines, edge compression,
        mild player stretch). Both endpoints have the same horizontal FOV, so
        `blend` only trades line-straightness vs distortion -- never blows up.
        """
        us = np.arange(out_w, dtype=np.float64) - (out_w - 1) / 2.0
        vs = np.arange(out_h, dtype=np.float64) - (out_h - 1) / 2.0
        uu, vv = np.meshgrid(us, vs)
        if projection == "cylindrical":
            hf = np.radians(fov_out_deg)
            f_rect = (out_w / 2.0) / np.tan(hf / 2.0)
            fpr = (out_w / 2.0) / (hf / 2.0)
            az = (1.0 - blend) * np.arctan(uu / f_rect) + blend * (uu / fpr)
            yn = vv / fpr
            rays = np.stack(
                [np.sin(az), -yn, np.cos(az)], axis=-1).astype(np.float64)
        elif projection == "panini":
            rays = _panini_image_to_rays(uu, vv, out_w, fov_out_deg, blend)
        else:
            f = (out_h / 2.0) / np.tan(np.radians(fov_out_deg) / 2.0)
            rays = np.stack([uu, -vv, np.full_like(uu, f)], axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

        r_v = (
            _rot_y(np.radians(yaw_deg))
            @ _rot_x(np.radians(pitch_deg))
            @ _rot_z(np.radians(roll_deg))
        )
        world = rays @ r_v.T
        return self.directions_to_src(world)

    def world_to_view(
        self, dirs: np.ndarray, yaw_deg: float, pitch_deg: float,
        fov_out_deg: float, out_w: int, out_h: int, roll_deg: float = 0.0,
        projection: str = "rectilinear", blend: float = 1.0,
    ) -> np.ndarray:
        """World rays -> output pixels (analytic inverse of view_maps).

        Used to draw debug markers into the rendered frame. Returns (N,2)
        (u,v); points behind the camera get NaN.
        """
        d = np.asarray(dirs, float).reshape(-1, 3)
        r_v = (_rot_y(np.radians(yaw_deg)) @ _rot_x(np.radians(pitch_deg))
               @ _rot_z(np.radians(roll_deg)))
        cam = d @ r_v  # world -> camera (r_v orthonormal)
        cx, cy, cz = cam[:, 0], cam[:, 1], cam[:, 2]
        cxc = (out_w - 1) / 2.0
        cyc = (out_h - 1) / 2.0
        if projection == "cylindrical":
            hf = np.radians(fov_out_deg)
            f_rect = (out_w / 2.0) / np.tan(hf / 2.0)
            fpr = (out_w / 2.0) / (hf / 2.0)
            az = np.arctan2(cx, cz)
            horiz = np.hypot(cx, cz)
            yn = -cy / np.maximum(horiz, 1e-9)
            # invert az = (1-b)*atan(u/f_rect) + b*(u/fpr) for u (monotonic)
            u = az * fpr  # cylindrical-side initial guess
            for _ in range(8):
                g = (1 - blend) * np.arctan(u / f_rect) + blend * (u / fpr) - az
                gp = (1 - blend) * (1.0 / (f_rect * (1 + (u / f_rect) ** 2))) \
                    + blend / fpr
                u = u - g / gp
            uu = u + cxc
            vv = yn * fpr + cyc
            behind = cz <= 0
        elif projection == "panini":
            # Forward Panini: ray (cx,cy,cz) -> image (u,v)
            # lambda = atan2(cx, cz); phi = atan2(-cy, hypot(cx, cz))
            d = float(blend)  # panini_d (re-using blend param slot)
            az = np.arctan2(cx, cz)
            horiz = np.hypot(cx, cz)
            phi = np.arctan2(-cy, np.maximum(horiz, 1e-9))
            hf = np.radians(fov_out_deg)
            u_max = (d + 1) * np.sin(hf / 2) / (d + np.cos(hf / 2))
            scale = (out_w / 2.0) / u_max
            denom = d + np.cos(az)
            u_norm = (d + 1) * np.sin(az) / np.where(denom != 0, denom, 1e-9)
            v_norm = (d + 1) * np.tan(phi) / np.where(denom != 0, denom, 1e-9)
            uu = u_norm * scale + cxc
            vv = v_norm * scale + cyc
            behind = (cz <= 0) | (denom <= 1e-6)
        else:
            f = (out_h / 2.0) / np.tan(np.radians(fov_out_deg) / 2.0)
            behind = cz <= 1e-6
            uu = f * cx / np.where(behind, np.nan, cz) + cxc
            vv = -f * cy / np.where(behind, np.nan, cz) + cyc
        out = np.column_stack([uu, vv])
        out[behind] = np.nan
        return out

    def render(
        self,
        frame: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        fov_out_deg: float,
        out_w: int,
        out_h: int,
        roll_deg: float = 0.0,
        interp: int = cv2.INTER_CUBIC,
        projection: str = "rectilinear",
        blend: float = 1.0,
    ) -> np.ndarray:
        mx, my = self.view_maps(yaw_deg, pitch_deg, fov_out_deg, out_w, out_h,
                                roll_deg, projection, blend)
        return cv2.remap(
            frame, mx, my, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

    def view_outline(
        self, yaw_deg: float, pitch_deg: float, fov_out_deg: float,
        out_w: int, out_h: int, samples_per_side: int = 128,
        roll_deg: float = 0.0, projection: str = "rectilinear",
        blend: float = 1.0,
    ) -> np.ndarray:
        """Source-pixel coords of the output frame's PERIMETER only.

        Cheap alternative to view_maps when you only need the crop outline
        (e.g. the debug-pano box) -- computes ~4*samples_per_side points
        instead of the full out_w*out_h grid, avoiding the per-pixel trig
        that dominates a full render. Returns (M, 2) array of (sx, sy).
        """
        sx = np.linspace(0, out_w - 1, samples_per_side)
        sy = np.linspace(0, out_h - 1, samples_per_side)
        top = np.column_stack([sx, np.zeros_like(sx)])
        right = np.column_stack([np.full_like(sy, out_w - 1), sy])
        bot = np.column_stack([sx[::-1], np.full_like(sx, out_h - 1)])
        left = np.column_stack([np.zeros_like(sy), sy[::-1]])
        perim = np.vstack([top, right, bot, left])  # (M, 2) in output px
        us = perim[:, 0] - (out_w - 1) / 2.0
        vs = perim[:, 1] - (out_h - 1) / 2.0
        if projection == "cylindrical":
            hf = np.radians(fov_out_deg)
            f_rect = (out_w / 2.0) / np.tan(hf / 2.0)
            fpr = (out_w / 2.0) / (hf / 2.0)
            az = (1.0 - blend) * np.arctan(us / f_rect) + blend * (us / fpr)
            yn = vs / fpr
            rays = np.stack([np.sin(az), -yn, np.cos(az)], axis=-1)
        elif projection == "panini":
            rays = _panini_image_to_rays(us, vs, out_w, fov_out_deg, blend)
        else:
            f = (out_h / 2.0) / np.tan(np.radians(fov_out_deg) / 2.0)
            rays = np.stack([us, -vs, np.full_like(us, f)], axis=-1)
        rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        r_v = (_rot_y(np.radians(yaw_deg)) @ _rot_x(np.radians(pitch_deg))
               @ _rot_z(np.radians(roll_deg)))
        world = rays @ r_v.T
        mx, my = self.directions_to_src(world)
        return np.column_stack([mx, my])


class GpuRenderer:
    """CUDA renderer for the cylindrical/rectilinear virtual camera.

    The CPU `PanoModel.render` recomputes a 2560x1440 projection map (per-
    pixel atan2/asin/radial trig) every frame, ~1 s/frame -- the render
    bottleneck. This does the identical math in torch on the GPU (idle
    during render) and samples with grid_sample, ~30-50x faster. Falls back
    to the CPU path if torch/CUDA is unavailable (see make_renderer()).

    The output-frame pixel grid is constant, so it's cached on the device
    once; per frame only the rotation + projection + sampling run.
    """

    def __init__(self, pano: "PanoModel", out_w: int, out_h: int,
                 projection: str, blend: float, device: str = "cuda",
                 edge_fill_mode: str = "zeros",
                 edge_fill_blur_deg: float = 10.0,
                 edge_fill_blur_sigma_px: float = 40.0,
                 edge_fill_blur_boundary_sigma_px: float = 8.0,
                 edge_fill_blur_fade_deg: float = 2.0,
                 sr_model=None, sr_min_upscale: float = 0.0):
        """edge_fill_mode controls what the renderer returns for rays
        that fall outside the pano's vfov coverage:
          "zeros"  - black (grid_sample padding_mode="zeros")
          "border" - clamp to nearest edge row (padding_mode="border")
          "blur"   - pre-pad the source image (caller must pass the
                     padded frame to render(); see _pad_source_for_blur)
        """
        import torch
        self.t = torch
        self.pano = pano
        self.out_w, self.out_h = out_w, out_h
        self.projection = projection
        self.blend = blend
        self.device = device
        self.edge_fill_mode = edge_fill_mode
        self.edge_fill_blur_deg = float(edge_fill_blur_deg)
        self.edge_fill_blur_sigma_px = float(edge_fill_blur_sigma_px)
        self.edge_fill_blur_boundary_sigma_px = float(edge_fill_blur_boundary_sigma_px)
        self.edge_fill_blur_fade_deg = float(edge_fill_blur_fade_deg)
        # Fade-band rows in original pano coords (used when caller calls
        # pad_source_for_blur).
        self._fade_rows = (int(round(self.edge_fill_blur_fade_deg
                                      * pano.src_h / pano.vfov_deg))
                           if edge_fill_mode == "blur" else 0)
        # For "blur" we render against a virtual pano with extended
        # vfov + height. The src_h shift is fixed at init time so the
        # grid math is consistent every frame.
        if edge_fill_mode == "blur":
            self._pad_rows = int(round(self.edge_fill_blur_deg
                                       * pano.src_h / pano.vfov_deg))
            self._padded_src_h = pano.src_h + 2 * self._pad_rows
            self._padded_vfov_deg = pano.vfov_deg + 2 * self.edge_fill_blur_deg
        else:
            self._pad_rows = 0
            self._padded_src_h = pano.src_h
            self._padded_vfov_deg = pano.vfov_deg
        # grid_sample padding mode:
        #   "zeros"  - literal: returns black for samples past source
        #   "border" - clamp to edge row (the natural edge-fill)
        #   "blur"   - same as "border" for the gs call, because the
        #              padded source's outermost row is already heavily
        #              blurred sky-like content. Asking past the padded
        #              edge clamps to that already-blurred row, which
        #              is exactly what we want (no visible black moon
        #              from the cases where padded vfov isn't quite
        #              enough -- e.g. clip 1's asymmetric mount needs
        #              10.5deg of padding but we have 10deg).
        self._gs_padding = ("zeros" if edge_fill_mode == "zeros"
                            else "border")
        us = torch.arange(out_w, dtype=torch.float32, device=device) - (out_w - 1) / 2.0
        vs = torch.arange(out_h, dtype=torch.float32, device=device) - (out_h - 1) / 2.0
        vv, uu = torch.meshgrid(vs, us, indexing="ij")
        self.uu, self.vv = uu, vv
        rs = pano._source_rotation()  # numpy 3x3 (identity when pitch0=roll0=0)
        self.rs = torch.tensor(rs, dtype=torch.float32, device=device)
        # Super-resolution (#41). When sr_model is set, the renderer
        # crops the source pano to the bbox of the sampling grid, runs
        # the SR model on the crop, and samples from the upscaled crop.
        # sr_min_upscale=0.0 (the default) means SR runs every frame --
        # there's no per-frame bypass. Earlier versions bypassed SR at
        # wide framings where the source crop was already larger than
        # the output, but the transition was visually noticeable as a
        # "pop" so the default is constant SR. Set this above 1.0 if
        # you want SR to skip when the natural upscale falls below the
        # threshold (accepting the visible transition).
        self.sr_model = sr_model
        self.sr_min_upscale = float(sr_min_upscale)

    def render(self, frame: np.ndarray, yaw_deg: float, pitch_deg: float,
               fov_out_deg: float, roll_deg: float = 0.0,
               blend: float | None = None) -> np.ndarray:
        """Render one frame.

        `blend` overrides self.blend for this call (used for adaptive
        Panini d where d changes per frame). When None, self.blend is
        used (legacy fixed-d behaviour).
        """
        torch = self.t
        W, H = self.out_w, self.out_h
        b = self.blend if blend is None else float(blend)
        if self.projection == "cylindrical":
            hf = np.radians(fov_out_deg)
            f_rect = (W / 2.0) / np.tan(hf / 2.0)
            fpr = (W / 2.0) / (hf / 2.0)
            az = ((1.0 - b) * torch.atan(self.uu / f_rect)
                  + b * (self.uu / fpr))
            yn = self.vv / fpr
            rays = torch.stack([torch.sin(az), -yn, torch.cos(az)], dim=-1)
        elif self.projection == "panini":
            d = b   # blend slot carries panini_d
            hf = np.radians(fov_out_deg)
            u_max = (d + 1.0) * np.sin(hf / 2.0) / (d + np.cos(hf / 2.0))
            scale = (W / 2.0) / u_max
            s = (self.uu / scale) / (d + 1.0)
            asin_arg = (s * d / torch.sqrt(1.0 + s * s)).clamp(-1.0, 1.0)
            lam = torch.atan(s) + torch.asin(asin_arg)
            t = (self.vv / scale) / (d + 1.0)
            phi = torch.atan(t * (d + torch.cos(lam)))
            cp = torch.cos(phi)
            rays = torch.stack([torch.sin(lam) * cp, -torch.sin(phi),
                                torch.cos(lam) * cp], dim=-1)
        else:
            f = (H / 2.0) / np.tan(np.radians(fov_out_deg) / 2.0)
            rays = torch.stack([self.uu, -self.vv,
                                torch.full_like(self.uu, float(f))], dim=-1)
        rays = rays / rays.norm(dim=-1, keepdim=True)
        r_v = (_rot_y(np.radians(yaw_deg)) @ _rot_x(np.radians(pitch_deg))
               @ _rot_z(np.radians(roll_deg)))
        # d = rays @ r_v.T @ rs.T = rays @ (rs @ r_v).T
        M = self.rs @ torch.tensor(r_v, dtype=torch.float32, device=self.device)
        d = rays @ M.t()
        dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
        lon = torch.atan2(dx, dz)
        lat = torch.asin(dy.clamp(-1.0, 1.0))
        hfov_p = np.radians(self.pano.hfov_deg)
        # For "blur" mode the rendered virtual pano is taller and has a
        # bigger vfov; the original-content rows sit in the middle of
        # the padded image. Use the effective dimensions for the
        # sx/sy computation so a ray at lat=0 still hits the right pixel.
        vfov_p = np.radians(self._padded_vfov_deg)
        sx = (lon / hfov_p + 0.5) * (self.pano.src_w - 1)
        sy = (0.5 - lat / vfov_p) * (self._padded_src_h - 1)
        if self.pano.k1 != 0.0 or self.pano.k2 != 0.0:
            # Distortion is in the *original* image's pixel space.
            # Shift sy into the original-rows frame, distort, then
            # shift back.
            cx, cy = self.pano._center()
            rn = self.pano._norm_radius()
            sy_orig = sy - self._pad_rows
            nx = (sx - cx) / rn
            ny = (sy_orig - cy) / rn
            r2 = nx * nx + ny * ny
            fac = 1.0 + self.pano.k1 * r2 + self.pano.k2 * r2 * r2
            sx = cx + nx * fac * rn
            sy = cy + ny * fac * rn + self._pad_rows
        # Upload source as a single normalized fp tensor.
        f_t = (torch.from_numpy(np.ascontiguousarray(frame)).to(self.device)
               .permute(2, 0, 1).unsqueeze(0).float())
        # Optional SR pass on the source crop bbox (#41).
        if self.sr_model is not None:
            sr_used, f_t, sx_eff, sy_eff, w_eff, h_eff = self._maybe_sr_source(
                f_t, sx, sy)
        else:
            sr_used = False
            sx_eff, sy_eff = sx, sy
            w_eff = self.pano.src_w
            h_eff = self._padded_src_h
        gx = 2.0 * sx_eff / (w_eff - 1) - 1.0
        gy = 2.0 * sy_eff / (h_eff - 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1,H,W,2)
        out = torch.nn.functional.grid_sample(
            f_t, grid, mode="bilinear", align_corners=True,
            padding_mode=self._gs_padding)
        return (out.squeeze(0).permute(1, 2, 0)
                .clamp(0, 255).byte().cpu().numpy())

    def _maybe_sr_source(self, f_t, sx, sy):
        """If SR helps this frame, crop the source to the grid bbox,
        upscale, and return adjusted grid coords. Returns:

            (sr_used, source_tensor, sx_adjusted, sy_adjusted, src_w, src_h)

        Bypasses SR (returns the unmodified inputs) when:
          * the crop is already as large as needed for the output
            (upscale ratio < self.sr_min_upscale), or
          * the crop bbox is degenerate / empty.
        """
        torch = self.t
        # Grid bbox in source pano coords. Clamp to the source extents
        # so a slightly-out-of-bounds grid (which grid_sample would
        # silently pad anyway) doesn't ask SR for negative dims.
        sx_min = max(0, int(sx.min().item()))
        sx_max = min(self.pano.src_w - 1,
                     int(sx.ceil().max().item()))
        sy_min = max(0, int(sy.min().item()))
        sy_max = min(self._padded_src_h - 1,
                     int(sy.ceil().max().item()))
        crop_w = sx_max - sx_min + 1
        crop_h = sy_max - sy_min + 1
        if crop_w < 8 or crop_h < 8:
            return False, f_t, sx, sy, self.pano.src_w, self._padded_src_h
        # Natural upscale ratio without SR.
        upscale_ratio = max(self.out_w / crop_w, self.out_h / crop_h)
        if upscale_ratio < self.sr_min_upscale:
            return False, f_t, sx, sy, self.pano.src_w, self._padded_src_h
        # Crop the source tensor and normalise to [0, 1] for the SR model.
        crop_t = (f_t[:, :, sy_min:sy_max + 1, sx_min:sx_max + 1] / 255.0)
        sr_t = self.sr_model.upscale_tensor(crop_t).clamp(0, 1) * 255.0
        sr_t = sr_t.float()
        sr_h, sr_w = sr_t.shape[-2:]
        s = self.sr_model.scale
        # New source coords inside the upscaled crop.
        sx_adj = (sx - sx_min) * s
        sy_adj = (sy - sy_min) * s
        return True, sr_t, sx_adj, sy_adj, sr_w, sr_h


def compute_required_pad_deg(pano: "PanoModel", hfovs, pitches, ds,
                              aspect: float,
                              safety_deg: float = 2.0) -> float:
    """Closed-form pad_deg needed so the blur extension covers every
    frame's ray demand at the top/bottom of the output, given a
    calibration and a campath.

    The renderer maps top-center output pixel -> source-lat:
        lat_top_src = phi_top - pv - pitch0
        lat_bot_src = -(phi_top + pv + pitch0)
    where phi_top is the Panini-General top-center ray pitch:
        phi_top = atan((d+1)*aspect*sin(hf/2) / (d+cos(hf/2)))

    pano covers source-lat in [-vfov/2, +vfov/2]. Padding must extend
    both sides to cover any frame's max demand:
        overflow_top = max(0, max(lat_top_src) - vfov/2)
        overflow_bot = max(0, -min(lat_bot_src) - vfov/2)
    Symmetric pad means using max of both, plus a safety margin so we
    don't sit exactly at the boundary (border-clamp would still handle
    that gracefully, but the gradient looks cleaner with a small buffer).

    Args:
        pano: source calibration (uses vfov_deg, pitch0_deg)
        hfovs, pitches, ds: per-frame arrays from the campath
        aspect: out_h / out_w of the campath output
        safety_deg: extra buffer added to the computed minimum

    Returns: pad_deg in degrees (>= safety_deg, even when no overflow).
    """
    hfovs = np.asarray(hfovs, float)
    pitches = np.asarray(pitches, float)
    ds = np.asarray(ds, float)
    hf_rad = np.radians(hfovs)
    phi_top = np.degrees(np.arctan(
        (ds + 1.0) * aspect * np.sin(hf_rad / 2.0) /
        np.maximum(ds + np.cos(hf_rad / 2.0), 1e-9)))
    lat_top = phi_top - pitches - pano.pitch0_deg
    lat_bot = -(phi_top + pitches + pano.pitch0_deg)
    half_v = pano.vfov_deg / 2.0
    overflow_top = float(np.maximum(0.0, lat_top - half_v).max())
    overflow_bot = float(np.maximum(0.0, -lat_bot - half_v).max())
    return max(overflow_top, overflow_bot) + float(safety_deg)


# Module-level torch availability cache. None = not yet probed, True/False
# after the first call. Lets pad_source_for_blur dispatch to GPU with one
# import attempt per process lifetime instead of per call. Forced back to
# False on any per-call GPU failure so we don't keep retrying a broken path.
_TORCH_CUDA_OK: bool | None = None


def _torch_cuda_available() -> bool:
    global _TORCH_CUDA_OK
    if _TORCH_CUDA_OK is None:
        try:
            import torch
            _TORCH_CUDA_OK = bool(torch.cuda.is_available())
        except Exception:
            _TORCH_CUDA_OK = False
    return _TORCH_CUDA_OK


def pad_source_for_blur(frame: np.ndarray, pad_rows: int,
                         sigma_max_px: float = 40.0,
                         sigma_boundary_px: float = 8.0,
                         fade_rows: int = 0) -> np.ndarray:
    """Pre-pad a pano frame with progressively-blurred extensions, plus
    an optional fade band inside the original that hides the seam.

    Region structure (top, mirror for bottom):

      pad row 0 ........... blur sigma = sigma_max_px       (heaviest,
                                                              top of image)
      pad row 1
      ...
      pad row pad_rows-1 ... blur sigma = sigma_boundary_px (seam)
      ------------- seam between padding and original -------------
      original row 0 ....... blur sigma = sigma_boundary_px
      original row 1
      ...
      original row fade_rows-1   blur sigma = 0  (unmodified)
      ------------- interior (untouched) -------------
        ...

    Blur is continuous across the seam: row pad_rows-1 (just-above-
    original) and original row 0 (just-below-padding) both have sigma =
    sigma_boundary_px. No sharp "now-blur-mode-engages" boundary, and
    distance into the padded extension keeps increasing the blur so
    vertical line-replication artefacts get smeared out.

    Args:
        frame: source pano (h, w, 3) -- typically 1728 x 4608 x 3
        pad_rows: rows of padding to add above and below
        sigma_max_px: horizontal blur sigma at the outermost padded row
        sigma_boundary_px: blur sigma at the seam (smaller; small enough
            that the boundary-blurred original row is still mostly
            recognisable as continuation of the visible content)
        fade_rows: rows INSIDE the original to fade-blur (each row a
            weighted average between original and boundary-blurred copy).
            0 = no fade band (sharp seam at the boundary blur level).

    Returns: array of shape (h + 2*pad_rows, w, 3).

    Implementation: if torch + CUDA are available, the broadcast-blend
    arithmetic runs on the GPU (~10 ms vs ~50 ms on CPU at the standard
    1728x4608 pano size); otherwise it falls back to the numpy version.
    Output is bit-identical or off by <=1 LSB per channel from rounding.
    """
    if pad_rows <= 0:
        return frame
    if _torch_cuda_available():
        try:
            return _pad_source_for_blur_gpu(
                frame, pad_rows,
                sigma_max_px=sigma_max_px,
                sigma_boundary_px=sigma_boundary_px,
                fade_rows=fade_rows)
        except Exception:
            # Any failure (OOM, driver hiccup, etc.) -- disable the GPU
            # path for the rest of the process and continue on CPU.
            global _TORCH_CUDA_OK
            _TORCH_CUDA_OK = False
    return _pad_source_for_blur_cpu(
        frame, pad_rows,
        sigma_max_px=sigma_max_px,
        sigma_boundary_px=sigma_boundary_px,
        fade_rows=fade_rows)


def _pad_seed_rows(frame: np.ndarray, sigma_max_px: float,
                    sigma_boundary_px: float) -> tuple[np.ndarray, ...]:
    """Compute the four 1-row blur seeds via cv2 (shared CPU/GPU path).

    These are tiny (1 x w x 3) and cv2.GaussianBlur runs in well under a
    millisecond at sigma=40 / w=4608, so there is no benefit to porting
    them to GPU. Keeping them on CPU also ensures bit-exact agreement
    between the CPU and GPU implementations of the surrounding math.
    """
    def kernel(sigma):
        return max(3, int(6 * sigma) | 1)

    top_row = frame[0:1]
    bot_row = frame[-1:]

    if sigma_boundary_px > 0.5:
        seed_top_b = cv2.GaussianBlur(
            top_row, (kernel(sigma_boundary_px), 1),
            float(sigma_boundary_px))
        seed_bot_b = cv2.GaussianBlur(
            bot_row, (kernel(sigma_boundary_px), 1),
            float(sigma_boundary_px))
    else:
        seed_top_b = top_row.copy()
        seed_bot_b = bot_row.copy()

    if sigma_max_px > 0.5:
        seed_top_h = cv2.GaussianBlur(
            top_row, (kernel(sigma_max_px), 1),
            float(sigma_max_px))
        seed_bot_h = cv2.GaussianBlur(
            bot_row, (kernel(sigma_max_px), 1),
            float(sigma_max_px))
    else:
        seed_top_h = top_row.copy()
        seed_bot_h = bot_row.copy()
    return seed_top_h, seed_top_b, seed_bot_h, seed_bot_b


def _pad_source_for_blur_cpu(frame: np.ndarray, pad_rows: int,
                              sigma_max_px: float = 40.0,
                              sigma_boundary_px: float = 8.0,
                              fade_rows: int = 0) -> np.ndarray:
    """numpy fallback for pad_source_for_blur. See the wrapper for docs."""
    if pad_rows <= 0:
        return frame
    h, w = frame.shape[:2]
    seed_top_h, seed_top_b, seed_bot_h, seed_bot_b = _pad_seed_rows(
        frame, sigma_max_px, sigma_boundary_px)

    # Pre-allocate the full output buffer once and fill regions in
    # place. Avoiding the np.vstack at the end + reusing intermediate
    # float32 buffers cuts pad_source_for_blur from ~130 ms to ~50 ms
    # per call on a 1728x4608 source.
    out = np.empty((h + 2 * pad_rows, w, 3), dtype=np.uint8)
    seed_top_h_f = seed_top_h.astype(np.float32)
    seed_top_b_f = seed_top_b.astype(np.float32)
    seed_bot_h_f = seed_bot_h.astype(np.float32)
    seed_bot_b_f = seed_bot_b.astype(np.float32)

    # Top padding: blend from heavy (top of image) to boundary
    # (just-above-original). In-place math, single cast at the end.
    w_top = np.linspace(1.0, 0.0, pad_rows, endpoint=True,
                         dtype=np.float32).reshape(-1, 1, 1)
    top_pad = w_top * seed_top_h_f
    top_pad += (1.0 - w_top) * seed_top_b_f
    np.clip(top_pad, 0, 255, out=top_pad)
    out[:pad_rows] = top_pad.astype(np.uint8)

    # Bottom padding: mirror.
    w_bot = np.linspace(0.0, 1.0, pad_rows, endpoint=True,
                         dtype=np.float32).reshape(-1, 1, 1)
    bot_pad = w_bot * seed_bot_h_f
    bot_pad += (1.0 - w_bot) * seed_bot_b_f
    np.clip(bot_pad, 0, 255, out=bot_pad)
    out[-pad_rows:] = bot_pad.astype(np.uint8)

    # Original middle (with optional fade band).
    if fade_rows > 0 and fade_rows * 2 < h:
        # Top fade: row 0 = fully boundary-blurred, row fade_rows-1 = original.
        fw_top = np.linspace(1.0, 0.0, fade_rows, endpoint=True,
                              dtype=np.float32).reshape(-1, 1, 1)
        top_fade = fw_top * seed_top_b_f
        top_fade += (1.0 - fw_top) * frame[:fade_rows].astype(np.float32)
        np.clip(top_fade, 0, 255, out=top_fade)
        out[pad_rows:pad_rows + fade_rows] = top_fade.astype(np.uint8)

        out[pad_rows + fade_rows:pad_rows + h - fade_rows] = frame[fade_rows:h - fade_rows]

        # Bottom fade: mirror.
        fw_bot = np.linspace(0.0, 1.0, fade_rows, endpoint=True,
                              dtype=np.float32).reshape(-1, 1, 1)
        bot_fade = fw_bot * seed_bot_b_f
        bot_fade += (1.0 - fw_bot) * frame[-fade_rows:].astype(np.float32)
        np.clip(bot_fade, 0, 255, out=bot_fade)
        out[pad_rows + h - fade_rows:pad_rows + h] = bot_fade.astype(np.uint8)
    else:
        out[pad_rows:pad_rows + h] = frame
    return out


def _pad_source_for_blur_gpu(frame: np.ndarray, pad_rows: int,
                              sigma_max_px: float = 40.0,
                              sigma_boundary_px: float = 8.0,
                              fade_rows: int = 0) -> np.ndarray:
    """torch-CUDA implementation of pad_source_for_blur.

    Strategy: keep the 4 tiny 1-row Gaussian blurs on CPU via cv2 (sub-
    millisecond, bit-exact with the CPU path), then move the heavy
    broadcast-blend arithmetic to GPU. The frame and seeds are uploaded
    once, the output is downloaded once.

    Bit-exactness: arithmetic differs from the numpy version only by the
    rounding behaviour of float32 add/mul, which agrees to within 1 LSB
    per channel in practice.
    """
    import torch

    if pad_rows <= 0:
        return frame
    h, w = frame.shape[:2]
    seed_top_h, seed_top_b, seed_bot_h, seed_bot_b = _pad_seed_rows(
        frame, sigma_max_px, sigma_boundary_px)

    device = torch.device("cuda")
    # Upload everything as float32 -- the broadcast-blend is float math.
    # frame_t stays uint8 so the middle-copy region needs no rounding.
    frame_t = torch.from_numpy(frame).to(device, non_blocking=True)
    seed_top_h_t = torch.from_numpy(seed_top_h).to(
        device, dtype=torch.float32, non_blocking=True)
    seed_top_b_t = torch.from_numpy(seed_top_b).to(
        device, dtype=torch.float32, non_blocking=True)
    seed_bot_h_t = torch.from_numpy(seed_bot_h).to(
        device, dtype=torch.float32, non_blocking=True)
    seed_bot_b_t = torch.from_numpy(seed_bot_b).to(
        device, dtype=torch.float32, non_blocking=True)

    out_t = torch.empty((h + 2 * pad_rows, w, 3),
                         dtype=torch.uint8, device=device)

    # Top padding: w_top * heavy + (1 - w_top) * boundary, linspaced rowwise.
    w_top = torch.linspace(1.0, 0.0, pad_rows, device=device,
                            dtype=torch.float32).view(-1, 1, 1)
    top_pad = w_top * seed_top_h_t + (1.0 - w_top) * seed_top_b_t
    out_t[:pad_rows] = top_pad.clamp_(0, 255).to(torch.uint8)

    # Bottom padding: mirror.
    w_bot = torch.linspace(0.0, 1.0, pad_rows, device=device,
                            dtype=torch.float32).view(-1, 1, 1)
    bot_pad = w_bot * seed_bot_h_t + (1.0 - w_bot) * seed_bot_b_t
    out_t[-pad_rows:] = bot_pad.clamp_(0, 255).to(torch.uint8)

    # Original middle (with optional fade band).
    if fade_rows > 0 and fade_rows * 2 < h:
        fw_top = torch.linspace(1.0, 0.0, fade_rows, device=device,
                                 dtype=torch.float32).view(-1, 1, 1)
        top_fade = (fw_top * seed_top_b_t
                    + (1.0 - fw_top) * frame_t[:fade_rows].float())
        out_t[pad_rows:pad_rows + fade_rows] = (
            top_fade.clamp_(0, 255).to(torch.uint8))

        out_t[pad_rows + fade_rows:pad_rows + h - fade_rows] = (
            frame_t[fade_rows:h - fade_rows])

        fw_bot = torch.linspace(0.0, 1.0, fade_rows, device=device,
                                 dtype=torch.float32).view(-1, 1, 1)
        bot_fade = (fw_bot * seed_bot_b_t
                    + (1.0 - fw_bot) * frame_t[-fade_rows:].float())
        out_t[pad_rows + h - fade_rows:pad_rows + h] = (
            bot_fade.clamp_(0, 255).to(torch.uint8))
    else:
        out_t[pad_rows:pad_rows + h] = frame_t

    return out_t.cpu().numpy()


def make_renderer(pano: "PanoModel", out_w: int, out_h: int,
                  projection: str, blend: float,
                  edge_fill_mode: str = "zeros",
                  edge_fill_blur_deg: float = 10.0,
                  edge_fill_blur_sigma_px: float = 40.0,
                  edge_fill_blur_boundary_sigma_px: float = 8.0,
                  edge_fill_blur_fade_deg: float = 2.0,
                  sr_model=None, sr_min_upscale: float = 0.0):
    """Return a GpuRenderer if torch+CUDA are available, else None (caller
    falls back to pano.render on CPU)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return GpuRenderer(pano, out_w, out_h, projection, blend,
                           edge_fill_mode=edge_fill_mode,
                           edge_fill_blur_deg=edge_fill_blur_deg,
                           edge_fill_blur_sigma_px=edge_fill_blur_sigma_px,
                           edge_fill_blur_boundary_sigma_px=edge_fill_blur_boundary_sigma_px,
                           edge_fill_blur_fade_deg=edge_fill_blur_fade_deg,
                           sr_model=sr_model, sr_min_upscale=sr_min_upscale)
    except Exception:
        return None
