# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Standalone video 2x super-resolution via Real-ESRGAN x2plus.

`waruka upscale input.mp4 --out output.mp4` reads any video, runs the
Real-ESRGAN x2plus model on each frame, and writes a 2x output at the
source fps. Mirrors `waruka interpolate`'s NVDEC -> GPU preprocess ->
NVENC architecture; no pair-cache or batching (SR is single-input
per-frame, the model is the dominant cost by orders of magnitude).

Performance note: at full 1440p input, SR takes ~10-15 s/frame on an
RTX 2080 Ti -- much slower than the in-renderer SR path (~1.2 s/frame
on cropped regions) because we feed it the whole frame at once. A
100-min match (120k frames) would be days; this command is really
intended for short clips (highlights, individual points, isolated
calls). For long-form upscale, use a smaller-input source or wait for
hardware that can amortise the cost.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import nvdecode
from .interpolate import (
    _preprocess_gpu_rgb, _model_out_to_bgr_numpy,
    _read_frame_nvdec, _read_frame_cv2,
)
from .progress import Progress
from .sr import SuperResolution, DEFAULT_WEIGHTS_PATH


def upscale_video(input_path: str | Path, output_path: str | Path,
                   weights_path: Optional[str | Path] = None,
                   fp16: bool = True,
                   t0: Optional[float] = None, t1: Optional[float] = None,
                   log_every: int = 50,
                   force_cv2: bool = False,
                   cq: int = 23) -> dict:
    """Upscale every frame of `input_path` by 2x using Real-ESRGAN.

    Output dimensions are exactly 2x the input. fps is preserved.
    """
    import torch
    import imageio_ffmpeg

    input_path = str(input_path)
    output_path = str(output_path)

    prog = Progress("upscale", source=input_path, out_path=output_path)
    prog.set_step("load_model")
    t_loadstart = time.time()
    sr = SuperResolution(weights_path or DEFAULT_WEIGHTS_PATH, fp16=fp16)
    prog.update(detail=f"loaded in {time.time()-t_loadstart:.1f} s")

    prog.set_step("probe_input")
    cap = cv2.VideoCapture(input_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    f0 = int(round((t0 or 0.0) * src_fps))
    f1 = int(round(t1 * src_fps)) if t1 is not None else src_n_total
    n_src = max(0, f1 - f0)
    if n_src < 1:
        prog.fail(f"need at least 1 source frame in [t0={t0}, t1={t1}]")
        raise ValueError("need at least 1 source frame")

    out_w, out_h = w * sr.scale, h * sr.scale
    out_hw = (out_h, out_w)

    def _open_writer(codec, params):
        wr = imageio_ffmpeg.write_frames(
            output_path, (out_w, out_h), pix_fmt_in="bgr24", fps=src_fps,
            codec=codec, quality=None, macro_block_size=1,
            output_params=params)
        wr.send(None)
        return wr
    # H.264 spec maxes out at 4096 in either dim. A 2x upscale of any
    # input >=2048 wide blows past that, so default to HEVC NVENC (8192
    # max). For smaller inputs the H.264 path stays preferred for
    # compatibility with players that still struggle with HEVC.
    over_h264 = (out_w > 4096) or (out_h > 4096)
    encoder = None
    if over_h264:
        try:
            writer = _open_writer("hevc_nvenc",
                                   ["-preset", "p4", "-rc", "vbr",
                                    "-cq", str(cq), "-pix_fmt", "yuv420p"])
            encoder = "hevc_nvenc"
        except Exception:
            pass
    if encoder is None and not over_h264:
        try:
            writer = _open_writer("h264_nvenc",
                                   ["-preset", "p4", "-rc", "vbr",
                                    "-cq", str(cq), "-pix_fmt", "yuv420p"])
            encoder = "h264_nvenc"
        except Exception:
            pass
    if encoder is None:
        # CPU fallback. libx265 if output is huge, libx264 otherwise.
        cpu_codec = "libx265" if over_h264 else "libx264"
        writer = _open_writer(cpu_codec,
                               ["-crf", "18", "-preset", "veryfast",
                                "-pix_fmt", "yuv420p"])
        encoder = f"{cpu_codec}-veryfast"

    use_nvdec = (not force_cv2) and nvdecode.is_available()
    frame_source = "nvdec" if use_nvdec else "cv2"
    _src_close = None
    read_next = None
    if use_nvdec:
        try:
            _dec = nvdecode.NvVideoDecoder(input_path)
            _dec_iter = _dec.frames(f_start=f0, f_end=f1)
            def read_next(_it=_dec_iter):
                return _read_frame_nvdec(_it, sr.dtype)
            _src_close = _dec.close
        except Exception as e:  # noqa: BLE001
            print(f"  NVDEC init failed ({e}); falling back to cv2",
                  flush=True)
            use_nvdec = False
            frame_source = "cv2"
    if not use_nvdec:
        _cap2 = cv2.VideoCapture(input_path)
        _cap2.set(cv2.CAP_PROP_POS_FRAMES, f0)
        def read_next(_c=_cap2):
            return _read_frame_cv2(_c, sr.dtype)
        _src_close = _cap2.release

    prog.update(detail=f"src {w}x{h} {src_fps:.1f} fps n={n_src} -> "
                        f"out {out_w}x{out_h} (scale={sr.scale}x, "
                        f"source={frame_source}, encoder={encoder})")

    prog.set_step("upscale", progress=0.0,
                   f_start=f0, f_end=f1, current_frame=f0,
                   fps_observed=0.0, eta_s=None)
    frame_times: list[float] = []
    overall_t0 = time.time()
    f_idx = 0
    while True:
        nxt = read_next()
        if nxt is None:
            break
        t, _bgr, _ = nxt
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_frame_start = time.perf_counter()
        with torch.no_grad():
            up = sr.upscale_tensor(t)
        bgr_out = _model_out_to_bgr_numpy(up, out_hw)
        writer.send(bgr_out)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        frame_times.append(time.perf_counter() - t_frame_start)
        f_idx += 1

        elapsed = time.time() - overall_t0
        fps_obs = f_idx / elapsed if elapsed > 0 else 0.0
        eta = (n_src - f_idx) / fps_obs if fps_obs > 0 else None
        prog.update(progress=f_idx / n_src,
                     current_frame=f0 + f_idx,
                     fps_observed=float(fps_obs),
                     eta_s=eta)
        if f_idx % log_every == 0:
            recent = frame_times[-log_every:]
            print(f"  frame {f_idx}/{n_src}  recent median = "
                  f"{np.median(recent)*1000:.0f} ms  "
                  f"emit fps = {fps_obs:.2f}", flush=True)

    _src_close()
    writer.close()

    ft = np.array(frame_times) * 1000.0
    report = {
        "input": input_path,
        "output": output_path,
        "encoder": encoder,
        "frame_source": frame_source,
        "src_fps": float(src_fps),
        "src_size_wh": [w, h],
        "out_size_wh": [out_w, out_h],
        "scale": int(sr.scale),
        "n_frames": int(len(frame_times)),
        "per_frame_ms_median": float(np.median(ft)) if ft.size else 0.0,
        "per_frame_ms_p90": float(np.percentile(ft, 90)) if ft.size else 0.0,
        "total_s": float(time.time() - overall_t0),
    }
    prog.done(**report)
    return report
