# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""NVDEC hardware video decode -> GPU tensors, via PyNvVideoCodec.

OpenCV's VideoCapture decodes on the CPU (~25-28 ms/frame for this 8 MP
source) and is the floor on detection throughput. NVDEC decodes on the
2080 Ti's hardware decoder (~3 ms/frame) and hands back a frame that is
*already a CUDA tensor*, so it pairs with the GPU tile remap to keep the
whole front of the pipeline GPU-resident (no CPU<->GPU frame copies).

Robustness notes (Windows + Python 3.13 + this driver):
 - The pip wheel ships two binaries, `_130` (CUDA 13) and `_121` (CUDA 12.1).
   Its auto-loader picks `_130` from the driver version, but the CUDA 13
   runtime isn't installed -- only the CUDA 12 runtime (nvidia-cuda-runtime
   -cu12, cudart64_12.dll). So we bypass the package loader and import the
   `_121` binary directly with the cudart dir on the DLL search path.
 - Everything is wrapped so any failure (missing wheel, missing runtime,
   unsupported codec) returns None and the caller falls back to OpenCV.

Output frames are interleaved RGB (HWC) uint8 on the GPU. The colorspace
differs from OpenCV's BGR by ~3 grey levels (YUV->RGB matrix), immaterial
to detection.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import site

_nvc = None
_load_error = None


def _find_cudart_dirs() -> list[str]:
    dirs = []
    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    for sp in roots:
        dirs += glob.glob(os.path.join(sp, "nvidia", "cuda_runtime", "bin"))
        dirs += glob.glob(os.path.join(sp, "nvidia", "*", "bin"))
    return [d for d in dirs if os.path.isdir(d)]


def _pkg_dir() -> str | None:
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        p = os.path.join(sp, "PyNvVideoCodec")
        if os.path.isdir(p):
            return p
    return None


def load_nvc():
    """Import the PyNvVideoCodec extension, or return None if unavailable.

    Tries the CUDA-13 binary first (if a cudart64_13 is reachable), then the
    CUDA-12 binary (cudart64_12 from nvidia-cuda-runtime-cu12). Result cached.
    """
    global _nvc, _load_error
    if _nvc is not None or _load_error is not None:
        return _nvc
    pkg = _pkg_dir()
    if pkg is None:
        _load_error = "PyNvVideoCodec not installed"
        return None
    for d in _find_cudart_dirs() + [pkg]:
        try:
            os.add_dll_directory(d)
        except Exception:
            pass
    for suffix in ("_130", "_121"):
        cand = glob.glob(os.path.join(pkg, f"PyNvVideoCodec{suffix}*.pyd"))
        if not cand:
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                "_PyNvVideoCodec", cand[0])
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            _nvc = m
            return _nvc
        except Exception as e:  # noqa: BLE001
            _load_error = f"{suffix}: {e}"
    return None


class NvVideoDecoder:
    """Sequential NVDEC decoder yielding (frame_index, gpu_rgb_tensor).

    gpu_rgb_tensor is a torch.uint8 CUDA tensor of shape (H, W, 3), RGB.
    Use is_available() before constructing; raises if NVDEC can't init.
    """

    def __init__(self, path: str, gpu_id: int = 0, batch: int = 32):
        nvc = load_nvc()
        if nvc is None:
            raise RuntimeError(f"NVDEC unavailable: {_load_error}")
        self.nvc = nvc
        self.batch = batch
        self._dec = nvc.CreateSimpleDecoder(
            path, gpu_id, 0, 0, True, 0, 0, 0, 4,
            nvc.OutputColorType.RGB, False)
        meta = self._dec.get_stream_metadata()
        self.num_frames = int(meta.num_frames)
        self.width = int(meta.width)
        self.height = int(meta.height)
        self.fps = float(meta.average_fps)

    def frames(self, f_start: int = 0, f_end: int | None = None):
        """Yield (frame_index, gpu_rgb_tensor) for [f_start, f_end)."""
        import torch
        end = self.num_frames if f_end is None else min(f_end, self.num_frames)
        if f_start > 0:
            try:
                self._dec.seek_to_index(f_start)
            except Exception:
                pass
        fi = f_start
        while fi < end:
            n = min(self.batch, end - fi)
            try:
                batch = self._dec.get_batch_frames(n)
            except Exception:
                break
            if not batch:
                break
            for fr in batch:
                if fi >= end:
                    break
                yield fi, torch.as_tensor(fr, device=f"cuda:{0}")
                fi += 1

    def close(self):
        try:
            self._dec.stop()
        except Exception:
            pass


def is_available() -> bool:
    return load_nvc() is not None
