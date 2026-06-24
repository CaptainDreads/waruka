# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Real-ESRGAN x2plus super-resolution wrapper (#41).

Used by `GpuRenderer` to upscale the source pano crop region before
the final resample to output resolution. Applied only when the crop
is small enough that resampling to output would otherwise be a net
upscale (typical action-zoom frames at hfov <= ~60deg on this
calibration). For wider framings (hfov >= ~75deg) the renderer
bypasses SR -- the source already has more pixels than the output.

Model: `RealESRGAN_x2plus.pth` from xinntao/Real-ESRGAN. Apache 2.0.
Architecture is the standard RRDBNet vendored at
`third_party/realesrgan/realesrgan/archs/rrdbnet_arch.py` (the
upstream basicsr pip dependency builds were unreliable on Windows +
recent torch combos -- the file is a standalone copy of basicsr's
arch with the helpers inlined).

Typical timing on an RTX 2080 Ti at fp16:

  hfov  20deg ( 486x270)  ~ 84 ms
  hfov  40deg ( 970x540)  ~300 ms
  hfov  60deg (1450x810)  ~680 ms
  hfov  80deg (1940x1090) ~1200 ms (but bypass triggers here)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VENDOR_DIR = _REPO_ROOT / "third_party" / "realesrgan"
DEFAULT_WEIGHTS_PATH = _VENDOR_DIR / "weights" / "RealESRGAN_x2plus.pth"
_RRDB_PATH = _VENDOR_DIR / "realesrgan" / "archs" / "rrdbnet_arch.py"


def _load_rrdbnet_arch():
    """Direct-import the vendored RRDBNet (sidesteps basicsr)."""
    spec = importlib.util.spec_from_file_location("rrdbnet_arch",
                                                    str(_RRDB_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RRDBNet


class SuperResolution:
    """Thin GPU-resident wrapper around Real-ESRGAN x2plus.

    All inputs / outputs are torch tensors on CUDA (the GpuRenderer is
    already on GPU; round-tripping through numpy would be wasteful).
    """

    def __init__(self, weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
                 fp16: bool = True, scale: int = 2):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("SR requires a CUDA-capable GPU")
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Real-ESRGAN weights not found: {weights_path}. "
                f"See third_party/realesrgan/")
        if scale != 2:
            raise NotImplementedError(
                f"only scale=2 is wired up (x2plus); got scale={scale}")
        RRDBNet = _load_rrdbnet_arch()
        m = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale,
                     num_feat=64, num_block=23, num_grow_ch=32)
        sd = torch.load(str(weights_path), map_location="cpu",
                          weights_only=False)
        key = "params_ema" if "params_ema" in sd else "params"
        m.load_state_dict(sd[key], strict=False)
        m.eval()
        self.dtype = torch.float16 if fp16 else torch.float32
        self.model = (m.half() if fp16 else m.float()).to("cuda")
        self.scale = int(scale)

    def upscale_tensor(self, t):
        """Upscale a (B, 3, H, W) tensor in [0, 1] by `self.scale`.

        Pads H/W to even before the call (the scale=2 RRDBNet path does
        a pixel_unshuffle by 2 internally and requires even input dims).
        The padded rows/cols are trimmed off the output, so the returned
        tensor is exactly (B, 3, H*scale, W*scale).
        """
        import torch
        import torch.nn.functional as F
        if t.dtype != self.dtype:
            t = t.to(self.dtype)
        h, w = t.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            t = F.pad(t, (0, pad_w, 0, pad_h), mode="replicate")
        with torch.no_grad():
            out = self.model(t)
        return out[..., : h * self.scale, : w * self.scale]


def make_sr_model(enable: bool = False,
                   weights_path: Optional[str | Path] = None,
                   fp16: bool = True) -> Optional[SuperResolution]:
    """Convenience factory mirroring `make_renderer()` in projection.py.

    Returns None when SR is disabled OR when prerequisites (CUDA / weights)
    are missing. The renderer treats sr_model=None as "skip SR".
    """
    if not enable:
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except Exception:
        return None
    try:
        return SuperResolution(weights_path or DEFAULT_WEIGHTS_PATH, fp16=fp16)
    except Exception as e:
        print(f"[sr] failed to initialise SR model: {e}", flush=True)
        return None
