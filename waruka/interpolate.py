# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Output-side frame interpolation (#18).

`waruka interpolate broadcast.mp4 --fps 60 --out smooth.mp4` reads a
rendered broadcast video, runs a frame interpolation model, and writes
a higher-fps mp4. Designed as a post-render step (does not touch
GpuRenderer). Renamed from `waruka polish` in v0.15 -- earlier versions
used "polish" as a generic umbrella but it never grew other steps;
"interpolate" describes what this actually does.

Two backends are supported:

* **rife** (default; recommended) -- RIFE 4.25 from `third_party/rife/`.
  ~250 ms/pair end-to-end at 2560x1440 on an RTX 2080 Ti, putting a
  100-min match at 3x at roughly 16 h of overnight compute.

* **film** -- FILM-Style from `third_party/film/film_net_fp16.pt`. About
  4x slower than RIFE end-to-end; only worth picking when you want
  FILM's slightly softer / more natural-looking in-betweens on very
  large motion AND you're willing to wait days for a long match. A
  100-min match at 3x is ~66 h of compute on the same GPU. Hits a
  cuDNN-kernel-selection cliff at 1440p single-call (~4 s/pair) so
  inputs >= 1920 px wide auto-route through a tile-and-stitch path
  (~800 ms/pair).

Toggles:
* `--backend` -- "rife" (default) or "film".
* `--fps` -- target output fps; must be an integer multiple of source.
   Default 60 (3x from a 20 fps render).
* `--model` -- override the model file path.
* `--fp32` -- use float32 instead of float16 (slower; for debugging).
* `--no-tile` -- force single-call inference (FILM only; default
   auto-tiles at width >= 1920).
* `--no-nvdec` -- force cv2 H264 decode instead of NVDEC. Default uses
   NVDEC + GPU-resident preprocess (~40% wall-time saving at 1440p,
   v0.15.1 in #43.1+2). Pick cv2 if you care about exact colour parity
   with v0.14 -- NVDEC's YUV->RGB matrix differs from cv2's; identical
   on grass/sky, slight shift on saturated jerseys.
* `--cq` -- NVENC constant-quality target (0-51, lower=better).
   Default 23 keeps source-frame quality close to the input. CQ 26-28
   halves file size with marginal quality loss. CQ 30 is the
   "model-floor" point: further reduction barely changes quality
   because the model-generated mid frames become the limit. Pre-#42
   (v0.15.x) effectively ran at default NVENC rate-control which
   under-budgeted bits at 60 fps and caused visible re-encode blur
   plus a 2-second VBR ramp-up transient.
* `--t0`, `--t1` -- process only a time window.

Background -- the cliff diagnostic and the 3-way quality comparison
that decided the default backend live in `_spike_film_cliff.py`,
`_bench_film_l1_vs_style.py`, `_bench_rife.py` and
`_spike_rife_vs_film_quality.py`.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import nvdecode
from .progress import Progress

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FILM_MODEL_PATH = _REPO_ROOT / "third_party" / "film" / "film_net_fp16.pt"
DEFAULT_RIFE_DIR = _REPO_ROOT / "third_party" / "rife"

# Width at or above which the FILM tile-stitch path is preferred. The cuDNN
# kernel-selection cliff sits between 1920 and 2560 px on the 2080 Ti; we
# tile from 1920 so anything above 1080p uses the safe path.
_FILM_TILE_WIDTH_THRESHOLD = 1920
# Half-tile overlap (each side of the centre); blends linearly across
# 2 * OVERLAP_PX columns. 128 px is enough to absorb FILM's flow
# disagreement between left and right halves on the test footage.
_TILE_OVERLAP_PX = 128
# Both FILM and RIFE conv stacks require multiples of 64.
_PAD_ALIGN = 64


# --------------------------------------------------------------------------
# Image <-> tensor helpers (shared by both backends)
# --------------------------------------------------------------------------

def _pad_align(img: np.ndarray, align: int = _PAD_ALIGN
                ) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = img.shape[:2]
    h_pad = (align - h % align) if h % align else 0
    w_pad = (align - w % align) if w % align else 0
    if h_pad or w_pad:
        img = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), mode="constant")
    return img, (h, w)


def _to_device_tensor(bgr_padded, dtype, device):
    import torch
    rgb = cv2.cvtColor(bgr_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(
        device, dtype=dtype, non_blocking=True)


def _to_bgr(t, orig_hw: tuple[int, int]) -> np.ndarray:
    h, w = orig_hw
    x = t.clamp(0, 1).float().squeeze(0).permute(1, 2, 0).cpu().numpy()
    x = (x * 255.0).astype(np.uint8)[:h, :w]
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# GPU-resident frame helpers (NVDEC fast path, quick wins #1 + #2 in #43)
# --------------------------------------------------------------------------

def _preprocess_gpu_rgb(rgb_hwc_u8, dtype):
    """NVDEC-style HWC uint8 RGB CUDA tensor -> (NCHW fp [0,1] padded, orig_hw).

    Does the work of `_to_device_tensor` entirely on the GPU: no host copy,
    no cv2.cvtColor, no numpy intermediate.
    """
    import torch.nn.functional as F
    h, w = int(rgb_hwc_u8.shape[0]), int(rgb_hwc_u8.shape[1])
    t = rgb_hwc_u8.permute(2, 0, 1).unsqueeze(0).to(dtype).div_(255.0)
    h_pad = (_PAD_ALIGN - h % _PAD_ALIGN) if h % _PAD_ALIGN else 0
    w_pad = (_PAD_ALIGN - w % _PAD_ALIGN) if w % _PAD_ALIGN else 0
    if h_pad or w_pad:
        t = F.pad(t, (0, w_pad, 0, h_pad))
    return t, (h, w)


def _model_out_to_bgr_numpy(pred, orig_hw):
    """NCHW fp [0,1] model output -> HWC u8 BGR numpy.

    Crop, scale, channel-flip, cast all happen on the GPU; one device->host
    transfer at the end. Replaces `_to_bgr` on the GPU fast path.
    """
    h, w = orig_hw
    x = (pred.clamp(0, 1).float() * 255.0).squeeze(0).byte()  # CHW u8 RGB
    x = x[:, :h, :w][[2, 1, 0]].permute(1, 2, 0).contiguous()  # HWC u8 BGR
    return x.cpu().numpy()


def _gpu_rgb_to_bgr_numpy(rgb_hwc_u8):
    """HWC u8 RGB CUDA tensor -> HWC u8 BGR numpy (for unmodified source pass-thru)."""
    return rgb_hwc_u8[:, :, [2, 1, 0]].contiguous().cpu().numpy()


# --------------------------------------------------------------------------
# Backend wrapper -- normalises the inference call across FILM and RIFE
# --------------------------------------------------------------------------

class InterpBackend:
    """Common interface around the two interpolation models.

    Hides the differences:
      * FILM expects `model(a, b, dt_scalar_tensor)` where dt is shape (1, 1).
      * RIFE expects `model.inference(a, b, timestep_tensor, scale=1.0)`
        where timestep is shape (1, 1, 1, 1).
    """

    def __init__(self, name: str, model, *, dtype):
        self.name = name
        self.model = model
        self.dtype = dtype

    def interp_tensor(self, a_t, b_t, dt: float):
        import torch
        if self.name == "film":
            dt_t = a_t.new_full((1, 1), dt)
            with torch.no_grad():
                return self.model(a_t, b_t, dt_t)
        # RIFE
        ts_t = a_t.new_full((1, 1, 1, 1), dt)
        with torch.no_grad():
            return self.model.inference(a_t, b_t, ts_t, scale=1.0)

    def interp_tensor_multi(self, a_t, b_t, dts):
        """Run multiple timesteps on the same (a, b) pair. Returns a list
        of (1, 3, H, W) tensors, one per dt.

        For RIFE the dts are batched into a single forward pass: a_t/b_t
        are repeated along the batch dim and the timestep tensor is
        shape (N, 1, 1, 1). RIFE's IFNet treats each batch element
        independently, so different dts in the same batch is fine and
        gives sub-linear scaling vs N sequential calls (quick win #4
        in #43). For FILM we fall back to sequential -- the TorchScript
        model wasn't validated for batched timesteps and the FILM path
        is rarely used anyway.
        """
        import torch
        if len(dts) == 1:
            return [self.interp_tensor(a_t, b_t, dts[0])]
        if self.name != "rife":
            return [self.interp_tensor(a_t, b_t, dt) for dt in dts]
        n = len(dts)
        a_batch = a_t.repeat(n, 1, 1, 1)
        b_batch = b_t.repeat(n, 1, 1, 1)
        ts = a_t.new_tensor(list(dts)).view(n, 1, 1, 1)
        with torch.no_grad():
            out = self.model.inference(a_batch, b_batch, ts, scale=1.0)
        return [out[i:i + 1] for i in range(n)]


def load_film_model(model_path: str | Path = DEFAULT_FILM_MODEL_PATH,
                     fp16: bool = True) -> InterpBackend:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("interpolate step requires a CUDA-capable GPU")
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"FILM weights not found: {model_path}. Download from "
            f"https://github.com/dajes/frame-interpolation-pytorch/releases")
    torch.backends.cudnn.benchmark = True
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    dtype = torch.float16 if fp16 else torch.float32
    model = model.half() if fp16 else model.float()
    model = model.to("cuda")
    return InterpBackend("film", model, dtype=dtype)


def load_rife_model(rife_dir: str | Path = DEFAULT_RIFE_DIR,
                     fp16: bool = True) -> InterpBackend:
    """Load RIFE 4.25 (or whatever flownet.pkl lives in train_log/).

    The RIFE codebase uses relative imports (`from model.warplayer ...`,
    `from train_log.IFNet_HDv3 ...`), so we temporarily chdir + augment
    sys.path during the load. The actual inference doesn't depend on cwd
    once the model is loaded.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("interpolate step requires a CUDA-capable GPU")
    rife_dir = Path(rife_dir).resolve()
    weights = rife_dir / "train_log" / "flownet.pkl"
    if not weights.exists():
        raise FileNotFoundError(
            f"RIFE weights not found: {weights}. See "
            f"https://github.com/hzwer/Practical-RIFE for downloads.")
    torch.backends.cudnn.benchmark = True
    sys_path_added = False
    if str(rife_dir) not in sys.path:
        sys.path.insert(0, str(rife_dir))
        sys_path_added = True
    orig_cwd = os.getcwd()
    try:
        os.chdir(rife_dir)
        from train_log.RIFE_HDv3 import Model
        m = Model()
        m.load_model("train_log", -1)
    finally:
        os.chdir(orig_cwd)
        if sys_path_added:
            try:
                sys.path.remove(str(rife_dir))
            except ValueError:
                pass
    m.eval()
    dtype = torch.float16 if fp16 else torch.float32
    m.flownet = (m.flownet.half() if fp16 else m.flownet.float()).to("cuda")
    return InterpBackend("rife", m, dtype=dtype)


def load_backend(name: str, model_path: Optional[str | Path] = None,
                  fp16: bool = True) -> InterpBackend:
    name = name.lower()
    if name == "rife":
        return load_rife_model(model_path or DEFAULT_RIFE_DIR, fp16=fp16)
    if name == "film":
        return load_film_model(model_path or DEFAULT_FILM_MODEL_PATH,
                                 fp16=fp16)
    raise ValueError(f"unknown backend {name!r}; pick 'rife' or 'film'")


# --------------------------------------------------------------------------
# Single-pair interpolation (single-call + tile-stitch variants)
# --------------------------------------------------------------------------

def _interp_single(backend: InterpBackend, fr_a: np.ndarray,
                    fr_b: np.ndarray, dt: float) -> np.ndarray:
    """Single full-frame call -- safe for RIFE always, FILM below the cliff."""
    a_p, orig_hw = _pad_align(fr_a)
    b_p, _ = _pad_align(fr_b)
    a_t = _to_device_tensor(a_p, backend.dtype, "cuda")
    b_t = _to_device_tensor(b_p, backend.dtype, "cuda")
    pred = backend.interp_tensor(a_t, b_t, dt)
    return _to_bgr(pred, orig_hw)


def _interp_tile_stitch(backend: InterpBackend, fr_a: np.ndarray,
                          fr_b: np.ndarray, dt: float,
                          overlap_px: int = _TILE_OVERLAP_PX) -> np.ndarray:
    """Split into left/right halves with overlap, run each, blend the seam.

    Used for FILM at 1440p where single-call hits the cuDNN cliff. RIFE
    doesn't need this -- its native 1440p call is already fast.
    """
    h, w = fr_a.shape[:2]
    half_w = w // 2 + overlap_px
    half_w_aligned = ((half_w + _PAD_ALIGN - 1) // _PAD_ALIGN) * _PAD_ALIGN

    a_left_src = fr_a[:, :half_w_aligned]
    b_left_src = fr_b[:, :half_w_aligned]
    a_right_src = fr_a[:, w - half_w_aligned:]
    b_right_src = fr_b[:, w - half_w_aligned:]

    a_lp, left_hw = _pad_align(a_left_src)
    b_lp, _ = _pad_align(b_left_src)
    a_rp, right_hw = _pad_align(a_right_src)
    b_rp, _ = _pad_align(b_right_src)

    a_l = _to_device_tensor(a_lp, backend.dtype, "cuda")
    b_l = _to_device_tensor(b_lp, backend.dtype, "cuda")
    a_r = _to_device_tensor(a_rp, backend.dtype, "cuda")
    b_r = _to_device_tensor(b_rp, backend.dtype, "cuda")

    pred_left = backend.interp_tensor(a_l, b_l, dt)
    pred_right = backend.interp_tensor(a_r, b_r, dt)

    left_bgr = _to_bgr(pred_left, left_hw)
    right_bgr = _to_bgr(pred_right, right_hw)

    out = np.empty_like(fr_a)
    seam_lo = w // 2 - overlap_px
    seam_hi = w // 2 + overlap_px
    right_start_src = w - half_w_aligned
    out[:, :seam_lo] = left_bgr[:, :seam_lo]
    out[:, seam_hi:] = right_bgr[:, seam_hi - right_start_src:]
    overlap_w = 2 * overlap_px
    weights = np.linspace(1.0, 0.0, overlap_w,
                           dtype=np.float32).reshape(1, -1, 1)
    left_overlap = left_bgr[:, seam_lo:seam_hi].astype(np.float32)
    right_overlap = right_bgr[:,
                                seam_lo - right_start_src:
                                seam_hi - right_start_src].astype(np.float32)
    blended = weights * left_overlap + (1 - weights) * right_overlap
    out[:, seam_lo:seam_hi] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _auto_tile(backend: InterpBackend, src_w: int,
                tile: Optional[bool]) -> bool:
    """Decide tile-stitch on/off. Honour the user's explicit choice; else
    auto-pick: tile when FILM + width above the cliff threshold."""
    if tile is not None:
        return tile
    if backend.name == "film" and src_w >= _FILM_TILE_WIDTH_THRESHOLD:
        return True
    return False


def interp_pair(backend: InterpBackend, fr_a: np.ndarray, fr_b: np.ndarray,
                 dt: float, *, tile: Optional[bool] = None) -> np.ndarray:
    """One in-between frame between fr_a and fr_b at fraction dt in [0, 1]."""
    if _auto_tile(backend, fr_a.shape[1], tile):
        return _interp_tile_stitch(backend, fr_a, fr_b, dt)
    return _interp_single(backend, fr_a, fr_b, dt)


# --------------------------------------------------------------------------
# Frame source readers -- NVDEC fast path + cv2 fallback share a common
# contract so the inner loop is identical. Each returns
# (preprocessed_gpu_tensor, bgr_numpy_for_writer, orig_hw) or None on EOF.
# --------------------------------------------------------------------------

def _read_frame_nvdec(dec_iter, dtype):
    try:
        _, rgb_u8 = next(dec_iter)
    except StopIteration:
        return None
    t, hw = _preprocess_gpu_rgb(rgb_u8, dtype)
    bgr = _gpu_rgb_to_bgr_numpy(rgb_u8)
    return t, bgr, hw


def _read_frame_cv2(cap, dtype):
    ok, bgr = cap.read()
    if not ok:
        return None
    a_p, hw = _pad_align(bgr)
    t = _to_device_tensor(a_p, dtype, "cuda")
    return t, bgr, hw


# --------------------------------------------------------------------------
# Inner-loop variants (#43.1+2 = sequential; #43.3 = pipelined)
# --------------------------------------------------------------------------

def _emit_progress(prog, i, n_pairs, multiplier, n_out, f0,
                    overall_t0, pair_times, log_every):
    elapsed = time.time() - overall_t0
    frames_emitted = 1 + i * multiplier
    fps_obs = frames_emitted / elapsed if elapsed > 0 else 0.0
    eta = (n_out - frames_emitted) / fps_obs if fps_obs > 0 else None
    prog.update(progress=i / n_pairs,
                 current_frame=f0 + i,
                 fps_observed=float(fps_obs),
                 eta_s=eta)
    if i % log_every == 0:
        recent = pair_times[-log_every:]
        print(f"  pair {i}/{n_pairs}  recent median = "
              f"{np.median(recent)*1000:.0f} ms  "
              f"emit fps = {fps_obs:.1f}", flush=True)


def _run_sequential_loop(bk, read_next, writer, a_t, a_bgr, orig_hw,
                          n_src, dts, multiplier, n_out, f0,
                          use_tile, prog, log_every, batch_dts=True):
    """Synchronous loop: read -> model -> encode, one pair at a time."""
    import torch
    pair_times: list[float] = []
    overall_t0 = time.time()
    n_pairs = n_src - 1
    for i in range(1, n_src):
        nxt = read_next()
        if nxt is None:
            break
        b_t, b_bgr, _ = nxt
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pair_start = time.perf_counter()
        if use_tile:
            for dt_v in dts:
                mid = _interp_tile_stitch(bk, a_bgr, b_bgr, dt_v)
                writer.send(mid)
        else:
            preds = (bk.interp_tensor_multi(a_t, b_t, dts) if batch_dts
                     else [bk.interp_tensor(a_t, b_t, dt) for dt in dts])
            for pred in preds:
                writer.send(_model_out_to_bgr_numpy(pred, orig_hw))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pair_times.append(time.perf_counter() - pair_start)
        writer.send(b_bgr)
        a_t = b_t
        a_bgr = b_bgr
        _emit_progress(prog, i, n_pairs, multiplier, n_out, f0,
                        overall_t0, pair_times, log_every)
    return pair_times, time.time() - overall_t0


def _run_pipelined_loop(bk, input_path, writer, use_nvdec_pref, dtype,
                         f0, f1, src_h, src_w, n_src, dts, multiplier,
                         n_out, prog, log_every, encoder, loop_mode,
                         batch_dts=True):
    """Three-stage pipeline. The decoder thread OPENS the source in-thread
    so its CUDA context is local: NVDEC binds its context to whichever
    thread called CreateSimpleDecoder, and cross-thread iteration trips
    CUDA_ERROR_INVALID_CONTEXT. The main thread (this caller) is the model
    thread; the encoder thread drains numpy frames to the ffmpeg pipe.

    Bounded queues cap VRAM (decode tensors) and RAM (encoded numpy
    frames) so the pipeline doesn't run away on long clips. None is the
    EOF sentinel on both queues.

    Returns (pair_times, loop_s, frame_source). `frame_source` may flip
    from "nvdec" to "cv2" if NVDEC init fails inside the decoder thread.
    """
    import queue
    import threading
    import torch

    orig_hw = (src_h, src_w)
    decode_q: "queue.Queue" = queue.Queue(maxsize=4)
    encode_q: "queue.Queue" = queue.Queue(maxsize=4)
    err: dict = {}
    state: dict = {"frame_source": "nvdec" if use_nvdec_pref else "cv2"}

    def decoder_worker():
        close = None
        try:
            # NVDEC binds its CUDA context to whichever thread creates the
            # decoder. PyTorch's primary context is per-thread lazily
            # initialized -- set_device alone doesn't init it. Force init by
            # running a real CUDA op so NVDEC + later torch.as_tensor
            # wrapping share the same context on this thread.
            torch.cuda.set_device(0)
            _ = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            use_nvdec = use_nvdec_pref
            if use_nvdec:
                try:
                    dec = nvdecode.NvVideoDecoder(input_path)
                    dec_iter = dec.frames(f_start=f0, f_end=f1)
                    def reader():
                        return _read_frame_nvdec(dec_iter, dtype)
                    close = dec.close
                except Exception as e:  # noqa: BLE001
                    print(f"  NVDEC init failed in decoder thread ({e}); "
                          f"falling back to cv2", flush=True)
                    use_nvdec = False
                    state["frame_source"] = "cv2"
            if not use_nvdec:
                cap = cv2.VideoCapture(input_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
                def reader():
                    return _read_frame_cv2(cap, dtype)
                close = cap.release
            while True:
                nxt = reader()
                decode_q.put(nxt)
                if nxt is None:
                    return
        except Exception as e:  # noqa: BLE001
            err["decode"] = e
            decode_q.put(None)
        finally:
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass

    def encoder_worker():
        try:
            while True:
                item = encode_q.get()
                if item is None:
                    return
                for fr in item:
                    writer.send(fr)
        except Exception as e:  # noqa: BLE001
            err["encode"] = e

    dec_thread = threading.Thread(target=decoder_worker,
                                   name="interp-decoder", daemon=True)
    dec_thread.start()

    first = decode_q.get()
    if first is None:
        dec_thread.join()
        if "decode" in err:
            raise err["decode"]
        raise RuntimeError("failed to read first frame")
    a_t, a_bgr, _ = first

    enc_thread = threading.Thread(target=encoder_worker,
                                   name="interp-encoder", daemon=True)
    enc_thread.start()
    encode_q.put([a_bgr])  # first source frame, pass-through

    # Warmup happens here (not in interpolate_video) because we need the
    # first GPU tensor that the decoder thread produced. Warm at the same
    # call shape production will use, so cuDNN picks the right algorithm
    # the first time (avoids the shimmer from #42).
    prog.set_step("warmup", detail="priming cuDNN algorithm cache")
    for _ in range(3):
        if batch_dts:
            _ = bk.interp_tensor_multi(a_t, a_t, dts)
        else:
            for dt_v in dts:
                _ = bk.interp_tensor(a_t, a_t, dt_v)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prog.set_step("interp_frames", progress=0.0,
                   detail=f"encoder={encoder} loop={loop_mode}",
                   f_start=f0, f_end=f1, current_frame=f0,
                   fps_observed=0.0, eta_s=None)

    pair_times: list[float] = []
    overall_t0 = time.time()
    n_pairs = n_src - 1
    for i in range(1, n_src):
        nxt = decode_q.get()
        if nxt is None:
            break
        b_t, b_bgr, _ = nxt
        pair_start = time.perf_counter()
        preds = (bk.interp_tensor_multi(a_t, b_t, dts) if batch_dts
                 else [bk.interp_tensor(a_t, b_t, dt) for dt in dts])
        out_frames = [_model_out_to_bgr_numpy(p, orig_hw) for p in preds]
        out_frames.append(b_bgr)
        encode_q.put(out_frames)
        pair_times.append(time.perf_counter() - pair_start)
        a_t = b_t
        a_bgr = b_bgr
        _emit_progress(prog, i, n_pairs, multiplier, n_out, f0,
                        overall_t0, pair_times, log_every)

    encode_q.put(None)
    dec_thread.join()
    enc_thread.join()
    if "decode" in err:
        raise err["decode"]
    if "encode" in err:
        raise err["encode"]
    return pair_times, time.time() - overall_t0, state["frame_source"]


# --------------------------------------------------------------------------
# Top-level video driver
# --------------------------------------------------------------------------

def interpolate_video(input_path: str | Path, output_path: str | Path,
                  target_fps: float,
                  backend: str = "rife",
                  model_path: Optional[str | Path] = None,
                  fp16: bool = True, tile: Optional[bool] = None,
                  t0: Optional[float] = None, t1: Optional[float] = None,
                  log_every: int = 50,
                  force_cv2: bool = False,
                  use_pipeline: bool = True,
                  batch_dts: bool = True,
                  cq: int = 23) -> dict:
    """Frame-interpolate a video to target_fps using FILM or RIFE.

    target_fps must be an integer >=2x multiple of the source fps.
    """
    import torch
    import imageio_ffmpeg

    input_path = str(input_path)
    output_path = str(output_path)

    prog = Progress("interpolate", source=input_path, out_path=output_path)
    prog.set_step("load_model", detail=f"backend={backend}")
    t_loadstart = time.time()
    bk = load_backend(backend, model_path=model_path, fp16=fp16)
    prog.update(detail=f"loaded in {time.time()-t_loadstart:.1f} s")

    prog.set_step("probe_input")
    cap = cv2.VideoCapture(input_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    multiplier = target_fps / src_fps
    multiplier_round = int(round(multiplier))
    if abs(multiplier - multiplier_round) > 1e-4 or multiplier_round < 2:
        cap.release()
        prog.fail(f"target fps {target_fps} is not an integer >=2x multiple "
                   f"of source fps {src_fps}")
        raise ValueError(f"target_fps {target_fps} must be an integer "
                          f">=2x multiple of source fps {src_fps}")
    multiplier = multiplier_round
    dts = [k / multiplier for k in range(1, multiplier)]

    f0 = int(round((t0 or 0.0) * src_fps))
    f1 = int(round(t1 * src_fps)) if t1 is not None else src_n_total
    n_src = max(0, f1 - f0)
    if n_src < 2:
        cap.release()
        prog.fail(f"need at least 2 source frames in [t0={t0}, t1={t1}]")
        raise ValueError("need at least 2 source frames")

    n_out = (n_src - 1) * multiplier + 1
    use_tile = _auto_tile(bk, w, tile)
    # Release the probe cap now -- the actual frame source is reopened below
    # via NVDEC (fast path) or its own cv2 cap (fallback / tile-stitch path).
    cap.release()

    # Output writer -- NVENC first, libx264 fallback (matches render.py).
    def _open_writer(codec, params):
        wr = imageio_ffmpeg.write_frames(
            output_path, (w, h), pix_fmt_in="bgr24", fps=target_fps,
            codec=codec, quality=None, macro_block_size=1,
            output_params=params)
        wr.send(None)
        return wr
    # NVENC's default rate-control is a low fixed bitrate that doesn't
    # scale with fps -- at 60 fps each frame gets ~40% of the bit budget
    # a 20 fps render gets, producing visible blur and a 2-second VBR
    # ramp-up transient ("shimmer" in #42, originally misdiagnosed as a
    # RIFE flow issue). Constant-quality mode (`-rc vbr -cq 23`) gives
    # each frame the bits it needs based on content, matches libx264
    # CRF 18 in perceived quality, and removes the ramp-up.
    try:
        writer = _open_writer("h264_nvenc",
                               ["-preset", "p4", "-rc", "vbr",
                                "-cq", str(cq), "-pix_fmt", "yuv420p"])
        encoder = "h264_nvenc"
    except Exception:
        writer = _open_writer("libx264",
                               ["-crf", "18", "-preset", "veryfast",
                                "-pix_fmt", "yuv420p"])
        encoder = "libx264-veryfast"

    # Source + loop selection.
    # NVDEC fast path: GPU-resident decode + preprocess (quick wins #1+#2).
    # Pipelined loop: decoder/model/encoder thread overlap (quick win #3).
    # Pipelined helper opens the source IN the decoder thread so its CUDA
    # context is local (NVDEC binds context per-thread).
    use_nvdec_pref = (not use_tile) and (not force_cv2) and nvdecode.is_available()
    run_pipelined = use_pipeline and (not use_tile)
    loop_mode = "pipelined" if run_pipelined else "sequential"
    frame_source = "nvdec" if use_nvdec_pref else "cv2"

    prog.update(detail=f"src {w}x{h} {src_fps:.1f} fps n={n_src} -> "
                        f"out {target_fps:.1f} fps n={n_out} "
                        f"(backend={bk.name}, mult={multiplier}, "
                        f"tile={use_tile}, source={frame_source}, "
                        f"loop={loop_mode})")

    if run_pipelined:
        pair_times, loop_s, frame_source = _run_pipelined_loop(
            bk, input_path, writer, use_nvdec_pref, bk.dtype, f0, f1,
            h, w, n_src, dts, multiplier, n_out, prog, log_every,
            encoder, loop_mode, batch_dts=batch_dts)
    else:
        # Sequential: open source on main thread (current thread holds the
        # CUDA context, so NVDEC is happy), warmup, then run the sync loop.
        use_nvdec = use_nvdec_pref
        _src_close = None
        read_next = None
        if use_nvdec:
            try:
                _dec = nvdecode.NvVideoDecoder(input_path)
                _dec_iter = _dec.frames(f_start=f0, f_end=f1)
                def read_next(_it=_dec_iter):
                    return _read_frame_nvdec(_it, bk.dtype)
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
                return _read_frame_cv2(_c, bk.dtype)
            _src_close = _cap2.release

        first = read_next()
        if first is None:
            _src_close()
            writer.close()
            prog.fail("failed to read first frame")
            raise RuntimeError("failed to read first frame")
        a_t, a_bgr, orig_hw = first
        writer.send(a_bgr)

        # cuDNN.benchmark warmup -- the first few RIFE/FILM calls at a new
        # input shape go through cuDNN's algorithm-selection phase, where
        # different conv algos can produce slightly different fp16 outputs.
        # Without this priming the first ~half-second of interpolated
        # frames had a visible "shimmer" (algorithm jitter). Running each
        # dt three times against fr_a duplicated as both inputs locks the
        # algo cache in for the rest of the run.
        prog.set_step("warmup", detail="priming cuDNN algorithm cache")
        for _ in range(3):
            if use_tile:
                for dt_v in dts:
                    _ = interp_pair(bk, a_bgr, a_bgr, dt_v, tile=True)
            elif batch_dts:
                _ = bk.interp_tensor_multi(a_t, a_t, dts)
            else:
                for dt_v in dts:
                    _ = bk.interp_tensor(a_t, a_t, dt_v)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prog.set_step("interp_frames", progress=0.0,
                       detail=f"encoder={encoder} loop={loop_mode}",
                       f_start=f0, f_end=f1, current_frame=f0,
                       fps_observed=0.0, eta_s=None)

        pair_times, loop_s = _run_sequential_loop(
            bk, read_next, writer, a_t, a_bgr, orig_hw,
            n_src, dts, multiplier, n_out, f0,
            use_tile, prog, log_every, batch_dts=batch_dts)
        _src_close()
    overall_t0 = time.time() - loop_s  # for the report's total_s below

    writer.close()

    pt = np.array(pair_times) * 1000.0
    report = {
        "input": input_path,
        "output": output_path,
        "encoder": encoder,
        "backend": bk.name,
        "src_fps": float(src_fps),
        "src_size_wh": [w, h],
        "target_fps": float(target_fps),
        "multiplier": int(multiplier),
        "n_pairs": int(len(pair_times)),
        "n_frames_out": int(1 + len(pair_times) * multiplier),
        "tile": bool(use_tile),
        "frame_source": frame_source,
        "loop_mode": loop_mode,
        "batch_dts": bool(batch_dts),
        "per_pair_ms_median": float(np.median(pt)) if pt.size else 0.0,
        "per_pair_ms_p90": float(np.percentile(pt, 90)) if pt.size else 0.0,
        "total_s": float(time.time() - overall_t0),
    }
    prog.done(**report)
    return report


# Kept for backward compat with any callers that imported the old name.
DEFAULT_MODEL_PATH = DEFAULT_FILM_MODEL_PATH
