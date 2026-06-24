# Waruka hardware requirements

Waruka is a Windows-only tool that uses NVIDIA CUDA for every heavy stage of
the pipeline. It will run on a CPU-only machine but with the GPU-only
stages disabled (`interpolate`, `upscale`, and `render --sr` will fail
with a clear "requires a CUDA-capable GPU" error; `track`, `classify`,
`campath`, and `render` fall back to CPU paths but are dramatically
slower).

## Recommended

| Tier | GPU | VRAM | RAM | CPU | Disk |
|---|---|---|---|---|---|
| Comfortable | RTX 2070 Super / RTX 3060 12GB / better | 8 GB+ | 32 GB | 8 cores / 16 threads | 200 GB |
| Minimum (validated) | RTX 2080 Ti (dev box) | 11 GB | 47 GB | i9-9900K (8c/16t) | -- |
| Minimum (theoretical) | GTX 1060 6GB / RTX 2060 6GB | 6 GB | 16 GB | 4 cores / 8 threads | 50 GB |

The minimum-theoretical row hasn't been tested -- if you run Waruka on a
GTX 1060 / RTX 2060, please share what works and what doesn't.

## Supported GPU generations

The runtime requirements come from three constraints stacked on top of
each other: the PyTorch CUDA-11.8 wheel's SM coverage, hardware video
decode for HEVC (Reolink Duo 2's output codec), and NVENC for the output
encoder.

### Fully supported (every GPU-accelerated stage works)

| Generation | Compute cap | Representative cards | Notes |
|---|---|---|---|
| Pascal | 6.1 | GTX 1060 6GB, 1070, 1070 Ti, 1080, 1080 Ti, Titan Xp | 6 GB models are the practical entry point |
| Turing | 7.5 | GTX 1650, 1660 (Super/Ti), RTX 2060/2070/2080 (Super/Ti), RTX Titan | Recommended baseline |
| Ampere | 8.0 / 8.6 | RTX 3050/3060/3070/3080/3090 (Ti), A2000-A6000 | RTX 3060 12GB sweet spot for budget builds |
| Ada Lovelace | 8.9 | RTX 4060/4070/4080/4090 (Ti) | All work; 4060 8GB is the practical floor |
| Hopper | 9.0 | H100, H200 | Overkill but supported |

### Partial support

- **Maxwell 2nd gen (compute 5.2)** -- GTX 950/960/970/980 (Ti), Titan X
  (Maxwell): `track` and `render` work, but there's **no hardware HEVC
  decode** so the Reolink source must be transcoded to H.264 first.
  NVENC h264 works; NVENC hevc does not. `interpolate` and `upscale` are
  unvalidated (probably OOM on 4 GB cards).

- **GT 1030 / mobile MX series** -- no NVENC engine at all. `render`
  falls back to libx264 software encode (much slower). 2 GB VRAM is too
  small for SR / interpolate.

### Not supported

- **Kepler (compute 3.x)** -- GTX 6xx/7xx, GT 6xx/7xx, original Titan,
  K-series workstation. The PyTorch+cu118 wheels don't ship SM 3.x
  binaries.

- **Pre-Maxwell (Fermi, Tesla)** -- same reason.

- **Blackwell (RTX 50xx, compute 12.0)** -- released after CUDA 11.8.
  The cu118 wheel doesn't include SM 12.0 binaries. Waruka will need a
  `torch+cu124` (or later) upgrade in `requirements.txt` to support
  Blackwell. Tracked as a 1.1+ task.

### CPU-only fallback

| Stage | Works on CPU? | Speed cost |
|---|---|---|
| `track` | Yes (ultralytics CPU + cv2 remap) | ~10-20x slower |
| `classify`, `campath` | Yes (always CPU) | N/A |
| `render` (no SR) | Yes (`pano.render` CPU + libx264) | ~5-10x slower |
| `render --sr` | **No** | Real-ESRGAN is CUDA-only |
| `interpolate` | **No** | RIFE/FILM are CUDA-only |
| `upscale` | **No** | Real-ESRGAN is CUDA-only |

The CPU paths exist for portability and emergency use; they are not
serious working configurations for a full match.

## VRAM budget by stage

| Stage | Approx VRAM | Note |
|---|---|---|
| `track` (YOLO11n + GpuRenderer) | ~1 GB | Cheap |
| `render` (no SR) | ~1.5 GB | |
| `render --sr` (Real-ESRGAN crop) | ~4-5 GB peak | |
| `interpolate` (RIFE / FILM) | ~3-4 GB | Tile-stitch path at full pano width |
| `upscale` (Real-ESRGAN full-frame) | ~5-6 GB peak | OOM-prone on 6 GB cards |

Run multiple stages concurrently and the VRAM peaks add up -- the
`pipeline` subcommand pipelines `track` and `render` together (~2.5 GB
peak); the `interpolate` stage runs after, so the peak of the GPU-only
post-process flow is dominated by interpolate alone.

## Storage

- A 100-minute Reolink Duo 2 recording is ~3-5 GB of HEVC source.
- The broadcast output is ~5 GB at default H.264 NVENC quality (`--cq 23`).
- Chunked-pipeline intermediates (`_pipeline_chunks/`) can grow to 10-20
  GB during a long run.
- The Waruka bundle itself is ~5.5 GB unpacked, ~3.2 GB zipped.

A 200 GB working drive comfortably handles a full match end-to-end.

## OS and drivers

- Windows 10 (1909+) or Windows 11
- NVIDIA driver supporting NVENC 13.0 (Game Ready 545+ as of late 2024)
- CUDA 11.8 user-mode bits are bundled with the PyTorch wheel; you don't
  install them separately
- Visual C++ 2015-2022 redistributable (almost always present on a
  modern Windows install)
