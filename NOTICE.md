# Waruka -- Third-Party Notices

Waruka is Copyright (C) 2026 Stefan Lewis and is licensed under the
**GNU Affero General Public License v3.0 or later** (AGPL-3.0-or-later).
The full license text is in [LICENSE](LICENSE).

This file lists the third-party software and model weights that Waruka
bundles or depends on at runtime, along with their licenses and source
URLs. None of the components listed here are owned by Waruka's
copyright holder; each retains its own copyright and license terms.

## Python libraries (runtime dependencies)

| Component | License | Source |
|---|---|---|
| **ultralytics** | **AGPL-3.0** (or commercial) | <https://github.com/ultralytics/ultralytics> |
| PyTorch (`torch`, `torchvision`) | BSD-3-Clause | <https://pytorch.org/> |
| NumPy | BSD-3-Clause | <https://numpy.org/> |
| SciPy | BSD-3-Clause | <https://scipy.org/> |
| OpenCV (`opencv-python`) | Apache-2.0 | <https://opencv.org/> |
| Pillow | HPND (MIT-like) | <https://python-pillow.org/> |
| filterpy | MIT | <https://github.com/rlabbe/filterpy> |
| imageio-ffmpeg (wrapper) | BSD-2-Clause | <https://github.com/imageio/imageio-ffmpeg> |
| tqdm | MIT and MPL-2.0 (dual) | <https://github.com/tqdm/tqdm> |
| psutil | BSD-3-Clause | <https://github.com/giampaolo/psutil> |
| reportlab | BSD-3-Clause | <https://www.reportlab.com/> |
| matplotlib | PSF / BSD-style | <https://matplotlib.org/> |
| pandas | BSD-3-Clause | <https://pandas.pydata.org/> |

Waruka's use of **ultralytics** is the binding constraint that requires
Waruka itself to be licensed under AGPL-3.0. The bundled `yolo11n.pt`
weights file is part of the ultralytics release and is covered by the
same AGPL-3.0.

## GUI framework

| Component | License | Source |
|---|---|---|
| PySide6 (Qt for Python) | LGPL-3.0 (or commercial) | <https://www.qt.io/qt-for-python> |

PySide6 is dynamically linked via separate DLLs in the bundle
(`_internal/PySide6/Qt6Core.dll`, `Qt6Gui.dll`, `Qt6Widgets.dll`).
Users may replace these DLLs with their own builds of Qt 6 to satisfy
LGPL-3.0 Section 4.

## NVIDIA proprietary components

Waruka bundles components from NVIDIA's CUDA Toolkit and Video Codec
SDK. These are redistributed under the terms of the **NVIDIA CUDA
Toolkit End User License Agreement** (CUDA EULA) and the **NVIDIA
Software License Agreement** for the Video Codec SDK. See:

- <https://docs.nvidia.com/cuda/eula/index.html>
- <https://developer.nvidia.com/nvidia-video-codec-sdk>

| Component | Notes |
|---|---|
| `nvidia-cuda-runtime-cu12` (`cudart64_12.dll`, etc.) | CUDA 12 runtime libraries, redistribution permitted under CUDA EULA section 1.3 |
| Bundled CUDA libs in torch wheel (`cublas`, `cudnn`, `cufft`, `cusolver`, `cusparse`, `cusolverMg`) | Redistributed with PyTorch per the same EULA |
| PyNvVideoCodec | NVIDIA Video Codec SDK Python bindings, redistribution permitted under the SDK EULA |

NVIDIA, CUDA, and NVENC/NVDEC are trademarks of NVIDIA Corporation.

## Vendored ML model code + weights

The `third_party/` directory contains source code and pre-trained model
weights from several research projects, vendored to make Waruka
self-contained.

| Component | License | Source |
|---|---|---|
| **Real-ESRGAN** (`third_party/realesrgan/`, `RealESRGAN_x2plus.pth`) | BSD-3-Clause | <https://github.com/xinntao/Real-ESRGAN> |
| **FILM-Style** (`third_party/film/`, `film_net_fp16.pt`) | Apache-2.0 | <https://github.com/dajes/frame-interpolation-pytorch> -- a PyTorch port of Google Research's [FILM](https://github.com/google-research/frame-interpolation) |
| **RIFE 4.25** (`third_party/rife/`, `flownet.pkl`) | MIT | <https://github.com/hzwer/Practical-RIFE> |
| **YOLO11n** (`yolo11n.pt`) | AGPL-3.0 (ultralytics release) | <https://github.com/ultralytics/ultralytics> |

Each subdirectory under `third_party/` retains its upstream `LICENSE`
file untouched -- check there for the authoritative text.

## Bundled ffmpeg binary

The `imageio-ffmpeg` package bundles a prebuilt ffmpeg binary, which
Waruka uses for video encoding (NVENC h.264/h.265 with libx264 software
fallback).

| Component | License | Source |
|---|---|---|
| ffmpeg (LGPL build, shipped via imageio-ffmpeg) | LGPL-2.1-or-later | <https://ffmpeg.org/> |

The imageio-ffmpeg project specifically distributes an LGPL-only build
of ffmpeg (no GPL-tainted libraries such as x264 or x265 statically
linked). Source code for the bundled ffmpeg binary is available from
ffmpeg.org.

## Bootloader

| Component | License | Source |
|---|---|---|
| PyInstaller bootloader (`waruka.exe`, `waruka-cli.exe`) | GPL-2.0 with **bootloader exception** | <https://github.com/pyinstaller/pyinstaller> |

The bootloader exception explicitly permits PyInstaller to embed
applications of any license. See the
[PyInstaller bootloader license](https://github.com/pyinstaller/pyinstaller/blob/develop/COPYING.txt)
for the exact text.

## Trademark note

"Waruka" is used here as the name of this open-source project. The
maintainer is aware that an unrelated commercial company also operates
under the name Waruka in a different product category (social video
watching). Should this become a source of confusion the project may be
renamed; no claim is made to the "Waruka" trademark.

## How to comply with this notice when redistributing

If you redistribute Waruka (or a fork of it):

1. Include this `NOTICE.md` and the `LICENSE` (AGPL-3.0) file.
2. Provide the full source code of Waruka (and any modifications) at the
   point of distribution, as required by AGPL-3.0 section 6.
3. If you redistribute the NVIDIA components, include the NVIDIA EULA
   acknowledgement (this file's "NVIDIA proprietary components"
   section satisfies that).
4. If you replace the bundled PySide6 Qt 6 DLLs with modified versions,
   you must comply with LGPL-3.0 section 4 (release the modifications
   or document how to reproduce them).
5. If you replace the bundled ffmpeg binary with a different build,
   ensure the new build is also license-compatible (LGPL or
   permissive).
