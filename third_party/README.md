# Vendored third-party model code + weights

This directory holds vendored copies of the upstream ML projects Waruka
uses for its **optional** post-processing stages:

| Folder        | Used by                         | Upstream | License |
|---------------|---------------------------------|----------|---------|
| `rife/`       | `waruka interpolate` (default)  | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) | MIT |
| `film/`       | `waruka interpolate --backend film` | [frame-interpolation-pytorch](https://github.com/dajes/frame-interpolation-pytorch) (PyTorch port of Google's [FILM](https://github.com/google-research/frame-interpolation)) | Apache-2.0 |
| `realesrgan/` | `waruka render --sr`, `waruka upscale` | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | BSD-3-Clause |

The **source code** for each is committed to this repo. The **model
weights are not** — GitHub rejects single files over 100 MB, and the
weights would bloat every clone. The base `track → classify → campath →
render` pipeline does **not** need them; you only need them for the
interpolation / super-resolution stages.

## Getting the weights

Three weight files are required for the optional stages:

```
third_party/film/film_net_fp16.pt              (~69 MB, FILM-Style interpolation)
third_party/rife/train_log/flownet.pkl         (~25 MB, RIFE 4.25 interpolation)
third_party/realesrgan/weights/RealESRGAN_x2plus.pth  (~67 MB, super-resolution)
```

**Option A — weights bundle (easiest).** Download
`waruka-weights-1.0.0.zip` from the
[v1.0.0 GitHub release](../../releases/tag/v1.0.0) and extract it at the
repository root. The archive already contains the correct paths, so the
files land in the right place:

```bash
# from the repo root
unzip waruka-weights-1.0.0.zip
```

**Option B — straight from upstream.** Fetch each file from its upstream
project (links in the table above) and drop it at the path listed above.
See [NOTICE.md](../NOTICE.md) for full attribution.

**Already have the Windows bundle?** The prebuilt `waruka-1.0.0.zip`
ships with all weights inside `_internal/third_party/`; you don't need
to fetch anything separately to run the `.exe`.
