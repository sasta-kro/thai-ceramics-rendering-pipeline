# `pot-3dgs` Environment

`pot-3dgs` is the pinned Gaussian Splatting training and evaluation environment.
Its reproducible definition is:

```text
environment-3dgs.yml
```

## Core binary stack

- Python 3.10
- PyTorch 2.4.1 with CUDA 12.4
- torchvision 0.19.1 with CUDA 12.4
- gsplat 1.5.3 compiled for PyTorch 2.4 and CUDA 12.4 on Windows
- pycolmap 3.x

It also contains the numerical, image, metric, logging, viewer, and build
packages used by the custom low-memory 3DGS pipeline.

## Create

From the repository root:

```powershell
micromamba create -f environment-3dgs.yml -y
```

## Activate

```powershell
micromamba activate pot-3dgs
```

## Verify

```powershell
python -c "import torch, torchvision, gsplat; print('Torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('gsplat:', gsplat.__version__); print('CUDA runtime:', torch.version.cuda); print('cuDNN:', torch.backends.cudnn.version()); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Check package consistency:

```powershell
python -m pip check
```

## Update from the YAML definition

```powershell
micromamba env update -n pot-3dgs -f environment-3dgs.yml
```

Do not independently upgrade `torch`, `torchvision`, or `gsplat`. The gsplat
wheel contains compiled CUDA extensions tied to the declared PyTorch and CUDA
versions. These three packages must remain a compatible set.

## Used for

- preparing COLMAP outputs for 3DGS;
- low-memory GTX 1650 smoke training;
- full Gaussian Splatting training;
- evaluation and dataset diagnostics;
- checkpoint export and viewer launch.

Relevant project guides include:

- `docs/gaussian_splatting.md`
- `docs/multiview_3dgs/commands.md`

## LightGlue boundary

The CUDA 12.4 runtime in this environment is older than the CUDA 12.8 runtime
required by the installed COLMAP 4.1.1 ONNX provider. Keep `pot-3dgs` unchanged
and use `pot-lightglue` for targeted SIFT-LightGlue matching.
