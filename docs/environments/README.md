# Project Environments

The project uses separate Micromamba environments so incompatible Python,
PyTorch, CUDA, masking, reconstruction, and Gaussian Splatting dependencies do
not overwrite one another.

Run environment commands from the repository root:

```text
C:\Users\USER\OneDrive\Documents\Programming\Python\CSX4213 (Computer Vision)\thai-ceramics-rendering-pipeline
```

## Environment index

| Environment | Definition | Primary purpose |
| --- | --- | --- |
| [`pot-masking`](pot-masking.md) | `environment-masking.yml` | Image processing, masks, dataset preparation, and classical reconstruction utilities |
| [`pot-3dgs`](pot-3dgs.md) | `environment-3dgs.yml` | Gaussian Splatting preparation, training, evaluation, export, and viewing |
| [`pot-lightglue`](pot-lightglue.md) | `environment-lightglue.yml` | CUDA 12.8 runtime for COLMAP SIFT-LightGlue matching |

## General commands

List environments:

```powershell
micromamba env list
```

Show the active environment:

```powershell
micromamba info
```

Leave the active environment:

```powershell
micromamba deactivate
```

Inspect installed packages without activating an environment:

```powershell
micromamba list -n pot-masking
micromamba list -n pot-3dgs
micromamba list -n pot-lightglue
```

## Environment boundaries

- Do not upgrade PyTorch independently inside `pot-3dgs`. Its gsplat wheel is
  compiled specifically for PyTorch 2.4 and CUDA 12.4.
- Do not use `pot-lightglue` for 3DGS training. It exists only to supply the
  CUDA 12.8 and cuDNN runtime required by COLMAP's ONNX matcher.
- Use `pot-masking` for non-3DGS Python utilities unless a command guide names a
  different environment.
- Treat each root-level `environment-*.yml` file as the reproducible definition.
  Packages installed manually afterward are not recorded in that definition.

Each environment guide contains creation, activation, verification, update,
and stage-specific commands.
