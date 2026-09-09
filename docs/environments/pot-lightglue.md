# `pot-lightglue` Environment

`pot-lightglue` is a small, dedicated runtime environment for COLMAP's
SIFT-LightGlue ONNX matcher. Its reproducible definition is:

```text
environment-lightglue.yml
```

## Included stack

- Python 3.10
- PyTorch 2.7.1 with CUDA 12.8
- the CUDA and cuDNN DLLs bundled with that PyTorch build
- NumPy 1.26 or newer (below 3.0)
- PyYAML 6.x

The Python process does not perform LightGlue inference itself. The project
runner validates the environment and exposes its PyTorch CUDA DLL directory
only to the external COLMAP child process.

## Create

From the repository root:

```powershell
micromamba create -f environment-lightglue.yml -y
```

The PyTorch CUDA package is large and requires internet access during creation.

## Activate

```powershell
micromamba activate pot-lightglue
```

## Verify

```powershell
python -c "import numpy, torch, yaml; print('NumPy:', numpy.__version__); print('Torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('cuDNN:', torch.backends.cudnn.version()); print('PyYAML:', yaml.__version__); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Required verification result:

- CUDA runtime 12.8 or newer;
- cuDNN version code 90500 or newer;
- GPU availability `True`;
- NVIDIA GeForce GTX 1650 detected.

Check package consistency:

```powershell
python -m pip check
```

## Update from the YAML definition

```powershell
micromamba env update -n pot-lightglue -f environment-lightglue.yml
```

## Validate the matching stage

Close the COLMAP GUI, then run:

```powershell
python scripts/reconstruction/run_lightglue_matching.py --dry-run
```

The dry run must validate 2,688 targeted pairs and print the resolved CUDA DLL
directory without creating a matching log or changing the database.

## Run matching

```powershell
python scripts/reconstruction/run_lightglue_matching.py
```

After matching, stop before camera registration so the side-to-top verified
pairs and inlier distribution can be measured.

The complete stage record is:

```text
docs/multiview_3dgs/lightglue_commands.md
```

Do not install gsplat here and do not use this environment for 3DGS training.
