# `pot-masking` Environment

`pot-masking` is the general image-processing and classical-reconstruction
environment. Its reproducible definition is:

```text
environment-masking.yml
```

## Included stack

- Python 3.12
- NumPy 1.26 or newer, below 3
- Pillow 10 or newer
- OpenCV 4.8 or newer
- PyYAML 6.x

The YAML does not define PyTorch, CUDA, gsplat, or the COLMAP executable.
COLMAP commands call the separately installed executable under `C:\Tools`.

## Create

From the repository root:

```powershell
micromamba create -f environment-masking.yml -y
```

## Activate

```powershell
micromamba activate pot-masking
```

## Verify

```powershell
python -c "import cv2, numpy, PIL, yaml; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('Pillow:', PIL.__version__); print('PyYAML:', yaml.__version__)"
```

Check package consistency:

```powershell
python -m pip check
```

## Update from the YAML definition

```powershell
micromamba env update -n pot-masking -f environment-masking.yml
```

An update installs or changes declared dependencies but may not remove packages
that were installed manually. Use `micromamba list -n pot-masking` when checking
whether an existing environment has drifted from the YAML definition.

## Used for

- frame and image utilities;
- mask processing and validation;
- multiview dataset staging;
- sparse and dense COLMAP runner scripts;
- mesh cleanup, simplification, and texturing helpers;
- YAML-driven classical-reconstruction stages.

Relevant project guides include:

- `docs/masking.md`
- `docs/colmap.md`
- `docs/capture.md`
- `docs/classical_reconstruction/`
- `docs/multiview_3dgs/commands.md`

Do not use this environment for the current SIFT-LightGlue GPU stage; its YAML
does not guarantee the CUDA 12.8 runtime required by that COLMAP build.
