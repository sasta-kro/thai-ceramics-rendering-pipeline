# Masked 3D Gaussian Splatting

This branch trains a low-memory 3D Gaussian Splatting model from the already
undistorted masked images and undistorted COLMAP model in
`colmap_dense_masked_sequential`. It does not rerun feature extraction, matching,
mapping, or dense stereo.

## Environment choice

The Windows environment is pinned to:

- Python 3.10
- PyTorch 2.4.1 with CUDA 12.4
- gsplat 1.5.3, official `cp310` Windows wheel for PyTorch 2.4/CUDA 12.4

This precompiled wheel supports the GTX 1650's compute capability 7.5 and avoids
requiring a local CUDA Toolkit or Visual Studio C++ compiler. Do not replace it
with an unpinned `pip install gsplat`, because that path may attempt a local CUDA
build on first use.

Official references:

- <https://github.com/nerfstudio-project/gsplat/tree/v1.5.3>
- <https://docs.gsplat.studio/whl/gsplat/>

## Package organization and command compatibility

The implementation is grouped by responsibility:

```text
scripts/gaussian_splatting/
├── core/              # Configuration, paths, shared validation
├── data/              # COLMAP scene loading and cache preparation
├── training/          # Gaussian initialization, rendering, optimization
├── postprocessing/    # Checkpoints, evaluation, export, viewer
├── diagnostics/       # Environment and dataset checks
└── cli/               # New python -m command entry points
```

The original top-level script names remain as compatibility wrappers. Therefore,
every command documented below continues to work unchanged. For example, these
two commands call the same trainer implementation:

```powershell
python scripts/gaussian_splatting/run_training.py --profile baseline_7k --dry-run
python -m scripts.gaussian_splatting.cli.train --profile baseline_7k --dry-run
```

New module entry points:

```powershell
python -m scripts.gaussian_splatting.cli.doctor
python -m scripts.gaussian_splatting.cli.validate
python -m scripts.gaussian_splatting.cli.prepare --all-profiles --dry-run
python -m scripts.gaussian_splatting.cli.train --profile baseline_7k --dry-run
python -m scripts.gaussian_splatting.cli.evaluate --run-name baseline_7k --help
python -m scripts.gaussian_splatting.cli.export --run-name baseline_7k --help
python -m scripts.gaussian_splatting.cli.view --run-name baseline_7k --dry-run
```

The wrappers contain no duplicated pipeline logic; they only bootstrap the
repository package and forward the existing command-line arguments.

## Create the environment

Run from the repository root in PowerShell:

```powershell
micromamba create -f environment-3dgs.yml -y
micromamba activate pot-3dgs
python scripts/gaussian_splatting/environment_doctor.py
```

The doctor imports the required packages, checks the pinned wheel matrix, and
confirms CUDA access. It does not launch a rasterization kernel or training.

If the environment already exists, update it instead:

```powershell
micromamba env update -n pot-3dgs -f environment-3dgs.yml
micromamba activate pot-3dgs
python scripts/gaussian_splatting/environment_doctor.py
```

Do not proceed to image preparation until the final line says:

```text
Environment check passed.
```

## Prepare downscaled masked images

The preparation stage creates PNG caches at factors 2 and 4. RGB is resized as
premultiplied color and then converted back to straight foreground color using
the aligned soft mask. This avoids introducing a dark resampling halo around the
pot. Masks are retained as soft alpha for later background compositing.

Preview the work without writing files:

```powershell
python scripts/gaussian_splatting/prepare_dataset.py --all-profiles --dry-run
```

Run the complete preparation yourself:

```powershell
python scripts/gaussian_splatting/prepare_dataset.py --all-profiles
```

The command is resumable by default. It writes factor-specific manifests, a
dataset manifest, and a deterministic every-eighth-image holdout split under
`gaussian_splatting_masked_sequential`. It does not start training.

## Initial smoke training

Validate the factor-4 cache, COLMAP poses and points, output path, and guarded
training settings without starting CUDA work:

```powershell
python scripts/gaussian_splatting/run_training.py --profile smoke_lowmem --dry-run
```

After the dry run passes, start the 300-step smoke test yourself:

```powershell
python scripts/gaussian_splatting/run_training.py --profile smoke_lowmem --run
```

The smoke profile uses 281×500 images, packed rasterization, sparse gradients,
28,491 COLMAP initialization points, a 100,000-Gaussian cap, and no training-time
viewer. It saves a checkpoint, CSV log, peak-VRAM summary, and three held-out
comparison renders. If the default run directory already exists, keep it and use
a new name such as `--run-name smoke_lowmem_retry1`.

## Full low-memory training

Validate the factor-2, 7,000-step profile without starting CUDA work:

```powershell
python scripts/gaussian_splatting/run_training.py --profile baseline_7k --dry-run
```

Start full training yourself:

```powershell
python scripts/gaussian_splatting/run_training.py --profile baseline_7k --run
```

This profile uses 562×1000 images, packed rasterization, sparse gradients,
spherical-harmonic degree 2, MCMC refinement, and a 500,000-Gaussian cap.

## Held-out evaluation

The evaluator renders all 35 deterministic test views over black, saves
reference/render comparisons, and reports full-frame PSNR/SSIM, foreground
PSNR/L1, and alpha IoU.

```powershell
python scripts/gaussian_splatting/evaluate_checkpoint.py --run-name baseline_7k --dry-run
python scripts/gaussian_splatting/evaluate_checkpoint.py --run-name baseline_7k --run
```

Results are written under
`runs/baseline_7k/evaluation/holdout_black`. Existing evaluation output is never
overwritten.

## Portable exports

The standard PLY retains all learned spherical-harmonic coefficients. The
compact `.splat` file stores base color and is useful for compatible WebGL
viewers.

```powershell
python scripts/gaussian_splatting/export_checkpoint.py --run-name baseline_7k --formats ply splat --dry-run
python scripts/gaussian_splatting/export_checkpoint.py --run-name baseline_7k --formats ply splat --run
```

Exports and their SHA-256 manifest are written under
`runs/baseline_7k/exports`. Existing exports are never overwritten.

## Local browser viewer

The local viewer does not upload the model. It uses the learned Gaussian
positions, covariance, opacity, and base spherical-harmonic color for interactive
inspection.

```powershell
python scripts/gaussian_splatting/view_checkpoint.py --run-name baseline_7k --dry-run
python scripts/gaussian_splatting/view_checkpoint.py --run-name baseline_7k --run
```

Open <http://localhost:8080>, orbit with the mouse, and press `Ctrl+C` in the
PowerShell terminal to stop the viewer. If 500,000 Gaussians are too slow in the
browser, retry with `--max-gaussians 250000`.
