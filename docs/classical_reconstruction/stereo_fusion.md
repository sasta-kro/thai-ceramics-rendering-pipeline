# Masked stereo fusion

Stereo fusion combines the completed geometric PatchMatch depth maps into one dense colored point cloud. The pottery masks are applied again during fusion so background depth samples are excluded.

## Prepared configuration

The project configuration is `configs/colmap_dense_pot1_unglazed_every6.yml`:

```yaml
fusion:
  input_type: geometric
  max_image_size: 1200
  cache_size_gb: 6
  num_threads: 4
  min_num_pixels: 3
```

The 1200-pixel limit matches the successful PatchMatch run. `min_num_pixels: 3` is a balanced starting value: a point must be supported by at least three images.

## Prerequisites

Before fusion, the workspace should contain all of the following:

- 273 geometric depth maps in `data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/stereo/depth_maps`
- 273 geometric normal maps in `data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/stereo/normal_maps`
- 273 aligned binary masks in `data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/masks`

The runner validates these inputs before starting COLMAP.

## Run from PowerShell

Open PowerShell in the repository root and activate the project environment:

```powershell
conda activate pot-masking
```

First perform a validation-only dry run. This prints the exact COLMAP command but does not create or modify fusion output:

```powershell
python scripts/reconstruction/run_colmap_dense.py --stage fusion --dry-run
```

If validation reports 273 fusion-ready inputs, start fusion:

```powershell
python scripts/reconstruction/run_colmap_dense.py --stage fusion
```

Do not close the terminal while the command is running. Runtime depends mainly on CPU speed, storage, and the number of valid depth samples; for this 273-image workspace, allow roughly tens of minutes and be prepared for it to take longer on a laptop.

## Equivalent direct COLMAP command

The Python runner is recommended because it validates the maps and masks, records a log, checks that the resulting PLY contains points, and safely resumes an existing valid result. The equivalent one-line PowerShell command is:

```powershell
& "C:\Tools\COLMAP-4.1.1\bin\colmap.exe" stereo_fusion --workspace_path "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential" --workspace_format COLMAP --input_type geometric --output_path "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\results\fused.ply" --StereoFusion.mask_path "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\masks" --StereoFusion.max_image_size 1200 --StereoFusion.cache_size 6 --StereoFusion.num_threads 4 --StereoFusion.min_num_pixels 3
```

The direct COLMAP command does not need the Conda environment, but its output is not automatically logged or validated by the project runner.

## Expected output

Successful fusion creates:

```text
data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/
├── results/
│   └── fused.ply
└── logs/
    └── 04_stereo_fusion.log
```

The runner should finish with exit code 0 and print a positive `Fused points` count. `fused.ply` is the dense point cloud used by the later meshing step; it is not yet a surface mesh.

Open the PLY in CloudCompare or MeshLab to inspect point density, background leakage, holes, and outliers. COLMAP's model viewer is intended primarily for sparse camera models and is not the best viewer for this dense PLY.

## If the result needs tuning

Keep the first `fused.ply` as a baseline before trying a different setting. Lowering `min_num_pixels` from 3 to 2 can recover more points but usually increases noise. Raising it to 4 or 5 produces fewer, more strongly supported points and may increase holes. Do not proceed to meshing until the pot is recognizable and background leakage is acceptably low.
