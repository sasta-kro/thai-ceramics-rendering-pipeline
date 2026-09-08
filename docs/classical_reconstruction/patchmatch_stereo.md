# COLMAP PatchMatch Stereo

This stage generates dense photometric and geometric depth and normal maps from
the 273 undistorted masked RGB images and the successful sequential COLMAP
camera model.

PatchMatch does not use the binary fusion masks directly. It processes the
black-background images in the dense workspace. The aligned binary masks will
be supplied during the later stereo-fusion stage.

## Current input

Run all commands from the project root:

```text
C:\Users\USER\OneDrive\Documents\Programming\Python\CSX4213 (Computer Vision)\thai-ceramics-rendering-pipeline
```

Required workspace:

```text
data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/
```

Required inputs:

```text
images/                    273 undistorted masked RGB images
sparse/                    Undistorted COLMAP camera model
stereo/patch-match.cfg     PatchMatch source-view configuration
```

The dense runner validates these inputs before starting COLMAP.

## Configured PatchMatch settings

The settings are stored in:

```text
configs/colmap_dense_pot1_unglazed_every6.yml
```

Current settings:

```yaml
patch_match:
  gpu_index: 0
  max_image_size: 1600
  cache_size_gb: 6
  num_threads: 2
  geom_consistency: true
  filter: true
```

These settings target the NVIDIA GeForce GTX 1650 with 4 GB of VRAM and a
computer with 16 GB of system RAM.

## Before starting

1. Activate the `pot-masking` environment.
2. Change to the project root.
3. Connect the computer to power.
4. Disable automatic sleep for the duration of the run.
5. Close the COLMAP GUI and other GPU-heavy programs.
6. Confirm that no earlier partial PatchMatch output is present.

The following folders should be empty before a fresh run:

```text
data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/stereo/depth_maps/
data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/stereo/normal_maps/
data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/stereo/consistency_graphs/
```

## Preview the command

Run a dry run first:

```powershell
python scripts/reconstruction/run_colmap_dense.py --stage patch-match --dry-run
```

The dry run validates the YAML configuration, checks that the undistorted
workspace contains all 273 images, and prints the COLMAP command without
starting PatchMatch.

## Recommended execution

Run PatchMatch through the project runner:

```powershell
python scripts/reconstruction/run_colmap_dense.py --stage patch-match
```

The runner:

- validates the dense workspace;
- refuses to mix a new run with incomplete existing maps;
- invokes `colmap.exe` without the problematic batch wrapper;
- streams COLMAP progress to the terminal;
- records the full output and elapsed time;
- validates the final depth-map and normal-map counts.

The log is written to:

```text
data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/logs/03_patch_match.log
```

## Equivalent direct COLMAP command

Use this only if the runner cannot be used:

```powershell
& "C:\Tools\COLMAP-4.1.1\bin\colmap.exe" patch_match_stereo --workspace_path "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential" --workspace_format COLMAP --PatchMatchStereo.gpu_index 0 --PatchMatchStereo.max_image_size 1600 --PatchMatchStereo.cache_size 6 --PatchMatchStereo.num_threads 2 --PatchMatchStereo.geom_consistency 1 --PatchMatchStereo.filter 1
```

## Expected runtime

For 273 images on the GTX 1650, PatchMatch may take several hours. A rough
working estimate is 4–10 hours, but actual runtime depends on GPU temperature,
power settings, source-view selection, and other system activity.

The desktop may become sluggish because PatchMatch uses the same GPU that
drives the display. Continuous image progress in the terminal or log indicates
that the process is still working.

## Expected outputs

Depth maps:

```text
data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/stereo/depth_maps/
```

Normal maps:

```text
data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/stereo/normal_maps/
```

With geometric consistency enabled, the target output is:

```text
273 photometric depth maps
273 geometric depth maps
273 photometric normal maps
273 geometric normal maps
```

The runner prints these four counts after COLMAP exits successfully.

## Manual output verification

If the direct COLMAP command was used, verify the result with:

```powershell
$stereo = "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\stereo"; Write-Host "Photometric depth maps:" (Get-ChildItem "$stereo\depth_maps\*.photometric.bin" -File).Count; Write-Host "Geometric depth maps:" (Get-ChildItem "$stereo\depth_maps\*.geometric.bin" -File).Count; Write-Host "Photometric normal maps:" (Get-ChildItem "$stereo\normal_maps\*.photometric.bin" -File).Count; Write-Host "Geometric normal maps:" (Get-ChildItem "$stereo\normal_maps\*.geometric.bin" -File).Count
```

Do not continue to stereo fusion unless PatchMatch exits successfully and all
required geometric map counts are 273.

## CUDA out-of-memory recovery

If COLMAP reports a CUDA out-of-memory error, do not mix the partial maps with
a retry. Preserve the failed output by renaming its generated folders:

```powershell
$stereo = "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\stereo"; $suffix = Get-Date -Format "yyyyMMdd-HHmmss"; if (Test-Path "$stereo\depth_maps") { Rename-Item -LiteralPath "$stereo\depth_maps" -NewName "depth_maps_failed_1600_$suffix" }; if (Test-Path "$stereo\normal_maps") { Rename-Item -LiteralPath "$stereo\normal_maps" -NewName "normal_maps_failed_1600_$suffix" }; if (Test-Path "$stereo\consistency_graphs") { Rename-Item -LiteralPath "$stereo\consistency_graphs" -NewName "consistency_graphs_failed_1600_$suffix" }; New-Item -ItemType Directory -Force "$stereo\depth_maps", "$stereo\normal_maps", "$stereo\consistency_graphs" | Out-Null
```

Then change the YAML setting:

```yaml
patch_match:
  max_image_size: 1200
```

Preview and rerun:

```powershell
python scripts/reconstruction/run_colmap_dense.py --stage patch-match --dry-run
python scripts/reconstruction/run_colmap_dense.py --stage patch-match
```

Reducing the maximum dimension from 1600 to 1200 lowers GPU-memory use and
runtime at the cost of some fine surface detail.

## Completion checkpoint

Stage 9 is complete when:

- COLMAP exits with code 0;
- all expected photometric and geometric depth maps exist;
- all expected photometric and geometric normal maps exist;
- no CUDA or workspace errors appear in the log;
- the runner prints `PatchMatch stereo complete.`

The next classical stage is masked stereo fusion, which will combine the
geometric depth maps into `results/fused.ply` while applying the 273 aligned
binary masks.
