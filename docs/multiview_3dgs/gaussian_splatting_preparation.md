# Repaired Multiview Gaussian Splatting Preparation

This guide converts the accepted 728-image LightGlue reconstruction into the
undistorted, mask-aligned dataset required by the project's 3D Gaussian
Splatting pipeline.

The global visual tilt of the COLMAP scene does not need correction before
training. A rigid world rotation changes neither camera-to-object geometry nor
the rendered result. Upright presentation can be applied after training.

Run every command from the repository root:

```text
C:\Users\USER\OneDrive\Documents\Programming\Python\CSX4213 (Computer Vision)\thai-ceramics-rendering-pipeline
```

Close the COLMAP GUI before running an undistortion stage.

## 1. Accepted sparse input

The source model is:

```text
data/processed/pot1-unglazed_multiview/
└── colmap_sparse_masked_multiview_lightglue_repaired/
    └── bundle_adjusted_final/
```

Accepted statistics:

```text
Registered images:       728 / 728
Cameras:                 3
Sparse points:           164,756
Observations:            1,717,485
Mean reprojection error: 0.898626 px
```

The three input cameras are `SIMPLE_RADIAL`. COLMAP undistortion will convert
them to three `PINHOLE` cameras while preserving every pose and 3D point.

## 2. Configurations

COLMAP input preparation and undistortion:

```text
configs/colmap_undistort_pot1_unglazed_multiview_lightglue_repaired.yml
```

Gaussian Splatting dataset, cache, and training profiles:

```text
configs/gaussian_splatting_pot1_unglazed_multiview_lightglue_repaired.yml
```

Generated undistorted dataset:

```text
data/processed/pot1-unglazed_multiview/
└── colmap_undistorted_masked_multiview_lightglue_repaired/
    ├── images/
    ├── masks/
    ├── sparse/
    ├── stereo/
    ├── inputs/
    ├── mask_undist/
    └── logs/
```

Original frames, masks, sparse models, and the accepted repaired model remain
unchanged.

## 3. Activate the preparation environment

Use `pot-masking` for image preparation and COLMAP undistortion:

```powershell
micromamba activate pot-masking
```

## 4. Validate the complete source dataset

```powershell
python scripts\reconstruction\run_multiview_3dgs_preparation.py --stage validate
```

The verified preflight result is:

```text
RGB images:        728
Masks:             728
Registered images: 728
Resolution:        2160x3840
```

This stage is read-only.

## 5. Prepare masked RGB and mask-image inputs

Preview the output paths without writing:

```powershell
python scripts\reconstruction\run_multiview_3dgs_preparation.py --stage prepare --dry-run
```

Create the prepared full-resolution inputs:

```powershell
python scripts\reconstruction\run_multiview_3dgs_preparation.py --stage prepare
```

Expected result:

```text
Prepared pairs: 728
```

This stage writes black-background RGB images and binary mask images under the
new undistortion workspace. It does not change the original frames or masks.

## 6. Undistort the masked RGB images

Preview the exact COLMAP command:

```powershell
python scripts\reconstruction\run_multiview_3dgs_preparation.py --stage undistort-rgb --dry-run
```

Run RGB undistortion:

```powershell
python scripts\reconstruction\run_multiview_3dgs_preparation.py --stage undistort-rgb
```

COLMAP writes the 728 undistorted RGB images, the three-camera `PINHOLE` sparse
model, and its workspace metadata. The configured maximum image size is 2000
pixels.

## 7. Undistort the binary mask images

The masks must pass through the same camera model and image-size settings as
the RGB images.

Preview:

```powershell
python scripts\reconstruction\run_multiview_3dgs_preparation.py --stage undistort-masks --dry-run
```

Run:

```powershell
python scripts\reconstruction\run_multiview_3dgs_preparation.py --stage undistort-masks
```

This writes a separate intermediate COLMAP workspace so it cannot overwrite
the RGB sparse model.

The short `inputs/mask_src` and `mask_undist` directory names are intentional.
Longer names exceeded the native Windows 260-character path limit in this
OneDrive project location. The runner checks all planned COLMAP input and
output image paths before launching the process.

## 8. Create and validate the aligned 3DGS masks

Preview:

```powershell
python scripts\reconstruction\run_multiview_3dgs_preparation.py --stage finalize-masks --dry-run
```

Create the final binary masks and verify every image-mask pair:

```powershell
python scripts\reconstruction\run_multiview_3dgs_preparation.py --stage finalize-masks
```

Required result:

```text
Aligned masks: expected 728; validated 728
```

No extra erosion is applied to the 3DGS masks.

## 9. Expected multiple-camera resolutions

The repaired camera calibrations correctly produce slightly different
undistorted widths:

```text
Side and top: 1125x2000
Underside:    1116x2000
```

The GS data loader supports these per-view dimensions and uses each image's own
`PINHOLE` intrinsics. At factor 4, the expected cache sizes are approximately:

```text
Side and top: 281x500
Underside:    279x500
```

Do not force the three cameras back into one calibration.

## 10. Activate the Gaussian Splatting environment

After all undistortion and mask stages pass:

```powershell
micromamba activate pot-3dgs
```

Check environment consistency:

```powershell
python -m pip check
```

## 11. Validate the undistorted 3DGS dataset

```powershell
python scripts\gaussian_splatting\validate_dataset.py --config configs\gaussian_splatting_pot1_unglazed_multiview_lightglue_repaired.yml
```

Required conditions:

- 728 undistorted RGB images;
- 728 pixel-aligned masks;
- 728 registered sparse images;
- three `PINHOLE` cameras accepted by the loader;
- no missing or unregistered filenames.

## 12. Prepare the factor-4 smoke-test cache

Preview without writing:

```powershell
python scripts\gaussian_splatting\prepare_dataset.py --config configs\gaussian_splatting_pot1_unglazed_multiview_lightglue_repaired.yml --profile smoke_multiview --dry-run
```

Create the cache:

```powershell
python scripts\gaussian_splatting\prepare_dataset.py --config configs\gaussian_splatting_pot1_unglazed_multiview_lightglue_repaired.yml --profile smoke_multiview
```

The deterministic every-eighth-image split should contain approximately 637
training images and 91 held-out images.

## 13. Validate the smoke training plan

Do not start training until dataset and cache validation pass.

```powershell
python scripts\gaussian_splatting\run_training.py --config configs\gaussian_splatting_pot1_unglazed_multiview_lightglue_repaired.yml --profile smoke_multiview --run-name smoke_multiview_repaired --dry-run
```

The smoke profile uses:

```text
Image factor:     4
Steps:            300
Initial points:   164,756
Gaussian cap:     200,000
SH degree:        1
Packed rendering: enabled
Sparse gradients: enabled
Viewer:           disabled during training
```

Stop after the dry run and inspect its plan before replacing `--dry-run` with
`--run`.

## 14. Run and evaluate the smoke test

Run the 300-step smoke test:

```powershell
python scripts\gaussian_splatting\run_training.py --config configs\gaussian_splatting_pot1_unglazed_multiview_lightglue_repaired.yml --profile smoke_multiview --run-name smoke_multiview_repaired --run
```

Evaluate all 91 deterministic held-out images:

```powershell
python scripts\gaussian_splatting\evaluate_checkpoint.py --config configs\gaussian_splatting_pot1_unglazed_multiview_lightglue_repaired.yml --run-name smoke_multiview_repaired --run
```

The evaluation writes per-image comparisons and aggregate metrics under:

```text
data/processed/pot1-unglazed_multiview/gaussian_splatting_masked_multiview_lightglue_repaired/runs/smoke_multiview_repaired/evaluation/holdout_black
```

Confirm that side, top-45-degree, and underside held-out views all render the
same coherent pot before continuing.

## 15. Prepare the factor-2 cache

The baseline and quality profiles share the same factor-2 cache. Create it
once:

```powershell
python scripts\gaussian_splatting\prepare_dataset.py --config configs\gaussian_splatting_pot1_unglazed_multiview_lightglue_repaired.yml --profile baseline_multiview_7k
```

Expected target resolutions are `562x1000` for side/top images and `558x1000`
for underside images, with 728 image-mask pairs.

## 16. Run and evaluate the 7,000-step baseline

Run the baseline training:

```powershell
python scripts\gaussian_splatting\run_training.py --config configs\gaussian_splatting_pot1_unglazed_multiview_lightglue_repaired.yml --profile baseline_multiview_7k --run-name baseline_multiview_repaired --run
```

Evaluate its final checkpoint:

```powershell
python scripts\gaussian_splatting\evaluate_checkpoint.py --config configs\gaussian_splatting_pot1_unglazed_multiview_lightglue_repaired.yml --run-name baseline_multiview_repaired --run
```

The accepted baseline completed 7,000 steps with 500,000 Gaussians. Its
91-view held-out result was 26.74 dB mean PSNR, 0.926 mean SSIM, and 0.914 mean
alpha IoU.

## 17. Run and evaluate the 15,000-step quality model

The quality run starts from the COLMAP initialization; it does not resume the
7,000-step checkpoint. It reuses the factor-2 cache created in section 15.

Run quality training:

```powershell
python scripts\gaussian_splatting\run_training.py --config configs\gaussian_splatting_pot1_unglazed_multiview_lightglue_repaired.yml --profile quality_multiview_15k --run-name quality_multiview_repaired --run
```

After training finishes, evaluate all held-out views before exporting:

```powershell
python scripts\gaussian_splatting\evaluate_checkpoint.py --config configs\gaussian_splatting_pot1_unglazed_multiview_lightglue_repaired.yml --run-name quality_multiview_repaired --run
```

Compare the quality metrics and representative side, top-45-degree, and
underside renders with the accepted baseline. Export only after the quality
checkpoint passes this inspection.

Completed result:

```text
Status:               complete
Training steps:       15,000
Elapsed:              44.6 minutes
Final loss:           0.0125201
Gaussians:            750,000
Peak training VRAM:   0.87 GB
Held-out images:      91
Mean PSNR:            26.9441 dB
Mean SSIM:            0.9319
Foreground PSNR:      26.1655 dB
Foreground L1:        0.02352
Alpha IoU:            0.92645
```

All side, top-45-degree, and underside groups remained connected. The quality
checkpoint was accepted as the final export source.

## 18. Open the local checkpoint viewer

The viewer reads the native checkpoint directly, so export is not required.
For the quality run:

```powershell
python scripts\gaussian_splatting\view_checkpoint.py --config configs\gaussian_splatting_pot1_unglazed_multiview_lightglue_repaired.yml --run-name quality_multiview_repaired --run
```

Open `http://localhost:8080`, orbit with the mouse, and press `Ctrl+C` in the
terminal to stop the server. The configured viewer limit is 500,000 Gaussians;
this affects interactive display only and does not modify the checkpoint.

To view the accepted baseline instead, replace the run name with
`baseline_multiview_repaired`.

## 19. Export PLY and SPLAT artifacts

Export the accepted quality checkpoint to both portable formats:

```powershell
python scripts\gaussian_splatting\export_checkpoint.py --config configs\gaussian_splatting_pot1_unglazed_multiview_lightglue_repaired.yml --run-name quality_multiview_repaired --formats ply splat --run
```

Outputs are written under:

```text
data/processed/pot1-unglazed_multiview/gaussian_splatting_masked_multiview_lightglue_repaired/runs/quality_multiview_repaired/exports
```

The `.ply` export retains spherical-harmonic coefficients and is the preferred
full-quality/archive artifact. The `.splat` export is compact and base-color
only for compatible viewers. These files represent Gaussian splats, not a
polygon mesh. The exporter also writes `export_manifest.json` with hashes,
sizes, checkpoint step, and normalization metadata, and refuses to overwrite
an existing export.

Completed export verification:

```text
PLY bytes:     114,000,952
PLY SHA-256:   6b6b991c724984dd6808ee5d41b88161d16dc2fa01725c147f7f7311193f0c7c
SPLAT bytes:   24,000,000
SPLAT SHA-256: 4c3f1749d671fd91551f30327136a76da3d6afa6a68bf7e2f438d380913f07e1
```

The PLY header and 750,000 fixed-size SPLAT records passed structural checks,
and both recalculated hashes matched `export_manifest.json`.

To export the accepted baseline instead, replace the run name with
`baseline_multiview_repaired`.

## 20. Failure safety

- Each stage resumes only outputs that pass validation.
- Incomplete outputs are not overwritten automatically.
- If a stage reports an incomplete nonempty workspace, preserve it and inspect
  the named files instead of deleting them blindly.
- Do not run RGB and mask undistortion simultaneously.
- Do not open the COLMAP GUI while an undistortion stage is running.
