# LightGlue Camera-Gap Repair Guide

This guide repairs the discontinuous side and top camera trajectories visible
in the first 728-image LightGlue model. It does not modify the original COLMAP
reconstruction or the original LightGlue workspace.

Run every command from the repository root:

```text
C:\Users\USER\OneDrive\Documents\Programming\Python\CSX4213 (Computer Vision)\thai-ceramics-rendering-pipeline
```

Activate the dedicated environment first:

```powershell
micromamba activate pot-lightglue
```

Close the COLMAP GUI before a command accesses a database or writes a model.

## 1. Why the repair is needed

The first LightGlue result registered all 728 images, but parts of its side and
top camera trajectories contained abnormally large jumps:

- the largest side step was approximately 26 times its sequence median;
- the largest top step was approximately 11.8 times its sequence median;
- adjacent images around those breaks had few or no shared triangulated points;
- the underside trajectory remained smooth.

The three videos were also forced to share one optimized camera calibration,
although their independently reconstructed focal lengths differed strongly.
The repair therefore uses a separate camera calibration for each capture ring.

## 2. Repair strategy

```text
Smooth original side poses
          +
Smooth original top poses
          +
Existing connected underside poses
          |
          v
Robust shared-landmark Sim(3) alignment
          |
          v
One point-free 728-camera model
          |
          v
Three camera calibrations in copied database
          |
          v
Fresh triangulation with poses fixed
          |
          v
Validation and conservative bundle adjustment
```

Only camera poses are transferred. Existing point clouds are not directly
merged. All 3D points are reconstructed again from the preserved feature
matches and verified geometries.

## 3. Isolated repair workspace

The repair uses:

```text
data/processed/pot1-unglazed_multiview/
└── colmap_sparse_masked_multiview_lightglue_repaired/
    ├── database.db
    ├── source_text/
    │   ├── side/
    │   ├── top/
    │   └── combined/
    ├── repaired_text/
    ├── pose_repaired_initial/
    ├── triangulated_fixed/
    ├── bundle_adjusted_final/
    ├── logs/
    └── reports/
```

The database is a copy. The original source models and the first LightGlue
model remain unchanged.

## 4. Install or update the environment

The reproducible environment file includes NumPy and all Python dependencies:

```powershell
micromamba env update -n pot-lightglue -f environment-lightglue.yml
```

Verify the installed packages:

```powershell
python -c "import numpy, torch, yaml; print('NumPy:', numpy.__version__); print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('cuDNN:', torch.backends.cudnn.version()); print('PyYAML:', yaml.__version__)"
```

## 5. Generate the repaired point-free pose model

An optional non-writing validation is:

```powershell
python scripts\reconstruction\repair_lightglue_camera_poses.py --dry-run
```

Create the repaired text model:

```powershell
python scripts\reconstruction\repair_lightglue_camera_poses.py
```

The completed run reported:

```text
Side-under alignment: 51/51 inliers, RMSE 0.002431
Top-side alignment: 1482/1853 inliers, RMSE 0.017108
side:      max/median adjacent spacing 1.153
top45:     max/median adjacent spacing 1.113
underside: max/median adjacent spacing 1.130
```

All trajectory ratios are below the acceptance limit of 2.0.

## 6. Convert the repaired model to COLMAP binary format

```powershell
& "C:\Tools\COLMAP-4.1.1\bin\colmap.exe" model_converter --input_path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue_repaired\repaired_text" --output_path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue_repaired\pose_repaired_initial" --output_type BIN --log_target stderr
```

`model_converter` normally returns to the prompt without printing anything.
The converted model must contain:

```text
Rigs:              3
Cameras:           3
Registered images: 728
Points:            0
```

Zero points are intentional at this stage.

## 7. Configure three cameras in the copied database

Optional non-writing validation:

```powershell
python scripts\reconstruction\configure_lightglue_repair_database.py --dry-run
```

Apply the camera assignments:

```powershell
python scripts\reconstruction\configure_lightglue_repair_database.py
```

Expected assignments:

```text
Camera 1: side       273 images
Camera 2: top45      233 images
Camera 3: underside  222 images
Database integrity: ok
```

The operation preserves all 728 feature records, all 16,499 match records, and
all 16,499 two-view-geometry records in the copied database.

## 8. Retriangulate from clean tracks

Run:

```powershell
& "C:\Tools\COLMAP-4.1.1\bin\colmap.exe" point_triangulator --database_path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue_repaired\database.db" --image_path "data\frames_output\pot1-unglazed_multiview_frames" --input_path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue_repaired\pose_repaired_initial" --output_path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue_repaired\triangulated_fixed" --clear_points 1 --refine_intrinsics 0 --Mapper.fix_existing_frames 1 --Mapper.num_threads 8 --Mapper.ba_use_gpu 0 --Mapper.tri_ignore_two_view_tracks 1 --log_target stderr
```

This command keeps all repaired poses and camera intrinsics fixed while
building a fresh point cloud. Stop after it finishes and validate the result
before bundle adjustment.

## 9. Analyze the triangulated model

Run this read-only summary:

```powershell
& "C:\Tools\COLMAP-4.1.1\bin\colmap.exe" model_analyzer --path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue_repaired\triangulated_fixed"
```

Required conditions before bundle adjustment:

- 728 registered images remain;
- all three cameras remain;
- a substantial number of 3D points and observations exist;
- the mean reprojection error is finite;
- cross-view tracks connect side, top, and underside images;
- the three camera trajectories remain smooth.

Do not run bundle adjustment if triangulation reports an error or if the model
loses registered images.

## 10. Conservative bundle adjustment

Run this only after the triangulated model passes validation:

```powershell
& "C:\Tools\COLMAP-4.1.1\bin\colmap.exe" bundle_adjuster --input_path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue_repaired\triangulated_fixed" --output_path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue_repaired\bundle_adjusted_final" --BundleAdjustment.refine_focal_length 0 --BundleAdjustment.refine_principal_point 0 --BundleAdjustment.refine_extra_params 0 --BundleAdjustment.refine_rig_from_world 1 --BundleAdjustment.refine_sensor_from_rig 0 --BundleAdjustment.refine_points3D 1 --BundleAdjustmentCeres.max_num_iterations 100 --BundleAdjustmentCeres.use_gpu 0 --log_target stderr
```

The first repair pass deliberately keeps all three intrinsics fixed. This
prevents bundle adjustment from collapsing the distinct side, top, and
underside calibrations back toward one compromise calibration.

## 11. Inspect the final model in the COLMAP GUI

Use the COLMAP batch launcher so its bundled Qt platform plugin is configured:

```powershell
& "C:\Tools\COLMAP-4.1.1\COLMAP.bat" gui --database_path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue_repaired\database.db" --image_path "data\frames_output\pot1-unglazed_multiview_frames" --import_path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue_repaired\bundle_adjusted_final" --log_target stderr
```

Check that:

- the side, top, and underside cameras form three smooth elevation rings;
- the large breaks previously visible in the side and top rings are gone;
- every camera points toward the same pot;
- the pot is not duplicated, folded, or split into remote components;
- the rim, interior, exterior, foot, and underside occupy one coordinate frame.

Do not use the repaired model for 3DGS until both numerical and GUI validation
pass.

The repaired model passed both checks. Continue with the undistortion and
masked 3DGS preparation workflow in:

```text
docs/multiview_3dgs/gaussian_splatting_preparation.md
```

## 12. Rerun safety

The output directories are intentionally required to be empty. If a command
reports that an output directory is not empty, do not delete or overwrite it
blindly. Preserve the existing result and inspect it before deciding whether a
new repair workspace is required.
