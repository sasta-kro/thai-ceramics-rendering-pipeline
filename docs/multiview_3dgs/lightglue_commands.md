# Pot 1 Multiview SIFT-LightGlue Commands

This stage attempts to connect the complete top-45-degree image sequence to the
existing side-and-underside COLMAP Model 2. It does not merge Models 1 and 2.
Top cameras will be registered into Model 2 only after the new cross-view
matches pass quality checks.

Run all commands from the repository root:

```text
C:\Users\USER\OneDrive\Documents\Programming\Python\CSX4213 (Computer Vision)\thai-ceramics-rendering-pipeline
```

Create the dedicated LightGlue environment once:

```powershell
micromamba create -f environment-lightglue.yml
```

Activate it before running this stage:

```powershell
micromamba activate pot-lightglue
```

Close the COLMAP GUI before any command that opens `database.db`.

## Files used by this stage

YAML configuration:

```text
configs/colmap_pot1_unglazed_multiview_lightglue.yml
```

Isolated workspace:

```text
data/processed/pot1-unglazed_multiview/colmap_sparse_masked_multiview_lightglue
```

The workspace contains:

```text
colmap_sparse_masked_multiview_lightglue/
├── database.db
├── bridge_pairs_lightglue.txt
├── base_model/
├── registered_pass1/
├── triangulated_pass1/
├── registered_final/
└── logs/
```

The original database and original sparse models remain unchanged in
`colmap_sparse_masked_multiview`.

## 1. Recreate the isolated workspace if starting again

Validate without writing:

```powershell
python scripts/reconstruction/prepare_lightglue_workspace.py --dry-run
```

Create the isolated database copy, base-model copy, output directories, and
2,688-pair bridge list:

```powershell
python scripts/reconstruction/prepare_lightglue_workspace.py
```

The preparation command refuses to overwrite an existing workspace.

## 2. Clean stale side-to-top records from the copied database

Preview the exact database changes:

```powershell
python scripts/reconstruction/cleanup_lightglue_database.py --dry-run
```

Apply the transactional cleanup:

```powershell
python scripts/reconstruction/cleanup_lightglue_database.py
```

The completed cleanup removed:

- 2,688 `matches` records containing 30 raw correspondence rows;
- 2,688 `two_view_geometries` records containing zero verified rows.

It preserved all 728 images, all 1,939,238 SIFT keypoints and descriptors, and
all non-target match groups. Its audit report is:

```text
data/processed/pot1-unglazed_multiview/colmap_sparse_masked_multiview_lightglue/logs/side_top_cleanup_report.json
```

## 3. Validate the YAML-driven LightGlue command

This checks the database, manifest, pair list, expected pair count, existing
SIFT features, cleanup state, COLMAP executable, and output log path. It prints
the resolved COLMAP command without starting matching:

```powershell
python scripts/reconstruction/run_lightglue_matching.py --dry-run
```

Expected summary:

```text
SIFT-LightGlue targeted matching
Validated targeted pairs: 2688
Dry run only: COLMAP was not started.
```

## 4. Run targeted SIFT-LightGlue matching

Run the complete configured matching stage with one command:

```powershell
python scripts/reconstruction/run_lightglue_matching.py
```

The runner reads this configuration automatically:

```text
configs/colmap_pot1_unglazed_multiview_lightglue.yml
```

The current GTX 1650 configuration uses:

- `SIFT_LIGHTGLUE` with the existing SIFT keypoints and descriptors;
- GPU index 0;
- one matching thread;
- at most 4,096 matches per pair;
- guided matching disabled (COLMAP 4.1.1 does not support it with LightGlue);
- LightGlue minimum score 0.1;
- at least 15 geometric inliers for a verified pair;
- exactly 2,688 targeted side-to-top pairs.

The installed COLMAP ONNX provider requires CUDA 12.8 or newer and cuDNN 9.5 or
newer. The runner validates these versions before launch and prepends the
configured DLL directory only to the external COLMAP process. The dedicated
`pot-lightglue` environment supplies the compatible runtime without changing
the pinned `pot-3dgs` training environment.

The database currently contains at most 3,894 descriptors for any single
image, so the 4,096 match limit does not truncate the stored SIFT feature set.

On the first run, COLMAP may download `sift-lightglue.onnx`. Internet access is
therefore required until the model has been cached. Keep the computer awake
during matching.

Console output is also written to:

```text
data/processed/pot1-unglazed_multiview/colmap_sparse_masked_multiview_lightglue/logs/lightglue_matching_retry1.log
```

The original `lightglue_matching.log` records the first failed launch, in which
COLMAP's ONNX Runtime could not discover `cublasLt64_12.dll`. That attempt
stopped before processing any pair and wrote no side-to-top database records.

The runner refuses to overwrite an existing matching log. If a run fails or
must be repeated, preserve the workspace and log for diagnosis rather than
deleting or manually modifying them.

## 5. Stop for bridge-quality validation

Do not run the mapper, image registration, triangulation, or bundle adjustment
immediately after matching. First inspect the copied database for:

- at least 10 to 20 verified side-to-top pairs;
- approximately 30 or more inliers in useful pairs;
- verified pairs distributed across several side and top frames;
- coverage around the capture rotation rather than one small angular region.

If matching completes, report the runner's exit code or preserve the generated
log and proceed to the bridge-quality inspection stage.

## 6. Completed bridge result

The targeted run completed in 17.77 minutes and produced:

- 240 verified side-to-top pairs;
- 154 pairs with at least 30 geometric inliers;
- median 41, mean 60.45, and maximum 232 inliers per verified pair;
- coverage of 49 selected side frames and 46 selected top frames.

This exceeded the bridge-quality threshold and authorized registration into
Model 2.

## 7. Register and triangulate

The first registration pass kept the 495 existing Model 2 frames fixed:

```powershell
C:\Tools\COLMAP-4.1.1\bin\colmap.exe image_registrator --database_path data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue\database.db --input_path data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue\base_model --output_path data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue\registered_pass1 --Mapper.fix_existing_frames 1 --Mapper.abs_pose_min_num_inliers 30 --Mapper.abs_pose_max_error 12 --Mapper.max_reg_trials 3 --Mapper.num_threads 8 --Mapper.ba_use_gpu 0 --log_target stderr
```

It registered 205 top images and produced a 700-image model. The following
pass preserved the existing sparse points and camera poses while triangulating
new observations:

```powershell
C:\Tools\COLMAP-4.1.1\bin\colmap.exe point_triangulator --database_path data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue\database.db --image_path data\frames_output\pot1-unglazed_multiview_frames --input_path data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue\registered_pass1 --output_path data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue\triangulated_pass1 --clear_points 0 --refine_intrinsics 0 --Mapper.fix_existing_frames 1 --Mapper.num_threads 8 --Mapper.ba_use_gpu 0 --Mapper.tri_ignore_two_view_tracks 1 --log_target stderr
```

Triangulation increased the model from 96,911 to 174,557 points. A second
registration pass then recovered every remaining top image:

```powershell
C:\Tools\COLMAP-4.1.1\bin\colmap.exe image_registrator --database_path data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue\database.db --input_path data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue\triangulated_pass1 --output_path data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue\registered_final --Mapper.fix_existing_frames 1 --Mapper.abs_pose_min_num_inliers 30 --Mapper.abs_pose_max_error 12 --Mapper.max_reg_trials 3 --Mapper.num_threads 8 --Mapper.ba_use_gpu 0 --log_target stderr
```

## 8. Registered result

`registered_final` contains all 728 cameras:

- side: 273 / 273;
- top-45-degree: 233 / 233;
- underside: 222 / 222.

It contains 174,557 points and 1,539,247 observations with a mean
reprojection error of 1.189781 pixels. Final triangulation, bundle adjustment,
and visual camera-ring validation remain pending.

## 9. Inspect `registered_final` in the COLMAP GUI

From the repository root, open the copied database, combined images, and final
registered model with one PowerShell command:

```powershell
& "C:\Tools\COLMAP-4.1.1\COLMAP.bat" gui --database_path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue\database.db" --image_path "data\frames_output\pot1-unglazed_multiview_frames" --import_path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview_lightglue\registered_final" --log_target stderr
```

Use `COLMAP.bat` for the GUI rather than launching `bin\colmap.exe` directly.
The launcher configures COLMAP's bundled Qt platform-plugin directory; without
it, Windows reports that the Qt `windows` platform plugin cannot be found.

Confirm that the viewer shows one coherent pot point cloud, a side camera ring,
a higher top-45-degree ring, and an underside ring. There should be no second
pot, remote camera cluster, or group of cameras pointing away from the object.

Close the COLMAP GUI before running final triangulation, bundle adjustment, or
any other command that accesses the copied database.
