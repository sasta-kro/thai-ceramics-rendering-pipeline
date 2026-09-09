# Pot 1 Multiview 3DGS Commands

This guide records the commands for reconstructing the main unglazed pot from
the existing side views, the new top-45-degree views, and the rotated underside
views. Run every command from the repository root in PowerShell.

The original view folders and the completed side-only reconstruction remain
unchanged. New results use the `pot1-unglazed_multiview` name.

## Pipeline

```text
Side RGB images and SAM2 masks
Top-45-degree RGB images and SAM2 masks
Rotated underside RGB images and SAM2 masks
                    ↓
Combined multiview COLMAP input
                    ↓
Masked sparse COLMAP reconstruction
                    ↓
Undistorted images, masks, cameras, and sparse points
                    ↓
3DGS preparation, training, evaluation, export, and viewer
```

## Inputs

```text
Side images:       data/frames_output/pot1-unglazed_every6_frames
Side masks:        data/processed/pot1-unglazed_every6/masks_colmap
Top-45 images:     data/frames_output/pot1-unglazed_top_frames
Top-45 masks:      data/processed/pot1-unglazed_top/masks_colmap
Underside images:  data/frames_output/pot1-unglazed_underside_rotate_frames
Underside masks:   data/processed/pot1-unglazed_underside/masks_colmap
```

Validated source totals:

- 273 side image-mask pairs
- 233 top-45-degree image-mask pairs
- 222 underside image-mask pairs
- 728 total image-mask pairs at 2160 by 3840 pixels

## 1. Activate the masking and COLMAP environment

```powershell
micromamba activate pot-masking
```

## 2. Validate the multiview sources without writing files

```powershell
python scripts/reconstruction/prepare_multiview_dataset.py --config configs/multiview_dataset_pot1_unglazed.yml --validate-only
```

Expected final output:

```text
Multiview dataset validation complete
Sources: 3
Images/masks: 728/728
Resolution: 2160x3840
No output files were written
```

Status: completed successfully.

## 3. Create the combined COLMAP staging dataset

```powershell
python scripts/reconstruction/prepare_multiview_dataset.py --config configs/multiview_dataset_pot1_unglazed.yml
```

The command copies the validated data without changing the original folders.
It prefixes colliding names with `side_`, `top45_`, or `underside_` and writes:

```text
data/frames_output/pot1-unglazed_multiview_frames
data/processed/pot1-unglazed_multiview/masks_colmap
data/processed/pot1-unglazed_multiview/dataset_manifest.csv
```

Do not add `--overwrite` for the first run. The tool refuses to replace an
existing generated dataset by default.

Expected final output:

```text
Multiview dataset validation complete
Sources: 3
Images/masks: 728/728
Resolution: 2160x3840
Combined COLMAP staging complete
```

Status: completed successfully. The 728 staged RGB files, 728 masks, and 728
manifest rows were verified with no missing or extra entries. Representative
SHA-256 checks passed for every view group.

## 4. Multisequence COLMAP matching strategy

The COLMAP runner supports `matching.method: multisequence`. This mode creates
one explicit pair list with:

- neighboring frames within each of the three ordered videos;
- loop-closure pairs between the beginning and end of each rotation;
- sampled all-angle bridge pairs from side to top-45-degree views;
- sampled all-angle bridge pairs from side to underside views;
- no unnecessary direct top-to-underside bridge set.

With overlap 15 and bridge stride 5, the validated 728-frame dataset produces:

```text
Within-sequence pairs: 10560
Loop-closure pairs: 675
Side-to-top bridge pairs: 2688
Side-to-underside bridge pairs: 2576
Total explicit pairs: 16499
```

COLMAP 4.1.1 processes the generated list with `matches_importer` and
`--match_type pairs`. This is substantially smaller than matching all 264,628
possible pairs while still connecting every capture ring through the side view.

Status: implemented and covered by tests. No COLMAP processing was started.

## 5. Validate the multiview COLMAP configuration

```powershell
python scripts/reconstruction/run_colmap.py --config configs/colmap_pot1_unglazed_multiview.yml --dry-run
```

The dry run validates the 728 RGB images, 728 masks, dataset manifest, explicit
pair plan, COLMAP executable, memory profile, and fresh output paths. It prints
the feature extraction, `matches_importer`, and mapper commands without running
COLMAP or creating output files.

Expected matching summary:

```text
Matching method: multisequence
Within-sequence pairs: 10560
Loop-closure pairs: 675
Bridge pairs side<->top45: 2688
Bridge pairs side<->underside: 2576
Total explicit pairs: 16499
```

Status: completed successfully. No COLMAP processing was started.

## 6. Run the multiview sparse COLMAP reconstruction

```powershell
python scripts/reconstruction/run_colmap.py --config configs/colmap_pot1_unglazed_multiview.yml
```

This command creates the explicit pair list, extracts masked SIFT features,
matches the 16,499 planned pairs, and runs sparse mapping. It writes only under:

```text
data/processed/pot1-unglazed_multiview/colmap_sparse_masked_multiview
```

The runner refuses to overwrite an existing database or non-empty sparse model
directory. This is a long-running command and must be run by the user.

Completed reconstruction result:

```text
Model 0: 10 top-45-degree images, 5,269 points, 0.820475 px error
Model 1: 233 top-45-degree images, 58,481 points, 0.797265 px error
Model 2: 273 side + 222 underside images, 96,911 points, 0.902436 px error
```

Model 2 is the strongest combined reconstruction. Model 1 is a strong but
separate top/interior reconstruction. Model 0 is a redundant top-view fragment.
No model currently contains all 728 images.

Open model 2 in the COLMAP GUI:

```powershell
$env:QT_PLUGIN_PATH="C:\Tools\COLMAP-4.1.1\plugins"; & "C:\Tools\COLMAP-4.1.1\bin\colmap.exe" gui --database_path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview\database.db" --image_path "data\frames_output\pot1-unglazed_multiview_frames" --import_path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview\sparse\2"
```

Open model 1 in the COLMAP GUI:

```powershell
$env:QT_PLUGIN_PATH="C:\Tools\COLMAP-4.1.1\plugins"; & "C:\Tools\COLMAP-4.1.1\bin\colmap.exe" gui --database_path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview\database.db" --image_path "data\frames_output\pot1-unglazed_multiview_frames" --import_path "data\processed\pot1-unglazed_multiview\colmap_sparse_masked_multiview\sparse\1"
```

Status: sparse reconstruction completed. Manual inspection and cross-model
alignment planning are required before undistortion or 3DGS preparation.

## Repaired LightGlue and 3DGS continuation

The original sparse models above were subsequently connected and repaired.
The current one-command workflow for undistortion, 3DGS dataset preparation,
smoke and full training, held-out evaluation, local checkpoint viewing, and
PLY/SPLAT export is documented in:

```text
docs/multiview_3dgs/gaussian_splatting_preparation.md
```

The LightGlue matching and camera-pose repair commands are documented in:

```text
docs/multiview_3dgs/lightglue_commands.md
docs/multiview_3dgs/lightglue_camera_gap_repair.md
```

Current final status:

```text
COLMAP:       728/728 images, 3 cameras, 164,756 points, 0.898626 px
3DGS:        15,000 steps, 750,000 Gaussians, final loss 0.0125201
Evaluation:  91 views, 26.9441 dB PSNR, 0.9319 SSIM, 0.92645 alpha IoU
Exports:     verified 114 MB PLY and 24 MB SPLAT
```

The final exported model is coherent across side, top-45-degree, and underside
views. A future 20–30-degree capture ring is recommended to strengthen the
remaining weak geometry around the upper rim.

Long-running image preparation, COLMAP reconstruction, 3DGS training,
evaluation, export, and viewer commands are run by the user. The project agent
only prepares and validates the corresponding scripts and configurations.
