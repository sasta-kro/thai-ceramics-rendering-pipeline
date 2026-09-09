# THAI CERAMICS MULTIVIEW 3DGS PROGRESS REPORT

Status updated: 9 September 2026

## 1. Scope of This Update

This report continues from `last_week_status.md` and records the work completed after the successful side-view 3D Gaussian Splatting baseline. The present update begins with the acquisition of new camera angles and continues through successful LightGlue alignment and registration of all 728 main-pot cameras.

The purpose of the new work was to improve the weakly observed areas of the first 3DGS model, especially:

- the pot opening and inner wall;
- the rim from elevated viewpoints;
- the foot and exact underside;
- the continuity of the reconstructed shape outside the original side-view orbit.

The previous side-only `baseline_7k` result remains preserved as the successful version-1 baseline. No existing reconstruction or training output was overwritten.

## 2. Reason for Additional Image Acquisition

The first 3DGS model produced strong side-view appearance and quantitative held-out results. However, free-viewpoint inspection showed that the upper interior became blurred near overhead views and the exact underside contained unstable colors and Gaussian artifacts. These defects were caused mainly by missing camera coverage rather than insufficient training steps.

Training for more iterations cannot reliably recover surfaces that were not visible in the registered images. For this reason, the next improvement stage focused on collecting new observations before changing the 3DGS training configuration.

The original side-view sequence was retained under its existing name:

```text
data/frames_output/pot1-unglazed_every6_frames
```

It was not renamed to `pot1_main_side`, which avoids breaking the completed version-1 pipeline and its documented commands.

## 3. New Multiview Capture

Two new videos of the main pot were recorded:

1. A top-45-degree rotation, showing the rim, opening, inner wall, upper ornament, and exterior body.
2. An underside rotation, showing the foot, lower body, and bottom surface.

The underside sequence was recorded while the pot was inverted. Its extracted images were subsequently rotated by 180 degrees with a separate Python utility to make their displayed orientation consistent and easier to annotate and inspect. The reversed orbital direction relative to the side and top sequences is not itself a COLMAP error because COLMAP estimates every camera pose from image correspondences. The important requirement is sufficient shared visible pottery surface between sequences.

Two additional physical components were also identified for later work:

- the separate pot lid;
- the broken lower support or bottom lift.

These components have not been included in the current main-pot COLMAP reconstruction. They must be processed as separate objects because they move independently from the pot. The broken bottom lift will also require mask-cleanup settings that preserve its genuine openings instead of automatically filling holes or retaining only one connected component.

## 4. Frame Organization

Each camera-position video remained in its own frame folder during extraction and SAM2 segmentation. The view groups were not mixed before masking because SAM2 video propagation assumes one continuous ordered sequence.

The three main-pot image sources are:

```text
Side images:
data/frames_output/pot1-unglazed_every6_frames

Top-45-degree images:
data/frames_output/pot1-unglazed_top_frames

Rotated underside images:
data/frames_output/pot1-unglazed_underside_rotate_frames
```

Source image counts and dimensions:

- Side: 273 images
- Top-45-degree: 233 images
- Underside: 222 images
- Total: 728 images
- Resolution: 2160 × 3840 pixels

Keeping these sequences separate at the masking stage preserved their original ordering, simplified annotation, and prevented SAM2 propagation from crossing a discontinuous change of viewpoint.

## 5. SAM2 Annotation and Segmentation

SAM2 was run independently for the top-45-degree and underside sequences. The original side masks were reused from the selected every-sixth-frame dataset.

For the top-45-degree sequence, positive points were placed on the pottery body and visible inner wall. Negative points were placed on the background and turntable. The annotation intentionally retained the opening and interior as part of the ceramic object.

For the underside sequence, positive points were placed on the lower body and bottom surface. Negative points were distributed around the white turntable, blue background, and visible cable. The bounding box surrounded the full visible pot without treating the turntable as foreground.

Final COLMAP mask sources:

```text
Side masks:
data/processed/pot1-unglazed_every6/masks_colmap

Top-45-degree masks:
data/processed/pot1-unglazed_top/masks_colmap

Underside masks:
data/processed/pot1-unglazed_underside/masks_colmap
```

Validation confirmed an exact image-to-mask mapping for every view:

- 273 side images and 273 side masks
- 233 top images and 233 top masks
- 222 underside images and 222 underside masks
- 728 total image-mask pairs
- No missing masks
- No extra masks
- Consistent 2160 × 3840 dimensions

Visual inspection showed clean foreground boundaries. The top masks retained the interior, while the underside masks retained the bottom and lower body and excluded the cable, turntable, and blue surroundings. The top sequence had no quality-control warnings. The side and underside reports only flagged the first or last frames for low rotation-loop IoU, which is expected when the endpoints of a nominal full rotation are not perfectly identical.

## 6. Multiview Dataset Staging

A dedicated preparation tool was created to combine the three independently masked sequences into one collision-free COLMAP input dataset:

```text
scripts/reconstruction/prepare_multiview_dataset.py
configs/multiview_dataset_pot1_unglazed.yml
tests/reconstruction/test_prepare_multiview_dataset.py
```

The tool performs the following operations:

- validates all source image and mask folders;
- confirms exact filename correspondence;
- verifies that masks are valid binary images;
- checks consistent image and mask dimensions;
- prefixes filenames according to their source view;
- copies the RGB images and masks without modifying their contents;
- writes a manifest connecting every staged file to its source sequence and sequence index;
- refuses to overwrite existing generated outputs unless explicitly requested;
- validates that source and destination paths do not overlap dangerously.

Staged filename prefixes are:

```text
side_
top45_
underside_
```

Combined outputs:

```text
data/frames_output/pot1-unglazed_multiview_frames
data/processed/pot1-unglazed_multiview/masks_colmap
data/processed/pot1-unglazed_multiview/dataset_manifest.csv
```

The initial validation-only run reported:

```text
Multiview dataset validation complete
Sources: 3
Images/masks: 728/728
Resolution: 2160x3840
No output files were written
```

The completed staging dataset contains:

- 728 RGB images
- 728 COLMAP masks
- 728 manifest records
- 273 side entries
- 233 top-45-degree entries
- 222 underside entries
- Zero missing or extra files

Representative SHA-256 comparisons confirmed that the staged RGB images and masks were byte-for-byte copies of their sources.

## 7. Dataset-Staging Problem and Resolution

During development, the first temporary-directory implementation failed under Windows and OneDrive because `tempfile.mkdtemp` created directories with restrictive permissions. This caused file-copy failures even though the project directory itself was writable.

The staging transaction was changed to create ordinary sibling directories with `Path.mkdir`. These directories inherit the permissions of the parent project directory. The tool still completes validation and copying in a temporary staging location before publishing the finished dataset, so incomplete output is not presented as a successful result.

Seven focused preparation tests passed after this correction.

## 8. Multisequence COLMAP Matching Strategy

The existing masked COLMAP runner was extended with a `multisequence` matching mode:

```text
scripts/reconstruction/run_colmap.py
configs/colmap_pot1_unglazed_multiview.yml
tests/reconstruction/test_run_colmap.py
```

The configuration uses:

- COLMAP 4.1.1 with CUDA support
- `SIMPLE_RADIAL` camera model
- One shared camera
- Masked SIFT feature extraction
- Maximum 8,192 features per image
- Balanced GPU memory profile
- Guided matching
- Sequential overlap of 15 frames
- Loop closure within every capture sequence
- Sampled cross-sequence bridge pairs

The pair plan contains:

```text
Within-sequence pairs:        10,560
Loop-closure pairs:              675
Side-to-top bridge pairs:      2,688
Side-to-underside pairs:       2,576
Total explicit pairs:         16,499
```

This strategy preserves the ordered-video advantage inside each sequence while adding deliberate connections between different elevation rings. It avoids the 264,628 comparisons required by complete exhaustive matching.

No direct top-to-underside bridge was initially included because those views have little common visible surface. Both new rings were instead intended to connect through the side sequence.

The complete `tests/reconstruction` suite contained 52 passing tests after the multisequence support was added. A dry run successfully validated all 728 images, masks, manifest entries, output paths, and 16,499 planned image pairs without starting COLMAP.

## 9. Multiview Sparse-Reconstruction Result

The project owner ran the multiview sparse reconstruction with:

```powershell
python scripts/reconstruction/run_colmap.py --config configs/colmap_pot1_unglazed_multiview.yml
```

Workspace:

```text
data/processed/pot1-unglazed_multiview/colmap_sparse_masked_multiview
```

COLMAP produced three disconnected sparse models:

### Model 0 — Small Top-View Fragment

- Registered images: 10
- View membership: top-45-degree only
- Sparse points: 5,269
- Mean reprojection error: 0.820475 pixels
- Mean track length: 5.648890

This is a redundant fragment of the top sequence and is not a candidate for the final reconstruction.

### Model 1 — Complete Top and Interior Reconstruction

- Registered images: 233 of 233 top images
- Side images: 0
- Underside images: 0
- Sparse points: 58,481
- Mean reprojection error: 0.797265 pixels
- Mean track length: 11.643371

Manual COLMAP GUI inspection showed one coherent elevated camera ring and a detailed sparse reconstruction of the rim, opening, inner wall, and upper exterior. Model 1 is therefore a successful top-only reconstruction.

### Model 2 — Complete Side and Underside Reconstruction

- Registered images: 495
- Side images: 273 of 273
- Underside images: 222 of 222
- Top images: 0
- Sparse points: 96,911
- Mean reprojection error: 0.902436 pixels
- Mean track length: 10.653754

Manual COLMAP GUI inspection showed a coherent main pot with a large side camera orbit and a second underside camera ring. Model 2 is the strongest partial multiview result and should be preserved as the base model for the next alignment attempt.

## 10. Cross-View Matching Diagnosis

Read-only inspection of the COLMAP database and model files established the reason for the split reconstruction:

- Side-to-top geometrically verified pairs: 0
- Side-to-underside geometrically verified pairs: 31
- Top-to-top matching: strong
- Underside-to-underside matching: strong

The sampled normal SIFT bridge pairs were sufficient for COLMAP to connect all side and underside images, even though the cross-view connection was relatively weak. They were not sufficient to connect any top image to the side model.

This does not mean the top reconstruction failed. Model 1 registered all 233 top images with a lower reprojection error than Model 2. The failure occurred specifically at the transition between the side and top camera elevations. The available shared rim and upper-body features changed too strongly in scale, perspective, and visible surface area for the current SIFT matching stage to verify them.

## 11. Why the Two Good Models Cannot Yet Be Directly Merged

COLMAP's `model_merger` requires common registered images between its input reconstructions so that it can estimate a reliable similarity transformation.

Current overlap:

```text
Model 1: 233 top images
Model 2: 273 side + 222 underside images
Shared registered images: 0
```

Therefore, simply concatenating Model 1 and Model 2 would be geometrically invalid. The two models currently have independent coordinate systems, scales, rotations, and translations. A forced merge could produce duplicated, rotated, or incorrectly scaled pottery geometry.

Undistortion and 3DGS training have intentionally not been started for the 728-image dataset. Training separate unaligned camera systems as if they were one scene would produce an invalid Gaussian model.

## 12. LightGlue Alignment and Registration

The alignment stage established reliable side-to-top correspondences with SIFT-LightGlue using the installed COLMAP 4.1.1 build.

LightGlue will not directly merge the two sparse models. Its role is to create stronger image correspondences across the larger elevation change. Those correspondences can then allow COLMAP to register top cameras directly into Model 2's coordinate system.

Planned flow:

```text
Preserve the original database and Models 1 and 2
                         ↓
Create an isolated LightGlue workspace
                         ↓
Copy the database and Model 2 as the base model
                         ↓
Clear only failed side-to-top match records in the copied database
                         ↓
Run targeted SIFT-LightGlue side-to-top matching
                         ↓
Measure verified cross-view pairs and inlier distribution
                         ↓
Register top cameras into Model 2
                         ↓
Triangulate newly connected rim and interior points
                         ↓
Repeat registration for remaining top frames
                         ↓
Run bundle adjustment and inspect the unified model
```

The first LightGlue pass used selected bridge pairs rather than all 63,609 possible side-to-top combinations. This limited runtime and memory use on the NVIDIA GeForce GTX 1650 with 4 GB VRAM.

The existing SIFT keypoints and descriptors can be reused. Frame extraction, SAM2 segmentation, mask generation, and dataset staging do not need to be repeated.

Before camera registration, the bridge result should satisfy practical checks such as:

- multiple geometrically verified side-to-top pairs;
- approximately 30 or more inliers in useful pairs;
- matches distributed across several side and top frames;
- correspondences located on genuine pottery features rather than repeated background or turntable patterns.

If the existing images still produce no reliable connection, additional 3DGS training will not solve the camera-alignment problem. The recommended fallback is a short intermediate 20-to-30-degree elevation ring that visibly overlaps both the original side orbit and the current top-45-degree orbit.

The isolated workspace and targeted bridge-pair preparation were completed on
September 9, 2026. The preparation script validates the manifest, database,
existing SIFT features, and Model 2 before writing anything. It refuses to
overwrite an existing workspace and creates the result through a temporary
sibling directory so an interrupted copy cannot leave a partial final output.

Prepared workspace:

```text
data/processed/pot1-unglazed_multiview/colmap_sparse_masked_multiview_lightglue
```

Prepared contents:

- an exact copy of the 459,407,360-byte database;
- Model 2 copied to `base_model`;
- 2,688 unique side-to-top pairs in `bridge_pairs_lightglue.txt`;
- 56 side anchors and 48 top anchors selected with stride 5;
- empty registration, triangulation, final-output, and log directories.

The copied database cleanup was completed on September 9, 2026. Exactly 2,688
side-to-top records were removed from `matches` and another 2,688 were removed
from `two_view_geometries`. Those records contained 30 raw correspondence rows
and zero geometrically verified rows. All other match groups, all 728 images,
and all 1,939,238 existing SIFT keypoints and descriptors were preserved.
SIFT-LightGlue matching completed in 17.77 minutes. All 2,688 targeted pairs
were processed. The copied database contains 240 geometrically verified
side-to-top pairs, including 154 pairs with at least 30 inliers. Verified
pairs have a median of 41 inliers, a mean of 60.45, and a maximum of 232.
They cover 49 of the 56 selected side anchors and 46 of the 48 selected top
anchors across all four rotation quartiles.

The first image-registration pass kept all 495 Model 2 frames fixed and added
205 top frames, producing a 700-image model. Triangulation then increased the
point count from 96,911 to 174,557. A second registration pass used those new
3D points to register the remaining 28 top frames. The resulting
`registered_final` model contains all 728 images:

- 273 side images;
- 233 top-45-degree images;
- 222 underside images.

The final registered model contains 174,557 points, 1,539,247 observations,
mean track length 8.818019, and mean reprojection error 1.189781 pixels. The
original database and original sparse models remain unchanged.

Cleanup audit:

```text
data/processed/pot1-unglazed_multiview/colmap_sparse_masked_multiview_lightglue/logs/side_top_cleanup_report.json
```

## 13. Completed Camera-Pose Repair and Final COLMAP Model

The first LightGlue registration succeeded numerically but visual inspection
revealed large breaks in the side and top camera trajectories. The registered
images therefore could not be passed directly to 3DGS.

The repair transferred the smooth original side and top trajectories into the
connected Model 2 coordinate system through robust shared-landmark similarity
alignment. It retained the already smooth underside trajectory, assigned one
camera calibration to each capture group, discarded the old mixed point cloud,
and triangulated all points again from the preserved database correspondences.

Alignment results:

```text
Side-under alignment: 51/51 inliers, RMSE 0.002431
Top-side alignment:   1482/1853 inliers, RMSE 0.017108
Side continuity:      max/median adjacent spacing 1.153
Top continuity:       max/median adjacent spacing 1.113
Underside continuity: max/median adjacent spacing 1.130
```

The copied database was configured with three cameras:

```text
Camera 1: side       273 images
Camera 2: top45      233 images
Camera 3: underside  222 images
Database integrity: ok
```

Fresh triangulation produced one connected model. Conservative global bundle
adjustment kept the three camera intrinsics fixed while refining camera poses
and 3D points. It converged in 26 iterations:

```text
Initial cost: 0.596448 px
Final cost:   0.594971 px
Termination:  convergence
```

Accepted final COLMAP result:

```text
Registered images:       728 / 728
Cameras and rigs:        3 / 3
Sparse points:           164,756
Observations:            1,717,485
Mean reprojection error: 0.898626 px
```

COLMAP GUI inspection confirmed one pot and three smooth elevation rings. The
global scene tilt was accepted because a rigid world rotation does not change
camera-to-object geometry or 3DGS training.

## 14. Completed Undistortion and 3DGS Preparation

All 728 RGB images and binary masks were undistorted with the accepted repaired
model. The three source `SIMPLE_RADIAL` cameras became three `PINHOLE` cameras.

```text
Side and top: 1125 × 2000
Underside:    1116 × 2000
RGB images:   728
Masks:        728
```

The 3DGS loader was extended to support multiple `PINHOLE` cameras and mixed
per-view resolutions. Every image-mask pair remained aligned, binary, and
nonempty. The deterministic split contains 637 training images and 91 held-out
images. Factor-4 and factor-2 caches were prepared successfully.

## 15. Completed Multiview 3DGS Training

Three guarded runs were completed in the `pot-3dgs` environment:

```text
Smoke:    300 steps, 164,756 Gaussians, final loss 0.045632
Baseline: 7,000 steps, 500,000 Gaussians, final loss 0.0167034
Quality:  15,000 steps, 750,000 Gaussians, final loss 0.0125201
```

The final quality run took 44.6 minutes and used 0.87 GB peak VRAM. Its
91-image held-out evaluation produced:

```text
Mean PSNR:       26.9441 dB
Mean SSIM:       0.9319
Foreground PSNR: 26.1655 dB
Foreground L1:   0.02352
Alpha IoU:       0.92645
```

Per capture group:

| Group | Views | PSNR | SSIM | Foreground PSNR | Alpha IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Side | 35 | 27.9743 | 0.95067 | 26.1531 | 0.91912 |
| Top-45 | 29 | 24.7372 | 0.91519 | 23.9453 | 0.92782 |
| Underside | 27 | 27.9790 | 0.92552 | 28.5661 | 0.93447 |

The 15,000-step model improved every group over the accepted 7,000-step
baseline and was selected as the final export source.

## 16. Final Exported Artifacts

The final checkpoint was exported and independently verified:

```text
Native checkpoint: 114,003,568 bytes
Gaussian PLY:      114,000,952 bytes
Compact SPLAT:      24,000,000 bytes
Gaussians:                 750,000
```

```text
PLY SHA-256:
6b6b991c724984dd6808ee5d41b88161d16dc2fa01725c147f7f7311193f0c7c

SPLAT SHA-256:
4c3f1749d671fd91551f30327136a76da3d6afa6a68bf7e2f438d380913f07e1
```

The PLY header, fixed-size SPLAT record count, and recalculated checksums all
matched `export_manifest.json`. Both the local checkpoint viewer and the
external SuperSplat editor loaded the result.

## 17. Problems Encountered and How They Were Solved

### Problem 1 — Windows and OneDrive staging permissions

The first dataset-staging transaction used `tempfile.mkdtemp`, which created a
temporary folder with permissions that blocked subsequent copies.

Resolution: the staging tool now creates an ordinary sibling directory with
`Path.mkdir`, inheriting the project directory permissions while retaining
transactional publication and cleanup behavior.

### Problem 2 — Copied `database.db` access errors

The first LightGlue dry runs reported Windows `WinError 5` while probing the
copied database path. A partially prepared or inaccessible workspace could not
be trusted for mutation.

Resolution: the isolated workspace was recreated and validated so
`database.db` was a normal copied SQLite file. All commands were run with the
COLMAP GUI closed, and the runner verified the file, database schema, image
records, SIFT records, and SQLite integrity before matching. The original
database was never modified.

### Problem 3 — Standard SIFT produced no side-to-top bridge

Normal SIFT matching reconstructed each sequence well but produced zero
geometrically verified side-to-top pairs. Model 1 and Model 2 therefore had no
shared registered images and could not be safely merged.

Resolution: 2,688 targeted side-to-top pairs were rematched with
SIFT-LightGlue. This produced 240 verified pairs, including 154 with at least
30 inliers, distributed across the complete rotation. Top cameras were then
registered directly into Model 2 rather than merging independent models.

### Problem 4 — Missing CUDA 12 LightGlue runtime DLL

COLMAP's ONNX Runtime failed to load `onnxruntime_providers_cuda.dll` because
`cublasLt64_12.dll` was unavailable to the process. The existing `pot-3dgs`
environment intentionally used a different pinned stack.

Resolution: a dedicated `pot-lightglue` environment was created with a CUDA
12.8 runtime and cuDNN 9.5 or newer. The YAML-driven runner validates the DLLs
and prepends their directory only for the child COLMAP process, leaving
`pot-3dgs` unchanged.

### Problem 5 — Missing NumPy in the repair environment

The camera-pose repair utility could not start because NumPy was absent from
the first `pot-lightglue` environment definition.

Resolution: NumPy and the remaining Python dependencies were added to
`environment-lightglue.yml`, and the environment guide now uses
`micromamba env update` plus an import/version verification command.

### Problem 6 — Guided matching unsupported by SIFT-LightGlue

COLMAP 4.1.1 does not support its guided-matching option with the
`SIFT_LIGHTGLUE` feature-matching type.

Resolution: `FeatureMatching.guided_matching` was disabled only for the
LightGlue stage. Geometric verification and the configured inlier threshold
were retained.

### Problem 7 — All images registered but camera trajectories had gaps

The first 728-image registered model contained side steps about 26 times the
sequence median and top steps about 11.8 times the median. The three capture
videos had also been forced through one compromise camera calibration.

Resolution: smooth source trajectories were robustly aligned by shared 3D
landmarks, one camera/rig was assigned per capture group, old points were
discarded, and the model was freshly triangulated and bundle-adjusted. Final
max/median spacing ratios fell to approximately 1.11–1.15.

### Problem 8 — COLMAP conversion appeared to print nothing

`model_converter` returned directly to the prompt without a success message,
which looked like a failed command.

Resolution: the generated binary files were checked explicitly. The converted
model contained 728 registered images, three rigs, three cameras, and zero
points as intentionally required before fresh triangulation.

### Problem 9 — COLMAP GUI could not find the Qt `windows` plugin

Launching `bin/colmap.exe gui` directly did not configure Qt's platform-plugin
search path.

Resolution: GUI commands use `COLMAP.bat`, or set `QT_PLUGIN_PATH` to
`C:\Tools\COLMAP-4.1.1\plugins` before launching the executable.

### Problem 10 — Windows maximum-path failure during mask undistortion

Native COLMAP image I/O exceeded the legacy 260-character Windows path limit
inside the long OneDrive project location.

Resolution: intermediate names were shortened to `inputs/mask_src` and
`mask_undist`. The preparation runner now checks planned native paths before
launching COLMAP.

### Problem 11 — Original 3DGS loader assumed one camera and one resolution

The repaired reconstruction correctly contained three calibrations and two
undistorted image widths, which the first loader contract did not accept.

Resolution: scene loading, preparation, and validation were generalized for
multiple `PINHOLE` cameras and mixed per-view dimensions. Each image uses its
own scaled intrinsic matrix.

### Problem 12 — Local viewer did not match CUDA evaluation quality

The lightweight local viewer showed unstable color and appearance even when
all 750,000 Gaussians were requested.

Resolution: checkpoint acceptance was based on the full gsplat held-out
rasterizer and external PLY inspection. The local viewer uses base SH color and
is retained as a convenience inspection tool rather than the authoritative
quality renderer.

### Problem 13 — Uneven upper rim in genuinely novel views

The final PLY remains coherent, but SuperSplat inspection exposes mild uneven
or floating Gaussians around part of the upper rim. Increasing training from
7,000 to 15,000 steps improved metrics but did not remove this geometry issue.

Resolution and next action: the remaining limitation is attributed primarily
to insufficient common surface coverage between the side and top-45-degree
rings. Capture a full 20–30-degree intermediate ring and connect
`side ↔ mid25 ↔ top45`. This requires new camera registration, triangulation,
bundle adjustment, undistortion, and fresh 3DGS training; more iterations on
the current images are not expected to solve it.

## 18. Current Accepted Paths and Preservation Requirements

```text
Final COLMAP database:
data/processed/pot1-unglazed_multiview/colmap_sparse_masked_multiview_lightglue_repaired/database.db

Final COLMAP model:
data/processed/pot1-unglazed_multiview/colmap_sparse_masked_multiview_lightglue_repaired/bundle_adjusted_final

Undistorted dataset:
data/processed/pot1-unglazed_multiview/colmap_undistorted_masked_multiview_lightglue_repaired

Final checkpoint:
data/processed/pot1-unglazed_multiview/gaussian_splatting_masked_multiview_lightglue_repaired/runs/quality_multiview_repaired/checkpoints/step_015000.pt

Verified exports:
data/processed/pot1-unglazed_multiview/gaussian_splatting_masked_multiview_lightglue_repaired/runs/quality_multiview_repaired/exports
```

Before storage cleanup, preserve the copied repaired database, final COLMAP
model, original frames and masks, final checkpoint, PLY, SPLAT, and export
manifest. Undistortion inputs, mask-undistortion workspaces, prepared 3DGS
caches, smoke runs, baseline runs, and comparison PNGs are regenerable.

No Git commit or push was performed during this work.
