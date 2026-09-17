# Week 14 Lid 3DGS Report Update Copy-Paste Text

This drafting guide adds the separately reconstructed terracotta lid to the
current computer-vision project report. It follows the structure of the Week
13 multiview update and uses the artifacts verified on September 17, 2026.

The lid must be described as an independent object. It was captured, masked,
reconstructed, trained, evaluated, and exported separately from the main pot
because its pose can change relative to the pot.

## Editing Map

| Report location | Action | Content to add |
| --- | --- | --- |
| Abstract | Append before the final concluding sentence | Separate lid result, 214 registered images, 7,000-step 3DGS result, and held-out metrics |
| III.E Image Acquisition and Dataset Preparation | Append | Lid side, top, and inverted underside capture experiment; selected side-only dataset |
| III.F Foreground Mask Generation and Quality Control | Append | Independent lid masks and mask-aligned side-only staging |
| III.G Sparse Reconstruction and Camera-Pose Selection | Append | Two disconnected complete models, failed bridge matching, and selection of the side model |
| III.H Mask-Aware 3D Gaussian Splatting Preparation | Append | Undistortion, 187/27 split, and factor-two preparation |
| III.I Low-Memory 3D Gaussian Splatting Training | Append | Lid smoke test and 7,000-step training result |
| III.J Evaluation Export and Interactive Viewing | Append | Lid held-out metrics, verified exports, and cleaned lower export |
| IV Results and Analysis | Add a new lid subsection | Visual result, quantitative result, failed underside integration, and current limitation |
| IV Limitations and Future Work | Append | Required bridge capture between exterior and underside |
| After the lid methodology text | Add Tables I-III below | Capture/reconstruction, training/evaluation, and export summary |

Do not describe the colorful unseen underside as reconstructed geometry. The
selected model was trained from the exterior side orbit, so its quantitative
evaluation applies to held-out views within that captured orbit.

## Abstract Addition

Append the following text before the final sentence of the current abstract.

```text
The removable terracotta lid was reconstructed as a separate object from 214 masked side-orbit images. COLMAP registered all 214 images into one model containing 31,303 sparse points, and a 7,000-step 3D Gaussian Splatting run produced 500,000 Gaussians. Evaluation on 27 held-out views achieved a mean PSNR of 37.7711 dB, a mean SSIM of 0.98494, a foreground PSNR of 30.9432 dB, and an alpha IoU of 0.96782. The exterior dome, knob, carved decoration, rim, and visible crack were reproduced accurately within the captured orbit. A separately recorded underside sequence could not be connected geometrically to the exterior sequence because the two views shared insufficient visible surface coverage, so the current accepted lid model is exterior-only.
```

## III Methodology

### E Separate Lid Capture and Dataset Selection

Append the following paragraphs to the image-acquisition subsection.

```text
The removable lid was treated as a separate reconstruction target because it can rotate and translate independently from the main pottery body. Three turn-around videos were recorded: an exterior side orbit, a higher exterior view, and an underside view recorded while the lid was inverted. Every sixth decoded frame was selected to retain strong neighboring overlap while limiting redundant images. This produced 214 side images and 202 underside images. The underside frames were rotated by 180 degrees after extraction to provide a consistent displayed orientation for annotation and inspection. The original files were retained unchanged.

The higher exterior sequence was not included in the selected model because the side orbit already observed the knob, sloped exterior, carved band, rim, and crack with sufficient overlap. The main reconstruction experiment therefore evaluated whether the 214-image side sequence and 202-image rotated underside sequence could be joined. When no geometrically valid connection could be established, the 214-image side sequence was selected as the final training dataset. This decision preserved a strong and internally consistent exterior reconstruction instead of manually forcing two independently scaled models together.
```

### F Lid Foreground Mask Generation and Quality Control

Append the following text to the masking subsection.

```text
SAM 2 foreground masks were generated independently for the side and rotated underside sequences. The lid prompt box enclosed the complete pottery silhouette. Positive prompts were placed on the knob, sloped body, carved exterior, rim, and concave underside surface as appropriate. Negative prompts were placed on the white turntable, fabric background, cable, and other visible non-object regions. The side sequence produced 214 image-mask pairs. The underside sequence produced 202 image-mask pairs, although 50 quality-control entries indicated boundary contact because the object was framed close to the image edge.

For the accepted side-only dataset, all 214 RGB images had exact corresponding binary COLMAP masks at 2160 by 3840 pixels. The staged filenames were prefixed with side so they matched the registered COLMAP image names. After camera undistortion, every RGB image retained a nonempty, binary, pixel-aligned mask. Foreground ratios were stable across representative final masks, and visual inspection confirmed that the knob, dome, carved decoration, rim, and crack remained inside the silhouettes.
```

### G Lid Sparse Reconstruction and View-Connection Experiment

Append the following paragraphs to the sparse-reconstruction subsection.

```text
Masked COLMAP reconstruction was first run on the combined 416-image lid dataset. Sequential matching was performed within each ordered sequence, and 1,848 sampled bridge pairs were planned between the side and underside groups. The mapper reconstructed both capture groups completely but placed them in two disconnected models. The 202-image underside model contained 36,468 points, 548,698 observations, a mean track length of 15.0460, and a mean reprojection error of 0.884665 pixels. The 214-image side model contained 31,303 points, 274,520 observations, a mean track length of 8.7698, and a mean reprojection error of 1.051550 pixels.

Database inspection found zero raw or geometrically verified side-to-underside correspondences. An isolated SIFT-LightGlue experiment then processed the same 1,848 targeted bridge pairs while preserving the original database and sparse models. It also produced zero usable cross-view correspondences. The concave underside and decorated exterior shared too little simultaneously visible texture, and the abrupt physical inversion created a viewpoint gap that feature matching could not bridge.

The two sparse models were not manually aligned because independent reconstructions have unrelated scale, orientation, and translation. A visual alignment would not create valid shared camera geometry for 3D Gaussian Splatting. The complete 214-image side model was therefore selected. COLMAP GUI inspection confirmed one smooth closed camera orbit around a coherent lid point cloud with no isolated cameras.
```

### H Lid 3D Gaussian Splatting Preparation

Append the following text to the 3DGS preparation subsection.

```text
The selected 214-image side model was prepared in a separate lid workspace. COLMAP undistortion converted the 2160 by 3840 source frames to 1116 by 2000 pixels while preserving all 214 registered cameras and 31,303 initialization points. The masks passed through the same camera model and resizing process as the RGB images, producing 214 aligned binary masks.

The full training profile used a factor-two cache at 558 by 1000 pixels. A deterministic every-eighth-image holdout rule divided the dataset into 187 training images and 27 test images. All dataset, camera, image, mask, and filename checks passed before training.
```

### I Lid Low-Memory 3D Gaussian Splatting Training

Append the following paragraphs to the training subsection.

```text
A 300-step smoke run was completed first using factor-four images at 279 by 500 pixels. It retained the 31,303 initialization Gaussians, reached a final loss of 0.0130714, required 11.81 seconds, and used 0.0772 GB peak GPU memory. The smoke result confirmed correct camera orientation, masked compositing, rasterization, checkpoint writing, and evaluation. Its low alpha IoU was expected because refinement began after the smoke run's final step.

The accepted lid model was then trained for 7,000 steps using factor-two images, spherical-harmonic degree two, packed rasterization, sparse gradients, and Markov Chain Monte Carlo refinement. The model grew from 31,303 initialization points to the configured limit of 500,000 Gaussians. Training completed in 642.89 seconds, or approximately 10 minutes 43 seconds, with a final loss of 0.0046292 and 0.6258 GB peak GPU memory. The run completed within the 4 GB memory limit of the NVIDIA GeForce GTX 1650.
```

### J Lid Evaluation, Export, and Viewer Cleanup

Append the following text to the evaluation and export subsection.

```text
The 7,000-step checkpoint was evaluated on all 27 held-out images at 558 by 1000 pixels. The model achieved a mean full-frame PSNR of 37.7711 dB and a mean SSIM of 0.98494. Foreground-only evaluation produced a PSNR of 30.9432 dB and a mean absolute error of 0.01477. The mean alpha IoU was 0.96782, confirming close silhouette agreement across the captured exterior orbit.

The complete checkpoint was exported as a 76,000,952-byte PLY file and a 16,000,000-byte SPLAT file. Both contain 500,000 Gaussian records, and their recalculated SHA-256 values match the export manifest. External viewer inspection showed stable exterior appearance, including the knob, carved band, rim, and crack. When the virtual camera moved beneath the object, unsupported colorful Gaussians became visible because no registered training camera observed the underside.

An export-only cleanup was therefore created for presentation. A lower clipping plane was inferred from the registered camera orientation and COLMAP support range. The operation removed 35,139 unsupported lower Gaussians, or 7.03 percent of the model, while retaining 464,861 Gaussians. It produced separate PLY and SPLAT files and did not modify the original checkpoint or complete exports. This cleanup reduces colorful lower floaters but does not reconstruct the real concave underside.
```

## Table I. Lid Capture and Sparse Reconstruction Summary

| Capture group | Images | Registered | Sparse points | Observations | Mean track length | Reprojection error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Side exterior | 214 | 214 | 31,303 | 274,520 | 8.7698 | 1.051550 px |
| Rotated underside | 202 | 202 | 36,468 | 548,698 | 15.0460 | 0.884665 px |
| Cross-view bridge | 1,848 planned pairs | 0 connected pairs | N/A | N/A | N/A | N/A |

Suggested caption:

```text
TABLE I. Independent lid reconstructions and the unsuccessful exterior-to-underside bridge experiment.
```

## Table II. Lid 3DGS Training and Held-Out Evaluation

| Item | Smoke run | Accepted run |
| --- | ---: | ---: |
| Training steps | 300 | 7,000 |
| Image resolution | 279 x 500 | 558 x 1000 |
| Training images | 187 | 187 |
| Held-out images | 27 | 27 |
| Final Gaussians | 31,303 | 500,000 |
| Final loss | 0.0130714 | 0.0046292 |
| Training time | 11.81 s | 642.89 s |
| Peak training VRAM | 0.0772 GB | 0.6258 GB |
| Mean PSNR | 31.0536 dB | 37.7711 dB |
| Mean SSIM | 0.87498 | 0.98494 |
| Foreground PSNR | 23.6738 dB | 30.9432 dB |
| Foreground L1 | 0.04189 | 0.01477 |
| Alpha IoU | 0.19118 | 0.96782 |

Suggested caption:

```text
TABLE II. Training and held-out evaluation results for the separate lid 3D Gaussian Splatting model.
```

## Table III. Verified Lid Export Artifacts

| Export | Gaussians | Size |
| --- | ---: | ---: |
| Complete PLY | 500,000 | 76,000,952 bytes |
| Complete SPLAT | 500,000 | 16,000,000 bytes |
| Cleaned lower PLY | 464,861 | 70,659,824 bytes |
| Cleaned lower SPLAT | 464,861 | 14,875,552 bytes |

Suggested caption:

```text
TABLE III. Verified complete and presentation-cleaned exports for the lid reconstruction.
```

## IV Results and Analysis

### Separate Lid Reconstruction Result

Add the following subsection to the results section.

```text
The separate lid experiment produced a strong exterior 3D Gaussian Splatting model. All 214 selected side images were registered in one closed camera orbit, and the sparse reconstruction supplied 31,303 initialization points. The final 7,000-step result reproduces the rounded knob, sloped dome, carved decorative band, circular rim, surface color, and visible structural crack. Held-out renders remain stable throughout the captured orbit and achieve 37.7711 dB mean PSNR, 0.98494 mean SSIM, and 0.96782 alpha IoU.

The quantitative result is stronger than the short smoke test across every reported metric. Densification increased the representation from 31,303 to 500,000 Gaussians, while the foreground mean absolute error decreased from 0.04189 to 0.01477. Visual inspection agrees with these values: fine decoration and the crack remain legible, the silhouette is stable, and the knob does not detach during rotation.

The underside remains the main limitation. Although all 202 underside images reconstructed successfully as an independent COLMAP model, neither standard SIFT nor targeted SIFT-LightGlue produced a valid connection to the side sequence. The accepted training cameras therefore observe only the exterior. Views placed below the rim expose view-dependent colors and elongated Gaussians that were never constrained by real underside images. The cleaned lower export removes the most visible unsupported floaters, but the missing surface is left empty rather than presented as measured geometry.

Separating the lid from the main pot remains the correct representation for later AR or VR assembly. The lid can be positioned on the pot, rotated, lifted, or removed without requiring the main pottery reconstruction to be retrained.
```

### Lid Reconstruction Limitations and Next Capture

Append the following text to the limitations and future-work discussion.

```text
The failed exterior-to-underside bridge demonstrates that two individually strong reconstructions cannot be combined safely without shared visual evidence. A future lid capture should include an additional full rotation with the lid held nearly edge-on or tilted approximately 60 to 80 degrees. Each bridge frame should show part of the decorated exterior, the rim and crack, and part of the concave underside simultaneously. The complete rim should remain inside the frame, lighting should remain fixed, and neighboring frames should retain high overlap. The intended registration chain is exterior side to bridge rotation to underside, followed by fresh triangulation, bundle adjustment, undistortion, and 3DGS training.

Additional optimization of the current side-only checkpoint is not expected to create an accurate underside because the required surface was absent from the registered training views. Until a bridge capture is available, interactive presentation should favor the exterior viewing range or use the cleaned export to suppress unsupported lower floaters.
```

## Problems Encountered and Solutions

### Problem 1 - Exterior and underside formed disconnected models

```text
COLMAP registered every image in both lid sequences, but the combined mapper produced two independent models. This indicated insufficient shared visual evidence rather than failure within either sequence.

Solution: preserve both models, inspect database bridge records, and avoid manual model alignment. The complete side model was selected for the accepted exterior result.
```

### Problem 2 - Targeted LightGlue produced no bridge correspondences

```text
Targeted matching processed 1,848 sampled side-to-underside pairs but produced no usable raw or geometrically verified correspondences. The two views were separated by an abrupt inversion and shared little visible texture.

Solution: clean only the targeted bridge records in an isolated database copy, verify that all original features remained intact, and stop the merge attempt when the second matcher also produced no geometric evidence.
```

### Problem 3 - Underside masks touched the image boundary

```text
Fifty underside masks were flagged because the lid was framed close to the image edge. The masks generally followed the pottery correctly, but the restricted framing reduced reliable transition coverage at the rim.

Solution: retain the sequence for the independent underside experiment and require wider framing in the future bridge capture.
```

### Problem 4 - Bottom view showed colorful unsupported Gaussians

```text
External viewer inspection showed excellent exterior quality but unstable color below the lid. These Gaussians were unconstrained because the selected model contained no registered underside cameras.

Solution: create a separate export-only lower cleanup. It removes 35,139 Gaussians below the inferred supported boundary while preserving the original checkpoint and complete exports. The cleaned output is identified as a presentation derivative rather than reconstructed underside geometry.
```

## Suggested Figure Captions

Insert only figures that remain legible in the report layout.

```text
Fig. X. Representative side-view lid frame and SAM 2 annotation, including positive pottery prompts and negative background prompts.

Fig. X. COLMAP sparse reconstruction of the lid exterior showing all 214 registered cameras in one closed orbit around the 31,303-point model.

Fig. X. Comparison of the disconnected side and underside lid reconstructions. Both capture groups registered completely, but no geometrically verified bridge connected their coordinate systems.

Fig. X. Held-out lid evaluation view showing the reference image and the rendering produced by the 7,000-step 3D Gaussian Splatting model.

Fig. X. External-viewer inspection of the exported lid showing stable knob, dome, carved decoration, rim, and crack reconstruction.

Fig. X. Unsupported lower Gaussians visible from beneath the side-only model before export cleanup.

Fig. X. Presentation-cleaned lid export after removing 35,139 unsupported lower Gaussians. The cleanup suppresses floaters but does not reconstruct the physical underside.
```

## Current Accepted Artifact Paths

```text
Selected side sparse model:
data/processed/lid_side_underside/colmap_sparse_masked/sparse/1

Undistorted side-only 3DGS dataset:
data/processed/lid_side_only/gs_input

Accepted training checkpoint:
data/processed/lid_side_only/gaussian_splatting/runs/baseline_lid_7k/checkpoints/step_007000.pt

Held-out evaluation:
data/processed/lid_side_only/gaussian_splatting/runs/baseline_lid_7k/evaluation/holdout_black

Complete exports:
data/processed/lid_side_only/gaussian_splatting/runs/baseline_lid_7k/exports

Presentation-cleaned exports:
data/processed/lid_side_only/gaussian_splatting/runs/baseline_lid_7k/exports/clean_lower_v1
```

Preserve the original frames, masks, selected COLMAP model, final checkpoint,
evaluation summary, complete PLY and SPLAT files, cleaned exports, and both
export manifests. Prepared caches and smoke-run outputs can be regenerated.

No Git commit or push was performed during this documentation update.
