# Week 14 Pot 1 Three-Angle Replacement Report Copy-Paste Text

This guide replaces the accepted Pot 1 multiview result described in
`w13 Multiview 3DGS Report Update Copy-Paste Text.md`. The earlier
side + top-45-degree + underside reconstruction remains a useful development
experiment, but the current accepted Pot 1 model uses the following three
capture groups:

- 273 side images;
- 221 intermediate images recorded at approximately 25 degrees; and
- 222 underside images.

The new dataset contains 716 registered images. It was introduced to give the
upper rim a smoother change in viewing elevation than the previous
top-45-degree sequence. All values below were read from the final reconstruction,
training, evaluation, and export records. SHA-256 values are intentionally
omitted from the report tables.

## Replacement Map

Apply these changes to the current Word report.

| Report location | Action |
| --- | --- |
| Abstract | Replace the old 728-image and top-45-degree Pot 1 result with the new 716-image intermediate-25-degree result. |
| III.E Image Acquisition and Dataset Preparation | Replace the Pot 1 top-45-degree capture description and its dataset table. |
| III.G Sparse Reconstruction and Camera-Pose Selection | Replace the old LightGlue and repaired 728-camera account with the new intermediate-ring registration account. |
| III.H Mask-Aware 3D Gaussian Splatting Preparation | Replace the old 637/91 split with the new 626/90 split. |
| III.I Low-Memory 3D Gaussian Splatting Training | Replace the old Pot 1 15,000-step result with the accepted 7,000-step result. |
| III.J Evaluation Export and Interactive Viewing | Replace the old 91-view metrics and 750,000-Gaussian export values. |
| IV.A Current Progress and Preliminary Analysis | Replace the final Pot 1 multiview progress paragraphs. |
| IV.K-O Pot 1 multiview results | Replace the old capture, reconstruction, evaluation, limitation, and next-experiment text. |
| Pot 1 multiview tables and figures | Replace references to `top45` and 728 images with `mid25` and 716 images. |

Keep the side-only Pot 1 baseline, the controlled classical-versus-side-only
comparison, and the separate lid material unchanged. When the previous
top-45-degree experiment is mentioned, label it as an earlier experiment rather
than the final Pot 1 reconstruction.

## Abstract Replacement

Replace the complete abstract with the following version if the abstract still
contains the earlier 728-image Pot 1 result.

```text
Abstract-This project investigates image-based three-dimensional reconstruction of pottery associated with Thai material culture for interactive AR or VR presentation. The original 273-image side-view dataset was processed through classical photogrammetry and 3D Gaussian Splatting branches to compare an explicit textured surface with a radiance-based representation under the same captured views. Additional camera coverage was then recorded for the first unglazed pot. An intermediate orbit at approximately 25 degrees was selected to provide gradual overlap with the side sequence, while an inverted underside sequence supplied direct observations of the foot and bottom surface. The accepted three-angle dataset contains 273 side images, 221 intermediate images, and 222 underside images. All 716 cameras were registered in one coordinate system. Fresh triangulation and final bundle adjustment produced 130,577 sparse points, 1,523,453 observations, a mean track length of 11.667, and a mean reprojection error of 0.996169 pixels. A mask-aware 3D Gaussian Splatting model was trained for 7,000 steps and reached 500,000 Gaussians. Evaluation on 90 held-out views produced a mean PSNR of 31.8255 dB, a mean SSIM of 0.95250, a foreground PSNR of 28.5096 dB, and an alpha IoU of 0.95555. The intermediate camera ring produces a more coherent Pot 1 result than the earlier top-45-degree experiment and is therefore selected as the current accepted multiview reconstruction.
```

## III Methodology

### E Image Acquisition and Dataset Preparation

Replace the old Pot 1 paragraph that describes the top-45-degree sequence with
the following text.

```text
Inspection of the first multiview Pot 1 reconstruction showed that the direct change from the side orbit to the approximately 45-degree elevated orbit left insufficient gradual overlap around part of the upper rim. A new intermediate sequence was therefore recorded at approximately 25 degrees above the original side orbit. This angle retained substantial visibility of the exterior body and rim while adding clearer observations of the opening and inner surface. The original underside sequence was retained and its extracted images remained rotated by 180 degrees so that their displayed orientation was consistent during annotation and reconstruction.

The accepted Pot 1 dataset contains 273 side images, 221 intermediate-25-degree images, and 222 underside images, for a total of 716 RGB frames. Every image has a corresponding binary foreground mask. The three groups were staged with the prefixes side, mid25, and underside so that their source sequence and camera elevation remained identifiable throughout COLMAP reconstruction, 3D Gaussian Splatting training, evaluation, and export. The earlier top-45-degree sequence is excluded from this accepted dataset.
```

Insert the following table after the capture description.

| Capture group | Images | Training | Held-out | Main visible area |
| --- | ---: | ---: | ---: | --- |
| Side | 273 | 239 | 34 | Exterior body, carved band, and lower profile |
| Intermediate 25-degree | 221 | 193 | 28 | Upper body, rim, opening, and visible interior |
| Underside | 222 | 194 | 28 | Foot and bottom surface |
| **Total** | **716** | **626** | **90** | **Accepted three-angle coverage** |

Suggested caption:

```text
TABLE X. Composition and purpose of the accepted three-angle Pot 1 dataset.
```

### F Foreground Mask Generation and Quality Control

Use this short replacement paragraph where the previous text describes masks
for the top-45-degree sequence.

```text
SAM 2 mask propagation was performed independently for the intermediate-25-degree sequence. Positive prompts were placed on the exterior pottery surface, rim, and visible interior, while negative prompts excluded the turntable and background. The previously validated side and underside masks were reused. Dataset validation confirmed one aligned binary mask for each of the 716 RGB images and preserved the source resolution of 2160 by 3840 pixels before COLMAP undistortion.
```

### G Sparse Reconstruction and Camera-Pose Selection

Replace the previous 728-camera Pot 1 reconstruction account with the following
text.

```text
The accepted reconstruction reused the connected 495-camera side-plus-underside model as its geometric base and registered the 221 intermediate-25-degree images into the same coordinate system. In the first incremental registration pass, the existing 495 poses were preserved exactly and 210 intermediate cameras were added, producing a 705-image model. After additional triangulation, a second registration pass recovered the remaining 11 intermediate cameras. The resulting model contained all 716 images and three camera calibrations.

The provisional point cloud was discarded after registration so that inconsistent points inherited from intermediate processing would not initialize the final Gaussian model. Fresh triangulation preserved all camera poses to numerical precision and increased the sparse model from 118,945 provisional points to 130,577 fresh points. Final bundle adjustment refined camera poses and 3D points while retaining the complete 716-image registration. The mean reprojection error decreased from 0.999984 to 0.996169 pixels, and the maximum camera-pose parameter change was 0.00709. The accepted COLMAP model contains three cameras, 130,577 sparse points, 1,523,453 observations, a mean track length of 11.667, and an average of 2,127.73 observations per image.
```

Insert the following reconstruction table after this paragraph.

| Reconstruction stage | Registered images | Intermediate images | Sparse points |
| --- | ---: | ---: | ---: |
| Side + underside base | 495 | 0 | Existing connected base model |
| Registration pass 1 | 705 | 210 of 221 | 101,575 |
| Registration pass 2 | 716 | 221 of 221 | 118,945 |
| Fresh triangulation and final bundle adjustment | 716 | 221 of 221 | 130,577 |

Suggested caption:

```text
TABLE X. Registration and fresh-triangulation progression of the accepted Pot 1 three-angle COLMAP model.
```

### H Mask-Aware 3D Gaussian Splatting Preparation

Replace the old Pot 1 preparation counts with the following paragraph.

```text
The final 716-camera COLMAP model, undistorted RGB images, and aligned masks were staged as the input to the mask-aware 3D Gaussian Splatting pipeline. The three capture groups use three PINHOLE camera calibrations. Factor-two preparation produced images with heights of 1000 pixels and widths of either 558 or 562 pixels, depending on the source camera. A deterministic ordered every-eighth-image rule created 626 training images and 90 held-out images. The held-out set contains 34 side views, 28 intermediate views, and 28 underside views.
```

### I Low-Memory 3D Gaussian Splatting Training

Replace the old 15,000-step Pot 1 quality-run paragraph with this text.

```text
The final sparse point cloud initialized training with 130,577 points. The accepted Pot 1 model was trained for 7,000 steps at factor-two resolution using degree-two spherical harmonics, packed rasterization, sparse gradients, random training backgrounds, and Markov Chain Monte Carlo refinement. Training completed in 864.23 seconds, or approximately 14.4 minutes. The model reached the configured limit of 500,000 Gaussians with a final loss of 0.0150615 and used 0.641 GB of peak GPU memory. The result was accepted after quantitative evaluation and external-viewer inspection showed better continuity than the earlier top-45-degree reconstruction.
```

### J Evaluation Export and Interactive Viewing

Replace the previous Pot 1 evaluation and export paragraph with the following
text.

```text
Evaluation was performed on all 90 held-out images using a black background. The accepted model achieved a mean PSNR of 31.8255 dB, a mean SSIM of 0.95250, a foreground PSNR of 28.5096 dB, a foreground mean absolute error of 0.01990, and an alpha IoU of 0.95555. The model was exported at step 7,000 as both PLY and SPLAT files. The PLY contains 500,000 Gaussian records and occupies 76,000,952 bytes, while the SPLAT export occupies 16,000,000 bytes. These exports represent the current accepted Pot 1 multiview artifact.
```

## IV Results and Analysis

### A Current Progress and Preliminary Analysis

Replace the old final Pot 1 multiview paragraphs with the following progress
summary.

```text
The earlier side, top-45-degree, and underside experiment successfully reconstructed the complete pot but retained uneven Gaussian structure around part of the upper rim. A new capture ring at approximately 25 degrees was recorded to reduce the elevation gap. The accepted replacement dataset contains 716 images: 273 side views, 221 intermediate views, and 222 underside views.

All 221 intermediate cameras were registered into the existing 495-camera side-plus-underside coordinate system. Fresh triangulation and final bundle adjustment produced a 130,577-point COLMAP model with all 716 images registered and a mean reprojection error of 0.996169 pixels. The repaired model was then used to train a 7,000-step Gaussian Splatting model containing 500,000 Gaussians.

Evaluation on 90 held-out images produced a mean PSNR of 31.8255 dB, a mean SSIM of 0.95250, a foreground PSNR of 28.5096 dB, and an alpha IoU of 0.95555. Visual inspection also showed a more coherent transition around the upper part of the pot. The side + intermediate-25-degree + underside model therefore replaces the earlier side + top-45-degree + underside result as the accepted Pot 1 multiview reconstruction.
```

### K Accepted Three-Angle Pot 1 Reconstruction

Use this subsection in place of the old Pot 1 multiview capture and initial
reconstruction subsection.

```text
The intermediate-25-degree sequence addressed a specific limitation discovered during free-viewpoint inspection of the first multiview result. The earlier elevated orbit observed the opening and interior clearly, but its angle differed sharply from the side orbit. The intermediate ring observes many of the same rim and exterior features as the side images while also looking into the opening. It therefore provides a more gradual transition in camera elevation.

Registration was performed against the already connected side-plus-underside model. The first pass registered 210 intermediate cameras without changing the 495 existing poses, and the second pass recovered the remaining 11 cameras. This produced complete registration without including any images from the old top-45-degree sequence. Fresh triangulation then rebuilt the scene points from the accepted camera set before the final bundle-adjustment stage.
```

### L Final Sparse Reconstruction Quality

```text
The final sparse reconstruction contains 716 registered images, three cameras, 130,577 points, and 1,523,453 observations. Its mean track length is 11.667, meaning that each reconstructed point is observed in approximately 11.7 images on average. Final bundle adjustment reduced the mean reprojection error to 0.996169 pixels while making only small changes to the accepted poses. COLMAP inspection showed the pottery surrounded by the three intended camera trajectories and confirmed that the intermediate orbit was placed between the side and underside coverage in the shared coordinate system.
```

### M Three-Angle Gaussian Splatting Results

```text
The accepted 7,000-step model reached 500,000 Gaussians and a final loss of 0.0150615. Across all 90 held-out images, it achieved a mean PSNR of 31.8255 dB, a mean SSIM of 0.95250, a foreground PSNR of 28.5096 dB, and an alpha IoU of 0.95555. The side group produced the highest PSNR and SSIM, while the underside group remained the most difficult in full-frame PSNR. The intermediate group achieved a mean PSNR of 32.3601 dB and an alpha IoU of 0.96322, indicating that its rim and upper-surface observations were reconstructed consistently.
```

Insert the following evaluation table after the paragraph.

| Capture group | Held-out views | PSNR dB | SSIM | Foreground PSNR dB | Foreground L1 | Alpha IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Side | 34 | 33.5091 | 0.97034 | 28.2456 | 0.02127 | 0.95201 |
| Intermediate 25-degree | 28 | 32.3601 | 0.95704 | 29.1554 | 0.02002 | 0.96322 |
| Underside | 28 | 29.2466 | 0.92630 | 28.1845 | 0.01812 | 0.95217 |
| **Overall** | **90** | **31.8255** | **0.95250** | **28.5096** | **0.01990** | **0.95555** |

Suggested caption:

```text
TABLE X. Held-out evaluation of the accepted 7,000-step Pot 1 three-angle Gaussian Splatting model.
```

### N Replacement of the Earlier Pot 1 Result

```text
The new result supersedes the earlier 728-image side + top-45-degree + underside model. The old model used 233 elevated images, 15,000 training steps, and 750,000 Gaussians. Its own 91-view evaluation reported a mean PSNR of 26.9441 dB, a mean SSIM of 0.93190, a foreground PSNR of 26.1655 dB, and an alpha IoU of 0.92645. The new model uses 221 intermediate images, 7,000 training steps, and 500,000 Gaussians. Its own 90-view evaluation reports stronger values for all four measurements and visual inspection shows a better upper-rim transition.

The two overall metric rows come from different elevated image sets and are therefore reported as evidence for their respective models rather than as a strictly controlled same-image benchmark. The selection of the newer model is based on both its held-out evaluation and its improved free-viewpoint appearance.
```

Insert the following replacement comparison table.

| Measurement | Earlier top-45-degree model | Accepted intermediate-25-degree model |
| --- | ---: | ---: |
| Capture groups | Side + top-45-degree + underside | Side + intermediate-25-degree + underside |
| Total images | 728 | 716 |
| Sparse points | 164,756 | 130,577 |
| Mean reprojection error | 0.898626 px | 0.996169 px |
| Training steps | 15,000 | 7,000 |
| Gaussians | 750,000 | 500,000 |
| Held-out views | 91 | 90 |
| Mean PSNR | 26.9441 dB | 31.8255 dB |
| Mean SSIM | 0.93190 | 0.95250 |
| Foreground PSNR | 26.1655 dB | 28.5096 dB |
| Alpha IoU | 0.92645 | 0.95555 |
| Report status | Previous experiment | **Accepted Pot 1 result** |

Suggested caption:

```text
TABLE X. Recorded results of the earlier top-45-degree experiment and the accepted intermediate-25-degree Pot 1 model. The held-out sets differ and the table is not a same-image benchmark.
```

### O Current Limitation and Next Step

```text
The accepted model improves the upper-rim transition and provides useful exterior, opening, interior, foot, and underside coverage. The underside remains the most difficult capture group according to SSIM and full-frame PSNR, and views far outside the three captured trajectories may still expose unsupported Gaussian structure. Future work should prioritize matched-view visual comparison, conservative removal of isolated floaters when necessary, and scene-level alignment of the separately reconstructed pot and lid. Additional Pot 1 training should be performed only when visual inspection identifies a specific remaining defect that the existing observations can support.
```

## Suggested Figure Captions

Use the next available figure numbers in the Word document.

```text
Fig. X. Representative side, intermediate-25-degree, and underside images from the accepted 716-image Pot 1 dataset.

Fig. X. Final COLMAP reconstruction of Pot 1 showing all 716 registered cameras and the 130,577-point sparse model.

Fig. X. Accepted 7,000-step Pot 1 Gaussian Splatting result rendered from side, elevated, and underside viewpoints.

Fig. X. External-viewer inspection of the accepted Pot 1 export showing the improved transition around the upper rim.

Fig. X. Comparison of the earlier top-45-degree Pot 1 result and the accepted intermediate-25-degree reconstruction from similar novel viewpoints.
```

For the figure showing the reconstructed pot with colored orbit guides, use:

```text
Fig. X. External-viewer inspection of the accepted three-angle Pot 1 Gaussian Splatting model with camera-orbit guides.
```

## Accepted Artifact Paths

These paths are for repository documentation and verification. They do not
need to appear in the final paper.

| Artifact | Path |
| --- | --- |
| Final COLMAP model | `data/processed/pot1-unglazed_side_underside_mid25/colmap_side_underside_mid25/bundle_adjusted_final` |
| Reconstruction report | `data/processed/pot1-unglazed_side_underside_mid25/colmap_side_underside_mid25/logs/final_bundle_adjustment_report.json` |
| Accepted checkpoint | `data/processed/pot1-unglazed_side_underside_mid25/gaussian_splatting/runs/baseline_side_under_mid25_7k/checkpoints/step_007000.pt` |
| Evaluation summary | `data/processed/pot1-unglazed_side_underside_mid25/gaussian_splatting/runs/baseline_side_under_mid25_7k/evaluation/holdout_black/evaluation_summary.json` |
| PLY export | `data/processed/pot1-unglazed_side_underside_mid25/gaussian_splatting/runs/baseline_side_under_mid25_7k/exports/thai_ceramics_baseline_side_under_mid25_7k.ply` |
| SPLAT export | `data/processed/pot1-unglazed_side_underside_mid25/gaussian_splatting/runs/baseline_side_under_mid25_7k/exports/thai_ceramics_baseline_side_under_mid25_7k.splat` |

## Final Consistency Checks

1. Use **716 images** throughout the accepted Pot 1 discussion.
2. Use **side + intermediate 25-degree + underside** for the accepted capture groups.
3. Do not include top-45-degree images in the accepted dataset.
4. Use **626 training images and 90 held-out images**.
5. Use **130,577 sparse points** and **0.996169 px** mean reprojection error.
6. Use **7,000 steps and 500,000 Gaussians** for the accepted model.
7. Use the new overall metrics: PSNR 31.8255 dB, SSIM 0.95250, foreground PSNR 28.5096 dB, and alpha IoU 0.95555.
8. Label the 728-image top-45-degree model as a previous experiment if it remains in the report.
9. State that the old and new evaluation sets differ when their values appear in the same table.
10. Keep the side-only classical comparison and the separate lid reconstruction unchanged.
11. Assign final table and figure numbers only after placing all content in Word.
12. Do not add SHA-256 columns to report tables.
