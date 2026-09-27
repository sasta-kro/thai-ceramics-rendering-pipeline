# Week 14 Complete Results and Analysis Replacement Copy-Paste Text

Replace the complete existing **IV. Results and Analysis** section with the
material below. This version removes outdated future-tense progress statements
and presents the latest accepted Pot 1 and lid results.

## Copy-Paste Blocks

## IV. Results and Analysis

### A. Current Progress and Preliminary Analysis

```text
The project has progressed from image capture and preprocessing to completed reconstruction experiments for the first unglazed pot and its separate lid. The work began with frame-sampling and mask comparisons, followed by a side-only COLMAP reconstruction, a side-only 3D Gaussian Splatting baseline, and a classical textured-mesh reconstruction. Additional camera angles were then tested to improve areas that were not visible in the original side orbit.
```

```text
The first multiview Pot 1 experiment used side, top-45-degree, and underside images. It reconstructed the main object but produced uneven structure around part of the upper rim. The elevated sequence was therefore replaced with a new orbit recorded at approximately 25 degrees. The accepted Pot 1 dataset now contains 716 images from side, intermediate-25-degree, and underside views.
```

```text
The lid was reconstructed separately because it is an independently movable part. Its exterior model is complete and visually strong. Its underside could not be connected to the exterior reconstruction because the two capture groups did not contain enough shared visual information.
```

### B. Frame Sampling and Foreground Masking

```text
The original side-view video of Pot 1 was tested at several frame intervals. The every-third-frame sequence contained 545 images, while the every-sixth-frame sequence contained 273 images. Their masks agreed with a mean intersection over union of 0.9984 at shared viewpoints. The smaller every-sixth-frame dataset was therefore selected because it retained strong view overlap while reducing processing time and storage.
```

```text
The selected masks were temporally stable, with a mean adjacent-frame intersection over union of 0.997814. Manual inspection confirmed that the rim, body, decoration, and base remained inside the foreground masks. Separate SAM 2 propagation was later performed for the intermediate-angle, underside, and lid sequences. Each sequence was validated before reconstruction to confirm a one-to-one correspondence between RGB images and binary masks.
```

| Dataset | Images | Main result |
| --- | ---: | --- |
| Every-third-frame Pot 1 side sequence | 545 | Denser but highly redundant |
| Every-sixth-frame Pot 1 side sequence | 273 | Selected side-view dataset |
| Mean mask IoU at shared viewpoints | — | 0.9984 |
| Mean adjacent-mask IoU for selected sequence | — | 0.997814 |

```text
**TABLE X.** Frame-sampling and foreground-mask results for the original Pot 1 side-view sequence.
```

### C. Side-Only Sparse Reconstruction

```text
The selected 273-image side sequence was reconstructed with both exhaustive and sequential COLMAP matching. Exhaustive matching required approximately 9 hours 31 minutes and produced 11 disconnected models. Its largest model registered only 126 images. Sequential matching completed in approximately 22 minutes, registered all 273 images in one closed camera orbit, and produced 28,491 sparse points with a mean reprojection error of 0.906756 pixels.
```

```text
Sequential matching was better suited to the ordered turntable video because neighboring frames had strong visual overlap. Complete camera coverage was more useful for Gaussian Splatting than the much larger number of unrelated image pairs tested by exhaustive matching. The sequential model was therefore selected for both the side-only Gaussian baseline and the classical reconstruction branch.
```

| Measurement | Exhaustive matching | Sequential matching |
| --- | ---: | ---: |
| Runtime | Approximately 9 h 31 min | Approximately 22 min |
| Number of reconstructed models | 11 | 1 |
| Images in selected model | 126 | 273 |
| Sparse points | 6,244 | 28,491 |
| Mean reprojection error | Higher than selected model | 0.906756 px |

```text
**TABLE X.** Comparison of exhaustive and sequential matching for the original Pot 1 side-view sequence.
```

### D. Side-Only 3D Gaussian Splatting Baseline

```text
A 300-step smoke test first verified the complete training pipeline. The full side-only baseline then trained for 7,000 steps using 238 training images and 35 held-out images. Markov Chain Monte Carlo refinement increased the model from 28,491 initialization points to 500,000 Gaussians. Training took approximately 10.04 minutes and used about 0.641 GB of peak GPU memory.
```

```text
The side-only model reproduced the exterior body, carved decoration, rim, silhouette, color, and foot throughout the captured orbit. Evaluation produced a mean PSNR of 35.364 dB, a mean SSIM of 0.97779, a foreground PSNR of 30.856 dB, and an alpha IoU of 0.97833. These measurements show strong interpolation within the captured side orbit. They do not represent overhead or underside viewpoints, which were absent from the training images.
```

| Side-only measurement | Result |
| --- | ---: |
| Training images | 238 |
| Held-out images | 35 |
| Training steps | 7,000 |
| Final Gaussians | 500,000 |
| Mean PSNR | 35.364 dB |
| Mean SSIM | 0.97779 |
| Foreground PSNR | 30.856 dB |
| Alpha IoU | 0.97833 |

```text
**TABLE X.** Training and held-out evaluation of the side-only Pot 1 Gaussian Splatting baseline.
```

### E. Classical Dense Reconstruction and Meshing

```text
The same 273 side images and selected COLMAP camera model were used for the classical branch. PatchMatch stereo and fusion produced 950,472 oriented points. The accepted trim-5 Poisson reconstruction contained 2,605,671 vertices and 5,208,278 faces. It was simplified to 260,723 vertices and 520,826 faces before texturing.
```

```text
Conservative cleanup removed 9,741 faces, or 1.8703 percent of the simplified mesh. The cleaned candidate contains 256,053 vertices and 511,085 faces. Its 8192 by 5020 texture atlas assigned source views to 94.75 percent of the faces, compared with 90.72 percent before cleanup. The result remains a candidate because texture coverage does not by itself prove that every reconstructed surface is geometrically accurate.
```

| Classical stage | Result |
| --- | --- |
| Stereo fusion | 950,472 oriented points |
| Trim-5 Poisson reconstruction | 2,605,671 vertices and 5,208,278 faces |
| Simplified mesh | 260,723 vertices and 520,826 faces |
| Cleaned candidate | 256,053 vertices and 511,085 faces |
| Removed geometry | 9,741 faces, or 1.8703% |
| Cleaned texture coverage | 94.75% of faces assigned to source views |
| Cleaned texture atlas | 8192 x 5020 pixels |

```text
**TABLE X.** Main outputs of the classical dense-reconstruction branch.
```

### F. Controlled Comparison of Classical Meshing and Side-Only 3DGS

```text
The cleaned classical candidate and side-only Gaussian baseline form the controlled comparison because both use the same 273 images, masks, and camera poses. The mesh provides an explicit surface suitable for conventional editing, measurement, and collision geometry after validation. The Gaussian model does not provide a triangle surface but produces smooth interactive appearance within the captured camera orbit.
```

```text
The side-only Gaussian baseline has measured held-out rendering metrics. An equivalent held-out rendering evaluation has not yet been completed for the classical mesh, so the Gaussian metrics cannot be used to claim quantitative superiority over the mesh. Their visual comparison should use matched camera views and examine silhouette, carved detail, texture seams, base artifacts, and behavior near the edge of the captured orbit.
```

| Criterion | Classical candidate | Side-only 3DGS baseline |
| --- | --- | --- |
| Input | Same 273 side images | Same 273 side images |
| Representation | Textured triangle mesh | 500,000 Gaussians |
| Main advantage | Explicit editable surface | Smooth interactive appearance |
| Held-out image metrics | Not yet measured | PSNR 35.364 dB, SSIM 0.97779 |
| Main limitation | Base and texture artifacts require review | Interior and underside are unsupported |

```text
**TABLE X.** Controlled comparison using the same Pot 1 side-view dataset.
```

### G. Previous Multiview Experiment

```text
The first Pot 1 multiview experiment combined 273 side images, 233 top-45-degree images, and 222 underside images. LightGlue was required to connect the side and elevated sequences after standard SIFT failed to produce verified bridge pairs. The repaired model registered all 728 images and was trained for 15,000 steps with 750,000 Gaussians.
```

```text
Evaluation on 91 held-out images produced a mean PSNR of 26.9441 dB, a mean SSIM of 0.93190, a foreground PSNR of 26.1655 dB, and an alpha IoU of 0.92645. The result added the visible interior and underside, but free-viewpoint inspection showed uneven Gaussian structure around part of the upper rim. This indicated that the change between the side and top-45-degree orbits did not provide enough gradual overlap. The model is retained as a previous experiment rather than the accepted Pot 1 result.
```

### H. Accepted Three-Angle Pot 1 Reconstruction

```text
The replacement dataset uses 273 side images, 221 images recorded at approximately 25 degrees, and 222 underside images. The new intermediate orbit shares more visible exterior and rim detail with the side sequence while still observing the opening and inner surface. The complete accepted dataset contains 716 images.
```

```text
Registration reused the connected 495-camera side-and-underside model. The first pass added 210 intermediate cameras, and a second pass recovered the remaining 11. The provisional points were then discarded and rebuilt through fresh triangulation. Final bundle adjustment produced 130,577 sparse points, 1,523,453 observations, a mean track length of 11.667, and a mean reprojection error of 0.996169 pixels. All 716 cameras remained registered.
```

| Accepted reconstruction measurement | Result |
| --- | ---: |
| Side images | 273 |
| Intermediate-25-degree images | 221 |
| Underside images | 222 |
| Total registered images | 716 |
| Sparse points | 130,577 |
| Observations | 1,523,453 |
| Mean track length | 11.667 |
| Mean reprojection error | 0.996169 px |

```text
**TABLE X.** Final sparse-reconstruction measurements for the accepted three-angle Pot 1 model.
```

### I. Accepted Three-Angle Pot 1 Gaussian Splatting Result

```text
The final sparse model initialized a 7,000-step Gaussian Splatting run. The dataset was divided into 626 training images and 90 held-out images. Training completed in approximately 14.4 minutes, reached 500,000 Gaussians, and ended with a loss of 0.0150615. Peak GPU memory use was approximately 0.641 GB.
```

```text
Across the 90 held-out views, the accepted model achieved a mean PSNR of 31.8255 dB, a mean SSIM of 0.95250, a foreground PSNR of 28.5096 dB, and an alpha IoU of 0.95555. The intermediate group achieved the highest foreground PSNR and alpha IoU, while the underside remained the most difficult group in full-frame PSNR and SSIM.
```

| Capture group | Test views | PSNR dB | SSIM | Foreground PSNR dB | Alpha IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Side | 34 | 33.5091 | 0.97034 | 28.2456 | 0.95201 |
| Intermediate 25-degree | 28 | 32.3601 | 0.95704 | 29.1554 | 0.96322 |
| Underside | 28 | 29.2466 | 0.92630 | 28.1845 | 0.95217 |
| **Overall** | **90** | **31.8255** | **0.95250** | **28.5096** | **0.95555** |

```text
**TABLE X.** Held-out evaluation of the accepted three-angle Pot 1 Gaussian Splatting model.
```

```text
The accepted model uses fewer images, steps, and Gaussians than the previous top-45-degree model while producing stronger recorded evaluation values and a better upper-rim transition during visual inspection. The two models use different elevated images and held-out sets, so their metrics are supporting evidence rather than a strict same-image benchmark. Selection of the new model is based on both quantitative evaluation and free-viewpoint inspection.
```

| Measurement | Previous top-45-degree model | Accepted intermediate model |
| --- | ---: | ---: |
| Total images | 728 | 716 |
| Training steps | 15,000 | 7,000 |
| Gaussians | 750,000 | 500,000 |
| Mean PSNR | 26.9441 dB | 31.8255 dB |
| Mean SSIM | 0.93190 | 0.95250 |
| Foreground PSNR | 26.1655 dB | 28.5096 dB |
| Alpha IoU | 0.92645 | 0.95555 |
| Status | Previous experiment | **Accepted result** |

```text
**TABLE X.** Recorded results of the previous and accepted Pot 1 multiview models. Their held-out image sets are different.
```

### J. Separate Lid Reconstruction Result

```text
The lid exterior was reconstructed from 214 side images. COLMAP registered all images in one closed camera orbit and produced 31,303 sparse initialization points. The 7,000-step Gaussian Splatting model reached 500,000 Gaussians and reproduced the knob, sloped dome, carved decoration, circular rim, surface color, and visible crack.
```

```text
Held-out evaluation produced a mean PSNR of 37.7711 dB, a mean SSIM of 0.98494, and an alpha IoU of 0.96782. The result is accepted for exterior viewing. A presentation-cleaned export removes 35,139 unsupported lower Gaussians while preserving the original checkpoint and complete export.
```

| Lid measurement | Result |
| --- | ---: |
| Registered exterior images | 214 |
| Sparse points | 31,303 |
| Training steps | 7,000 |
| Complete model Gaussians | 500,000 |
| Mean PSNR | 37.7711 dB |
| Mean SSIM | 0.98494 |
| Alpha IoU | 0.96782 |
| Gaussians removed from cleaned export | 35,139 |

```text
**TABLE X.** Sparse reconstruction, evaluation, and presentation-cleanup results for the separate lid model.
```

```text
The lid underside was also reconstructed independently from 202 images, but it could not be connected to the exterior model. Standard SIFT and targeted LightGlue matching produced no valid bridge correspondences between the two capture groups. Manual alignment was rejected because it would not provide verified shared geometry. The cleaned exterior export suppresses visible lower floaters but does not claim to reconstruct the physical underside.
```

### K. Implementation Problems and Solutions

```text
Several problems affected the reconstruction work. Exhaustive COLMAP matching was slow and fragmented the ordered side video, so it was replaced with sequential matching. The original top-45-degree Pot 1 sequence produced an uneven upper-rim transition, so a new intermediate-25-degree sequence was captured and registered. Fresh triangulation was used after registration to prevent provisional points from entering the final Gaussian initialization.
```

```text
The first Poisson trial used trim 10 and returned only 36 vertices and 48 faces despite completing without a process error. Output-size validation was added, and trim 5 produced the accepted detailed mesh. The lid exterior and underside formed two strong but disconnected models. Because targeted LightGlue also failed to find bridge evidence, the exterior was retained as the accepted lid model rather than forcing an unreliable merge.
```

```text
Low-memory training settings allowed all accepted Gaussian models to run on the GTX 1650. Packed rasterization, sparse gradients, factor-two images, and guarded smoke tests kept peak memory use well below the available 4 GB.
```

### L. Visual Quality, Limitations, and Next Work

```text
The accepted Pot 1 model provides the strongest current multiview result. It preserves the exterior body, carved decoration, rim, visible interior, foot, and underside across the three recorded camera elevations. The intermediate orbit improves continuity around the upper rim. The underside remains the most difficult group, and views far outside the recorded trajectories may still reveal unsupported Gaussian structure.
```

```text
The lid model is strong across its captured exterior orbit, but its physical underside remains unavailable. A future bridge capture should hold the lid near an edge-on angle so that the decorated exterior, rim, crack, and concave underside are visible together. This would provide the shared evidence required to register the two lid models safely.
```

```text
The glossy second pot remains a separate future reconstruction challenge because its moving highlights can reduce feature consistency and create strong view-dependent appearance. The accepted Pot 1 and lid outputs should therefore be presented as completed results for the first unglazed pottery object rather than as results for every selected object.
```

## Suggested Figure Captions

Use the next available figure numbers in the Word document.

```text
Fig. X. Comparison of exhaustive and sequential COLMAP reconstruction for the original 273-image Pot 1 side sequence.
```

```text
Fig. X. Side-only Pot 1 Gaussian Splatting baseline rendered within the captured camera orbit.
```

```text
Fig. X. Cleaned classical textured-mesh candidate reconstructed from the same 273 Pot 1 side images.
```

```text
Fig. X. Previous top-45-degree Pot 1 result showing uneven Gaussian structure around part of the upper rim.
```

```text
Fig. X. Final COLMAP reconstruction of the accepted Pot 1 dataset showing all 716 registered cameras and 130,577 sparse points.
```

```text
Fig. X. Accepted three-angle Pot 1 Gaussian Splatting model reconstructed from side, intermediate-25-degree, and underside images.
```

```text
Fig. X. External-viewer inspection of the accepted Pot 1 model showing the improved upper-rim transition.
```

```text
Fig. X. COLMAP sparse reconstruction of the separate lid exterior showing all 214 registered cameras.
```

```text
Fig. X. Accepted 7,000-step lid model showing the knob, dome, carved decoration, rim, and visible crack.
```

```text
Fig. X. Presentation-cleaned lid export after removing unsupported lower Gaussians. The cleanup does not reconstruct the physical underside.
```


## Final Consistency Check

- Treat the 273-image side-only Pot 1 result as the first baseline.
- Compare classical meshing and 3DGS only with the same 273 side images.
- Treat the 728-image top-45-degree model as a previous experiment.
- Treat the 716-image intermediate-25-degree model as the accepted Pot 1 result.
- Use 130,577 sparse points, 7,000 steps, and 500,000 Gaussians for the accepted Pot 1 model.
- Use 90 held-out images and the updated three-angle evaluation values.
- Present the lid as a separate 214-image exterior reconstruction.
- State that lid cleanup removes floaters but does not reconstruct the underside.
- Do not include SHA-256 columns in report tables.
