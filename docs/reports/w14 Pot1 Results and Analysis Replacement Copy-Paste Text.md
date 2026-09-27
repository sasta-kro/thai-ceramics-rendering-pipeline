# Week 14 Pot 1 Results and Analysis Replacement Copy-Paste Text

Copy the following material into **IV Results and Analysis**. Replace the old
Pot 1 multiview result with this version. Keep the existing classical
reconstruction and side-only comparison sections unchanged.

## IV Results and Analysis

### A. Current Progress and Preliminary Analysis

```text
The first Pot 1 multiview experiment combined side, top-45-degree, and underside images. It reconstructed the complete object, but inspection from novel viewpoints showed uneven structure around part of the upper rim. This result indicated that the change in elevation between the side and top-45-degree camera orbits was too large.

To improve the transition, the top-45-degree sequence was replaced with a new orbit recorded at approximately 25 degrees. The accepted dataset contains 273 side images, 221 intermediate-25-degree images, and 222 underside images, giving 716 images in total. All images and masks passed dataset validation.

All 716 cameras were registered in one coordinate system. Fresh triangulation and final bundle adjustment produced 130,577 sparse points with a mean reprojection error of 0.996169 pixels. The model was then trained for 7,000 steps and reached 500,000 Gaussians.

Evaluation on 90 held-out images produced a mean PSNR of 31.8255 dB, a mean SSIM of 0.95250, a foreground PSNR of 28.5096 dB, and an alpha IoU of 0.95555. Visual inspection also showed a smoother and more coherent upper-rim transition. The side, intermediate-25-degree, and underside model is therefore selected as the current Pot 1 multiview result.
```

### B. Three-Angle Sparse Reconstruction Result

```text
The reconstruction reused the connected 495-camera side-and-underside model as its base. The first registration pass added 210 of the 221 intermediate cameras. A second pass recovered the remaining 11 cameras, producing complete registration of all 716 images.

The provisional points were discarded after registration and rebuilt through fresh triangulation. This produced 130,577 sparse points from 1,523,453 observations. The mean track length was 11.667 observations per point, and each registered image contained an average of 2,127.73 point observations. Final bundle adjustment reduced the mean reprojection error from 0.999984 to 0.996169 pixels while preserving the complete camera set.
```

| Reconstruction measurement | Result |
| --- | ---: |
| Side images | 273 |
| Intermediate-25-degree images | 221 |
| Underside images | 222 |
| Total registered images | 716 |
| Camera calibrations | 3 |
| Sparse points | 130,577 |
| Observations | 1,523,453 |
| Mean track length | 11.667 |
| Mean reprojection error | 0.996169 px |

Suggested caption:

```text
TABLE X. Final sparse-reconstruction results for the accepted three-angle Pot 1 dataset.
```

### C. Three-Angle 3D Gaussian Splatting Result

```text
The final sparse reconstruction initialized the Gaussian Splatting model with 130,577 points. Training used 626 images, while 90 images were reserved for evaluation. The model completed 7,000 training steps in approximately 14.4 minutes and reached the configured limit of 500,000 Gaussians. Its final training loss was 0.0150615.

The complete held-out set achieved a mean PSNR of 31.8255 dB and a mean SSIM of 0.95250. Foreground evaluation produced a PSNR of 28.5096 dB and an alpha IoU of 0.95555. The side views achieved the highest PSNR and SSIM. The intermediate views achieved the highest foreground PSNR and alpha IoU, supporting the use of the new camera angle around the rim and opening. The underside remained the most difficult group in full-frame PSNR and SSIM.
```

| Capture group | Test views | PSNR dB | SSIM | Foreground PSNR dB | Alpha IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Side | 34 | 33.5091 | 0.97034 | 28.2456 | 0.95201 |
| Intermediate 25-degree | 28 | 32.3601 | 0.95704 | 29.1554 | 0.96322 |
| Underside | 28 | 29.2466 | 0.92630 | 28.1845 | 0.95217 |
| **Overall** | **90** | **31.8255** | **0.95250** | **28.5096** | **0.95555** |

Suggested caption:

```text
TABLE X. Held-out evaluation of the accepted 7,000-step Pot 1 Gaussian Splatting model.
```

### D. Improvement over the Previous Multiview Result

```text
The previous model used a top-45-degree orbit and produced visible unevenness around part of the upper rim. The accepted model replaces this orbit with intermediate views recorded at approximately 25 degrees. These views share more visible surface with the side sequence and provide a more gradual transition toward the opening.

The accepted model uses fewer images, training steps, and Gaussians than the previous model while producing stronger held-out measurements and better visual continuity. Its mean PSNR increased from 26.9441 to 31.8255 dB, its SSIM increased from 0.93190 to 0.95250, and its alpha IoU increased from 0.92645 to 0.95555. Because the two models use different elevated images and different held-out sets, these measurements are supporting evidence rather than a same-image benchmark. Final selection was based on both evaluation results and visual inspection.
```

| Measurement | Previous top-45-degree model | Accepted intermediate-25-degree model |
| --- | ---: | ---: |
| Total images | 728 | 716 |
| Training steps | 15,000 | 7,000 |
| Gaussians | 750,000 | 500,000 |
| Mean PSNR | 26.9441 dB | 31.8255 dB |
| Mean SSIM | 0.93190 | 0.95250 |
| Foreground PSNR | 26.1655 dB | 28.5096 dB |
| Alpha IoU | 0.92645 | 0.95555 |
| Status | Previous experiment | **Accepted result** |

Suggested caption:

```text
TABLE X. Summary of the previous Pot 1 experiment and the accepted intermediate-angle result. The evaluation sets are different.
```

### E. Visual Quality and Limitation

```text
The accepted model reconstructs the exterior body, carved decoration, rim, visible interior, foot, and underside. The intermediate camera orbit improves continuity around the upper part of the pot and reduces the uneven appearance observed in the previous model.

The underside remains the most difficult region, and viewpoints far outside the recorded camera paths may still reveal unsupported Gaussian structure. These limitations are mainly related to camera coverage. The accepted model nevertheless provides the best current balance of visual quality, complete three-angle coverage, training cost, and held-out accuracy for Pot 1.
```

## Suggested Figure Captions

Use the next available figure numbers in the Word report.

```text
Fig. X. Final COLMAP reconstruction of Pot 1 showing the 716 registered cameras and the 130,577-point sparse model.

Fig. X. Accepted three-angle Pot 1 Gaussian Splatting model reconstructed from side, intermediate-25-degree, and underside images.

Fig. X. External-viewer inspection of the accepted Pot 1 model showing the improved upper-rim transition.

Fig. X. Comparison of the previous top-45-degree result and the accepted intermediate-25-degree Pot 1 model.
```

## Short Final Check

- Use **716 images**, not 728, for the accepted result.
- Use **intermediate 25-degree**, not top-45-degree, for the accepted elevated view.
- Use **130,577 sparse points** and **0.996169 px** reprojection error.
- Use **7,000 steps** and **500,000 Gaussians**.
- Use **90 held-out images** and the updated evaluation values.
- Label the top-45-degree model as the previous experiment.
- Do not add a SHA-256 column to any report table.
