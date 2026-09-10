# Week 13 Multiview 3DGS Report Update Copy-Paste Text

This drafting guide updates `Computer Vision Project w12 Update.docx` with the
multiview reconstruction completed after the side-only 3DGS baseline. The text
is based on `docs/progress/current_status.md` as updated on September 9, 2026.

## Editing Map

Use the following placement plan in the Week 12 Word report.

| Report location | Action | Content to add |
| --- | --- | --- |
| Abstract | Replace the complete existing abstract | Current multiview objective, LightGlue registration, final COLMAP model, 15,000-step 3DGS result, and remaining rim limitation |
| Keywords | Replace the existing keyword line | Add SAM 2, SIFT-LightGlue, COLMAP, and 3D Gaussian Splatting |
| III.E Image Acquisition and Dataset Preparation | Append before the next subsection | Top-45-degree and underside captures, image counts, orientation correction, and staged multiview dataset |
| III.F Foreground Mask Generation and Quality Control | Append before the next subsection | Independent SAM 2 propagation for each capture group and validation of 728 masks |
| III.G Sparse Reconstruction and Camera-Pose Selection | Append before the next subsection | Initial split models, SIFT failure, targeted LightGlue matching, registration passes, pose repair, triangulation, and bundle adjustment |
| III.H Mask-Aware 3D Gaussian Splatting Preparation | Append before the next subsection | Three PINHOLE cameras, two image widths, 637/91 split, and multiview caches |
| III.I Low-Memory 3D Gaussian Splatting Training | Append before the next subsection | Multiview smoke, 7,000-step baseline, and selected 15,000-step quality run |
| III.J Evaluation Export and Interactive Viewing | Append before Results and Analysis | Ninety-one-view evaluation, verified PLY and SPLAT exports, and the role of SuperSplat inspection |
| III.K Classical Dense Reconstruction and Texturing | Add after subsection J | Side-only PatchMatch, fusion, Poisson meshing, simplification, cleanup, and texturing method |
| IV.A Current Progress and Preliminary Analysis | Append after the existing side-only update | A concise chronological summary of the classical branch and completed multiview stage |
| IV, after the current subsection H | Add new subsections I-O | Classical result, controlled side-only comparison, separate multiview results, limitations, and next experiment |
| III.E, after the new capture paragraphs | Add Table I | Capture-group composition, split, visible areas, and undistorted resolutions |
| IV.I, after the classical result | Add Table II | Classical dense, meshing, cleanup, and texturing results |
| IV.J, after the comparison discussion | Add Table III | Controlled comparison of classical meshing and side-only 3DGS using the same 273 images |
| IV.K, after the initial multiview reconstruction discussion | Add Table IV | Progression from the side baseline to the repaired 728-image model |
| IV.L, after the LightGlue discussion | Add Table V | Bridge-pair success, inlier quality, rotational coverage, and runtime |
| IV.M, after the multiview evaluation paragraph | Add Table VI | Per-capture-group held-out metrics |
| Throughout the new material | Add only available figures | Capture groups, split COLMAP models, LightGlue matches, repaired camera rings, final 3DGS views, and rim limitation |

Keep the Introduction, Literature Review, Methodology subsections A-D, and the
existing Week 12 side-only results unchanged. The new text should present the
side-only reconstruction as the version-1 baseline and the 728-image model as
the completed multiview improvement.

## Abstract

Replace the complete existing abstract with the following text.

```text
Abstract-This project investigates image-based three-dimensional reconstruction of pottery associated with Thai material culture for interactive AR or VR presentation. The original 273-image side-view dataset was processed through two output branches. Classical dense reconstruction produced a cleaned textured-mesh candidate with 256,053 vertices and 511,085 faces, while the side-only 3D Gaussian Splatting baseline produced 500,000 Gaussians. These outputs provide a controlled comparison between an explicit surface representation and a radiance-based representation using the same captured views. The team then recorded 233 top-45-degree images and 222 underside images to improve the opening, inner wall, foot, and bottom surface. Standard SIFT matching reconstructed the individual capture groups but produced no geometrically verified connection between the side and elevated sequences. Targeted SIFT-LightGlue matching established 240 verified side-to-top pairs and enabled all 728 cameras to be registered in one coordinate system. Camera-trajectory repair, fresh triangulation, and bundle adjustment produced a final COLMAP model with 164,756 sparse points and a mean reprojection error of 0.898626 pixels. A multiview 3D Gaussian Splatting model was trained for 15,000 steps with 750,000 Gaussians. Evaluation on 91 held-out views produced a mean PSNR of 26.9441 dB, a mean SSIM of 0.9319, and an alpha IoU of 0.92645. The multiview result improves interior and underside coverage, although mild unevenness remains around part of the upper rim because the side and elevated camera rings have limited common surface coverage.
```

## Keywords

Replace the existing keyword line with:

```text
Keywords-Thai arts and culture, image-based reconstruction, photogrammetry, classical meshing, PatchMatch stereo, SAM 2, SIFT-LightGlue, COLMAP, 3D Gaussian Splatting
```

## III Methodology

### E Image Acquisition and Dataset Preparation

Keep the existing Week 12 paragraphs describing the original side-view video
and every-sixth-frame selection. Append the following paragraphs before
subsection F.

```text
Free-viewpoint inspection of the side-only baseline showed that the captured orbit did not sufficiently observe the deep opening, inner wall, foot, or exact underside. Additional optimization could not recover surfaces that were absent from the registered training images. Two new turn-around videos were therefore recorded for the first unglazed pot. The first used an approximately 45-degree elevated camera angle to observe the rim, opening, upper decoration, and interior. The second recorded the underside while the pot was inverted. The underside frames were rotated by 180 degrees after extraction so that their displayed orientation was consistent during annotation and inspection.

The selected multiview dataset contains 273 original side images, 233 top-45-degree images, and 222 underside images, for a total of 728 RGB frames at 2160 by 3840 pixels. Each video remained in a separate ordered folder during extraction and segmentation because SAM 2 propagation assumes a continuous image sequence. A dedicated staging program later combined the three groups using the prefixes side, top45, and underside. It also wrote a manifest containing the source group and sequence index for every staged image. Validation confirmed 728 RGB images, 728 corresponding masks, and no missing or extra files.

The lid and broken lower support were not added to this reconstruction. They move independently from the main pot and must therefore be captured, registered, trained, and exported as separate objects. Their final placement can be adjusted later inside the AR or VR scene or another compatible 3D editor.
```

Insert Table I immediately after the preceding capture paragraphs. The table
is complete and can be copied as one unit.

| Capture group | Images | Training | Held-out | Main visible area | Undistorted resolution |
| --- | ---: | ---: | ---: | --- | --- |
| Side | 273 | 238 | 35 | Exterior body and ornament | 1125 x 2000 |
| Top-45-degree | 233 | 204 | 29 | Rim, opening, and interior | 1125 x 2000 |
| Underside | 222 | 195 | 27 | Foot and bottom | 1116 x 2000 |
| Total | 728 | 637 | 91 | Complete multiview coverage | Mixed |

Suggested table caption:

```text
TABLE I. Composition and purpose of the three capture groups used for multiview reconstruction.
```

### F Foreground Mask Generation and Quality Control

Keep the existing side-view masking description. Append the following text
before subsection G.

```text
SAM 2 mask propagation was performed independently for the two new capture sequences. For the top-45-degree sequence, positive prompts were placed on the pottery body and visible inner wall, while negative prompts were placed on the turntable and background. For the underside sequence, positive prompts covered the lower body and bottom surface, while negative prompts excluded the white turntable, blue background, and visible cable. The opening and visible interior were intentionally retained as part of the pottery object.

The original 273 side masks were reused without modification. Validation confirmed exact image-to-mask correspondence for 273 side views, 233 elevated views, and 222 underside views. All 728 masks had the same 2160 by 3840 dimensions as their source images. Visual inspection showed that the elevated masks retained the rim and interior, while the underside masks preserved the bottom surface and excluded the surrounding capture equipment.
```

### G Sparse Reconstruction and Camera-Pose Selection

Keep the existing paragraphs describing the successful 273-image sequential
COLMAP baseline. Append the following text before subsection H.

```text
For the multiview experiment, sequential matching was retained within each ordered capture group and explicit bridge pairs were added between groups. The first pair plan contained 10,560 within-sequence pairs, 675 loop-closure pairs, 2,688 side-to-top pairs, and 2,576 side-to-underside pairs. This produced 16,499 planned comparisons instead of the 264,628 comparisons required by exhaustive matching across all 728 images.

The initial reconstruction produced three disconnected models. A complete elevated model registered all 233 top-45-degree images and contained 58,481 points. A second model registered all 273 side images and all 222 underside images and contained 96,911 points. Database inspection found 31 verified side-to-underside pairs but zero verified side-to-top pairs. The two complete models could not be directly combined because they shared no registered image and therefore had independent scales and coordinate systems.

An isolated copy of the database and the side-plus-underside model was created for targeted SIFT-LightGlue matching. The existing SIFT keypoints and descriptors were reused, and only the 2,688 failed side-to-top pair records were cleared in the copied database. LightGlue processed the selected pairs in 17.77 minutes and produced 240 geometrically verified side-to-top pairs. Of these, 154 contained at least 30 inliers. The verified matches covered 49 of 56 selected side anchors and 46 of 48 selected elevated anchors across the full rotation.

The first registration pass kept the 495 side and underside cameras fixed and added 205 elevated cameras. After triangulation increased the model to 174,557 points, a second pass registered the remaining 28 elevated cameras. Although all 728 images were now connected, visual inspection revealed abnormal gaps in parts of the side and elevated camera trajectories. The smooth source trajectories were therefore aligned into the connected coordinate system using shared 3D landmarks. One camera calibration was assigned to each of the three capture groups, the inconsistent point cloud was discarded, and all points were triangulated again.

Conservative global bundle adjustment kept the three camera intrinsics fixed while refining camera poses and 3D points. It converged in 26 iterations, reducing the optimization cost from 0.596448 to 0.594971 pixels. The accepted COLMAP model contains all 728 registered images, three cameras, 164,756 sparse points, 1,717,485 observations, and a mean reprojection error of 0.898626 pixels. COLMAP GUI inspection confirmed one pottery object surrounded by three smooth camera rings.
```

### H Mask-Aware 3D Gaussian Splatting Preparation

Keep the existing explanation of premultiplied mask-aware resizing. Append the
following paragraphs before subsection I.

```text
The repaired multiview model was undistorted together with all 728 masks. The three original SIMPLE_RADIAL camera groups became three PINHOLE cameras. Side and elevated images were produced at 1125 by 2000 pixels, while underside images were produced at 1116 by 2000 pixels. Every undistorted image retained an aligned, binary, and nonempty mask.

The 3D Gaussian Splatting data loader was extended to support multiple PINHOLE cameras and mixed per-view image widths. Each image uses the intrinsic matrix associated with its capture group and rescales that matrix to the prepared resolution. A deterministic every-eighth-image holdout rule divided the complete dataset into 637 training images and 91 test images. Factor-four and factor-two preparation profiles were used for the guarded smoke and full-resolution experiments.
```

### I Low-Memory 3D Gaussian Splatting Training

Keep the existing side-only baseline description. Append the following text
before subsection J.

```text
The repaired COLMAP point cloud initialized the multiview experiment with 164,756 Gaussians. A 300-step smoke test first verified the three-camera dataset, mixed image widths, masks, rasterization, optimization, and checkpoint path. It completed with a final loss of 0.045632. A 7,000-step multiview baseline then reached 500,000 Gaussians with a final loss of 0.0167034.

The selected quality model was trained for 15,000 steps at the factor-two resolution with spherical-harmonic degree two, packed rasterization, sparse gradients, and Markov Chain Monte Carlo refinement. It reached the configured limit of 750,000 Gaussians and a final loss of 0.0125201. Training required approximately 44.6 minutes and used 0.87 GB of peak GPU memory, remaining within the 4 GB capacity of the NVIDIA GeForce GTX 1650.
```

### J Evaluation Export and Interactive Viewing

Keep the existing explanation of the evaluation metrics and export formats.
Append the following text before section IV.

```text
The 15,000-step multiview checkpoint was evaluated on all 91 held-out images. The complete test set produced a mean PSNR of 26.9441 dB, a mean SSIM of 0.9319, a foreground PSNR of 26.1655 dB, a foreground mean absolute error of 0.02352, and an alpha IoU of 0.92645. Metrics were also grouped by capture sequence to distinguish side, elevated, and underside performance.

The final checkpoint contains 750,000 Gaussians and was exported as a 114,000,952-byte PLY file and a 24,000,000-byte SPLAT file. The PLY header, SPLAT record count, and SHA-256 values were independently checked against the export manifest. The PLY was also loaded into the external SuperSplat editor for free-viewpoint inspection. Held-out CUDA renders were treated as the authoritative appearance evaluation because the lightweight local viewer displays only base spherical-harmonic color and can appear less stable than the full trained renderer.
```

### K Classical Dense Reconstruction and Texturing

Add this new methodology subsection after subsection J. It describes the
classical branch derived from the same 273-image side-view dataset as the
side-only 3DGS baseline.

```text
The original 273-image side-view dataset was also processed through a classical dense reconstruction branch using the same selected COLMAP camera model. The masked RGB images and aligned masks were undistorted before geometric PatchMatch stereo was run at a maximum image size of 1200 pixels. The depth and normal maps were fused into a dense point cloud containing 950,472 oriented points.

Poisson surface reconstruction was selected as the detailed mesh branch. A first trial using trim 10 returned only 36 vertices and 48 faces despite a successful process exit, so output-size validation was added and trim 5 was used for the accepted run. The resulting mesh contained 2,605,671 vertices and 5,208,278 faces. It was simplified to approximately 10 percent of the original face count before texture generation. A Delaunay mesh was preserved as a lighter geometric alternative but was not selected as the primary textured result.

The simplified Poisson mesh was textured from the calibrated source images and preserved as the raw classical baseline. A conservative cleanup stage removed duplicate faces, faces without nearby fused-cloud support, and small disconnected components. It did not smooth the carved ornament, fill holes, invent unseen surfaces, or apply an automatic base cut. The cleaned geometry was textured again into a separate output so that the raw mesh, cleaned candidate, removed geometry, and corresponding texture atlases remained independently reviewable.
```

## IV Results and Analysis

### A Current Progress and Preliminary Analysis

Append the following paragraphs after the current final paragraph in this
subsection. Do not delete the existing Week 12 side-only progress paragraphs.

```text
In parallel with the multiview work, the team completed the classical dense-reconstruction branch for the original 273-image side-view dataset. PatchMatch stereo and fusion produced a 950,472-point dense cloud. The selected Poisson surface was simplified, conservatively cleaned, and retextured. The cleaned classical candidate contains 256,053 vertices and 511,085 faces, with source views assigned to 94.75 percent of its faces. It remains labeled as a candidate until its final matched-view MeshLab inspection is complete.

Following the successful side-only baseline, the team completed a multiview reconstruction of the first unglazed pot. Two additional rotations were captured and masked: 233 images from an elevated top-45-degree angle and 222 images of the underside. Together with the original 273 side images, the combined dataset contains 728 registered views of the main pottery body.

Normal SIFT matching successfully connected the side and underside sequences but could not verify any bridge between the side and elevated views. Targeted SIFT-LightGlue matching solved this connection problem by producing 240 verified side-to-top pairs. Two registration and triangulation passes then registered every elevated image into the side-plus-underside coordinate system.

The first fully registered model still contained visible discontinuities in two camera trajectories, so it was not used directly for Gaussian Splatting. Robust trajectory alignment, separate calibration for the three capture groups, fresh triangulation, and global bundle adjustment produced the accepted 728-image COLMAP model. The final sparse reconstruction contains 164,756 points and has a mean reprojection error of 0.898626 pixels.

The repaired model was used to train a 15,000-step multiview Gaussian Splatting model containing 750,000 Gaussians. The exported PLY and SPLAT files reproduce the exterior, opening, interior, foot, and underside across a much wider viewing range than the side-only baseline. A remaining uneven region around part of the upper rim was visible in genuinely novel views. This limitation is attributed mainly to the abrupt change between the side and top-45-degree capture elevations rather than insufficient training iterations.
```

### I Classical Meshing Results

Add this new subsection after the existing subsection H. The classical result
must remain labeled as a cleaned candidate until the matched-view MeshLab review
is complete.

```text
The classical branch completed every planned dense reconstruction stage for the original 273-image side-view dataset. Geometric PatchMatch stereo was the dominant compute stage and required approximately 2 hours 14 minutes 43 seconds. Stereo fusion produced 950,472 oriented points, and the accepted trim-5 Poisson reconstruction converted them into a detailed surface with 2,605,671 vertices and 5,208,278 faces. The alternative Delaunay reconstruction contained 40,719 vertices and 81,604 faces but retained substantially less surface detail.

The Poisson mesh was simplified to 260,723 vertices and 520,826 faces before texturing. Conservative cleanup removed 9,741 faces, or 1.8703 percent of the simplified mesh, leaving 256,053 vertices and 511,085 faces. The cleanup included 302 duplicate faces, 6,876 faces without nearby fused-cloud support, and 2,563 faces belonging to small disconnected components. No nonfinite or degenerate faces were detected.

The raw texture stage assigned source views to 472,517 of 520,826 faces, or 90.72 percent, and produced an 8192 by 4950 texture atlas. Retexturing the cleaned candidate assigned source views to 484,228 of 511,085 faces, or 94.75 percent, and produced an 8192 by 5020 atlas. The increased assignment percentage indicates improved texture coverage, but it does not by itself prove surface accuracy. The raw baseline, cleaned candidate, and removed-geometry audit remain preserved for visual comparison.
```

Insert Table II after the preceding paragraphs. The recorded durations exclude
manual work and unsuccessful retries and should not be presented as a complete
end-to-end runtime.

| Classical stage | Result | Recorded duration |
| --- | --- | ---: |
| RGB undistortion | Dense calibrated image workspace | 5 min 39 s |
| PatchMatch stereo | Geometric depth and normal maps | 2 h 14 min 43 s |
| Stereo fusion | 950,472-point dense cloud | 3 min 2 s |
| Poisson mesh with trim 5 | 2,605,671 vertices and 5,208,278 faces | 75.366 s |
| Delaunay alternative | 40,719 vertices and 81,604 faces | 10 min 16 s |
| Poisson simplification | 260,723 vertices and 520,826 faces | 63.564 s |
| Raw texturing | 90.72% of faces assigned to source views | 115.421 s |
| Geometry cleanup | 256,053 vertices and 511,085 faces | 5.178 s |
| Cleaned-mesh texturing | 94.75% of faces assigned to source views | Approximately 106.315 s |

Suggested table caption:

```text
TABLE II. Recorded outputs and processing durations of the side-view classical dense-reconstruction pipeline.
```

### J Controlled Comparison of Classical Meshing and Side-Only 3DGS

```text
The cleaned classical candidate was compared only with the side-only 3D Gaussian Splatting baseline for the controlled method comparison. Both outputs use the same 273 selected side images, masks, and COLMAP camera reconstruction. The newer 728-image multiview Gaussian model was excluded from this comparison because its additional elevated and underside observations would confound the effect of the reconstruction representation with the effect of camera coverage.

The classical result represents the pottery as an explicit triangle surface with a projected texture atlas. This form is compatible with conventional mesh editing, collision geometry, and measurement workflows after geometric accuracy has been validated. The Gaussian result represents the pottery with 500,000 anisotropic primitives and degree-two spherical-harmonic appearance. It does not provide an explicit surface but produces smooth interactive rendering within the captured side-view orbit.

The side-only Gaussian baseline has measured held-out rendering results, including a PSNR of 35.364 dB, an SSIM of 0.97779, and an alpha IoU of 0.97833. Equivalent held-out image metrics have not yet been calculated for the classical renderer, so these values must not be used to claim that Gaussian Splatting quantitatively outperforms the mesh. The controlled comparison should instead use matched camera viewpoints, equal framing, and consistent backgrounds to examine silhouette, carved detail, texture seams, base artifacts, and behavior near the limits of the captured side orbit.
```

Insert Table III after the controlled-comparison discussion.

| Criterion | Cleaned classical candidate | Side-only 3DGS baseline |
| --- | --- | --- |
| Input images | 273 side views | 273 side views |
| Camera reconstruction | Same selected COLMAP model | Same selected COLMAP model |
| Representation | Triangle mesh with texture atlas | 500,000 anisotropic Gaussians |
| Geometry | 256,053 vertices and 511,085 faces | No explicit triangle surface |
| Appearance | 8192 x 5020 projected texture atlas | Degree-two spherical harmonics |
| Appearance coverage | Source views assigned to 94.75% of faces | Appearance optimized directly from training images |
| Main compute stage | PatchMatch stereo took approximately 2 h 14 min 43 s | 7,000-step training took approximately 10.04 min |
| Held-out evaluation | Not yet measured | PSNR 35.364 dB, SSIM 0.97779, alpha IoU 0.97833 |
| Main strength | Explicit editable surface for conventional 3D workflows | Smooth real-time appearance within the captured orbit |
| Main limitation | Base artifacts, dark speckles, and uncertain unseen surfaces require review | Interior and underside remain weak outside the side orbit |
| Current status | Cleaned candidate awaiting matched-view visual acceptance | Accepted side-only baseline |

Suggested table caption:

```text
TABLE III. Controlled comparison between classical meshing and 3D Gaussian Splatting using the same 273-image side-view dataset.
```

### K Multiview Capture and Initial Reconstruction Results

Present the 728-image model as a separate camera-coverage experiment, not as a
direct method comparison with the side-only classical candidate.

```text
The multiview experiment is reported separately from the controlled classical-mesh comparison because it uses 728 images from three camera elevations rather than the original 273-image side-view dataset. Its purpose is to evaluate the improvement obtained from additional camera coverage, not to measure the difference between meshing and Gaussian Splatting under identical inputs.

The new capture strategy increased the dataset from one side orbit with 273 images to three elevation groups with 728 images. The top-45-degree sequence supplied direct observations of the opening and inner wall, while the underside sequence supplied observations of the foot and bottom surface that were absent from the first model. Independent masking and later manifest-based staging preserved the temporal order of each video and prevented filename collisions.

The initial multisequence COLMAP result demonstrated that strong reconstruction within each capture group did not guarantee a connected multiview model. All 233 elevated images formed a detailed top reconstruction, and all 495 side and underside images formed a second coherent reconstruction. However, zero side-to-top pairs passed geometric verification with standard SIFT. This showed that the failure was located specifically at the transition in camera elevation rather than within either image sequence.
```

Insert Table IV immediately after the preceding discussion. The LightGlue
registered row describes the connected model before camera-pose repair.

| Reconstruction stage | Registered images | Sparse points | Mean reprojection error |
| --- | ---: | ---: | ---: |
| Original side baseline | 273 | 28,491 | 0.906756 px |
| Initial elevated model | 233 | 58,481 | 0.797265 px |
| Initial side and underside model | 495 | 96,911 | 0.902436 px |
| LightGlue registered model | 728 | 174,557 | 1.189781 px |
| Repaired and bundle-adjusted model | 728 | 164,756 | 0.898626 px |

Suggested table caption:

```text
TABLE IV. Progression from separate capture-group reconstructions to the accepted repaired 728-image COLMAP model.
```

### L LightGlue Registration and Camera-Pose Repair

```text
SIFT-LightGlue was applied only to 2,688 selected side-to-top bridge pairs rather than all 63,609 possible combinations. It produced 240 verified pairs with a median of 41 inliers and a maximum of 232 inliers. The matches were distributed across all four rotation quartiles instead of being concentrated around a single similar viewpoint. This distribution was sufficient to register the elevated cameras directly into the existing side-plus-underside coordinate system without forcing a merge between independent COLMAP models.

Numerical image registration alone was not accepted as proof of geometric quality. COLMAP GUI inspection exposed side and top trajectory steps approximately 26 and 11.8 times their respective sequence medians. After robust alignment and reconstruction repair, the maximum-to-median adjacent spacing ratios fell to approximately 1.11-1.15 for all three groups. Fresh triangulation and bundle adjustment then produced a coherent model with all 728 images. This result emphasizes that camera-path continuity and visual inspection are necessary checks in addition to image count and reprojection error.
```

Insert Table V immediately after the LightGlue and repair paragraphs.

| LightGlue bridge measurement | Result |
| --- | ---: |
| All possible side-to-top pairs | 63,609 |
| Targeted pairs processed | 2,688 |
| Geometrically verified pairs | 240 |
| Pairs with at least 30 inliers | 154 |
| Median verified inliers | 41 |
| Mean verified inliers | 60.45 |
| Maximum verified inliers | 232 |
| Side anchors covered | 49 of 56 |
| Top-45-degree anchors covered | 46 of 48 |
| Matching time | 17.77 minutes |

Suggested table caption:

```text
TABLE V. Quality and rotational coverage of the targeted SIFT-LightGlue side-to-top correspondences.
```

### M Multiview Gaussian Splatting Results

```text
The multiview smoke, baseline, and quality runs completed successfully. Increasing the quality run from 7,000 to 15,000 steps increased the representation from 500,000 to 750,000 Gaussians and reduced the final training loss from 0.0167034 to 0.0125201. The quality run improved held-out measurements for the side, top-45-degree, and underside groups and was therefore selected for export.

Across all 91 held-out views, the selected model achieved a mean PSNR of 26.9441 dB, a mean SSIM of 0.9319, a foreground PSNR of 26.1655 dB, and an alpha IoU of 0.92645. The side and underside groups produced similar full-frame PSNR values of approximately 27.97 dB. The elevated group produced the lowest PSNR at 24.7372 dB, indicating that the rim, opening, inner wall, and transition from the side surface remain the most difficult parts of the reconstruction.
```

Insert Table VI immediately after the preceding paragraph. This table is also
complete and can be copied as one unit.

| Capture group | Held-out views | PSNR dB | SSIM | Foreground PSNR dB | Alpha IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Side | 35 | 27.9743 | 0.95067 | 26.1531 | 0.91912 |
| Top-45-degree | 29 | 24.7372 | 0.91519 | 23.9453 | 0.92782 |
| Underside | 27 | 27.9790 | 0.92552 | 28.5661 | 0.93447 |
| Overall | 91 | 26.9441 | 0.93190 | 26.1655 | 0.92645 |

Suggested table caption:

```text
TABLE VI. Held-out evaluation of the 15,000-step multiview Gaussian Splatting model by capture group.
```

### N Visual Quality and Remaining Limitation

```text
Free-viewpoint inspection showed a substantial improvement over the side-only baseline. The final model preserves the pot body, carved decorative band, opening, visible interior, foot, and bottom surface while the virtual camera moves between the three recorded elevation ranges. The PLY file also loaded successfully in SuperSplat, confirming that the standard export can be inspected outside the training code.

The main remaining defect is mild unevenness or floating Gaussian structure around part of the upper rim when the model is viewed from angles not closely represented by the training cameras. Increasing the training duration from 7,000 to 15,000 steps improved the numerical results but did not eliminate this defect. The evidence therefore suggests a camera-coverage limitation: the side orbit and the top-45-degree orbit do not share enough gradually changing views of the same rim surface. The artifact should not be described as a complete reconstruction failure because the pot remains coherent and the defect is localized, but it should be reported as a limitation of unrestricted novel-view quality.
```

### O Next Experiment

```text
The next capture should add a complete intermediate ring at approximately 20-30 degrees above the original side orbit. This sequence should maintain substantial overlap with both the side and top-45-degree images so that the registration chain becomes side to intermediate to top. The intermediate views must be masked, staged, matched, registered, triangulated, and bundle-adjusted with the existing camera groups before the 3D Gaussian Splatting model is retrained. Additional optimization of the current images alone is not expected to reconstruct missing transition geometry reliably.

The lid and repaired lower support should remain separate reconstruction targets because they are independent movable objects. Each component can be trained and exported separately, then positioned relative to the main pot in the final AR or VR scene. The glossy second pot also remains a separate future reconstruction because its moving highlights create a different feature-matching and appearance-modeling challenge.
```

## Implementation Problems and Solutions Placement

The Week 12 report already contains an Implementation Problems and Solutions
subsection. Append only the following academically relevant problems there.
The remaining operational issues, such as Qt plugin discovery, silent command
output, or a missing Python package, can stay in the repository progress log or
an appendix if the report has a strict page limit.

```text
The first Poisson trial used trim 10 and exited normally, but its output contained only 36 vertices and 48 faces. Process completion was therefore not treated as proof of a valid mesh. The pipeline added output-size validation, preserved the failed diagnostic mesh, and reran Poisson reconstruction with trim 5 to obtain 2,605,671 vertices and 5,208,278 faces.

The simplified classical mesh retained a jagged base skirt, small detached components, and dark surface speckles. A conservative geometry cleanup removed only 1.8703 percent of the faces and preserved the original raw mesh for comparison. The cleaned candidate was retextured separately because applying the raw texture atlas to changed geometry could create invalid texture coordinates. Dark speckles are still treated cautiously because they can originate from texture projection or shading rather than incorrect geometry.

The principal reconstruction problem was the absence of geometrically verified side-to-top correspondences under standard SIFT matching. Rather than concatenating two independent sparse models, the team preserved the stronger side-plus-underside model and used targeted SIFT-LightGlue matching to register the elevated cameras directly into its coordinate system. This avoided an invalid forced merge and retained the original reconstruction as an unchanged backup.

A second problem appeared after all 728 images registered: parts of the camera paths contained large discontinuities even though the image count was complete. The team treated GUI trajectory inspection as a required validation step. Smooth source trajectories were aligned using shared landmarks, each capture group received its own calibration, and the points were freshly triangulated and bundle-adjusted. This reduced the abnormal adjacent-camera spacing and produced one coherent model.

The final repaired dataset contained three PINHOLE cameras and two undistorted image widths, while the original Gaussian Splatting loader assumed one camera and one resolution. The loader and preparation checks were generalized so that every image uses its own camera intrinsics and prepared dimensions. This allowed all three capture groups to be trained and evaluated together without forcing an inaccurate shared calibration.
```

## Suggested New Figure Captions

The current Week 12 document already reaches Fig. 18. Start the next available
figure at Fig. 19, but renumber these captions if the Word document changes.
Insert only figures that are available and readable in the two-column layout.

```text
Fig. 19. Representative side, top-45-degree, and underside frames from the 728-image multiview dataset for the first unglazed pot.

Fig. 20. Raw classical textured mesh and conservatively cleaned classical candidate reconstructed from the original 273-image side-view dataset.

Fig. 21. Same-input comparison between the cleaned classical textured-mesh candidate and the side-only 3D Gaussian Splatting baseline using matched camera viewpoints.

Fig. 22. Initial multisequence COLMAP result showing the separate complete elevated model and the connected side-plus-underside model before LightGlue registration.

Fig. 23. Example geometrically verified SIFT-LightGlue correspondences between a side image and a top-45-degree image.

Fig. 24. Accepted repaired COLMAP reconstruction containing all 728 registered images and three continuous elevation rings around one pottery object.

Fig. 25. Free-viewpoint renderings of the separate 15,000-step multiview Gaussian Splatting model showing the exterior, opening and inner wall, and underside.

Fig. 26. SuperSplat inspection of the multiview PLY model showing localized uneven Gaussian structure around part of the upper rim in a novel view.
```

Recommended placement:

- Place Fig. 19 in Methodology subsection E after the new capture description.
- Place Fig. 20 in Results subsection I after the classical cleanup result.
- Place Fig. 21 in Results subsection J after the controlled-comparison table.
- Place Fig. 22 in Results subsection K after the split-model explanation.
- Place Fig. 23 in Results subsection L after the LightGlue statistics.
- Place Fig. 24 after the repaired COLMAP result in subsection L.
- Place Fig. 25 after the quantitative results in subsection M.
- Place Fig. 26 in subsection N beside the multiview limitation discussion.

## References to Add

Add references for the following methods if they are not already present. Use
the final numbering assigned by Word after all references are ordered.

- Segment Anything Model 2 for video mask propagation.
- Poisson surface reconstruction for the classical mesh branch.
- LightGlue for learned local-feature matching.
- The original 3D Gaussian Splatting method.
- The gsplat implementation used for low-memory training and evaluation.

Do not cite COLMAP only as software documentation if the existing paper already
includes the Structure-from-Motion and Multi-View Stereo papers. Add only the
references that are actually cited in the new prose.

## Final Consistency Checks

Before submitting the updated Word report:

1. Change future-tense statements that claim masking or reconstruction is still under development.
2. Call the classical output a cleaned candidate until matched-view MeshLab inspection is complete.
3. Compare the cleaned classical candidate only with the side-only 3DGS baseline when making same-input method claims.
4. Keep the side-only 273-image and 7,000-step 3DGS results labeled as the version-1 baseline.
5. Present the 728-image and 15,000-step model as a separate multiview camera-coverage experiment.
6. Do not compare the side-only and multiview PSNR values as though their held-out image sets have equal difficulty.
7. Do not claim that the lid, lower support, or glossy second pot has been reconstructed.
8. Distinguish held-out interpolation metrics from unrestricted novel-view quality.
9. Report both the classical base and texture artifacts and the multiview 3DGS upper-rim artifact.
10. Include the planned 20-30-degree intermediate capture ring as the next multiview experiment.
11. Confirm that every table, figure number, and in-text reference matches the final Word layout.
12. Keep operational file paths, environment commands, hashes, and storage-cleanup details in the repository documentation unless the instructor specifically requests reproducibility details in the paper.
