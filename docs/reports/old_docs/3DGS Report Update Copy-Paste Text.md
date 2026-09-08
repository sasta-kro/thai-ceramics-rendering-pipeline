# 3DGS Report Update Copy-Paste Text

This drafting file contains only the proposed report additions and replacements.
Methodology subsections A-D should remain unchanged in the Word report.

## III. Methodology

### E. Image Acquisition and Dataset Preparation

```text
The team implemented a Python program that extracts individual frames from a 60 FPS video and saves them as a consistently named image sequence. This provides a repeatable image source for the reconstruction pipeline and avoids the need to capture every viewpoint as a separate photograph. Since adjacent video frames can be almost identical, different sampling intervals were tested to reduce redundant computation while retaining enough overlap for reliable camera-pose estimation. The unglazed pottery recording has a resolution of 2160 by 3840 pixels and a duration of approximately 27.2 seconds.

Two main sampling intervals were processed and compared for the first pot. Selecting every third frame produced 545 images, while selecting every sixth frame produced 273 images and an effective sampling rate of approximately 10 FPS. The every-sixth-frame sequence retained sufficient overlap between neighboring viewpoints, required approximately half the masking time and storage, and produced masks that agreed with the denser sequence by an average intersection over union of 0.9984. For these reasons, the 273-image every-sixth-frame dataset was selected for the reconstruction experiments described in this report.

The selected frames were stored without modifying their original RGB values. A frame manifest recorded the relationship between each extracted image, its original video-frame number, and its timestamp. Contact sheets were also generated for rapid visual inspection. Blurred, incorrectly exposed, or obstructed images can be removed during review, although all 273 selected images from the first pot were suitable for the completed camera reconstruction.
```

#### 3DGS Pipeline Diagram

Place this diagram after subsection E and before subsection F.

```text
Video
  ↓
Frame extraction
  ↓
SAM 2 segmentation
  ↓
Masked images and masks
  ↓
COLMAP reconstruction
  ↓
Undistorted images + camera poses + sparse 3D points
  ↓
3DGS image preparation
  ↓
3D Gaussian Splatting training
  ↓
Checkpoint, evaluation, export, and viewer
```

```text
Fig. 8. Overall processing pipeline used to reconstruct and evaluate the first unglazed Thai pottery object with 3D Gaussian Splatting.
```

### F. Foreground Mask Generation and Quality Control

```text
The pottery object was separated from the blue background and white turntable using the Segment Anything Model 2, or SAM 2. The first frame was manually annotated with a tight bounding box, a positive point inside the pot, and negative points on the turntable and background. SAM 2 then propagated the object mask through the ordered video sequence. The visible opening and interior were retained as part of the pottery object so that the reconstruction would preserve the rim and inner surface that could be observed from the captured camera angle.

The masking program produced a full-resolution object mask, a slightly eroded mask suitable for feature extraction, a transparent RGBA image, and a visual overlay for every selected frame. It also generated a frame-level manifest and a quality-control contact sheet. For the every-sixth-frame dataset, the average adjacent-frame mask intersection over union was 0.997814, the minimum adjacent-frame value was 0.950811, and the rotation-loop intersection over union was 0.872302. Visual inspection confirmed that the masks consistently followed the rim, body, carved ornament, and base while excluding the surrounding capture area.
```

### G. Sparse Reconstruction and Camera-Pose Selection

```text
COLMAP was used to estimate camera parameters and construct the sparse three-dimensional initialization required by the Gaussian Splatting stage. Feature extraction used the original selected RGB images together with the corresponding foreground masks. A SIMPLE_RADIAL camera model with one shared camera was used for the original frames. The feature-extraction settings were adjusted to remain stable on the available computer, including a maximum image dimension of 3200 pixels, a maximum of 8192 SIFT features per image, and one extraction worker.

Two matching strategies were evaluated. Exhaustive matching required approximately 9 hours and 31 minutes and divided the reconstruction into 11 disconnected models. Its largest model registered only 126 of the 273 images and contained 6244 sparse points. Sequential matching was then tested because the source data consisted of ordered video frames. With an overlap of 15 neighboring frames and guided matching enabled, sequential matching completed in approximately 22 minutes and registered all 273 images in one connected model. The selected model contained 28,491 sparse points, 319,683 observations, a mean track length of 11.220491, and a mean reprojection error of 0.906756 pixels.

The selected model was undistorted to a shared PINHOLE camera representation. This produced 273 registered masked images, 273 geometrically aligned masks, and an undistorted sparse model. The resulting images had a resolution of 1125 by 2000 pixels. Only the undistorted images, aligned masks, and sparse camera model were required as input to the 3D Gaussian Splatting pipeline.
```

### H. Mask-Aware 3D Gaussian Splatting Preparation

```text
A separate preparation stage was developed for 3D Gaussian Splatting. The stage verifies that each image has a corresponding mask and registered camera pose before creating reusable image caches. The masked images were resized together with soft alpha masks so that boundary pixels remained correctly aligned. Premultiplied color was used during resizing and was converted back to foreground color afterward. This prevented dark pixels from the original black background from contaminating the edge of the pot when random training backgrounds were applied.

Two prepared resolutions were created. The factor-four cache contained 273 images and masks at 281 by 500 pixels for a low-memory smoke test. The factor-two cache contained 273 images and masks at 562 by 1000 pixels for full training. A deterministic every-eighth-image holdout rule divided the dataset into 238 training views and 35 test views. This same split was retained throughout training and evaluation so that the final measurements could be reproduced.
```

### I. Low-Memory 3D Gaussian Splatting Training

```text
The Gaussian Splatting environment was configured for an NVIDIA GeForce GTX 1650 with 4 GB of video memory. It used Python 3.10.21, PyTorch 2.4.1 with CUDA 12.4, and gsplat 1.5.3. Packed rasterization and sparse gradients were enabled to reduce memory use, and the interactive viewer was disabled during training. The initial Gaussian positions were taken from the 28,491 points in the selected COLMAP sparse model.

A 300-step smoke test was performed first at the factor-four resolution. Densification was disabled during this short test so that the complete image loading, camera conversion, rendering, loss calculation, optimization, checkpoint, and preview path could be verified safely. After the smoke test completed successfully, the full baseline model was trained for 7000 steps using the factor-two images, spherical-harmonic degree two, and Markov Chain Monte Carlo refinement. The refinement stage relocated low-opacity Gaussians and added new Gaussians until the configured limit of 500,000 was reached.
```

### J. Evaluation, Export, and Interactive Viewing

```text
The final checkpoint was evaluated on all 35 held-out views at the factor-two resolution. Full-frame peak signal-to-noise ratio and structural similarity were used to measure image agreement. Foreground-only peak signal-to-noise ratio and mean absolute error were also calculated so that the large black background did not dominate the appearance measurements. Alpha intersection over union was used to measure agreement between the reconstructed and reference silhouettes. A side-by-side comparison image was saved for every held-out view.

The trained model was exported as both a standard PLY file and a compact SPLAT file. The PLY representation retained positions, opacity, scale, rotation, and the complete degree-two spherical-harmonic coefficients, while the compact SPLAT representation stored the base color required by compatible real-time viewers. An export manifest recorded file sizes and SHA-256 checksums. A local browser viewer was also implemented with Viser so that the reconstruction could be rotated, panned, zoomed, and inspected without uploading the model to an external service.
```

## IV. Results and Analysis

### A. Current Progress and Preliminary Analysis

Keep the existing Week 11 paragraphs under **Current Progress and Preliminary Analysis** unchanged. Append the following paragraphs after the current final paragraph of that material.

```text
Following the capture and preliminary feature-matching work, the team completed foreground masking for the first unglazed pot. The every-third-frame sequence contained 545 images and required approximately 37 minutes to process, while the every-sixth-frame sequence contained 273 images and required approximately 19 minutes. The masks from the two sequences agreed by an average intersection over union of 0.9984 across the shared viewpoints. Since the denser sequence did not provide a meaningful improvement in segmentation quality, the every-sixth-frame dataset was selected to reduce storage and later reconstruction time while maintaining substantial visual overlap.

The selected 273-image dataset was then processed with COLMAP. The initial exhaustive-matching experiment produced several disconnected models and registered only 126 images in its largest component. Replacing exhaustive matching with sequential matching reduced the runtime from approximately 9 hours and 31 minutes to approximately 22 minutes. More importantly, the sequential method registered all 273 images in one connected camera trajectory and produced 28,491 sparse points with a mean reprojection error of 0.906756 pixels. This result provided a reliable camera and geometry initialization for the first Gaussian Splatting experiment.

After image and mask undistortion, the team implemented a low-memory 3D Gaussian Splatting pipeline for the available GTX 1650 GPU. The pipeline includes dataset validation, mask-aware image preparation, Gaussian initialization, training, held-out evaluation, model export, and local interactive viewing. A 300-step smoke test confirmed that the complete training path worked before a larger experiment was attempted. The full baseline then completed 7000 training steps and reached the configured limit of 500,000 Gaussians without exceeding the available GPU memory.

The completed baseline reproduced the overall pot shape, silhouette, rim opening, carved decorative band, body color, and lower foot across the captured side-view orbit. Quantitative evaluation on 35 held-out views produced a mean full-frame peak signal-to-noise ratio of 35.364 dB, a mean structural similarity of 0.97779, a foreground peak signal-to-noise ratio of 30.856 dB, and an alpha intersection over union of 0.97833. The trained model was exported to PLY and SPLAT formats and was successfully loaded into a local interactive browser viewer.
```

### B. Frame Sampling and Masking Results

```text
The frame-sampling comparison showed that adding more nearly identical video frames did not automatically improve the useful information available to later stages. Although the every-third-frame dataset contained approximately twice as many images as the every-sixth-frame dataset, the two masking results were approximately 99.84 percent identical over their shared viewpoints. The every-sixth-frame sequence therefore provided a better balance between adjacent-view overlap, processing time, storage, and dataset size for this capture.

The masks were also temporally stable. The every-sixth-frame sequence achieved an average adjacent-frame intersection over union of 0.997814, showing that the predicted silhouette changed smoothly as the pot rotated. Only two quality-control records were flagged, and manual inspection confirmed that the first, middle, and final frames retained clean object boundaries. This was important because inconsistent masks could introduce false Gaussians around the background or remove thin parts of the rim and base.
```

### C. Sparse-Reconstruction Comparison

```text
Sequential matching was substantially more suitable than exhaustive matching for the ordered turn-around video. Exhaustive matching compared many image pairs that were visually unrelated or separated by large viewpoint changes. This required much more time and still produced 11 disconnected models. Sequential matching concentrated the comparisons on neighboring frames, where overlap was high, while guided matching strengthened geometrically consistent correspondences. As a result, it was approximately 26 times faster, registered every input image, produced more than four times as many sparse points in the selected model, and achieved a lower mean reprojection error.

The complete camera orbit was more important to the Gaussian Splatting stage than maximizing the number of candidate image pairs. Each registered camera supplied another observation of the pottery surface, while the sparse points supplied initial Gaussian positions. The sequential result therefore provided both broader view coverage and a stronger initialization than the largest component of the exhaustive result.
```

### D. Gaussian Splatting Training Results

```text
The 300-step smoke test completed in approximately 17 seconds. Its final loss was 0.020068, its final logged structural similarity was 0.953751, and its peak allocated GPU memory was approximately 0.075 GB. The model remained relatively soft because the smoke configuration used low-resolution images, a short optimization period, and no densification. However, it correctly reproduced the pot's orientation, scale, silhouette, opening, and main decorative band, confirming that the input cameras and training implementation were consistent.

The full baseline used 562 by 1000 pixel images and completed 7000 steps in approximately 10.04 minutes. The training loss decreased from 0.125020 to 0.007804, while the final logged structural similarity reached 0.977806. Markov Chain Monte Carlo refinement increased the model from 28,491 initial Gaussians to the configured limit of 500,000. Peak allocated GPU memory was approximately 0.641 GB, which remained safely below the GTX 1650's 4 GB capacity. These results demonstrate that the selected packed and sparse training settings provided a practical low-memory configuration for the available hardware.
```

### E. Held-Out Evaluation

```text
Evaluation on the 35 withheld images showed strong agreement within the camera range represented by the original side-view capture. The mean full-frame peak signal-to-noise ratio was 35.364 dB, and the mean structural similarity was 0.97779. Since the black background is comparatively easy to reproduce, foreground-only measurements were included as a stricter assessment of the pottery appearance. The foreground peak signal-to-noise ratio was 30.856 dB, while the foreground mean absolute error was 0.012923 on a zero-to-one color scale.

The mean alpha intersection over union was 0.97833, indicating close agreement between the rendered and reference silhouettes. Across the individual test views, full-frame peak signal-to-noise ratio ranged from 33.473 to 36.016 dB, and alpha intersection over union ranged from 0.96159 to 0.98235. Inspection of the lowest-scoring views did not reveal a catastrophic reconstruction failure. Instead, the differences were mainly associated with fine surface appearance and object boundaries.

These measurements should be interpreted as interpolation results rather than proof of unrestricted novel-view performance. The held-out frames came from the same ordered capture session and were positioned between nearby training cameras. The evaluation therefore demonstrates that the model can reproduce unseen views within the captured side orbit, but it does not demonstrate equal accuracy under different lighting, backgrounds, or camera elevations.
```

### F. Implementation Problems and Solutions

```text
Several technical problems were encountered while adapting the pipeline to the Windows computer and the installed library versions. An initial memory-monitoring call failed because the installed PyTorch build rejected an explicitly supplied CUDA device argument. The runner was corrected by selecting the current CUDA device first and then using the no-argument memory functions. A second error occurred because the installed gsplat rasterizer rejected the supplied batched background tensor shape. The final implementation rasterized without a background and then combined the rendered color with a random background using the returned alpha image. This preserved the intended training behavior and allowed both the smoke and full runs to complete successfully.

The long exhaustive COLMAP result also demonstrated that more pair comparisons do not necessarily produce a stronger reconstruction for an ordered video sequence. Changing the matching strategy was more effective than increasing computation. Together, these corrections show the importance of matching the software configuration to the actual library versions, hardware limits, and structure of the input dataset.
```

### G. Visual Quality and Current Limitations

```text
Visual inspection showed that the reconstruction is strongest around the middle side-view orbit represented by the training cameras. The overall body, rim, carved ornament, foot, and matte terracotta appearance remain stable when the model is viewed from these directions. No major holes, detached background components, or serious framing errors were visible in the normal captured range. The result is suitable as a successful first interactive baseline for the unglazed pot.

The reconstruction becomes less reliable when the virtual camera moves far above or below the captured orbit. Near-overhead inspection shows a soft and incomplete inner surface, while the exact underside contains unstable colorful Gaussians. Fine side texture also appears smoother than the original source when viewed closely. These limitations are mainly caused by missing observations rather than insufficient optimization. The original recording contains many overlapping side views but does not adequately observe the deep interior or underside of the pot. Additional training steps cannot reliably reconstruct geometry that was never visible in the input images.

The local browser viewer introduces another visual limitation because it displays the base spherical-harmonic color rather than the full degree-two view-dependent appearance used by the trained CUDA renderer. It is therefore useful for inspecting geometry and coverage, but its appearance can be smoother than the quantitative evaluation renders.
```

### H. Exported Model and Next Experiment

```text
The final checkpoint was exported as a standard PLY file of approximately 76 MB and a compact SPLAT file of 16 MB. Both exports contain 500,000 Gaussian records, and their computed SHA-256 values match the export manifest. The PLY file preserves the full degree-two spherical-harmonic representation, while the SPLAT file is intended for compact real-time viewing. The successful local Viser test confirmed that the exported reconstruction can be interactively inspected through a web browser without transferring the object to an external service.

The current baseline should be preserved so that later experiments can be compared against the same 35 held-out views. The next quality experiment should first test native-resolution images at 1125 by 2000 pixels with a short low-memory smoke run. More importantly, a later capture should include a middle ring, multiple upper-angle rings, near-overhead views, a low-angle ring around the foot, and genuine underside observations. Neighboring views should retain approximately 60 to 80 percent overlap, and changes in elevation should be gradual so that all new frames can be registered into a consistent camera model.

Only the first unglazed pot has completed the Gaussian Splatting process at this stage. The glossy second pot remains a separate reconstruction challenge because moving reflections may reduce feature consistency and create viewpoint-dependent appearance changes. Results for the first pot should therefore not be presented as completed results for both selected objects.
```

## Suggested New Figure Captions

The following captions follow the style of the existing report. Insert only the figures that are available and clearly legible in the two-column layout.

```text
Fig. 9. Sample foreground-masking result for the selected every-sixth-frame dataset, showing the original frame, propagated pot mask, and masked output.

Fig. 10. Sequential COLMAP reconstruction of the first unglazed pot. All 273 input images were registered in one connected camera orbit around the sparse pottery point cloud.

Fig. 11. Held-out evaluation example comparing the reference image with the rendering produced by the 7000-step Gaussian Splatting baseline.

Fig. 12. Local interactive visualization of the exported 500,000-Gaussian pottery reconstruction using the Viser browser interface.

Fig. 13. Free-viewpoint inspection showing the strong reconstructed side surface and the less reliable top or underside caused by missing camera coverage.
```

## References to Add Later

The final Word report should add properly formatted references for SAM 2, the original 3D Gaussian Splatting method, and gsplat. Their reference numbers should be assigned only after confirming the final order of citations in the complete report.
