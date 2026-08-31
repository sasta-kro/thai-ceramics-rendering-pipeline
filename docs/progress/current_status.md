SAM 2 POT MASKING PROGRESS REPORT

1. Objective

The purpose of this stage was to isolate the Thai ceramic pot from the blue background and white turntable before performing 3D reconstruction. SAM 2 was used to create pot silhouettes, transparent-background images, COLMAP-compatible masks, visual overlays, and quality-control reports.

2. Environment and Hardware

The masking environment was created with Micromamba under the name pot-masking.

Software:
- Python 3.12.14
- NumPy 2.5.2
- OpenCV 5.0.0
- Pillow 12.3.0
- PyTorch 2.13.0 with CUDA 12.6
- SAM 2 version 1.0
- SAM 2.1 Hiera Tiny checkpoint

Hardware:
- NVIDIA GeForce GTX 1650
- 4 GB GPU memory
- Float16 inference
- Video and model-state CPU offloading enabled

The complete SAM 2 smoke test passed in 2.42 seconds, confirming that the model, checkpoint, CUDA device, and environment were functioning correctly.

3. Source Dataset

The first experiment used the unglazed Thai pot video:

data/raw/videos/pot1-unglazed.mp4

Video properties:
- Resolution: 2160 × 3840 pixels
- Frame rate: approximately 60 FPS
- Duration: approximately 27.2 seconds
- Material type: matte terracotta

4. Frame-Sampling Experiments

Two sampling intervals were evaluated.

Experiment A — Every Third Frame:
- Sampling interval: 3
- Effective frame rate: approximately 20 FPS
- Extracted frames: 545
- Processing time: 2,231.37 seconds, or approximately 37 minutes 11 seconds

Experiment B — Every Sixth Frame:
- Sampling interval: 6
- Effective frame rate: approximately 10 FPS
- Extracted frames: 273
- Processing time: 1,122.76 seconds, or approximately 18 minutes 43 seconds

5. Annotation Process

The first frame of each dataset was manually annotated using:
- One tight bounding box around the complete pot
- One positive point inside the pot
- Negative points on the white turntable and blue background

The pot opening and visible interior were retained as part of the pot object.

The annotation window was modified to fit within the computer display. Its preview is now limited to 1200 × 800 pixels by default while preserving coordinate mapping to the original 2160 × 3840 image.

6. Generated Outputs

SAM 2 generated the following products for every processed frame:

- masks_object: full-resolution binary pot masks
- masks_colmap: slightly eroded masks formatted for COLMAP
- rgba: original RGB images with transparent backgrounds
- overlays: images showing the predicted mask boundary
- mask_manifest.csv: frame-level measurements and QC information
- run_metadata.json: environment, hardware, model, and runtime information
- qc_contact_sheet.jpg: sampled and flagged masks for visual inspection

The original extracted RGB frames were not modified.

7. Every-Third-Frame Results

Output directory:

data/processed/pot1-unglazed_every3/

Results:
- Completed masks: 545
- COLMAP masks: 545
- RGBA images: 545
- Review overlays: 545
- Average mask-area ratio: 0.223602
- Mask-area range: 0.221309–0.226583
- Average adjacent-frame IoU: 0.998631
- Minimum adjacent-frame IoU: 0.943949
- Rotation-loop IoU: 0.868319
- QC warning records: 2

The masks were generally accurate and stable. A small boundary defect was visible near the lower-left base of the pot in the first frame. The final frame was also flagged for low previous-frame IoU and low rotation-loop IoU.

8. Every-Sixth-Frame Results

Output directory:

data/processed/pot1-unglazed_every6/

Results:
- Completed masks: 273
- COLMAP masks: 273
- RGBA images: 273
- Review overlays: 273
- Average mask-area ratio: 0.223323
- Mask-area range: 0.220759–0.226142
- Average adjacent-frame IoU: 0.997814
- Minimum adjacent-frame IoU: 0.950811
- Rotation-loop IoU: 0.872302
- QC warning records: 2

Visual inspection confirmed that the masks accurately followed the pot rim, body, and base. The blue background and white turntable were excluded. The first, middle, and final frames all retained clean pot boundaries.

9. Experimental Comparison

The every-third-frame dataset contained approximately twice as many images and required almost twice the processing time. However, it did not produce a meaningful improvement in mask accuracy.

The masks produced by the two experiments had an average IoU agreement of 0.998400 across their 273 shared frames. This means the results were approximately 99.84% identical.

The every-sixth-frame experiment provided:
- Comparable segmentation accuracy
- Slightly better rotation-loop IoU
- A cleaner first-frame boundary
- Half the number of images
- Approximately half the processing time
- Lower storage and later reconstruction requirements

10. Final Sampling Decision

Every-sixth-frame sampling was selected for the remaining pipeline.

For a 60 FPS source video, this produces an effective rate of approximately 10 FPS. This rate preserves sufficient overlap between consecutive views while avoiding excessive duplicate images.

The selected dataset is:

data/frames_output/pot1-unglazed_every6_frames/

The selected masking output is:

data/processed/pot1-unglazed_every6/

11. Current Project Status

The following stages are complete:
- Micromamba environment setup
- CUDA-enabled PyTorch installation
- SAM 2 installation
- SAM 2.1 Hiera Tiny checkpoint setup
- Runtime and GPU validation
- Video-frame extraction
- Manual pot annotation
- Background-mask generation
- COLMAP-mask generation
- Sampling-rate comparison
- Selection of every-sixth-frame sampling

The next project stage is feature extraction, feature matching, camera-pose estimation, sparse reconstruction, dense reconstruction, mesh generation, and texture mapping using COLMAP.