THAI CERAMICS 3D RECONSTRUCTION PROGRESS REPORT

Status updated: 1 September 2026

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

11. COLMAP Setup and Sparse-Reconstruction Configuration

COLMAP 4.1.1 with CUDA support was installed at:

C:/Tools/COLMAP-4.1.1/

The executable used by the reconstruction scripts is:

C:/Tools/COLMAP-4.1.1/bin/colmap.exe

The executable is called directly rather than through COLMAP.bat. The batch wrapper could not correctly handle the parentheses in the project directory name, "CSX4213 (Computer Vision)".

Masked feature extraction used:
- RGB images: data/frames_output/pot1-unglazed_every6_frames/
- COLMAP masks: data/processed/pot1-unglazed_every6/masks_colmap/
- Camera model: SIMPLE_RADIAL
- One shared camera
- Maximum 8,192 SIFT features per image
- Balanced GPU memory profile
- Maximum feature-extraction image size: 3,200 pixels
- First SIFT octave: 0
- Feature-extraction workers: 1
- Affine-shape estimation disabled
- Domain-size pooling disabled

The balanced GPU settings were selected after covariant CPU SIFT exhausted available memory and terminated with Windows access-violation code 0xC0000005. The revised settings completed without materially reducing the sparse-reconstruction quality.

The sparse runner and configuration are:
- scripts/reconstruction/run_colmap.py
- configs/colmap_pot1_unglazed_every6.yml

12. Exhaustive-Matching Sparse Reconstruction

The first masked sparse reconstruction used exhaustive matching.

Output directory:

data/processed/pot1-unglazed_every6/colmap_sparse_masked_exhaustion/

Runtime:
- Approximately 9 hours 31 minutes
- Approximately 571 minutes

Results:
- Input images: 273
- Disconnected sparse models: 11
- Largest model: sparse/5
- Registered images in the largest model: 126 of 273
- Sparse points in the largest model: 6,244
- Mean track length: approximately 13.63
- Mean reprojection error: approximately 1.015 pixels

The reconstruction produced a partial camera trajectory and several disconnected models. Although its largest component reconstructed the pot, it did not register the complete turntable sequence and was therefore retained as an experimental baseline rather than selected for dense reconstruction.

13. Sequential-Matching Sparse Reconstruction

The second sparse reconstruction used sequential matching because the source images are ordered video frames.

Matching settings:
- Matching method: sequential
- Sequential overlap: 15 frames
- Guided matching enabled

Output directory:

data/processed/pot1-unglazed_every6/colmap_sparse_masked_sequential/

Selected model:

data/processed/pot1-unglazed_every6/colmap_sparse_masked_sequential/sparse/0/

Runtime:
- Approximately 22 minutes

Results:
- Input images: 273
- Registered images: 273 of 273
- Registration rate: 100 percent
- Connected sparse models: 1
- Sparse points: 28,491
- Observations: 319,683
- Mean track length: 11.220491
- Mean observations per image: 1,171
- Mean reprojection error: 0.906756 pixels

Visual inspection in the COLMAP GUI showed a smooth, closed camera orbit around a compact pottery point cloud. The top and angled views showed consistent camera placement around the object without the disconnected trajectory found in the exhaustive result.

The sequential model was selected as the camera-pose and sparse-geometry source for both the classical dense reconstruction and the future 3D Gaussian Splatting branch.

14. Sparse-Matching Comparison

Exhaustive matching:
- Runtime: approximately 9 hours 31 minutes
- Registered images in largest model: 126 of 273
- Number of models: 11
- Largest-model sparse points: 6,244
- Largest-model reprojection error: approximately 1.015 pixels

Sequential matching:
- Runtime: approximately 22 minutes
- Registered images: 273 of 273
- Number of models: 1
- Sparse points: 28,491
- Reprojection error: 0.906756 pixels

Sequential matching was approximately 26 times faster than exhaustive matching for this dataset. It also registered every image in one connected model, produced substantially more sparse points, and achieved a lower mean reprojection error. This confirms that sequential matching is the appropriate method for the ordered every-sixth-frame video dataset.

15. Classical Dense-Reconstruction Preparation and Current State

The classical reconstruction branch uses the following planned sequence:

Masked RGB preparation → image undistortion → PatchMatch stereo → stereo fusion → dense point cloud → Poisson and Delaunay meshes → mesh texturing

The dense workspace is:

data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/

The dense-pipeline files are:
- configs/colmap_dense_pot1_unglazed_every6.yml
- scripts/reconstruction/run_colmap_dense.py

Completed dense-preparation products:
- 273 masked RGB input images
- 273 undistorted masked RGB images
- 273 geometrically aligned undistorted masks
- One undistorted PINHOLE sparse model
- All 273 registered camera poses retained
- Undistortion log: logs/02_undistort_rgb.log

The undistorted masks were generated through the same camera transformation and cropping process as the RGB images. This preserves pixel alignment between the color images, foreground masks, and camera model.

The following classical dense stages have not been run:
- PatchMatch stereo
- Geometric stereo fusion
- Dense fused point-cloud generation
- Poisson meshing
- Delaunay meshing
- Mesh texturing

No fused.ply, meshed-poisson.ply, or meshed-delaunay.ply result currently exists. The prepared undistorted images, masks, and sparse model were instead reused as the verified input to the 3D Gaussian Splatting branch.

16. 3D Gaussian Splatting Objective and Input Reuse

The 3D Gaussian Splatting branch was created to reconstruct and render the masked ceramic pot directly from the verified sequential COLMAP solution.

It uses:
- Undistorted masked images: colmap_dense_masked_sequential/images/
- Undistorted aligned masks: colmap_dense_masked_sequential/masks/
- Undistorted COLMAP model: colmap_dense_masked_sequential/sparse/

Verified source properties:
- Images: 273
- Masks: 273
- Source image resolution: 1125 × 2000 pixels
- Registered images: 273 of 273
- Camera model: one shared PINHOLE camera
- COLMAP initialization points: 28,491
- Mean COLMAP reprojection error: 0.906756 pixels

The source images already contain the pot over a black masked background. The aligned masks are retained separately as soft alpha so that training can use random-background compositing without introducing a dark boundary halo.

The 3DGS workspace is:

data/processed/pot1-unglazed_every6/gaussian_splatting_masked_sequential/

17. 3DGS Environment and Hardware Validation

A separate Micromamba environment named pot-3dgs was created to isolate the Gaussian Splatting dependencies.

Validated software:
- Windows 10 build 10.0.26200
- Python 3.10.21
- PyTorch 2.4.1+cu124
- PyTorch CUDA runtime 12.4
- gsplat 1.5.3+pt24cu124
- pycolmap 3.13.0
- viser 0.2.23

Validated hardware:
- NVIDIA GeForce GTX 1650
- CUDA compute capability 7.5
- 4.00 GB GPU memory

The official precompiled gsplat Windows wheel for Python 3.10, PyTorch 2.4, and CUDA 12.4 was used. This avoided requiring a local CUDA Toolkit and Visual Studio C++ compilation toolchain.

The environment doctor confirmed that Python, PyTorch, CUDA, gsplat, the GPU, and all required runtime libraries were available. It explicitly did not start image preparation, rasterization, or training.

Environment files:
- environment-3dgs.yml
- scripts/gaussian_splatting/environment_doctor.py

18. 3DGS Configuration, Safety Controls, and Test Coverage

The main 3DGS configuration is:

configs/gaussian_splatting_pot1_unglazed_every6.yml

Implemented scripts:
- scripts/gaussian_splatting/core/common.py
- scripts/gaussian_splatting/data/scene.py
- scripts/gaussian_splatting/data/preparation.py
- scripts/gaussian_splatting/training/runner.py
- scripts/gaussian_splatting/postprocessing/checkpoint.py
- scripts/gaussian_splatting/postprocessing/evaluation.py
- scripts/gaussian_splatting/postprocessing/exporter.py
- scripts/gaussian_splatting/postprocessing/viewer.py
- scripts/gaussian_splatting/diagnostics/environment.py
- scripts/gaussian_splatting/diagnostics/dataset.py

Command-line modules:
- scripts/gaussian_splatting/cli/doctor.py
- scripts/gaussian_splatting/cli/validate.py
- scripts/gaussian_splatting/cli/prepare.py
- scripts/gaussian_splatting/cli/train.py
- scripts/gaussian_splatting/cli/evaluate.py
- scripts/gaussian_splatting/cli/export.py
- scripts/gaussian_splatting/cli/view.py

The original top-level command paths were retained as lightweight compatibility wrappers:
- scripts/gaussian_splatting/environment_doctor.py
- scripts/gaussian_splatting/validate_dataset.py
- scripts/gaussian_splatting/prepare_dataset.py
- scripts/gaussian_splatting/run_training.py
- scripts/gaussian_splatting/evaluate_checkpoint.py
- scripts/gaussian_splatting/export_checkpoint.py
- scripts/gaussian_splatting/view_checkpoint.py

The wrappers contain no feature implementation and forward the original arguments to the organized package. Existing commands in the project documentation therefore remain valid. Equivalent new commands are available through python -m scripts.gaussian_splatting.cli.<command>.

Implemented controls:
- Project-boundary path validation
- Exact input-count and sparse-model validation
- Explicit --dry-run or --run acknowledgement for expensive stages
- Refusal to overwrite existing training, evaluation, or export outputs
- Deterministic train/test splitting
- Reproducible random seed
- Low-memory packed rasterization
- Sparse gradients
- Disabled training-time viewer
- Gaussian-count caps for the 4 GB GPU
- Run manifests, configuration snapshots, CSV logs, summaries, and checksums

The long-running image preparation, training, evaluation, export, and viewer stages were run manually by the project owner. Development checks were limited to dry runs, CPU tests, and read-only artifact inspection.

The final Gaussian Splatting unit-test suite contains 23 tests. All 23 tests pass. The tests cover configuration and path safety, sparse-model validation, normalization, image/mask resizing, environment compatibility, CUDA-memory API behavior, random-background compositing, checkpoint integrity, covariance conversion, viewer filtering, export naming, and old/new command compatibility.

No Git commit or push was performed during this development session.

19. Mask-Aware 3DGS Image Preparation

Prepared caches were generated at factors 2 and 4.

Factor-2 cache:
- Resolution: 562 × 1000 pixels
- Source images: 273
- Prepared images: 273
- Prepared soft masks: 273
- Output: gaussian_splatting_masked_sequential/cache/factor_2/

Factor-4 cache:
- Resolution: 281 × 500 pixels
- Source images: 273
- Prepared images: 273
- Prepared soft masks: 273
- Output: gaussian_splatting_masked_sequential/cache/factor_4/

The resizer treats the masked RGB images as premultiplied color, resizes RGB and alpha consistently, and converts the result back to straight foreground color. This avoids dark edge contamination when the pot is composited over random training backgrounds.

A deterministic every-eighth-image holdout split was created:
- Training images: 238
- Test images: 35

Preparation outputs:
- dataset_manifest.json
- holdout_split.json
- factor-specific cache manifests

The preparation process completed successfully and did not start training.

20. Smoke Training and Problems Resolved

The smoke_lowmem profile was designed to validate the complete training path on the GTX 1650 before attempting full training.

Smoke profile:
- Prepared factor: 4
- Resolution: 281 × 500
- Training/test images: 238/35
- Steps: 300
- Initial Gaussians: 28,491
- Gaussian cap: 100,000
- Spherical-harmonic degree: 1
- Packed rasterization: enabled
- Sparse gradients: enabled
- Densification disabled during the 300-step smoke window

Three compatibility problems were found and corrected before the successful run.

Problem 1 — PyTorch CUDA peak-memory device string:
- torch.cuda.reset_peak_memory_stats("cuda:0") raised RuntimeError: Invalid device argument.
- The runner was changed to use an integer CUDA device index.

Problem 2 — PyTorch CUDA peak-memory integer index:
- The same Windows PyTorch 2.4 build also rejected reset_peak_memory_stats(0).
- Direct testing confirmed that the no-argument API works after torch.cuda.set_device(0).
- All peak-memory reset and query calls now use the explicitly selected current CUDA device without a device argument.

Problem 3 — gsplat packed-background tensor shape:
- gsplat 1.5.3 raised an assertion because the documented batched background shape [1, 3] was rejected by the packed CUDA wrapper.
- Rasterization was changed to use backgrounds=None.
- Random RGB backgrounds are now composited after rasterization using the returned alpha map.
- This is mathematically equivalent and matches the compositing pattern used by the official gsplat trainer.

The failed smoke attempts stopped before a completed iteration and produced no checkpoint. Their incomplete run directory was preserved rather than overwritten.

Successful smoke run:

gaussian_splatting_masked_sequential/runs/smoke_lowmem_retry1/

Results:
- Completed steps: 300 of 300
- Runtime: 16.96 seconds, or approximately 0.28 minutes
- Initial logged loss: 0.127449
- Final loss: 0.020068
- Final logged SSIM: 0.953751
- Final Gaussians: 28,491
- Peak allocated VRAM: 0.075 GB
- Checkpoint: checkpoints/step_000300.pt
- Three held-out comparison renders created

Visual inspection confirmed correct pot silhouette, scale, pose, rim, and ornament alignment. The smoke render remained softer than the reference, which was expected after only 300 steps without densification.

21. Full 3DGS Training

The selected full profile was baseline_7k.

Profile configuration:
- Prepared factor: 2
- Resolution: 562 × 1000
- Training/test images: 238/35
- Steps: 7,000
- Initial Gaussians: 28,491
- Maximum Gaussians: 500,000
- Spherical-harmonic degree: 2
- Packed rasterization: enabled
- Sparse gradients: enabled
- MCMC refinement start: step 500
- MCMC refinement stop: step 6,500
- Refinement interval: 100 steps

During MCMC refinement, low-opacity Gaussians were relocated and approximately five percent new Gaussians were periodically added. Observed relocation counts of approximately 133–186 Gaussians per cycle were normal and represented only a small fraction of the model.

Full-training results:
- Completed steps: 7,000 of 7,000
- Runtime: 602.23 seconds, or approximately 10.04 minutes
- Initial logged loss: 0.125020
- Final loss: 0.007804
- Final logged SSIM: 0.977806
- Final Gaussians: 500,000
- Gaussian cap reached successfully
- Peak allocated VRAM: 0.641 GB
- Final checkpoint size: 76,003,504 bytes

Final checkpoint:

gaussian_splatting_masked_sequential/runs/baseline_7k/checkpoints/step_007000.pt

The checkpoint contains finite tensors with the expected shapes:
- means: 500,000 × 3
- scales: 500,000 × 3
- quaternions: 500,000 × 4
- opacities: 500,000
- SH base coefficients: 500,000 × 1 × 3
- SH higher-order coefficients: 500,000 × 8 × 3

Visual comparison on held-out views showed correct geometry, viewpoint, silhouette, color, rim opening, foot, and carved ornament. No major holes, floaters, framing errors, or background leakage were visible in the normal side-view camera range.

22. Held-Out Evaluation

The completed baseline checkpoint was evaluated on all 35 deterministic held-out images at 562 × 1000 resolution.

Evaluation output:

gaussian_splatting_masked_sequential/runs/baseline_7k/evaluation/holdout_black/

Mean results:
- Full-frame PSNR: 35.364 dB
- Full-frame SSIM: 0.97779
- Foreground-only PSNR: 30.856 dB
- Foreground L1 error: 0.012923 on a 0–1 scale
- Alpha silhouette IoU: 0.97833
- Peak allocated VRAM: 0.382 GB
- Runtime: 5.51 seconds

Per-view ranges:
- PSNR: 33.473–36.016 dB
- SSIM: 0.97230–0.97941
- Foreground PSNR: 27.895–31.552 dB
- Foreground L1: 0.01163–0.02041
- Alpha IoU: 0.96159–0.98235

All 35 comparisons were saved. Read-only inspection included the lowest-PSNR view, the lowest-alpha-IoU view, and representative views. These cases retained the correct pot structure and showed no catastrophic side-view failure.

Metric interpretation:
- PSNR measures pixel-level agreement; values above 30 dB are strong for this reconstruction.
- SSIM measures structural and contrast similarity; a value near 0.978 indicates high structural agreement.
- Foreground PSNR excludes most of the easy black background and is the more informative appearance score.
- Foreground L1 indicates approximately 1.3 percent average absolute color error per foreground channel.
- Alpha IoU measures overlap between rendered and reference silhouettes; 0.978 indicates highly consistent boundaries.

The holdout images come from the same ordered capture session. Therefore, the results demonstrate strong interpolation around the captured side orbit, but they do not prove performance under unrelated lighting, backgrounds, or camera elevations outside the training distribution.

23. Model Export and Local Viewer

The final checkpoint was exported to two portable formats.

Export directory:

gaussian_splatting_masked_sequential/runs/baseline_7k/exports/

Standard PLY export:
- File: thai_ceramics_baseline_7k.ply
- Size: 76,000,952 bytes
- Vertices/Gaussians: 500,000
- Contains positions, full degree-2 SH coefficients, opacity, scale, and rotation
- SHA-256: a330e951dc5ca28ec95930d692ba41dffd45d55fba7cd3856f81548dc58320d6

Compact splat export:
- File: thai_ceramics_baseline_7k.splat
- Size: 16,000,000 bytes
- Record size: 32 bytes
- Gaussian records: 500,000
- Stores compact base-color splat data
- SHA-256: 470a3d3e213fd434778596e068756c037d2e8c7575e363410a659b7cb0109296

Both computed checksums match export_manifest.json. The PLY header was inspected and correctly declares 500,000 binary little-endian vertices and the complete degree-2 spherical-harmonic fields.

A local viewer was implemented with viser and tested successfully at:

http://localhost:8080

The viewer:
- Runs locally without uploading the model
- Is configured to load up to 500,000 Gaussians after opacity filtering
- Supports orbit, pan, and zoom
- Can use a lower --max-gaussians value for slower browsers
- Uses position, covariance, opacity, and base SH color

The current local viewer uses only the base spherical-harmonic color, not the complete SH2 view-dependent appearance. Consequently, it can look smoother than the CUDA evaluation renders and is intended primarily for interactive geometry and coverage inspection.

24. Current Reconstruction Strengths and Limitations

Strengths:
- Complete 273-image camera registration
- Strong side-view geometry and silhouette
- Stable carved ornament and rim alignment
- High held-out PSNR, SSIM, and alpha IoU
- Efficient training within the GTX 1650 memory limit
- Complete checkpoint, evaluation records, portable exports, and local viewer

Limitations discovered through free-viewpoint inspection:
- The upper interior becomes blurry when viewed from near-overhead positions.
- The exact underside contains colorful floaters and unstable Gaussian appearance.
- Fine side texture is smoother than the original source at close zoom.
- The present capture is dominated by one side orbit and does not adequately observe the top interior or bottom surface.
- 3DGS is a radiance-field representation and does not inherently provide a watertight, measurement-ready surface mesh.

The top and bottom artifacts are not primarily training failures. They occur because these regions were not sufficiently observed by registered cameras. Additional optimization cannot reliably invent missing geometry.

The side-view softness has three contributing factors:
- Training used factor-2 images at 562 × 1000 instead of the 1125 × 2000 source resolution.
- The browser viewer displays only base SH color.
- The source video frames provide many overlapping side views but limited new elevation and close-detail information.

The existing 500,000-Gaussian, 7,000-step model has converged strongly on its available data. Simply adding more nearly identical frames or more factor-2 steps is expected to produce diminishing returns.

25. Recommended Detail and Coverage Improvements

The current baseline_7k model should be preserved as the successful v1 baseline. New experiments should use separate run and dataset directories.

Recommended current-data quality experiment:
- Prepare a factor-1 cache at 1125 × 2000
- Run a 300–500-step factor-1 smoke test first
- Use packed rasterization and sparse gradients
- Begin with a 500,000-Gaussian cap
- Use SH degree 2 because the matte pot does not primarily require higher-order view dependence
- If memory remains safe, run approximately 10,000 native-resolution steps
- Compare factor 1 directly with the current factor-2 baseline

More steps alone are lower priority than native-resolution input. Raising SH degree primarily improves view-dependent appearance, not physical shape. Increasing the Gaussian cap can add capacity but should be tested only after verifying the factor-1 memory requirement.

Recommended multilevel v2 capture:
- Preserve approximately 60–80 percent overlap between neighboring frames
- Capture one middle 360-degree ring
- Capture an upper 30–45-degree ring
- Capture an upper 60–75-degree ring that sees the inner wall
- Capture approximately 8–12 near-overhead frames
- Capture a low-angle 360-degree ring around the foot
- Capture genuine underside views with the pot raised on a narrow or transparent support
- Add closer, sharp side frames when they provide genuinely higher ornament resolution
- Use diffuse, consistent lighting and avoid motion blur
- Transition gradually between elevations so sequential COLMAP matching retains overlap

If the pot is turned upside down, those images must not simply be mixed into the same rigid sequence. Turning the pot changes its pose relative to the scene. The recommended options are either keeping it fixed on a support that permits underside photography or reconstructing the upside-down sequence separately and aligning the two reconstructions later.

Every new v2 frame must pass through:

masking → COLMAP registration → undistortion → aligned-mask preparation → 3DGS cache preparation → retraining → top/bottom holdout evaluation

New files must not be copied directly into the current prepared cache because each image requires an aligned mask and a registered COLMAP camera pose.

Optional future methods:
- 2D Gaussian Splatting for stronger surface-oriented geometry and normal/depth consistency
- SuGaR for surface alignment and mesh extraction
- Mip-Splatting or antialiased rasterization for zoom and sampling-rate artifacts
- An SH-aware interactive viewer for a closer match to the trained CUDA renderer

These methods have not been added to the current pipeline. Better camera coverage and native-resolution training should be evaluated before introducing a more complex reconstruction model.

26. Current Project Status

Completed stages:
- Micromamba masking and 3DGS environment setup
- CUDA-enabled PyTorch and gsplat validation
- Video-frame extraction and sampling comparison
- SAM 2 annotation, segmentation, and mask quality control
- Selection of every-sixth-frame sampling
- COLMAP installation and Windows path handling
- Exhaustive sparse-reconstruction baseline
- Sequential sparse reconstruction with 273 of 273 registered images
- Masked RGB preparation
- RGB and mask undistortion
- Verified aligned PINHOLE input model for 3DGS
- Factor-2 and factor-4 3DGS cache preparation
- Deterministic 238/35 training/test split
- 300-step low-memory smoke training
- Resolution of PyTorch CUDA-memory API compatibility problems
- Resolution of gsplat packed-background compatibility problem
- 7,000-step MCMC full training
- 500,000-Gaussian final checkpoint
- 35-view quantitative evaluation
- PLY and compact splat export
- Checksum and format validation
- Local interactive viewer test
- 23 passing Gaussian Splatting tests
- Responsibility-based 3DGS package organization
- Backward-compatible old script commands and new module commands

Pending or optional stages:
- Classical PatchMatch stereo, fusion, meshing, and texturing
- Native-resolution factor-1 smoke and quality experiment
- Multilevel v2 capture with top, interior, low-angle, and underside coverage
- Retraining and dedicated top/bottom evaluation
- Optional surface-oriented 2DGS or SuGaR experiment
- Optional SH-aware interactive viewer

The immediate recommended next step is to preserve the successful v1 artifacts, then add a guarded factor-1 smoke profile and cache for the current dataset. This can quantify how much side detail is recoverable from the existing source images before committing to the multilevel v2 recapture.
