# System Architecture

This document describes only the project folder structure and the function of
each folder. The classical reconstruction branch is intentionally excluded.

## Repository Folder Structure

```text
thai-ceramics-rendering-pipeline/
├── configs/
│   └── gaussian_splatting_pot1_unglazed_every6.yml
├── data/
│   ├── raw/
│   ├── frames_output/
│   └── processed/
├── docs/
│   ├── capture.md
│   ├── masking.md
│   ├── gaussian_splatting.md
│   ├── system-architecture.md
│   ├── progress/
│   ├── reports/
│   └── templates/
├── models/
├── scripts/
│   ├── capture/
│   ├── features/
│   ├── masking/
│   └── gaussian_splatting/
├── tests/
│   ├── masking/
│   └── gaussian_splatting/
├── environment-masking.yml
├── environment-3dgs.yml
├── .gitignore
├── IMPORTANT_INFO.md
└── README.md
```

## Top-Level Folder Functions

| Path | Function |
| --- | --- |
| `configs/` | Stores reproducible YAML settings for 3DGS dataset paths, preparation profiles, training profiles, evaluation, and export. |
| `data/raw/` | Stores original source videos and photographs without modification. |
| `data/frames_output/` | Stores frames extracted or selected from source videos. |
| `data/processed/` | Stores masks, prepared 3DGS datasets, training runs, evaluations, and exports, grouped by ceramic object. |
| `docs/` | Stores technical documentation, progress records, reports, and course templates. |
| `models/` | Stores external model weights required by local processing, such as SAM 2 checkpoints. |
| `scripts/capture/` | Extracts video frames and builds contact sheets for dataset review. |
| `scripts/features/` | Contains standalone local-feature matching demonstrations and experiments. |
| `scripts/masking/` | Creates and reviews object masks, transparent previews, and quality-control outputs. |
| `scripts/gaussian_splatting/` | Validates, prepares, trains, evaluates, exports, and displays the 3D Gaussian Splatting model. |
| `tests/masking/` | Tests mask processing, output generation, prompts, quality checks, and overwrite protection. |
| `tests/gaussian_splatting/` | Tests 3DGS configuration, dataset preparation, training utilities, postprocessing, diagnostics, and command compatibility. |
| `environment-masking.yml` | Defines the Micromamba environment used for SAM 2 masking. |
| `environment-3dgs.yml` | Defines the low-memory PyTorch and gsplat environment used for 3DGS on the GTX 1650. |

## Data Folder Structure

```text
data/
├── raw/
│   └── videos/
│       └── pot1-unglazed.mp4
├── frames_output/
│   └── pot1-unglazed_every6_frames/
│       ├── frame images
│       └── frames_manifest.csv
└── processed/
    └── pot1-unglazed_every6/
        ├── masks_object/
        ├── masks_colmap/
        ├── rgba/
        ├── overlays/
        ├── mask_manifest.csv
        ├── run_metadata.json
        ├── qc_contact_sheet.jpg
        ├── colmap_dense_masked_sequential/
        │   ├── images/
        │   ├── masks/
        │   └── sparse/
        └── gaussian_splatting_masked_sequential/
            ├── cache/
            │   ├── factor_2/
            │   └── factor_4/
            ├── dataset_manifest.json
            ├── holdout_split.json
            └── runs/
                ├── smoke_lowmem_retry1/
                └── baseline_7k/
                    ├── checkpoints/
                    ├── evaluation/
                    ├── exports/
                    ├── previews/
                    ├── training_log.csv
                    └── training_summary.json
```

## Data Folder Functions

| Path | Function |
| --- | --- |
| `data/raw/videos/` | Keeps the original ceramic videos as read-only source material. |
| `data/frames_output/<dataset>/` | Contains sampled RGB frames and their extraction manifest. |
| `masks_object/` | Contains full-resolution binary pot silhouettes. |
| `masks_colmap/` | Contains slightly eroded masks using the required image-aligned naming convention. |
| `rgba/` | Contains the original RGB pixels with the pot mask stored as transparency. |
| `overlays/` | Contains lightweight visual previews for mask inspection. |
| `mask_manifest.csv` | Records per-frame mask measurements, paths, and quality warnings. |
| `run_metadata.json` | Records the masking environment, hardware, model, and processing settings. |
| `qc_contact_sheet.jpg` | Shows sampled and flagged masks for rapid quality review. |
| `colmap_dense_masked_sequential/images/` | Supplies the verified undistorted masked images used by 3DGS. |
| `colmap_dense_masked_sequential/masks/` | Supplies masks aligned pixel-for-pixel with the undistorted images. |
| `colmap_dense_masked_sequential/sparse/` | Supplies registered camera parameters, poses, and sparse initialization points. |
| `gaussian_splatting_masked_sequential/cache/` | Stores reusable, mask-aware image caches at the resolution required by each training profile. |
| `dataset_manifest.json` | Records source files, prepared profiles, image dimensions, and dataset metadata. |
| `holdout_split.json` | Stores deterministic training and held-out evaluation image lists. |
| `runs/<run-name>/checkpoints/` | Stores serialized 3DGS training checkpoints. |
| `runs/<run-name>/evaluation/` | Stores held-out renders, comparisons, and quantitative metrics. |
| `runs/<run-name>/exports/` | Stores portable PLY and SPLAT models, checksums, and the export manifest. |
| `runs/<run-name>/previews/` | Stores selected reconstruction previews. |
| `training_log.csv` | Stores step-level loss, quality, Gaussian-count, and memory measurements. |
| `training_summary.json` | Stores final run settings, duration, loss, Gaussian count, and peak VRAM. |

## Script Folder Structure

```text
scripts/
├── capture/
│   ├── video_frame_extraction.py
│   └── build_frame_contact_sheet.py
├── features/
│   └── demo_feature_matching.py
├── masking/
│   ├── mask_dataset.py
│   └── masking_core.py
└── gaussian_splatting/
    ├── cli/
    │   ├── doctor.py
    │   ├── validate.py
    │   ├── prepare.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── export.py
    │   └── view.py
    ├── core/
    │   └── common.py
    ├── data/
    │   ├── scene.py
    │   └── preparation.py
    ├── diagnostics/
    │   ├── environment.py
    │   └── dataset.py
    ├── training/
    │   └── runner.py
    ├── postprocessing/
    │   ├── checkpoint.py
    │   ├── evaluation.py
    │   ├── exporter.py
    │   └── viewer.py
    ├── environment_doctor.py
    ├── validate_dataset.py
    ├── prepare_dataset.py
    ├── run_training.py
    ├── evaluate_checkpoint.py
    ├── export_checkpoint.py
    └── view_checkpoint.py
```

## Script Folder Functions

| Path | Function |
| --- | --- |
| `scripts/capture/video_frame_extraction.py` | Samples frames from a source video and records frame metadata. |
| `scripts/capture/build_frame_contact_sheet.py` | Creates a labeled image sheet for quickly reviewing extracted frames. |
| `scripts/features/demo_feature_matching.py` | Demonstrates SIFT detection, descriptor matching, and geometric verification. |
| `scripts/masking/mask_dataset.py` | Provides the command-line interface for masking environment checks, annotation, processing, and review. |
| `scripts/masking/masking_core.py` | Implements mask propagation, cleanup, export, manifest generation, and quality control. |
| `scripts/gaussian_splatting/cli/` | Contains the organized `python -m` command entry points. |
| `scripts/gaussian_splatting/core/` | Provides shared configuration loading, project paths, validation, and utility functions. |
| `scripts/gaussian_splatting/data/` | Loads the prepared camera scene and creates mask-aware image caches and train/test splits. |
| `scripts/gaussian_splatting/diagnostics/` | Checks the Python/CUDA/gsplat environment and validates the prepared dataset. |
| `scripts/gaussian_splatting/training/` | Initializes Gaussians from sparse points and performs low-memory gsplat optimization. |
| `scripts/gaussian_splatting/postprocessing/` | Loads checkpoints, evaluates held-out views, exports PLY/SPLAT files, and starts the local viewer. |
| Top-level files in `scripts/gaussian_splatting/` | Preserve the original command paths as lightweight wrappers around the organized modules. |

## Test Folder Structure

```text
tests/
├── masking/
│   └── test_masking_pipeline.py
└── gaussian_splatting/
    ├── test_cli_compatibility.py
    ├── test_common.py
    ├── test_dataset.py
    ├── test_environment_doctor.py
    ├── test_postprocess.py
    ├── test_prepare_dataset.py
    └── test_run_training.py
```

## Test Folder Functions

| Path | Function |
| --- | --- |
| `tests/masking/test_masking_pipeline.py` | Verifies the masking pipeline without requiring a live SAM 2 processing run. |
| `tests/gaussian_splatting/test_cli_compatibility.py` | Confirms that the original commands and organized module commands call the same implementations. |
| Other files in `tests/gaussian_splatting/` | Verify shared utilities, dataset handling, environment checks, preparation, training behavior, checkpoints, evaluation, export, and viewer helpers. |
