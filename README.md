# Thai Ceramics Rendering Pipeline

This repository contains the current preprocessing tools for reconstructing
Thai pottery from video frames and photographs.

## Current features

- `scripts/capture` extracts sampled video frames and builds contact sheets.
- `scripts/features` demonstrates SIFT, FLANN, Lowe-ratio, and RANSAC matching.
- `scripts/masking` annotates pots, propagates SAM 2 masks, exports COLMAP masks
  and RGBA images, and generates quality-control reports.
- `tests/masking` verifies the masking pipeline without requiring a real SAM 2
  model during the automated run.
- `docs` contains feature guides, weekly reports, and document templates.

## Environment

Create and activate the environment from the repository root.

```powershell
micromamba create -f environment-masking.yml -y
micromamba activate pot-masking
```

## Main commands

```powershell
python scripts/capture/video_frame_extraction.py INPUT_VIDEO --every-n 3
python scripts/capture/build_frame_contact_sheet.py OUTPUT_FRAMES CONTACT_SHEET.jpg
python scripts/features/demo_feature_matching.py FRAME_1.jpg FRAME_2.jpg MATCHES.jpg
python scripts/masking/mask_dataset.py doctor
```

See `docs/capture.md` and `docs/masking.md` for detailed usage. The complete
component map and data flow are documented in `docs/system-architecture.md`.

When `OUTPUT_FRAMES` is omitted, frame extraction writes to
`data/frames_output/<video-name>_frames` by default.
