# Thai Ceramics 3D Reconstruction Pipeline

This project reconstructs Thai ceramic objects from turntable videos using
foreground masking, COLMAP camera reconstruction, and 3D Gaussian Splatting
(3DGS). It also includes a classical dense point-cloud and textured-mesh
pipeline for the first unglazed pot.

## Project status

The reconstruction experiments are complete. The repository now retains only
source code, YAML configurations, documentation, final models, evaluations,
and exported results. Raw videos, extracted frames, masks, caches, smoke runs,
and intermediate workspaces were removed after completion.

Restoring the original videos or frames is required before rerunning capture,
masking, COLMAP reconstruction, or training. Final COLMAP models remain, but
their deleted image sets and feature databases are no longer available for GUI
image inspection.

## Pipeline

1. Sample frames from turntable videos.
2. Annotate and propagate SAM 2 foreground masks.
3. Estimate cameras and sparse geometry with COLMAP and LightGlue-assisted
   matching where needed.
4. Undistort RGB images and binary masks.
5. Train, evaluate, view, and export gsplat 3D Gaussian models.
6. For Pot 1 side-only, generate dense point clouds and textured meshes.

## Environments

The project uses Micromamba environments with separate dependencies:

| Environment | Definition | Purpose |
|---|---|---|
| `pot-masking` | `environment-masking.yml` | Capture, masks, dataset preparation, and COLMAP orchestration |
| `pot-lightglue` | `environment-lightglue.yml` | LightGlue feature matching and repair |
| `pot-3dgs` | `environment-3dgs.yml` | gsplat training, evaluation, viewing, and export |

Activate an existing environment with:

```powershell
micromamba activate pot-3dgs
```

## Retained final results

| Result | Capture/model | Final output directory |
|---|---|---|
| Pot 1 side-only | Side orbit, 7k 3DGS baseline | `data/processed/pot1-unglazed_every6/gaussian_splatting_masked_sequential/runs/baseline_7k` |
| Pot 1 classical | Dense cloud, Poisson/Delaunay meshes, textured mesh | `data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/results` |
| Pot 1 repaired multiview | Side, elevated, and underside; 15k quality model | `data/processed/pot1-unglazed_multiview/gaussian_splatting_masked_multiview_lightglue_repaired/runs/quality_multiview_repaired` |
| Pot 1 three-angle | Side, intermediate 25-degree, and underside; 7k model | `data/processed/pot1-unglazed_side_underside_mid25/gaussian_splatting/runs/baseline_side_under_mid25_7k` |
| Lid | Side-only 7k model with original and lower-cleaned exports | `data/processed/lid_side_only/gaussian_splatting/runs/baseline_lid_7k` |
| Lift | Four-view 7k openwork model | `data/processed/lift_all_angles/gaussian_splatting/runs/baseline_lift_7k` |

Each accepted 3DGS run retains:

- a `.pt` training checkpoint;
- PLY and SPLAT exports;
- evaluation metrics and comparison renders;
- training configuration and summary metadata.

Final COLMAP sparse models are retained beside their respective pipeline
outputs. The Pot 1 classical result additionally includes fused point clouds,
mesh variants, and textured meshes.

## Known limitation

The lift reconstructs its interior most reliably. Exterior side and bottom
free views are softer because separately captured camera rings did not provide
fully consistent geometry between orientations. Additional optimization alone
cannot replace missing continuous overlap between those captures.

## Repository layout

```text
configs/       Pipeline configuration files
scripts/       Capture, masking, reconstruction, training, export, and cleanup tools
tests/         Automated pipeline tests
docs/          Technical guides and project reports
models/        SAM 2 model dependency
data/processed Final retained models and results
```

Detailed implementation notes and reports are available under `docs/`. The
guarded final-data audit is implemented in
`scripts/cleanup_completed_project.py`; its dry run currently reports no
unnecessary data remaining.
