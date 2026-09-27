# Lid side and underside reconstruction

This workspace is for the lid alone. Its side and rotated underside captures
are processed separately from the main pot. The extracted top capture remains
available but is not part of this first reconstruction attempt.

The first step validates and stages the image and COLMAP mask pairs using
`configs/lid_side_underside_dataset.yml`. The staged data and all later lid
reconstruction outputs use `lid_side_underside` paths, leaving the source
captures and pot reconstruction unchanged.

The underside video is cropped against the image boundary in part of its orbit.
Its masks are usable for a trial, but the missing rim pixels cannot be restored
by masking. Inspect cross-view matches and camera poses before 3DGS training.

`run_sparse.py` calls the shared, validated COLMAP runner with
`configs/colmap_lid_side_underside.yml`. Its workspace is under
`data/processed/lid_side_underside/colmap_sparse_masked` and is separate from
all pot reconstructions. Run `--dry-run` first to check staged inputs and the
within-sequence and side-to-underside pair plan without starting COLMAP.

The initial standard SIFT run registered all 214 side and 202 underside images
in separate models, with no cross-view correspondences. `prepare_lightglue.py`
copies the database and side model into a second isolated workspace for a
targeted side-to-underside SIFT-LightGlue trial. It does not modify either
original sparse model or database.

`cleanup_lightglue.py` removes only failed side-to-underside SIFT match records
from that copied database. Run its `--dry-run` mode and inspect the reported
record counts before applying the cleanup.

`run_lightglue.py` uses the lid-only LightGlue configuration to match the 1,848
targeted bridge pairs. Run `--dry-run` in `pot-lightglue` first. Inspect verified
cross-view pairs before attempting camera registration.

The LightGlue trial produced no raw or geometrically verified bridge matches.
The accepted lid baseline therefore uses only sparse model `1`, which contains
all 214 side cameras. `configs/lid_side_only_dataset.yml` stages the side RGB
images and masks with the exact `side_` filename prefix used by that model.
The underside models and data remain preserved as separate experiments.

`configs/colmap_undistort_lid_side_only.yml` uses the 214-image side model and
the side-only staged filenames for masked RGB and mask undistortion. All 3DGS
input files are written under `data/processed/lid_side_only/gs_input`.

`configs/gaussian_splatting_lid_side_only.yml` defines a 300-step factor-4
smoke profile and a factor-2 7,000-step baseline. Its cache, holdout split,
training runs, evaluations, and exports remain inside the lid-side-only root.
