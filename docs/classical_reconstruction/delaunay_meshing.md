# Dense Delaunay meshing

Delaunay meshing creates a visibility-aware triangle surface from the fused point cloud and its camera-visibility records. It is an alternative to Poisson meshing, not a process applied to the Poisson mesh. Keeping both results makes it possible to compare Poisson's smoother, gap-filling surface with Delaunay's generally more outlier-robust but less smooth surface.

## Prepared configuration

The settings are stored in `configs/colmap_dense_pot1_unglazed_every6.yml`:

```yaml
meshing:
  run_delaunay: true
  delaunay:
    input_type: dense
    max_proj_dist: 20.0
    max_depth_dist: 0.05
    visibility_sigma: 3.0
    distance_sigma_factor: 1.0
    quality_regularization: 1.0
    max_side_length_factor: 25.0
    max_side_length_percentile: 95.0
    num_threads: 4
```

These are COLMAP's balanced dense-meshing values with the thread count limited to four. Delaunay meshing is CPU- and RAM-intensive.

## Workspace preparation

COLMAP requires `fused.ply` and `fused.ply.vis` at the dense workspace root. This project stores the authoritative fusion artifacts under `results/`. The Python runner automatically creates workspace-root hard links to those files and falls back to copying if hard links are unavailable. It does not move or modify the original fusion artifacts.

## Run from PowerShell

Open PowerShell in the repository root and activate the project environment:

```powershell
conda activate pot-masking
```

Validate the dense workspace and print the command without creating links or starting COLMAP:

```powershell
python scripts/reconstruction/run_colmap_dense.py --stage delaunay-mesh --dry-run
```

The dry run should report `950472` Delaunay-ready fused points. Start meshing with:

```powershell
python scripts/reconstruction/run_colmap_dense.py --stage delaunay-mesh
```

For this dataset, allow tens of minutes and be prepared for a longer run. Memory use can be substantial during triangulation and graph-cut optimization.

## Equivalent direct COLMAP command

The Python runner is recommended because it prepares the required workspace layout, checks the visibility data, logs the run, protects existing output, and validates the resulting mesh. After `fused.ply` and `fused.ply.vis` have been linked or copied to the dense workspace root, the equivalent one-line PowerShell command is:

```powershell
& "C:\Tools\COLMAP-4.1.1\bin\colmap.exe" delaunay_mesher --input_path "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential" --input_type dense --output_path "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\results\meshed-delaunay.ply" --DelaunayMeshing.max_proj_dist 20.0 --DelaunayMeshing.max_depth_dist 0.05 --DelaunayMeshing.visibility_sigma 3.0 --DelaunayMeshing.distance_sigma_factor 1.0 --DelaunayMeshing.quality_regularization 1.0 --DelaunayMeshing.max_side_length_factor 25.0 --DelaunayMeshing.max_side_length_percentile 95.0 --DelaunayMeshing.num_threads 4
```

## Expected output

Successful meshing creates:

```text
data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/
├── results/
│   ├── meshed-poisson.ply
│   └── meshed-delaunay.ply
└── logs/
    └── 06_delaunay_meshing.log
```

The runner verifies that the PLY contains meaningful vertex and face counts. Inspect both meshes in MeshLab or CloudCompare and compare:

- Pot silhouette and completeness
- Rim, inner wall, and base behavior
- Background surfaces and floating components
- Fine carved detail
- Smoothing, holes, and incorrectly closed regions

Choose the visually stronger mesh as the input for the later texture-atlas stage. A higher face count alone does not prove that one mesh is better.

## If memory or quality needs tuning

If memory use is excessive, increase `max_proj_dist` from 20 to 25 or 30 to insert fewer points into the triangulation, producing a coarser mesh. If the result is too coarse and memory remains safe, reduce it gradually. Increasing `quality_regularization` makes the result smoother; reducing it retains more local variation but can increase noise.

Preserve the first `meshed-delaunay.ply` before running tuning experiments.
