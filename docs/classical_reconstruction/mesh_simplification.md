# Poisson mesh simplification

The successful Poisson mesh contains 5,208,278 faces, which is unnecessarily heavy for interactive inspection and texture-atlas generation. This stage uses COLMAP's quadric-error mesh simplifier to create a separate mesh at approximately 10% of the original face count while preserving the full-resolution mesh unchanged.

The Delaunay mesh is also preserved as a lightweight alternative. The simplified Poisson mesh is the initial texturing candidate because it retains substantially more surface detail than the 81,604-face Delaunay result.

## Prepared configuration

The settings are stored in `configs/colmap_dense_pot1_unglazed_every6.yml`:

```yaml
output:
  simplified_poisson_mesh: results/meshed-poisson-simplified.ply

meshing:
  simplification:
    target_face_ratio: 0.1
    max_error: 0.0
    boundary_weight: 1000.0
    interpolate_colors: true
    num_threads: 4
```

The target is approximately 520,827 faces. `max_error: 0` disables an additional error cutoff, and the high boundary weight protects open boundaries. Although color interpolation is requested, COLMAP 4.1.1 writes this simplified Poisson result as geometry-only PLY; appearance will be supplied by the later texture atlas.

## Run from PowerShell

Open PowerShell in the repository root and activate the project environment:

```powershell
conda activate pot-masking
```

Validate the original Poisson mesh and print the command without starting simplification:

```powershell
python scripts/reconstruction/run_colmap_dense.py --stage simplify-mesh --dry-run
```

The dry run should report 2,605,671 vertices, 5,208,278 faces, and an expected target of approximately 520,827 faces. Start simplification with:

```powershell
python scripts/reconstruction/run_colmap_dense.py --stage simplify-mesh
```

This operation is CPU- and RAM-dependent but should normally be much shorter than PatchMatch. Keep the terminal open until it reports the achieved face ratio.

## Equivalent direct COLMAP command

The Python runner is recommended because it validates the source mesh, protects existing output, records a log, and verifies that the output actually has fewer faces. The equivalent one-line PowerShell command is:

```powershell
& "C:\Tools\COLMAP-4.1.1\bin\colmap.exe" mesh_simplifier --input_path "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\results\meshed-poisson.ply" --output_path "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\results\meshed-poisson-simplified.ply" --MeshSimplification.target_face_ratio 0.1 --MeshSimplification.max_error 0.0 --MeshSimplification.boundary_weight 1000.0 --MeshSimplification.interpolate_colors 1 --MeshSimplification.num_threads 4
```

## Expected output

Successful simplification creates:

```text
data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/
├── results/
│   ├── meshed-poisson.ply
│   ├── meshed-poisson-simplified.ply
│   └── meshed-delaunay.ply
└── logs/
    └── 07_mesh_simplification.log
```

The runner prints the simplified vertex count, face count, and achieved face ratio. Inspect the simplified mesh alongside the full Poisson and Delaunay versions in MeshLab or CloudCompare. Check that the silhouette, rim, carved ornament, and open boundaries remain acceptable and that simplification has not introduced collapsed or stretched regions.

This stage does not create a texture atlas. The generated PLY contains geometry only, so texture generation is the following stage after the simplified mesh passes visual inspection.

## If more detail is needed

Preserve the first simplified output before tuning. Increase `target_face_ratio` to 0.2 for approximately one million faces if the 10% version visibly loses important detail. Reduce it to 0.05 only if a smaller, faster showcase asset is more important than fine geometry.
