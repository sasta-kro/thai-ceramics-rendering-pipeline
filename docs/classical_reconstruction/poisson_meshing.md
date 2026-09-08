# Poisson surface meshing

Poisson meshing converts the fused oriented point cloud into a continuous triangle surface. This step uses `results/fused.ply` as its only reconstruction input; it does not rerun PatchMatch or stereo fusion.

## Prepared configuration

The settings are stored in `configs/colmap_dense_pot1_unglazed_every6.yml`:

```yaml
meshing:
  run_poisson: true
  run_delaunay: true
  poisson:
    point_weight: 1.0
    depth: 11
    color: true
    trim: 5.0
    num_threads: 4
```

Depth 11 is a memory-conscious starting point for the 950,472-point input. COLMAP's default depth 13 can require substantially more RAM. Color is retained in the output mesh. A first run with the default trim value of 10 retained only 36 vertices and 48 faces, so this dataset uses trim 5 to preserve more of the reconstructed surface.

## Run from PowerShell

Open PowerShell in the repository root and activate the project environment:

```powershell
conda activate pot-masking
```

Validate the fused point cloud and print the exact COLMAP command without starting meshing:

```powershell
python scripts/reconstruction/run_colmap_dense.py --stage mesh --dry-run
```

The dry run should report `950472` Poisson-ready fused points. Then start Poisson meshing:

```powershell
python scripts/reconstruction/run_colmap_dense.py --stage mesh
```

### Retry after the trim-10 result

The first trim-10 attempt exited successfully but produced only 36 vertices and 48 faces. Preserve that diagnostic result and its log before rerunning:

```powershell
Move-Item -LiteralPath "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\results\meshed-poisson.ply" -Destination "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\results\meshed-poisson-trim10-failed-36v-48f.ply"
Move-Item -LiteralPath "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\logs\05_poisson_meshing.log" -Destination "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\logs\05_poisson_meshing_trim10_failed.log"
python scripts/reconstruction/run_colmap_dense.py --stage mesh
```

These moves preserve the failed attempt and free the standard output path for the trim-5 retry.

Poisson meshing is CPU- and RAM-dependent. At depth 11, expect anything from several minutes to tens of minutes on this dataset; slower machines can take longer.

## Equivalent direct COLMAP command

The Python runner is recommended because it validates the input normals and colors, records a log, protects an existing result, and verifies that the output has both vertices and faces. The equivalent one-line PowerShell command is:

```powershell
& "C:\Tools\COLMAP-4.1.1\bin\colmap.exe" poisson_mesher --input_path "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\results\fused.ply" --output_path "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\results\meshed-poisson.ply" --PoissonMeshing.point_weight 1.0 --PoissonMeshing.depth 11 --PoissonMeshing.color 1 --PoissonMeshing.trim 5.0 --PoissonMeshing.num_threads 4
```

The direct COLMAP command does not require the Conda environment, but it does not perform the runner's validation or create the project runtime log.

## Expected output

Successful meshing creates:

```text
data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/
├── results/
│   ├── fused.ply
│   └── meshed-poisson.ply
└── logs/
    └── 05_poisson_meshing.log
```

The runner should finish with exit code 0 and print meaningful mesh vertex and face counts. For this input, it rejects results below 950 vertices or faces as implausibly small. Open `meshed-poisson.ply` in MeshLab or CloudCompare and inspect:

- Whether the pottery silhouette and carved surface remain recognizable
- Whether the rim, inner wall, base, or other weakly observed regions were incorrectly closed
- Floating components, stretched triangles, or background surfaces
- Excessive smoothing of fine details

Poisson reconstruction naturally fills gaps and can create surfaces in unseen regions. Treat a closed-looking top, bottom, or interior as an interpolation unless those areas were genuinely visible in the source cameras.

## If memory or quality needs tuning

If the process runs out of RAM, lower `meshing.poisson.depth` from 11 to 10. If depth 11 succeeds but lacks detail and ample RAM remains, preserve the first mesh and test depth 12 separately. Each increase in octree depth can raise memory and runtime considerably.

Do not overwrite a useful baseline while tuning. Move or rename `meshed-poisson.ply` first, then rerun with the revised YAML value.
