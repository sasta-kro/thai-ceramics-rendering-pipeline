# Calibrated mesh texturing

This stage projects the 273 masked, undistorted RGB images onto the selected simplified Poisson surface. COLMAP chooses suitable camera views for the mesh faces, generates per-face UV coordinates, and bakes the appearance into a texture atlas.

## Selected inputs

```text
Dense workspace:
data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential

Mesh:
results/meshed-poisson-simplified.ply

Input mesh size:
260,723 vertices and 520,826 faces

Calibrated images:
273 masked, undistorted images in images/
```

The full Poisson mesh and the Delaunay mesh remain unchanged.

## Prepared configuration

The settings are stored in `configs/colmap_dense_pot1_unglazed_every6.yml`:

```yaml
texturing:
  output_type: BIN
  min_cos_normal_angle: 0.1
  min_visible_vertices: 3
  view_selection_smoothing_iterations: 3
  atlas_patch_padding: 4
  inpaint_radius: 5
  apply_color_correction: true
  color_correction_regularization: 0.1
  texture_scale_factor: 1.0
  num_threads: 4
```

Full texture resolution is retained. Four pixels of patch padding and a five-pixel inpainting radius help reduce atlas-edge seams. Color correction compensates for moderate exposure differences between source frames.

## Run from PowerShell

Open PowerShell in the repository root and activate the project environment:

```powershell
conda activate pot-masking
```

Validate the mesh, all calibrated images, and the dense workspace while printing the command without running COLMAP:

```powershell
python scripts/reconstruction/run_colmap_dense.py --stage texture --dry-run
```

The dry run should report:

```text
Texture-source images: 273
Texture-input mesh: 260723 vertices, 520826 faces
```

Start texture mapping with:

```powershell
python scripts/reconstruction/run_colmap_dense.py --stage texture
```

This operation uses the CPU and can consume substantial RAM while selecting views and building the atlas. Allow several minutes to an hour, with longer runtimes possible depending on the machine and atlas size.

## Equivalent direct COLMAP command

The Python runner is recommended because it validates every prerequisite, protects existing texture output, records a runtime log, and checks the generated mesh, UV coordinates, texture reference, and PNG atlas. The equivalent one-line PowerShell command is:

```powershell
& "C:\Tools\COLMAP-4.1.1\bin\colmap.exe" mesh_texturer --workspace_path "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential" --input_path "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\results\meshed-poisson-simplified.ply" --output_path "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\results\raw_classical_baseline" --output_type BIN --MeshTextureMapping.min_cos_normal_angle 0.1 --MeshTextureMapping.min_visible_vertices 3 --MeshTextureMapping.view_selection_smoothing_iterations 3 --MeshTextureMapping.atlas_patch_padding 4 --MeshTextureMapping.inpaint_radius 5 --MeshTextureMapping.apply_color_correction 1 --MeshTextureMapping.color_correction_regularization 0.1 --MeshTextureMapping.num_threads 4 --MeshTextureMapping.texture_scale_factor 1.0
```

The direct command does not require the Conda environment, but it does not provide the runner's additional validation and logging.

## Expected output

Successful texturing creates:

```text
data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/
├── results/
│   └── raw_classical_baseline/
│       ├── mesh.ply
│       └── texture.png
└── logs/
    └── 08_mesh_texturing.log
```

The runner verifies that:

- The textured mesh retains all 520,826 input faces
- The PLY contains per-face UV coordinates
- The PLY references `texture.png`
- The texture atlas is a readable, nonempty PNG

## Inspect in MeshLab

In MeshLab, choose **File → Import Mesh** and open:

```text
data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/results/raw_classical_baseline/mesh.ply
```

Keep `mesh.ply` and `texture.png` in the same directory. MeshLab should load the atlas automatically. Unused black regions inside the atlas image are normal; inspect the pottery itself for seams, incorrect camera projections, blurred areas, and black patches on visible faces.

## If a run is interrupted

The runner will not overwrite a partial nonempty output directory. Preserve it before retrying:

```powershell
Move-Item -LiteralPath "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\results\raw_classical_baseline" -Destination "data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential\results\raw_classical_baseline-partial"
python scripts/reconstruction/run_colmap_dense.py --stage texture
```

Use a different destination name if `raw_classical_baseline-partial` already exists.
