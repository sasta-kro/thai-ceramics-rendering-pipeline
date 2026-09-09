# Texture the cleaned classical candidate

The cleanup report confirms 256,053 vertices and 511,085 faces remain. This run
uses the same calibrated images and texturing settings as the raw baseline.
The existing Python runner's `--stage texture` still targets the original mesh;
use the command below to texture the cleaned geometry instead.

From PowerShell in the repository root, paste this block:

```powershell
$dense = 'data\processed\pot1-unglazed_every6\colmap_dense_masked_sequential'
$cleanMesh = Join-Path $dense 'results\cleaned_classical_candidate\meshed-poisson-cleaned.ply'
$cleanTexture = Join-Path $dense 'results\cleaned_classical_textured'
$cleanLog = Join-Path $dense 'logs\09_cleaned_mesh_texturing.log'
if (!(Test-Path -LiteralPath $cleanMesh)) { throw "Cleaned mesh missing: $cleanMesh" }
if (Test-Path -LiteralPath $cleanTexture) { throw "Output already exists; preserve it and choose a new output directory: $cleanTexture" }
& 'C:\Tools\COLMAP-4.1.1\bin\colmap.exe' mesh_texturer --workspace_path $dense --input_path $cleanMesh --output_path $cleanTexture --output_type BIN --MeshTextureMapping.min_cos_normal_angle 0.1 --MeshTextureMapping.min_visible_vertices 3 --MeshTextureMapping.view_selection_smoothing_iterations 3 --MeshTextureMapping.atlas_patch_padding 4 --MeshTextureMapping.inpaint_radius 5 --MeshTextureMapping.apply_color_correction 1 --MeshTextureMapping.color_correction_regularization 0.1 --MeshTextureMapping.num_threads 4 --MeshTextureMapping.texture_scale_factor 1.0 2>&1 | Tee-Object -FilePath $cleanLog
if ($LASTEXITCODE -ne 0) { throw "COLMAP failed with exit code $LASTEXITCODE. Inspect $cleanLog" }
```

This directly runs COLMAP, so activating `pot-masking` is optional. Allow a few
minutes; the raw baseline texture run took 115 seconds, but runtime can differ.

Expected outputs under the dense workspace:

```text
results/cleaned_classical_textured/mesh.ply
results/cleaned_classical_textured/texture.png
logs/09_cleaned_mesh_texturing.log
```

Open the new `mesh.ply` with MeshLab's File > Import Mesh. Keep the atlas beside
the PLY and hide the raw model layer during inspection. The mesh should retain
511,085 faces. Review the log's assigned-face count and inspect the foot, rim,
interior, seams, and black speckles. A successful exit alone does not establish
texture quality. The old atlas cannot simply be reused after faces were removed.

The raw textured result remains in `results/raw_classical_baseline/`. Retain both
versions and report the cleanup when comparing classical reconstruction with 3DGS.
If a run fails, preserve its output directory and log before retrying under new names.
