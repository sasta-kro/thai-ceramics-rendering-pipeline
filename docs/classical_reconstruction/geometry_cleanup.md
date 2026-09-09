# Classical geometry cleanup

The current textured output is now named `results/raw_classical_baseline/`.
It contains the original `mesh.ply` and `texture.png`, unchanged by the rename.
This is the raw classical baseline, before any cleanup, for comparison with 3DGS.
The original simplified Poisson mesh remains the cleanup input.

## Run

From the repository root, with `pot-masking` active, inspect the proposed changes:

```powershell
python scripts/reconstruction/cleanup_mesh.py --dry-run
```

Then generate the candidate:

```powershell
python scripts/reconstruction/cleanup_mesh.py
```

Settings are in `configs/mesh_cleanup_pot1_unglazed_every6.yml`. NumPy and PyYAML
are already dependencies of `pot-masking`; no extra installation is required.

## Method and limitations

The script removes nonfinite/degenerate faces, duplicate triangles (including
opposite winding duplicates), unsupported faces, and vertex-connected components
with fewer than 100 faces. Unreferenced vertices are removed during export.
It preserves the positions and winding of retained triangles. It does not smooth
carving, infer a base plane, repair non-manifold edges, or fill the pot opening.

Support is measured against the fused point cloud using a voxel grid. Cell width
is 0.003 times the cloud bounding-box diagonal; each occupied cell and its 26
neighbors count as supported. A triangle is removed only when all three vertices
lack support. This is an approximate, conservative support test, not an exact
surface distance or proof that geometry is wrong. It can leave some skirt geometry
and may open boundaries in sparsely observed areas. Inspect the candidate before
tightening this threshold or using it as the final cleaned model.

The real-data dry run retained 511,085 of 520,826 faces (98.13%). It proposed removing
302 duplicate faces, 6,876 unsupported faces, and 2,563 faces in tiny fragments.
The script stops before writing if more than 15% would be removed.

## Outputs and review

Under `data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential/results/`:

```text
raw_classical_baseline/
    mesh.ply
    texture.png
cleaned_classical_candidate/
    meshed-poisson-cleaned.ply
    removed-geometry.ply
    kept-source-face-indices.npy
    cleanup_report.json
```

Import `meshed-poisson-cleaned.ply` into MeshLab. Compare with the original
`meshed-poisson-simplified.ply` at the same viewpoint, especially the foot, rim,
and interior. Import `removed-geometry.ply` separately to inspect what was removed.
The cleaned mesh will be gray; no texture has been generated for its new topology.

The JSON report records settings, counts, input/output SHA256 hashes and runtime.
The NPY array maps retained faces to their original face indices. Existing candidate
directories are never overwritten; choose another output directory in the YAML
for a new experiment.

After visual review, the next step is to texture the cleaned mesh into a separate
directory. The existing dense runner still targets the raw baseline, so do not
rerun its texture stage expecting the new mesh to be used. Keep the raw baseline
and cleaned result separate in evaluations and document the cleanup applied.

Black body speckles can also be texture-projection artifacts. Geometry cleanup
alone is not evidence that those have been resolved.
