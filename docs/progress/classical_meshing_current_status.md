# Classical Meshing Pipeline — Current Status

Updated: 10 September 2026

## Summary and scope

The original 273-image, side-view classical reconstruction has reached textured
mesh output. Both the raw classical baseline and a separately cleaned, retextured
candidate are preserved. The next step is visual comparison and acceptance of
the cleaned candidate, not another reconstruction run.

This report concerns `pot1-unglazed_every6` only. It does not claim that the newer
728-image multiview/LightGlue dataset has been processed through this dense
meshing pipeline. The separate 3DGS progress report in `current_status.md` is
unchanged.

## Pipeline and completion

SAM 2 masks → sampled RGB images → masked SIFT and sequential matching → sparse
cameras → masked RGB preparation and undistortion → aligned binary fusion masks
→ PatchMatch stereo → stereo fusion → Poisson/Delaunay meshes → simplification
→ raw texturing → conservative geometry cleanup → cleaned-mesh texturing.

All listed stages have produced outputs. Completion of texturing does not by
itself establish geometric accuracy, watertightness, or artifact-free appearance.

## Inputs and selected sparse reconstruction

Paths below are relative to the repository root.

- RGB: `data/frames_output/pot1-unglazed_every6_frames`
- Masks: `data/processed/pot1-unglazed_every6/masks_colmap`
- Selected sparse model: `data/processed/pot1-unglazed_every6/colmap_sparse_masked_sequential/sparse/0`
- Selected model registered all 273 images; earlier GUI inspection showed a coherent camera orbit.

Historical sparse-run durations from [last_week_status.md](last_week_status.md):

| Strategy | Recorded whole-run duration | Decision |
| --- | --- | --- |
| Exhaustive matching reconstruction | Approximately 9 h 31 min | Preserved in `colmap_sparse_masked_exhaustion`; fragmented result |
| Sequential matching reconstruction | Approximately 22 min | Selected for the classical baseline |

These are historical approximate reconstruction-run durations, not newly measured
matcher-only timings.

## Dense workspace and configuration

Workspace: `data/processed/pot1-unglazed_every6/colmap_dense_masked_sequential`

Runner: `scripts/reconstruction/run_colmap_dense.py`

Configuration: `configs/colmap_dense_pot1_unglazed_every6.yml`

Current settings include undistortion at maximum image size 2000, PatchMatch and
fusion at 1200, geometric consistency enabled, a 6 GB cache setting, and limited
thread counts. Poisson uses depth 11 and trim 5; simplification targets 10% of
the original faces. These are the current configuration values, not a guarantee
that every earlier trial used identical settings.

Workspace locations:

- `images/`: undistorted masked RGB images
- `sparse/`: undistorted camera model
- `masks/`: aligned masks
- `stereo/depth_maps/`: PatchMatch depth maps
- `stereo/normal_maps/`: PatchMatch normal maps
- `results/`: point cloud, meshes and textured outputs
- `logs/`: per-stage execution records

## Results and runtimes

| Stage | Result | Recorded duration |
| --- | --- | --- |
| RGB undistortion | Dense image workspace | 339.080 s (5 min 39 s) |
| PatchMatch stereo | Geometric depth/normal outputs | 8,083.014 s (2 h 14 min 43 s) |
| Stereo fusion | 950,472 points | 181.858 s (3 min 2 s) |
| Poisson, trim 5 | 2,605,671 vertices / 5,208,278 faces | 75.366 s |
| Delaunay | 40,719 vertices / 81,604 faces | 615.847 s (10 min 16 s) |
| Poisson simplification | 260,723 vertices / 520,826 faces | 63.564 s |
| Raw texturing | Textured simplified Poisson mesh | 115.421 s |
| Geometry cleanup | 256,053 vertices / 511,085 faces | 5.178 s |
| Cleaned-mesh texturing | Separate textured candidate | Approximately 106.315 s |

Dense stage durations are from `logs/02_undistort_rgb.log` through
`logs/08_mesh_texturing.log`; cleanup timing is from `cleanup_report.json`.
Cleaned texturing duration is inferred from the first and final COLMAP log
timestamps (8 September, 19:50:24.304–19:52:10.620), not an explicit wall-clock
summary. These timings exclude manual work and should not be summed as an
end-to-end experimental runtime without accounting for retries and preparation.

The initial Poisson trim-10 trial produced only 36 vertices / 48 faces despite
exit code 0. It is preserved as
`results/meshed-poisson-trim10-failed-36v-48f.ply`, with its failed-quality log.
The trim-5 result replaced it as the usable baseline.

## Preserved artifacts

All paths in this section are relative to the dense workspace.

| Artifact | Path |
| --- | --- |
| Dense cloud | `results/fused.ply` and `results/fused.ply.vis` |
| Full Poisson mesh | `results/meshed-poisson.ply` |
| Delaunay comparison | `results/meshed-delaunay.ply` |
| Simplified raw geometry | `results/meshed-poisson-simplified.ply` |
| Raw classical baseline | `results/raw_classical_baseline/mesh.ply` and `texture.png` |
| Cleaned geometry | `results/cleaned_classical_candidate/meshed-poisson-cleaned.ply` |
| Cleanup audit | `results/cleaned_classical_candidate/cleanup_report.json` |
| Removed geometry | `results/cleaned_classical_candidate/removed-geometry.ply` |
| Source-face mapping | `results/cleaned_classical_candidate/kept-source-face-indices.npy` |
| Cleaned textured candidate | `results/cleaned_classical_textured/mesh.ply` and `texture.png` |

The old `results/textured` directory was renamed to `raw_classical_baseline`;
the historical texturing log still mentions its original destination.
Keep each texture atlas beside its matching mesh. Do not substitute the raw
atlas for the cleaned mesh's atlas.

## Cleanup policy and measured changes

Script: `scripts/reconstruction/cleanup_mesh.py`

Config: `configs/mesh_cleanup_pot1_unglazed_every6.yml`

The cleanup removes duplicates, faces unsupported by the fused cloud under an
approximate voxel-neighborhood test, and small connected components. It does
not smooth the ornament, fill holes, invent an underside, or apply an automatic
base-plane cut.

- Removed: 9,741 faces (1.8703% of the simplified mesh).
- Duplicate faces: 302.
- Unsupported faces: 6,876.
- Small-component faces: 2,563.
- Nonfinite/degenerate faces removed: 0 / 0.
- Retained: 511,085 faces and 256,053 vertices.
- Support voxel fraction: 0.003 of fused-cloud bounding-box diagonal.
- Minimum retained component size: 100 faces.
- Safety stop: removal exceeding 15%.

These tests are conservative heuristics, not proof that every retained face is
correct or every removed face was erroneous. The raw geometry and removal audit
remain available for review.

## Texturing status

| Version | Faces assigned to views | Atlas |
| --- | --- | --- |
| Raw baseline | 472,517 / 520,826 (90.72%) | 8192 × 4950 |
| Cleaned candidate | 484,228 / 511,085 (94.75%) | 8192 × 5020 |

The cleaned log reports `Mesh texture mapping complete`, and both output files
exist. It leaves 26,857 faces without a selected source view. Assignment
percentage is not a surface-accuracy metric or a substitute for visual review.

The cleaned log contains PowerShell `NativeCommandError` formatting around
COLMAP's stderr informational output; its final completion message and generated
files indicate that texturing proceeded. No explicit exit-code summary was
saved in that direct-command log. This status update checked the log and file
presence, not a new full binary validation or MeshLab inspection.

Earlier screenshots showed recognizable pottery shape and ornament, but also a
jagged base skirt and dark speckles. Cleanup/retexturing must still be inspected
to establish which defects improved; dark speckles are not necessarily geometry
errors and can also involve visibility, texture, or shading.

## Next step and acceptance checklist

1. Import `results/cleaned_classical_textured/mesh.ply` into MeshLab and hide the raw layer.
2. Compare raw and cleaned versions at the same view and shading settings.
3. Inspect the base skirt, rim, interior, ornament, detached fragments and texture seams.
4. Check `removed-geometry.ply` if any genuine pottery detail appears missing.
5. Save comparable screenshots and record remaining limitations before naming the candidate an accepted cleaned baseline.

Do not claim reconstruction of unseen interior/underside surfaces from the
side-only dataset. For comparison with 3DGS, distinguish raw from cleaned
classical results and use matched input data, held-out views and rendering
conditions; a newer multiview 3DGS result is not a controlled same-input comparison.

## Existing command guides

- [PatchMatch stereo](../classical_reconstruction/patchmatch_stereo.md)
- [Stereo fusion](../classical_reconstruction/stereo_fusion.md)
- [Poisson meshing](../classical_reconstruction/poisson_meshing.md)
- [Delaunay meshing](../classical_reconstruction/delaunay_meshing.md)
- [Mesh simplification](../classical_reconstruction/mesh_simplification.md)
- [Raw mesh texturing](../classical_reconstruction/mesh_texturing.md)
- [Geometry cleanup](../classical_reconstruction/geometry_cleanup.md)
- [Cleaned mesh texturing](../classical_reconstruction/cleaned_mesh_texturing.md)

The main runner's `--stage texture` still targets the original simplified mesh;
the cleaned-texturing guide uses a separate command and destination. The project
owner runs reconstruction commands. No reconstruction was launched and no Git
commands, commit, or push were performed for this status update.
