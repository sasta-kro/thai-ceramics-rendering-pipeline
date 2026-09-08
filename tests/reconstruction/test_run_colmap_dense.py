from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "reconstruction"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_colmap_dense  # noqa: E402


class DenseColmapRunnerTests(unittest.TestCase):
    def test_expected_mask_uses_colmap_double_extension(self) -> None:
        image_root = Path("images")
        mask_root = Path("masks")
        image_path = image_root / "nested" / "frame_000000.jpg"

        self.assertEqual(
            run_colmap_dense.expected_mask_path(image_path, image_root, mask_root),
            mask_root / "nested" / "frame_000000.jpg.png",
        )

    def test_output_child_cannot_escape_dense_workspace(self) -> None:
        workspace = run_colmap_dense.PROJECT_ROOT / "data" / "processed" / "dense"
        with self.assertRaises(run_colmap_dense.PipelineError):
            run_colmap_dense.resolve_output_child(
                workspace, "../outside.ply", "output.fused_point_cloud"
            )

    def test_sparse_component_validation_accepts_binary_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sparse_model = Path(directory)
            for component in run_colmap_dense.SPARSE_MODEL_COMPONENTS:
                (sparse_model / f"{component}.bin").write_bytes(b"model")

            run_colmap_dense.validate_sparse_components(sparse_model)

    def test_sparse_component_validation_reports_missing_component(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sparse_model = Path(directory)
            (sparse_model / "cameras.bin").write_bytes(b"model")
            (sparse_model / "images.bin").write_bytes(b"model")

            with self.assertRaisesRegex(
                run_colmap_dense.PipelineError, "points3D"
            ):
                run_colmap_dense.validate_sparse_components(sparse_model)

    def test_reads_registered_names_from_colmap_text_model(self) -> None:
        contents = """# Image list with two lines of data per image:
# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
1 1 0 0 0 0 0 0 1 frame_000000.jpg
10.0 20.0 -1
2 1 0 0 0 0 0 0 1 nested/frame_000006.jpg

"""
        with tempfile.TemporaryDirectory() as directory:
            images_txt = Path(directory) / "images.txt"
            images_txt.write_text(contents, encoding="utf-8")

            self.assertEqual(
                run_colmap_dense.read_text_model_image_names(images_txt),
                ["frame_000000.jpg", "nested/frame_000006.jpg"],
            )

    def test_stage_selection_preserves_dense_order(self) -> None:
        self.assertEqual(
            run_colmap_dense.selected_stages("all"),
            run_colmap_dense.PIPELINE_STAGES,
        )
        self.assertEqual(
            run_colmap_dense.selected_stages("validate"), ("validate",)
        )

    def test_prepare_image_pair_masks_background_and_preserves_name(self) -> None:
        from PIL import Image

        settings = run_colmap_dense.MaskPreparationSettings(
            background_value=0,
            threshold=128,
            reconstruction_dilation_pixels=0,
            jpeg_quality=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_image = root / "source" / "frame_000000.png"
            source_mask = root / "masks" / "frame_000000.png.png"
            masked_output = root / "prepared_rgb" / "frame_000000.png"
            mask_output = root / "prepared_masks" / "frame_000000.png"
            source_image.parent.mkdir()
            source_mask.parent.mkdir()

            rgb = Image.new("RGB", (5, 5), (200, 50, 25))
            mask = Image.new("L", (5, 5), 0)
            mask.putpixel((2, 2), 255)
            rgb.save(source_image)
            mask.save(source_mask)
            rgb.close()
            mask.close()

            run_colmap_dense.prepare_image_pair(
                source_image,
                source_mask,
                masked_output,
                mask_output,
                settings,
            )

            with Image.open(masked_output) as prepared_rgb:
                self.assertEqual(prepared_rgb.getpixel((0, 0)), (0, 0, 0))
                self.assertEqual(prepared_rgb.getpixel((2, 2)), (200, 50, 25))
            with Image.open(mask_output) as prepared_mask:
                self.assertEqual(prepared_mask.convert("L").getpixel((0, 0)), 0)
                self.assertEqual(prepared_mask.convert("L").getpixel((2, 2)), 255)

            self.assertTrue(
                run_colmap_dense.prepared_pair_is_valid(
                    masked_output, mask_output, (5, 5)
                )
            )

    def test_undistortion_settings_are_loaded_for_colmap_workspace(self) -> None:
        settings = run_colmap_dense.undistortion_settings(
            {
                "undistortion": {
                    "output_type": "COLMAP",
                    "max_image_size": 2000,
                    "jpeg_quality": 100,
                    "num_threads": 2,
                }
            }
        )

        self.assertEqual(settings.output_type, "COLMAP")
        self.assertEqual(settings.max_image_size, 2000)
        self.assertEqual(settings.jpeg_quality, 100)
        self.assertEqual(settings.num_threads, 2)

    def test_non_colmap_undistortion_workspace_is_rejected(self) -> None:
        with self.assertRaisesRegex(run_colmap_dense.PipelineError, "must be COLMAP"):
            run_colmap_dense.undistortion_settings(
                {"undistortion": {"output_type": "PMVS"}}
            )

    def test_patch_match_settings_use_memory_conscious_defaults(self) -> None:
        settings = run_colmap_dense.patch_match_settings(
            {
                "patch_match": {
                    "gpu_index": 0,
                    "max_image_size": 1600,
                    "cache_size_gb": 6,
                    "num_threads": 2,
                    "geom_consistency": True,
                    "filter": True,
                }
            }
        )

        self.assertEqual(settings.gpu_index, 0)
        self.assertEqual(settings.max_image_size, 1600)
        self.assertEqual(settings.cache_size_gb, 6.0)
        self.assertEqual(settings.num_threads, 2)
        self.assertTrue(settings.geom_consistency)
        self.assertTrue(settings.filter)

    def test_patch_match_command_is_an_argument_list(self) -> None:
        root = Path("C:/Project With Parentheses (CV)")
        paths = run_colmap_dense.DensePaths(
            config=root / "config.yml",
            colmap=Path("C:/Tools/COLMAP/bin/colmap.exe"),
            source_images=root / "source_images",
            source_masks=root / "source_masks",
            sparse_model=root / "source_sparse",
            workspace=root / "dense",
            masked_images=root / "masked",
            mask_undistortion_images=root / "mask_inputs",
            undistorted_images=root / "dense" / "images",
            undistorted_masks=root / "dense" / "masks",
            fused_point_cloud=root / "dense" / "results" / "fused.ply",
            poisson_mesh=root / "dense" / "results" / "poisson.ply",
            simplified_poisson_mesh=root / "dense" / "results" / "poisson-simple.ply",
            delaunay_mesh=root / "dense" / "results" / "delaunay.ply",
            textured_mesh=root / "dense" / "results" / "textured",
            logs=root / "dense" / "logs",
            run_manifest=root / "dense" / "run_manifest.json",
        )
        settings = run_colmap_dense.PatchMatchSettings(
            gpu_index=0,
            max_image_size=1600,
            cache_size_gb=6,
            num_threads=2,
            geom_consistency=True,
            filter=True,
        )

        command = run_colmap_dense.build_patch_match_arguments(paths, settings)

        self.assertEqual(command[0], str(paths.colmap))
        self.assertEqual(command[1], "patch_match_stereo")
        self.assertIn(str(paths.workspace), command)
        self.assertIn("--PatchMatchStereo.max_image_size", command)
        self.assertIn("1600", command)

    def test_fusion_settings_are_loaded(self) -> None:
        settings = run_colmap_dense.fusion_settings(
            {
                "fusion": {
                    "input_type": "geometric",
                    "max_image_size": 1200,
                    "cache_size_gb": 6,
                    "num_threads": 4,
                    "min_num_pixels": 3,
                }
            }
        )

        self.assertEqual(settings.input_type, "geometric")
        self.assertEqual(settings.max_image_size, 1200)
        self.assertEqual(settings.cache_size_gb, 6.0)
        self.assertEqual(settings.num_threads, 4)
        self.assertEqual(settings.min_num_pixels, 3)

    def test_fusion_command_uses_geometric_maps_and_masks(self) -> None:
        root = Path("C:/Project With Parentheses (CV)")
        paths = run_colmap_dense.DensePaths(
            config=root / "config.yml",
            colmap=Path("C:/Tools/COLMAP/bin/colmap.exe"),
            source_images=root / "source_images",
            source_masks=root / "source_masks",
            sparse_model=root / "source_sparse",
            workspace=root / "dense",
            masked_images=root / "masked",
            mask_undistortion_images=root / "mask_inputs",
            undistorted_images=root / "dense" / "images",
            undistorted_masks=root / "dense" / "masks",
            fused_point_cloud=root / "dense" / "results" / "fused.ply",
            poisson_mesh=root / "dense" / "results" / "poisson.ply",
            simplified_poisson_mesh=root / "dense" / "results" / "poisson-simple.ply",
            delaunay_mesh=root / "dense" / "results" / "delaunay.ply",
            textured_mesh=root / "dense" / "results" / "textured",
            logs=root / "dense" / "logs",
            run_manifest=root / "dense" / "run_manifest.json",
        )
        settings = run_colmap_dense.FusionSettings(
            input_type="geometric",
            max_image_size=1200,
            cache_size_gb=6,
            num_threads=4,
            min_num_pixels=3,
        )

        command = run_colmap_dense.build_fusion_arguments(paths, settings)

        self.assertEqual(command[:2], [str(paths.colmap), "stereo_fusion"])
        self.assertEqual(command[command.index("--input_type") + 1], "geometric")
        self.assertEqual(
            command[command.index("--StereoFusion.mask_path") + 1],
            str(paths.undistorted_masks),
        )
        self.assertEqual(
            command[command.index("--StereoFusion.max_image_size") + 1], "1200"
        )
        self.assertEqual(
            command[command.index("--StereoFusion.min_num_pixels") + 1], "3"
        )

    def test_ply_vertex_count_reads_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            point_cloud = Path(directory) / "fused.ply"
            point_cloud.write_bytes(
                b"ply\nformat binary_little_endian 1.0\n"
                b"element vertex 28491\nproperty float x\nend_header\n"
            )

            self.assertEqual(run_colmap_dense.ply_vertex_count(point_cloud), 28491)

    def test_poisson_meshing_settings_are_loaded(self) -> None:
        settings = run_colmap_dense.poisson_meshing_settings(
            {
                "meshing": {
                    "run_poisson": True,
                    "poisson": {
                        "point_weight": 1.0,
                        "depth": 11,
                        "color": True,
                        "trim": 10.0,
                        "num_threads": 4,
                    },
                }
            }
        )

        self.assertEqual(settings.point_weight, 1.0)
        self.assertEqual(settings.depth, 11)
        self.assertTrue(settings.color)
        self.assertEqual(settings.trim, 10.0)
        self.assertEqual(settings.num_threads, 4)

    def test_poisson_meshing_command_uses_fused_cloud(self) -> None:
        root = Path("C:/Project With Parentheses (CV)")
        paths = run_colmap_dense.DensePaths(
            config=root / "config.yml",
            colmap=Path("C:/Tools/COLMAP/bin/colmap.exe"),
            source_images=root / "source_images",
            source_masks=root / "source_masks",
            sparse_model=root / "source_sparse",
            workspace=root / "dense",
            masked_images=root / "masked",
            mask_undistortion_images=root / "mask_inputs",
            undistorted_images=root / "dense" / "images",
            undistorted_masks=root / "dense" / "masks",
            fused_point_cloud=root / "dense" / "results" / "fused.ply",
            poisson_mesh=root / "dense" / "results" / "poisson.ply",
            simplified_poisson_mesh=root / "dense" / "results" / "poisson-simple.ply",
            delaunay_mesh=root / "dense" / "results" / "delaunay.ply",
            textured_mesh=root / "dense" / "results" / "textured",
            logs=root / "dense" / "logs",
            run_manifest=root / "dense" / "run_manifest.json",
        )
        settings = run_colmap_dense.PoissonMeshingSettings(
            point_weight=1.0,
            depth=11,
            color=True,
            trim=10.0,
            num_threads=4,
        )

        command = run_colmap_dense.build_poisson_meshing_arguments(paths, settings)

        self.assertEqual(command[:2], [str(paths.colmap), "poisson_mesher"])
        self.assertEqual(
            command[command.index("--input_path") + 1], str(paths.fused_point_cloud)
        )
        self.assertEqual(
            command[command.index("--output_path") + 1], str(paths.poisson_mesh)
        )
        self.assertEqual(command[command.index("--PoissonMeshing.depth") + 1], "11")
        self.assertEqual(command[command.index("--PoissonMeshing.color") + 1], "1")

    def test_ply_mesh_counts_require_vertices_and_faces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory) / "mesh.ply"
            mesh.write_bytes(
                b"ply\nformat binary_little_endian 1.0\n"
                b"element vertex 1200\nproperty float x\n"
                b"element face 2300\nproperty list uchar int vertex_indices\n"
                b"end_header\n"
            )

            self.assertEqual(run_colmap_dense.ply_mesh_counts(mesh), (1200, 2300))

    def test_poisson_mesh_validation_rejects_implausibly_small_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory) / "tiny-mesh.ply"
            mesh.write_bytes(
                b"ply\nformat binary_little_endian 1.0\n"
                b"element vertex 36\nproperty float x\n"
                b"element face 48\nproperty list uchar int vertex_indices\n"
                b"end_header\n"
            )

            with self.assertRaisesRegex(
                run_colmap_dense.PipelineError, "implausibly small"
            ):
                run_colmap_dense.validate_poisson_mesh(mesh, 950472)

    def test_delaunay_meshing_settings_are_loaded(self) -> None:
        settings = run_colmap_dense.delaunay_meshing_settings(
            {
                "meshing": {
                    "run_delaunay": True,
                    "delaunay": {
                        "input_type": "dense",
                        "max_proj_dist": 20.0,
                        "max_depth_dist": 0.05,
                        "visibility_sigma": 3.0,
                        "distance_sigma_factor": 1.0,
                        "quality_regularization": 1.0,
                        "max_side_length_factor": 25.0,
                        "max_side_length_percentile": 95.0,
                        "num_threads": 4,
                    },
                }
            }
        )

        self.assertEqual(settings.input_type, "dense")
        self.assertEqual(settings.max_proj_dist, 20.0)
        self.assertEqual(settings.max_depth_dist, 0.05)
        self.assertEqual(settings.quality_regularization, 1.0)
        self.assertEqual(settings.max_side_length_percentile, 95.0)
        self.assertEqual(settings.num_threads, 4)

    def test_delaunay_meshing_command_uses_dense_workspace(self) -> None:
        root = Path("C:/Project With Parentheses (CV)")
        paths = run_colmap_dense.DensePaths(
            config=root / "config.yml",
            colmap=Path("C:/Tools/COLMAP/bin/colmap.exe"),
            source_images=root / "source_images",
            source_masks=root / "source_masks",
            sparse_model=root / "source_sparse",
            workspace=root / "dense",
            masked_images=root / "masked",
            mask_undistortion_images=root / "mask_inputs",
            undistorted_images=root / "dense" / "images",
            undistorted_masks=root / "dense" / "masks",
            fused_point_cloud=root / "dense" / "results" / "fused.ply",
            poisson_mesh=root / "dense" / "results" / "poisson.ply",
            simplified_poisson_mesh=root / "dense" / "results" / "poisson-simple.ply",
            delaunay_mesh=root / "dense" / "results" / "delaunay.ply",
            textured_mesh=root / "dense" / "results" / "textured",
            logs=root / "dense" / "logs",
            run_manifest=root / "dense" / "run_manifest.json",
        )
        settings = run_colmap_dense.DelaunayMeshingSettings(
            input_type="dense",
            max_proj_dist=20.0,
            max_depth_dist=0.05,
            visibility_sigma=3.0,
            distance_sigma_factor=1.0,
            quality_regularization=1.0,
            max_side_length_factor=25.0,
            max_side_length_percentile=95.0,
            num_threads=4,
        )

        command = run_colmap_dense.build_delaunay_meshing_arguments(paths, settings)

        self.assertEqual(command[:2], [str(paths.colmap), "delaunay_mesher"])
        self.assertEqual(command[command.index("--input_path") + 1], str(paths.workspace))
        self.assertEqual(command[command.index("--input_type") + 1], "dense")
        self.assertEqual(
            command[command.index("--output_path") + 1], str(paths.delaunay_mesh)
        )
        self.assertEqual(
            command[command.index("--DelaunayMeshing.num_threads") + 1], "4"
        )

    def test_prepare_delaunay_workspace_preserves_fusion_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            fused = results / "fused.ply"
            visibility = results / "fused.ply.vis"
            fused.write_bytes(b"point cloud")
            visibility.write_bytes(b"visibility")
            paths = type(
                "Paths",
                (),
                {"workspace": root, "fused_point_cloud": fused},
            )()

            run_colmap_dense.prepare_delaunay_workspace(paths)

            self.assertEqual((root / "fused.ply").read_bytes(), b"point cloud")
            self.assertEqual((root / "fused.ply.vis").read_bytes(), b"visibility")
            self.assertEqual(fused.read_bytes(), b"point cloud")
            self.assertEqual(visibility.read_bytes(), b"visibility")

    def test_mesh_simplification_settings_are_loaded(self) -> None:
        settings = run_colmap_dense.mesh_simplification_settings(
            {
                "meshing": {
                    "simplification": {
                        "target_face_ratio": 0.1,
                        "max_error": 0.0,
                        "boundary_weight": 1000.0,
                        "interpolate_colors": True,
                        "num_threads": 4,
                    }
                }
            }
        )

        self.assertEqual(settings.target_face_ratio, 0.1)
        self.assertEqual(settings.max_error, 0.0)
        self.assertEqual(settings.boundary_weight, 1000.0)
        self.assertTrue(settings.interpolate_colors)
        self.assertEqual(settings.num_threads, 4)

    def test_mesh_simplification_command_preserves_input_mesh(self) -> None:
        paths = SimpleNamespace(
            colmap=Path("C:/Tools/COLMAP/bin/colmap.exe"),
            poisson_mesh=Path("C:/Project/results/meshed-poisson.ply"),
            simplified_poisson_mesh=Path(
                "C:/Project/results/meshed-poisson-simplified.ply"
            ),
        )
        settings = run_colmap_dense.MeshSimplificationSettings(
            target_face_ratio=0.1,
            max_error=0.0,
            boundary_weight=1000.0,
            interpolate_colors=True,
            num_threads=4,
        )

        command = run_colmap_dense.build_mesh_simplification_arguments(
            paths, settings
        )

        self.assertEqual(command[:2], [str(paths.colmap), "mesh_simplifier"])
        self.assertEqual(command[command.index("--input_path") + 1], str(paths.poisson_mesh))
        self.assertEqual(
            command[command.index("--output_path") + 1],
            str(paths.simplified_poisson_mesh),
        )
        self.assertEqual(
            command[command.index("--MeshSimplification.target_face_ratio") + 1],
            "0.1",
        )

    def test_simplified_mesh_must_reduce_face_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory) / "not-simplified.ply"
            mesh.write_bytes(
                b"ply\nformat binary_little_endian 1.0\n"
                b"element vertex 100\nproperty float x\n"
                b"element face 200\nproperty list uchar int vertex_indices\n"
                b"end_header\n"
            )

            with self.assertRaisesRegex(
                run_colmap_dense.PipelineError, "did not reduce"
            ):
                run_colmap_dense.validate_simplified_mesh(mesh, 200)


if __name__ == "__main__":
    unittest.main()
