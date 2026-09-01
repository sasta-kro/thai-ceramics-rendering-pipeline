from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
