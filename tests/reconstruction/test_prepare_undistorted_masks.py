from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "reconstruction"
sys.path.insert(0, str(SCRIPTS_DIR))

import prepare_undistorted_masks  # noqa: E402


class PrepareUndistortedMasksTests(unittest.TestCase):
    def test_fusion_mask_keeps_image_extension_and_adds_png(self) -> None:
        dense_root = Path("images")
        output_root = Path("masks")
        dense_image = dense_root / "nested" / "frame_000000.jpg"

        self.assertEqual(
            prepare_undistorted_masks.fusion_mask_path(
                dense_image, dense_root, output_root
            ),
            output_root / "nested" / "frame_000000.jpg.png",
        )

    def test_binary_conversion_and_validation(self) -> None:
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense_root = root / "dense_images"
            source_root = root / "undistorted_mask_images"
            output_root = root / "masks"
            dense_root.mkdir()
            source_root.mkdir()

            dense_path = dense_root / "frame_000000.jpg"
            source_path = source_root / "frame_000000.jpg"
            dense_image = Image.new("RGB", (9, 9), (200, 80, 30))
            source_image = Image.new("L", (9, 9), 0)
            draw = ImageDraw.Draw(source_image)
            draw.rectangle((2, 2, 6, 6), fill=255)
            dense_image.save(dense_path, quality=100)
            source_image.save(source_path, quality=100)
            dense_image.close()
            source_image.close()

            expected, written, resumed = prepare_undistorted_masks.convert_masks(
                source_root,
                dense_root,
                output_root,
                threshold=128,
                erosion_pixels=1,
                resume=True,
            )

            output_path = output_root / "frame_000000.jpg.png"
            self.assertEqual((expected, written, resumed), (1, 1, 0))
            self.assertTrue(output_path.is_file())
            self.assertTrue(
                prepare_undistorted_masks.mask_is_valid(output_path, (9, 9))
            )
            self.assertEqual(
                prepare_undistorted_masks.validate_masks(dense_root, output_root),
                1,
            )

    def test_resume_reuses_valid_mask(self) -> None:
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense_root = root / "dense_images"
            source_root = root / "source"
            output_root = root / "output"
            dense_root.mkdir()
            source_root.mkdir()

            dense = Image.new("RGB", (7, 7), "white")
            source = Image.new("L", (7, 7), 0)
            ImageDraw.Draw(source).rectangle((1, 1, 5, 5), fill=255)
            dense.save(dense_root / "frame.jpg")
            source.save(source_root / "frame.jpg")
            dense.close()
            source.close()

            prepare_undistorted_masks.convert_masks(
                source_root, dense_root, output_root, 128, 0, True
            )
            expected, written, resumed = prepare_undistorted_masks.convert_masks(
                source_root, dense_root, output_root, 128, 0, True
            )

            self.assertEqual((expected, written, resumed), (1, 0, 1))


if __name__ == "__main__":
    unittest.main()
