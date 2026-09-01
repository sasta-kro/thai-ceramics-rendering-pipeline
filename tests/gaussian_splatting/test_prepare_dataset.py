from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gaussian_splatting.data import preparation as prepare_dataset  # noqa: E402


class PrepareDatasetTests(unittest.TestCase):
    def test_target_name_changes_image_extension_to_png(self) -> None:
        self.assertEqual(
            prepare_dataset.target_relative_path(Path("nested/frame_000006.jpg")),
            Path("nested/frame_000006.png"),
        )

    def test_target_size_matches_configured_factor(self) -> None:
        self.assertEqual(prepare_dataset.target_size((1125, 2000), 2), (562, 1000))
        self.assertEqual(prepare_dataset.target_size((1125, 2000), 4), (281, 500))

    def test_holdout_split_is_deterministic_for_ordered_images(self) -> None:
        names = [f"frame_{index:03d}.jpg" for index in range(10)]
        split = prepare_dataset.holdout_split(names, 4)

        self.assertEqual(split["test"], [names[0], names[4], names[8]])
        self.assertEqual(split["train_count"], 7)
        self.assertEqual(split["test_count"], 3)

    def test_alpha_aware_resize_preserves_foreground_color(self) -> None:
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            mask_path = root / "mask.png"

            rgb = Image.new("RGB", (8, 8), (0, 0, 0))
            mask = Image.new("L", (8, 8), 0)
            for y in range(2, 6):
                for x in range(2, 6):
                    rgb.putpixel((x, y), (200, 80, 40))
                    mask.putpixel((x, y), 255)
            rgb.save(image_path)
            mask.save(mask_path)
            rgb.close()
            mask.close()

            resized_rgb, resized_mask = prepare_dataset.alpha_aware_resize(
                image_path,
                mask_path,
                (4, 4),
                Image.Resampling.BICUBIC,
                Image.Resampling.LANCZOS,
                1 / 255,
            )
            try:
                rgb_values = np.asarray(resized_rgb)
                alpha_values = np.asarray(resized_mask)
                visible = alpha_values > 1
                self.assertTrue(visible.any())
                self.assertGreaterEqual(int(rgb_values[..., 0][visible].max()), 190)
                self.assertEqual(int(rgb_values[~visible].max(initial=0)), 0)
            finally:
                resized_rgb.close()
                resized_mask.close()


if __name__ == "__main__":
    unittest.main()
