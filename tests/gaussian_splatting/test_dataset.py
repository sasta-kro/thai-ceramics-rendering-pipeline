from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gaussian_splatting.data import scene as dataset  # noqa: E402


class DatasetTests(unittest.TestCase):
    def test_multiple_pinhole_cameras_are_supported(self) -> None:
        cameras = {
            1: SimpleNamespace(model=SimpleNamespace(name="PINHOLE")),
            2: SimpleNamespace(model=SimpleNamespace(name="PINHOLE")),
            3: SimpleNamespace(model=SimpleNamespace(name="PINHOLE")),
        }

        dataset.validate_undistorted_cameras(cameras)

    def test_distorted_camera_is_rejected(self) -> None:
        cameras = {
            1: SimpleNamespace(model=SimpleNamespace(name="PINHOLE")),
            2: SimpleNamespace(model=SimpleNamespace(name="SIMPLE_RADIAL")),
        }

        with self.assertRaisesRegex(dataset.PipelineError, "SIMPLE_RADIAL"):
            dataset.validate_undistorted_cameras(cameras)

    def test_multiple_resolution_label(self) -> None:
        self.assertEqual(
            dataset.resolution_text(((279, 500), (281, 500))),
            "279x500, 281x500",
        )

    def test_normalization_centers_and_scales_camera_orbit(self) -> None:
        cameras = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        cameras[0, :3, 3] = (-2.0, 0.0, 0.0)
        cameras[1, :3, 3] = (2.0, 0.0, 0.0)
        points = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)

        normalized_cameras, normalized_points, center, scale = dataset.normalize_scene(
            cameras, points
        )

        np.testing.assert_allclose(center, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(scale, 0.5)
        self.assertAlmostEqual(
            np.linalg.norm(normalized_cameras[:, :3, 3], axis=1).max(), 1.0
        )
        np.testing.assert_allclose(normalized_points[:, 1], (0.5, -0.5))


if __name__ == "__main__":
    unittest.main()
