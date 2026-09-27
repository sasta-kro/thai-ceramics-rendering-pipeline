from __future__ import annotations

import importlib
import struct
import tempfile
from pathlib import Path
import unittest

import numpy as np


MODULE = importlib.import_module(
    "scripts.gaussian_splatting.postprocessing.lower_cleanup"
)


class LowerCleanupTests(unittest.TestCase):
    def test_reads_camera_image_up_axis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            images_bin = Path(temporary) / "images.bin"
            with images_bin.open("wb") as output:
                output.write(struct.pack("<Q", 3))
                for image_id in range(1, 4):
                    output.write(struct.pack("<i", image_id))
                    output.write(struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))
                    output.write(struct.pack("<3d", 0.0, 0.0, 0.0))
                    output.write(struct.pack("<i", 1))
                    output.write(f"side_{image_id}.jpg".encode() + b"\0")
                    output.write(struct.pack("<Q", 0))
            axis = MODULE.read_colmap_up_axis(images_bin)
            np.testing.assert_allclose(axis, [0.0, -1.0, 0.0], atol=1e-12)

    def test_reads_colmap_sparse_points(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            points_bin = Path(temporary) / "points3D.bin"
            with points_bin.open("wb") as output:
                output.write(struct.pack("<Q", 100))
                for point_id in range(100):
                    output.write(struct.pack("<Q", point_id + 1))
                    output.write(struct.pack("<3d", point_id, 2.0, 3.0))
                    output.write(bytes((100, 80, 60)))
                    output.write(struct.pack("<d", 0.1))
                    output.write(struct.pack("<Q", 1))
                    output.write(struct.pack("<II", 1, point_id))
            points = MODULE.read_colmap_points(points_bin)
            self.assertEqual(points.shape, (100, 3))
            np.testing.assert_allclose(points[-1], [99.0, 2.0, 3.0])

    def test_cleanup_plane_uses_sparse_support_and_margin(self) -> None:
        sparse = np.column_stack(
            (
                np.zeros(1000),
                np.zeros(1000),
                np.linspace(-1.0, 2.0, 1000),
            )
        )
        geometry = MODULE.infer_cleanup_geometry(
            axis=np.asarray([0.0, 0.0, 1.0]),
            sparse_points=sparse,
            normalization_center=np.zeros(3),
            normalization_scale=1.0,
            lower_quantile=0.01,
            margin_scale=0.02,
        )
        self.assertAlmostEqual(geometry.sparse_low, -0.97, places=2)
        self.assertAlmostEqual(geometry.sparse_high, 1.97, places=2)
        self.assertAlmostEqual(geometry.cutoff, -1.0288, places=3)


if __name__ == "__main__":
    unittest.main()
