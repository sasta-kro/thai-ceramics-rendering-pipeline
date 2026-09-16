from __future__ import annotations

import importlib
import struct
import tempfile
from pathlib import Path
import unittest

import numpy as np


MODULE = importlib.import_module(
    "scripts.gaussian_splatting.postprocessing.interior_patch"
)


def splat_dtype() -> np.dtype:
    names = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    return np.dtype([(name, "<f4") for name in names])


class InteriorPatchTests(unittest.TestCase):
    def test_robust_circle_fit_ignores_central_points(self) -> None:
        rng = np.random.default_rng(3)
        angle = np.linspace(0, 2 * np.pi, 800, endpoint=False)
        expected = np.asarray([0.13, -0.08])
        ring = expected + np.column_stack((np.cos(angle), np.sin(angle))) * 0.5
        ring += rng.normal(0.0, 0.003, ring.shape)
        central_outliers = expected + rng.normal(0.0, 0.08, (400, 2))
        fitted = MODULE.fit_rim_circle_center(
            np.concatenate((ring, central_outliers), axis=0)
        )
        np.testing.assert_allclose(fitted, expected, atol=0.01)

    def test_reads_colmap_centers_and_orients_axis_toward_mid25(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            images_bin = Path(temporary) / "images.bin"
            records = [
                (1, "mid25_frame_000000.jpg", (0.0, 0.0, 2.0)),
                (2, "mid25_frame_000006.jpg", (1.0, 0.0, 2.0)),
                (3, "mid25_frame_000012.jpg", (-1.0, 0.0, 2.0)),
                (4, "underside_frame_000000.jpg", (0.0, 0.0, -2.0)),
                (5, "underside_frame_000006.jpg", (1.0, 0.0, -2.0)),
                (6, "underside_frame_000012.jpg", (-1.0, 0.0, -2.0)),
            ]
            with images_bin.open("wb") as output:
                output.write(struct.pack("<Q", len(records)))
                for image_id, name, center in records:
                    output.write(struct.pack("<i", image_id))
                    output.write(struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))
                    output.write(struct.pack("<3d", *(-np.asarray(center))))
                    output.write(struct.pack("<i", 1))
                    output.write(name.encode("utf-8") + b"\0")
                    output.write(struct.pack("<Q", 0))
            centers = MODULE.read_colmap_camera_centers(images_bin)
            axis = MODULE.normalized_axis(centers, np.zeros(3), 1.0)
            np.testing.assert_allclose(axis, [0.0, 0.0, 1.0], atol=1e-12)

    def test_infers_rim_and_builds_finite_concave_patch(self) -> None:
        rng = np.random.default_rng(7)
        dtype = splat_dtype()
        vertices = np.zeros(6000, dtype=dtype)
        body_count = 4800
        angle = rng.uniform(0, 2 * np.pi, body_count)
        z = rng.uniform(-1.0, 0.94, body_count)
        radius = 0.62 - 0.14 * np.abs(z)
        vertices["x"][:body_count] = radius * np.cos(angle)
        vertices["y"][:body_count] = radius * np.sin(angle)
        vertices["z"][:body_count] = z
        rim_angle = rng.uniform(0, 2 * np.pi, len(vertices) - body_count)
        rim_radius = rng.normal(0.42, 0.012, len(rim_angle))
        vertices["x"][body_count:] = rim_radius * np.cos(rim_angle)
        vertices["y"][body_count:] = rim_radius * np.sin(rim_angle)
        vertices["z"][body_count:] = rng.uniform(0.95, 1.0, len(rim_angle))
        vertices["opacity"] = 4.0
        vertices["f_dc_0"] = 0.3
        vertices["f_dc_1"] = -0.2
        vertices["f_dc_2"] = -0.5

        geometry = MODULE.infer_geometry(
            vertices, np.asarray([0.0, 0.0, 1.0]), 0.72, 0.90, 0.16, 0.48
        )
        self.assertAlmostEqual(geometry.rim_outer_radius, 0.43, delta=0.04)
        self.assertLess(geometry.center_axial, geometry.edge_axial)
        patch = MODULE.create_patch_vertices(
            dtype, geometry, 2000, opacity=0.97, color_scale=0.68, seed=42
        )
        self.assertEqual(len(patch), 2000)
        for name in dtype.names or ():
            self.assertTrue(np.all(np.isfinite(patch[name])), name)
        patch_xyz = MODULE.vertex_xyz(patch)
        patch_axial = patch_xyz @ geometry.axis
        self.assertLess(float(patch_axial.min()), float(patch_axial.max()))

    def test_round_trips_binary_ply_with_updated_vertex_count(self) -> None:
        dtype = splat_dtype()
        vertices = np.zeros(3, dtype=dtype)
        properties = "\n".join(
            f"property float {name}" for name in (dtype.names or ())
        )
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            "element vertex 3\n" + properties + "\nend_header\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.ply"
            with source.open("wb") as output:
                output.write(header.encode("ascii"))
                vertices.tofile(output)
            ply = MODULE.read_ply(source)
            destination = Path(temporary) / "patched.ply"
            MODULE.write_ply(destination, ply, np.zeros(5, dtype=dtype))
            reread = MODULE.read_ply(destination)
            self.assertEqual(reread.vertex_count, 5)
            self.assertEqual(len(reread.vertices), 5)


if __name__ == "__main__":
    unittest.main()
