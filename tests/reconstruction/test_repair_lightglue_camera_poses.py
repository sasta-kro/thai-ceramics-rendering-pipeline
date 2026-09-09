from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/reconstruction/repair_lightglue_camera_poses.py"
)
SPEC = importlib.util.spec_from_file_location("repair_lightglue_camera_poses", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SimilarityTests(unittest.TestCase):
    def test_estimate_similarity_recovers_known_transform(self) -> None:
        source = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        angle = np.deg2rad(31.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        target = (2.4 * (rotation @ source.T)).T + np.array([4.0, -2.0, 7.0])

        result = MODULE.estimate_similarity(source, target)

        np.testing.assert_allclose(result.transform_points(source), target, atol=1e-10)
        self.assertAlmostEqual(result.scale, 2.4)
        np.testing.assert_allclose(result.rotation, rotation, atol=1e-10)

    def test_robust_similarity_rejects_outliers(self) -> None:
        generator = np.random.default_rng(8)
        source = generator.normal(size=(80, 3))
        target = 1.7 * source + np.array([2.0, -4.0, 0.5])
        target[-15:] += 20.0

        result, inliers = MODULE.robust_similarity(
            source, target, threshold=0.01, iterations=800
        )

        self.assertEqual(result.inlier_count, 65)
        self.assertEqual(int(inliers.sum()), 65)
        self.assertLess(result.residual_rmse, 1e-10)


class PoseTests(unittest.TestCase):
    def test_quaternion_rotation_round_trip(self) -> None:
        qvec = np.array([0.7, 0.2, -0.4, 0.5])
        qvec /= np.linalg.norm(qvec)
        rotation = MODULE.qvec_to_rotation(qvec)

        recovered = MODULE.rotation_to_qvec(rotation)

        np.testing.assert_allclose(
            MODULE.qvec_to_rotation(recovered), rotation, atol=1e-12
        )

    def test_transform_pose_maps_camera_center(self) -> None:
        pose = MODULE.make_pose(
            3,
            "side_frame_000012.jpg",
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([-1.0, -2.0, -3.0]),
        )
        angle = np.deg2rad(90.0)
        similarity = MODULE.Similarity(
            2.0,
            np.array(
                [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
            ),
            np.array([10.0, 20.0, 30.0]),
        )

        transformed = MODULE.transform_pose(pose, similarity)

        expected_center = similarity.transform_points(pose.center[None, :])[0]
        np.testing.assert_allclose(transformed.center, expected_center, atol=1e-12)
        np.testing.assert_allclose(
            transformed.rotation, pose.rotation @ similarity.rotation.T, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
