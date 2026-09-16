from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/reconstruction_all_angles/analyze_incremental_camera_trajectory.py"
)
spec = importlib.util.spec_from_file_location(
    "analyze_incremental_camera_trajectory", SCRIPT
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class CameraTrajectoryAnalysisTests(unittest.TestCase):
    def test_identity_pose_center_is_negative_translation(self) -> None:
        pose = module.ImagePose((1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0), 1)
        self.assertEqual(module.camera_center(pose), (-1.0, -2.0, -3.0))

    def test_quaternion_sign_does_not_change_rotation_difference(self) -> None:
        first = module.ImagePose((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1)
        second = module.ImagePose((-1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1)
        self.assertAlmostEqual(module.rotation_difference_degrees(first, second), 0.0)

    def test_frame_index_uses_final_numeric_token(self) -> None:
        self.assertEqual(module.frame_index("mid25_frame_001320.jpg"), 1320)


if __name__ == "__main__":
    unittest.main()
