from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts/reconstruction_all_angles"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_incremental_registration as registration  # noqa: E402


class RunIncrementalRegistrationTests(unittest.TestCase):
    def test_parse_images_text(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            path = Path(directory) / "images.txt"
            path.write_text(
                "# images\n"
                "1 1 0 0 0 0 0 1 1 side_0.jpg\n"
                "10 20 1\n"
                "2 1 0 0 0 1 2 3 4 mid25_0.jpg\n"
                "\n",
                encoding="utf-8",
            )
            images = registration.parse_images_text(path)
            self.assertEqual(len(images), 2)
            self.assertEqual(images["mid25_0.jpg"].camera_id, 4)

    def test_validate_result_preserves_base_and_counts_mid25(self) -> None:
        base_pose = registration.ImagePose((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1)
        mid_pose = registration.ImagePose((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 1.0), 4)
        base = registration.ModelSummary({"side_0.jpg": base_pose}, 3, 10)
        output = registration.ModelSummary(
            {"side_0.jpg": base_pose, "mid25_0.jpg": mid_pose}, 4, 11
        )
        plan = registration.RegistrationPlan(
            Path("colmap"),
            Path("workspace"),
            Path("database"),
            Path("input"),
            Path("output"),
            Path("log"),
            Path("report"),
            base,
            (),
        )
        original = registration.summarize_model
        try:
            registration.summarize_model = lambda _colmap, _model: output
            report = registration.validate_result(plan)
        finally:
            registration.summarize_model = original
        self.assertEqual(report["mid25_registered"], 1)
        self.assertEqual(report["base_pose_max_absolute_change"], 0.0)

    def test_max_pose_change_detects_change(self) -> None:
        base = {"side.jpg": registration.ImagePose((1, 0, 0, 0), (0, 0, 0), 1)}
        changed = {
            "side.jpg": registration.ImagePose((1, 0, 0, 0), (0, 0.25, 0), 1)
        }
        self.assertEqual(registration.max_pose_change(base, changed), 0.25)


if __name__ == "__main__":
    unittest.main()
