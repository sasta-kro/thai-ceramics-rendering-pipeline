from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/reconstruction_all_angles/run_incremental_triangulation.py"
)
spec = importlib.util.spec_from_file_location("run_incremental_triangulation", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


ImagePose = module.ImagePose
ModelSummary = module.ModelSummary
TriangulationPlan = module.TriangulationPlan


def pose(value: float = 0.0) -> ImagePose:
    return ImagePose((1.0, value, 0.0, 0.0), (value, 0.0, 0.0), 1)


class IncrementalTriangulationTests(unittest.TestCase):
    def test_validate_result_preserves_images_and_accepts_fresh_points(self) -> None:
        input_summary = ModelSummary(
            {"side_a.jpg": pose(), "mid25_a.jpg": pose(0.1)}, 4, 100
        )
        output_summary = ModelSummary(dict(input_summary.images), 4, 85)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = TriangulationPlan(
                root / "colmap.exe",
                root,
                root / "database.db",
                root / "images",
                root / "registered_pass1",
                root / "triangulated_pass1",
                root / "run.log",
                root / "report.json",
                input_summary,
                (),
            )
            original = module.summarize_model
            module.summarize_model = lambda *_args: output_summary
            try:
                report = module.validate_result(plan)
            finally:
                module.summarize_model = original
        self.assertEqual(report["registered_images_preserved"], 2)
        self.assertEqual(report["mid25_images_preserved"], 1)
        self.assertEqual(report["fresh_sparse_points"], 85)
        self.assertEqual(report["fresh_minus_provisional_points"], -15)

    def test_validate_result_rejects_pose_change(self) -> None:
        input_summary = ModelSummary(
            {"side_a.jpg": pose(), "mid25_a.jpg": pose(0.1)}, 4, 100
        )
        output_summary = ModelSummary(
            {"side_a.jpg": pose(0.01), "mid25_a.jpg": pose(0.1)}, 4, 135
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = TriangulationPlan(
                root / "colmap.exe",
                root,
                root / "database.db",
                root / "images",
                root / "registered_pass1",
                root / "triangulated_pass1",
                root / "run.log",
                root / "report.json",
                input_summary,
                (),
            )
            original = module.summarize_model
            module.summarize_model = lambda *_args: output_summary
            try:
                with self.assertRaisesRegex(
                    module.IncrementalRegistrationError, "poses changed"
                ):
                    module.validate_result(plan)
            finally:
                module.summarize_model = original


if __name__ == "__main__":
    unittest.main()
