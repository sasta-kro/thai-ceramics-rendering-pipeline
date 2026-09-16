from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/reconstruction_all_angles/run_final_bundle_adjustment.py"
)
spec = importlib.util.spec_from_file_location("run_final_bundle_adjustment", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class FinalBundleAdjustmentTests(unittest.TestCase):
    def test_analyze_model_parses_colmap_output(self) -> None:
        sample = (
            "Registered images: 716\n"
            "Points: 130577\n"
            "Observations: 1523453\n"
            "Mean track length: 11.667085\n"
            "Mean observations per image: 2127.727654\n"
            "Mean reprojection error: 0.999984px\n"
        )
        completed = module.subprocess.CompletedProcess([], 0, "", sample)
        original = module.subprocess.run
        module.subprocess.run = lambda *_args, **_kwargs: completed
        try:
            result = module.analyze_model(Path("colmap"), Path("model"))
        finally:
            module.subprocess.run = original
        self.assertEqual(result.images, 716)
        self.assertEqual(result.points, 130577)
        self.assertAlmostEqual(result.mean_reprojection_error, 0.999984)

    def test_validate_result_accepts_improved_error(self) -> None:
        pose = module.ImagePose((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1)
        summary = module.ModelSummary({"side_a.jpg": pose}, 3, 10)
        analysis_before = module.AnalyzerSummary(1, 10, 20, 2.0, 20.0, 1.0)
        analysis_after = module.AnalyzerSummary(1, 10, 20, 2.0, 20.0, 0.8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = module.BundleAdjustmentPlan(
                root / "colmap.exe",
                root,
                root / "input",
                root / "output",
                root / "run.log",
                root / "report.json",
                summary,
                analysis_before,
                (),
            )
            original_summary = module.summarize_model
            original_analysis = module.analyze_model
            module.summarize_model = lambda *_args: summary
            module.analyze_model = lambda *_args: analysis_after
            try:
                report = module.validate_result(plan)
            finally:
                module.summarize_model = original_summary
                module.analyze_model = original_analysis
        self.assertAlmostEqual(report["mean_reprojection_error_after_px"], 0.8)
        self.assertAlmostEqual(report["mean_reprojection_error_improvement_px"], 0.2)


if __name__ == "__main__":
    unittest.main()
