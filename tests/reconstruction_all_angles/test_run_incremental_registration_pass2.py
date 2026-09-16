from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/reconstruction_all_angles/run_incremental_registration_pass2.py"
)
spec = importlib.util.spec_from_file_location("run_incremental_registration_pass2", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def pose(value: float = 0.0) -> module.ImagePose:
    return module.ImagePose((1.0, value, 0.0, 0.0), (value, 0.0, 0.0), 1)


class IncrementalRegistrationPass2Tests(unittest.TestCase):
    def make_plan(
        self, root: Path, input_summary: module.ModelSummary
    ) -> module.RegistrationPass2Plan:
        return module.RegistrationPass2Plan(
            root / "colmap.exe",
            root,
            root / "database.db",
            root / "triangulated_pass1",
            root / "registered_final",
            root / "run.log",
            root / "report.json",
            input_summary,
            (),
        )

    def test_validate_result_recovers_mid25_and_preserves_input(self) -> None:
        input_summary = module.ModelSummary(
            {"side_a.jpg": pose(), "mid25_a.jpg": pose(0.1)}, 4, 100
        )
        output_summary = module.ModelSummary(
            {
                **input_summary.images,
                "mid25_b.jpg": pose(0.2),
                "mid25_c.jpg": pose(0.3),
            },
            4,
            100,
        )
        with tempfile.TemporaryDirectory() as directory:
            plan = self.make_plan(Path(directory), input_summary)
            original = module.summarize_model
            module.summarize_model = lambda *_args: output_summary
            try:
                report = module.validate_result(plan)
            finally:
                module.summarize_model = original
        self.assertEqual(report["mid25_recovered"], 2)
        self.assertEqual(report["input_images_preserved"], 2)
        self.assertEqual(report["existing_pose_max_absolute_change"], 0.0)

    def test_validate_result_rejects_no_recovery(self) -> None:
        input_summary = module.ModelSummary(
            {"side_a.jpg": pose(), "mid25_a.jpg": pose(0.1)}, 4, 100
        )
        with tempfile.TemporaryDirectory() as directory:
            plan = self.make_plan(Path(directory), input_summary)
            original = module.summarize_model
            module.summarize_model = lambda *_args: input_summary
            try:
                with self.assertRaisesRegex(
                    module.IncrementalRegistrationError, "did not recover"
                ):
                    module.validate_result(plan)
            finally:
                module.summarize_model = original


if __name__ == "__main__":
    unittest.main()
