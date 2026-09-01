from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gaussian_splatting.postprocessing import checkpoint as postprocess  # noqa: E402
from scripts.gaussian_splatting.training import runner as run_training  # noqa: E402


def fake_checkpoint(count: int = 2) -> dict:
    settings = run_training.load_training_settings(
        run_training.load_yaml(run_training.DEFAULT_CONFIG), "baseline_7k"
    )
    return {
        "step": 7000,
        "profile": settings.__dict__,
        "splats": {
            "means": torch.zeros((count, 3)),
            "scales": torch.zeros((count, 3)),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
            "opacities": torch.zeros((count,)),
            "sh0": torch.zeros((count, 1, 3)),
            "shN": torch.zeros((count, 8, 3)),
        },
        "normalization_center": torch.zeros(3).numpy(),
        "normalization_scale": 1.0,
    }


class PostprocessTests(unittest.TestCase):
    def test_resolves_only_complete_run_and_final_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory)
            run_dir = runs / "baseline_7k"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            (run_dir / "training_summary.json").write_text(
                json.dumps({"status": "complete", "steps": 7000}),
                encoding="utf-8",
            )
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"profile": {"profile_name": "baseline_7k"}}),
                encoding="utf-8",
            )
            (checkpoint_dir / "step_007000.pt").write_bytes(b"checkpoint")

            resolved = postprocess.resolve_complete_run(
                SimpleNamespace(runs=runs), "baseline_7k"
            )

            self.assertEqual(resolved.step, 7000)
            self.assertEqual(resolved.profile_name, "baseline_7k")
            self.assertEqual(resolved.checkpoint.name, "step_007000.pt")

    def test_checkpoint_loader_validates_expected_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            torch.save(fake_checkpoint(), checkpoint_path)

            loaded = postprocess.load_checkpoint_cpu(checkpoint_path)

            self.assertEqual(tuple(loaded["splats"]["means"].shape), (2, 3))

    def test_identity_quaternion_produces_diagonal_covariance(self) -> None:
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        log_scales = torch.log(torch.tensor([[1.0, 2.0, 3.0]]))

        covariance = postprocess.quaternion_covariances(quats, log_scales)

        torch.testing.assert_close(
            covariance[0], torch.diag(torch.tensor([1.0, 4.0, 9.0]))
        )

    def test_viewer_arrays_apply_opacity_limit(self) -> None:
        checkpoint = fake_checkpoint(count=3)
        checkpoint["splats"]["opacities"] = torch.tensor([-10.0, 0.0, 10.0])

        centers, covariances, colors, opacities = postprocess.viewer_arrays(
            checkpoint, opacity_threshold=0.1, max_gaussians=1
        )

        self.assertEqual(centers.shape, (1, 3))
        self.assertEqual(covariances.shape, (1, 3, 3))
        self.assertEqual(colors.shape, (1, 3))
        self.assertGreater(float(opacities[0, 0]), 0.99)

    def test_export_names_are_run_scoped(self) -> None:
        run = postprocess.CompleteRun(
            run_name="baseline_7k",
            run_dir=Path("runs") / "baseline_7k",
            checkpoint=Path("checkpoint.pt"),
            step=7000,
            profile_name="baseline_7k",
            summary={},
        )

        self.assertEqual(
            postprocess.export_path(run, "ply").name,
            "thai_ceramics_baseline_7k.ply",
        )
        self.assertEqual(
            postprocess.export_path(run, "splat").name,
            "thai_ceramics_baseline_7k.splat",
        )


if __name__ == "__main__":
    unittest.main()
