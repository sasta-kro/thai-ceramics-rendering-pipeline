from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gaussian_splatting.core import common  # noqa: E402
from scripts.gaussian_splatting.training import runner as run_training  # noqa: E402


class RunTrainingTests(unittest.TestCase):
    def test_smoke_profile_is_guarded_and_low_memory(self) -> None:
        config_path = common.DEFAULT_CONFIG
        settings = run_training.load_training_settings(
            common.load_yaml(config_path), "smoke_lowmem"
        )

        self.assertEqual(settings.image_factor, 4)
        self.assertEqual(settings.max_steps, 300)
        self.assertEqual(settings.max_gaussians, 100000)
        self.assertTrue(settings.packed)
        self.assertTrue(settings.sparse_grad)
        self.assertTrue(settings.disable_viewer)

    def test_run_name_rejects_path_escape(self) -> None:
        with self.assertRaises(run_training.PipelineError):
            run_training.validated_run_name("../outside")

    def test_cuda_memory_helpers_use_current_device(self) -> None:
        torch_module = Mock()
        torch_module.cuda.max_memory_allocated.return_value = 2 * 1024**3

        run_training.reset_cuda_peak_memory_stats(torch_module)
        peak_gb = run_training.peak_cuda_memory_gb(torch_module)

        torch_module.cuda.reset_peak_memory_stats.assert_called_once_with()
        torch_module.cuda.max_memory_allocated.assert_called_once_with()
        self.assertEqual(peak_gb, 2.0)

    def test_background_is_composited_from_raster_alpha(self) -> None:
        rendered = torch.zeros((2, 2, 3), dtype=torch.float32)
        alpha = torch.full((2, 2, 1), 0.25, dtype=torch.float32)
        background = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)

        composited = run_training.composite_background(
            rendered, alpha, background
        )

        expected = torch.tensor([0.75, 0.375, 0.0], dtype=torch.float32)
        torch.testing.assert_close(composited[0, 0], expected)
        self.assertEqual(tuple(composited.shape), (2, 2, 3))


if __name__ == "__main__":
    unittest.main()
