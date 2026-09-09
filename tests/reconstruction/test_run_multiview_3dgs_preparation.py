from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/reconstruction/run_multiview_3dgs_preparation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_multiview_3dgs_preparation", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MultiviewPreparationTests(unittest.TestCase):
    def test_undistortion_arguments_use_requested_image_and_output_paths(self) -> None:
        paths = SimpleNamespace(
            colmap=Path("colmap.exe"), sparse_model=Path("model")
        )
        settings = SimpleNamespace(
            output_type="COLMAP", max_image_size=2000, jpeg_quality=100, num_threads=2
        )

        arguments = MODULE.undistortion_arguments(
            paths,
            settings,
            image_path=Path("mask_inputs"),
            output_path=Path("mask_workspace"),
        )

        self.assertIn("mask_inputs", arguments)
        self.assertIn("mask_workspace", arguments)
        self.assertEqual(arguments[1], "image_undistorter")
        self.assertEqual(arguments[arguments.index("--max_image_size") + 1], "2000")

    def test_mask_workspace_stays_inside_output_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            paths = SimpleNamespace(workspace=workspace)
            config = {"output": {"mask_undistortion_workspace": "mask_workspace"}}

            result = MODULE.mask_workspace(paths, config)

            self.assertEqual(result, workspace / "mask_workspace")

    def test_path_length_guard_rejects_long_colmap_image_path(self) -> None:
        with self.assertRaisesRegex(MODULE.dense.PipelineError, "path limit"):
            MODULE.validate_colmap_image_path_lengths(
                Path("input"),
                Path("output"),
                [Path("frame.jpg")],
                max_length=5,
            )

    def test_path_length_guard_accepts_short_paths(self) -> None:
        MODULE.validate_colmap_image_path_lengths(
            Path("input"),
            Path("output"),
            [Path("frame.jpg")],
            max_length=1000,
        )


if __name__ == "__main__":
    unittest.main()
