from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "reconstruction"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_colmap  # noqa: E402


class ColmapRunnerTests(unittest.TestCase):
    def test_image_files_ignores_frame_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frame_000000.jpg").write_bytes(b"image")
            (root / "frame_000006.PNG").write_bytes(b"image")
            (root / "frames_manifest.csv").write_text("filename\n", encoding="utf-8")

            self.assertEqual(
                [path.name for path in run_colmap.image_files(root)],
                ["frame_000000.jpg", "frame_000006.PNG"],
            )

    def test_validate_masks_accepts_colmap_double_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            image = images / "frame_000000.jpg"
            image.write_bytes(b"image")
            (masks / "frame_000000.jpg.png").write_bytes(b"mask")

            run_colmap.validate_masks([image], images, masks)

    def test_validate_masks_reports_missing_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            image = images / "frame_000000.jpg"
            image.write_bytes(b"image")

            with self.assertRaisesRegex(
                run_colmap.PipelineError, "frame_000000.jpg.png"
            ):
                run_colmap.validate_masks([image], images, masks)

    def test_output_child_cannot_escape_workspace(self) -> None:
        workspace = run_colmap.PROJECT_ROOT / "data" / "processed" / "example"
        with self.assertRaises(run_colmap.PipelineError):
            run_colmap.resolve_output_child(workspace, "../database.db", "database")

    def test_balanced_gpu_profile_limits_resolution_and_threads(self) -> None:
        name, settings = run_colmap.feature_memory_profile(
            {"memory_profile": "balanced_gpu"}
        )
        self.assertEqual(name, "balanced_gpu")
        self.assertEqual(settings["max_image_size"], 3200)
        self.assertEqual(settings["first_octave"], 0)
        self.assertEqual(settings["num_threads"], 1)

    def test_unknown_memory_profile_is_rejected(self) -> None:
        with self.assertRaises(run_colmap.PipelineError):
            run_colmap.feature_memory_profile({"memory_profile": "unlimited"})

    def test_execution_preserves_parenthesized_path_argument(self) -> None:
        image_path = "C:/Projects/CSX4213 (Computer Vision)/frames"
        command = run_colmap.execution_command(
            Path("C:/Tools/COLMAP/bin/colmap.exe"),
            ["feature_extractor", "--image_path", image_path],
        )
        self.assertEqual(
            command,
            [
                str(Path("C:/Tools/COLMAP/bin/colmap.exe")),
                "feature_extractor",
                "--image_path",
                image_path,
            ],
        )


if __name__ == "__main__":
    unittest.main()
