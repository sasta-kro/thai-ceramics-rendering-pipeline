from __future__ import annotations

import csv
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

    def test_multisequence_pairs_connect_each_view_through_side(self) -> None:
        sequences = {
            "side": ["side_0.jpg", "side_1.jpg", "side_2.jpg", "side_3.jpg"],
            "top45": ["top45_0.jpg", "top45_1.jpg", "top45_2.jpg"],
            "underside": ["underside_0.jpg", "underside_1.jpg"],
        }

        plan = run_colmap.build_multisequence_pair_plan(
            sequences,
            sequential_overlap=1,
            loop_closure=False,
            bridge_stride=2,
            bridges=[("side", "top45"), ("side", "underside")],
        )

        self.assertEqual(plan.within_sequence, 6)
        self.assertEqual(plan.loop_closure, 0)
        self.assertEqual(
            dict(plan.bridge_counts),
            {"side<->top45": 6, "side<->underside": 6},
        )
        self.assertEqual(len(plan.pairs), 18)
        for left, right in plan.pairs:
            self.assertFalse(
                {left.split("_", 1)[0], right.split("_", 1)[0]}
                == {"top45", "underside"}
            )

    def test_multisequence_loop_closes_each_capture_sequence(self) -> None:
        plan = run_colmap.build_multisequence_pair_plan(
            {
                "side": ["side_0.jpg", "side_1.jpg", "side_2.jpg"],
                "top45": ["top45_0.jpg", "top45_1.jpg", "top45_2.jpg"],
            },
            sequential_overlap=1,
            loop_closure=True,
            bridge_stride=3,
            bridges=[("side", "top45")],
        )

        self.assertEqual(plan.within_sequence, 4)
        self.assertEqual(plan.loop_closure, 2)
        self.assertIn(("side_2.jpg", "side_0.jpg"), plan.pairs)
        self.assertIn(("top45_2.jpg", "top45_0.jpg"), plan.pairs)

    def test_manifest_must_describe_every_combined_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            image_paths = [images / "side_0.jpg", images / "top45_0.jpg"]
            for path in image_paths:
                path.write_bytes(b"image")
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("view", "view_frame_index", "combined_filename"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "view": "side",
                        "view_frame_index": 0,
                        "combined_filename": "side_0.jpg",
                    }
                )

            with self.assertRaisesRegex(
                run_colmap.PipelineError, "manifest/image mismatch"
            ):
                run_colmap.load_multiview_sequences(
                    manifest, image_paths, images
                )

    def test_explicit_pair_list_uses_colmap_pair_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.txt"
            run_colmap.write_pair_list(
                path,
                [("side_0.jpg", "top45_0.jpg"), ("side_1.jpg", "underside_1.jpg")],
            )

            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                [
                    "side_0.jpg top45_0.jpg",
                    "side_1.jpg underside_1.jpg",
                ],
            )

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
