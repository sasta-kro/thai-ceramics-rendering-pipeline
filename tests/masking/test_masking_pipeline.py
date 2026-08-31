from __future__ import annotations

import csv
import contextlib
import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "masking"
sys.path.insert(0, str(SCRIPTS_DIR))

import mask_dataset  # noqa: E402
import masking_core as core  # noqa: E402


class DatasetTests(unittest.TestCase):
    def test_natural_sort_orders_numeric_frame_names(self) -> None:
        values = ["frame_10.jpg", "frame_2.jpg", "frame_001.jpg"]
        self.assertEqual(
            sorted(values, key=core.natural_sort_key),
            ["frame_001.jpg", "frame_2.jpg", "frame_10.jpg"],
        )

    def test_discover_frames_reads_and_preserves_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((18, 24, 3), 90, dtype=np.uint8)
            cv2.imwrite(str(root / "frame_000003.jpg"), image)
            cv2.imwrite(str(root / "frame_000000.jpg"), image)
            with (root / "frames_manifest.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=["filename", "timestamp"])
                writer.writeheader()
                writer.writerow({"filename": "frame_000000.jpg", "timestamp": "zero"})
                writer.writerow({"filename": "frame_000003.jpg", "timestamp": "three"})

            frames, fields = core.discover_frames(root)
            self.assertEqual([frame.filename for frame in frames], ["frame_000000.jpg", "frame_000003.jpg"])
            self.assertEqual(fields, ["filename", "timestamp"])
            self.assertEqual(frames[1].manifest["timestamp"], "three")

    def test_discover_frames_rejects_mixed_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / "frame_0.jpg"), np.zeros((10, 10, 3), np.uint8))
            cv2.imwrite(str(root / "frame_1.jpg"), np.zeros((12, 10, 3), np.uint8))
            with self.assertRaises(ValueError):
                core.discover_frames(root)

    def test_numeric_staging_preserves_order_and_uses_numeric_jpegs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in (6, 0, 3):
                cv2.imwrite(
                    str(root / f"frame_{index:06d}.jpg"),
                    np.full((8, 9, 3), index, np.uint8),
                )
            frames, _ = core.discover_frames(root)
            with core.numeric_frame_staging(frames) as (staging, methods):
                self.assertEqual(
                    [path.name for path in sorted(staging.iterdir())],
                    ["000000.jpg", "000001.jpg", "000002.jpg"],
                )
                self.assertEqual(len(methods), 3)
                self.assertTrue(all(method in {"symlink", "hardlink", "copy"} for method in methods))


class PromptTests(unittest.TestCase):
    def test_annotation_display_scale_fits_both_dimensions(self) -> None:
        self.assertEqual(
            mask_dataset.annotation_display_scale(800, 600, 1200, 800),
            1.0,
        )
        self.assertAlmostEqual(
            mask_dataset.annotation_display_scale(2160, 3840, 1200, 800),
            800 / 3840,
        )

    def test_prompt_coordinates_round_trip(self) -> None:
        box = [10, 20, 80, 90]
        normalized = core.normalized_box(box, 100, 200)
        restored = core.denormalized_box(normalized, 100, 200)
        np.testing.assert_allclose(restored, box)

        points = [(20, 50, 1), (90, 180, 0)]
        normalized_points = core.normalized_points(points, 100, 200)
        coordinates, labels = core.denormalized_points(normalized_points, 100, 200)
        np.testing.assert_allclose(coordinates, [[20, 50], [90, 180]])
        np.testing.assert_array_equal(labels, [1, 0])

    def test_noninteractive_annotation_creates_prompt_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "frames"
            output = root / "output"
            input_dir.mkdir()
            cv2.imwrite(
                str(input_dir / "frame_000000.jpg"),
                np.full((100, 120, 3), 100, np.uint8),
            )
            args = Namespace(
                input_frames=input_dir,
                output=output,
                material="matte",
                frame="0",
                box=[20.0, 10.0, 100.0, 90.0],
                positive=[[50.0, 50.0]],
                negative=[[10.0, 95.0]],
                overwrite=False,
            )
            self.assertEqual(mask_dataset.command_annotate(args), 0)
            document = json.loads((output / "prompts.json").read_text(encoding="utf-8"))
            self.assertEqual(document["material"], "matte")
            self.assertEqual(document["prompts"][0]["filename"], "frame_000000.jpg")
            self.assertEqual(len(document["prompts"][0]["points_normalized"]), 2)


class MaskTests(unittest.TestCase):
    def test_postprocessing_removes_island_and_fills_hole(self) -> None:
        raw = np.zeros((100, 120), dtype=np.uint8)
        raw[20:85, 30:95] = 255
        raw[45:55, 55:65] = 0
        raw[5:10, 5:10] = 255
        previous = np.zeros_like(raw)
        previous[20:85, 30:95] = 255

        result = core.postprocess_mask(raw, previous)
        self.assertEqual(result.raw_component_count, 2)
        self.assertEqual(int(result.mask[50, 60]), 255)
        self.assertEqual(int(result.mask[7, 7]), 0)
        self.assertGreater(result.cleanup_change_ratio, 0)

    def test_colmap_erosion_moves_boundary_inward(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[3:17, 3:17] = 255
        eroded = core.erode_mask(mask, 2)
        self.assertEqual(np.count_nonzero(eroded), 10 * 10)

    def test_product_outputs_preserve_rgb_and_colmap_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "frames"
            output = root / "output"
            input_dir.mkdir()
            for name in mask_dataset.GENERATED_DIRECTORIES:
                (output / name).mkdir(parents=True)
            source = np.zeros((40, 50, 3), dtype=np.uint8)
            source[:, :, 0] = 17
            source[:, :, 1] = 83
            source[:, :, 2] = 201
            source_path = input_dir / "frame_000003.png"
            cv2.imwrite(str(source_path), source)
            frame = core.FrameInfo(0, source_path, 50, 40, {})
            mask = np.zeros((40, 50), dtype=np.uint8)
            mask[8:32, 12:38] = 255

            products = mask_dataset.save_products(core, frame, mask, output, 3, 1280)
            self.assertEqual(products["colmap_mask"], "masks_colmap/frame_000003.png.png")
            rgba = cv2.imread(str(output / products["rgba_image"]), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(rgba[:, :, :3], source)
            np.testing.assert_array_equal(rgba[:, :, 3], mask)
            object_mask = cv2.imread(
                str(output / products["object_mask"]), cv2.IMREAD_GRAYSCALE
            )
            self.assertEqual(set(np.unique(object_mask)), {0, 255})

    def test_sequence_qc_flags_expected_failures(self) -> None:
        records = []
        for index, area in enumerate([0.20, 0.20, 0.35, 0.20, 0.20]):
            records.append(
                {
                    "mask_area": int(area * 1000),
                    "mask_area_ratio": area,
                    "centroid_x_normalized": 0.5 + (0.05 if index == 3 else 0),
                    "centroid_y_normalized": 0.5,
                    "touches_boundary": 0,
                    "raw_component_count": 1,
                    "cleanup_change_ratio": 0.0,
                    "previous_iou": 0.8 if index == 4 else 1.0,
                    "review_status": "unreviewed",
                    "qc_warnings": "",
                }
            )
        core.calculate_sequence_flags(records, "matte")
        self.assertIn("area_jump", records[2]["qc_warnings"])
        self.assertIn("centroid_jump", records[3]["qc_warnings"])
        self.assertIn("low_previous_iou", records[4]["qc_warnings"])


class PipelineUtilityTests(unittest.TestCase):
    def test_chunk_ranges_include_overlap(self) -> None:
        self.assertEqual(
            list(mask_dataset.chunk_ranges(250, 120, 8)),
            [(0, 0, 120), (1, 112, 232), (2, 224, 250)],
        )

    def test_prepare_output_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "masks_object").mkdir()
            (output / "masks_object" / "old.png").write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                mask_dataset.prepare_output(output, overwrite=False)
            mask_dataset.prepare_output(output, overwrite=True)
            self.assertFalse((output / "masks_object" / "old.png").exists())
            self.assertTrue((output / "rgba").is_dir())

    def test_full_process_with_fake_predictor_writes_aligned_outputs(self) -> None:
        class FakeTensor:
            def __init__(self, value: np.ndarray):
                self.value = np.asarray(value)

            def __getitem__(self, key):
                return FakeTensor(self.value[key])

            def __gt__(self, value):
                return FakeTensor(self.value > value)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class FakePredictor:
            def init_state(self, video_path: str, **_kwargs):
                paths = sorted(Path(video_path).glob("*.jpg"))
                image = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
                return {"count": len(paths), "height": image.shape[0], "width": image.shape[1]}

            def reset_state(self, _state):
                return None

            def add_new_mask(self, **_kwargs):
                return None

            def add_new_points_or_box(self, **_kwargs):
                return None

            def propagate_in_video(self, state):
                for index in range(state["count"]):
                    logits = np.full((1, 1, state["height"], state["width"]), -1.0, np.float32)
                    logits[:, :, 12:48, 22:58] = 1.0
                    yield index, [1], FakeTensor(logits)

        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "fake"
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        )
        fake_torch.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        fake_torch.device = lambda name: types.SimpleNamespace(type=name)
        fake_torch.inference_mode = contextlib.nullcontext

        fake_sam2 = types.ModuleType("sam2")
        fake_sam2.__file__ = str(PROJECT_ROOT / "fake_sam2" / "__init__.py")
        fake_build = types.ModuleType("sam2.build_sam")
        fake_build.build_sam2_video_predictor = lambda *_args, **_kwargs: FakePredictor()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "frames"
            output = root / "output"
            input_dir.mkdir()
            for index in range(7):
                image = np.full((60, 80, 3), (30, 80, 130), dtype=np.uint8)
                cv2.imwrite(str(input_dir / f"frame_{index * 3:06d}.jpg"), image)
            annotate_args = Namespace(
                input_frames=input_dir,
                output=output,
                material="glossy",
                frame="0",
                box=[20.0, 10.0, 60.0, 50.0],
                positive=[],
                negative=[[5.0, 55.0]],
                overwrite=False,
            )
            mask_dataset.command_annotate(annotate_args)
            checkpoint = root / "tiny.pt"
            checkpoint.write_bytes(b"fake checkpoint")
            process_args = Namespace(
                input_frames=input_dir,
                output=output,
                device="auto",
                checkpoint=checkpoint,
                model_config=mask_dataset.DEFAULT_MODEL_CONFIG,
                chunk_size=4,
                chunk_overlap=1,
                colmap_erosion=3,
                overlay_max_dimension=1280,
                qc_sample_stride=2,
                overwrite=False,
            )
            with mock.patch.dict(
                sys.modules,
                {"torch": fake_torch, "sam2": fake_sam2, "sam2.build_sam": fake_build},
            ):
                self.assertEqual(mask_dataset.command_process(process_args), 0)

            fields, rows = core.read_mask_manifest(output / "mask_manifest.csv")
            self.assertEqual(len(rows), 7)
            self.assertIn("input_filename", fields)
            self.assertEqual(len(list((output / "masks_object").glob("*.png"))), 7)
            self.assertEqual(len(list((output / "masks_colmap").glob("*.png"))), 7)
            self.assertEqual(len(list((output / "rgba").glob("*.png"))), 7)
            self.assertTrue((output / "run_metadata.json").is_file())
            self.assertTrue((output / "qc_contact_sheet.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
