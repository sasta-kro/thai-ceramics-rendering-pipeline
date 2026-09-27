from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "reconstruction"
sys.path.insert(0, str(SCRIPTS_DIR))

import prepare_multiview_dataset as multiview  # noqa: E402


class PrepareMultiviewDatasetTests(unittest.TestCase):
    def create_source(
        self,
        root: Path,
        name: str,
        filenames: tuple[str, ...] = ("frame_000000.jpg", "frame_000006.jpg"),
        size: tuple[int, int] = (9, 7),
    ) -> multiview.SourceSpec:
        from PIL import Image, ImageDraw

        images = root / name / "images"
        masks = root / name / "masks"
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        for index, filename in enumerate(filenames):
            rgb = Image.new("RGB", size, (120 + index, 80, 40))
            mask = Image.new("L", size, 0)
            ImageDraw.Draw(mask).rectangle((1, 1, size[0] - 2, size[1] - 2), fill=255)
            rgb.save(images / filename, quality=100)
            mask.save(masks / f"{filename}.png")
            rgb.close()
            mask.close()
        return multiview.SourceSpec(name=name, images=images, masks=masks)

    def test_stages_prefixed_pairs_and_manifest_without_changing_sources(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            side = self.create_source(root, "side")
            top = self.create_source(root, "top45")
            output_images = root / "combined" / "images"
            output_masks = root / "combined" / "masks"
            manifest = root / "combined" / "dataset_manifest.csv"
            original_side = (side.images / "frame_000000.jpg").read_bytes()

            report = multiview.prepare_dataset(
                [side, top],
                output_images,
                output_masks,
                manifest,
                validate_only=False,
                overwrite=False,
            )

            self.assertEqual(report.images, 4)
            self.assertEqual(report.masks, 4)
            self.assertTrue((output_images / "side_frame_000000.jpg").is_file())
            self.assertTrue((output_images / "top45_frame_000000.jpg").is_file())
            self.assertTrue((output_masks / "side_frame_000000.jpg.png").is_file())
            self.assertEqual(
                (side.images / "frame_000000.jpg").read_bytes(), original_side
            )
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["view"] for row in rows}, {"side", "top45"})
            self.assertTrue(all(len(row["image_sha256"]) == 64 for row in rows))
            self.assertTrue(all(len(row["mask_sha256"]) == 64 for row in rows))

    def test_validate_only_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            source = self.create_source(root, "side")
            output_images = root / "combined_images"
            output_masks = root / "combined_masks"
            manifest = root / "manifest.csv"

            report = multiview.prepare_dataset(
                [source],
                output_images,
                output_masks,
                manifest,
                validate_only=True,
                overwrite=False,
            )

            self.assertFalse(report.written)
            self.assertFalse(output_images.exists())
            self.assertFalse(output_masks.exists())
            self.assertFalse(manifest.exists())

    def test_staging_directory_is_a_normal_sibling(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            destination = root / "combined_images"

            staging = multiview.create_staging_directory(destination)
            try:
                self.assertEqual(staging.parent, destination.parent)
                self.assertTrue(staging.is_dir())
                self.assertTrue(staging.name.startswith(".combined_images-"))
                self.assertTrue(staging.name.endswith(".tmp"))
            finally:
                staging.rmdir()

    def test_publish_staged_output_copies_when_windows_rename_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            staging = root / ".combined_images-test.tmp"
            destination = root / "combined_images"
            staging.mkdir()
            (staging / "frame.jpg").write_bytes(b"complete staged output")

            with mock.patch.object(multiview.os, "name", "nt"), mock.patch.object(
                Path, "replace", side_effect=PermissionError(13, "Access is denied")
            ):
                multiview.publish_staged_output(staging, destination)

            self.assertFalse(staging.exists())
            self.assertEqual(
                (destination / "frame.jpg").read_bytes(), b"complete staged output"
            )

    def test_missing_mask_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            source = self.create_source(root, "side")
            (source.masks / "frame_000006.jpg.png").unlink()

            with self.assertRaisesRegex(
                multiview.DatasetPreparationError, "missing 1 COLMAP mask"
            ):
                multiview.collect_pairs([source])

    def test_dimension_mismatch_is_rejected(self) -> None:
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            source = self.create_source(root, "side")
            replacement = Image.new("L", (5, 5), 0)
            ImageDraw.Draw(replacement).rectangle((1, 1, 3, 3), fill=255)
            replacement.save(source.masks / "frame_000000.jpg.png")
            replacement.close()

            with self.assertRaisesRegex(
                multiview.DatasetPreparationError, "Dimension mismatch"
            ):
                multiview.collect_pairs([source])

    def test_explicit_exclusions_preserve_sources_and_skip_bad_pairs(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            source = self.create_source(root, "side")
            excluded = multiview.SourceSpec(
                name=source.name,
                images=source.images,
                masks=source.masks,
                exclude=("frame_000006.jpg",),
            )

            pairs = multiview.collect_pairs([excluded])

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].image.name, "frame_000000.jpg")
            self.assertTrue((source.images / "frame_000006.jpg").is_file())
            self.assertTrue((source.masks / "frame_000006.jpg.png").is_file())

    def test_unknown_exclusion_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            source = self.create_source(root, "side")
            excluded = multiview.SourceSpec(
                name=source.name,
                images=source.images,
                masks=source.masks,
                exclude=("missing.jpg",),
            )

            with self.assertRaisesRegex(
                multiview.DatasetPreparationError, "were not found"
            ):
                multiview.collect_pairs([excluded])

    def test_existing_output_is_not_overwritten_by_default(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            source = self.create_source(root, "side")
            output_images = root / "combined_images"
            output_images.mkdir()

            with self.assertRaisesRegex(
                multiview.DatasetPreparationError, "already exists"
            ):
                multiview.prepare_dataset(
                    [source],
                    output_images,
                    root / "combined_masks",
                    root / "manifest.csv",
                    validate_only=False,
                    overwrite=False,
                )

    def test_output_parent_of_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            source = self.create_source(root, "side")

            with self.assertRaisesRegex(
                multiview.DatasetPreparationError, "must be separate"
            ):
                multiview.prepare_dataset(
                    [source],
                    root,
                    root / "combined" / "masks",
                    root / "combined" / "manifest.csv",
                    validate_only=True,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()
