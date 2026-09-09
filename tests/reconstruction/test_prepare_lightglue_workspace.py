from __future__ import annotations

import csv
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "reconstruction"
sys.path.insert(0, str(SCRIPTS_DIR))

import prepare_lightglue_workspace as lightglue  # noqa: E402


class PrepareLightGlueWorkspaceTests(unittest.TestCase):
    def create_sources(
        self,
        root: Path,
        *,
        side_count: int = 7,
        top_count: int = 6,
    ) -> tuple[Path, Path, Path, set[str]]:
        manifest = root / "dataset_manifest.csv"
        rows = []
        filenames: set[str] = set()
        for view, count in (("side", side_count), ("top45", top_count)):
            for index in range(count):
                filename = f"{view}_frame_{index * 6:06d}.jpg"
                filenames.add(filename)
                rows.append(
                    {
                        "view": view,
                        "view_frame_index": index,
                        "combined_filename": filename,
                    }
                )
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("view", "view_frame_index", "combined_filename"),
            )
            writer.writeheader()
            writer.writerows(rows)

        database = root / "database.db"
        connection = sqlite3.connect(database)
        connection.executescript(
            """
            CREATE TABLE images (
                image_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE keypoints (
                image_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL
            );
            CREATE TABLE descriptors (
                image_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL
            );
            """
        )
        for image_id, filename in enumerate(sorted(filenames), start=1):
            connection.execute("INSERT INTO images VALUES (?, ?)", (image_id, filename))
            connection.execute("INSERT INTO keypoints VALUES (?, 20)", (image_id,))
            connection.execute("INSERT INTO descriptors VALUES (?, 20)", (image_id,))
        connection.commit()
        connection.close()

        model = root / "sparse" / "2"
        model.mkdir(parents=True)
        for name in lightglue.REQUIRED_MODEL_FILES:
            (model / name).write_bytes(name.encode("ascii"))
        (model / "project.ini").write_text("model=2\n", encoding="utf-8")
        return manifest, database, model, filenames

    def build_plan(
        self, root: Path, manifest: Path, database: Path, model: Path, stride: int = 5
    ) -> lightglue.PreparationPlan:
        return lightglue.build_plan(
            source_database=database,
            source_model=model,
            manifest=manifest,
            workspace=root / "lightglue",
            left_view="side",
            right_view="top45",
            stride=stride,
        )

    def test_real_sequence_sizes_produce_2688_pairs(self) -> None:
        side = tuple(
            lightglue.ManifestFrame("side", index, f"side_{index}.jpg")
            for index in range(273)
        )
        top = tuple(
            lightglue.ManifestFrame("top45", index, f"top45_{index}.jpg")
            for index in range(233)
        )

        pairs = lightglue.build_bridge_pairs(side, top, stride=5)

        self.assertEqual(len(lightglue.anchor_frames(side, 5)), 56)
        self.assertEqual(len(lightglue.anchor_frames(top, 5)), 48)
        self.assertEqual(len(pairs), 2688)
        self.assertEqual(pairs[0], ("side_0.jpg", "top45_0.jpg"))
        self.assertEqual(pairs[-1], ("side_272.jpg", "top45_232.jpg"))
        self.assertEqual(len(set(pairs)), len(pairs))

    def test_prepares_isolated_workspace_without_changing_sources(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            manifest, database, model, _ = self.create_sources(root)
            database_bytes = database.read_bytes()
            model_bytes = (model / "images.bin").read_bytes()
            plan = self.build_plan(root, manifest, database, model, stride=3)

            lightglue.prepare_workspace(plan)

            workspace = root / "lightglue"
            self.assertEqual((workspace / "database.db").read_bytes(), database_bytes)
            self.assertEqual(
                (workspace / "base_model" / "images.bin").read_bytes(), model_bytes
            )
            self.assertEqual(database.read_bytes(), database_bytes)
            self.assertEqual((model / "images.bin").read_bytes(), model_bytes)
            self.assertTrue((workspace / "bridge_pairs_lightglue.txt").is_file())
            self.assertFalse((workspace / "database.db-wal").exists())
            self.assertFalse((workspace / "database.db-shm").exists())
            self.assertEqual(
                len(
                    (workspace / "bridge_pairs_lightglue.txt")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                len(plan.pairs),
            )
            for output_directory in lightglue.OUTPUT_DIRECTORIES:
                self.assertTrue((workspace / output_directory).is_dir())

    def test_existing_workspace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            manifest, database, model, _ = self.create_sources(root)
            (root / "lightglue").mkdir()

            with self.assertRaisesRegex(
                lightglue.WorkspacePreparationError, "refusing to overwrite"
            ):
                self.build_plan(root, manifest, database, model)

    def test_manifest_database_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            manifest, database, model, _ = self.create_sources(root)
            connection = sqlite3.connect(database)
            connection.execute("DELETE FROM descriptors WHERE image_id = 1")
            connection.commit()
            connection.close()

            with self.assertRaisesRegex(
                lightglue.WorkspacePreparationError, "SIFT features are incomplete"
            ):
                self.build_plan(root, manifest, database, model)

    def test_nonpositive_stride_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            lightglue.WorkspacePreparationError, "positive integer"
        ):
            lightglue.anchor_frames((), 0)


if __name__ == "__main__":
    unittest.main()
