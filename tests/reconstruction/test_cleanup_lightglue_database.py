from __future__ import annotations

import csv
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "reconstruction"
sys.path.insert(0, str(SCRIPTS_DIR))

import cleanup_lightglue_database as cleanup  # noqa: E402


class CleanupLightGlueDatabaseTests(unittest.TestCase):
    def create_database(self, root: Path) -> tuple[Path, Path]:
        filenames = (
            (1, "side_frame_000000.jpg", "side", 0),
            (2, "side_frame_000006.jpg", "side", 1),
            (3, "top45_frame_000000.jpg", "top45", 0),
            (4, "top45_frame_000006.jpg", "top45", 1),
            (5, "underside_frame_000000.jpg", "underside", 0),
            (6, "underside_frame_000006.jpg", "underside", 1),
        )
        manifest = root / "dataset_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("view", "view_frame_index", "combined_filename"),
            )
            writer.writeheader()
            for _, filename, view, index in filenames:
                writer.writerow(
                    {
                        "view": view,
                        "view_frame_index": index,
                        "combined_filename": filename,
                    }
                )

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
            CREATE TABLE matches (
                pair_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL
            );
            CREATE TABLE two_view_geometries (
                pair_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL
            );
            """
        )
        for image_id, filename, _, _ in filenames:
            connection.execute("INSERT INTO images VALUES (?, ?)", (image_id, filename))
            connection.execute("INSERT INTO keypoints VALUES (?, 20)", (image_id,))
            connection.execute("INSERT INTO descriptors VALUES (?, 20)", (image_id,))

        pairs = (
            (1, 2, 11),
            (3, 4, 12),
            (5, 6, 13),
            (1, 3, 3),
            (2, 4, 0),
            (1, 5, 14),
        )
        for image_id1, image_id2, rows in pairs:
            pair_id = cleanup.pair_id_from_image_ids(image_id1, image_id2)
            connection.execute("INSERT INTO matches VALUES (?, ?)", (pair_id, rows))
            geometry_rows = 0 if {image_id1, image_id2} <= {1, 2, 3, 4} else rows
            connection.execute(
                "INSERT INTO two_view_geometries VALUES (?, ?)",
                (pair_id, geometry_rows),
            )
        connection.commit()
        connection.close()
        return database, manifest

    def run_cleanup(
        self,
        database: Path,
        manifest: Path,
        *,
        dry_run: bool,
        report_path: Path | None = None,
    ) -> cleanup.CleanupResult:
        return cleanup.cleanup_database(
            database=database,
            manifest=manifest,
            left_view="side",
            right_view="top45",
            dry_run=dry_run,
            report_path=report_path,
        )

    def test_dry_run_changes_nothing(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            database, manifest = self.create_database(root)
            original = database.read_bytes()

            result = self.run_cleanup(database, manifest, dry_run=True)

            target = result.before.match_groups["matches"]["side<->top45"]
            self.assertEqual(target, cleanup.GroupCounts(records=2, rows=3))
            self.assertEqual(database.read_bytes(), original)
            self.assertEqual(result.before, result.after)

    def test_cleanup_removes_only_target_group_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            database, manifest = self.create_database(root)
            report = root / "logs" / "cleanup.json"

            result = self.run_cleanup(
                database, manifest, dry_run=False, report_path=report
            )

            self.assertEqual(result.after.images, 6)
            self.assertEqual(result.after.keypoint_rows, 120)
            self.assertEqual(result.after.descriptor_rows, 120)
            for table in cleanup.MATCH_TABLES:
                self.assertNotIn("side<->top45", result.after.match_groups[table])
                before_preserved = dict(result.before.match_groups[table])
                before_preserved.pop("side<->top45")
                self.assertEqual(before_preserved, result.after.match_groups[table])
            self.assertTrue(report.is_file())
            self.assertFalse(Path(f"{database}-wal").exists())
            self.assertFalse(Path(f"{database}-shm").exists())

    def test_validation_failure_rolls_back_deletions(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            database, manifest = self.create_database(root)
            before = self.run_cleanup(database, manifest, dry_run=True).before

            with mock.patch.object(
                cleanup,
                "validate_transition",
                side_effect=cleanup.DatabaseCleanupError("forced validation failure"),
            ):
                with self.assertRaisesRegex(
                    cleanup.DatabaseCleanupError, "forced validation failure"
                ):
                    self.run_cleanup(database, manifest, dry_run=False)

            after = self.run_cleanup(database, manifest, dry_run=True).before
            self.assertEqual(after, before)

    def test_second_cleanup_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            database, manifest = self.create_database(root)
            self.run_cleanup(database, manifest, dry_run=False)

            with self.assertRaisesRegex(
                cleanup.DatabaseCleanupError, "No side<->top45 records remain"
            ):
                self.run_cleanup(database, manifest, dry_run=False)

    def test_pair_id_roundtrip(self) -> None:
        pair_id = cleanup.pair_id_from_image_ids(27, 4)
        self.assertEqual(cleanup.image_ids_from_pair_id(pair_id), (4, 27))


if __name__ == "__main__":
    unittest.main()
