from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "reconstruction_all_angles"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_incremental_sift_matching as matching  # noqa: E402


class RunIncrementalSiftMatchingTests(unittest.TestCase):
    def create_inputs(self, root: Path) -> tuple[Path, tuple[str, ...]]:
        workspace = root / "workspace"
        (workspace / "logs").mkdir(parents=True)
        database = workspace / "database.db"
        connection = sqlite3.connect(database)
        connection.executescript(
            """
            CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT UNIQUE);
            CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER);
            CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY, rows INTEGER);
            CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER);
            CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER);
            """
        )
        names = tuple(f"mid25_frame_{index:06d}.jpg" for index in range(949))
        for image_id, name in enumerate(names, start=1):
            connection.execute("INSERT INTO images VALUES (?, ?)", (image_id, name))
            connection.execute("INSERT INTO keypoints VALUES (?, 100)", (image_id,))
            connection.execute("INSERT INTO descriptors VALUES (?, 100)", (image_id,))
        connection.execute("INSERT INTO matches VALUES (1, 50)")
        connection.execute("INSERT INTO two_view_geometries VALUES (1, 40)")
        connection.commit()
        connection.close()
        # Reuse image names cyclically while keeping canonical pair IDs unique.
        pairs = tuple(
            (names[index % 900], names[900 + index // 900]) for index in range(3420)
        )
        (workspace / "new_view_sequential_pairs.txt").write_text(
            "".join(f"{left} {right}\n" for left, right in pairs), encoding="utf-8"
        )
        return workspace, names

    def test_pair_id_is_order_independent(self) -> None:
        self.assertEqual(matching.pair_id(4, 9), matching.pair_id(9, 4))

    def test_build_plan_validates_3420_fresh_pairs(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            workspace, _ = self.create_inputs(root)
            colmap = root / "colmap.exe"
            colmap.write_bytes(b"executable")
            plan = matching.build_plan(colmap=colmap, workspace=workspace)
            self.assertEqual(len(plan.pairs), 3420)
            self.assertIn("SIFT_BRUTEFORCE", plan.command)

    def test_failed_log_uses_separate_retry_log(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            workspace, _ = self.create_inputs(root)
            (workspace / "logs/incremental_sift_matching.log").write_text(
                "parse failure\n", encoding="utf-8"
            )
            colmap = root / "colmap.exe"
            colmap.write_bytes(b"executable")
            plan = matching.build_plan(colmap=colmap, workspace=workspace)
            self.assertEqual(plan.log.name, "incremental_sift_matching_retry1.log")

    def test_rejects_stale_target_pair(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            workspace, names = self.create_inputs(root)
            connection = sqlite3.connect(workspace / "database.db")
            stale = matching.pair_id(1, 901)
            connection.execute("INSERT INTO matches VALUES (?, 20)", (stale,))
            connection.commit()
            connection.close()
            colmap = root / "colmap.exe"
            colmap.write_bytes(b"executable")
            with self.assertRaisesRegex(
                matching.IncrementalMatchingError, "already contains"
            ):
                matching.build_plan(colmap=colmap, workspace=workspace)

    def test_validate_result_reports_verified_pairs(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            workspace, _ = self.create_inputs(root)
            colmap = root / "colmap.exe"
            colmap.write_bytes(b"executable")
            plan = matching.build_plan(colmap=colmap, workspace=workspace)
            connection = sqlite3.connect(workspace / "database.db")
            for index, record_id in enumerate(plan.pair_ids):
                rows = 25 if index < 3000 else 5
                connection.execute("INSERT INTO matches VALUES (?, ?)", (record_id, rows))
                connection.execute(
                    "INSERT INTO two_view_geometries VALUES (?, ?)", (record_id, rows)
                )
            connection.commit()
            connection.close()
            report = matching.validate_result(plan)
            self.assertEqual(report["planned_pairs"], 3420)
            self.assertEqual(report["verified_pairs_15plus"], 3000)


if __name__ == "__main__":
    unittest.main()
