from __future__ import annotations

from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts/reconstruction_all_angles"
sys.path.insert(0, str(SCRIPTS_DIR))

import prepare_incremental_view_workspace as preparation  # noqa: E402


class PrepareFilteredWorkspaceTests(unittest.TestCase):
    def test_pair_id_decode(self) -> None:
        pair_id = 4 * preparation.MAX_IMAGE_ID + 19
        self.assertEqual(preparation.image_ids_from_pair_id(pair_id), (4, 19))

    def test_filter_bridge_pairs_keeps_only_retained_names(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            path = Path(directory) / "pairs.txt"
            side_mid = [f"side_{index}.jpg mid25_{index}.jpg" for index in range(2520)]
            path.write_text(
                "\n".join(side_mid + ["mid25_0.jpg top45_0.jpg"]) + "\n",
                encoding="utf-8",
            )
            retained = {
                *(f"side_{index}.jpg" for index in range(2520)),
                *(f"mid25_{index}.jpg" for index in range(2520)),
            }
            result = preparation.filter_bridge_pairs(path, retained)
        self.assertEqual(len(result), 2520)
        self.assertNotIn("top45", "\n".join(result))

    def test_filter_database_removes_only_excluded_image_records(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            database = Path(directory) / "database.db"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER);
                CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER);
                CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY, rows INTEGER);
                CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER);
                CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER);
                CREATE TABLE frames (frame_id INTEGER PRIMARY KEY, rig_id INTEGER);
                CREATE TABLE frame_data (
                    frame_id INTEGER, data_id INTEGER, sensor_id INTEGER, sensor_type INTEGER
                );
                CREATE TABLE pose_priors (
                    corr_data_id INTEGER, corr_sensor_id INTEGER, corr_sensor_type INTEGER
                );
                """
            )
            for image_id, name in (
                (1, "side_a.jpg"),
                (2, "top45_a.jpg"),
                (3, "mid25_a.jpg"),
            ):
                connection.execute("INSERT INTO images VALUES (?, ?, 1)", (image_id, name))
                connection.execute("INSERT INTO keypoints VALUES (?, 10)", (image_id,))
                connection.execute("INSERT INTO descriptors VALUES (?, 10)", (image_id,))
                connection.execute("INSERT INTO frames VALUES (?, 1)", (image_id,))
                connection.execute(
                    "INSERT INTO frame_data VALUES (?, ?, 1, 0)", (image_id, image_id)
                )
            kept_pair = 1 * preparation.MAX_IMAGE_ID + 3
            stale_pair = 1 * preparation.MAX_IMAGE_ID + 2
            for table in ("matches", "two_view_geometries"):
                connection.execute(f"INSERT INTO {table} VALUES (?, 20)", (kept_pair,))
                connection.execute(f"INSERT INTO {table} VALUES (?, 30)", (stale_pair,))
            connection.commit()
            connection.close()

            preparation.filter_database(database, {2})

            connection = sqlite3.connect(database)
            try:
                names = [row[0] for row in connection.execute("SELECT name FROM images")]
                self.assertEqual(names, ["side_a.jpg", "mid25_a.jpg"])
                for table in ("matches", "two_view_geometries"):
                    rows = list(connection.execute(f"SELECT pair_id, rows FROM {table}"))
                    self.assertEqual(rows, [(kept_pair, 20)])
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0], 2
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0], 2
                )
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
