from __future__ import annotations

import importlib.util
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/reconstruction/configure_lightglue_repair_database.py"
)
SPEC = importlib.util.spec_from_file_location(
    "configure_lightglue_repair_database", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DatabaseConfigurationTests(unittest.TestCase):
    def test_image_group(self) -> None:
        self.assertEqual(MODULE.image_group("side_frame_000006.jpg"), "side")
        self.assertEqual(MODULE.image_group("top45_frame_000006.jpg"), "top45")
        self.assertEqual(
            MODULE.image_group("underside_frame_000006.jpg"), "underside"
        )
        with self.assertRaises(MODULE.DatabaseConfigurationError):
            MODULE.image_group("unknown.jpg")

    def test_camera_blob_is_little_endian_double(self) -> None:
        camera = MODULE.CameraDefinition(1, 2, 10, 20, (1.5, 2.5))
        self.assertEqual(len(camera.params_blob), 16)

    def test_configure_database_assigns_three_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "database.db"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE cameras (
                    camera_id INTEGER PRIMARY KEY NOT NULL,
                    model INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    params BLOB,
                    prior_focal_length INTEGER NOT NULL
                );
                CREATE TABLE images (
                    image_id INTEGER PRIMARY KEY NOT NULL,
                    name TEXT NOT NULL,
                    camera_id INTEGER NOT NULL
                );
                CREATE TABLE rigs (
                    rig_id INTEGER PRIMARY KEY NOT NULL,
                    ref_sensor_id INTEGER NOT NULL,
                    ref_sensor_type INTEGER NOT NULL
                );
                CREATE TABLE frames (
                    frame_id INTEGER PRIMARY KEY NOT NULL,
                    rig_id INTEGER NOT NULL
                );
                CREATE TABLE frame_data (
                    frame_id INTEGER NOT NULL,
                    data_id INTEGER NOT NULL,
                    sensor_id INTEGER NOT NULL,
                    sensor_type INTEGER NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT INTO cameras VALUES (1, 2, 2160, 3840, ?, 0)",
                (b"old",),
            )
            connection.execute("INSERT INTO rigs VALUES (1, 1, 0)")
            image_id = 0
            for prefix, count in MODULE.EXPECTED_GROUP_COUNTS.items():
                for index in range(count):
                    image_id += 1
                    connection.execute(
                        "INSERT INTO images VALUES (?, ?, 1)",
                        (image_id, f"{prefix}_frame_{index:06}.jpg"),
                    )
                    connection.execute(
                        "INSERT INTO frames VALUES (?, 1)", (image_id,)
                    )
                    connection.execute(
                        "INSERT INTO frame_data VALUES (?, ?, 1, 0)",
                        (image_id, image_id),
                    )
            connection.commit()
            connection.close()
            cameras = (
                MODULE.CameraDefinition(1, 2, 2160, 3840, (1.0, 2.0, 3.0, 4.0)),
                MODULE.CameraDefinition(2, 2, 2160, 3840, (5.0, 2.0, 3.0, 4.0)),
                MODULE.CameraDefinition(3, 2, 2160, 3840, (6.0, 2.0, 3.0, 4.0)),
            )

            MODULE.configure_database(database, cameras, dry_run=False)

            connection = sqlite3.connect(database)
            self.assertEqual(
                dict(
                    connection.execute(
                        "SELECT camera_id, COUNT(*) FROM images GROUP BY camera_id"
                    )
                ),
                {1: 273, 2: 233, 3: 222},
            )
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM rigs").fetchone()[0], 3)
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM frames WHERE rig_id NOT IN (1, 2, 3)"
                ).fetchone()[0],
                0,
            )
            connection.close()


if __name__ == "__main__":
    unittest.main()
