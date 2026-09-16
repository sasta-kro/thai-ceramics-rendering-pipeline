from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "reconstruction_all_angles"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_incremental_feature_extraction as extraction  # noqa: E402


class RunIncrementalFeatureExtractionTests(unittest.TestCase):
    def create_database(self, path: Path, base_count: int = 728) -> list[str]:
        connection = sqlite3.connect(path)
        connection.executescript(
            """
            CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY);
            CREATE TABLE images (
                image_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                camera_id INTEGER NOT NULL
            );
            CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER);
            CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY, rows INTEGER);
            CREATE TABLE matches (pair_id INTEGER PRIMARY KEY);
            CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY);
            """
        )
        for camera_id in (1, 2, 3):
            connection.execute("INSERT INTO cameras VALUES (?)", (camera_id,))
        names = [f"side_frame_{index:06d}.jpg" for index in range(base_count)]
        for image_id, name in enumerate(names, start=1):
            camera_id = 1 + (image_id - 1) % 3
            connection.execute(
                "INSERT INTO images VALUES (?, ?, ?)", (image_id, name, camera_id)
            )
            connection.execute("INSERT INTO keypoints VALUES (?, 100)", (image_id,))
            connection.execute("INSERT INTO descriptors VALUES (?, 100)", (image_id,))
        connection.execute("INSERT INTO matches VALUES (1)")
        connection.execute("INSERT INTO two_view_geometries VALUES (1)")
        connection.commit()
        connection.close()
        return names

    def create_inputs(self, root: Path) -> tuple[Path, Path, Path, tuple[str, ...]]:
        workspace = root / "workspace"
        images = root / "images"
        masks = root / "masks"
        (workspace / "logs").mkdir(parents=True)
        images.mkdir()
        masks.mkdir()
        self.create_database(workspace / "database.db")
        new_names = tuple(f"mid25_frame_{index:06d}.jpg" for index in range(4))
        (workspace / "new_view_images.txt").write_text(
            "\n".join(new_names) + "\n", encoding="utf-8"
        )
        for name in new_names:
            (images / name).write_bytes(b"rgb")
            (masks / f"{name}.png").write_bytes(b"mask")
        return workspace, images, masks, new_names

    def test_build_plan_targets_only_new_images(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            workspace, images, masks, new_names = self.create_inputs(root)
            colmap = root / "colmap.exe"
            colmap.write_bytes(b"executable")

            plan = extraction.build_plan(
                colmap=colmap, workspace=workspace, images=images, masks=masks
            )

            self.assertEqual(plan.new_names, new_names)
            self.assertEqual(len(plan.before.images), 728)
            self.assertIn("--image_list_path", plan.command)
            self.assertIn(str(workspace / "new_view_images.txt"), plan.command)

    def test_rejects_new_image_already_in_database(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            workspace, images, masks, new_names = self.create_inputs(root)
            connection = sqlite3.connect(workspace / "database.db")
            connection.execute(
                "INSERT INTO images VALUES (729, ?, 1)", (new_names[0],)
            )
            connection.commit()
            connection.close()
            colmap = root / "colmap.exe"
            colmap.write_bytes(b"executable")

            with self.assertRaisesRegex(
                extraction.IncrementalFeatureError, "already contains"
            ):
                extraction.build_plan(
                    colmap=colmap, workspace=workspace, images=images, masks=masks
                )

    def test_validate_result_preserves_base_and_accepts_new_camera(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            workspace, images, masks, new_names = self.create_inputs(root)
            colmap = root / "colmap.exe"
            colmap.write_bytes(b"executable")
            plan = extraction.build_plan(
                colmap=colmap, workspace=workspace, images=images, masks=masks
            )
            connection = sqlite3.connect(workspace / "database.db")
            connection.execute("INSERT INTO cameras VALUES (4)")
            for offset, name in enumerate(new_names, start=729):
                connection.execute("INSERT INTO images VALUES (?, ?, 4)", (offset, name))
                connection.execute("INSERT INTO keypoints VALUES (?, 200)", (offset,))
                connection.execute("INSERT INTO descriptors VALUES (?, 200)", (offset,))
            connection.commit()
            connection.close()

            report = extraction.validate_result(plan)

            self.assertEqual(report["total_images"], 732)
            self.assertEqual(report["total_cameras"], 4)
            self.assertEqual(report["new_camera_id"], 4)
            self.assertEqual(report["new_keypoints_total"], 800)

    def test_validate_result_rejects_base_feature_change(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            workspace, images, masks, new_names = self.create_inputs(root)
            colmap = root / "colmap.exe"
            colmap.write_bytes(b"executable")
            plan = extraction.build_plan(
                colmap=colmap, workspace=workspace, images=images, masks=masks
            )
            connection = sqlite3.connect(workspace / "database.db")
            connection.execute("INSERT INTO cameras VALUES (4)")
            for offset, name in enumerate(new_names, start=729):
                connection.execute("INSERT INTO images VALUES (?, ?, 4)", (offset, name))
                connection.execute("INSERT INTO keypoints VALUES (?, 200)", (offset,))
                connection.execute("INSERT INTO descriptors VALUES (?, 200)", (offset,))
            connection.execute("UPDATE keypoints SET rows = 99 WHERE image_id = 1")
            connection.commit()
            connection.close()

            with self.assertRaisesRegex(
                extraction.IncrementalFeatureError, "base SIFT features changed"
            ):
                extraction.validate_result(plan)


if __name__ == "__main__":
    unittest.main()
