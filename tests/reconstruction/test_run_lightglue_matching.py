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
import run_lightglue_matching as runner  # noqa: E402


class RunLightGlueMatchingTests(unittest.TestCase):
    def create_inputs(self, root: Path) -> tuple[Path, Path, Path]:
        manifest = root / "manifest.csv"
        images = (
            (1, "side_0.jpg", "side", 0),
            (2, "side_1.jpg", "side", 1),
            (3, "top45_0.jpg", "top45", 0),
            (4, "top45_1.jpg", "top45", 1),
        )
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("view", "view_frame_index", "combined_filename"),
            )
            writer.writeheader()
            for _, filename, view, index in images:
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
            CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT UNIQUE);
            CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER);
            CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY, rows INTEGER);
            CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER);
            CREATE TABLE two_view_geometries (
                pair_id INTEGER PRIMARY KEY,
                rows INTEGER
            );
            """
        )
        for image_id, filename, _, _ in images:
            connection.execute("INSERT INTO images VALUES (?, ?)", (image_id, filename))
            connection.execute("INSERT INTO keypoints VALUES (?, 20)", (image_id,))
            connection.execute("INSERT INTO descriptors VALUES (?, 20)", (image_id,))
        connection.commit()
        connection.close()

        pairs = root / "pairs.txt"
        pairs.write_text(
            "side_0.jpg top45_0.jpg\nside_1.jpg top45_1.jpg\n",
            encoding="utf-8",
        )
        return database, manifest, pairs

    def create_config(
        self,
        root: Path,
        database: Path,
        manifest: Path,
        pairs: Path,
    ) -> dict[str, object]:
        executable = root / "colmap.exe"
        executable.write_bytes(b"test executable")
        cuda_dll_directory = root / "torch" / "lib"
        cuda_dll_directory.mkdir(parents=True)
        required_cuda_dlls = (
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "cudart64_12.dll",
            "cudnn64_9.dll",
        )
        for name in required_cuda_dlls:
            (cuda_dll_directory / name).write_bytes(b"test DLL")
        return {
            "colmap": {"executable": str(executable)},
            "runtime": {
                "cuda_dll_directory": str(cuda_dll_directory),
                "minimum_cuda_version": "12.8",
                "minimum_cudnn_version": 90500,
                "required_cuda_dlls": list(required_cuda_dlls),
            },
            "input": {
                "database": str(database),
                "match_pairs": str(pairs),
                "manifest": str(manifest),
            },
            "output": {"log": str(root / "logs" / "matching.log")},
            "matching": {
                "feature_type": "SIFT_LIGHTGLUE",
                "use_gpu": True,
                "gpu_index": 0,
                "num_threads": 1,
                "max_num_matches": 4096,
                "guided_matching": True,
                "lightglue_min_score": 0.1,
                "min_num_inliers": 15,
                "expected_pair_count": 2,
            },
        }

    def test_plan_uses_lightglue_and_low_memory_settings(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            database, manifest, pairs = self.create_inputs(root)
            plan = runner.build_plan(
                self.create_config(root, database, manifest, pairs)
            )

            self.assertEqual(plan.pair_count, 2)
            self.assertIn("SIFT_LIGHTGLUE", plan.command)
            self.assertIn("--FeatureMatching.num_threads", plan.command)
            self.assertIn("--FeatureMatching.max_num_matches", plan.command)
            self.assertEqual(plan.command[0], str(root / "colmap.exe"))
            self.assertEqual(plan.cuda_dll_directory, root / "torch" / "lib")

    def test_non_bridge_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            database, manifest, pairs = self.create_inputs(root)
            pairs.write_text("side_0.jpg side_1.jpg\n", encoding="utf-8")
            config = self.create_config(root, database, manifest, pairs)
            config["matching"]["expected_pair_count"] = 1  # type: ignore[index]

            with self.assertRaisesRegex(
                runner.LightGlueRunnerError, "not side<->top45"
            ):
                runner.build_plan(config)

    def test_stale_bridge_records_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            database, manifest, pairs = self.create_inputs(root)
            connection = sqlite3.connect(database)
            pair_id = cleanup.pair_id_from_image_ids(1, 3)
            connection.execute("INSERT INTO matches VALUES (?, 0)", (pair_id,))
            connection.commit()
            connection.close()

            with self.assertRaisesRegex(
                runner.LightGlueRunnerError, "run cleanup before matching"
            ):
                runner.build_plan(
                    self.create_config(root, database, manifest, pairs)
                )

    def test_run_streams_to_new_log_and_checks_exit_code(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            database, manifest, pairs = self.create_inputs(root)
            plan = runner.build_plan(
                self.create_config(root, database, manifest, pairs)
            )
            with mock.patch.object(runner, "stream_process", return_value=0) as process:
                runner.run_matching(plan)

            process.assert_called_once()
            self.assertTrue(plan.log.is_file())

    def test_child_environment_prepends_cuda_dll_directory(self) -> None:
        cuda_dll_directory = Path("C:/example/torch/lib")

        environment = runner.child_environment(cuda_dll_directory)

        self.assertEqual(
            environment["PATH"].split(runner.os.pathsep)[0],
            str(cuda_dll_directory),
        )

    def test_existing_log_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            database, manifest, pairs = self.create_inputs(root)
            config = self.create_config(root, database, manifest, pairs)
            log = root / "logs" / "matching.log"
            log.parent.mkdir()
            log.write_text("old run\n", encoding="utf-8")

            with self.assertRaisesRegex(
                runner.LightGlueRunnerError, "refusing to overwrite"
            ):
                runner.build_plan(config)


if __name__ == "__main__":
    unittest.main()
