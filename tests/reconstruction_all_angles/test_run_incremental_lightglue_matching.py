from __future__ import annotations

import csv
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts/reconstruction_all_angles"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_incremental_lightglue_matching as lightglue  # noqa: E402


class RunIncrementalLightGlueMatchingTests(unittest.TestCase):
    def test_parse_bridges_accepts_two_groups(self) -> None:
        result = lightglue.parse_bridges(
            {
                "bridges": [
                    {"left_view": "side", "right_view": "mid25", "expected_pair_count": 2},
                    {"left_view": "mid25", "right_view": "top45", "expected_pair_count": 3},
                ]
            }
        )
        self.assertEqual(result["mid25<->side"], 2)
        self.assertEqual(result["mid25<->top45"], 3)

    def test_validate_pairs_checks_each_bridge_count(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            path = Path(directory) / "pairs.txt"
            path.write_text(
                "side.jpg mid_a.jpg\nside.jpg mid_b.jpg\nmid_a.jpg top.jpg\n",
                encoding="utf-8",
            )
            views = {
                "side.jpg": "side",
                "mid_a.jpg": "mid25",
                "mid_b.jpg": "mid25",
                "top.jpg": "top45",
            }
            pairs = lightglue.validate_pairs(
                path, views, {"mid25<->side": 2, "mid25<->top45": 1}
            )
            self.assertEqual(len(pairs), 3)

    def test_validate_pairs_rejects_unexpected_group(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            path = Path(directory) / "pairs.txt"
            path.write_text("side.jpg top.jpg\n", encoding="utf-8")
            with self.assertRaisesRegex(
                lightglue.LightGlueRunnerError, "unexpected bridge group"
            ):
                lightglue.validate_pairs(
                    path,
                    {"side.jpg": "side", "top.jpg": "top45"},
                    {"mid25<->side": 1},
                )

    def test_next_output_paths_preserves_failed_log(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            base = Path(directory) / "matching.log"
            base.write_text("failed\n", encoding="utf-8")
            log, report = lightglue.next_output_paths(base)
            self.assertEqual(log.name, "matching_retry1.log")
            self.assertEqual(report.name, "matching_retry1_report.json")


if __name__ == "__main__":
    unittest.main()
