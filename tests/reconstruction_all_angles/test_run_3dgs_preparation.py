from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/reconstruction_all_angles/run_3dgs_preparation.py"


class ThreeAnglePreparationWrapperTests(unittest.TestCase):
    def test_help_exposes_shared_stages(self) -> None:
        completed = subprocess.run(
            (sys.executable, str(SCRIPT), "--help"),
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("undistort-rgb", completed.stdout)
        self.assertIn("finalize-masks", completed.stdout)


if __name__ == "__main__":
    unittest.main()
