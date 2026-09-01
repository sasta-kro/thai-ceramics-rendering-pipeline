from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gaussian_splatting import environment_doctor as old_doctor  # noqa: E402
from scripts.gaussian_splatting import evaluate_checkpoint as old_evaluate  # noqa: E402
from scripts.gaussian_splatting import export_checkpoint as old_export  # noqa: E402
from scripts.gaussian_splatting import prepare_dataset as old_prepare  # noqa: E402
from scripts.gaussian_splatting import run_training as old_train  # noqa: E402
from scripts.gaussian_splatting import validate_dataset as old_validate  # noqa: E402
from scripts.gaussian_splatting import view_checkpoint as old_view  # noqa: E402
from scripts.gaussian_splatting.cli import doctor  # noqa: E402
from scripts.gaussian_splatting.cli import evaluate  # noqa: E402
from scripts.gaussian_splatting.cli import export  # noqa: E402
from scripts.gaussian_splatting.cli import prepare  # noqa: E402
from scripts.gaussian_splatting.cli import train  # noqa: E402
from scripts.gaussian_splatting.cli import validate  # noqa: E402
from scripts.gaussian_splatting.cli import view  # noqa: E402


class CliCompatibilityTests(unittest.TestCase):
    def test_old_wrappers_forward_to_new_cli_modules(self) -> None:
        pairs = (
            (old_doctor.main, doctor.main),
            (old_validate.main, validate.main),
            (old_prepare.main, prepare.main),
            (old_train.main, train.main),
            (old_evaluate.main, evaluate.main),
            (old_export.main, export.main),
            (old_view.main, view.main),
        )
        for old_main, new_main in pairs:
            with self.subTest(entry_point=old_main.__module__):
                self.assertIs(old_main, new_main)


if __name__ == "__main__":
    unittest.main()
