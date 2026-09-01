from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gaussian_splatting.diagnostics import environment as environment_doctor  # noqa: E402


class EnvironmentDoctorTests(unittest.TestCase):
    def test_pinned_versions_are_compatible(self) -> None:
        self.assertEqual(
            environment_doctor.compatibility_problems(
                (3, 10), "2.4.1+cu124", "12.4", "1.5.3+pt24cu124"
            ),
            [],
        )

    def test_wrong_python_and_cuda_are_reported(self) -> None:
        problems = environment_doctor.compatibility_problems(
            (3, 12), "2.4.1+cu124", "12.6", "1.5.3+pt24cu124"
        )

        self.assertTrue(any("Python 3.12" in problem for problem in problems))
        self.assertTrue(any("CUDA 12.6" in problem for problem in problems))

    def test_local_version_suffix_is_ignored_for_public_version_check(self) -> None:
        self.assertEqual(
            environment_doctor.normalized_public_version("1.5.3+pt24cu124"),
            "1.5.3",
        )


if __name__ == "__main__":
    unittest.main()
