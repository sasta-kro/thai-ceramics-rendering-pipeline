#!/usr/bin/env python3
"""Run the shared COLMAP pipeline with the isolated lid configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLMAP_RUNNER = PROJECT_ROOT / "scripts/reconstruction/run_colmap.py"
LID_CONFIG = PROJECT_ROOT / "configs/colmap_lid_side_underside.yml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print the COLMAP plan."
    )
    args = parser.parse_args()

    command = [sys.executable, str(COLMAP_RUNNER), "--config", str(LID_CONFIG)]
    if args.dry_run:
        command.append("--dry-run")
    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
