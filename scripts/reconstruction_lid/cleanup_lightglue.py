#!/usr/bin/env python3
"""Clear failed side-to-underside SIFT records from the copied lid database."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LID_ROOT = PROJECT_ROOT / "data/processed/lid_side_underside"
CLEANER = PROJECT_ROOT / "scripts/reconstruction/cleanup_lightglue_database.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = [
        sys.executable,
        str(CLEANER),
        "--workspace",
        str(LID_ROOT / "colmap_sparse_masked_lightglue"),
        "--manifest",
        str(LID_ROOT / "dataset_manifest.csv"),
        "--left-view",
        "side",
        "--right-view",
        "underside",
    ]
    if args.dry_run:
        command.append("--dry-run")
    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
