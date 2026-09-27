#!/usr/bin/env python3
"""Prepare an isolated lid workspace for side-to-underside LightGlue matching."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LID_ROOT = PROJECT_ROOT / "data/processed/lid_side_underside"
SOURCE_WORKSPACE = LID_ROOT / "colmap_sparse_masked"
LIGHTGLUE_WORKSPACE = LID_ROOT / "colmap_sparse_masked_lightglue"
PREPARER = PROJECT_ROOT / "scripts/reconstruction/prepare_lightglue_workspace.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = [
        sys.executable,
        str(PREPARER),
        "--source-database",
        str(SOURCE_WORKSPACE / "database.db"),
        "--source-model",
        str(SOURCE_WORKSPACE / "sparse/1"),
        "--manifest",
        str(LID_ROOT / "dataset_manifest.csv"),
        "--workspace",
        str(LIGHTGLUE_WORKSPACE),
        "--left-view",
        "side",
        "--right-view",
        "underside",
        "--stride",
        "5",
    ]
    if args.dry_run:
        command.append("--dry-run")
    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
