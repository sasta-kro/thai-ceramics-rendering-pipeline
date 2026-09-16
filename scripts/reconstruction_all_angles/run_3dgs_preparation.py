#!/usr/bin/env python3
"""Run the shared masked-3DGS preparation with the three-angle config."""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SHARED_SCRIPTS = PROJECT_ROOT / "scripts/reconstruction"
sys.path.insert(0, str(SHARED_SCRIPTS))

import run_multiview_3dgs_preparation as shared  # noqa: E402


shared.DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/colmap_undistort_pot1_unglazed_side_underside_mid25.yml"
)


if __name__ == "__main__":
    raise SystemExit(shared.main())
