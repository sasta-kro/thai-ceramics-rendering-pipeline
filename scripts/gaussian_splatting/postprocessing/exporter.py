#!/usr/bin/env python3
"""Export one complete 3DGS checkpoint to portable splat formats."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from ..core.common import (
    DEFAULT_CONFIG,
    PipelineError,
    load_yaml,
    resolve_config_path,
    resolve_paths,
)
from .checkpoint import (
    EXPORT_SUFFIXES,
    export_path,
    load_checkpoint_cpu,
    resolve_complete_run,
    sha256_file,
)
from ..training.runner import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one complete 3DGS run.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default="baseline_7k")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=tuple(EXPORT_SUFFIXES),
        default=["ply", "splat"],
        help="PLY retains SH coefficients; splat is compact and base-color only.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def export(args: argparse.Namespace) -> int:
    config_path = resolve_config_path(args.config)
    paths = resolve_paths(config_path, load_yaml(config_path))
    run = resolve_complete_run(paths, args.run_name)
    checkpoint = load_checkpoint_cpu(run.checkpoint)
    formats = tuple(dict.fromkeys(args.formats))
    destinations = {name: export_path(run, name) for name in formats}
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        raise PipelineError(
            "Refusing to overwrite existing export(s): "
            + ", ".join(str(path) for path in existing)
        )

    gaussian_count = int(checkpoint["splats"]["means"].shape[0])
    print("3DGS export plan")
    print(f"Run: {run.run_name}")
    print(f"Checkpoint: {run.checkpoint}")
    print(f"Gaussians: {gaussian_count}")
    for name, destination in destinations.items():
        print(f"{name}: {destination}")
    if args.dry_run:
        print("Dry run passed. No model export was created.")
        return 0

    try:
        from gsplat import export_splats
    except (ImportError, ModuleNotFoundError) as error:
        raise PipelineError("gsplat exporter is unavailable in this environment.") from error

    export_dir = run.run_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    splats = checkpoint["splats"]
    records: list[dict[str, object]] = []
    for export_format, destination in destinations.items():
        export_splats(
            means=splats["means"],
            scales=splats["scales"],
            quats=splats["quats"],
            opacities=splats["opacities"],
            sh0=splats["sh0"],
            shN=splats["shN"],
            format=export_format,
            save_to=str(destination),
        )
        if not destination.is_file() or destination.stat().st_size == 0:
            raise PipelineError(f"gsplat did not create a valid export: {destination}")
        records.append(
            {
                "format": export_format,
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
        print(f"Exported {export_format}: {destination}")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run.run_name,
        "checkpoint": str(run.checkpoint),
        "checkpoint_step": run.step,
        "gaussians": gaussian_count,
        "normalization_center": checkpoint["normalization_center"].tolist(),
        "normalization_scale": float(checkpoint["normalization_scale"]),
        "exports": records,
    }
    atomic_write_json(manifest, export_dir / "export_manifest.json")
    print(f"Export manifest: {export_dir / 'export_manifest.json'}")
    return 0


def main() -> int:
    try:
        return export(parse_args())
    except PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
