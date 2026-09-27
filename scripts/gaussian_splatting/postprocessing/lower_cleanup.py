#!/usr/bin/env python3
"""Remove unsupported Gaussians below a side-only reconstruction.

The cleanup is export-only. It estimates the object's upward direction from
COLMAP image-up vectors and places a conservative lower clipping plane from
the sparse points that initialized training. The checkpoint and original
exports are never changed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import sys
from typing import BinaryIO

import numpy as np

from ..core.common import (
    DEFAULT_CONFIG,
    PipelineError,
    load_yaml,
    resolve_config_path,
    resolve_paths,
)
from .checkpoint import load_checkpoint_cpu, resolve_complete_run, sha256_file
from ..training.runner import atomic_write_json


@dataclass(frozen=True)
class CleanupGeometry:
    axis: np.ndarray
    sparse_low: float
    sparse_high: float
    cutoff: float


@dataclass(frozen=True)
class CleanupPlan:
    source_checkpoint: str
    output_directory: str
    source_gaussians: int
    removed_gaussians: int
    retained_gaussians: int
    removed_fraction: float
    axis: list[float]
    sparse_low: float
    sparse_high: float
    cutoff: float
    removed_axial_min: float | None
    removed_axial_max: float | None
    retained_axial_min: float
    retained_axial_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a clean 3DGS export without unsupported lower Gaussians."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--clean-name",
        default="clean_lower_v1",
        help="Subdirectory/name for the cleaned export.",
    )
    parser.add_argument(
        "--lower-quantile",
        type=float,
        default=0.01,
        help="Sparse-point quantile used for the supported lower boundary.",
    )
    parser.add_argument(
        "--margin-scale",
        type=float,
        default=0.02,
        help="Safety margin below the boundary as a fraction of object height.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("ply", "splat"),
        default=["ply", "splat"],
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if not args.clean_name or any(character not in safe for character in args.clean_name):
        raise PipelineError(
            "clean-name may contain only letters, numbers, '.', '_' and '-'."
        )
    if not 0.0 <= args.lower_quantile < 0.1:
        raise PipelineError("lower-quantile must be in [0, 0.1).")
    if not 0.0 <= args.margin_scale <= 0.2:
        raise PipelineError("margin-scale must be in [0, 0.2].")


def read_exact(source: BinaryIO, size: int, label: str) -> bytes:
    value = source.read(size)
    if len(value) != size:
        raise PipelineError(f"COLMAP {label} ended unexpectedly.")
    return value


def quaternion_rotation(qvec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(qvec))
    if not math.isfinite(norm) or norm <= 0:
        raise PipelineError("COLMAP contains an invalid camera quaternion.")
    qw, qx, qy, qz = qvec / norm
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def read_colmap_up_axis(images_bin: Path) -> np.ndarray:
    """Average the world-space image-up direction of registered cameras."""

    try:
        source = images_bin.open("rb")
    except OSError as error:
        raise PipelineError(f"Could not open COLMAP image poses: {images_bin}") from error
    image_ups: list[np.ndarray] = []
    with source:
        image_count = struct.unpack("<Q", read_exact(source, 8, "images.bin"))[0]
        for _ in range(image_count):
            _image_id = struct.unpack("<i", read_exact(source, 4, "images.bin"))[0]
            qvec = np.asarray(
                struct.unpack("<4d", read_exact(source, 32, "images.bin")),
                dtype=np.float64,
            )
            _tvec = read_exact(source, 24, "images.bin")
            _camera_id = read_exact(source, 4, "images.bin")
            while read_exact(source, 1, "images.bin") != b"\x00":
                pass
            point_count = struct.unpack(
                "<Q", read_exact(source, 8, "images.bin")
            )[0]
            source.seek(point_count * 24, 1)
            rotation = quaternion_rotation(qvec)
            image_up = -(rotation.T @ np.asarray([0.0, 1.0, 0.0]))
            image_ups.append(image_up / np.linalg.norm(image_up))
    if len(image_ups) < 3:
        raise PipelineError("At least three registered cameras are required.")
    axis = np.mean(image_ups, axis=0)
    length = float(np.linalg.norm(axis))
    if not math.isfinite(length) or length < 0.25:
        raise PipelineError("Registered camera orientations do not define a stable up axis.")
    return axis / length


def read_colmap_points(points_bin: Path) -> np.ndarray:
    """Read XYZ coordinates from a COLMAP points3D.bin file."""

    try:
        source = points_bin.open("rb")
    except OSError as error:
        raise PipelineError(f"Could not open COLMAP sparse points: {points_bin}") from error
    points: list[tuple[float, float, float]] = []
    with source:
        point_count = struct.unpack("<Q", read_exact(source, 8, "points3D.bin"))[0]
        for _ in range(point_count):
            _point_id = read_exact(source, 8, "points3D.bin")
            xyz = struct.unpack("<3d", read_exact(source, 24, "points3D.bin"))
            _rgb = read_exact(source, 3, "points3D.bin")
            _error = read_exact(source, 8, "points3D.bin")
            track_length = struct.unpack(
                "<Q", read_exact(source, 8, "points3D.bin")
            )[0]
            source.seek(track_length * 8, 1)
            points.append(xyz)
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1:] != (3,) or len(result) < 100:
        raise PipelineError("COLMAP sparse model contains too few 3D points.")
    finite = np.all(np.isfinite(result), axis=1)
    result = result[finite]
    if len(result) < 100:
        raise PipelineError("COLMAP sparse model contains too few finite 3D points.")
    return result


def infer_cleanup_geometry(
    axis: np.ndarray,
    sparse_points: np.ndarray,
    normalization_center: np.ndarray,
    normalization_scale: float,
    lower_quantile: float,
    margin_scale: float,
) -> CleanupGeometry:
    normalized = (sparse_points - normalization_center[None, :]) * normalization_scale
    axial = normalized @ axis
    sparse_low = float(np.quantile(axial, lower_quantile))
    sparse_high = float(np.quantile(axial, 1.0 - lower_quantile))
    height = sparse_high - sparse_low
    if not math.isfinite(height) or height <= 1e-6:
        raise PipelineError("Sparse points do not define a valid object height.")
    cutoff = sparse_low - margin_scale * height
    return CleanupGeometry(axis, sparse_low, sparse_high, cutoff)


def selected_gaussians(means, geometry: CleanupGeometry):
    import torch

    axis = torch.as_tensor(geometry.axis, dtype=means.dtype, device=means.device)
    axial = means @ axis
    keep = axial >= geometry.cutoff
    return keep, axial


def make_plan(run, output_dir: Path, means, keep, axial, geometry) -> CleanupPlan:
    count = int(len(means))
    retained = int(keep.sum().item())
    removed = count - retained
    removed_axial = axial[~keep]
    retained_axial = axial[keep]
    return CleanupPlan(
        source_checkpoint=str(run.checkpoint),
        output_directory=str(output_dir),
        source_gaussians=count,
        removed_gaussians=removed,
        retained_gaussians=retained,
        removed_fraction=removed / count,
        axis=[float(value) for value in geometry.axis],
        sparse_low=geometry.sparse_low,
        sparse_high=geometry.sparse_high,
        cutoff=geometry.cutoff,
        removed_axial_min=(
            float(removed_axial.min().item()) if removed > 0 else None
        ),
        removed_axial_max=(
            float(removed_axial.max().item()) if removed > 0 else None
        ),
        retained_axial_min=float(retained_axial.min().item()),
        retained_axial_max=float(retained_axial.max().item()),
    )


def print_plan(plan: CleanupPlan, dry_run: bool) -> None:
    label = "dry run" if dry_run else "complete"
    print(f"3DGS lower cleanup {label}")
    print(f"Source Gaussians: {plan.source_gaussians}")
    print(
        f"Remove/retain: {plan.removed_gaussians}/{plan.retained_gaussians} "
        f"({plan.removed_fraction:.2%} removed)"
    )
    print("Up axis: " + ", ".join(f"{value:.6f}" for value in plan.axis))
    print(
        "Sparse lower/upper support: "
        f"{plan.sparse_low:.6f}/{plan.sparse_high:.6f}"
    )
    print(f"Lower clipping plane: {plan.cutoff:.6f}")
    if plan.removed_axial_min is not None:
        print(
            "Removed axial range: "
            f"{plan.removed_axial_min:.6f} to {plan.removed_axial_max:.6f}"
        )
    print(
        "Retained axial range: "
        f"{plan.retained_axial_min:.6f} to {plan.retained_axial_max:.6f}"
    )
    print(f"Output directory: {plan.output_directory}")
    if dry_run:
        print("Dry run passed. No cleaned export was created.")


def clean_export(args: argparse.Namespace) -> int:
    validate_args(args)
    config_path = resolve_config_path(args.config)
    paths = resolve_paths(config_path, load_yaml(config_path))
    run = resolve_complete_run(paths, args.run_name)
    output_dir = run.run_dir / "exports" / args.clean_name
    manifest_path = output_dir / "lower_cleanup_manifest.json"
    formats = tuple(dict.fromkeys(args.formats))
    destinations = {
        name: output_dir / f"thai_ceramics_{run.run_name}_{args.clean_name}.{name}"
        for name in formats
    }
    existing = [path for path in (*destinations.values(), manifest_path) if path.exists()]
    if existing:
        raise PipelineError(
            "Refusing to overwrite existing cleanup output(s): "
            + ", ".join(str(path) for path in existing)
        )

    checkpoint = load_checkpoint_cpu(run.checkpoint)
    center = np.asarray(checkpoint["normalization_center"], dtype=np.float64)
    scale = float(checkpoint["normalization_scale"])
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise PipelineError("Checkpoint contains an invalid normalization center.")
    if not math.isfinite(scale) or scale <= 0:
        raise PipelineError("Checkpoint contains an invalid normalization scale.")

    sparse_dir = paths.sparse_model
    axis = read_colmap_up_axis(sparse_dir / "images.bin")
    sparse_points = read_colmap_points(sparse_dir / "points3D.bin")
    geometry = infer_cleanup_geometry(
        axis,
        sparse_points,
        center,
        scale,
        args.lower_quantile,
        args.margin_scale,
    )
    splats = checkpoint["splats"]
    keep, axial = selected_gaussians(splats["means"], geometry)
    plan = make_plan(run, output_dir, splats["means"], keep, axial, geometry)
    if plan.removed_gaussians < 10:
        raise PipelineError(
            "The lower clipping plane selected fewer than 10 Gaussians; refusing to export."
        )
    if plan.removed_fraction > 0.25:
        raise PipelineError(
            "The lower clipping plane selected more than 25% of the model; refusing to export."
        )
    print_plan(plan, args.dry_run)
    for name, destination in destinations.items():
        print(f"{name}: {destination}")
    if args.dry_run:
        return 0

    try:
        from gsplat import export_splats
    except (ImportError, ModuleNotFoundError) as error:
        raise PipelineError("gsplat exporter is unavailable in this environment.") from error

    output_dir.mkdir(parents=True, exist_ok=False)
    filtered = {key: value[keep].contiguous() for key, value in splats.items()}
    records: list[dict[str, object]] = []
    try:
        for export_format, destination in destinations.items():
            export_splats(
                means=filtered["means"],
                scales=filtered["scales"],
                quats=filtered["quats"],
                opacities=filtered["opacities"],
                sh0=filtered["sh0"],
                shN=filtered["shN"],
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
            "operation": "remove_unsupported_lower_gaussians",
            "run_name": run.run_name,
            "source_checkpoint": str(run.checkpoint),
            "checkpoint_step": run.step,
            "settings": {
                "lower_quantile": args.lower_quantile,
                "margin_scale": args.margin_scale,
                "formats": list(formats),
            },
            "plan": asdict(plan),
            "exports": records,
        }
        atomic_write_json(manifest, manifest_path)
    except Exception:
        for destination in destinations.values():
            if destination.exists():
                destination.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
        if output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        raise
    print(f"Manifest: {manifest_path}")
    return 0


def main() -> int:
    try:
        return clean_export(parse_args())
    except (PipelineError, OSError, ValueError, struct.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
