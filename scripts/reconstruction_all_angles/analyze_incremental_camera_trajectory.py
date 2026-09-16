#!/usr/bin/env python3
"""Check camera-ring continuity before final triangulation and adjustment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from run_incremental_registration import (  # noqa: E402
    ImagePose,
    IncrementalRegistrationError,
    summarize_model,
)


DEFAULT_COLMAP = Path("C:/Tools/COLMAP-4.1.1/bin/colmap.exe")
DEFAULT_WORKSPACE = (
    PROJECT_ROOT
    / "data/processed/pot1-unglazed_side_underside_mid25/colmap_side_underside_mid25"
)
EXPECTED_COUNTS = {"side": 273, "mid25": 221, "underside": 222}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze adjacent camera spacing and rotation for all four rings."
    )
    parser.add_argument("--colmap", type=Path, default=DEFAULT_COLMAP)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--model-name",
        choices=("registered_final", "triangulated_final", "bundle_adjusted_final"),
        default="bundle_adjusted_final",
    )
    parser.add_argument("--max-spacing-ratio", type=float, default=2.0)
    return parser.parse_args()


def frame_index(name: str) -> int:
    match = re.search(r"(\d+)(?=\.[^.]+$)", Path(name).name)
    if match is None:
        raise IncrementalRegistrationError(f"Cannot read frame index from {name!r}.")
    return int(match.group(1))


def view_group(name: str) -> str | None:
    basename = Path(name).name.lower()
    for group in EXPECTED_COUNTS:
        if basename.startswith(f"{group}_"):
            return group
    return None


def camera_center(pose: ImagePose) -> tuple[float, float, float]:
    qw, qx, qy, qz = pose.qvec
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0:
        raise IncrementalRegistrationError("Model contains a zero camera quaternion.")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    rotation = (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
        (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
    )
    return tuple(
        -sum(rotation[row][column] * pose.tvec[row] for row in range(3))
        for column in range(3)
    )


def distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def rotation_difference_degrees(first: ImagePose, second: ImagePose) -> float:
    first_norm = math.sqrt(sum(value * value for value in first.qvec))
    second_norm = math.sqrt(sum(value * value for value in second.qvec))
    dot = abs(
        sum(a * b for a, b in zip(first.qvec, second.qvec))
        / (first_norm * second_norm)
    )
    return math.degrees(2 * math.acos(min(1.0, dot)))


def ring_report(
    images: Mapping[str, ImagePose], group: str
) -> dict[str, object]:
    sequence = sorted(
        ((name, pose) for name, pose in images.items() if view_group(name) == group),
        key=lambda item: frame_index(item[0]),
    )
    if len(sequence) != EXPECTED_COUNTS[group]:
        raise IncrementalRegistrationError(
            f"Expected {EXPECTED_COUNTS[group]} {group} images; found {len(sequence)}."
        )
    centers = [(name, camera_center(pose)) for name, pose in sequence]
    adjacent = [
        (first_name, second_name, distance(first_center, second_center))
        for (first_name, first_center), (second_name, second_center) in zip(
            centers, centers[1:]
        )
    ]
    rotations = [
        (first_name, second_name, rotation_difference_degrees(first_pose, second_pose))
        for (first_name, first_pose), (second_name, second_pose) in zip(
            sequence, sequence[1:]
        )
    ]
    median_spacing = statistics.median(item[2] for item in adjacent)
    maximum_spacing = max(adjacent, key=lambda item: item[2])
    median_rotation = statistics.median(item[2] for item in rotations)
    maximum_rotation = max(rotations, key=lambda item: item[2])
    return {
        "images": len(sequence),
        "median_adjacent_distance": median_spacing,
        "max_adjacent_distance": maximum_spacing[2],
        "max_over_median_spacing": maximum_spacing[2] / median_spacing,
        "worst_spacing_pair": [maximum_spacing[0], maximum_spacing[1]],
        "closure_distance": distance(centers[-1][1], centers[0][1]),
        "median_adjacent_rotation_degrees": median_rotation,
        "max_adjacent_rotation_degrees": maximum_rotation[2],
        "max_over_median_rotation": maximum_rotation[2] / median_rotation,
        "worst_rotation_pair": [maximum_rotation[0], maximum_rotation[1]],
    }


def main() -> int:
    args = parse_args()
    try:
        colmap = args.colmap.expanduser().resolve()
        workspace = args.workspace.expanduser()
        if not workspace.is_absolute():
            workspace = PROJECT_ROOT / workspace
        workspace = workspace.resolve()
        model = workspace / args.model_name
        report_path = workspace / f"logs/camera_trajectory_{args.model_name}.json"
        if report_path.exists():
            raise IncrementalRegistrationError(
                f"Trajectory report already exists: {report_path}"
            )
        summary = summarize_model(colmap, model)
        rings = {
            group: ring_report(summary.images, group) for group in EXPECTED_COUNTS
        }
        failures = [
            group
            for group, values in rings.items()
            if float(values["max_over_median_spacing"]) > args.max_spacing_ratio
        ]
        report = {
            "model": str(model),
            "registered_images": len(summary.images),
            "spacing_ratio_limit": args.max_spacing_ratio,
            "rings": rings,
            "continuity_passed": not failures,
            "failed_rings": failures,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Camera trajectory analysis: {args.model_name}")
        for group, values in rings.items():
            print(
                f"{group}: {values['images']} images, spacing max/median "
                f"{float(values['max_over_median_spacing']):.3f}, rotation "
                f"max/median {float(values['max_over_median_rotation']):.3f}"
            )
            print(
                "  worst spacing: "
                f"{values['worst_spacing_pair'][0]} -> "
                f"{values['worst_spacing_pair'][1]}"
            )
        print(f"Report: {report_path}")
        if failures:
            raise IncrementalRegistrationError(
                "Camera continuity failed for: " + ", ".join(failures)
            )
        print("Camera trajectory continuity check passed.")
        return 0
    except (IncrementalRegistrationError, OSError, ZeroDivisionError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
