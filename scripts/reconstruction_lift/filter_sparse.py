#!/usr/bin/env python3
"""Conservatively filter the lift sparse model into a separate output."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COLMAP = Path("C:/Tools/COLMAP-4.1.1/bin/colmap.exe")
DEFAULT_WORKSPACE = (
    PROJECT_ROOT
    / "data/processed/lift_all_angles/colmap_sparse_masked_multiview"
)


class FilterError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap", type=Path, default=DEFAULT_COLMAP)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--min-track-len", type=int, default=3)
    parser.add_argument("--max-reproj-error", type=float, default=3.0)
    parser.add_argument("--min-tri-angle", type=float, default=1.5)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def analyze(colmap: Path, model: Path) -> dict[str, float | int]:
    completed = subprocess.run(
        (str(colmap), "model_analyzer", "--path", str(model)),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = "\n".join((completed.stdout, completed.stderr))
    if completed.returncode != 0:
        raise FilterError(f"COLMAP could not analyze {model}.")

    def integer(label: str) -> int:
        match = re.search(rf"{re.escape(label)}:\s+(\d+)", output)
        if match is None:
            raise FilterError(f"Could not parse {label} from model_analyzer.")
        return int(match.group(1))

    def decimal(label: str, suffix: str = "") -> float:
        match = re.search(
            rf"{re.escape(label)}:\s+([0-9eE+.-]+){re.escape(suffix)}", output
        )
        if match is None:
            raise FilterError(f"Could not parse {label} from model_analyzer.")
        return float(match.group(1))

    return {
        "registered_images": integer("Registered images"),
        "points": integer("Points"),
        "observations": integer("Observations"),
        "mean_track_length": decimal("Mean track length"),
        "mean_reprojection_error_px": decimal("Mean reprojection error", "px"),
    }


def main() -> int:
    args = parse_args()
    try:
        colmap = args.colmap.expanduser().resolve()
        workspace = args.workspace.expanduser().resolve()
        input_model = workspace / "sparse" / "0"
        output_model = workspace / "sparse_filtered"
        logs = workspace / "logs"
        report_path = logs / "point_filtering_report.json"
        log_path = logs / "point_filtering.log"

        if not colmap.is_file():
            raise FilterError(f"COLMAP executable was not found: {colmap}")
        if not input_model.is_dir():
            raise FilterError(f"Input sparse model was not found: {input_model}")
        if args.min_track_len < 2:
            raise FilterError("min-track-len must be at least 2.")
        if args.max_reproj_error <= 0 or args.min_tri_angle < 0:
            raise FilterError("Filtering thresholds are invalid.")
        if output_model.exists() or report_path.exists() or log_path.exists():
            raise FilterError(
                "Filtered output already exists; the original model was not changed."
            )

        before = analyze(colmap, input_model)
        command = (
            str(colmap),
            "point_filtering",
            "--input_path",
            str(input_model),
            "--output_path",
            str(output_model),
            "--min_track_len",
            str(args.min_track_len),
            "--max_reproj_error",
            str(args.max_reproj_error),
            "--min_tri_angle",
            str(args.min_tri_angle),
            "--log_target",
            "stderr",
        )
        print("Lift sparse-point filtering plan")
        print(f"Input: {input_model}")
        print(f"Output: {output_model}")
        print(f"Registered images: {before['registered_images']}")
        print(f"Sparse points: {before['points']}")
        print(f"Mean reprojection error: {before['mean_reprojection_error_px']:.6f}px")
        print(
            "Thresholds: "
            f"track >= {args.min_track_len}, "
            f"error <= {args.max_reproj_error}px, "
            f"angle >= {args.min_tri_angle} degrees"
        )
        print("Command: " + subprocess.list2cmdline(command))
        if args.dry_run:
            print("Dry run complete. No filtered model was created.")
            return 0

        output_model.mkdir(parents=True)
        logs.mkdir(parents=True, exist_ok=True)
        with log_path.open("x", encoding="utf-8", newline="\n") as log:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        if completed.returncode != 0:
            shutil.rmtree(output_model, ignore_errors=True)
            raise FilterError(
                f"COLMAP point_filtering failed with exit code {completed.returncode}."
            )

        after = analyze(colmap, output_model)
        if after["registered_images"] != before["registered_images"]:
            raise FilterError("Point filtering changed the registered image count.")
        if after["points"] <= 0 or after["points"] > before["points"]:
            raise FilterError("Point filtering produced an invalid point count.")
        retained_fraction = after["points"] / before["points"]
        if retained_fraction < 0.60:
            raise FilterError(
                "Filtering retained less than 60% of the sparse points; use the original model."
            )
        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "operation": "colmap_point_filtering",
            "input_model": str(input_model),
            "output_model": str(output_model),
            "settings": {
                "min_track_len": args.min_track_len,
                "max_reproj_error": args.max_reproj_error,
                "min_tri_angle": args.min_tri_angle,
            },
            "before": before,
            "after": after,
            "removed_points": before["points"] - after["points"],
            "retained_fraction": retained_fraction,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Sparse-point filtering complete")
        print(f"Points: {before['points']} -> {after['points']}")
        print(f"Retained: {retained_fraction:.2%}")
        print(
            "Mean reprojection error: "
            f"{before['mean_reprojection_error_px']:.6f}px -> "
            f"{after['mean_reprojection_error_px']:.6f}px"
        )
        print(f"Report: {report_path}")
        return 0
    except (FilterError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
