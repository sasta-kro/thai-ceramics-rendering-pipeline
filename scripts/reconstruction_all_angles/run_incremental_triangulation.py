#!/usr/bin/env python3
"""Freshly triangulate the final side + underside + mid25 model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from run_incremental_registration import (  # noqa: E402
    ImagePose,
    IncrementalRegistrationError,
    ModelSummary,
    max_pose_change,
    resolve_workspace,
    summarize_model,
)


DEFAULT_COLMAP = Path("C:/Tools/COLMAP-4.1.1/bin/colmap.exe")
DEFAULT_WORKSPACE = (
    PROJECT_ROOT
    / "data/processed/pot1-unglazed_side_underside_mid25/colmap_side_underside_mid25"
)
DEFAULT_IMAGES = (
    PROJECT_ROOT / "data/frames_output/pot1-unglazed_side_underside_mid25_frames"
)
EXPECTED_BASE_IMAGES = 495


@dataclass(frozen=True)
class TriangulationPlan:
    colmap: Path
    workspace: Path
    database: Path
    images: Path
    input_model: Path
    output_model: Path
    log: Path
    report: Path
    input_summary: ModelSummary
    command: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discard provisional points and freshly triangulate the final "
            "side + underside + mid25 model with all camera poses fixed."
        )
    )
    parser.add_argument("--colmap", type=Path, default=DEFAULT_COLMAP)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_project_directory(path: Path, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise IncrementalRegistrationError(
            f"{label} must stay inside the project: {candidate}"
        ) from error
    if not candidate.is_dir():
        raise IncrementalRegistrationError(f"{label} was not found: {candidate}")
    return candidate


def next_output_paths(workspace: Path) -> tuple[Path, Path]:
    logs = workspace / "logs"
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"_retry{attempt}"
        log = logs / f"final_triangulation{suffix}.log"
        report = logs / f"final_triangulation{suffix}_report.json"
        if report.exists():
            raise IncrementalRegistrationError(
                "Final triangulation already has a completed report."
            )
        if not log.exists():
            return log, report
    raise IncrementalRegistrationError("No unused triangulation retry log remains.")


def build_plan(
    *, colmap: Path, workspace: Path, images: Path
) -> TriangulationPlan:
    colmap = colmap.expanduser().resolve()
    workspace = resolve_workspace(workspace)
    images = resolve_project_directory(images, "Combined image directory")
    if not colmap.is_file():
        raise IncrementalRegistrationError(f"COLMAP executable was not found: {colmap}")
    database = workspace / "database.db"
    input_model = workspace / "registered_final"
    output_model = workspace / "triangulated_final"
    registration_report = workspace / "logs/incremental_registration_pass2_report.json"
    if not database.is_file():
        raise IncrementalRegistrationError(f"Database was not found: {database}")
    if not registration_report.is_file():
        raise IncrementalRegistrationError(
            f"Completed pass-2 registration report was not found: {registration_report}"
        )
    if not output_model.is_dir() or any(output_model.iterdir()):
        raise IncrementalRegistrationError(
            f"Triangulation output must exist and be empty: {output_model}"
        )
    input_summary = summarize_model(colmap, input_model)
    mid25_count = sum(
        name.startswith("mid25_") for name in input_summary.images
    )
    if len(input_summary.images) != 716 or mid25_count != 221:
        raise IncrementalRegistrationError(
            "Final registration input must contain all 716 images, including "
            f"221 mid25 images; found {len(input_summary.images)} and {mid25_count}."
        )
    if any(name.startswith("top45_") for name in input_summary.images):
        raise IncrementalRegistrationError("Final registration still contains top45 images.")
    missing_files = [name for name in input_summary.images if not (images / name).is_file()]
    if missing_files:
        raise IncrementalRegistrationError(
            f"Combined image directory is missing {len(missing_files)} registered image(s)."
        )
    log, report = next_output_paths(workspace)
    command = (
        str(colmap),
        "point_triangulator",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--input_path",
        str(input_model),
        "--output_path",
        str(output_model),
        "--clear_points",
        "1",
        "--refine_intrinsics",
        "0",
        "--Mapper.fix_existing_frames",
        "1",
        "--Mapper.num_threads",
        "8",
        "--Mapper.ba_use_gpu",
        "0",
        "--Mapper.tri_ignore_two_view_tracks",
        "1",
        "--log_target",
        "stderr",
    )
    return TriangulationPlan(
        colmap,
        workspace,
        database,
        images,
        input_model,
        output_model,
        log,
        report,
        input_summary,
        command,
    )


def run_command(command: Sequence[str], log: Path) -> None:
    with log.open("x", encoding="utf-8", newline="\n") as handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise IncrementalRegistrationError(
            f"COLMAP point triangulation failed with exit code {return_code}."
        )


def validate_result(plan: TriangulationPlan) -> dict[str, object]:
    output = summarize_model(plan.colmap, plan.output_model)
    input_names = set(plan.input_summary.images)
    output_names = set(output.images)
    if output_names != input_names:
        raise IncrementalRegistrationError(
            "Triangulation changed the registered image set: "
            f"lost {len(input_names - output_names)}, "
            f"added {len(output_names - input_names)}."
        )
    pose_change = max_pose_change(plan.input_summary.images, output.images)
    if pose_change > 1e-10:
        raise IncrementalRegistrationError(
            f"Registered poses changed by up to {pose_change:.3e}."
        )
    if output.points3d <= 0:
        raise IncrementalRegistrationError(
            "Fresh triangulation produced no sparse points."
        )
    mid25_count = sum(name.startswith("mid25_") for name in output.images)
    return {
        "registered_images_preserved": len(output.images),
        "mid25_images_preserved": mid25_count,
        "camera_pose_max_absolute_change": pose_change,
        "cameras": output.cameras,
        "provisional_sparse_points_discarded": plan.input_summary.points3d,
        "fresh_sparse_points": output.points3d,
        "fresh_minus_provisional_points": output.points3d - plan.input_summary.points3d,
    }


def main() -> int:
    args = parse_args()
    try:
        plan = build_plan(
            colmap=args.colmap, workspace=args.workspace, images=args.images
        )
        mid25_count = sum(
            name.startswith("mid25_") for name in plan.input_summary.images
        )
        print("Final fresh triangulation")
        print(f"Input model: {plan.input_model}")
        print(f"Output model: {plan.output_model}")
        print(f"Registered images: {len(plan.input_summary.images)}")
        print(f"Registered mid25 images: {mid25_count}")
        print(f"Provisional sparse points to discard: {plan.input_summary.points3d}")
        print(f"Log: {plan.log}")
        print(f"Command: {subprocess.list2cmdline(plan.command)}")
        if args.dry_run:
            print("Dry run complete. No final triangulation output was written.")
            return 0
        run_command(plan.command, plan.log)
        report = validate_result(plan)
        plan.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Final fresh triangulation complete")
        for key, value in report.items():
            print(f"{key}: {value}")
        print(f"Report: {plan.report}")
        return 0
    except (IncrementalRegistrationError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
