#!/usr/bin/env python3
"""Recover remaining mid25 cameras after the first triangulation pass."""

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
TOTAL_MID25_IMAGES = 221
EXPECTED_BASE_IMAGES = 495


@dataclass(frozen=True)
class RegistrationPass2Plan:
    colmap: Path
    workspace: Path
    database: Path
    input_model: Path
    output_model: Path
    log: Path
    report: Path
    input_summary: ModelSummary
    command: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register remaining mid25 images using newly triangulated tracks."
    )
    parser.add_argument("--colmap", type=Path, default=DEFAULT_COLMAP)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def next_output_paths(workspace: Path) -> tuple[Path, Path]:
    logs = workspace / "logs"
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"_retry{attempt}"
        log = logs / f"incremental_registration_pass2{suffix}.log"
        report = logs / f"incremental_registration_pass2{suffix}_report.json"
        if report.exists():
            raise IncrementalRegistrationError(
                "Second registration pass already has a completed report."
            )
        if not log.exists():
            return log, report
    raise IncrementalRegistrationError("No unused pass-2 registration log remains.")


def build_plan(*, colmap: Path, workspace: Path) -> RegistrationPass2Plan:
    colmap = colmap.expanduser().resolve()
    workspace = resolve_workspace(workspace)
    if not colmap.is_file():
        raise IncrementalRegistrationError(f"COLMAP executable was not found: {colmap}")
    database = workspace / "database.db"
    input_model = workspace / "triangulated_pass1"
    output_model = workspace / "registered_final"
    triangulation_report = (
        workspace / "logs/incremental_triangulation_pass1_report.json"
    )
    if not database.is_file():
        raise IncrementalRegistrationError(f"Database was not found: {database}")
    if not triangulation_report.is_file():
        raise IncrementalRegistrationError(
            f"Completed triangulation report was not found: {triangulation_report}"
        )
    if not output_model.is_dir() or any(output_model.iterdir()):
        raise IncrementalRegistrationError(
            f"Second registration output must exist and be empty: {output_model}"
        )
    input_summary = summarize_model(colmap, input_model)
    input_mid25 = sum(name.startswith("mid25_") for name in input_summary.images)
    if len(input_summary.images) != EXPECTED_BASE_IMAGES + input_mid25:
        raise IncrementalRegistrationError(
            f"Triangulated input does not contain the fixed {EXPECTED_BASE_IMAGES} "
            "side/underside images plus "
            "registered mid25 images."
        )
    if not 0 < input_mid25 < TOTAL_MID25_IMAGES:
        raise IncrementalRegistrationError(
            f"Expected 1-{TOTAL_MID25_IMAGES - 1} registered mid25 images before "
            f"pass 2; found {input_mid25}."
        )
    log, report = next_output_paths(workspace)
    command = (
        str(colmap),
        "image_registrator",
        "--database_path",
        str(database),
        "--input_path",
        str(input_model),
        "--output_path",
        str(output_model),
        "--Mapper.fix_existing_frames",
        "1",
        "--Mapper.abs_pose_min_num_inliers",
        "30",
        "--Mapper.abs_pose_max_error",
        "12",
        "--Mapper.max_reg_trials",
        "3",
        "--Mapper.num_threads",
        "8",
        "--Mapper.ba_use_gpu",
        "0",
        "--log_target",
        "stderr",
    )
    return RegistrationPass2Plan(
        colmap,
        workspace,
        database,
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
            f"COLMAP second registration pass failed with exit code {return_code}."
        )


def validate_result(plan: RegistrationPass2Plan) -> dict[str, object]:
    output = summarize_model(plan.colmap, plan.output_model)
    input_names = set(plan.input_summary.images)
    output_names = set(output.images)
    missing_input = input_names - output_names
    if missing_input:
        raise IncrementalRegistrationError(
            f"Second registration pass lost {len(missing_input)} input image(s)."
        )
    additions = output_names - input_names
    unexpected = [name for name in additions if not name.startswith("mid25_")]
    if unexpected:
        raise IncrementalRegistrationError(
            f"Second registration pass added {len(unexpected)} unexpected image(s)."
        )
    if not additions:
        raise IncrementalRegistrationError(
            "Second registration pass did not recover any remaining mid25 images."
        )
    pose_change = max_pose_change(plan.input_summary.images, output.images)
    if pose_change > 1e-10:
        raise IncrementalRegistrationError(
            f"Existing registered poses changed by up to {pose_change:.3e}."
        )
    output_mid25 = sum(name.startswith("mid25_") for name in output.images)
    if output_mid25 > TOTAL_MID25_IMAGES:
        raise IncrementalRegistrationError(
            f"Output contains too many mid25 images: {output_mid25}."
        )
    return {
        "input_images_preserved": len(input_names),
        "existing_pose_max_absolute_change": pose_change,
        "mid25_recovered": len(additions),
        "mid25_registered": output_mid25,
        "mid25_remaining": TOTAL_MID25_IMAGES - output_mid25,
        "total_registered": len(output.images),
        "cameras": output.cameras,
        "sparse_points": output.points3d,
        "recovered_images": sorted(additions),
    }


def main() -> int:
    args = parse_args()
    try:
        plan = build_plan(colmap=args.colmap, workspace=args.workspace)
        input_mid25 = sum(
            name.startswith("mid25_") for name in plan.input_summary.images
        )
        print("Incremental image registration pass 2")
        print(f"Input model: {plan.input_model}")
        print(f"Output model: {plan.output_model}")
        print(f"Fixed input images: {len(plan.input_summary.images)}")
        print(f"Registered mid25 before pass 2: {input_mid25}")
        print(f"Remaining mid25 candidates: {TOTAL_MID25_IMAGES - input_mid25}")
        print(f"Log: {plan.log}")
        print(f"Command: {subprocess.list2cmdline(plan.command)}")
        if args.dry_run:
            print("Dry run complete. No registration output was written.")
            return 0
        run_command(plan.command, plan.log)
        report = validate_result(plan)
        plan.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Incremental registration pass 2 complete")
        for key, value in report.items():
            print(f"{key}: {value}")
        print(f"Report: {plan.report}")
        return 0
    except (IncrementalRegistrationError, OSError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
