#!/usr/bin/env python3
"""Bundle-adjust the final side + underside + mid25 sparse model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
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


@dataclass(frozen=True)
class AnalyzerSummary:
    images: int
    points: int
    observations: int
    mean_track_length: float
    mean_observations_per_image: float
    mean_reprojection_error: float


@dataclass(frozen=True)
class BundleAdjustmentPlan:
    colmap: Path
    workspace: Path
    input_model: Path
    output_model: Path
    log: Path
    report: Path
    input_summary: ModelSummary
    input_analysis: AnalyzerSummary
    command: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Conservatively bundle-adjust the final three-angle model without "
            "changing camera intrinsics."
        )
    )
    parser.add_argument("--colmap", type=Path, default=DEFAULT_COLMAP)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def analyze_model(colmap: Path, model: Path) -> AnalyzerSummary:
    completed = subprocess.run(
        (str(colmap), "model_analyzer", "--path", str(model)),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        raise IncrementalRegistrationError(
            f"COLMAP model_analyzer failed with exit code {completed.returncode}."
        )

    def integer(label: str) -> int:
        match = re.search(rf"{re.escape(label)}:\s+(\d+)", output)
        if match is None:
            raise IncrementalRegistrationError(
                f"Could not parse {label!r} from COLMAP model_analyzer."
            )
        return int(match.group(1))

    def decimal(label: str, suffix: str = "") -> float:
        match = re.search(
            rf"{re.escape(label)}:\s+([0-9eE+.-]+){re.escape(suffix)}", output
        )
        if match is None:
            raise IncrementalRegistrationError(
                f"Could not parse {label!r} from COLMAP model_analyzer."
            )
        return float(match.group(1))

    return AnalyzerSummary(
        images=integer("Registered images"),
        points=integer("Points"),
        observations=integer("Observations"),
        mean_track_length=decimal("Mean track length"),
        mean_observations_per_image=decimal("Mean observations per image"),
        mean_reprojection_error=decimal("Mean reprojection error", "px"),
    )


def next_output_paths(workspace: Path) -> tuple[Path, Path]:
    logs = workspace / "logs"
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"_retry{attempt}"
        log = logs / f"final_bundle_adjustment{suffix}.log"
        report = logs / f"final_bundle_adjustment{suffix}_report.json"
        if report.exists():
            raise IncrementalRegistrationError(
                "Final bundle adjustment already has a completed report."
            )
        if not log.exists():
            return log, report
    raise IncrementalRegistrationError("No unused bundle-adjustment log remains.")


def build_plan(*, colmap: Path, workspace: Path) -> BundleAdjustmentPlan:
    colmap = colmap.expanduser().resolve()
    workspace = resolve_workspace(workspace)
    if not colmap.is_file():
        raise IncrementalRegistrationError(f"COLMAP executable was not found: {colmap}")
    input_model = workspace / "triangulated_final"
    output_model = workspace / "bundle_adjusted_final"
    triangulation_report = workspace / "logs/final_triangulation_report.json"
    if not triangulation_report.is_file():
        raise IncrementalRegistrationError(
            f"Completed final triangulation report was not found: {triangulation_report}"
        )
    if not output_model.is_dir() or any(output_model.iterdir()):
        raise IncrementalRegistrationError(
            f"Bundle-adjustment output must exist and be empty: {output_model}"
        )
    input_summary = summarize_model(colmap, input_model)
    if (
        len(input_summary.images) != 716
        or sum(name.startswith("mid25_") for name in input_summary.images) != 221
        or any(name.startswith("top45_") for name in input_summary.images)
    ):
        raise IncrementalRegistrationError(
            "Final triangulated input is not the complete 716-image three-angle model."
        )
    input_analysis = analyze_model(colmap, input_model)
    log, report = next_output_paths(workspace)
    command = (
        str(colmap),
        "bundle_adjuster",
        "--input_path",
        str(input_model),
        "--output_path",
        str(output_model),
        "--BundleAdjustment.refine_focal_length",
        "0",
        "--BundleAdjustment.refine_principal_point",
        "0",
        "--BundleAdjustment.refine_extra_params",
        "0",
        "--BundleAdjustment.refine_rig_from_world",
        "1",
        "--BundleAdjustment.refine_sensor_from_rig",
        "0",
        "--BundleAdjustment.refine_points3D",
        "1",
        "--BundleAdjustmentCeres.max_num_iterations",
        "100",
        "--BundleAdjustmentCeres.use_gpu",
        "0",
        "--log_target",
        "stderr",
    )
    return BundleAdjustmentPlan(
        colmap,
        workspace,
        input_model,
        output_model,
        log,
        report,
        input_summary,
        input_analysis,
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
            f"COLMAP bundle adjustment failed with exit code {return_code}."
        )


def validate_result(plan: BundleAdjustmentPlan) -> dict[str, object]:
    output_summary = summarize_model(plan.colmap, plan.output_model)
    if set(output_summary.images) != set(plan.input_summary.images):
        raise IncrementalRegistrationError(
            "Bundle adjustment changed the registered image set."
        )
    if output_summary.cameras != plan.input_summary.cameras:
        raise IncrementalRegistrationError(
            "Bundle adjustment changed the number of camera calibrations."
        )
    if output_summary.points3d != plan.input_summary.points3d:
        raise IncrementalRegistrationError(
            "Bundle adjustment changed the sparse-point count: "
            f"{plan.input_summary.points3d} -> {output_summary.points3d}."
        )
    output_analysis = analyze_model(plan.colmap, plan.output_model)
    if output_analysis.mean_reprojection_error > (
        plan.input_analysis.mean_reprojection_error + 1e-6
    ):
        raise IncrementalRegistrationError(
            "Bundle adjustment increased mean reprojection error: "
            f"{plan.input_analysis.mean_reprojection_error:.6f} -> "
            f"{output_analysis.mean_reprojection_error:.6f}px."
        )
    pose_change = max_pose_change(plan.input_summary.images, output_summary.images)
    return {
        "registered_images": len(output_summary.images),
        "mid25_images": sum(
            name.startswith("mid25_") for name in output_summary.images
        ),
        "top45_images": sum(
            name.startswith("top45_") for name in output_summary.images
        ),
        "cameras": output_summary.cameras,
        "sparse_points": output_summary.points3d,
        "observations": output_analysis.observations,
        "mean_track_length": output_analysis.mean_track_length,
        "mean_observations_per_image": output_analysis.mean_observations_per_image,
        "mean_reprojection_error_before_px": plan.input_analysis.mean_reprojection_error,
        "mean_reprojection_error_after_px": output_analysis.mean_reprojection_error,
        "mean_reprojection_error_improvement_px": (
            plan.input_analysis.mean_reprojection_error
            - output_analysis.mean_reprojection_error
        ),
        "max_pose_parameter_change": pose_change,
    }


def main() -> int:
    args = parse_args()
    try:
        plan = build_plan(colmap=args.colmap, workspace=args.workspace)
        print("Final conservative bundle adjustment")
        print(f"Input model: {plan.input_model}")
        print(f"Output model: {plan.output_model}")
        print(f"Registered images: {len(plan.input_summary.images)}")
        print(f"Sparse points: {plan.input_summary.points3d}")
        print(
            "Mean reprojection error before: "
            f"{plan.input_analysis.mean_reprojection_error:.6f}px"
        )
        print(f"Log: {plan.log}")
        print(f"Command: {subprocess.list2cmdline(plan.command)}")
        if args.dry_run:
            print("Dry run complete. No bundle-adjusted model was written.")
            return 0
        run_command(plan.command, plan.log)
        report = validate_result(plan)
        plan.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Final bundle adjustment complete")
        for key, value in report.items():
            print(f"{key}: {value}")
        print(f"Report: {plan.report}")
        return 0
    except (IncrementalRegistrationError, OSError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
