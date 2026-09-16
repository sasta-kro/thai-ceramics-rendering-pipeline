#!/usr/bin/env python3
"""Register mid25 cameras against the fixed side + underside base model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_COLMAP = Path("C:/Tools/COLMAP-4.1.1/bin/colmap.exe")
DEFAULT_WORKSPACE = (
    PROJECT_ROOT
    / "data/processed/pot1-unglazed_side_underside_mid25/colmap_side_underside_mid25"
)
EXPECTED_BASE_IMAGES = 495
EXPECTED_BASE_CAMERAS = 3
MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")


class IncrementalRegistrationError(RuntimeError):
    """A user-correctable incremental registration error."""


@dataclass(frozen=True)
class ImagePose:
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int


@dataclass(frozen=True)
class ModelSummary:
    images: Mapping[str, ImagePose]
    cameras: int
    points3d: int


@dataclass(frozen=True)
class RegistrationPlan:
    colmap: Path
    workspace: Path
    database: Path
    input_model: Path
    output_model: Path
    log: Path
    report: Path
    base_summary: ModelSummary
    command: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register mid25 images into the fixed 495-view side/underside model."
    )
    parser.add_argument("--colmap", type=Path, default=DEFAULT_COLMAP)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_workspace(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise IncrementalRegistrationError(
            f"Workspace must stay inside the project: {candidate}"
        ) from error
    return candidate


def require_model(path: Path, label: str) -> None:
    if not path.is_dir():
        raise IncrementalRegistrationError(f"{label} was not found: {path}")
    missing = [name for name in MODEL_FILES if not (path / name).is_file()]
    if missing:
        raise IncrementalRegistrationError(
            f"{label} is missing file(s): " + ", ".join(missing)
        )


def parse_images_text(path: Path) -> dict[str, ImagePose]:
    result: dict[str, ImagePose] = {}
    expecting_image = True
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            continue
        if expecting_image:
            if not stripped:
                continue
            fields = stripped.split(maxsplit=9)
            if len(fields) != 10:
                raise IncrementalRegistrationError(
                    f"Malformed COLMAP image record: {raw_line}"
                )
            name = fields[9].replace("\\", "/")
            if name in result:
                raise IncrementalRegistrationError(f"Duplicate model image: {name}")
            result[name] = ImagePose(
                qvec=tuple(float(value) for value in fields[1:5]),  # type: ignore[arg-type]
                tvec=tuple(float(value) for value in fields[5:8]),  # type: ignore[arg-type]
                camera_id=int(fields[8]),
            )
            expecting_image = False
        else:
            expecting_image = True
    if not expecting_image:
        raise IncrementalRegistrationError("COLMAP images.txt ended unexpectedly.")
    return result


def count_text_records(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def summarize_model(colmap: Path, model: Path) -> ModelSummary:
    require_model(model, "Sparse model")
    with tempfile.TemporaryDirectory(prefix="thai_pot_registration_check_") as directory:
        text_model = Path(directory)
        completed = subprocess.run(
            (
                str(colmap),
                "model_converter",
                "--input_path",
                str(model),
                "--output_path",
                str(text_model),
                "--output_type",
                "TXT",
                "--log_target",
                "stderr",
            ),
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise IncrementalRegistrationError(
                "Could not convert sparse model for validation: "
                + (completed.stdout or completed.stderr)
            )
        images = parse_images_text(text_model / "images.txt")
        cameras = count_text_records(text_model / "cameras.txt")
        points = count_text_records(text_model / "points3D.txt")
    return ModelSummary(images, cameras, points)


def next_output_paths(workspace: Path) -> tuple[Path, Path]:
    logs = workspace / "logs"
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"_retry{attempt}"
        log = logs / f"incremental_registration_pass1{suffix}.log"
        report = logs / f"incremental_registration_pass1{suffix}_report.json"
        if report.exists():
            raise IncrementalRegistrationError(
                "First registration pass already has a completed report."
            )
        if not log.exists():
            return log, report
    raise IncrementalRegistrationError("No unused registration retry log remains.")


def build_plan(*, colmap: Path, workspace: Path) -> RegistrationPlan:
    colmap = colmap.expanduser().resolve()
    workspace = resolve_workspace(workspace)
    if not colmap.is_file():
        raise IncrementalRegistrationError(f"COLMAP executable was not found: {colmap}")
    if not workspace.is_dir():
        raise IncrementalRegistrationError(f"Workspace was not found: {workspace}")
    database = workspace / "database.db"
    input_model = workspace / "base_model"
    output_model = workspace / "registered_pass1"
    if not database.is_file():
        raise IncrementalRegistrationError(f"Database was not found: {database}")
    require_model(input_model, "Accepted base model")
    if not output_model.is_dir() or any(output_model.iterdir()):
        raise IncrementalRegistrationError(
            f"Registration output must exist and be empty: {output_model}"
        )
    log, report = next_output_paths(workspace)
    base_summary = summarize_model(colmap, input_model)
    if (
        len(base_summary.images) != EXPECTED_BASE_IMAGES
        or base_summary.cameras != EXPECTED_BASE_CAMERAS
    ):
        raise IncrementalRegistrationError(
            "Expected the side/underside base model to contain "
            f"{EXPECTED_BASE_IMAGES} images and {EXPECTED_BASE_CAMERAS} cameras; "
            f"found {len(base_summary.images)} and {base_summary.cameras}."
        )
    if any(name.startswith("mid25_") for name in base_summary.images):
        raise IncrementalRegistrationError("Base model already contains mid25 images.")
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
    return RegistrationPlan(
        colmap,
        workspace,
        database,
        input_model,
        output_model,
        log,
        report,
        base_summary,
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
            f"COLMAP image registration failed with exit code {return_code}."
        )


def max_pose_change(
    base: Mapping[str, ImagePose], output: Mapping[str, ImagePose]
) -> float:
    return max(
        abs(first - second)
        for name, pose in base.items()
        for first, second in zip(
            (*pose.qvec, *pose.tvec), (*output[name].qvec, *output[name].tvec)
        )
    )


def validate_result(plan: RegistrationPlan) -> dict[str, object]:
    output = summarize_model(plan.colmap, plan.output_model)
    base_names = set(plan.base_summary.images)
    output_names = set(output.images)
    missing_base = base_names - output_names
    if missing_base:
        raise IncrementalRegistrationError(
            f"Registration output lost {len(missing_base)} accepted base image(s)."
        )
    unexpected = [
        name for name in output_names - base_names if not name.startswith("mid25_")
    ]
    if unexpected:
        raise IncrementalRegistrationError(
            f"Registration output contains {len(unexpected)} unexpected image(s)."
        )
    registered_mid25 = sorted(
        name for name in output_names if name.startswith("mid25_")
    )
    if not registered_mid25:
        raise IncrementalRegistrationError("The first pass registered no mid25 images.")
    pose_change = max_pose_change(plan.base_summary.images, output.images)
    if pose_change > 1e-10:
        raise IncrementalRegistrationError(
            f"Accepted base poses changed by up to {pose_change:.3e}."
        )
    return {
        "base_images_preserved": len(base_names),
        "base_pose_max_absolute_change": pose_change,
        "mid25_registered": len(registered_mid25),
        "mid25_remaining": 221 - len(registered_mid25),
        "total_registered": len(output.images),
        "cameras": output.cameras,
        "sparse_points": output.points3d,
        "first_mid25": registered_mid25[0],
        "last_mid25": registered_mid25[-1],
    }


def main() -> int:
    args = parse_args()
    try:
        plan = build_plan(colmap=args.colmap, workspace=args.workspace)
        print("Incremental image registration pass 1")
        print(f"Input model: {plan.input_model}")
        print(f"Output model: {plan.output_model}")
        print(f"Accepted fixed images: {len(plan.base_summary.images)}")
        print(f"Accepted sparse points: {plan.base_summary.points3d}")
        print(f"Log: {plan.log}")
        print(f"Command: {subprocess.list2cmdline(plan.command)}")
        if args.dry_run:
            print("Dry run complete. No registration output was written.")
            return 0
        run_command(plan.command, plan.log)
        report = validate_result(plan)
        plan.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Incremental registration pass 1 complete")
        for key, value in report.items():
            print(f"{key}: {value}")
        print(f"Report: {plan.report}")
        return 0
    except (IncrementalRegistrationError, OSError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
