#!/usr/bin/env python3
"""Append SIFT features for one new view to an isolated COLMAP database."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_COLMAP = Path("C:/Tools/COLMAP-4.1.1/bin/colmap.exe")
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/processed/pot1-unglazed_side_underside_mid25"
DEFAULT_WORKSPACE = DEFAULT_DATA_ROOT / "colmap_side_underside_mid25"
DEFAULT_IMAGES = (
    PROJECT_ROOT / "data/frames_output/pot1-unglazed_side_underside_mid25_frames"
)
DEFAULT_MASKS = DEFAULT_DATA_ROOT / "masks_colmap"
REQUIRED_TABLES = {
    "cameras",
    "images",
    "keypoints",
    "descriptors",
    "matches",
    "two_view_geometries",
}


class IncrementalFeatureError(RuntimeError):
    """A user-correctable incremental feature-extraction error."""


@dataclass(frozen=True)
class DatabaseSnapshot:
    cameras: frozenset[int]
    images: dict[str, tuple[int, int]]
    keypoint_rows: dict[str, int]
    descriptor_rows: dict[str, int]
    matches: int
    two_view_geometries: int


@dataclass(frozen=True)
class ExtractionPlan:
    colmap: Path
    workspace: Path
    database: Path
    images: Path
    masks: Path
    image_list: Path
    log: Path
    report: Path
    new_names: tuple[str, ...]
    before: DatabaseSnapshot
    command: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract masked SIFT features only for images listed in the isolated "
            "incremental workspace."
        )
    )
    parser.add_argument("--colmap", type=Path, default=DEFAULT_COLMAP)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--masks", type=Path, default=DEFAULT_MASKS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the COLMAP command without changing the database.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise IncrementalFeatureError(
            f"{label} must stay inside the project directory: {candidate}"
        ) from error
    return candidate


def database_snapshot(path: Path) -> DatabaseSnapshot:
    if not path.is_file():
        raise IncrementalFeatureError(f"Database was not found: {path}")
    try:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
        try:
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            if integrity != ("ok",):
                raise IncrementalFeatureError(
                    f"Database quick_check failed: {integrity}"
                )
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            missing = REQUIRED_TABLES - tables
            if missing:
                raise IncrementalFeatureError(
                    "Database is missing table(s): " + ", ".join(sorted(missing))
                )
            cameras = frozenset(
                int(row[0]) for row in connection.execute("SELECT camera_id FROM cameras")
            )
            images = {
                str(name): (int(image_id), int(camera_id))
                for image_id, name, camera_id in connection.execute(
                    "SELECT image_id, name, camera_id FROM images"
                )
            }
            keypoint_rows = {
                str(name): int(rows)
                for name, rows in connection.execute(
                    "SELECT images.name, keypoints.rows FROM images "
                    "JOIN keypoints ON keypoints.image_id = images.image_id"
                )
            }
            descriptor_rows = {
                str(name): int(rows)
                for name, rows in connection.execute(
                    "SELECT images.name, descriptors.rows FROM images "
                    "JOIN descriptors ON descriptors.image_id = images.image_id"
                )
            }
            matches = int(connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
            geometries = int(
                connection.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
            )
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise IncrementalFeatureError(f"Could not inspect database: {error}") from error
    return DatabaseSnapshot(
        cameras=cameras,
        images=images,
        keypoint_rows=keypoint_rows,
        descriptor_rows=descriptor_rows,
        matches=matches,
        two_view_geometries=geometries,
    )


def read_image_list(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise IncrementalFeatureError(f"New-view image list was not found: {path}")
    names = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    if not names or any(not name for name in names):
        raise IncrementalFeatureError("New-view image list is empty or malformed.")
    if len(names) != len(set(names)):
        raise IncrementalFeatureError("New-view image list contains duplicates.")
    if any(Path(name).name != name for name in names):
        raise IncrementalFeatureError("New-view image names must be plain filenames.")
    return names


def build_plan(
    *, colmap: Path, workspace: Path, images: Path, masks: Path
) -> ExtractionPlan:
    colmap = colmap.expanduser().resolve()
    workspace = resolve_project_path(workspace, "Workspace")
    images = resolve_project_path(images, "Image directory")
    masks = resolve_project_path(masks, "Mask directory")
    if not colmap.is_file():
        raise IncrementalFeatureError(f"COLMAP executable was not found: {colmap}")
    if not workspace.is_dir():
        raise IncrementalFeatureError(f"Incremental workspace was not found: {workspace}")
    if not images.is_dir() or not masks.is_dir():
        raise IncrementalFeatureError("Expanded image or mask directory was not found.")

    database = workspace / "database.db"
    image_list = workspace / "new_view_images.txt"
    log = workspace / "logs/incremental_feature_extraction.log"
    report = workspace / "logs/incremental_feature_extraction_report.json"
    if log.exists() or report.exists():
        raise IncrementalFeatureError(
            "Feature-extraction output already exists; refusing to repeat this stage."
        )
    new_names = read_image_list(image_list)
    missing_images = [name for name in new_names if not (images / name).is_file()]
    missing_masks = [name for name in new_names if not (masks / f"{name}.png").is_file()]
    if missing_images or missing_masks:
        raise IncrementalFeatureError(
            "New-view inputs are incomplete: "
            f"missing images={len(missing_images)}, missing masks={len(missing_masks)}."
        )

    before = database_snapshot(database)
    overlap = set(new_names) & set(before.images)
    if overlap:
        raise IncrementalFeatureError(
            f"Database already contains {len(overlap)} new-view image(s)."
        )
    if len(before.images) != 728:
        raise IncrementalFeatureError(
            f"Expected 728 accepted base images, found {len(before.images)}."
        )
    if len(before.cameras) != 3:
        raise IncrementalFeatureError(
            f"Expected 3 accepted base cameras, found {len(before.cameras)}."
        )
    missing_base_keypoints = set(before.images) - set(before.keypoint_rows)
    missing_base_descriptors = set(before.images) - set(before.descriptor_rows)
    if missing_base_keypoints or missing_base_descriptors:
        raise IncrementalFeatureError(
            "Accepted base features are incomplete: "
            f"keypoints={len(missing_base_keypoints)}, "
            f"descriptors={len(missing_base_descriptors)}."
        )

    command = (
        str(colmap),
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--image_list_path",
        str(image_list),
        "--ImageReader.mask_path",
        str(masks),
        "--ImageReader.camera_model",
        "SIMPLE_RADIAL",
        "--ImageReader.single_camera",
        "1",
        "--FeatureExtraction.use_gpu",
        "1",
        "--FeatureExtraction.gpu_index",
        "0",
        "--FeatureExtraction.num_threads",
        "1",
        "--FeatureExtraction.max_image_size",
        "3200",
        "--SiftExtraction.max_num_features",
        "8192",
        "--SiftExtraction.first_octave",
        "0",
        "--SiftExtraction.estimate_affine_shape",
        "0",
        "--SiftExtraction.domain_size_pooling",
        "0",
    )
    return ExtractionPlan(
        colmap=colmap,
        workspace=workspace,
        database=database,
        images=images,
        masks=masks,
        image_list=image_list,
        log=log,
        report=report,
        new_names=new_names,
        before=before,
        command=command,
    )


def run_command(command: Sequence[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
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
            sys.stdout.write(line)
            handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise IncrementalFeatureError(
            f"COLMAP feature extraction failed with exit code {return_code}."
        )


def validate_result(plan: ExtractionPlan) -> dict[str, object]:
    after = database_snapshot(plan.database)
    base_names = set(plan.before.images)
    new_names = set(plan.new_names)
    if set(after.images) != base_names | new_names:
        raise IncrementalFeatureError(
            "Database image set is incorrect after feature extraction."
        )
    if any(after.images[name] != plan.before.images[name] for name in base_names):
        raise IncrementalFeatureError("Existing base image IDs or camera IDs changed.")
    if any(
        after.keypoint_rows.get(name) != plan.before.keypoint_rows.get(name)
        or after.descriptor_rows.get(name) != plan.before.descriptor_rows.get(name)
        for name in base_names
    ):
        raise IncrementalFeatureError("Existing base SIFT features changed.")
    if (
        after.matches != plan.before.matches
        or after.two_view_geometries != plan.before.two_view_geometries
    ):
        raise IncrementalFeatureError("Existing match records changed unexpectedly.")
    incomplete = [
        name
        for name in plan.new_names
        if after.keypoint_rows.get(name, 0) <= 0
        or after.descriptor_rows.get(name, 0) <= 0
    ]
    if incomplete:
        raise IncrementalFeatureError(
            f"New SIFT features are incomplete for {len(incomplete)} image(s)."
        )
    new_camera_ids = {after.images[name][1] for name in new_names}
    if len(new_camera_ids) != 1 or new_camera_ids & plan.before.cameras:
        raise IncrementalFeatureError(
            "New images were not assigned exactly one separate camera calibration."
        )
    if len(after.cameras) != 4:
        raise IncrementalFeatureError(
            f"Expected 4 cameras after extraction, found {len(after.cameras)}."
        )
    keypoint_counts = [after.keypoint_rows[name] for name in plan.new_names]
    return {
        "base_images_preserved": len(base_names),
        "new_images": len(new_names),
        "total_images": len(after.images),
        "total_cameras": len(after.cameras),
        "new_camera_id": next(iter(new_camera_ids)),
        "new_keypoints_total": sum(keypoint_counts),
        "new_keypoints_min": min(keypoint_counts),
        "new_keypoints_max": max(keypoint_counts),
        "base_matches_preserved": after.matches,
        "base_two_view_geometries_preserved": after.two_view_geometries,
    }


def print_plan(plan: ExtractionPlan, dry_run: bool) -> None:
    print("Incremental masked SIFT feature extraction")
    print(f"Database: {plan.database}")
    print(f"Accepted base images: {len(plan.before.images)}")
    print(f"Accepted base cameras: {len(plan.before.cameras)}")
    print(f"New images to extract: {len(plan.new_names)}")
    print("New camera policy: one separate SIMPLE_RADIAL camera")
    print("SIFT profile: max image 3200, first octave 0, max 8192 features")
    print(f"Command: {subprocess.list2cmdline(plan.command)}")
    if dry_run:
        print("Dry run complete. Database and logs were not modified.")


def main() -> int:
    args = parse_args()
    try:
        plan = build_plan(
            colmap=args.colmap,
            workspace=args.workspace,
            images=args.images,
            masks=args.masks,
        )
        print_plan(plan, args.dry_run)
        if args.dry_run:
            return 0
        run_command(plan.command, plan.log)
        report = validate_result(plan)
        plan.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Incremental feature extraction complete")
        for key, value in report.items():
            print(f"{key}: {value}")
        print(f"Log: {plan.log}")
        print(f"Report: {plan.report}")
        return 0
    except IncrementalFeatureError as error:
        print(f"ERROR: {error}")
        return 1
    except OSError as error:
        print(f"ERROR: Could not run incremental feature extraction: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
