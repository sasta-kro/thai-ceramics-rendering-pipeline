#!/usr/bin/env python3
"""Prepare a side + underside + mid25 reconstruction from existing results.

The completed four-angle database is copied into an isolated workspace. Only
top45 images and records involving those images are removed from the copy. The
accepted 728-camera base model is filtered to its 495 side and underside poses.
No source database, sparse model, image, or mask is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from typing import Mapping
import uuid


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from run_incremental_registration import (  # noqa: E402
    IncrementalRegistrationError,
    max_pose_change,
    summarize_model,
)


DEFAULT_COLMAP = Path("C:/Tools/COLMAP-4.1.1/bin/colmap.exe")
SOURCE_ROOT = PROJECT_ROOT / "data/processed/pot1-unglazed_multiview_mid25"
DEFAULT_SOURCE_WORKSPACE = SOURCE_ROOT / "colmap_incremental_mid25"
TARGET_ROOT = PROJECT_ROOT / "data/processed/pot1-unglazed_side_underside_mid25"
DEFAULT_MANIFEST = TARGET_ROOT / "dataset_manifest.csv"
DEFAULT_IMAGES = (
    PROJECT_ROOT / "data/frames_output/pot1-unglazed_side_underside_mid25_frames"
)
DEFAULT_MASKS = TARGET_ROOT / "masks_colmap"
DEFAULT_WORKSPACE = TARGET_ROOT / "colmap_side_underside_mid25"
MAX_IMAGE_ID = 2**31 - 1
EXPECTED_COUNTS = {"side": 273, "mid25": 221, "underside": 222}
OUTPUT_DIRECTORIES = (
    "registered_pass1",
    "triangulated_pass1",
    "registered_final",
    "triangulated_final",
    "bundle_adjusted_final",
    "logs",
)


class WorkspacePreparationError(RuntimeError):
    """A user-correctable filtered-workspace preparation error."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse the four-angle features/matches while excluding top45 from "
            "a new side + underside + mid25 COLMAP workspace."
        )
    )
    parser.add_argument("--colmap", type=Path, default=DEFAULT_COLMAP)
    parser.add_argument(
        "--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--masks", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_project_path(path: Path, label: str, *, must_exist: bool = True) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise WorkspacePreparationError(
            f"{label} must stay inside the project: {candidate}"
        ) from error
    if not relative.parts:
        raise WorkspacePreparationError(f"{label} cannot be the project root.")
    if must_exist and not candidate.exists():
        raise WorkspacePreparationError(f"{label} was not found: {candidate}")
    return candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"view", "combined_filename"}
    if not rows or not required.issubset(rows[0]):
        raise WorkspacePreparationError(
            "Filtered manifest is empty or missing view/combined_filename columns."
        )
    result: dict[str, str] = {}
    counts = {group: 0 for group in EXPECTED_COUNTS}
    for row in rows:
        view = row["view"]
        name = row["combined_filename"]
        if view not in EXPECTED_COUNTS:
            raise WorkspacePreparationError(
                f"Filtered manifest contains unexpected view {view!r}."
            )
        if name in result:
            raise WorkspacePreparationError(f"Duplicate manifest image: {name}")
        result[name] = view
        counts[view] += 1
    if counts != EXPECTED_COUNTS:
        raise WorkspacePreparationError(
            f"Filtered manifest counts differ: {counts} != {EXPECTED_COUNTS}."
        )
    return result


def validate_staging(
    names: Mapping[str, str], images: Path, masks: Path
) -> None:
    image_names = {path.name for path in images.iterdir() if path.is_file()}
    mask_names = {
        path.name[: -len(".png")]
        for path in masks.iterdir()
        if path.is_file() and path.name.endswith(".png")
    }
    expected = set(names)
    if image_names != expected:
        raise WorkspacePreparationError(
            "Image staging/manifest mismatch: "
            f"{len(expected - image_names)} missing, "
            f"{len(image_names - expected)} unexpected."
        )
    if mask_names != expected:
        raise WorkspacePreparationError(
            "Mask staging/manifest mismatch: "
            f"{len(expected - mask_names)} missing, "
            f"{len(mask_names - expected)} unexpected."
        )


def image_ids_from_pair_id(pair_id: int) -> tuple[int, int]:
    second = pair_id % MAX_IMAGE_ID
    first = (pair_id - second) // MAX_IMAGE_ID
    if first <= 0 or second <= 0 or first >= second:
        raise WorkspacePreparationError(f"Invalid COLMAP pair_id: {pair_id}")
    return first, second


def database_snapshot(path: Path) -> dict[str, object]:
    try:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
        try:
            if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise WorkspacePreparationError(f"Database quick_check failed: {path}")
            images = {
                int(image_id): (str(name), int(camera_id))
                for image_id, name, camera_id in connection.execute(
                    "SELECT image_id, name, camera_id FROM images"
                )
            }
            keypoints = {
                int(image_id): int(rows)
                for image_id, rows in connection.execute(
                    "SELECT image_id, rows FROM keypoints"
                )
            }
            descriptors = {
                int(image_id): int(rows)
                for image_id, rows in connection.execute(
                    "SELECT image_id, rows FROM descriptors"
                )
            }
            pairs = {
                table: {
                    int(pair_id): int(rows)
                    for pair_id, rows in connection.execute(
                        f"SELECT pair_id, rows FROM {table}"
                    )
                }
                for table in ("matches", "two_view_geometries")
            }
            frames = int(connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
            frame_data = int(
                connection.execute("SELECT COUNT(*) FROM frame_data").fetchone()[0]
            )
            cameras = int(
                connection.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
            )
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise WorkspacePreparationError(f"Could not inspect database {path}: {error}") from error
    return {
        "images": images,
        "keypoints": keypoints,
        "descriptors": descriptors,
        "pairs": pairs,
        "frames": frames,
        "frame_data": frame_data,
        "cameras": cameras,
    }


def filter_database(path: Path, excluded_ids: set[int]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        for table in ("matches", "two_view_geometries"):
            stale = [
                (int(pair_id),)
                for (pair_id,) in connection.execute(f"SELECT pair_id FROM {table}")
                if excluded_ids.intersection(image_ids_from_pair_id(int(pair_id)))
            ]
            connection.executemany(f"DELETE FROM {table} WHERE pair_id = ?", stale)
        placeholders = ",".join("?" for _ in excluded_ids)
        parameters = tuple(sorted(excluded_ids))
        frame_ids = [
            int(row[0])
            for row in connection.execute(
                f"SELECT frame_id FROM frame_data WHERE sensor_type = 0 "
                f"AND data_id IN ({placeholders})",
                parameters,
            )
        ]
        connection.execute(
            f"DELETE FROM pose_priors WHERE corr_sensor_type = 0 "
            f"AND corr_data_id IN ({placeholders})",
            parameters,
        )
        connection.execute(
            f"DELETE FROM frame_data WHERE sensor_type = 0 "
            f"AND data_id IN ({placeholders})",
            parameters,
        )
        if frame_ids:
            frame_placeholders = ",".join("?" for _ in frame_ids)
            connection.execute(
                f"DELETE FROM frames WHERE frame_id IN ({frame_placeholders})",
                tuple(frame_ids),
            )
        connection.execute(
            f"DELETE FROM keypoints WHERE image_id IN ({placeholders})", parameters
        )
        connection.execute(
            f"DELETE FROM descriptors WHERE image_id IN ({placeholders})", parameters
        )
        connection.execute(
            f"DELETE FROM images WHERE image_id IN ({placeholders})", parameters
        )
        connection.commit()
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise WorkspacePreparationError("Filtered database quick_check failed.")
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def validate_database_transition(
    before: Mapping[str, object], after: Mapping[str, object], expected_names: set[str]
) -> dict[str, int]:
    before_images = before["images"]
    after_images = after["images"]
    assert isinstance(before_images, dict) and isinstance(after_images, dict)
    if len(before_images) != 949 or len(after_images) != 716:
        raise WorkspacePreparationError(
            f"Unexpected database image transition: {len(before_images)} -> {len(after_images)}."
        )
    if {record[0] for record in after_images.values()} != expected_names:
        raise WorkspacePreparationError("Filtered database image names do not match manifest.")
    retained_ids = set(after_images)
    for feature in ("keypoints", "descriptors"):
        before_rows = before[feature]
        after_rows = after[feature]
        assert isinstance(before_rows, dict) and isinstance(after_rows, dict)
        expected_rows = {
            image_id: rows
            for image_id, rows in before_rows.items()
            if image_id in retained_ids
        }
        if after_rows != expected_rows:
            raise WorkspacePreparationError(
                f"Filtered database changed retained {feature}."
            )
    pair_counts: dict[str, int] = {}
    before_pairs = before["pairs"]
    after_pairs = after["pairs"]
    assert isinstance(before_pairs, dict) and isinstance(after_pairs, dict)
    for table in ("matches", "two_view_geometries"):
        expected_pairs = {
            pair_id: rows
            for pair_id, rows in before_pairs[table].items()
            if set(image_ids_from_pair_id(pair_id)).issubset(retained_ids)
        }
        if after_pairs[table] != expected_pairs:
            raise WorkspacePreparationError(
                f"Filtered database changed retained {table} records."
            )
        pair_counts[table] = len(expected_pairs)
    return pair_counts


def filter_bridge_pairs(source: Path, retained_names: set[str]) -> tuple[str, ...]:
    pairs: list[str] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if len(fields) != 2:
            raise WorkspacePreparationError(
                f"Malformed bridge pair at line {line_number}: {line}"
            )
        if fields[0] in retained_names and fields[1] in retained_names:
            pairs.append(f"{fields[0]} {fields[1]}")
    if len(pairs) != 2520:
        raise WorkspacePreparationError(
            f"Expected 2520 retained side-mid25 bridge pairs; found {len(pairs)}."
        )
    return tuple(pairs)


def run_image_deleter(
    colmap: Path,
    source_model: Path,
    output_model: Path,
    excluded_names: Path,
    log: Path,
) -> None:
    command = (
        str(colmap),
        "image_deleter",
        "--input_path",
        str(source_model),
        "--output_path",
        str(output_model),
        "--image_names_path",
        str(excluded_names),
        "--log_target",
        "stderr",
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = completed.stdout or completed.stderr
    log.write_text(output, encoding="utf-8")
    if completed.returncode != 0:
        raise WorkspacePreparationError(
            f"COLMAP image_deleter failed with exit code {completed.returncode}."
        )


def prepare_workspace(args: argparse.Namespace) -> dict[str, object]:
    colmap = args.colmap.expanduser().resolve()
    source_workspace = resolve_project_path(args.source_workspace, "Source workspace")
    manifest = resolve_project_path(args.manifest, "Filtered manifest")
    images = resolve_project_path(args.images, "Filtered images")
    masks = resolve_project_path(args.masks, "Filtered masks")
    workspace = resolve_project_path(args.workspace, "Output workspace", must_exist=False)
    if not colmap.is_file():
        raise WorkspacePreparationError(f"COLMAP executable was not found: {colmap}")
    if workspace.exists():
        raise WorkspacePreparationError(
            f"Output workspace already exists; refusing to overwrite: {workspace}"
        )
    source_database = source_workspace / "database.db"
    source_model = source_workspace / "base_model"
    source_bridge_pairs = source_workspace / "bridge_pairs_lightglue.txt"
    source_sequential_pairs = source_workspace / "new_view_sequential_pairs.txt"
    source_new_images = source_workspace / "new_view_images.txt"
    for path in (
        source_database,
        source_bridge_pairs,
        source_sequential_pairs,
        source_new_images,
    ):
        if not path.is_file():
            raise WorkspacePreparationError(f"Required source file was not found: {path}")
    retained_views = load_manifest(manifest)
    retained_names = set(retained_views)
    validate_staging(retained_views, images, masks)
    before = database_snapshot(source_database)
    before_images = before["images"]
    assert isinstance(before_images, dict)
    excluded = {
        image_id: record[0]
        for image_id, record in before_images.items()
        if record[0].startswith("top45_")
    }
    if len(excluded) != 233:
        raise WorkspacePreparationError(
            f"Expected 233 top45 database images; found {len(excluded)}."
        )
    retained_database_ids = {
        image_id
        for image_id, record in before_images.items()
        if record[0] in retained_names
    }
    if set(before_images) - set(excluded) != retained_database_ids:
        raise WorkspacePreparationError(
            "Four-angle database contains unexpected images outside the filtered manifest."
        )
    source_summary = summarize_model(colmap, source_model)
    source_top_names = {
        name for name in source_summary.images if name.startswith("top45_")
    }
    if len(source_summary.images) != 728 or len(source_top_names) != 233:
        raise WorkspacePreparationError(
            "Accepted source model must contain 728 images including 233 top45 images."
        )
    retained_base_names = set(source_summary.images) - source_top_names
    if len(retained_base_names) != 495 or any(
        name.startswith("mid25_") for name in retained_base_names
    ):
        raise WorkspacePreparationError("Filtered base pose set is not side + underside.")
    bridges = filter_bridge_pairs(source_bridge_pairs, retained_names)
    source_database_hash = sha256(source_database)
    source_model_hashes = {
        name: sha256(source_model / name)
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    }
    plan = {
        "source_database": str(source_database),
        "source_model": str(source_model),
        "workspace": str(workspace),
        "retained_database_images": len(retained_names),
        "excluded_top45_images": len(excluded),
        "retained_base_images": len(retained_base_names),
        "retained_side_mid25_bridge_pairs": len(bridges),
    }
    if args.dry_run:
        return {**plan, "written": False}

    staging = workspace.parent / f".{workspace.name}-staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        target_database = staging / "database.db"
        shutil.copy2(source_database, target_database)
        filter_database(target_database, set(excluded))
        after = database_snapshot(target_database)
        pair_counts = validate_database_transition(before, after, retained_names)

        excluded_names = staging / "excluded_top45_images.txt"
        excluded_names.write_text(
            "\n".join(sorted(source_top_names)) + "\n", encoding="utf-8"
        )
        target_model = staging / "base_model"
        target_model.mkdir()
        logs = staging / "logs"
        logs.mkdir()
        run_image_deleter(
            colmap,
            source_model,
            target_model,
            excluded_names,
            logs / "prepare_base_model_without_top45.log",
        )
        filtered_summary = summarize_model(colmap, target_model)
        if set(filtered_summary.images) != retained_base_names:
            raise WorkspacePreparationError(
                "Filtered base model does not contain exactly the 495 retained poses."
            )
        pose_change = max_pose_change(
            {name: source_summary.images[name] for name in retained_base_names},
            filtered_summary.images,
        )
        if pose_change > 1e-10:
            raise WorkspacePreparationError(
                f"Filtered base poses changed by up to {pose_change:.3e}."
            )

        shutil.copy2(source_sequential_pairs, staging / source_sequential_pairs.name)
        shutil.copy2(source_new_images, staging / source_new_images.name)
        (staging / "bridge_pairs_lightglue.txt").write_text(
            "\n".join(bridges) + "\n", encoding="utf-8"
        )
        for name in OUTPUT_DIRECTORIES:
            (staging / name).mkdir(exist_ok=True)
        report = {
            **plan,
            "written": True,
            "database_images": len(after["images"]),
            "database_cameras_retained": after["cameras"],
            "database_match_records": pair_counts["matches"],
            "database_verified_records": pair_counts["two_view_geometries"],
            "base_model_images": len(filtered_summary.images),
            "base_model_cameras": filtered_summary.cameras,
            "base_model_points": filtered_summary.points3d,
            "base_pose_max_absolute_change": pose_change,
            "source_database_sha256": source_database_hash,
            "source_model_sha256": source_model_hashes,
        }
        (staging / "preparation_manifest.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        if sha256(source_database) != source_database_hash:
            raise WorkspacePreparationError("Source database changed during preparation.")
        for name, expected_hash in source_model_hashes.items():
            if sha256(source_model / name) != expected_hash:
                raise WorkspacePreparationError(
                    f"Source base model file changed during preparation: {name}"
                )
        staging.replace(workspace)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> int:
    args = parse_args()
    try:
        report = prepare_workspace(args)
        print("Side + underside + mid25 workspace preparation")
        for key, value in report.items():
            if key != "source_model_sha256":
                print(f"{key}: {value}")
        if args.dry_run:
            print("Dry run complete. No workspace was written.")
        else:
            print("Filtered workspace preparation complete.")
        return 0
    except (WorkspacePreparationError, IncrementalRegistrationError, OSError, sqlite3.Error) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
