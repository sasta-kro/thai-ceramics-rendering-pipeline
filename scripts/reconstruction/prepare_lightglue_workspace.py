#!/usr/bin/env python3
"""Prepare an isolated COLMAP workspace for targeted SIFT-LightGlue matching."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MULTIVIEW_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "pot1-unglazed_multiview"
)
DEFAULT_SOURCE_WORKSPACE = (
    DEFAULT_MULTIVIEW_ROOT / "colmap_sparse_masked_multiview"
)
DEFAULT_MANIFEST = DEFAULT_MULTIVIEW_ROOT / "dataset_manifest.csv"
DEFAULT_WORKSPACE = (
    DEFAULT_MULTIVIEW_ROOT / "colmap_sparse_masked_multiview_lightglue"
)
REQUIRED_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
OUTPUT_DIRECTORIES = (
    "registered_pass1",
    "triangulated_pass1",
    "registered_final",
    "logs",
)


class WorkspacePreparationError(RuntimeError):
    """A user-correctable LightGlue workspace preparation error."""


@dataclass(frozen=True)
class ManifestFrame:
    view: str
    view_frame_index: int
    filename: str


@dataclass(frozen=True)
class PreparationPlan:
    source_database: Path
    source_model: Path
    manifest: Path
    workspace: Path
    left_view: str
    right_view: str
    stride: int
    left_images: int
    right_images: int
    left_anchors: int
    right_anchors: int
    pairs: tuple[tuple[str, str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the multiview dataset, copy the existing COLMAP database and "
            "Model 2, and generate targeted side-to-top LightGlue pairs."
        )
    )
    parser.add_argument(
        "--source-database",
        type=Path,
        default=DEFAULT_SOURCE_WORKSPACE / "database.db",
        help="Existing multiview COLMAP database to copy.",
    )
    parser.add_argument(
        "--source-model",
        type=Path,
        default=DEFAULT_SOURCE_WORKSPACE / "sparse" / "2",
        help="Existing side-and-underside sparse model to copy as base_model.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Combined multiview dataset manifest.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="New isolated LightGlue workspace. It must not already exist.",
    )
    parser.add_argument("--left-view", default="side")
    parser.add_argument("--right-view", default="top45")
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Select every Nth frame in each view, plus each final frame (default: 5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report the plan without writing any files.",
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
        raise WorkspacePreparationError(
            f"{label} must stay inside the project directory: {candidate}"
        ) from error
    return candidate


def load_manifest(path: Path) -> dict[str, tuple[ManifestFrame, ...]]:
    required_fields = {"view", "view_frame_index", "combined_filename"}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = required_fields - set(reader.fieldnames or ())
            if missing:
                raise WorkspacePreparationError(
                    "Manifest is missing field(s): " + ", ".join(sorted(missing))
                )
            rows = list(reader)
    except FileNotFoundError as error:
        raise WorkspacePreparationError(f"Manifest was not found: {path}") from error
    except OSError as error:
        raise WorkspacePreparationError(
            f"Could not read manifest {path}: {error}"
        ) from error

    frames_by_view: dict[str, list[ManifestFrame]] = {}
    filenames: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        view = (row.get("view") or "").strip()
        filename = (row.get("combined_filename") or "").strip()
        if not view or any(character.isspace() for character in view):
            raise WorkspacePreparationError(
                f"Invalid view name in manifest row {row_number}."
            )
        if (
            not filename
            or Path(filename).name != filename
            or any(character.isspace() for character in filename)
        ):
            raise WorkspacePreparationError(
                f"Invalid combined filename in manifest row {row_number}: {filename!r}"
            )
        if filename in filenames:
            raise WorkspacePreparationError(f"Duplicate manifest image: {filename}")
        filenames.add(filename)
        try:
            frame_index = int(row.get("view_frame_index", ""))
        except ValueError as error:
            raise WorkspacePreparationError(
                f"Invalid view_frame_index in manifest row {row_number}."
            ) from error
        if frame_index < 0:
            raise WorkspacePreparationError(
                f"Negative view_frame_index in manifest row {row_number}."
            )
        frames_by_view.setdefault(view, []).append(
            ManifestFrame(view, frame_index, filename)
        )

    result: dict[str, tuple[ManifestFrame, ...]] = {}
    for view, frames in frames_by_view.items():
        frames.sort(key=lambda frame: frame.view_frame_index)
        actual_indices = [frame.view_frame_index for frame in frames]
        if actual_indices != list(range(len(frames))):
            raise WorkspacePreparationError(
                f"Manifest indices for view '{view}' must be contiguous from 0."
            )
        result[view] = tuple(frames)
    return result


def sqlite_read_only_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro&immutable=1"


def validate_database(path: Path, manifest_filenames: set[str]) -> None:
    if not path.is_file():
        raise WorkspacePreparationError(f"Source database was not found: {path}")
    try:
        connection = sqlite3.connect(sqlite_read_only_uri(path), uri=True)
        try:
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            if integrity != ("ok",):
                raise WorkspacePreparationError(
                    f"Source database quick_check failed: {integrity}"
                )
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            required_tables = {"images", "keypoints", "descriptors"}
            missing_tables = required_tables - tables
            if missing_tables:
                raise WorkspacePreparationError(
                    "Source database is missing table(s): "
                    + ", ".join(sorted(missing_tables))
                )
            database_images = {
                row[0] for row in connection.execute("SELECT name FROM images")
            }
            keypoint_images = {
                row[0]
                for row in connection.execute(
                    "SELECT images.name FROM images "
                    "JOIN keypoints ON keypoints.image_id = images.image_id"
                )
            }
            descriptor_images = {
                row[0]
                for row in connection.execute(
                    "SELECT images.name FROM images "
                    "JOIN descriptors ON descriptors.image_id = images.image_id"
                )
            }
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise WorkspacePreparationError(
            f"Could not validate source database {path}: {error}"
        ) from error

    missing_from_database = manifest_filenames - database_images
    extra_in_database = database_images - manifest_filenames
    if missing_from_database or extra_in_database:
        details = []
        if missing_from_database:
            details.append(f"{len(missing_from_database)} manifest image(s) missing")
        if extra_in_database:
            details.append(f"{len(extra_in_database)} unexpected database image(s)")
        raise WorkspacePreparationError(
            "Manifest/database image mismatch: " + ", ".join(details)
        )
    missing_keypoints = database_images - keypoint_images
    missing_descriptors = database_images - descriptor_images
    if missing_keypoints or missing_descriptors:
        raise WorkspacePreparationError(
            "Existing SIFT features are incomplete: "
            f"{len(missing_keypoints)} image(s) lack keypoints and "
            f"{len(missing_descriptors)} image(s) lack descriptors."
        )


def validate_model(path: Path) -> None:
    if not path.is_dir():
        raise WorkspacePreparationError(f"Source base model was not found: {path}")
    missing = [name for name in REQUIRED_MODEL_FILES if not (path / name).is_file()]
    if missing:
        raise WorkspacePreparationError(
            "Source base model is missing file(s): " + ", ".join(missing)
        )


def anchor_frames(
    sequence: Sequence[ManifestFrame], stride: int
) -> tuple[ManifestFrame, ...]:
    if stride <= 0:
        raise WorkspacePreparationError("Stride must be a positive integer.")
    anchors = list(sequence[::stride])
    if sequence and anchors[-1] != sequence[-1]:
        anchors.append(sequence[-1])
    return tuple(anchors)


def build_bridge_pairs(
    left_sequence: Sequence[ManifestFrame],
    right_sequence: Sequence[ManifestFrame],
    stride: int,
) -> tuple[tuple[str, str], ...]:
    left_anchors = anchor_frames(left_sequence, stride)
    right_anchors = anchor_frames(right_sequence, stride)
    return tuple(
        (left.filename, right.filename)
        for left, right in product(left_anchors, right_anchors)
    )


def build_plan(
    *,
    source_database: Path,
    source_model: Path,
    manifest: Path,
    workspace: Path,
    left_view: str,
    right_view: str,
    stride: int,
) -> PreparationPlan:
    source_database = resolve_project_path(source_database, "Source database")
    source_model = resolve_project_path(source_model, "Source model")
    manifest = resolve_project_path(manifest, "Manifest")
    workspace = resolve_project_path(workspace, "Workspace")
    if left_view == right_view:
        raise WorkspacePreparationError("Bridge views must be different.")
    if stride <= 0:
        raise WorkspacePreparationError("Stride must be a positive integer.")
    for source, label in (
        (source_database, "source database"),
        (source_model, "source model"),
        (manifest, "manifest"),
    ):
        if source == workspace or workspace in source.parents:
            raise WorkspacePreparationError(
                f"Workspace must be separate from the {label}: {source}"
            )
    if workspace.exists():
        raise WorkspacePreparationError(
            f"LightGlue workspace already exists; refusing to overwrite it: {workspace}"
        )

    sequences = load_manifest(manifest)
    missing_views = [view for view in (left_view, right_view) if view not in sequences]
    if missing_views:
        raise WorkspacePreparationError(
            "Manifest is missing bridge view(s): " + ", ".join(missing_views)
        )
    manifest_filenames = {
        frame.filename for sequence in sequences.values() for frame in sequence
    }
    validate_database(source_database, manifest_filenames)
    validate_model(source_model)

    left_sequence = sequences[left_view]
    right_sequence = sequences[right_view]
    left_anchors = anchor_frames(left_sequence, stride)
    right_anchors = anchor_frames(right_sequence, stride)
    pairs = build_bridge_pairs(left_sequence, right_sequence, stride)
    if not pairs:
        raise WorkspacePreparationError("Targeted bridge-pair plan is empty.")
    if len(set(pairs)) != len(pairs):
        raise WorkspacePreparationError(
            "Targeted bridge-pair plan contains duplicates."
        )
    return PreparationPlan(
        source_database=source_database,
        source_model=source_model,
        manifest=manifest,
        workspace=workspace,
        left_view=left_view,
        right_view=right_view,
        stride=stride,
        left_images=len(left_sequence),
        right_images=len(right_sequence),
        left_anchors=len(left_anchors),
        right_anchors=len(right_anchors),
        pairs=pairs,
    )


def write_pair_list(path: Path, pairs: Sequence[tuple[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for left, right in pairs:
            handle.write(f"{left} {right}\n")


def prepare_workspace(plan: PreparationPlan) -> None:
    plan.workspace.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{plan.workspace.name}-",
            suffix=".tmp",
            dir=plan.workspace.parent,
        )
    )
    try:
        shutil.copy2(plan.source_database, staging / "database.db")
        if (
            (staging / "database.db").stat().st_size
            != plan.source_database.stat().st_size
        ):
            raise WorkspacePreparationError(
                "Copied database size does not match source."
            )
        copied_manifest = load_manifest(plan.manifest)
        validate_database(
            staging / "database.db",
            {
                frame.filename
                for sequence in copied_manifest.values()
                for frame in sequence
            },
        )
        shutil.copytree(plan.source_model, staging / "base_model")
        write_pair_list(staging / "bridge_pairs_lightglue.txt", plan.pairs)
        for directory in OUTPUT_DIRECTORIES:
            (staging / directory).mkdir()
        staging.rename(plan.workspace)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def print_plan(plan: PreparationPlan, dry_run: bool) -> None:
    mode = "Dry run" if dry_run else "Prepared"
    print(f"{mode}: isolated SIFT-LightGlue workspace")
    print(f"Source database: {plan.source_database}")
    print(f"Source base model: {plan.source_model}")
    print(f"Manifest: {plan.manifest}")
    print(f"Workspace: {plan.workspace}")
    print(
        f"{plan.left_view}: {plan.left_images} images, "
        f"{plan.left_anchors} anchors"
    )
    print(
        f"{plan.right_view}: {plan.right_images} images, "
        f"{plan.right_anchors} anchors"
    )
    print(f"Bridge stride: {plan.stride}")
    print(f"Targeted bridge pairs: {len(plan.pairs)}")
    if not dry_run:
        print("Original database and sparse models were not modified.")


def main() -> int:
    args = parse_args()
    try:
        plan = build_plan(
            source_database=args.source_database,
            source_model=args.source_model,
            manifest=args.manifest,
            workspace=args.workspace,
            left_view=args.left_view.strip(),
            right_view=args.right_view.strip(),
            stride=args.stride,
        )
        if not args.dry_run:
            prepare_workspace(plan)
        print_plan(plan, args.dry_run)
        return 0
    except WorkspacePreparationError as error:
        print(f"ERROR: {error}")
        return 1
    except OSError as error:
        print(f"ERROR: Could not prepare LightGlue workspace: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
