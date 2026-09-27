#!/usr/bin/env python3
"""Remove reconstructable project data while preserving accepted final artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


# Complete output directories retained in their original locations.
KEEP_DIRECTORIES = (
    # Pot 1 side-only COLMAP, classical reconstruction, and 3DGS baseline.
    "processed/pot1-unglazed_every6/colmap_sparse_masked_sequential/sparse/0",
    "processed/pot1-unglazed_every6/colmap_dense_masked_sequential/results",
    "processed/pot1-unglazed_every6/gaussian_splatting_masked_sequential/runs/baseline_7k",
    # Pot 1 repaired multiview COLMAP and quality 3DGS result.
    "processed/pot1-unglazed_multiview/colmap_sparse_masked_multiview_lightglue_repaired/bundle_adjusted_final",
    "processed/pot1-unglazed_multiview/gaussian_splatting_masked_multiview_lightglue_repaired/runs/quality_multiview_repaired",
    # Pot 1 side + intermediate-25 + underside result.
    "processed/pot1-unglazed_side_underside_mid25/colmap_side_underside_mid25/bundle_adjusted_final",
    "processed/pot1-unglazed_side_underside_mid25/gaussian_splatting/runs/baseline_side_under_mid25_7k",
    # Lid result.
    "processed/lid_side_only/gs_input/sparse",
    "processed/lid_side_only/gaussian_splatting/runs/baseline_lid_7k",
    # Lift result.
    "processed/lift_all_angles/colmap_sparse_masked_multiview/sparse_filtered_openwork",
    "processed/lift_all_angles/gaussian_splatting/runs/baseline_lift_7k",
)


# Small final reports and manifests retained beside the accepted models.
KEEP_FILES = (
    "processed/pot1-unglazed_every6/colmap_sparse_masked_sequential/sparse_model_0.ply",
    "processed/pot1-unglazed_every6/gaussian_splatting_masked_sequential/dataset_manifest.json",
    "processed/pot1-unglazed_multiview/dataset_manifest.csv",
    "processed/pot1-unglazed_multiview/colmap_sparse_masked_multiview_lightglue_repaired/reports/pose_alignment.json",
    "processed/pot1-unglazed_multiview/gaussian_splatting_masked_multiview_lightglue_repaired/dataset_manifest.json",
    "processed/pot1-unglazed_side_underside_mid25/dataset_manifest.csv",
    "processed/pot1-unglazed_side_underside_mid25/colmap_side_underside_mid25/logs/camera_trajectory_bundle_adjusted_final.json",
    "processed/pot1-unglazed_side_underside_mid25/colmap_side_underside_mid25/logs/final_bundle_adjustment_report.json",
    "processed/pot1-unglazed_side_underside_mid25/colmap_side_underside_mid25/logs/final_triangulation_report.json",
    "processed/pot1-unglazed_side_underside_mid25/gaussian_splatting/dataset_manifest.json",
    "processed/lid_side_only/dataset_manifest.csv",
    "processed/lid_side_only/gaussian_splatting/dataset_manifest.json",
    "processed/lift_all_angles/dataset_manifest.csv",
    "processed/lift_all_angles/colmap_sparse_masked_multiview/logs/point_filtering_report.json",
    "processed/lift_all_angles/gaussian_splatting/dataset_manifest.json",
)


# This rejected workspace has a legacy restrictive ACL and is removed with an
# explicit elevated PowerShell operation after the normal file cleanup.
SEPARATE_REMOVE_DIRECTORIES = (
    "processed/lid_side_underside",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete raw captures, extracted frames, caches, masks, and intermediate "
            "reconstruction files while retaining accepted final artifacts."
        )
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Perform deletion. Without this flag, only print the cleanup audit.",
    )
    return parser.parse_args()


def resolve_inside_data(relative: str) -> Path:
    candidate = (DATA_ROOT / relative).resolve()
    candidate.relative_to(DATA_ROOT.resolve())
    return candidate


def is_inside(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def is_ancestor(path: Path, target: Path) -> bool:
    return path == target or path in target.parents


def walk_data(
    keep_directories: tuple[Path, ...],
    separate_remove_directories: tuple[Path, ...],
) -> tuple[list[Path], list[OSError]]:
    files: list[Path] = []
    errors: list[OSError] = []

    def collect_error(error: OSError) -> None:
        errors.append(error)

    for root, directory_names, file_names in os.walk(
        DATA_ROOT, topdown=True, onerror=collect_error, followlinks=False
    ):
        root_path = Path(root)
        for name in file_names:
            files.append(root_path / name)
        # Symlinked directories are treated as entries and never followed.
        for name in list(directory_names):
            candidate = root_path / name
            if candidate.is_symlink():
                files.append(candidate)
                directory_names.remove(name)
            elif candidate.resolve() in (
                *keep_directories,
                *separate_remove_directories,
            ):
                # Accepted result directories are preserved as complete units.
                # Do not traverse them during the removal audit; some historical
                # COLMAP outputs carry restrictive inherited Windows ACLs.
                directory_names.remove(name)
    return files, errors


def file_size(path: Path) -> int:
    try:
        return path.lstat().st_size
    except OSError:
        return 0


def format_bytes(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def main() -> int:
    args = parse_args()
    project = PROJECT_ROOT.resolve()
    data = DATA_ROOT.resolve()
    data.relative_to(project)
    if data.name != "data":
        raise RuntimeError(f"Refusing unexpected data root: {data}")
    if not data.is_dir():
        raise RuntimeError(f"Data directory does not exist: {data}")

    keep_directories = tuple(resolve_inside_data(path) for path in KEEP_DIRECTORIES)
    keep_files = tuple(resolve_inside_data(path) for path in KEEP_FILES)
    separate_remove_directories = tuple(
        resolve_inside_data(path) for path in SEPARATE_REMOVE_DIRECTORIES
    )
    existing_separate_remove_directories = tuple(
        path for path in separate_remove_directories if path.exists()
    )
    missing = [path for path in (*keep_directories, *keep_files) if not path.exists()]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise RuntimeError(f"Required final artifact is missing:\n{formatted}")

    all_files, walk_errors = walk_data(
        keep_directories, existing_separate_remove_directories
    )
    if walk_errors:
        formatted = "\n".join(f"  {error}" for error in walk_errors)
        raise RuntimeError(f"Could not fully audit the data directory:\n{formatted}")

    retained: list[Path] = []
    removable: list[Path] = []
    for path in all_files:
        resolved = path.resolve()
        if resolved in keep_files or any(
            is_inside(resolved, directory) for directory in keep_directories
        ):
            retained.append(path)
        else:
            removable.append(path)

    retained_directory_files: list[Path] = []
    for directory in keep_directories:
        for root, _, names in os.walk(directory, onerror=lambda _: None):
            retained_directory_files.extend(Path(root) / name for name in names)
    retained.extend(retained_directory_files)
    retained_bytes = sum(file_size(path) for path in retained)
    removable_bytes = sum(file_size(path) for path in removable)
    print("Completed-project cleanup audit")
    print(f"Data root: {data}")
    print(f"Retained files: {len(retained)} ({format_bytes(retained_bytes)})")
    print(f"Files to remove: {len(removable)} ({format_bytes(removable_bytes)})")
    print(
        "Protected legacy directories to remove separately: "
        f"{len(existing_separate_remove_directories)}"
    )
    print(f"Mode: {'DELETE' if args.run else 'DRY RUN'}")

    if not args.run:
        print("No files were deleted.")
        return 0

    failures: list[str] = []
    deleted_bytes = 0
    for number, path in enumerate(removable, start=1):
        try:
            size = file_size(path)
            path.unlink()
            deleted_bytes += size
        except OSError as error:
            failures.append(f"{path}: {error}")
        if number == 1 or number % 1000 == 0 or number == len(removable):
            print(f"Deleted {number}/{len(removable)} file entries", flush=True)

    required_directories = set(keep_directories)
    required_files = set(keep_files)

    def directory_is_required(path: Path) -> bool:
        resolved = path.resolve()
        if any(is_inside(resolved, directory) for directory in keep_directories):
            return True
        if any(is_ancestor(resolved, directory) for directory in keep_directories):
            return True
        if any(is_ancestor(resolved, file) for file in keep_files):
            return True
        return False

    for root, directory_names, _ in os.walk(DATA_ROOT, topdown=False):
        root_path = Path(root)
        for name in directory_names:
            directory = root_path / name
            if directory_is_required(directory):
                continue
            try:
                directory.rmdir()
            except OSError:
                # Non-empty directories remain only when a deletion failed.
                pass

    final_missing = [
        path for path in (*required_directories, *required_files) if not path.exists()
    ]
    if final_missing:
        formatted = "\n".join(f"  {path}" for path in final_missing)
        raise RuntimeError(f"Final-artifact verification failed:\n{formatted}")
    if failures:
        print("Cleanup completed with deletion failures:")
        for failure in failures[:20]:
            print(f"  {failure}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1

    print(f"Cleanup complete. Removed {format_bytes(deleted_bytes)}.")
    print("All required final artifacts were verified after deletion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
