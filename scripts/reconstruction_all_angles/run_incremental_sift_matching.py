#!/usr/bin/env python3
"""Match only the sequential pairs from the newly added capture ring."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import statistics
import subprocess
import sys
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_COLMAP = Path("C:/Tools/COLMAP-4.1.1/bin/colmap.exe")
DEFAULT_WORKSPACE = (
    PROJECT_ROOT
    / "data/processed/pot1-unglazed_side_underside_mid25/colmap_side_underside_mid25"
)
MAX_IMAGE_ID = 2**31 - 1


class IncrementalMatchingError(RuntimeError):
    """A user-correctable incremental SIFT-matching error."""


@dataclass(frozen=True)
class MatchSnapshot:
    images: dict[str, int]
    keypoint_rows: dict[str, int]
    descriptor_rows: dict[str, int]
    matches: dict[int, int]
    geometries: dict[int, int]


@dataclass(frozen=True)
class MatchingPlan:
    colmap: Path
    workspace: Path
    database: Path
    pair_list: Path
    log: Path
    report: Path
    pairs: tuple[tuple[str, str], ...]
    pair_ids: frozenset[int]
    before: MatchSnapshot
    command: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ordinary SIFT matching only within the new view ring."
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
        raise IncrementalMatchingError(
            f"Workspace must stay inside the project: {candidate}"
        ) from error
    return candidate


def pair_id(image_id1: int, image_id2: int) -> int:
    smaller, larger = sorted((image_id1, image_id2))
    return smaller * MAX_IMAGE_ID + larger


def snapshot(path: Path) -> MatchSnapshot:
    if not path.is_file():
        raise IncrementalMatchingError(f"Database was not found: {path}")
    try:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
        try:
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            if integrity != ("ok",):
                raise IncrementalMatchingError(
                    f"Database quick_check failed: {integrity}"
                )
            images = {
                str(name): int(image_id)
                for image_id, name in connection.execute(
                    "SELECT image_id, name FROM images"
                )
            }
            keypoints = {
                str(name): int(rows)
                for name, rows in connection.execute(
                    "SELECT images.name, keypoints.rows FROM images "
                    "JOIN keypoints ON keypoints.image_id = images.image_id"
                )
            }
            descriptors = {
                str(name): int(rows)
                for name, rows in connection.execute(
                    "SELECT images.name, descriptors.rows FROM images "
                    "JOIN descriptors ON descriptors.image_id = images.image_id"
                )
            }
            matches = {
                int(record_id): int(rows)
                for record_id, rows in connection.execute(
                    "SELECT pair_id, rows FROM matches"
                )
            }
            geometries = {
                int(record_id): int(rows)
                for record_id, rows in connection.execute(
                    "SELECT pair_id, rows FROM two_view_geometries"
                )
            }
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise IncrementalMatchingError(f"Could not inspect database: {error}") from error
    return MatchSnapshot(images, keypoints, descriptors, matches, geometries)


def read_pairs(
    path: Path, image_ids: dict[str, int]
) -> tuple[tuple[tuple[str, str], ...], frozenset[int]]:
    if not path.is_file():
        raise IncrementalMatchingError(f"Sequential pair list was not found: {path}")
    pairs: list[tuple[str, str]] = []
    ids: set[int] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if len(fields) != 2:
            raise IncrementalMatchingError(
                f"Pair-list line {line_number} must contain two filenames."
            )
        left, right = fields
        if left not in image_ids or right not in image_ids:
            raise IncrementalMatchingError(
                f"Pair-list line {line_number} refers to an unknown database image."
            )
        if not left.startswith("mid25_") or not right.startswith("mid25_"):
            raise IncrementalMatchingError(
                f"Pair-list line {line_number} is not a mid25-to-mid25 pair."
            )
        record_id = pair_id(image_ids[left], image_ids[right])
        if record_id in ids:
            raise IncrementalMatchingError("Sequential pair list contains duplicates.")
        ids.add(record_id)
        pairs.append((left, right))
    if not pairs:
        raise IncrementalMatchingError("Sequential pair list is empty.")
    return tuple(pairs), frozenset(ids)


def next_output_paths(workspace: Path) -> tuple[Path, Path]:
    logs = workspace / "logs"
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"_retry{attempt}"
        log = logs / f"incremental_sift_matching{suffix}.log"
        report = logs / f"incremental_sift_matching{suffix}_report.json"
        if report.exists():
            raise IncrementalMatchingError(
                "Sequential matching already has a completed report; "
                "refusing to repeat this stage."
            )
        if not log.exists():
            return log, report
    raise IncrementalMatchingError("No unused sequential-matching retry log remains.")


def build_plan(*, colmap: Path, workspace: Path) -> MatchingPlan:
    colmap = colmap.expanduser().resolve()
    workspace = resolve_workspace(workspace)
    if not colmap.is_file():
        raise IncrementalMatchingError(f"COLMAP executable was not found: {colmap}")
    if not workspace.is_dir():
        raise IncrementalMatchingError(f"Workspace was not found: {workspace}")
    database = workspace / "database.db"
    pair_list = workspace / "new_view_sequential_pairs.txt"
    log, report = next_output_paths(workspace)
    before = snapshot(database)
    if len(before.images) != 949:
        raise IncrementalMatchingError(
            f"Expected 949 database images, found {len(before.images)}."
        )
    if set(before.images) != set(before.keypoint_rows) or set(before.images) != set(
        before.descriptor_rows
    ):
        raise IncrementalMatchingError("Database SIFT features are incomplete.")
    pairs, pair_ids = read_pairs(pair_list, before.images)
    if len(pairs) != 3420:
        raise IncrementalMatchingError(
            f"Expected 3420 sequential pairs, found {len(pairs)}."
        )
    stale_matches = pair_ids & set(before.matches)
    stale_geometries = pair_ids & set(before.geometries)
    if stale_matches or stale_geometries:
        raise IncrementalMatchingError(
            "Database already contains new-ring match records: "
            f"matches={len(stale_matches)}, geometries={len(stale_geometries)}."
        )
    command = (
        str(colmap),
        "matches_importer",
        "--database_path",
        str(database),
        "--match_list_path",
        str(pair_list),
        "--match_type",
        "pairs",
        "--FeatureMatching.type",
        "SIFT_BRUTEFORCE",
        "--FeatureMatching.use_gpu",
        "1",
        "--FeatureMatching.gpu_index",
        "0",
        "--FeatureMatching.num_threads",
        "1",
        "--FeatureMatching.max_num_matches",
        "4096",
        "--FeatureMatching.guided_matching",
        "1",
        "--TwoViewGeometry.min_num_inliers",
        "15",
        "--log_target",
        "stderr",
    )
    return MatchingPlan(
        colmap, workspace, database, pair_list, log, report, pairs, pair_ids, before, command
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
        raise IncrementalMatchingError(
            f"COLMAP sequential matching failed with exit code {return_code}."
        )


def validate_result(plan: MatchingPlan) -> dict[str, object]:
    after = snapshot(plan.database)
    if after.images != plan.before.images:
        raise IncrementalMatchingError("Database image records changed during matching.")
    if (
        after.keypoint_rows != plan.before.keypoint_rows
        or after.descriptor_rows != plan.before.descriptor_rows
    ):
        raise IncrementalMatchingError("SIFT feature records changed during matching.")
    if any(after.matches.get(key) != value for key, value in plan.before.matches.items()):
        raise IncrementalMatchingError("An existing raw-match record changed.")
    if any(
        after.geometries.get(key) != value
        for key, value in plan.before.geometries.items()
    ):
        raise IncrementalMatchingError("An existing geometry record changed.")
    missing_matches = plan.pair_ids - set(after.matches)
    missing_geometries = plan.pair_ids - set(after.geometries)
    if missing_matches or missing_geometries:
        raise IncrementalMatchingError(
            "Sequential matching did not record every planned pair: "
            f"matches missing={len(missing_matches)}, "
            f"geometries missing={len(missing_geometries)}."
        )
    raw_rows = [after.matches[record_id] for record_id in plan.pair_ids]
    geometry_rows = [after.geometries[record_id] for record_id in plan.pair_ids]
    verified = [rows for rows in geometry_rows if rows >= 15]
    if not verified:
        raise IncrementalMatchingError("No new-ring pair reached 15 geometric inliers.")
    return {
        "planned_pairs": len(plan.pairs),
        "raw_pairs_with_matches": sum(rows > 0 for rows in raw_rows),
        "verified_pairs_15plus": len(verified),
        "verified_pair_percent": round(100.0 * len(verified) / len(plan.pairs), 3),
        "verified_inliers_median": float(statistics.median(verified)),
        "verified_inliers_mean": round(statistics.mean(verified), 3),
        "verified_inliers_max": max(verified),
        "existing_match_records_preserved": len(plan.before.matches),
        "existing_geometry_records_preserved": len(plan.before.geometries),
    }


def main() -> int:
    args = parse_args()
    try:
        plan = build_plan(colmap=args.colmap, workspace=args.workspace)
        print("Incremental within-ring SIFT matching")
        print(f"Database: {plan.database}")
        print(f"Validated pairs: {len(plan.pairs)}")
        print(f"Log: {plan.log}")
        print(f"Command: {subprocess.list2cmdline(plan.command)}")
        if args.dry_run:
            print("Dry run complete. Database and logs were not modified.")
            return 0
        run_command(plan.command, plan.log)
        report = validate_result(plan)
        plan.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Incremental within-ring SIFT matching complete")
        for key, value in report.items():
            print(f"{key}: {value}")
        print(f"Log: {plan.log}")
        print(f"Report: {plan.report}")
        return 0
    except IncrementalMatchingError as error:
        print(f"ERROR: {error}")
        return 1
    except OSError as error:
        print(f"ERROR: Could not run incremental SIFT matching: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
