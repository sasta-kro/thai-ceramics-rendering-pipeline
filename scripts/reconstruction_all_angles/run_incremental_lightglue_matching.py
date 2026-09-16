#!/usr/bin/env python3
"""Run SIFT-LightGlue only between the side and new mid25 rings."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RECONSTRUCTION_SCRIPTS_DIR = PROJECT_ROOT / "scripts/reconstruction"
sys.path.insert(0, str(RECONSTRUCTION_SCRIPTS_DIR))

from cleanup_lightglue_database import (  # noqa: E402
    DatabaseCleanupError,
    GroupCounts,
    group_label,
    image_ids_from_pair_id,
    manifest_views,
    read_only_snapshot,
)
from run_lightglue_matching import (  # noqa: E402
    LightGlueRunnerError,
    bool_arg,
    child_environment,
    load_yaml,
    required_bool,
    required_float,
    required_int,
    required_string,
    required_string_list,
    resolve_cuda_dll_directory,
    resolve_executable,
    resolve_project_path,
    section,
    stream_process,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/colmap_pot1_unglazed_multiview_mid25_lightglue.yml"
)
PROTECTED_DATABASE = (
    PROJECT_ROOT
    / "data/processed/pot1-unglazed_multiview/"
    "colmap_sparse_masked_multiview_lightglue_repaired/database.db"
)


@dataclass(frozen=True)
class IncrementalLightGluePlan:
    executable: Path
    cuda_dll_directory: Path
    database: Path
    pair_list: Path
    manifest: Path
    log: Path
    report: Path
    expected_groups: Mapping[str, int]
    pair_count: int
    before: Any
    command: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match side-to-mid25 with SIFT-LightGlue."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_bridges(matching: Mapping[str, Any]) -> dict[str, int]:
    raw = matching.get("bridges")
    if not isinstance(raw, list) or not raw:
        raise LightGlueRunnerError("'matching.bridges' must be a non-empty list.")
    result: dict[str, int] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise LightGlueRunnerError(f"matching.bridges[{index}] must be a mapping.")
        left = required_string(item, "left_view", f"matching.bridges[{index}]")
        right = required_string(item, "right_view", f"matching.bridges[{index}]")
        if left == right:
            raise LightGlueRunnerError("A LightGlue bridge cannot connect a view to itself.")
        label = group_label(left, right)
        if label in result:
            raise LightGlueRunnerError(f"Duplicate LightGlue bridge group: {label}")
        result[label] = required_int(
            item, "expected_pair_count", f"matching.bridges[{index}]", minimum=1
        )
    return result


def validate_pairs(
    path: Path,
    filename_to_view: Mapping[str, str],
    expected_groups: Mapping[str, int],
) -> tuple[tuple[str, str], ...]:
    if not path.is_file():
        raise LightGlueRunnerError(f"LightGlue pair list was not found: {path}")
    pairs: list[tuple[str, str]] = []
    canonical: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if len(fields) != 2:
            raise LightGlueRunnerError(
                f"Pair-list line {line_number} must contain two filenames."
            )
        left, right = fields
        if left not in filename_to_view or right not in filename_to_view:
            raise LightGlueRunnerError(
                f"Pair-list line {line_number} refers to an unknown image."
            )
        label = group_label(filename_to_view[left], filename_to_view[right])
        if label not in expected_groups:
            raise LightGlueRunnerError(
                f"Pair-list line {line_number} has unexpected bridge group {label}."
            )
        key = tuple(sorted((left, right)))
        if key in canonical:
            raise LightGlueRunnerError("LightGlue pair list contains duplicates.")
        canonical.add(key)
        counts[label] += 1
        pairs.append((left, right))
    if dict(counts) != dict(expected_groups):
        raise LightGlueRunnerError(
            f"LightGlue bridge counts differ: expected={dict(expected_groups)}, "
            f"actual={dict(counts)}."
        )
    return tuple(pairs)


def next_output_paths(base_log: Path) -> tuple[Path, Path]:
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"_retry{attempt}"
        log = base_log.with_name(f"{base_log.stem}{suffix}{base_log.suffix}")
        report = base_log.with_name(f"{base_log.stem}{suffix}_report.json")
        if report.exists():
            raise LightGlueRunnerError(
                "Incremental LightGlue already has a completed report."
            )
        if not log.exists():
            return log, report
    raise LightGlueRunnerError("No unused LightGlue retry log remains.")


def build_plan(config: Mapping[str, Any]) -> IncrementalLightGluePlan:
    colmap = section(config, "colmap")
    runtime = section(config, "runtime")
    inputs = section(config, "input")
    output = section(config, "output")
    matching = section(config, "matching")
    executable = resolve_executable(required_string(colmap, "executable", "colmap"))
    required_dlls = required_string_list(runtime, "required_cuda_dlls", "runtime")
    cuda_directory = resolve_cuda_dll_directory(
        required_string(runtime, "cuda_dll_directory", "runtime"),
        required_dlls,
        required_string(runtime, "minimum_cuda_version", "runtime"),
        required_int(runtime, "minimum_cudnn_version", "runtime", minimum=1),
    )
    database = resolve_project_path(
        required_string(inputs, "database", "input"), "Input database"
    )
    pair_list = resolve_project_path(
        required_string(inputs, "match_pairs", "input"), "Input pair list"
    )
    manifest = resolve_project_path(
        required_string(inputs, "manifest", "input"), "Input manifest"
    )
    base_log = resolve_project_path(
        required_string(output, "log", "output"), "Output log"
    )
    if database.resolve() == PROTECTED_DATABASE.resolve():
        raise LightGlueRunnerError("Refusing to modify the accepted 728-view database.")
    if not database.is_file():
        raise LightGlueRunnerError(f"Incremental database was not found: {database}")
    log, report = next_output_paths(base_log)
    expected_groups = parse_bridges(matching)
    filename_to_view = manifest_views(manifest)
    pairs = validate_pairs(pair_list, filename_to_view, expected_groups)
    before = read_only_snapshot(database, filename_to_view)
    if before.images != 949 or before.keypoint_images != 949 or before.descriptor_images != 949:
        raise LightGlueRunnerError(
            "Expected complete SIFT features for all 949 database images."
        )
    for table in ("matches", "two_view_geometries"):
        for label in expected_groups:
            existing = before.match_groups[table].get(label, GroupCounts())
            if existing.records:
                raise LightGlueRunnerError(
                    f"Database already contains {existing.records} {label} records in {table}."
                )
    feature_type = required_string(matching, "feature_type", "matching").upper()
    if feature_type != "SIFT_LIGHTGLUE":
        raise LightGlueRunnerError("Feature type must be SIFT_LIGHTGLUE.")
    command = (
        str(executable),
        "matches_importer",
        "--database_path",
        str(database),
        "--match_list_path",
        str(pair_list),
        "--match_type",
        "pairs",
        "--FeatureMatching.type",
        feature_type,
        "--FeatureMatching.use_gpu",
        bool_arg(required_bool(matching, "use_gpu", "matching")),
        "--FeatureMatching.gpu_index",
        str(required_int(matching, "gpu_index", "matching", minimum=0)),
        "--FeatureMatching.num_threads",
        str(required_int(matching, "num_threads", "matching", minimum=1)),
        "--FeatureMatching.max_num_matches",
        str(required_int(matching, "max_num_matches", "matching", minimum=1)),
        "--FeatureMatching.guided_matching",
        bool_arg(required_bool(matching, "guided_matching", "matching")),
        "--SiftMatching.lightglue_min_score",
        str(
            required_float(
                matching,
                "lightglue_min_score",
                "matching",
                minimum=0.0,
                maximum=1.0,
            )
        ),
        "--TwoViewGeometry.min_num_inliers",
        str(required_int(matching, "min_num_inliers", "matching", minimum=1)),
        "--log_target",
        "stderr",
    )
    return IncrementalLightGluePlan(
        executable,
        cuda_directory,
        database,
        pair_list,
        manifest,
        log,
        report,
        expected_groups,
        len(pairs),
        before,
        command,
    )


def target_rows(
    database: Path, filename_to_view: Mapping[str, str], labels: set[str]
) -> dict[str, list[int]]:
    result = {label: [] for label in labels}
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        names = dict(connection.execute("SELECT image_id, name FROM images"))
        for record_id, rows in connection.execute(
            "SELECT pair_id, rows FROM two_view_geometries"
        ):
            first, second = image_ids_from_pair_id(int(record_id))
            label = group_label(
                filename_to_view[names[first]], filename_to_view[names[second]]
            )
            if label in result:
                result[label].append(int(rows))
    finally:
        connection.close()
    return result


def validate_result(plan: IncrementalLightGluePlan) -> dict[str, object]:
    filename_to_view = manifest_views(plan.manifest)
    after = read_only_snapshot(plan.database, filename_to_view)
    for field in (
        "images",
        "keypoint_images",
        "keypoint_rows",
        "descriptor_images",
        "descriptor_rows",
    ):
        if getattr(after, field) != getattr(plan.before, field):
            raise LightGlueRunnerError(f"LightGlue changed database field '{field}'.")
    labels = set(plan.expected_groups)
    for table in ("matches", "two_view_geometries"):
        preserved_before = {
            key: value
            for key, value in plan.before.match_groups[table].items()
            if key not in labels
        }
        preserved_after = {
            key: value
            for key, value in after.match_groups[table].items()
            if key not in labels
        }
        if preserved_before != preserved_after:
            raise LightGlueRunnerError(f"LightGlue changed preserved groups in {table}.")
    rows_by_group = target_rows(plan.database, filename_to_view, labels)
    groups: dict[str, object] = {}
    for label, expected in plan.expected_groups.items():
        rows = rows_by_group[label]
        if len(rows) != expected:
            raise LightGlueRunnerError(
                f"Expected {expected} geometry records for {label}, found {len(rows)}."
            )
        verified = [value for value in rows if value >= 15]
        if not verified:
            raise LightGlueRunnerError(f"No {label} pair reached 15 geometric inliers.")
        groups[label] = {
            "planned_pairs": expected,
            "verified_pairs_15plus": len(verified),
            "verified_pairs_30plus": sum(value >= 30 for value in rows),
            "inliers_median": float(statistics.median(verified)),
            "inliers_mean": round(statistics.mean(verified), 3),
            "inliers_max": max(verified),
        }
    return {
        "images_preserved": after.images,
        "keypoints_preserved": after.keypoint_rows,
        "descriptors_preserved": after.descriptor_rows,
        "planned_bridge_pairs": plan.pair_count,
        "groups": groups,
        "preserved_match_groups": {
            table: {
                label: asdict(counts)
                for label, counts in after.match_groups[table].items()
                if label not in labels
            }
            for table in ("matches", "two_view_geometries")
        },
    }


def run_matching(plan: IncrementalLightGluePlan) -> None:
    plan.log.parent.mkdir(parents=True, exist_ok=True)
    with plan.log.open("x", encoding="utf-8", newline="\n") as handle:
        exit_code = stream_process(
            plan.command, handle, child_environment(plan.cuda_dll_directory)
        )
    if exit_code != 0:
        raise LightGlueRunnerError(
            f"COLMAP SIFT-LightGlue failed with exit code {exit_code}. See {plan.log}"
        )


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_project_path(args.config, "Configuration")
        plan = build_plan(load_yaml(config_path))
        print("Incremental cross-angle SIFT-LightGlue matching")
        print(f"CUDA DLL directory: {plan.cuda_dll_directory}")
        print(f"Database: {plan.database}")
        for label, count in plan.expected_groups.items():
            print(f"Bridge pairs {label}: {count}")
        print(f"Total bridge pairs: {plan.pair_count}")
        print(f"Log: {plan.log}")
        print(f"Command: {subprocess.list2cmdline(plan.command)}")
        if args.dry_run:
            print("Dry run complete. Database and logs were not modified.")
            return 0
        run_matching(plan)
        report = validate_result(plan)
        plan.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Incremental SIFT-LightGlue matching complete")
        for label, values in report["groups"].items():
            print(f"{label}: {values}")
        print(f"Report: {plan.report}")
        return 0
    except (LightGlueRunnerError, DatabaseCleanupError, OSError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
