#!/usr/bin/env python3
"""Run targeted SIFT-LightGlue matching from a validated YAML config."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence, TextIO

from cleanup_lightglue_database import (
    DatabaseCleanupError,
    GroupCounts,
    group_label,
    manifest_views,
    read_only_snapshot,
)
from prepare_lightglue_workspace import WorkspacePreparationError


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "colmap_pot1_unglazed_multiview_lightglue.yml"
)
ORIGINAL_DATABASE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pot1-unglazed_multiview"
    / "colmap_sparse_masked_multiview"
    / "database.db"
)


class LightGlueRunnerError(RuntimeError):
    """A user-correctable LightGlue configuration or execution error."""


@dataclass(frozen=True)
class MatchingPlan:
    executable: Path
    cuda_dll_directory: Path
    database: Path
    pair_list: Path
    manifest: Path
    log: Path
    left_view: str
    right_view: str
    expected_pair_count: int
    pair_count: int
    command: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run targeted SIFT-LightGlue matching for the multiview pot."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "YAML configuration file "
            f"(default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the COLMAP command without running it.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise LightGlueRunnerError(
            "PyYAML is required. Activate the project environment before running."
        ) from error
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except FileNotFoundError as error:
        raise LightGlueRunnerError(f"Configuration file not found: {path}") from error
    except yaml.YAMLError as error:
        raise LightGlueRunnerError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(config, Mapping):
        raise LightGlueRunnerError(f"YAML root must be a mapping: {path}")
    return config


def section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise LightGlueRunnerError(f"Missing or invalid YAML section '{name}'.")
    return value


def required_string(values: Mapping[str, Any], key: str, section_name: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LightGlueRunnerError(
            f"'{section_name}.{key}' must be a non-empty string."
        )
    return value.strip()


def required_bool(values: Mapping[str, Any], key: str, section_name: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise LightGlueRunnerError(f"'{section_name}.{key}' must be true or false.")
    return value


def required_int(
    values: Mapping[str, Any],
    key: str,
    section_name: str,
    *,
    minimum: int,
) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LightGlueRunnerError(
            f"'{section_name}.{key}' must be an integer of at least {minimum}."
        )
    return value


def required_float(
    values: Mapping[str, Any],
    key: str,
    section_name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LightGlueRunnerError(f"'{section_name}.{key}' must be numeric.")
    result = float(value)
    if not minimum <= result <= maximum:
        raise LightGlueRunnerError(
            f"'{section_name}.{key}' must be between {minimum} and {maximum}."
        )
    return result


def required_string_list(
    values: Mapping[str, Any], key: str, section_name: str
) -> tuple[str, ...]:
    value = values.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise LightGlueRunnerError(
            f"'{section_name}.{key}' must be a non-empty list of strings."
        )
    return tuple(item.strip() for item in value)


def resolve_project_path(raw_path: str | Path, label: str) -> Path:
    candidate = Path(os.path.expandvars(str(raw_path))).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise LightGlueRunnerError(
            f"{label} must stay inside the project directory: {candidate}"
        ) from error
    return candidate


def resolve_executable(raw_path: str) -> Path:
    candidate = Path(os.path.expandvars(raw_path)).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise LightGlueRunnerError(f"COLMAP executable was not found: {candidate}")
    return candidate


def resolve_cuda_dll_directory(
    raw_path: str,
    required_dlls: Sequence[str],
    minimum_cuda_version: str,
    minimum_cudnn_version: int,
) -> Path:
    if raw_path.strip().lower() == "auto_pytorch":
        try:
            import torch
        except ModuleNotFoundError as error:
            raise LightGlueRunnerError(
                "PyTorch is required to resolve the automatic CUDA DLL directory."
            ) from error
        detected_cuda = torch.version.cuda
        detected_cudnn = torch.backends.cudnn.version()
        if not isinstance(detected_cuda, str):
            raise LightGlueRunnerError(
                "The active PyTorch installation has no CUDA runtime."
            )
        try:
            detected_cuda_parts = tuple(
                int(part) for part in detected_cuda.split(".")[:2]
            )
            minimum_cuda_parts = tuple(
                int(part) for part in minimum_cuda_version.split(".")[:2]
            )
        except ValueError as error:
            raise LightGlueRunnerError(
                "CUDA versions must use a numeric major.minor format."
            ) from error
        if detected_cuda_parts < minimum_cuda_parts:
            raise LightGlueRunnerError(
                f"Active PyTorch CUDA {detected_cuda} is older than the "
                f"COLMAP ONNX requirement CUDA {minimum_cuda_version}. "
                "Use a compatible external CUDA runtime or CPU matching."
            )
        if (
            not isinstance(detected_cudnn, int)
            or detected_cudnn < minimum_cudnn_version
        ):
            raise LightGlueRunnerError(
                f"Active PyTorch cuDNN {detected_cudnn} is older than the "
                f"required version code {minimum_cudnn_version}."
            )
        candidate = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    else:
        candidate = Path(os.path.expandvars(raw_path)).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise LightGlueRunnerError(
            "CUDA DLL directory was not found. Activate pot-3dgs before running: "
            f"{candidate}"
        )
    invalid_names = [name for name in required_dlls if Path(name).name != name]
    if invalid_names:
        raise LightGlueRunnerError(
            "Runtime DLL names must not include directory components: "
            + ", ".join(invalid_names)
        )
    missing = [name for name in required_dlls if not (candidate / name).is_file()]
    if missing:
        raise LightGlueRunnerError(
            f"CUDA runtime directory {candidate} is missing DLL(s): "
            + ", ".join(missing)
        )
    return candidate


def bool_arg(value: bool) -> str:
    return "1" if value else "0"


def load_and_validate_pairs(
    path: Path,
    filename_to_view: Mapping[str, str],
    left_view: str,
    right_view: str,
) -> tuple[tuple[str, str], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise LightGlueRunnerError(
            f"LightGlue pair list was not found: {path}"
        ) from error
    except OSError as error:
        raise LightGlueRunnerError(
            f"Could not read pair list {path}: {error}"
        ) from error

    pairs: list[tuple[str, str]] = []
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 2:
            raise LightGlueRunnerError(
                f"Pair-list line {line_number} must contain exactly two filenames."
            )
        left, right = fields
        if left not in filename_to_view or right not in filename_to_view:
            raise LightGlueRunnerError(
                f"Pair-list line {line_number} refers to an unknown image."
            )
        actual_group = group_label(
            filename_to_view[left], filename_to_view[right]
        )
        expected_group = group_label(left_view, right_view)
        if actual_group != expected_group:
            raise LightGlueRunnerError(
                f"Pair-list line {line_number} is {actual_group}, not {expected_group}."
            )
        pairs.append((left, right))
    if not pairs:
        raise LightGlueRunnerError("LightGlue pair list is empty.")
    if len(set(pairs)) != len(pairs):
        raise LightGlueRunnerError("LightGlue pair list contains duplicates.")
    return tuple(pairs)


def build_plan(config: Mapping[str, Any]) -> MatchingPlan:
    colmap = section(config, "colmap")
    runtime = section(config, "runtime")
    inputs = section(config, "input")
    output = section(config, "output")
    matching = section(config, "matching")

    executable = resolve_executable(required_string(colmap, "executable", "colmap"))
    required_cuda_dlls = required_string_list(
        runtime, "required_cuda_dlls", "runtime"
    )
    minimum_cuda_version = required_string(
        runtime, "minimum_cuda_version", "runtime"
    )
    minimum_cudnn_version = required_int(
        runtime, "minimum_cudnn_version", "runtime", minimum=1
    )
    cuda_dll_directory = resolve_cuda_dll_directory(
        required_string(runtime, "cuda_dll_directory", "runtime"),
        required_cuda_dlls,
        minimum_cuda_version,
        minimum_cudnn_version,
    )
    database = resolve_project_path(
        required_string(inputs, "database", "input"), "Input database"
    )
    pair_list = resolve_project_path(
        required_string(inputs, "match_pairs", "input"), "Input match-pair list"
    )
    manifest = resolve_project_path(
        required_string(inputs, "manifest", "input"), "Input manifest"
    )
    log = resolve_project_path(required_string(output, "log", "output"), "Output log")
    if database == ORIGINAL_DATABASE.resolve():
        raise LightGlueRunnerError(
            "Refusing to run LightGlue against the original multiview database."
        )
    if not database.is_file():
        raise LightGlueRunnerError(f"Copied COLMAP database was not found: {database}")
    if log.exists():
        raise LightGlueRunnerError(
            f"LightGlue log already exists; refusing to overwrite it: {log}"
        )

    left_view = (
        required_string(matching, "left_view", "matching")
        if "left_view" in matching
        else "side"
    )
    right_view = (
        required_string(matching, "right_view", "matching")
        if "right_view" in matching
        else "top45"
    )
    if left_view == right_view:
        raise LightGlueRunnerError("Matching bridge views must be different.")
    filename_to_view = manifest_views(manifest)
    pairs = load_and_validate_pairs(
        pair_list, filename_to_view, left_view, right_view
    )
    expected_pair_count = required_int(
        matching,
        "expected_pair_count",
        "matching",
        minimum=1,
    )
    if len(pairs) != expected_pair_count:
        raise LightGlueRunnerError(
            f"Expected {expected_pair_count} pairs but found {len(pairs)} "
            f"in {pair_list}."
        )

    snapshot = read_only_snapshot(database, filename_to_view)
    target_group = group_label(left_view, right_view)
    for table in ("matches", "two_view_geometries"):
        remaining = snapshot.match_groups[table].get(target_group, GroupCounts())
        if remaining.records:
            raise LightGlueRunnerError(
                f"Copied database still contains {remaining.records} {target_group} "
                f"records in {table}; run cleanup before matching."
            )

    feature_type = required_string(matching, "feature_type", "matching").upper()
    if feature_type != "SIFT_LIGHTGLUE":
        raise LightGlueRunnerError(
            "'matching.feature_type' must be SIFT_LIGHTGLUE for this stage."
        )
    use_gpu = required_bool(matching, "use_gpu", "matching")
    gpu_index = required_int(matching, "gpu_index", "matching", minimum=0)
    num_threads = required_int(matching, "num_threads", "matching", minimum=1)
    max_num_matches = required_int(
        matching, "max_num_matches", "matching", minimum=1
    )
    guided_matching = required_bool(matching, "guided_matching", "matching")
    lightglue_min_score = required_float(
        matching,
        "lightglue_min_score",
        "matching",
        minimum=0.0,
        maximum=1.0,
    )
    min_num_inliers = required_int(
        matching, "min_num_inliers", "matching", minimum=1
    )

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
        bool_arg(use_gpu),
        "--FeatureMatching.gpu_index",
        str(gpu_index),
        "--FeatureMatching.num_threads",
        str(num_threads),
        "--FeatureMatching.max_num_matches",
        str(max_num_matches),
        "--FeatureMatching.guided_matching",
        bool_arg(guided_matching),
        "--SiftMatching.lightglue_min_score",
        str(lightglue_min_score),
        "--TwoViewGeometry.min_num_inliers",
        str(min_num_inliers),
        "--log_target",
        "stderr",
    )
    return MatchingPlan(
        executable=executable,
        cuda_dll_directory=cuda_dll_directory,
        database=database,
        pair_list=pair_list,
        manifest=manifest,
        log=log,
        left_view=left_view,
        right_view=right_view,
        expected_pair_count=expected_pair_count,
        pair_count=len(pairs),
        command=command,
    )


def display_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def child_environment(cuda_dll_directory: Path) -> dict[str, str]:
    environment = os.environ.copy()
    existing_path = environment.get("PATH", "")
    path_entries = [str(cuda_dll_directory)]
    if existing_path:
        path_entries.append(existing_path)
    environment["PATH"] = os.pathsep.join(path_entries)
    return environment


def stream_process(
    command: Sequence[str],
    log_handle: TextIO,
    environment: Mapping[str, str],
) -> int:
    process = subprocess.Popen(
        list(command),
        cwd=PROJECT_ROOT,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_handle.write(line)
        log_handle.flush()
    return process.wait()


def run_matching(plan: MatchingPlan) -> None:
    plan.log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with plan.log.open("x", encoding="utf-8", newline="\n") as log_handle:
            exit_code = stream_process(
                plan.command,
                log_handle,
                child_environment(plan.cuda_dll_directory),
            )
    except FileExistsError as error:
        raise LightGlueRunnerError(
            f"LightGlue log already exists; refusing to overwrite it: {plan.log}"
        ) from error
    except OSError as error:
        raise LightGlueRunnerError(f"Could not run COLMAP: {error}") from error
    if exit_code != 0:
        raise LightGlueRunnerError(
            f"COLMAP SIFT-LightGlue matching failed with exit code {exit_code}. "
            f"See {plan.log}"
        )


def print_plan(plan: MatchingPlan, dry_run: bool) -> None:
    print("SIFT-LightGlue targeted matching")
    print(f"CUDA DLL directory: {plan.cuda_dll_directory}")
    print(f"Database: {plan.database}")
    print(f"Pair list: {plan.pair_list}")
    print(f"Validated targeted pairs: {plan.pair_count}")
    print(f"Log: {plan.log}")
    print(f"Command: {display_command(plan.command)}")
    if dry_run:
        print("Dry run only: COLMAP was not started.")


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_project_path(args.config, "Configuration")
        plan = build_plan(load_yaml(config_path))
        print_plan(plan, args.dry_run)
        if not args.dry_run:
            run_matching(plan)
            print("SIFT-LightGlue matching completed successfully.")
            print("Do not register cameras until bridge quality has been checked.")
        return 0
    except (
        LightGlueRunnerError,
        DatabaseCleanupError,
        WorkspacePreparationError,
        OSError,
    ) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
