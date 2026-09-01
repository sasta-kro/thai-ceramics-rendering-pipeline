#!/usr/bin/env python3
"""Run a validated, masked COLMAP sparse reconstruction from YAML config."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "colmap_pot1_unglazed_every6.yml"
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
FEATURE_MEMORY_PROFILES = {
    "quality_gpu": {
        "max_image_size": -1,
        "first_octave": -1,
        "num_threads": 1,
    },
    "balanced_gpu": {
        "max_image_size": 3200,
        "first_octave": 0,
        "num_threads": 1,
    },
    "low_memory_gpu": {
        "max_image_size": 2400,
        "first_octave": 0,
        "num_threads": 1,
    },
}


class PipelineError(RuntimeError):
    """A user-correctable configuration or pipeline error."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run masked COLMAP feature extraction, matching, and sparse mapping."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML configuration file (default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without creating outputs or running COLMAP.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise PipelineError(
            "PyYAML is required. Update the project environment with: "
            "micromamba env update -n pot-masking -f environment-masking.yml"
        ) from error

    try:
        with path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file)
    except FileNotFoundError as error:
        raise PipelineError(f"Configuration file not found: {path}") from error
    except yaml.YAMLError as error:
        raise PipelineError(f"Invalid YAML in {path}: {error}") from error

    if not isinstance(data, Mapping):
        raise PipelineError(f"YAML root must be a mapping: {path}")
    return data


def section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise PipelineError(f"Missing or invalid '{name}' section in the YAML config.")
    return value


def required_string(values: Mapping[str, Any], key: str, section_name: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(
            f"'{section_name}.{key}' must be a non-empty string in the YAML config."
        )
    return value.strip()


def optional_bool(
    values: Mapping[str, Any], key: str, section_name: str, default: bool
) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise PipelineError(f"'{section_name}.{key}' must be true or false.")
    return value


def optional_positive_int(
    values: Mapping[str, Any], key: str, section_name: str, default: int
) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PipelineError(f"'{section_name}.{key}' must be a positive integer.")
    return value


def feature_memory_profile(values: Mapping[str, Any]) -> tuple[str, Mapping[str, int]]:
    profile = values.get("memory_profile", "balanced_gpu")
    if not isinstance(profile, str):
        raise PipelineError("'feature_extraction.memory_profile' must be a string.")
    profile = profile.strip().lower()
    settings = FEATURE_MEMORY_PROFILES.get(profile)
    if settings is None:
        choices = ", ".join(sorted(FEATURE_MEMORY_PROFILES))
        raise PipelineError(
            f"Unknown feature extraction memory profile '{profile}'. Choose: {choices}."
        )
    return profile, settings


def resolve_project_path(raw_path: str, label: str) -> Path:
    candidate = Path(os.path.expandvars(raw_path)).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()

    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise PipelineError(
            f"'{label}' must stay inside the project directory. Resolved path: {candidate}"
        ) from error
    return candidate


def resolve_output_child(workspace: Path, raw_path: str, label: str) -> Path:
    configured = Path(raw_path)
    if configured.is_absolute():
        raise PipelineError(f"'{label}' must be relative to output.workspace.")
    candidate = (workspace / configured).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as error:
        raise PipelineError(f"'{label}' must stay inside output.workspace.") from error
    return candidate


def resolve_colmap_path(raw_path: str) -> Path:
    candidate = Path(os.path.expandvars(raw_path)).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def image_files(image_root: Path) -> list[Path]:
    return sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_masks(images: Iterable[Path], image_root: Path, mask_root: Path) -> None:
    missing: list[tuple[Path, Path]] = []
    for image_path in images:
        relative_image = image_path.relative_to(image_root)
        expected_mask = mask_root / Path(f"{relative_image}.png")
        if not expected_mask.is_file():
            missing.append((relative_image, expected_mask.relative_to(mask_root)))

    if missing:
        examples = "\n".join(f"  {image} -> {mask}" for image, mask in missing[:10])
        remainder = "" if len(missing) <= 10 else f"\n  ...and {len(missing) - 10} more"
        raise PipelineError(
            f"Missing COLMAP masks for {len(missing)} image(s). Expected naming is "
            f"'<image filename>.<extension>.png':\n{examples}{remainder}"
        )


def bool_arg(value: bool) -> str:
    return "1" if value else "0"


def display_command(executable: Path, arguments: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(executable), *arguments])


def execution_command(executable: Path, arguments: Sequence[str]) -> list[str]:
    return [str(executable), *arguments]


def run_colmap(
    executable: Path,
    arguments: Sequence[str],
    stage: str,
    dry_run: bool = False,
) -> None:
    print(f"\n[{stage}]\n{display_command(executable, arguments)}", flush=True)
    if dry_run:
        return

    try:
        completed = subprocess.run(
            execution_command(executable, arguments),
            cwd=PROJECT_ROOT,
            check=False,
        )
    except OSError as error:
        raise PipelineError(f"Could not launch COLMAP during '{stage}': {error}") from error

    if completed.returncode != 0:
        unsigned_code = completed.returncode & 0xFFFFFFFF
        detail = ""
        if unsigned_code == 0xC0000005:
            detail = (
                " Windows reported access violation 0xC0000005. During feature "
                "extraction this can follow excessive CPU/covariant-SIFT memory use; "
                "use balanced_gpu or low_memory_gpu with affine shape and domain-size "
                "pooling disabled."
            )
        raise PipelineError(
            f"COLMAP stage '{stage}' failed with exit code {completed.returncode}."
            f"{detail}"
        )


def ensure_clean_output(database_path: Path, sparse_path: Path) -> None:
    if database_path.exists():
        raise PipelineError(
            f"COLMAP database already exists: {database_path}\n"
            "Move the existing workspace before starting a fresh run."
        )
    if sparse_path.exists() and any(sparse_path.iterdir()):
        raise PipelineError(
            f"Sparse output directory is not empty: {sparse_path}\n"
            "Move the existing workspace before starting a fresh run."
        )


def find_sparse_models(sparse_path: Path) -> list[Path]:
    models = []
    for path in sparse_path.iterdir():
        if not path.is_dir():
            continue
        if (path / "cameras.bin").is_file() or (path / "cameras.txt").is_file():
            models.append(path)
    return sorted(models, key=lambda path: (not path.name.isdigit(), path.name))


def main() -> int:
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    config = load_yaml(config_path)
    project_config = section(config, "project")
    colmap_config = section(config, "colmap")
    input_config = section(config, "input")
    output_config = section(config, "output")
    extraction_config = section(config, "feature_extraction")
    matching_config = section(config, "matching")
    postprocess_config = section(config, "postprocess")

    project_name = required_string(project_config, "name", "project")
    colmap_path = resolve_colmap_path(
        required_string(colmap_config, "executable", "colmap")
    )
    use_gpu = optional_bool(colmap_config, "use_gpu", "colmap", True)
    camera_model = required_string(colmap_config, "camera_model", "colmap")
    single_camera = optional_bool(colmap_config, "single_camera", "colmap", True)

    images_path = resolve_project_path(
        required_string(input_config, "images", "input"), "input.images"
    )
    masks_path = resolve_project_path(
        required_string(input_config, "masks", "input"), "input.masks"
    )
    workspace_path = resolve_project_path(
        required_string(output_config, "workspace", "output"), "output.workspace"
    )
    database_path = resolve_output_child(
        workspace_path,
        required_string(output_config, "database", "output"),
        "output.database",
    )
    sparse_path = resolve_output_child(
        workspace_path,
        required_string(output_config, "sparse", "output"),
        "output.sparse",
    )

    if not colmap_path.is_file():
        raise PipelineError(
            f"COLMAP executable was not found: {colmap_path}\n"
            "Update 'colmap.executable' in the YAML config after installing COLMAP."
        )
    if not images_path.is_dir():
        raise PipelineError(f"RGB image directory was not found: {images_path}")
    if not masks_path.is_dir():
        raise PipelineError(f"COLMAP mask directory was not found: {masks_path}")

    images = image_files(images_path)
    if not images:
        raise PipelineError(f"No supported RGB images were found under: {images_path}")
    validate_masks(images, images_path, masks_path)
    ensure_clean_output(database_path, sparse_path)

    max_num_features = optional_positive_int(
        extraction_config, "max_num_features", "feature_extraction", 8192
    )
    memory_profile, memory_settings = feature_memory_profile(extraction_config)
    max_image_size = memory_settings["max_image_size"]
    first_octave = memory_settings["first_octave"]
    extraction_threads = memory_settings["num_threads"]
    estimate_affine_shape = optional_bool(
        extraction_config, "estimate_affine_shape", "feature_extraction", False
    )
    domain_size_pooling = optional_bool(
        extraction_config, "domain_size_pooling", "feature_extraction", False
    )
    if use_gpu and (estimate_affine_shape or domain_size_pooling):
        raise PipelineError(
            "Affine-shape or domain-size-pooling SIFT forces COLMAP onto the CPU even "
            "when colmap.use_gpu is true. Disable both for the GPU memory profiles."
        )
    guided_matching = optional_bool(
        matching_config, "guided_matching", "matching", True
    )
    matching_method = required_string(matching_config, "method", "matching").lower()
    if matching_method not in {"exhaustive", "sequential"}:
        raise PipelineError("'matching.method' must be 'exhaustive' or 'sequential'.")
    sequential_overlap = optional_positive_int(
        matching_config, "sequential_overlap", "matching", 10
    )
    analyze_models = optional_bool(
        postprocess_config, "analyze_models", "postprocess", True
    )
    export_ply = optional_bool(
        postprocess_config, "export_ply", "postprocess", True
    )

    print(f"Project: {project_name}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Images: {len(images)}")
    print(f"Masks: all {len(images)} required masks found")
    print(f"Workspace: {workspace_path}")
    print(
        "Feature memory profile: "
        f"{memory_profile} (max image {max_image_size}, first octave {first_octave}, "
        f"threads {extraction_threads})"
    )

    feature_arguments = [
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_path),
        "--ImageReader.mask_path",
        str(masks_path),
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.single_camera",
        bool_arg(single_camera),
        "--FeatureExtraction.use_gpu",
        bool_arg(use_gpu),
        "--FeatureExtraction.num_threads",
        str(extraction_threads),
        "--FeatureExtraction.max_image_size",
        str(max_image_size),
        "--SiftExtraction.max_num_features",
        str(max_num_features),
        "--SiftExtraction.first_octave",
        str(first_octave),
        "--SiftExtraction.estimate_affine_shape",
        bool_arg(estimate_affine_shape),
        "--SiftExtraction.domain_size_pooling",
        bool_arg(domain_size_pooling),
    ]

    matcher_arguments = [
        f"{matching_method}_matcher",
        "--database_path",
        str(database_path),
        "--FeatureMatching.use_gpu",
        bool_arg(use_gpu),
        "--FeatureMatching.guided_matching",
        bool_arg(guided_matching),
    ]
    if matching_method == "sequential":
        matcher_arguments.extend(["--SequentialMatching.overlap", str(sequential_overlap)])

    mapper_arguments = [
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_path),
        "--output_path",
        str(sparse_path),
    ]

    if args.dry_run:
        run_colmap(colmap_path, feature_arguments, "Feature extraction", dry_run=True)
        run_colmap(colmap_path, matcher_arguments, "Feature matching", dry_run=True)
        run_colmap(colmap_path, mapper_arguments, "Sparse mapping", dry_run=True)
        print("\nDry run complete. No files were created and COLMAP was not executed.")
        return 0

    workspace_path.mkdir(parents=True, exist_ok=True)
    sparse_path.mkdir(parents=True, exist_ok=True)

    run_colmap(colmap_path, ["version"], "COLMAP version")
    run_colmap(colmap_path, feature_arguments, "Feature extraction")
    run_colmap(colmap_path, matcher_arguments, "Feature matching")
    run_colmap(colmap_path, mapper_arguments, "Sparse mapping")

    models = find_sparse_models(sparse_path)
    if not models:
        raise PipelineError(
            "COLMAP finished without producing a sparse model. Inspect the terminal output "
            "for insufficient features, matches, or a failed initial image pair."
        )

    print(f"\nSparse models created: {len(models)}")
    for model_path in models:
        print(f"  {model_path}")
        if analyze_models:
            run_colmap(
                colmap_path,
                ["model_analyzer", "--path", str(model_path)],
                f"Analyze sparse model {model_path.name}",
            )
        if export_ply:
            ply_path = workspace_path / f"sparse_model_{model_path.name}.ply"
            run_colmap(
                colmap_path,
                [
                    "model_converter",
                    "--input_path",
                    str(model_path),
                    "--output_path",
                    str(ply_path),
                    "--output_type",
                    "PLY",
                ],
                f"Export sparse model {model_path.name} to PLY",
            )

    print(f"\nMasked sparse reconstruction complete: {workspace_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except PipelineError as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
