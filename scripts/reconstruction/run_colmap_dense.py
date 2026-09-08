#!/usr/bin/env python3
"""Run the masked COLMAP dense-reconstruction pipeline from YAML config."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "colmap_dense_pot1_unglazed_every6.yml"
PIPELINE_STAGES = (
    "validate",
    "prepare",
    "undistort",
    "patch-match",
    "fusion",
    "mesh",
    "delaunay-mesh",
    "simplify-mesh",
    "texture",
)
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
SPARSE_MODEL_COMPONENTS = ("cameras", "images", "points3D")


class PipelineError(RuntimeError):
    """A user-correctable configuration or dense-pipeline error."""


@dataclass(frozen=True)
class DensePaths:
    """Resolved paths used by the dense reconstruction."""

    config: Path
    colmap: Path
    source_images: Path
    source_masks: Path
    sparse_model: Path
    workspace: Path
    masked_images: Path
    mask_undistortion_images: Path
    undistorted_images: Path
    undistorted_masks: Path
    fused_point_cloud: Path
    poisson_mesh: Path
    simplified_poisson_mesh: Path
    delaunay_mesh: Path
    textured_mesh: Path
    logs: Path
    run_manifest: Path


@dataclass(frozen=True)
class ValidationReport:
    """Summary of a successfully validated dense-reconstruction input set."""

    image_count: int
    mask_count: int
    registered_image_count: int
    resolutions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class MaskPreparationSettings:
    """Settings for creating dense RGB and mask-undistortion inputs."""

    background_value: int
    threshold: int
    reconstruction_dilation_pixels: int
    jpeg_quality: int


@dataclass(frozen=True)
class PreparationReport:
    """Summary of prepared or resumed image pairs."""

    total: int
    written: int
    resumed: int


@dataclass(frozen=True)
class UndistortionSettings:
    """COLMAP image-undistortion settings."""

    output_type: str
    max_image_size: int
    jpeg_quality: int
    num_threads: int


@dataclass(frozen=True)
class UndistortionReport:
    """Summary of the masked-RGB undistortion stage."""

    image_count: int
    elapsed_seconds: float
    resumed: bool


@dataclass(frozen=True)
class PatchMatchSettings:
    """Memory-conscious PatchMatch settings for the selected GPU."""

    gpu_index: int
    max_image_size: int
    cache_size_gb: float
    num_threads: int
    geom_consistency: bool
    filter: bool


@dataclass(frozen=True)
class PatchMatchReport:
    """Summary of PatchMatch depth and normal map generation."""

    photometric_depth_maps: int
    geometric_depth_maps: int
    photometric_normal_maps: int
    geometric_normal_maps: int
    elapsed_seconds: float
    resumed: bool


@dataclass(frozen=True)
class FusionSettings:
    """Settings for masked geometric depth-map fusion."""

    input_type: str
    max_image_size: int
    cache_size_gb: float
    num_threads: int
    min_num_pixels: int


@dataclass(frozen=True)
class FusionReport:
    """Summary of the fused dense point cloud."""

    point_count: int
    elapsed_seconds: float
    resumed: bool


@dataclass(frozen=True)
class PoissonMeshingSettings:
    """Memory-conscious settings for Poisson surface reconstruction."""

    point_weight: float
    depth: int
    color: bool
    trim: float
    num_threads: int


@dataclass(frozen=True)
class MeshReport:
    """Summary of a validated surface mesh."""

    vertex_count: int
    face_count: int
    elapsed_seconds: float
    resumed: bool


@dataclass(frozen=True)
class DelaunayMeshingSettings:
    """Settings for visibility-aware dense Delaunay meshing."""

    input_type: str
    max_proj_dist: float
    max_depth_dist: float
    visibility_sigma: float
    distance_sigma_factor: float
    quality_regularization: float
    max_side_length_factor: float
    max_side_length_percentile: float
    num_threads: int


@dataclass(frozen=True)
class MeshSimplificationSettings:
    """Settings for QEM simplification of the Poisson mesh."""

    target_face_ratio: float
    max_error: float
    boundary_weight: float
    interpolate_colors: bool
    num_threads: int


@dataclass(frozen=True)
class SimplificationReport:
    """Summary of a validated simplified surface mesh."""

    vertex_count: int
    face_count: int
    input_face_count: int
    elapsed_seconds: float
    resumed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare masked inputs and run COLMAP undistortion, PatchMatch stereo, "
            "fusion, meshing, and texturing."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML configuration file (default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--stage",
        choices=("all", *PIPELINE_STAGES),
        default="all",
        help="Run one pipeline stage or all stages in order (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print the selected stages without writing outputs.",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Reuse completed stage outputs when they pass validation.",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not reuse completed stage outputs.",
    )
    parser.set_defaults(resume=None)
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


def optional_nonnegative_int(
    values: Mapping[str, Any], key: str, section_name: str, default: int
) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PipelineError(f"'{section_name}.{key}' must be a non-negative integer.")
    return value


def optional_positive_int(
    values: Mapping[str, Any], key: str, section_name: str, default: int
) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PipelineError(f"'{section_name}.{key}' must be a positive integer.")
    return value


def optional_positive_number(
    values: Mapping[str, Any], key: str, section_name: str, default: float
) -> float:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise PipelineError(f"'{section_name}.{key}' must be a positive number.")
    return float(value)


def optional_nonnegative_number(
    values: Mapping[str, Any], key: str, section_name: str, default: float
) -> float:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise PipelineError(f"'{section_name}.{key}' must be a non-negative number.")
    return float(value)


def resolve_config_path(raw_path: Path) -> Path:
    candidate = raw_path
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise PipelineError(
            f"The configuration must stay inside the project directory: {candidate}"
        ) from error
    return candidate


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


def resolve_paths(config_path: Path, config: Mapping[str, Any]) -> DensePaths:
    colmap_config = section(config, "colmap")
    input_config = section(config, "input")
    output_config = section(config, "output")

    workspace = resolve_project_path(
        required_string(output_config, "workspace", "output"), "output.workspace"
    )

    def output_path(key: str) -> Path:
        return resolve_output_child(
            workspace,
            required_string(output_config, key, "output"),
            f"output.{key}",
        )

    return DensePaths(
        config=config_path,
        colmap=resolve_colmap_path(
            required_string(colmap_config, "executable", "colmap")
        ),
        source_images=resolve_project_path(
            required_string(input_config, "images", "input"), "input.images"
        ),
        source_masks=resolve_project_path(
            required_string(input_config, "masks", "input"), "input.masks"
        ),
        sparse_model=resolve_project_path(
            required_string(input_config, "sparse_model", "input"),
            "input.sparse_model",
        ),
        workspace=workspace,
        masked_images=output_path("masked_images"),
        mask_undistortion_images=output_path("mask_undistortion_images"),
        undistorted_images=output_path("undistorted_images"),
        undistorted_masks=output_path("undistorted_masks"),
        fused_point_cloud=output_path("fused_point_cloud"),
        poisson_mesh=output_path("poisson_mesh"),
        simplified_poisson_mesh=output_path("simplified_poisson_mesh"),
        delaunay_mesh=output_path("delaunay_mesh"),
        textured_mesh=output_path("textured_mesh"),
        logs=output_path("logs"),
        run_manifest=output_path("run_manifest"),
    )


def image_files(image_root: Path) -> list[Path]:
    return sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def expected_mask_path(image_path: Path, image_root: Path, mask_root: Path) -> Path:
    relative_image = image_path.relative_to(image_root)
    return mask_root / Path(f"{relative_image}.png")


def format_examples(values: Iterable[str], limit: int = 10) -> str:
    examples = list(values)
    shown = "\n".join(f"  {value}" for value in examples[:limit])
    if len(examples) > limit:
        shown += f"\n  ...and {len(examples) - limit} more"
    return shown


def validate_sparse_components(sparse_model: Path) -> None:
    if not sparse_model.is_dir():
        raise PipelineError(f"Sparse model directory was not found: {sparse_model}")

    missing = [
        component
        for component in SPARSE_MODEL_COMPONENTS
        if not any(
            (sparse_model / f"{component}.{extension}").is_file()
            for extension in ("bin", "txt")
        )
    ]
    if missing:
        raise PipelineError(
            "Sparse model is missing required component(s): " + ", ".join(missing)
        )


def read_text_model_image_names(images_txt: Path) -> list[str]:
    """Read image names from COLMAP's two-lines-per-image text format."""

    names: list[str] = []
    expecting_image_line = True
    try:
        lines = images_txt.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise PipelineError(f"Could not read converted sparse model: {error}") from error

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            continue
        if expecting_image_line:
            if not stripped:
                continue
            fields = stripped.split(maxsplit=9)
            if len(fields) < 10:
                raise PipelineError(
                    f"Malformed image record in converted model: {raw_line}"
                )
            names.append(fields[9].replace("\\", "/"))
            expecting_image_line = False
        else:
            # The points2D record may be empty, but it still occupies the next line.
            expecting_image_line = True

    if not expecting_image_line:
        raise PipelineError("Converted images.txt ended before its points2D record.")
    if not names:
        raise PipelineError("The sparse model contains no registered images.")
    if len(names) != len(set(names)):
        raise PipelineError("The sparse model contains duplicate registered image names.")
    return names


def registered_image_names(colmap: Path, sparse_model: Path) -> list[str]:
    """Ask COLMAP to export a temporary text model and return registered names."""

    with tempfile.TemporaryDirectory(prefix="colmap_dense_validate_") as directory:
        text_model = Path(directory)
        arguments = [
            str(colmap),
            "model_converter",
            "--input_path",
            str(sparse_model),
            "--output_path",
            str(text_model),
            "--output_type",
            "TXT",
        ]
        try:
            completed = subprocess.run(
                arguments,
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise PipelineError(
                f"Could not launch COLMAP while validating the sparse model: {error}"
            ) from error

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            if detail:
                detail = f"\n{detail[-2000:]}"
            raise PipelineError(
                "COLMAP could not convert the sparse model for validation "
                f"(exit code {completed.returncode}).{detail}"
            )
        return read_text_model_image_names(text_model / "images.txt")


def validate_image_mask_pairs(
    images: Iterable[Path], image_root: Path, mask_root: Path
) -> tuple[int, tuple[tuple[int, int], ...]]:
    try:
        from PIL import Image, UnidentifiedImageError
    except ModuleNotFoundError as error:
        raise PipelineError(
            "Pillow is required to validate RGB images and masks. Update the "
            "pot-masking environment from environment-masking.yml."
        ) from error

    missing_masks: list[str] = []
    unreadable_pairs: list[str] = []
    dimension_mismatches: list[str] = []
    empty_masks: list[str] = []
    resolutions: set[tuple[int, int]] = set()
    checked = 0

    for image_path in images:
        relative_image = image_path.relative_to(image_root)
        mask_path = expected_mask_path(image_path, image_root, mask_root)
        if not mask_path.is_file():
            missing_masks.append(
                f"{relative_image.as_posix()} -> "
                f"{mask_path.relative_to(mask_root).as_posix()}"
            )
            continue

        try:
            with Image.open(image_path) as rgb_image, Image.open(mask_path) as mask_image:
                rgb_image.verify()
                mask_image.verify()
            with Image.open(image_path) as rgb_image, Image.open(mask_path) as mask_image:
                rgb_size = rgb_image.size
                mask_size = mask_image.size
                resolutions.add(rgb_size)
                if rgb_size != mask_size:
                    dimension_mismatches.append(
                        f"{relative_image.as_posix()}: RGB {rgb_size[0]}x{rgb_size[1]}, "
                        f"mask {mask_size[0]}x{mask_size[1]}"
                    )
                grayscale_mask = mask_image.convert("L")
                try:
                    if grayscale_mask.getbbox() is None:
                        empty_masks.append(relative_image.as_posix())
                finally:
                    grayscale_mask.close()
        except (OSError, UnidentifiedImageError) as error:
            unreadable_pairs.append(f"{relative_image.as_posix()}: {error}")
        checked += 1

    problems: list[str] = []
    if missing_masks:
        problems.append(
            f"Missing masks for {len(missing_masks)} image(s):\n"
            + format_examples(missing_masks)
        )
    if unreadable_pairs:
        problems.append(
            f"Unreadable RGB/mask pairs: {len(unreadable_pairs)}:\n"
            + format_examples(unreadable_pairs)
        )
    if dimension_mismatches:
        problems.append(
            f"Dimension mismatches: {len(dimension_mismatches)}:\n"
            + format_examples(dimension_mismatches)
        )
    if empty_masks:
        problems.append(
            f"Empty masks: {len(empty_masks)}:\n" + format_examples(empty_masks)
        )
    if problems:
        raise PipelineError("\n\n".join(problems))

    return checked, tuple(sorted(resolutions))


def validate_inputs(paths: DensePaths) -> ValidationReport:
    if not paths.colmap.is_file():
        raise PipelineError(f"COLMAP executable was not found: {paths.colmap}")
    if not paths.source_images.is_dir():
        raise PipelineError(f"RGB image directory was not found: {paths.source_images}")
    if not paths.source_masks.is_dir():
        raise PipelineError(f"COLMAP mask directory was not found: {paths.source_masks}")

    validate_sparse_components(paths.sparse_model)
    images = image_files(paths.source_images)
    if not images:
        raise PipelineError(
            f"No supported RGB images were found under: {paths.source_images}"
        )

    checked, resolutions = validate_image_mask_pairs(
        images, paths.source_images, paths.source_masks
    )
    registered_names = registered_image_names(paths.colmap, paths.sparse_model)
    source_names = {
        path.relative_to(paths.source_images).as_posix() for path in images
    }
    sparse_names = {name.replace("\\", "/") for name in registered_names}

    missing_source_images = sorted(sparse_names - source_names)
    unregistered_source_images = sorted(source_names - sparse_names)
    if missing_source_images or unregistered_source_images:
        details: list[str] = []
        if missing_source_images:
            details.append(
                "Registered images missing from the RGB directory:\n"
                + format_examples(missing_source_images)
            )
        if unregistered_source_images:
            details.append(
                "RGB images not registered in the selected sparse model:\n"
                + format_examples(unregistered_source_images)
            )
        raise PipelineError("\n\n".join(details))

    mask_count = sum(
        1
        for path in paths.source_masks.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    )
    return ValidationReport(
        image_count=checked,
        mask_count=mask_count,
        registered_image_count=len(registered_names),
        resolutions=resolutions,
    )


def validate_config_schema(config: Mapping[str, Any]) -> None:
    project_config = section(config, "project")
    masking_config = section(config, "masking")
    undistortion_config = section(config, "undistortion")
    patch_match_config = section(config, "patch_match")
    fusion_config = section(config, "fusion")
    meshing_config = section(config, "meshing")
    runtime_config = section(config, "runtime")

    required_string(project_config, "name", "project")

    background_value = optional_nonnegative_int(
        masking_config, "background_value", "masking", 0
    )
    threshold = optional_positive_int(masking_config, "threshold", "masking", 128)
    jpeg_quality = optional_positive_int(
        masking_config, "jpeg_quality", "masking", 100
    )
    if background_value > 255:
        raise PipelineError("'masking.background_value' must be between 0 and 255.")
    if threshold > 255:
        raise PipelineError("'masking.threshold' must be between 1 and 255.")
    if jpeg_quality > 100:
        raise PipelineError("'masking.jpeg_quality' must be between 1 and 100.")
    optional_nonnegative_int(
        masking_config,
        "reconstruction_dilation_pixels",
        "masking",
        0,
    )
    optional_nonnegative_int(
        masking_config, "fusion_erosion_pixels", "masking", 0
    )

    required_string(undistortion_config, "output_type", "undistortion")
    optional_positive_int(
        undistortion_config, "max_image_size", "undistortion", 2000
    )
    undistortion_jpeg_quality = optional_positive_int(
        undistortion_config, "jpeg_quality", "undistortion", 100
    )
    if undistortion_jpeg_quality > 100:
        raise PipelineError("'undistortion.jpeg_quality' must be between 1 and 100.")
    optional_positive_int(
        undistortion_config, "num_threads", "undistortion", 2
    )

    optional_nonnegative_int(patch_match_config, "gpu_index", "patch_match", 0)
    optional_positive_int(
        patch_match_config, "max_image_size", "patch_match", 1600
    )
    optional_positive_number(
        patch_match_config, "cache_size_gb", "patch_match", 6
    )
    optional_positive_int(
        patch_match_config, "num_threads", "patch_match", 2
    )
    optional_bool(
        patch_match_config, "geom_consistency", "patch_match", True
    )
    optional_bool(patch_match_config, "filter", "patch_match", True)

    required_string(fusion_config, "input_type", "fusion")
    optional_positive_int(fusion_config, "max_image_size", "fusion", 1600)
    optional_positive_number(fusion_config, "cache_size_gb", "fusion", 6)
    optional_positive_int(fusion_config, "num_threads", "fusion", 4)
    optional_positive_int(fusion_config, "min_num_pixels", "fusion", 3)

    optional_bool(meshing_config, "run_poisson", "meshing", True)
    optional_bool(meshing_config, "run_delaunay", "meshing", True)
    poisson_config = section(meshing_config, "poisson")
    optional_positive_number(
        poisson_config, "point_weight", "meshing.poisson", 1.0
    )
    poisson_depth = optional_positive_int(
        poisson_config, "depth", "meshing.poisson", 11
    )
    if poisson_depth > 15:
        raise PipelineError(
            "'meshing.poisson.depth' must not exceed 15; larger values can require "
            "extreme amounts of RAM."
        )
    optional_bool(poisson_config, "color", "meshing.poisson", True)
    optional_positive_number(poisson_config, "trim", "meshing.poisson", 5.0)
    optional_positive_int(
        poisson_config, "num_threads", "meshing.poisson", 4
    )
    delaunay_config = section(meshing_config, "delaunay")
    input_type = required_string(
        delaunay_config, "input_type", "meshing.delaunay"
    ).lower()
    if input_type != "dense":
        raise PipelineError("'meshing.delaunay.input_type' must be 'dense'.")
    optional_positive_number(
        delaunay_config, "max_proj_dist", "meshing.delaunay", 20.0
    )
    max_depth_dist = optional_positive_number(
        delaunay_config, "max_depth_dist", "meshing.delaunay", 0.05
    )
    if max_depth_dist > 1:
        raise PipelineError("'meshing.delaunay.max_depth_dist' must not exceed 1.")
    optional_positive_number(
        delaunay_config, "visibility_sigma", "meshing.delaunay", 3.0
    )
    optional_positive_number(
        delaunay_config, "distance_sigma_factor", "meshing.delaunay", 1.0
    )
    optional_positive_number(
        delaunay_config, "quality_regularization", "meshing.delaunay", 1.0
    )
    optional_positive_number(
        delaunay_config, "max_side_length_factor", "meshing.delaunay", 25.0
    )
    max_side_percentile = optional_positive_number(
        delaunay_config,
        "max_side_length_percentile",
        "meshing.delaunay",
        95.0,
    )
    if max_side_percentile > 100:
        raise PipelineError(
            "'meshing.delaunay.max_side_length_percentile' must not exceed 100."
        )
    optional_positive_int(
        delaunay_config, "num_threads", "meshing.delaunay", 4
    )
    simplification_config = section(meshing_config, "simplification")
    target_face_ratio = optional_positive_number(
        simplification_config,
        "target_face_ratio",
        "meshing.simplification",
        0.1,
    )
    if target_face_ratio > 1:
        raise PipelineError(
            "'meshing.simplification.target_face_ratio' must not exceed 1."
        )
    optional_nonnegative_number(
        simplification_config, "max_error", "meshing.simplification", 0.0
    )
    optional_positive_number(
        simplification_config,
        "boundary_weight",
        "meshing.simplification",
        1000.0,
    )
    optional_bool(
        simplification_config,
        "interpolate_colors",
        "meshing.simplification",
        True,
    )
    optional_positive_int(
        simplification_config, "num_threads", "meshing.simplification", 4
    )
    optional_bool(runtime_config, "resume", "runtime", True)
    optional_bool(runtime_config, "stop_on_error", "runtime", True)


def mask_preparation_settings(
    config: Mapping[str, Any]
) -> MaskPreparationSettings:
    masking_config = section(config, "masking")
    return MaskPreparationSettings(
        background_value=optional_nonnegative_int(
            masking_config, "background_value", "masking", 0
        ),
        threshold=optional_positive_int(
            masking_config, "threshold", "masking", 128
        ),
        reconstruction_dilation_pixels=optional_nonnegative_int(
            masking_config,
            "reconstruction_dilation_pixels",
            "masking",
            0,
        ),
        jpeg_quality=optional_positive_int(
            masking_config, "jpeg_quality", "masking", 100
        ),
    )


def undistortion_settings(config: Mapping[str, Any]) -> UndistortionSettings:
    values = section(config, "undistortion")
    output_type = required_string(values, "output_type", "undistortion").upper()
    if output_type != "COLMAP":
        raise PipelineError(
            "'undistortion.output_type' must be COLMAP for PatchMatch stereo."
        )
    return UndistortionSettings(
        output_type=output_type,
        max_image_size=optional_positive_int(
            values, "max_image_size", "undistortion", 2000
        ),
        jpeg_quality=optional_positive_int(
            values, "jpeg_quality", "undistortion", 100
        ),
        num_threads=optional_positive_int(
            values, "num_threads", "undistortion", 2
        ),
    )


def patch_match_settings(config: Mapping[str, Any]) -> PatchMatchSettings:
    values = section(config, "patch_match")
    return PatchMatchSettings(
        gpu_index=optional_nonnegative_int(
            values, "gpu_index", "patch_match", 0
        ),
        max_image_size=optional_positive_int(
            values, "max_image_size", "patch_match", 1600
        ),
        cache_size_gb=optional_positive_number(
            values, "cache_size_gb", "patch_match", 6
        ),
        num_threads=optional_positive_int(
            values, "num_threads", "patch_match", 2
        ),
        geom_consistency=optional_bool(
            values, "geom_consistency", "patch_match", True
        ),
        filter=optional_bool(values, "filter", "patch_match", True),
    )


def fusion_settings(config: Mapping[str, Any]) -> FusionSettings:
    values = section(config, "fusion")
    input_type = required_string(values, "input_type", "fusion").lower()
    if input_type not in {"photometric", "geometric"}:
        raise PipelineError(
            "'fusion.input_type' must be 'photometric' or 'geometric'."
        )
    return FusionSettings(
        input_type=input_type,
        max_image_size=optional_positive_int(
            values, "max_image_size", "fusion", 1200
        ),
        cache_size_gb=optional_positive_number(
            values, "cache_size_gb", "fusion", 6
        ),
        num_threads=optional_positive_int(values, "num_threads", "fusion", 4),
        min_num_pixels=optional_positive_int(
            values, "min_num_pixels", "fusion", 3
        ),
    )


def poisson_meshing_settings(config: Mapping[str, Any]) -> PoissonMeshingSettings:
    meshing_config = section(config, "meshing")
    if not optional_bool(meshing_config, "run_poisson", "meshing", True):
        raise PipelineError(
            "Poisson meshing is disabled by 'meshing.run_poisson' in the YAML config."
        )
    values = section(meshing_config, "poisson")
    depth = optional_positive_int(values, "depth", "meshing.poisson", 11)
    if depth > 15:
        raise PipelineError(
            "'meshing.poisson.depth' must not exceed 15; larger values can require "
            "extreme amounts of RAM."
        )
    return PoissonMeshingSettings(
        point_weight=optional_positive_number(
            values, "point_weight", "meshing.poisson", 1.0
        ),
        depth=depth,
        color=optional_bool(values, "color", "meshing.poisson", True),
        trim=optional_positive_number(values, "trim", "meshing.poisson", 5.0),
        num_threads=optional_positive_int(
            values, "num_threads", "meshing.poisson", 4
        ),
    )


def delaunay_meshing_settings(config: Mapping[str, Any]) -> DelaunayMeshingSettings:
    meshing_config = section(config, "meshing")
    if not optional_bool(meshing_config, "run_delaunay", "meshing", True):
        raise PipelineError(
            "Delaunay meshing is disabled by 'meshing.run_delaunay' in the YAML config."
        )
    values = section(meshing_config, "delaunay")
    input_type = required_string(values, "input_type", "meshing.delaunay").lower()
    if input_type != "dense":
        raise PipelineError("'meshing.delaunay.input_type' must be 'dense'.")
    max_depth_dist = optional_positive_number(
        values, "max_depth_dist", "meshing.delaunay", 0.05
    )
    max_side_percentile = optional_positive_number(
        values,
        "max_side_length_percentile",
        "meshing.delaunay",
        95.0,
    )
    if max_depth_dist > 1:
        raise PipelineError("'meshing.delaunay.max_depth_dist' must not exceed 1.")
    if max_side_percentile > 100:
        raise PipelineError(
            "'meshing.delaunay.max_side_length_percentile' must not exceed 100."
        )
    return DelaunayMeshingSettings(
        input_type=input_type,
        max_proj_dist=optional_positive_number(
            values, "max_proj_dist", "meshing.delaunay", 20.0
        ),
        max_depth_dist=max_depth_dist,
        visibility_sigma=optional_positive_number(
            values, "visibility_sigma", "meshing.delaunay", 3.0
        ),
        distance_sigma_factor=optional_positive_number(
            values, "distance_sigma_factor", "meshing.delaunay", 1.0
        ),
        quality_regularization=optional_positive_number(
            values, "quality_regularization", "meshing.delaunay", 1.0
        ),
        max_side_length_factor=optional_positive_number(
            values, "max_side_length_factor", "meshing.delaunay", 25.0
        ),
        max_side_length_percentile=max_side_percentile,
        num_threads=optional_positive_int(
            values, "num_threads", "meshing.delaunay", 4
        ),
    )


def mesh_simplification_settings(
    config: Mapping[str, Any]
) -> MeshSimplificationSettings:
    values = section(section(config, "meshing"), "simplification")
    target_face_ratio = optional_positive_number(
        values, "target_face_ratio", "meshing.simplification", 0.1
    )
    if target_face_ratio > 1:
        raise PipelineError(
            "'meshing.simplification.target_face_ratio' must not exceed 1."
        )
    return MeshSimplificationSettings(
        target_face_ratio=target_face_ratio,
        max_error=optional_nonnegative_number(
            values, "max_error", "meshing.simplification", 0.0
        ),
        boundary_weight=optional_positive_number(
            values, "boundary_weight", "meshing.simplification", 1000.0
        ),
        interpolate_colors=optional_bool(
            values, "interpolate_colors", "meshing.simplification", True
        ),
        num_threads=optional_positive_int(
            values, "num_threads", "meshing.simplification", 4
        ),
    )


def pillow_save_format(path: Path) -> str:
    formats = {
        ".bmp": "BMP",
        ".jpeg": "JPEG",
        ".jpg": "JPEG",
        ".png": "PNG",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".webp": "WEBP",
    }
    try:
        return formats[path.suffix.lower()]
    except KeyError as error:
        raise PipelineError(f"Unsupported prepared-image extension: {path}") from error


def atomic_save_image(image: Any, destination: Path, jpeg_quality: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    image_format = pillow_save_format(destination)
    save_options: dict[str, Any] = {}
    if image_format == "JPEG":
        save_options.update(quality=jpeg_quality, subsampling=0)
    elif image_format == "WEBP":
        save_options.update(quality=jpeg_quality)

    try:
        image.save(temporary, format=image_format, **save_options)
        temporary.replace(destination)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise PipelineError(f"Could not write prepared image {destination}: {error}") from error


def prepared_pair_is_valid(
    masked_image: Path, mask_image: Path, expected_size: tuple[int, int]
) -> bool:
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise PipelineError(
            "Pillow is required to validate prepared images. Update the "
            "pot-masking environment from environment-masking.yml."
        ) from error

    try:
        with Image.open(masked_image) as rgb_output:
            if rgb_output.size != expected_size:
                return False
            rgb_output.verify()
        with Image.open(mask_image) as mask_output:
            if mask_output.size != expected_size:
                return False
            mask_output.verify()
    except OSError:
        return False
    return True


def prepare_image_pair(
    source_image: Path,
    source_mask: Path,
    masked_output: Path,
    mask_output: Path,
    settings: MaskPreparationSettings,
) -> None:
    try:
        from PIL import Image, ImageFilter, UnidentifiedImageError
    except ModuleNotFoundError as error:
        raise PipelineError(
            "Pillow is required to prepare masked RGB inputs. Update the "
            "pot-masking environment from environment-masking.yml."
        ) from error

    rgb_image = None
    grayscale_mask = None
    binary_mask = None
    reconstruction_mask = None
    background = None
    masked_rgb = None
    mask_rgb = None
    try:
        with Image.open(source_image) as opened_rgb:
            rgb_image = opened_rgb.convert("RGB")
        with Image.open(source_mask) as opened_mask:
            grayscale_mask = opened_mask.convert("L")

        if rgb_image.size != grayscale_mask.size:
            raise PipelineError(
                f"Cannot prepare mismatched RGB and mask: {source_image.name} "
                f"is {rgb_image.size[0]}x{rgb_image.size[1]}, mask is "
                f"{grayscale_mask.size[0]}x{grayscale_mask.size[1]}."
            )

        binary_mask = grayscale_mask.point(
            lambda value: 255 if value >= settings.threshold else 0,
            mode="L",
        )
        if binary_mask.getbbox() is None:
            raise PipelineError(f"Cannot prepare an empty mask: {source_mask}")

        if settings.reconstruction_dilation_pixels > 0:
            kernel_size = settings.reconstruction_dilation_pixels * 2 + 1
            reconstruction_mask = binary_mask.filter(
                ImageFilter.MaxFilter(kernel_size)
            )
        else:
            reconstruction_mask = binary_mask.copy()

        background_color = (settings.background_value,) * 3
        background = Image.new("RGB", rgb_image.size, background_color)
        masked_rgb = Image.composite(rgb_image, background, reconstruction_mask)
        mask_rgb = binary_mask.convert("RGB")

        atomic_save_image(masked_rgb, masked_output, settings.jpeg_quality)
        atomic_save_image(mask_rgb, mask_output, settings.jpeg_quality)
    except (OSError, UnidentifiedImageError) as error:
        raise PipelineError(
            f"Could not prepare {source_image.name}: {error}"
        ) from error
    finally:
        for image in (
            rgb_image,
            grayscale_mask,
            binary_mask,
            reconstruction_mask,
            background,
            masked_rgb,
            mask_rgb,
        ):
            if image is not None:
                image.close()


def prepare_masked_inputs(
    paths: DensePaths,
    settings: MaskPreparationSettings,
    resume: bool,
) -> PreparationReport:
    images = image_files(paths.source_images)
    if not images:
        raise PipelineError(f"No RGB images were found under: {paths.source_images}")

    if not resume:
        existing_outputs = []
        for source_image in images:
            relative_image = source_image.relative_to(paths.source_images)
            for destination_root in (
                paths.masked_images,
                paths.mask_undistortion_images,
            ):
                destination = destination_root / relative_image
                if destination.exists():
                    existing_outputs.append(destination)
        if existing_outputs:
            raise PipelineError(
                "Prepared outputs already exist while --no-resume is selected:\n"
                + format_examples(str(path) for path in existing_outputs)
                + "\nMove the existing prepared outputs before starting a fresh run."
            )

    paths.masked_images.mkdir(parents=True, exist_ok=True)
    paths.mask_undistortion_images.mkdir(parents=True, exist_ok=True)
    written = 0
    resumed = 0

    for index, source_image in enumerate(images, start=1):
        relative_image = source_image.relative_to(paths.source_images)
        source_mask = expected_mask_path(
            source_image, paths.source_images, paths.source_masks
        )
        masked_output = paths.masked_images / relative_image
        mask_output = paths.mask_undistortion_images / relative_image

        if resume:
            try:
                from PIL import Image
                with Image.open(source_image) as opened_source:
                    expected_size = opened_source.size
            except (ModuleNotFoundError, OSError) as error:
                raise PipelineError(
                    f"Could not inspect source image {source_image}: {error}"
                ) from error
            if prepared_pair_is_valid(masked_output, mask_output, expected_size):
                resumed += 1
                continue

        prepare_image_pair(
            source_image,
            source_mask,
            masked_output,
            mask_output,
            settings,
        )
        written += 1
        if index == 1 or index % 25 == 0 or index == len(images):
            print(f"Prepared {index}/{len(images)} image pairs", flush=True)

    return PreparationReport(total=len(images), written=written, resumed=resumed)


def validate_prepared_inputs(paths: DensePaths) -> int:
    source_images = image_files(paths.source_images)
    invalid: list[str] = []
    for source_image in source_images:
        relative_image = source_image.relative_to(paths.source_images)
        masked_image = paths.masked_images / relative_image
        mask_image = paths.mask_undistortion_images / relative_image
        try:
            from PIL import Image
            with Image.open(source_image) as opened_source:
                expected_size = opened_source.size
        except (ModuleNotFoundError, OSError) as error:
            raise PipelineError(
                f"Could not inspect source image {source_image}: {error}"
            ) from error
        if not prepared_pair_is_valid(masked_image, mask_image, expected_size):
            invalid.append(relative_image.as_posix())

    if invalid:
        raise PipelineError(
            f"Missing or invalid prepared image pairs: {len(invalid)}:\n"
            + format_examples(invalid)
            + "\nRun --stage prepare before undistortion."
        )
    return len(source_images)


def display_command(arguments: list[str]) -> str:
    return subprocess.list2cmdline(arguments)


def run_colmap_logged(
    arguments: list[str], stage_name: str, log_path: Path
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = display_command(arguments)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    print(f"\n[{stage_name}]\n{command_text}", flush=True)

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"Stage: {stage_name}\n")
            log_file.write(f"Started UTC: {started_at.isoformat()}\n")
            log_file.write(f"Command: {command_text}\n\n")
            process = subprocess.Popen(
                arguments,
                cwd=PROJECT_ROOT,
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
                log_file.write(line)
            return_code = process.wait()
            elapsed = time.perf_counter() - started
            log_file.write(f"\nElapsed seconds: {elapsed:.3f}\n")
            log_file.write(f"Exit code: {return_code}\n")
    except OSError as error:
        raise PipelineError(f"Could not run COLMAP during '{stage_name}': {error}") from error

    if return_code != 0:
        raise PipelineError(
            f"COLMAP stage '{stage_name}' failed with exit code {return_code}. "
            f"Inspect {log_path}."
        )
    return elapsed


def undistorted_workspace_is_complete(paths: DensePaths, expected_count: int) -> bool:
    try:
        validate_sparse_components(paths.workspace / "sparse")
    except PipelineError:
        return False
    if len(image_files(paths.undistorted_images)) != expected_count:
        return False
    return all(
        (paths.workspace / "stereo" / filename).is_file()
        for filename in ("patch-match.cfg", "fusion.cfg")
    )


def generated_undistortion_files(paths: DensePaths) -> list[Path]:
    generated: list[Path] = []
    for root in (
        paths.undistorted_images,
        paths.workspace / "sparse",
        paths.workspace / "stereo",
    ):
        if root.is_dir():
            generated.extend(path for path in root.rglob("*") if path.is_file())
    return generated


def run_rgb_undistortion(
    paths: DensePaths,
    settings: UndistortionSettings,
    expected_count: int,
    resume: bool,
) -> UndistortionReport:
    if resume and undistorted_workspace_is_complete(paths, expected_count):
        return UndistortionReport(
            image_count=expected_count,
            elapsed_seconds=0.0,
            resumed=True,
        )

    existing = generated_undistortion_files(paths)
    if existing:
        raise PipelineError(
            "The undistorted COLMAP workspace is incomplete but already contains "
            "generated files:\n"
            + format_examples(str(path) for path in existing)
            + "\nMove the incomplete generated workspace files before retrying."
        )

    paths.workspace.mkdir(parents=True, exist_ok=True)
    arguments = [
        str(paths.colmap),
        "image_undistorter",
        "--image_path",
        str(paths.masked_images),
        "--input_path",
        str(paths.sparse_model),
        "--output_path",
        str(paths.workspace),
        "--output_type",
        settings.output_type,
        "--max_image_size",
        str(settings.max_image_size),
        "--jpeg_quality",
        str(settings.jpeg_quality),
        "--num_threads",
        str(settings.num_threads),
    ]
    elapsed = run_colmap_logged(
        arguments,
        "Undistort masked RGB images",
        paths.logs / "02_undistort_rgb.log",
    )
    if not undistorted_workspace_is_complete(paths, expected_count):
        raise PipelineError(
            "COLMAP returned success, but the undistorted workspace is incomplete. "
            f"Inspect {paths.logs / '02_undistort_rgb.log'}."
        )
    return UndistortionReport(
        image_count=expected_count,
        elapsed_seconds=elapsed,
        resumed=False,
    )


def colmap_bool(value: bool) -> str:
    return "1" if value else "0"


def build_patch_match_arguments(
    paths: DensePaths, settings: PatchMatchSettings
) -> list[str]:
    return [
        str(paths.colmap),
        "patch_match_stereo",
        "--workspace_path",
        str(paths.workspace),
        "--workspace_format",
        "COLMAP",
        "--PatchMatchStereo.gpu_index",
        str(settings.gpu_index),
        "--PatchMatchStereo.max_image_size",
        str(settings.max_image_size),
        "--PatchMatchStereo.cache_size",
        str(settings.cache_size_gb),
        "--PatchMatchStereo.num_threads",
        str(settings.num_threads),
        "--PatchMatchStereo.geom_consistency",
        colmap_bool(settings.geom_consistency),
        "--PatchMatchStereo.filter",
        colmap_bool(settings.filter),
    ]


def patch_match_map_counts(paths: DensePaths) -> tuple[int, int, int, int]:
    depth_root = paths.workspace / "stereo" / "depth_maps"
    normal_root = paths.workspace / "stereo" / "normal_maps"
    return (
        len(list(depth_root.rglob("*.photometric.bin"))) if depth_root.is_dir() else 0,
        len(list(depth_root.rglob("*.geometric.bin"))) if depth_root.is_dir() else 0,
        len(list(normal_root.rglob("*.photometric.bin"))) if normal_root.is_dir() else 0,
        len(list(normal_root.rglob("*.geometric.bin"))) if normal_root.is_dir() else 0,
    )


def patch_match_is_complete(
    paths: DensePaths, expected_count: int, geom_consistency: bool
) -> bool:
    photometric_depth, geometric_depth, photometric_normal, geometric_normal = (
        patch_match_map_counts(paths)
    )
    if photometric_depth != expected_count or photometric_normal != expected_count:
        return False
    if geom_consistency:
        return geometric_depth == expected_count and geometric_normal == expected_count
    return True


def generated_patch_match_files(paths: DensePaths) -> list[Path]:
    generated: list[Path] = []
    for root in (
        paths.workspace / "stereo" / "depth_maps",
        paths.workspace / "stereo" / "normal_maps",
        paths.workspace / "stereo" / "consistency_graphs",
    ):
        if root.is_dir():
            generated.extend(path for path in root.rglob("*") if path.is_file())
    return generated


def validate_patch_match_readiness(paths: DensePaths) -> int:
    if not paths.colmap.is_file():
        raise PipelineError(f"COLMAP executable was not found: {paths.colmap}")
    expected_count = len(image_files(paths.undistorted_images))
    if expected_count == 0:
        raise PipelineError(
            f"No undistorted dense RGB images were found: {paths.undistorted_images}"
        )
    if not undistorted_workspace_is_complete(paths, expected_count):
        raise PipelineError(
            "The undistorted COLMAP workspace is incomplete. Run --stage undistort "
            "and verify its output before PatchMatch."
        )
    patch_match_config = paths.workspace / "stereo" / "patch-match.cfg"
    if patch_match_config.stat().st_size == 0:
        raise PipelineError(f"PatchMatch configuration is empty: {patch_match_config}")
    return expected_count


def run_patch_match(
    paths: DensePaths,
    settings: PatchMatchSettings,
    expected_count: int,
    resume: bool,
) -> PatchMatchReport:
    if resume and patch_match_is_complete(
        paths, expected_count, settings.geom_consistency
    ):
        counts = patch_match_map_counts(paths)
        return PatchMatchReport(*counts, elapsed_seconds=0.0, resumed=True)

    existing = generated_patch_match_files(paths)
    if existing:
        counts = patch_match_map_counts(paths)
        raise PipelineError(
            "PatchMatch output is incomplete but already contains generated files. "
            f"Current photometric/geometric depth counts: {counts[0]}/{counts[1]}; "
            f"normal counts: {counts[2]}/{counts[3]}. Move the partial depth_maps, "
            "normal_maps, and consistency_graphs directories before starting a "
            "fresh PatchMatch run."
        )

    elapsed = run_colmap_logged(
        build_patch_match_arguments(paths, settings),
        "PatchMatch stereo",
        paths.logs / "03_patch_match.log",
    )
    counts = patch_match_map_counts(paths)
    if not patch_match_is_complete(paths, expected_count, settings.geom_consistency):
        raise PipelineError(
            "COLMAP returned success, but PatchMatch output is incomplete. "
            f"Photometric/geometric depth counts: {counts[0]}/{counts[1]}; "
            f"normal counts: {counts[2]}/{counts[3]}. Inspect "
            f"{paths.logs / '03_patch_match.log'}."
        )
    return PatchMatchReport(*counts, elapsed_seconds=elapsed, resumed=False)


def validate_fusion_masks(paths: DensePaths, expected_count: int) -> int:
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise PipelineError(
            "Pillow is required to validate fusion masks. Update the pot-masking "
            "environment from environment-masking.yml."
        ) from error

    dense_images = image_files(paths.undistorted_images)
    if len(dense_images) != expected_count:
        raise PipelineError(
            f"Expected {expected_count} undistorted images, found {len(dense_images)}."
        )

    expected_masks: set[Path] = set()
    problems: list[str] = []
    valid_count = 0
    for dense_image in dense_images:
        relative_image = dense_image.relative_to(paths.undistorted_images)
        mask_path = paths.undistorted_masks / Path(f"{relative_image}.png")
        expected_masks.add(mask_path.resolve())
        if not mask_path.is_file():
            problems.append(f"Missing fusion mask: {mask_path}")
            continue
        try:
            with Image.open(dense_image) as opened_image, Image.open(mask_path) as opened_mask:
                if opened_image.size != opened_mask.size:
                    problems.append(
                        f"Dimension mismatch for {relative_image.as_posix()}: image "
                        f"{opened_image.size[0]}x{opened_image.size[1]}, mask "
                        f"{opened_mask.size[0]}x{opened_mask.size[1]}"
                    )
                    continue
                mask = opened_mask.convert("L")
                try:
                    colors = mask.getcolors(maxcolors=3)
                    if colors is None or {value for _, value in colors} != {0, 255}:
                        problems.append(f"Mask is not binary and nonempty: {mask_path}")
                        continue
                finally:
                    mask.close()
        except OSError as error:
            problems.append(f"Unreadable fusion mask {mask_path}: {error}")
            continue
        valid_count += 1

    actual_masks = {
        path.resolve()
        for path in paths.undistorted_masks.rglob("*.png")
        if path.is_file()
    }
    extra_masks = sorted(str(path) for path in actual_masks - expected_masks)
    if extra_masks:
        problems.append(
            f"Unexpected fusion masks: {len(extra_masks)}:\n"
            + format_examples(extra_masks)
        )
    if problems:
        raise PipelineError(
            f"Fusion-mask validation failed with {len(problems)} problem(s):\n"
            + format_examples(problems)
        )
    if valid_count != expected_count:
        raise PipelineError(
            f"Validated {valid_count} fusion masks, expected {expected_count}."
        )
    return valid_count


def validate_fusion_readiness(paths: DensePaths) -> int:
    expected_count = validate_patch_match_readiness(paths)
    if not patch_match_is_complete(paths, expected_count, geom_consistency=True):
        counts = patch_match_map_counts(paths)
        raise PipelineError(
            "Geometric PatchMatch output is incomplete. "
            f"Photometric/geometric depth counts: {counts[0]}/{counts[1]}; "
            f"normal counts: {counts[2]}/{counts[3]}."
        )
    validated_masks = validate_fusion_masks(paths, expected_count)
    if validated_masks != expected_count:
        raise PipelineError(
            f"Validated {validated_masks} fusion masks, expected {expected_count}."
        )
    return expected_count


def build_fusion_arguments(paths: DensePaths, settings: FusionSettings) -> list[str]:
    return [
        str(paths.colmap),
        "stereo_fusion",
        "--workspace_path",
        str(paths.workspace),
        "--workspace_format",
        "COLMAP",
        "--input_type",
        settings.input_type,
        "--output_path",
        str(paths.fused_point_cloud),
        "--StereoFusion.mask_path",
        str(paths.undistorted_masks),
        "--StereoFusion.max_image_size",
        str(settings.max_image_size),
        "--StereoFusion.cache_size",
        str(settings.cache_size_gb),
        "--StereoFusion.num_threads",
        str(settings.num_threads),
        "--StereoFusion.min_num_pixels",
        str(settings.min_num_pixels),
    ]


def ply_header_text(path: Path) -> str:
    if not path.is_file():
        raise PipelineError(f"PLY file was not found: {path}")
    try:
        with path.open("rb") as ply_file:
            header_bytes = ply_file.read(1024 * 1024)
    except OSError as error:
        raise PipelineError(f"Could not read PLY file {path}: {error}") from error

    end_marker = b"end_header"
    end_index = header_bytes.find(end_marker)
    if end_index < 0:
        raise PipelineError(f"PLY header is missing end_header: {path}")
    try:
        return header_bytes[: end_index + len(end_marker)].decode("ascii")
    except UnicodeDecodeError as error:
        raise PipelineError(f"PLY header is not valid ASCII: {path}") from error


def ply_element_counts(path: Path) -> dict[str, int]:
    header = ply_header_text(path)

    counts: dict[str, int] = {}
    for line in header.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] == "element":
            try:
                count = int(fields[2])
            except ValueError as error:
                raise PipelineError(
                    f"Invalid PLY element count for '{fields[1]}' in {path}"
                ) from error
            if count < 0:
                raise PipelineError(
                    f"Negative PLY element count for '{fields[1]}' in {path}"
                )
            counts[fields[1]] = count
    if not counts:
        raise PipelineError(f"PLY header contains no element counts: {path}")
    return counts


def ply_vertex_properties(path: Path) -> set[str]:
    properties: set[str] = set()
    in_vertex_element = False
    for line in ply_header_text(path).splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "element":
            in_vertex_element = fields[1] == "vertex"
        elif in_vertex_element and len(fields) >= 3 and fields[0] == "property":
            properties.add(fields[-1])
    return properties


def ply_vertex_count(path: Path) -> int:
    count = ply_element_counts(path).get("vertex")
    if count is None:
        raise PipelineError(f"PLY header contains no vertex count: {path}")
    if count <= 0:
        raise PipelineError(f"PLY contains no vertices: {path}")
    return count


def ply_mesh_counts(path: Path) -> tuple[int, int]:
    counts = ply_element_counts(path)
    vertices = counts.get("vertex")
    faces = counts.get("face")
    if vertices is None or vertices <= 0:
        raise PipelineError(f"Mesh PLY contains no vertices: {path}")
    if faces is None or faces <= 0:
        raise PipelineError(f"Mesh PLY contains no faces: {path}")
    return vertices, faces


def validate_poisson_mesh(path: Path, input_point_count: int) -> tuple[int, int]:
    vertices, faces = ply_mesh_counts(path)
    minimum_count = max(10, int(input_point_count * 0.001))
    if vertices < minimum_count or faces < minimum_count:
        raise PipelineError(
            "Poisson meshing produced a structurally valid but implausibly small "
            f"surface ({vertices} vertices, {faces} faces) from {input_point_count} "
            f"input points. Expected at least {minimum_count} of each. The trim "
            "threshold likely removed most of the surface."
        )
    return vertices, faces


def run_stereo_fusion(
    paths: DensePaths, settings: FusionSettings, resume: bool
) -> FusionReport:
    if paths.fused_point_cloud.exists():
        if resume:
            return FusionReport(
                point_count=ply_vertex_count(paths.fused_point_cloud),
                elapsed_seconds=0.0,
                resumed=True,
            )
        raise PipelineError(
            "Fused point cloud already exists while --no-resume is selected: "
            f"{paths.fused_point_cloud}"
        )

    paths.fused_point_cloud.parent.mkdir(parents=True, exist_ok=True)
    elapsed = run_colmap_logged(
        build_fusion_arguments(paths, settings),
        "Masked stereo fusion",
        paths.logs / "04_stereo_fusion.log",
    )
    point_count = ply_vertex_count(paths.fused_point_cloud)
    return FusionReport(
        point_count=point_count,
        elapsed_seconds=elapsed,
        resumed=False,
    )


def validate_poisson_readiness(paths: DensePaths) -> int:
    if not paths.colmap.is_file():
        raise PipelineError(f"COLMAP executable was not found: {paths.colmap}")
    point_count = ply_vertex_count(paths.fused_point_cloud)
    required_properties = {"x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"}
    missing_properties = sorted(
        required_properties - ply_vertex_properties(paths.fused_point_cloud)
    )
    if missing_properties:
        raise PipelineError(
            "The fused point cloud lacks properties required for colored Poisson "
            f"meshing: {', '.join(missing_properties)}"
        )
    return point_count


def build_poisson_meshing_arguments(
    paths: DensePaths, settings: PoissonMeshingSettings
) -> list[str]:
    return [
        str(paths.colmap),
        "poisson_mesher",
        "--input_path",
        str(paths.fused_point_cloud),
        "--output_path",
        str(paths.poisson_mesh),
        "--PoissonMeshing.point_weight",
        str(settings.point_weight),
        "--PoissonMeshing.depth",
        str(settings.depth),
        "--PoissonMeshing.color",
        "1" if settings.color else "0",
        "--PoissonMeshing.trim",
        str(settings.trim),
        "--PoissonMeshing.num_threads",
        str(settings.num_threads),
    ]


def run_poisson_meshing(
    paths: DensePaths, settings: PoissonMeshingSettings, resume: bool
) -> MeshReport:
    input_point_count = ply_vertex_count(paths.fused_point_cloud)
    if paths.poisson_mesh.exists():
        if resume:
            vertices, faces = validate_poisson_mesh(
                paths.poisson_mesh, input_point_count
            )
            return MeshReport(vertices, faces, elapsed_seconds=0.0, resumed=True)
        raise PipelineError(
            "Poisson mesh already exists while --no-resume is selected: "
            f"{paths.poisson_mesh}"
        )

    paths.poisson_mesh.parent.mkdir(parents=True, exist_ok=True)
    elapsed = run_colmap_logged(
        build_poisson_meshing_arguments(paths, settings),
        "Poisson surface meshing",
        paths.logs / "05_poisson_meshing.log",
    )
    vertices, faces = validate_poisson_mesh(paths.poisson_mesh, input_point_count)
    return MeshReport(vertices, faces, elapsed_seconds=elapsed, resumed=False)


def fusion_visibility_path(paths: DensePaths) -> Path:
    return Path(f"{paths.fused_point_cloud}.vis")


def validate_delaunay_readiness(paths: DensePaths) -> int:
    if not paths.colmap.is_file():
        raise PipelineError(f"COLMAP executable was not found: {paths.colmap}")
    validate_sparse_components(paths.workspace / "sparse")
    point_count = ply_vertex_count(paths.fused_point_cloud)
    visibility_path = fusion_visibility_path(paths)
    if not visibility_path.is_file() or visibility_path.stat().st_size == 0:
        raise PipelineError(
            "Dense Delaunay meshing requires the stereo-fusion visibility file: "
            f"{visibility_path}"
        )
    return point_count


def prepare_delaunay_workspace(paths: DensePaths) -> None:
    sources = (
        paths.fused_point_cloud,
        fusion_visibility_path(paths),
    )
    destinations = (
        paths.workspace / "fused.ply",
        paths.workspace / "fused.ply.vis",
    )
    for source, destination in zip(sources, destinations):
        if destination.exists():
            try:
                same_file = source.samefile(destination)
            except OSError:
                same_file = False
            source_stat = source.stat()
            destination_stat = destination.stat()
            equivalent_copy = (
                source_stat.st_size == destination_stat.st_size
                and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
            )
            if same_file or equivalent_copy:
                continue
            raise PipelineError(
                "Delaunay workspace file already exists but does not match the "
                f"fusion artifact: {destination}"
            )
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def build_delaunay_meshing_arguments(
    paths: DensePaths, settings: DelaunayMeshingSettings
) -> list[str]:
    return [
        str(paths.colmap),
        "delaunay_mesher",
        "--input_path",
        str(paths.workspace),
        "--input_type",
        settings.input_type,
        "--output_path",
        str(paths.delaunay_mesh),
        "--DelaunayMeshing.max_proj_dist",
        str(settings.max_proj_dist),
        "--DelaunayMeshing.max_depth_dist",
        str(settings.max_depth_dist),
        "--DelaunayMeshing.visibility_sigma",
        str(settings.visibility_sigma),
        "--DelaunayMeshing.distance_sigma_factor",
        str(settings.distance_sigma_factor),
        "--DelaunayMeshing.quality_regularization",
        str(settings.quality_regularization),
        "--DelaunayMeshing.max_side_length_factor",
        str(settings.max_side_length_factor),
        "--DelaunayMeshing.max_side_length_percentile",
        str(settings.max_side_length_percentile),
        "--DelaunayMeshing.num_threads",
        str(settings.num_threads),
    ]


def validate_delaunay_mesh(path: Path, input_point_count: int) -> tuple[int, int]:
    vertices, faces = ply_mesh_counts(path)
    minimum_count = max(10, int(input_point_count * 0.001))
    if vertices < minimum_count or faces < minimum_count:
        raise PipelineError(
            "Delaunay meshing produced an implausibly small surface "
            f"({vertices} vertices, {faces} faces) from {input_point_count} input "
            f"points. Expected at least {minimum_count} of each."
        )
    return vertices, faces


def run_delaunay_meshing(
    paths: DensePaths, settings: DelaunayMeshingSettings, resume: bool
) -> MeshReport:
    input_point_count = validate_delaunay_readiness(paths)
    if paths.delaunay_mesh.exists():
        if resume:
            vertices, faces = validate_delaunay_mesh(
                paths.delaunay_mesh, input_point_count
            )
            return MeshReport(vertices, faces, elapsed_seconds=0.0, resumed=True)
        raise PipelineError(
            "Delaunay mesh already exists while --no-resume is selected: "
            f"{paths.delaunay_mesh}"
        )

    prepare_delaunay_workspace(paths)
    paths.delaunay_mesh.parent.mkdir(parents=True, exist_ok=True)
    elapsed = run_colmap_logged(
        build_delaunay_meshing_arguments(paths, settings),
        "Dense Delaunay surface meshing",
        paths.logs / "06_delaunay_meshing.log",
    )
    vertices, faces = validate_delaunay_mesh(
        paths.delaunay_mesh, input_point_count
    )
    return MeshReport(vertices, faces, elapsed_seconds=elapsed, resumed=False)


def validate_simplification_readiness(paths: DensePaths) -> tuple[int, int]:
    input_point_count = validate_poisson_readiness(paths)
    return validate_poisson_mesh(paths.poisson_mesh, input_point_count)


def build_mesh_simplification_arguments(
    paths: DensePaths, settings: MeshSimplificationSettings
) -> list[str]:
    return [
        str(paths.colmap),
        "mesh_simplifier",
        "--input_path",
        str(paths.poisson_mesh),
        "--output_path",
        str(paths.simplified_poisson_mesh),
        "--MeshSimplification.target_face_ratio",
        str(settings.target_face_ratio),
        "--MeshSimplification.max_error",
        str(settings.max_error),
        "--MeshSimplification.boundary_weight",
        str(settings.boundary_weight),
        "--MeshSimplification.interpolate_colors",
        "1" if settings.interpolate_colors else "0",
        "--MeshSimplification.num_threads",
        str(settings.num_threads),
    ]


def validate_simplified_mesh(
    path: Path, input_face_count: int
) -> tuple[int, int]:
    vertices, faces = ply_mesh_counts(path)
    if faces >= input_face_count:
        raise PipelineError(
            "Mesh simplification did not reduce the face count: "
            f"{faces} output faces versus {input_face_count} input faces."
        )
    return vertices, faces


def run_mesh_simplification(
    paths: DensePaths, settings: MeshSimplificationSettings, resume: bool
) -> SimplificationReport:
    _, input_faces = validate_simplification_readiness(paths)
    if paths.simplified_poisson_mesh.exists():
        if resume:
            vertices, faces = validate_simplified_mesh(
                paths.simplified_poisson_mesh, input_faces
            )
            return SimplificationReport(
                vertices, faces, input_faces, elapsed_seconds=0.0, resumed=True
            )
        raise PipelineError(
            "Simplified Poisson mesh already exists while --no-resume is selected: "
            f"{paths.simplified_poisson_mesh}"
        )

    paths.simplified_poisson_mesh.parent.mkdir(parents=True, exist_ok=True)
    elapsed = run_colmap_logged(
        build_mesh_simplification_arguments(paths, settings),
        "Poisson mesh simplification",
        paths.logs / "07_mesh_simplification.log",
    )
    vertices, faces = validate_simplified_mesh(
        paths.simplified_poisson_mesh, input_faces
    )
    return SimplificationReport(
        vertices, faces, input_faces, elapsed_seconds=elapsed, resumed=False
    )


def selected_stages(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return PIPELINE_STAGES
    return (stage,)


def main() -> int:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = load_yaml(config_path)
    validate_config_schema(config)
    paths = resolve_paths(config_path, config)

    project_config = section(config, "project")
    runtime_config = section(config, "runtime")
    resume = (
        optional_bool(runtime_config, "resume", "runtime", True)
        if args.resume is None
        else args.resume
    )
    stages = selected_stages(args.stage)

    print(f"Project: {required_string(project_config, 'name', 'project')}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Configuration: {paths.config}")
    print(f"Sparse model: {paths.sparse_model}")
    print(f"Dense workspace: {paths.workspace}")
    print(f"Stages: {', '.join(stages)}")
    print(f"Resume: {resume}")

    if args.dry_run:
        if args.stage == "patch-match":
            expected_count = validate_patch_match_readiness(paths)
            print(f"PatchMatch-ready undistorted images: {expected_count}")
            print(
                "\n[PatchMatch command]\n"
                + display_command(
                    build_patch_match_arguments(paths, patch_match_settings(config))
                )
            )
        elif args.stage == "fusion":
            expected_count = validate_fusion_readiness(paths)
            print(f"Fusion-ready geometric depth maps and masks: {expected_count}")
            print(
                "\n[Stereo fusion command]\n"
                + display_command(
                    build_fusion_arguments(paths, fusion_settings(config))
                )
            )
        elif args.stage == "mesh":
            point_count = validate_poisson_readiness(paths)
            print(f"Poisson-ready fused points: {point_count}")
            print(
                "\n[Poisson meshing command]\n"
                + display_command(
                    build_poisson_meshing_arguments(
                        paths, poisson_meshing_settings(config)
                    )
                )
            )
        elif args.stage == "delaunay-mesh":
            point_count = validate_delaunay_readiness(paths)
            print(f"Delaunay-ready fused points: {point_count}")
            print(
                "\n[Delaunay meshing command]\n"
                + display_command(
                    build_delaunay_meshing_arguments(
                        paths, delaunay_meshing_settings(config)
                    )
                )
            )
            print(
                "The execution run will prepare workspace-root hard links for "
                "fused.ply and fused.ply.vis."
            )
        elif args.stage == "simplify-mesh":
            input_vertices, input_faces = validate_simplification_readiness(paths)
            settings = mesh_simplification_settings(config)
            print(
                f"Simplification-ready Poisson mesh: {input_vertices} vertices, "
                f"{input_faces} faces"
            )
            print(
                "Expected target faces: "
                f"approximately {int(input_faces * settings.target_face_ratio)}"
            )
            print(
                "\n[Mesh simplification command]\n"
                + display_command(
                    build_mesh_simplification_arguments(paths, settings)
                )
            )
        print("\nDry run complete. No files were created and COLMAP was not run.")
        return 0

    if args.stage == "patch-match":
        print("\n[Validate PatchMatch workspace]")
        expected_count = validate_patch_match_readiness(paths)
        print(f"Undistorted images ready: {expected_count}")
        settings = patch_match_settings(config)
        patch_match = run_patch_match(
            paths,
            settings,
            expected_count=expected_count,
            resume=resume,
        )
        if patch_match.resumed:
            print("PatchMatch already complete; reused the validated map set.")
        else:
            print(f"PatchMatch runtime: {patch_match.elapsed_seconds / 3600:.2f} hours")
        print(f"Photometric depth maps: {patch_match.photometric_depth_maps}")
        print(f"Geometric depth maps: {patch_match.geometric_depth_maps}")
        print(f"Photometric normal maps: {patch_match.photometric_normal_maps}")
        print(f"Geometric normal maps: {patch_match.geometric_normal_maps}")
        print("\nPatchMatch stereo complete.")
        return 0

    if args.stage == "fusion":
        print("\n[Validate stereo-fusion inputs]")
        expected_count = validate_fusion_readiness(paths)
        print(f"Geometric depth-map sets ready: {expected_count}")
        print(f"Aligned binary fusion masks ready: {expected_count}")
        report = run_stereo_fusion(
            paths,
            fusion_settings(config),
            resume=resume,
        )
        if report.resumed:
            print("Stereo fusion already complete; reused the validated point cloud.")
        else:
            print(f"Stereo-fusion runtime: {report.elapsed_seconds / 60:.2f} minutes")
        print(f"Fused points: {report.point_count}")
        print(f"Fused point cloud: {paths.fused_point_cloud}")
        print("\nMasked stereo fusion complete.")
        return 0

    if args.stage == "mesh":
        print("\n[Validate Poisson-meshing input]")
        point_count = validate_poisson_readiness(paths)
        print(f"Fused points ready: {point_count}")
        report = run_poisson_meshing(
            paths,
            poisson_meshing_settings(config),
            resume=resume,
        )
        if report.resumed:
            print("Poisson meshing already complete; reused the validated mesh.")
        else:
            print(f"Poisson-meshing runtime: {report.elapsed_seconds / 60:.2f} minutes")
        print(f"Mesh vertices: {report.vertex_count}")
        print(f"Mesh faces: {report.face_count}")
        print(f"Poisson mesh: {paths.poisson_mesh}")
        print("\nPoisson surface meshing complete.")
        return 0

    if args.stage == "delaunay-mesh":
        print("\n[Validate dense Delaunay-meshing input]")
        point_count = validate_delaunay_readiness(paths)
        print(f"Fused points and visibility records ready: {point_count}")
        report = run_delaunay_meshing(
            paths,
            delaunay_meshing_settings(config),
            resume=resume,
        )
        if report.resumed:
            print("Delaunay meshing already complete; reused the validated mesh.")
        else:
            print(f"Delaunay-meshing runtime: {report.elapsed_seconds / 60:.2f} minutes")
        print(f"Mesh vertices: {report.vertex_count}")
        print(f"Mesh faces: {report.face_count}")
        print(f"Delaunay mesh: {paths.delaunay_mesh}")
        print("\nDense Delaunay surface meshing complete.")
        return 0

    if args.stage == "simplify-mesh":
        print("\n[Validate Poisson mesh for simplification]")
        input_vertices, input_faces = validate_simplification_readiness(paths)
        print(f"Input mesh vertices: {input_vertices}")
        print(f"Input mesh faces: {input_faces}")
        report = run_mesh_simplification(
            paths,
            mesh_simplification_settings(config),
            resume=resume,
        )
        if report.resumed:
            print("Mesh simplification already complete; reused the validated mesh.")
        else:
            print(
                f"Mesh-simplification runtime: {report.elapsed_seconds / 60:.2f} minutes"
            )
        achieved_ratio = report.face_count / report.input_face_count
        print(f"Simplified vertices: {report.vertex_count}")
        print(f"Simplified faces: {report.face_count}")
        print(f"Achieved face ratio: {achieved_ratio:.4f}")
        print(f"Simplified Poisson mesh: {paths.simplified_poisson_mesh}")
        print("\nPoisson mesh simplification complete.")
        return 0

    if args.stage in {"validate", "prepare", "undistort"}:
        print("\n[Validate dense-reconstruction inputs]")
        report = validate_inputs(paths)
        resolution_text = ", ".join(
            f"{width}x{height}" for width, height in report.resolutions
        )
        print(f"RGB images checked: {report.image_count}")
        print(f"PNG masks present: {report.mask_count}")
        print(f"Registered sparse images: {report.registered_image_count}")
        print(f"RGB resolutions: {resolution_text}")
        print("All RGB images and masks are readable, nonempty, and dimension-aligned.")
        print("All source image names exactly match the selected sparse model.")
        if args.stage == "validate":
            print("\nValidation complete. No dense reconstruction outputs were created.")
            return 0

        if args.stage == "prepare":
            print("\n[Prepare masked RGB and mask-undistortion inputs]")
            preparation = prepare_masked_inputs(
                paths,
                mask_preparation_settings(config),
                resume=resume,
            )
            print(f"Prepared image pairs: {preparation.total}")
            print(f"Newly written pairs: {preparation.written}")
            print(f"Resumed valid pairs: {preparation.resumed}")
            print(f"Masked RGB directory: {paths.masked_images}")
            print(f"Mask-undistortion directory: {paths.mask_undistortion_images}")
            print(
                "\nPreparation complete. Original RGB images and masks were not modified."
            )
            return 0

        print("\n[Validate prepared inputs]")
        prepared_count = validate_prepared_inputs(paths)
        print(f"Prepared image pairs ready: {prepared_count}")
        undistortion = run_rgb_undistortion(
            paths,
            undistortion_settings(config),
            expected_count=prepared_count,
            resume=resume,
        )
        if undistortion.resumed:
            print("Undistortion already complete; reused the validated workspace.")
        else:
            print(f"Undistorted images: {undistortion.image_count}")
            print(
                f"Undistortion runtime: {undistortion.elapsed_seconds / 60:.2f} minutes"
            )
        print(f"Dense COLMAP workspace: {paths.workspace}")
        print("\nMasked RGB undistortion complete.")
        return 0

    raise PipelineError(
        "Only the 'validate', 'prepare', 'undistort', 'patch-match', 'fusion', "
        "'mesh', 'delaunay-mesh', and 'simplify-mesh' execution stages are enabled "
        "so far. Select one of those stages, or use --dry-run while texturing is "
        "implemented."
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except PipelineError as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
