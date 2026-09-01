#!/usr/bin/env python3
"""Run the masked COLMAP dense-reconstruction pipeline from YAML config."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
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
    optional_bool(
        patch_match_config, "geom_consistency", "patch_match", True
    )
    optional_bool(patch_match_config, "filter", "patch_match", True)

    required_string(fusion_config, "input_type", "fusion")
    optional_positive_int(fusion_config, "max_image_size", "fusion", 1600)
    optional_positive_number(fusion_config, "cache_size_gb", "fusion", 6)
    optional_positive_int(fusion_config, "min_num_pixels", "fusion", 3)

    optional_bool(meshing_config, "run_poisson", "meshing", True)
    optional_bool(meshing_config, "run_delaunay", "meshing", True)
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
        print("\nDry run complete. No files were created and COLMAP was not run.")
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
        "Only the 'validate', 'prepare', and 'undistort' execution stages are "
        "enabled so far. Select one of those stages, or use --dry-run while the "
        "remaining stages are being implemented step by step."
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
