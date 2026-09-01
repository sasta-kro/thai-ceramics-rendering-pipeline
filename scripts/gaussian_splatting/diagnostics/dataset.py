#!/usr/bin/env python3
"""Validate the existing undistorted masked COLMAP dataset for 3DGS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..core.common import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    PipelineError,
    ValidationReport,
    expected_mask_path,
    format_examples,
    image_files,
    load_yaml,
    registered_image_names,
    resolve_config_path,
    resolve_paths,
    validate_sparse_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the undistorted RGB images, object masks, and COLMAP sparse "
            "model without preparing images or launching 3DGS training."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML configuration (default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)}).",
    )
    return parser.parse_args()


def validate_dataset(config_path: Path) -> ValidationReport:
    try:
        from PIL import Image, UnidentifiedImageError
    except ModuleNotFoundError as error:
        raise PipelineError(
            "Pillow is required for dataset validation. Update the pot-masking "
            "environment from environment-masking.yml."
        ) from error

    config_path = resolve_config_path(config_path)
    config = load_yaml(config_path)
    paths = resolve_paths(config_path, config)

    if not paths.colmap.is_file():
        raise PipelineError(f"COLMAP executable was not found: {paths.colmap}")
    if not paths.images.is_dir():
        raise PipelineError(f"Undistorted image directory was not found: {paths.images}")
    if not paths.masks.is_dir():
        raise PipelineError(f"Undistorted mask directory was not found: {paths.masks}")

    sparse_format = validate_sparse_components(paths.sparse_model)
    images = image_files(paths.images)
    if not images:
        raise PipelineError(f"No supported images were found under: {paths.images}")

    missing_masks: list[str] = []
    unreadable_pairs: list[str] = []
    dimension_mismatches: list[str] = []
    empty_masks: list[str] = []
    resolutions: set[tuple[int, int]] = set()

    for image_path in images:
        relative_name = image_path.relative_to(paths.images).as_posix()
        mask_path = expected_mask_path(image_path, paths.images, paths.masks)
        if not mask_path.is_file():
            missing_masks.append(relative_name)
            continue
        try:
            with Image.open(image_path) as rgb_image, Image.open(mask_path) as mask_image:
                rgb_size = rgb_image.size
                mask_size = mask_image.size
                resolutions.add(rgb_size)
                if rgb_size != mask_size:
                    dimension_mismatches.append(
                        f"{relative_name}: RGB {rgb_size[0]}x{rgb_size[1]}, "
                        f"mask {mask_size[0]}x{mask_size[1]}"
                    )
                grayscale_mask = mask_image.convert("L")
                try:
                    if grayscale_mask.getbbox() is None:
                        empty_masks.append(relative_name)
                finally:
                    grayscale_mask.close()
        except (OSError, UnidentifiedImageError) as error:
            unreadable_pairs.append(f"{relative_name}: {error}")

    problems: list[str] = []
    if missing_masks:
        problems.append(
            f"Missing masks for {len(missing_masks)} image(s):\n"
            + format_examples(missing_masks)
        )
    if unreadable_pairs:
        problems.append(
            f"Unreadable image/mask pairs: {len(unreadable_pairs)}:\n"
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

    registered_names = registered_image_names(paths.colmap, paths.sparse_model)
    source_names = {
        path.relative_to(paths.images).as_posix() for path in images
    }
    sparse_names = {name.replace("\\", "/") for name in registered_names}
    missing_images = sorted(sparse_names - source_names)
    unregistered_images = sorted(source_names - sparse_names)
    if missing_images or unregistered_images:
        details: list[str] = []
        if missing_images:
            details.append(
                "Registered images missing from the image directory:\n"
                + format_examples(missing_images)
            )
        if unregistered_images:
            details.append(
                "Images not registered in the selected model:\n"
                + format_examples(unregistered_images)
            )
        raise PipelineError("\n\n".join(details))

    mask_count = sum(
        1
        for path in paths.masks.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    )
    return ValidationReport(
        image_count=len(images),
        mask_count=mask_count,
        registered_image_count=len(registered_names),
        resolutions=tuple(sorted(resolutions)),
        sparse_format=sparse_format,
    )


def main() -> int:
    args = parse_args()
    try:
        report = validate_dataset(args.config)
    except PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    resolutions = ", ".join(f"{width}x{height}" for width, height in report.resolutions)
    print("3DGS dataset validation passed")
    print(f"Images: {report.image_count}")
    print(f"Masks: {report.mask_count}")
    print(f"Registered COLMAP images: {report.registered_image_count}")
    print(f"Resolution(s): {resolutions}")
    print(f"Sparse model format: {report.sparse_format}")
    print("No images were prepared and no training was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
