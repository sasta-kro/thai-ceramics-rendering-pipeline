#!/usr/bin/env python3
"""Create and validate binary COLMAP fusion masks after image undistortion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DENSE_WORKSPACE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pot1-unglazed_every6"
    / "colmap_dense_masked_sequential"
)
DEFAULT_SOURCE = DENSE_WORKSPACE / "mask_undistortion_workspace" / "images"
DEFAULT_DENSE_IMAGES = DENSE_WORKSPACE / "images"
DEFAULT_OUTPUT = DENSE_WORKSPACE / "masks"
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class MaskPreparationError(RuntimeError):
    """A user-correctable fusion-mask preparation error."""


@dataclass(frozen=True)
class MaskReport:
    """Summary of conversion and validation results."""

    expected: int
    written: int
    resumed: int
    validated: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Threshold COLMAP-undistorted mask images, optionally erode the object "
            "boundary, save <image filename>.png fusion masks, and validate them "
            "against the undistorted dense RGB images."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=(
            "Undistorted black-and-white image directory produced by COLMAP "
            f"(default: {DEFAULT_SOURCE.relative_to(PROJECT_ROOT)})."
        ),
    )
    parser.add_argument(
        "--dense-images",
        type=Path,
        default=DEFAULT_DENSE_IMAGES,
        help=(
            "Undistorted dense RGB directory used for filename and dimension checks "
            f"(default: {DEFAULT_DENSE_IMAGES.relative_to(PROJECT_ROOT)})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Destination for binary fusion masks "
            f"(default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Values at or above this threshold become foreground (default: 128).",
    )
    parser.add_argument(
        "--erosion-pixels",
        type=int,
        default=1,
        help="Object-boundary erosion radius in pixels (default: 1).",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Reuse masks that already pass validation (default).",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Refuse to run if any expected output mask already exists.",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args()


def resolve_project_path(raw_path: Path, label: str) -> Path:
    expanded = Path(os.path.expandvars(str(raw_path))).expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    resolved = expanded.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise MaskPreparationError(
            f"'{label}' must stay inside the project directory: {resolved}"
        ) from error
    return resolved


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def fusion_mask_path(dense_image: Path, dense_root: Path, output_root: Path) -> Path:
    relative_image = dense_image.relative_to(dense_root)
    return output_root / Path(f"{relative_image}.png")


def source_mask_image_path(
    dense_image: Path, dense_root: Path, source_root: Path
) -> Path:
    return source_root / dense_image.relative_to(dense_root)


def format_examples(values: Iterable[str], limit: int = 10) -> str:
    examples = list(values)
    output = "\n".join(f"  {value}" for value in examples[:limit])
    if len(examples) > limit:
        output += f"\n  ...and {len(examples) - limit} more"
    return output


def mask_is_valid(mask_path: Path, expected_size: tuple[int, int]) -> bool:
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise MaskPreparationError(
            "Pillow is required. Update the pot-masking environment from "
            "environment-masking.yml."
        ) from error

    try:
        with Image.open(mask_path) as opened_mask:
            if opened_mask.size != expected_size:
                return False
            mask = opened_mask.convert("L")
            try:
                colors = mask.getcolors(maxcolors=3)
                if colors is None:
                    return False
                values = {value for _, value in colors}
                return values == {0, 255}
            finally:
                mask.close()
    except OSError:
        return False


def write_binary_mask(
    source_path: Path,
    output_path: Path,
    expected_size: tuple[int, int],
    threshold: int,
    erosion_pixels: int,
) -> None:
    try:
        from PIL import Image, ImageFilter, UnidentifiedImageError
    except ModuleNotFoundError as error:
        raise MaskPreparationError(
            "Pillow is required. Update the pot-masking environment from "
            "environment-masking.yml."
        ) from error

    binary_mask = None
    eroded_mask = None
    try:
        with Image.open(source_path) as source_image:
            if source_image.size != expected_size:
                raise MaskPreparationError(
                    f"Undistorted mask size mismatch for {source_path.name}: "
                    f"expected {expected_size[0]}x{expected_size[1]}, found "
                    f"{source_image.size[0]}x{source_image.size[1]}."
                )
            grayscale = source_image.convert("L")
            try:
                binary_mask = grayscale.point(
                    lambda value: 255 if value >= threshold else 0,
                    mode="L",
                )
            finally:
                grayscale.close()

        if erosion_pixels > 0:
            kernel_size = erosion_pixels * 2 + 1
            eroded_mask = binary_mask.filter(ImageFilter.MinFilter(kernel_size))
        else:
            eroded_mask = binary_mask.copy()

        colors = eroded_mask.getcolors(maxcolors=3)
        if colors is None or {value for _, value in colors} != {0, 255}:
            raise MaskPreparationError(
                f"Converted mask is empty, full-frame, or non-binary: {source_path}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        try:
            eroded_mask.save(temporary_path, format="PNG")
            temporary_path.replace(output_path)
        except OSError as error:
            temporary_path.unlink(missing_ok=True)
            raise MaskPreparationError(
                f"Could not write fusion mask {output_path}: {error}"
            ) from error
    except (OSError, UnidentifiedImageError) as error:
        raise MaskPreparationError(
            f"Could not read undistorted mask image {source_path}: {error}"
        ) from error
    finally:
        if binary_mask is not None:
            binary_mask.close()
        if eroded_mask is not None:
            eroded_mask.close()


def convert_masks(
    source_root: Path,
    dense_root: Path,
    output_root: Path,
    threshold: int,
    erosion_pixels: int,
    resume: bool,
) -> tuple[int, int, int]:
    dense_images = image_files(dense_root)
    if not dense_images:
        raise MaskPreparationError(
            f"No undistorted dense RGB images were found: {dense_root}"
        )
    if not source_root.is_dir():
        raise MaskPreparationError(
            "Undistorted mask-image directory was not found. Run COLMAP "
            f"image_undistorter first: {source_root}"
        )

    missing_sources: list[str] = []
    for dense_image in dense_images:
        source_path = source_mask_image_path(dense_image, dense_root, source_root)
        if not source_path.is_file():
            missing_sources.append(source_path.relative_to(source_root).as_posix())
    if missing_sources:
        raise MaskPreparationError(
            f"Missing undistorted mask images: {len(missing_sources)}:\n"
            + format_examples(missing_sources)
        )

    if not resume:
        existing = [
            fusion_mask_path(image, dense_root, output_root)
            for image in dense_images
            if fusion_mask_path(image, dense_root, output_root).exists()
        ]
        if existing:
            raise MaskPreparationError(
                "Fusion masks already exist while --no-resume is selected:\n"
                + format_examples(str(path) for path in existing)
            )

    output_root.mkdir(parents=True, exist_ok=True)
    written = 0
    resumed = 0
    for index, dense_image in enumerate(dense_images, start=1):
        try:
            from PIL import Image
            with Image.open(dense_image) as opened_dense:
                expected_size = opened_dense.size
        except (ModuleNotFoundError, OSError) as error:
            raise MaskPreparationError(
                f"Could not inspect dense RGB image {dense_image}: {error}"
            ) from error

        source_path = source_mask_image_path(dense_image, dense_root, source_root)
        output_path = fusion_mask_path(dense_image, dense_root, output_root)
        if resume and mask_is_valid(output_path, expected_size):
            resumed += 1
            continue

        write_binary_mask(
            source_path,
            output_path,
            expected_size,
            threshold,
            erosion_pixels,
        )
        written += 1
        if index == 1 or index % 25 == 0 or index == len(dense_images):
            print(f"Created {index}/{len(dense_images)} fusion masks", flush=True)

    return len(dense_images), written, resumed


def validate_masks(dense_root: Path, output_root: Path) -> int:
    dense_images = image_files(dense_root)
    expected_paths = {
        fusion_mask_path(image, dense_root, output_root).resolve()
        for image in dense_images
    }
    actual_paths = {
        path.resolve()
        for path in output_root.rglob("*.png")
        if path.is_file()
    }

    problems: list[str] = []
    missing = sorted(str(path) for path in expected_paths - actual_paths)
    extra = sorted(str(path) for path in actual_paths - expected_paths)
    if missing:
        problems.append(
            f"Missing fusion masks: {len(missing)}:\n" + format_examples(missing)
        )
    if extra:
        problems.append(
            f"Unexpected fusion masks: {len(extra)}:\n" + format_examples(extra)
        )

    validated = 0
    for dense_image in dense_images:
        output_path = fusion_mask_path(dense_image, dense_root, output_root)
        try:
            from PIL import Image
            with Image.open(dense_image) as opened_dense:
                expected_size = opened_dense.size
        except (ModuleNotFoundError, OSError) as error:
            problems.append(f"Unreadable dense image {dense_image}: {error}")
            continue
        if output_path.is_file() and mask_is_valid(output_path, expected_size):
            validated += 1
        elif output_path.is_file():
            problems.append(
                f"Invalid binary mask or dimension mismatch: {output_path}"
            )

    if problems:
        raise MaskPreparationError("\n\n".join(problems))
    if validated != len(dense_images):
        raise MaskPreparationError(
            f"Validated {validated} masks, expected {len(dense_images)}."
        )
    return validated


def main() -> int:
    args = parse_args()
    source_root = resolve_project_path(args.source, "source")
    dense_root = resolve_project_path(args.dense_images, "dense-images")
    output_root = resolve_project_path(args.output, "output")

    if not 1 <= args.threshold <= 255:
        raise MaskPreparationError("--threshold must be between 1 and 255.")
    if args.erosion_pixels < 0:
        raise MaskPreparationError("--erosion-pixels must be zero or greater.")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Undistorted mask images: {source_root}")
    print(f"Dense RGB images: {dense_root}")
    print(f"Fusion-mask output: {output_root}")
    print(f"Threshold: {args.threshold}")
    print(f"Erosion: {args.erosion_pixels} pixel(s)")
    print(f"Resume: {args.resume}")

    expected, written, resumed = convert_masks(
        source_root,
        dense_root,
        output_root,
        args.threshold,
        args.erosion_pixels,
        args.resume,
    )
    validated = validate_masks(dense_root, output_root)
    report = MaskReport(
        expected=expected,
        written=written,
        resumed=resumed,
        validated=validated,
    )

    print(f"\nExpected masks: {report.expected}")
    print(f"Newly written masks: {report.written}")
    print(f"Resumed valid masks: {report.resumed}")
    print(f"Validated masks: {report.validated}")
    print("Fusion-mask preparation and validation complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except MaskPreparationError as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
