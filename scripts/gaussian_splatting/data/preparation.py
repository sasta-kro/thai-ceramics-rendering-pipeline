#!/usr/bin/env python3
"""Prepare mask-aware downscaled caches for user-run 3DGS training."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from ..core.common import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    GaussianSplattingPaths,
    PipelineError,
    expected_mask_path,
    image_files,
    load_yaml,
    preparation_factors,
    profile,
    required_positive_int,
    required_positive_number,
    required_string,
    resolve_config_path,
    resolve_paths,
    section,
)
from ..diagnostics.dataset import validate_dataset


@dataclass(frozen=True)
class FactorPreparationReport:
    factor: int
    source_count: int
    written_count: int
    resumed_count: int
    source_resolution: tuple[int, int]
    target_resolution: tuple[int, int]
    cache_directory: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare alpha-aware PNG image/mask caches. This command does not "
            "launch gsplat training."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--profile",
        help="Prepare the image factor required by one configured training profile.",
    )
    selection.add_argument(
        "--all-profiles",
        action="store_true",
        help="Prepare every unique image factor used by the configured profiles.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show planned outputs without creating cache files.",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def target_relative_path(source_relative_path: Path) -> Path:
    return source_relative_path.with_suffix(".png")


def target_size(source_size: tuple[int, int], factor: int) -> tuple[int, int]:
    if factor <= 0:
        raise PipelineError("Image preparation factor must be positive.")
    return tuple(max(1, round(dimension / factor)) for dimension in source_size)


def resampling_filter(name: str) -> int:
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise PipelineError("Pillow is required for image preparation.") from error

    filters = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    selected = filters.get(name.strip().lower())
    if selected is None:
        raise PipelineError(
            f"Unsupported resampling filter '{name}'. Choose: {', '.join(filters)}."
        )
    return selected


def alpha_aware_resize(
    source_image: Path,
    source_mask: Path,
    destination_size: tuple[int, int],
    rgb_filter: int,
    mask_filter: int,
    alpha_epsilon: float,
):
    """Resize premultiplied masked RGB and return straight RGB plus soft alpha."""

    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as error:
        raise PipelineError("NumPy and Pillow are required for image preparation.") from error

    with Image.open(source_image) as opened_rgb, Image.open(source_mask) as opened_mask:
        rgb = opened_rgb.convert("RGB")
        alpha = opened_mask.convert("L")
        try:
            if rgb.size != alpha.size:
                raise PipelineError(
                    f"Image/mask dimensions differ for {source_image.name}: "
                    f"RGB {rgb.size}, mask {alpha.size}."
                )
            resized_premultiplied = rgb.resize(destination_size, rgb_filter)
            resized_alpha = alpha.resize(destination_size, mask_filter)
        finally:
            rgb.close()
            alpha.close()

    premultiplied = np.asarray(resized_premultiplied, dtype=np.float32) / 255.0
    alpha_values = np.asarray(resized_alpha, dtype=np.float32) / 255.0
    resized_premultiplied.close()

    straight = np.zeros_like(premultiplied)
    visible = alpha_values > alpha_epsilon
    straight[visible] = premultiplied[visible] / alpha_values[visible, None]
    straight = np.clip(straight, 0.0, 1.0)
    straight_rgb = Image.fromarray(np.rint(straight * 255.0).astype(np.uint8))
    return straight_rgb, resized_alpha


def atomic_save_png(image: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    image.save(temporary, format="PNG", optimize=False)
    temporary.replace(destination)


def atomic_write_json(data: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def prepared_pair_is_valid(
    image_path: Path, mask_path: Path, expected_size: tuple[int, int]
) -> bool:
    try:
        from PIL import Image

        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            return (
                image.format == "PNG"
                and mask.format == "PNG"
                and image.mode == "RGB"
                and mask.mode == "L"
                and image.size == expected_size
                and mask.size == expected_size
                and mask.getbbox() is not None
            )
    except (FileNotFoundError, OSError):
        return False


def holdout_split(image_names: list[str], holdout_every: int) -> dict[str, Any]:
    if holdout_every <= 1:
        raise PipelineError("evaluation.holdout_every must be greater than one.")
    test = [name for index, name in enumerate(image_names) if index % holdout_every == 0]
    train = [name for index, name in enumerate(image_names) if index % holdout_every != 0]
    return {
        "method": "ordered_every_n",
        "holdout_every": holdout_every,
        "train_count": len(train),
        "test_count": len(test),
        "train": train,
        "test": test,
    }


def selected_factors(config: Mapping[str, Any], profile_name: str | None) -> tuple[int, ...]:
    if profile_name is None:
        return preparation_factors(config)
    values = profile(config, profile_name)
    return (
        required_positive_int(values, "image_factor", f"profiles.{profile_name}"),
    )


def prepare_factor(
    paths: GaussianSplattingPaths,
    images: list[Path],
    factor: int,
    settings: Mapping[str, Any],
    resume: bool,
    dry_run: bool,
) -> FactorPreparationReport:
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise PipelineError("Pillow is required for image preparation.") from error

    rgb_filter = resampling_filter(required_string(settings, "rgb_resample", "preparation"))
    mask_filter = resampling_filter(
        required_string(settings, "mask_resample", "preparation")
    )
    alpha_epsilon = required_positive_number(
        settings, "alpha_epsilon", "preparation"
    )
    output_format = required_string(settings, "output_format", "preparation").upper()
    if output_format != "PNG":
        raise PipelineError("preparation.output_format must be PNG.")

    factor_root = paths.cache / f"factor_{factor}"
    output_images = factor_root / "images"
    output_masks = factor_root / "masks"
    entries: list[dict[str, Any]] = []
    written = 0
    resumed = 0
    source_resolution: tuple[int, int] | None = None
    destination_resolution: tuple[int, int] | None = None

    for source_image in images:
        relative_source = source_image.relative_to(paths.images)
        source_mask = expected_mask_path(source_image, paths.images, paths.masks)
        relative_output = target_relative_path(relative_source)
        output_image = output_images / relative_output
        output_mask = output_masks / relative_output

        with Image.open(source_image) as opened:
            current_source_size = opened.size
        current_target_size = target_size(current_source_size, factor)
        if source_resolution is None:
            source_resolution = current_source_size
            destination_resolution = current_target_size
        elif current_source_size != source_resolution:
            raise PipelineError(
                "All source images must have the same resolution for this dataset."
            )

        if resume and prepared_pair_is_valid(
            output_image, output_mask, current_target_size
        ):
            resumed += 1
        elif not dry_run:
            if output_image.exists() or output_mask.exists():
                raise PipelineError(
                    "Existing cache pair is incomplete or invalid; refusing to overwrite: "
                    f"{relative_output.as_posix()}"
                )
            prepared_rgb, prepared_mask = alpha_aware_resize(
                source_image,
                source_mask,
                current_target_size,
                rgb_filter,
                mask_filter,
                alpha_epsilon,
            )
            try:
                atomic_save_png(prepared_rgb, output_image)
                atomic_save_png(prepared_mask, output_mask)
            finally:
                prepared_rgb.close()
                prepared_mask.close()
            written += 1

        entries.append(
            {
                "source_name": relative_source.as_posix(),
                "image": output_image.relative_to(factor_root).as_posix(),
                "mask": output_mask.relative_to(factor_root).as_posix(),
            }
        )

    if source_resolution is None or destination_resolution is None:
        raise PipelineError("No source images were available for preparation.")

    if not dry_run:
        atomic_write_json(
            {
                "schema_version": 1,
                "factor": factor,
                "source_dataset": str(paths.dataset),
                "source_resolution": list(source_resolution),
                "target_resolution": list(destination_resolution),
                "image_count": len(entries),
                "images": entries,
            },
            factor_root / "manifest.json",
        )

    return FactorPreparationReport(
        factor=factor,
        source_count=len(images),
        written_count=written,
        resumed_count=resumed,
        source_resolution=source_resolution,
        target_resolution=destination_resolution,
        cache_directory=str(factor_root),
    )


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_config_path(args.config)
        config = load_yaml(config_path)
        paths = resolve_paths(config_path, config)
        validate_dataset(config_path)
        images = image_files(paths.images)
        factors = selected_factors(config, args.profile)
        preparation = section(config, "preparation")
        evaluation = section(config, "evaluation")
        holdout_every = required_positive_int(
            evaluation, "holdout_every", "evaluation"
        )
        split = holdout_split(
            [path.relative_to(paths.images).as_posix() for path in images],
            holdout_every,
        )

        reports = [
            prepare_factor(
                paths,
                images,
                factor,
                preparation,
                resume=args.resume,
                dry_run=args.dry_run,
            )
            for factor in factors
        ]

        if not args.dry_run:
            atomic_write_json(split, paths.holdout_split)
            atomic_write_json(
                {
                    "schema_version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_dataset": str(paths.dataset),
                    "source_images": str(paths.images),
                    "source_masks": str(paths.masks),
                    "source_sparse_model": str(paths.sparse_model),
                    "image_count": len(images),
                    "prepared_factors": [asdict(report) for report in reports],
                },
                paths.dataset_manifest,
            )
    except PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("3DGS image preparation " + ("dry run" if args.dry_run else "complete"))
    for report in reports:
        print(
            f"Factor {report.factor}: {report.source_resolution[0]}x"
            f"{report.source_resolution[1]} -> {report.target_resolution[0]}x"
            f"{report.target_resolution[1]}, source {report.source_count}, "
            f"written {report.written_count}, resumed {report.resumed_count}"
        )
        print(f"  Cache: {report.cache_directory}")
    print(f"Holdout split: train {split['train_count']}, test {split['test_count']}")
    print("No gsplat training was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
