#!/usr/bin/env python3
"""Stage multiple masked capture sequences for one COLMAP reconstruction.

The source RGB images and SAM2 masks are never modified. Each staged filename
is prefixed with its view name so independent videos can safely contain the
same original frame names.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable
import uuid


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
VIEW_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
MANIFEST_FIELDS = (
    "view",
    "view_frame_index",
    "source_filename",
    "combined_filename",
    "source_image",
    "source_mask",
    "width",
    "height",
    "image_sha256",
    "mask_sha256",
)


class DatasetPreparationError(RuntimeError):
    """A user-correctable multiview dataset preparation error."""


@dataclass(frozen=True)
class SourceSpec:
    """One ordered capture sequence and its matching COLMAP masks."""

    name: str
    images: Path
    masks: Path


@dataclass(frozen=True)
class FramePair:
    """Validated source image/mask pair and its collision-free output name."""

    view: str
    view_frame_index: int
    image: Path
    mask: Path
    combined_filename: str
    width: int
    height: int


@dataclass(frozen=True)
class PreparationReport:
    """Summary returned after validation or staging."""

    sources: int
    images: int
    masks: int
    width: int
    height: int
    written: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and non-destructively stage multiple RGB/SAM2-mask "
            "sequences for a combined COLMAP reconstruction."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run every preflight check without creating output files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the generated outputs declared by the configuration.",
    )
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path, label: str) -> Path:
    expanded = Path(os.path.expandvars(str(raw_path))).expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    resolved = expanded.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise DatasetPreparationError(
            f"'{label}' must stay inside the project directory: {resolved}"
        ) from error
    if not relative.parts:
        raise DatasetPreparationError(f"'{label}' cannot be the project root.")
    return resolved


def read_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise DatasetPreparationError(
            "PyYAML is required. Activate the pot-masking environment."
        ) from error

    path = resolve_project_path(config_path, "config")
    if not path.is_file():
        raise DatasetPreparationError(f"Configuration file not found: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DatasetPreparationError(f"Could not read configuration {path}: {error}") from error
    if not isinstance(document, dict):
        raise DatasetPreparationError("Configuration root must be a mapping.")
    return document


def config_paths(
    document: dict[str, Any],
) -> tuple[str, list[SourceSpec], Path, Path, Path]:
    project = document.get("project")
    project_name = project.get("name") if isinstance(project, dict) else None
    if not isinstance(project_name, str) or not project_name.strip():
        raise DatasetPreparationError("project.name must be a non-empty string.")

    raw_sources = document.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise DatasetPreparationError("sources must be a non-empty list.")

    sources: list[SourceSpec] = []
    seen_names: set[str] = set()
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict):
            raise DatasetPreparationError(f"sources[{index}] must be a mapping.")
        name = raw_source.get("name")
        if not isinstance(name, str) or not VIEW_NAME_PATTERN.fullmatch(name):
            raise DatasetPreparationError(
                f"sources[{index}].name must match {VIEW_NAME_PATTERN.pattern!r}."
            )
        if name in seen_names:
            raise DatasetPreparationError(f"Duplicate source name: {name}")
        seen_names.add(name)
        if "images" not in raw_source or "masks" not in raw_source:
            raise DatasetPreparationError(
                f"sources[{index}] requires both images and masks paths."
            )
        sources.append(
            SourceSpec(
                name=name,
                images=resolve_project_path(raw_source["images"], f"{name}.images"),
                masks=resolve_project_path(raw_source["masks"], f"{name}.masks"),
            )
        )

    output = document.get("output")
    if not isinstance(output, dict):
        raise DatasetPreparationError("output must be a mapping.")
    required = ("images", "masks", "manifest")
    missing = [name for name in required if name not in output]
    if missing:
        raise DatasetPreparationError(
            "output is missing required path(s): " + ", ".join(missing)
        )
    output_images = resolve_project_path(output["images"], "output.images")
    output_masks = resolve_project_path(output["masks"], "output.masks")
    output_manifest = resolve_project_path(output["manifest"], "output.manifest")
    if output_manifest.suffix.lower() != ".csv":
        raise DatasetPreparationError("output.manifest must use the .csv extension.")
    if len({output_images, output_masks, output_manifest}) != 3:
        raise DatasetPreparationError("Output image, mask, and manifest paths must differ.")
    if output_images in output_masks.parents or output_masks in output_images.parents:
        raise DatasetPreparationError(
            "output.images and output.masks cannot contain one another."
        )
    if output_images in output_manifest.parents or output_masks in output_manifest.parents:
        raise DatasetPreparationError(
            "output.manifest cannot be stored inside an output image or mask directory."
        )
    return project_name, sources, output_images, output_masks, output_manifest


def image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise DatasetPreparationError(f"Image directory not found: {root}")
    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise DatasetPreparationError(f"No supported images found: {root}")
    return files


def format_examples(values: Iterable[str], limit: int = 8) -> str:
    examples = list(values)
    rendered = "\n".join(f"  {value}" for value in examples[:limit])
    if len(examples) > limit:
        rendered += f"\n  ...and {len(examples) - limit} more"
    return rendered


def inspect_rgb(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image, UnidentifiedImageError
    except ModuleNotFoundError as error:
        raise DatasetPreparationError(
            "Pillow is required. Activate the pot-masking environment."
        ) from error
    try:
        with Image.open(path) as image:
            size = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise DatasetPreparationError(f"Unreadable RGB image {path}: {error}") from error
    return size


def inspect_binary_mask(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image, UnidentifiedImageError
    except ModuleNotFoundError as error:
        raise DatasetPreparationError(
            "Pillow is required. Activate the pot-masking environment."
        ) from error
    try:
        with Image.open(path) as opened:
            size = opened.size
            mask = opened.convert("L")
            try:
                colors = mask.getcolors(maxcolors=3)
            finally:
                mask.close()
    except (OSError, UnidentifiedImageError) as error:
        raise DatasetPreparationError(f"Unreadable mask {path}: {error}") from error
    if colors is None or {value for _, value in colors} != {0, 255}:
        raise DatasetPreparationError(
            f"Mask must contain both binary values 0 and 255 only: {path}"
        )
    return size


def validate_output_separation(
    sources: list[SourceSpec],
    output_images: Path,
    output_masks: Path,
    output_manifest: Path,
) -> None:
    for output_name, output_path in (
        ("output.images", output_images),
        ("output.masks", output_masks),
        ("output.manifest", output_manifest),
    ):
        for source in sources:
            for source_name, source_path in (
                (f"{source.name}.images", source.images),
                (f"{source.name}.masks", source.masks),
            ):
                if (
                    output_path == source_path
                    or source_path in output_path.parents
                    or output_path in source_path.parents
                ):
                    raise DatasetPreparationError(
                        f"{output_name} must be separate from {source_name}: {output_path}"
                    )


def collect_pairs(sources: list[SourceSpec]) -> list[FramePair]:
    pairs: list[FramePair] = []
    output_names: set[str] = set()
    common_size: tuple[int, int] | None = None

    for source in sources:
        images = image_files(source.images)
        if not source.masks.is_dir():
            raise DatasetPreparationError(f"Mask directory not found: {source.masks}")
        expected_names = {f"{image.name}.png" for image in images}
        actual_names = {
            path.name
            for path in source.masks.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        }
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        if missing:
            raise DatasetPreparationError(
                f"{source.name} is missing {len(missing)} COLMAP mask(s):\n"
                + format_examples(missing)
            )
        if extra:
            raise DatasetPreparationError(
                f"{source.name} has {len(extra)} unexpected COLMAP mask(s):\n"
                + format_examples(extra)
            )

        for index, image in enumerate(images):
            mask = source.masks / f"{image.name}.png"
            image_size = inspect_rgb(image)
            mask_size = inspect_binary_mask(mask)
            if mask_size != image_size:
                raise DatasetPreparationError(
                    f"Dimension mismatch for {source.name}/{image.name}: RGB is "
                    f"{image_size[0]}x{image_size[1]}, mask is "
                    f"{mask_size[0]}x{mask_size[1]}."
                )
            if common_size is None:
                common_size = image_size
            elif image_size != common_size:
                raise DatasetPreparationError(
                    f"All source images must share one size. Expected "
                    f"{common_size[0]}x{common_size[1]}, found "
                    f"{image_size[0]}x{image_size[1]} in {image}."
                )

            combined_filename = f"{source.name}_{image.name}"
            if combined_filename in output_names:
                raise DatasetPreparationError(
                    f"Combined filename collision: {combined_filename}"
                )
            output_names.add(combined_filename)
            pairs.append(
                FramePair(
                    view=source.name,
                    view_frame_index=index,
                    image=image,
                    mask=mask,
                    combined_filename=combined_filename,
                    width=image_size[0],
                    height=image_size[1],
                )
            )
    return pairs


def sha256_copy(source: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        for block in iter(lambda: input_handle.read(1024 * 1024), b""):
            digest.update(block)
            output_handle.write(block)
    shutil.copystat(source, destination)
    return digest.hexdigest()


def manifest_source_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def safe_remove_generated(path: Path) -> None:
    resolved = path.resolve()
    relative = resolved.relative_to(PROJECT_ROOT)
    if len(relative.parts) < 3:
        raise DatasetPreparationError(
            f"Refusing to remove a broad generated-output path: {resolved}"
        )
    if resolved.is_dir():
        shutil.rmtree(resolved)
    elif resolved.exists():
        resolved.unlink()


def ensure_outputs_available(
    output_images: Path, output_masks: Path, output_manifest: Path, overwrite: bool
) -> None:
    existing = [
        path for path in (output_images, output_masks, output_manifest) if path.exists()
    ]
    if existing and not overwrite:
        raise DatasetPreparationError(
            "Generated output already exists. Use --overwrite only when replacement "
            "is intended:\n" + format_examples(str(path) for path in existing)
        )


def create_staging_directory(destination: Path) -> Path:
    """Create a sibling staging directory with permissions inherited normally.

    ``tempfile.mkdtemp`` requests mode 0700. Recent Windows Python versions can
    translate that into a restrictive ACL which survives a later rename and
    prevents sandboxed project tools from reading the final dataset. A normal
    ``Path.mkdir`` inherits the OneDrive/project parent permissions instead.
    """

    for _ in range(20):
        candidate = destination.parent / (
            f".{destination.name}-{uuid.uuid4().hex}.tmp"
        )
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise DatasetPreparationError(
        f"Could not allocate a staging directory beside {destination}."
    )


def write_staged_dataset(
    pairs: list[FramePair],
    output_images: Path,
    output_masks: Path,
    output_manifest: Path,
    overwrite: bool,
) -> None:
    ensure_outputs_available(output_images, output_masks, output_manifest, overwrite)
    output_images.parent.mkdir(parents=True, exist_ok=True)
    output_masks.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    image_staging = create_staging_directory(output_images)
    mask_staging = create_staging_directory(output_masks)
    manifest_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{output_manifest.stem}-",
        suffix=".csv.tmp",
        dir=output_manifest.parent,
        delete=False,
    )
    manifest_staging = Path(manifest_handle.name)
    try:
        with manifest_handle:
            writer = csv.DictWriter(manifest_handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for number, pair in enumerate(pairs, start=1):
                image_destination = image_staging / pair.combined_filename
                mask_destination = mask_staging / f"{pair.combined_filename}.png"
                image_hash = sha256_copy(pair.image, image_destination)
                mask_hash = sha256_copy(pair.mask, mask_destination)
                writer.writerow(
                    {
                        "view": pair.view,
                        "view_frame_index": pair.view_frame_index,
                        "source_filename": pair.image.name,
                        "combined_filename": pair.combined_filename,
                        "source_image": manifest_source_path(pair.image),
                        "source_mask": manifest_source_path(pair.mask),
                        "width": pair.width,
                        "height": pair.height,
                        "image_sha256": image_hash,
                        "mask_sha256": mask_hash,
                    }
                )
                if number == 1 or number % 25 == 0 or number == len(pairs):
                    print(f"Staged {number}/{len(pairs)} image-mask pairs", flush=True)

        if overwrite:
            for path in (output_images, output_masks, output_manifest):
                if path.exists():
                    safe_remove_generated(path)
        image_staging.replace(output_images)
        mask_staging.replace(output_masks)
        manifest_staging.replace(output_manifest)
    finally:
        if image_staging.exists():
            shutil.rmtree(image_staging)
        if mask_staging.exists():
            shutil.rmtree(mask_staging)
        manifest_staging.unlink(missing_ok=True)


def prepare_dataset(
    sources: list[SourceSpec],
    output_images: Path,
    output_masks: Path,
    output_manifest: Path,
    *,
    validate_only: bool,
    overwrite: bool,
) -> PreparationReport:
    validate_output_separation(
        sources, output_images, output_masks, output_manifest
    )
    pairs = collect_pairs(sources)
    if not pairs:
        raise DatasetPreparationError("No image-mask pairs were collected.")
    if validate_only:
        ensure_outputs_available(output_images, output_masks, output_manifest, overwrite)
    else:
        write_staged_dataset(
            pairs, output_images, output_masks, output_manifest, overwrite
        )
    return PreparationReport(
        sources=len(sources),
        images=len(pairs),
        masks=len(pairs),
        width=pairs[0].width,
        height=pairs[0].height,
        written=not validate_only,
    )


def main() -> int:
    args = parse_args()
    document = read_config(args.config)
    project_name, sources, output_images, output_masks, output_manifest = config_paths(
        document
    )

    print(f"Multiview dataset: {project_name}")
    for source in sources:
        print(f"Source {source.name}: {source.images}")
        print(f"Masks  {source.name}: {source.masks}")
    print(f"Combined images: {output_images}")
    print(f"Combined masks:  {output_masks}")
    print(f"Manifest:        {output_manifest}")
    print(f"Mode:            {'validate only' if args.validate_only else 'write'}")

    report = prepare_dataset(
        sources,
        output_images,
        output_masks,
        output_manifest,
        validate_only=args.validate_only,
        overwrite=args.overwrite,
    )
    print("\nMultiview dataset validation complete")
    print(f"Sources: {report.sources}")
    print(f"Images/masks: {report.images}/{report.masks}")
    print(f"Resolution: {report.width}x{report.height}")
    if report.written:
        print("Combined COLMAP staging complete")
    else:
        print("No output files were written")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except DatasetPreparationError as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
