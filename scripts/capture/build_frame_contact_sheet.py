#!/usr/bin/env python3
"""Build one labeled contact-sheet image from every image in a folder.

If frames_manifest.csv exists in the image folder, source-frame and timestamp
information from the manifest is included below each thumbnail.

Example:
    python scripts/capture/build_frame_contact_sheet.py output_frames contact_sheet.jpg
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine a folder of images into one labeled contact sheet."
    )
    parser.add_argument("image_dir", type=Path, help="Folder containing images.")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output image. Defaults to <folder-name>_contact_sheet.jpg.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        help="Number of grid columns. Default: automatically create a square grid.",
    )
    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=280,
        metavar="PIXELS",
        help="Width of each thumbnail area. Default: 280.",
    )
    parser.add_argument(
        "--thumbnail-height",
        type=int,
        default=210,
        metavar="PIXELS",
        help="Height of each thumbnail area. Default: 210.",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=10,
        metavar="PIXELS",
        help="Space between tiles. Default: 10.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include images in subdirectories.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        metavar="1-100",
        help="JPEG output quality. Default: 92.",
    )
    parser.add_argument(
        "--max-canvas-megapixels",
        type=float,
        default=100.0,
        metavar="MP",
        help="Refuse larger canvases to avoid exhausting memory. Default: 100.",
    )
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help="Allow a canvas larger than --max-canvas-megapixels.",
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def find_images(image_dir: Path, recursive: bool, output: Path) -> list[Path]:
    candidates = image_dir.rglob("*") if recursive else image_dir.iterdir()
    output_resolved = output.resolve()
    images = [
        path
        for path in candidates
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and path.resolve() != output_resolved
    ]
    return sorted(images, key=natural_sort_key)


def load_manifest(image_dir: Path) -> dict[str, dict[str, str]]:
    manifest_path = image_dir / "frames_manifest.csv"
    if not manifest_path.is_file():
        return {}

    with manifest_path.open(newline="", encoding="utf-8") as csv_file:
        return {
            row["filename"]: row
            for row in csv.DictReader(csv_file)
            if row.get("filename")
        }


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    new_width = max(1, round(source_width * scale))
    new_height = max(1, round(source_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    fitted = np.full((height, width, 3), 28, dtype=np.uint8)
    x = (width - new_width) // 2
    y = (height - new_height) // 2
    fitted[y : y + new_height, x : x + new_width] = resized
    return fitted


def shorten(text: str, max_characters: int) -> str:
    if len(text) <= max_characters:
        return text
    return text[: max(1, max_characters - 3)] + "..."


def put_label(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.43,
    color: tuple[int, int, int] = (225, 225, 225),
    max_width: int | None = None,
    thickness: int = 1,
) -> None:
    if max_width is not None:
        text_width = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )[0][0]
        if text_width > max_width:
            scale *= max_width / text_width

    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def tile_information(
    image_path: Path,
    image: np.ndarray,
    index: int,
    total: int,
    manifest: dict[str, dict[str, str]],
) -> tuple[str, str]:
    height, width = image.shape[:2]
    first_line = f"{index + 1}/{total}  {shorten(image_path.name, 30)}"

    row = manifest.get(image_path.name)
    if row:
        frame_number = row.get("source_frame_index", "?")
        timestamp = row.get("timestamp", "unknown time")
        second_line = f"frame {frame_number} | {timestamp} | {width}x{height}"
    else:
        second_line = f"{width}x{height} | {image_path.suffix.lower()[1:].upper()}"

    return first_line, second_line


def write_output(output: Path, canvas: np.ndarray, jpeg_quality: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    extension = output.suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Output must end in .jpg, .jpeg, or .png")

    parameters = (
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        if extension in {".jpg", ".jpeg"}
        else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    if not cv2.imwrite(str(output), canvas, parameters):
        raise RuntimeError(f"Failed to write contact sheet: {output}")


def build_contact_sheet(args: argparse.Namespace) -> tuple[Path, int, int, int]:
    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {args.image_dir}")
    if args.columns is not None and args.columns < 1:
        raise ValueError("--columns must be at least 1")
    if args.thumbnail_width < 40 or args.thumbnail_height < 40:
        raise ValueError("Thumbnail dimensions must each be at least 40 pixels")
    if args.gap < 0:
        raise ValueError("--gap cannot be negative")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.max_canvas_megapixels <= 0:
        raise ValueError("--max-canvas-megapixels must be greater than zero")

    output = args.output or args.image_dir.with_name(
        f"{args.image_dir.name}_contact_sheet.jpg"
    )
    output = output.resolve()
    image_paths = find_images(args.image_dir, args.recursive, output)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {args.image_dir}")

    total = len(image_paths)
    columns = args.columns or math.ceil(math.sqrt(total))
    columns = min(columns, total)
    rows = math.ceil(total / columns)

    label_height = 52
    header_height = 86
    tile_width = args.thumbnail_width
    tile_height = args.thumbnail_height + label_height
    canvas_width = args.gap + columns * (tile_width + args.gap)
    canvas_height = header_height + args.gap + rows * (tile_height + args.gap)
    canvas_megapixels = canvas_width * canvas_height / 1_000_000

    if canvas_megapixels > args.max_canvas_megapixels and not args.allow_large:
        raise MemoryError(
            f"The contact sheet would be {canvas_width}x{canvas_height} "
            f"({canvas_megapixels:.1f} MP). Reduce --thumbnail-width/height, "
            "or use --allow-large if the computer has enough memory."
        )

    canvas = np.full((canvas_height, canvas_width, 3), 18, dtype=np.uint8)
    manifest = load_manifest(args.image_dir)

    title = f"Frame Contact Sheet | {args.image_dir.name} | {total} images"
    subtitle = (
        f"Grid: {columns} columns x {rows} rows | "
        f"Canvas: {canvas_width}x{canvas_height}"
    )
    put_label(
        canvas,
        title,
        args.gap,
        34,
        0.78,
        (255, 255, 255),
        canvas_width - 2 * args.gap,
        2,
    )
    put_label(
        canvas,
        subtitle,
        args.gap,
        62,
        0.5,
        (180, 180, 180),
        canvas_width - 2 * args.gap,
    )

    for index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not read image: {image_path}")

        row, column = divmod(index, columns)
        x = args.gap + column * (tile_width + args.gap)
        y = header_height + args.gap + row * (tile_height + args.gap)

        thumbnail = fit_image(
            image, args.thumbnail_width, args.thumbnail_height
        )
        canvas[
            y : y + args.thumbnail_height,
            x : x + args.thumbnail_width,
        ] = thumbnail
        cv2.rectangle(
            canvas,
            (x, y),
            (x + args.thumbnail_width - 1, y + args.thumbnail_height - 1),
            (85, 85, 85),
            1,
        )

        first_line, second_line = tile_information(
            image_path, image, index, total, manifest
        )
        label_y = y + args.thumbnail_height
        cv2.rectangle(
            canvas,
            (x, label_y),
            (x + tile_width - 1, label_y + label_height - 1),
            (36, 36, 36),
            -1,
        )
        put_label(
            canvas,
            first_line,
            x + 7,
            label_y + 20,
            max_width=tile_width - 14,
        )
        put_label(
            canvas,
            shorten(second_line, 42),
            x + 7,
            label_y + 41,
            0.4,
            (175, 205, 230),
            tile_width - 14,
        )

        if (index + 1) % 100 == 0:
            print(f"Placed {index + 1}/{total} images...")

    write_output(output, canvas, args.jpeg_quality)
    return output, total, canvas_width, canvas_height


def main() -> None:
    args = parse_args()
    output, total, width, height = build_contact_sheet(args)
    print(f"Images included: {total}")
    print(f"Canvas size:     {width}x{height}")
    print(f"Contact sheet:   {output}")


if __name__ == "__main__":
    main()
