"""Reusable helpers for the pottery background-removal pipeline.

This module deliberately contains no SAM 2 or PyTorch imports.  Dataset
inspection, annotation, output generation, review, and unit tests can therefore
run in the light-weight OpenCV environment used by the existing project tools.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PROMPT_VERSION = 1


@dataclass(frozen=True)
class FrameInfo:
    """One validated input frame and any row from frames_manifest.csv."""

    index: int
    path: Path
    width: int
    height: int
    manifest: dict[str, str]

    @property
    def filename(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class MaskCleanupResult:
    mask: np.ndarray
    raw_area: int
    clean_area: int
    cleanup_change_ratio: float
    raw_component_count: int


def natural_sort_key(value: str | Path) -> list[object]:
    """Return a human-friendly key that sorts digit runs numerically."""

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def load_source_manifest(image_dir: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Load an optional frame manifest while preserving its column order."""

    manifest_path = image_dir / "frames_manifest.csv"
    if not manifest_path.is_file():
        return [], {}
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = {
            row["filename"]: dict(row)
            for row in reader
            if row.get("filename")
        }
    return fieldnames, rows


def discover_frames(image_dir: Path) -> tuple[list[FrameInfo], list[str]]:
    """Find, sort, and validate a folder of same-sized image frames."""

    image_dir = image_dir.resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Frame directory not found: {image_dir}")

    paths = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=natural_sort_key,
    )
    if not paths:
        raise FileNotFoundError(f"No JPG, JPEG, or PNG frames found in {image_dir}")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Input frame filenames must be unique")

    manifest_fields, manifest = load_source_manifest(image_dir)
    frames: list[FrameInfo] = []
    expected_size: tuple[int, int] | None = None
    for index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim < 2:
            raise RuntimeError(f"OpenCV could not read frame: {path}")
        height, width = image.shape[:2]
        size = (width, height)
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise ValueError(
                f"All frames must have one resolution. {path.name} is "
                f"{width}x{height}, expected {expected_size[0]}x{expected_size[1]}."
            )
        frames.append(
            FrameInfo(
                index=index,
                path=path,
                width=width,
                height=height,
                manifest=manifest.get(path.name, {}),
            )
        )
    return frames, manifest_fields


def image_directory_fingerprint(frames: Sequence[FrameInfo]) -> str:
    """Fingerprint the ordered frame identity without hashing large image files."""

    digest = hashlib.sha256()
    for frame in frames:
        stat = frame.path.stat()
        digest.update(frame.filename.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def normalized_box(box_xyxy: Sequence[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in box_xyxy]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("The prompt box must be inside the selected frame")
    return [x0 / width, y0 / height, x1 / width, y1 / height]


def denormalized_box(box: Sequence[float], width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = [float(value) for value in box]
    return np.asarray(
        [x0 * width, y0 * height, x1 * width, y1 * height], dtype=np.float32
    )


def normalized_points(
    points: Sequence[tuple[float, float, int]], width: int, height: int
) -> list[dict[str, float | int]]:
    result: list[dict[str, float | int]] = []
    for x, y, label in points:
        if label not in {0, 1}:
            raise ValueError("Point labels must be 0 for background or 1 for foreground")
        if not (0 <= x <= width and 0 <= y <= height):
            raise ValueError("Prompt points must be inside the selected frame")
        result.append({"x": x / width, "y": y / height, "label": int(label)})
    return result


def denormalized_points(
    points: Sequence[dict[str, float | int]], width: int, height: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not points:
        return None, None
    coordinates = np.asarray(
        [[float(point["x"]) * width, float(point["y"]) * height] for point in points],
        dtype=np.float32,
    )
    labels = np.asarray([int(point["label"]) for point in points], dtype=np.int32)
    return coordinates, labels


def validate_prompt_document(
    document: dict[str, Any], frames: Sequence[FrameInfo]
) -> None:
    if document.get("version") != PROMPT_VERSION:
        raise ValueError(
            f"Unsupported prompts.json version {document.get('version')!r}; "
            f"expected {PROMPT_VERSION}."
        )
    if document.get("material") not in {"matte", "glossy"}:
        raise ValueError("prompts.json material must be 'matte' or 'glossy'")
    if not document.get("prompts"):
        raise ValueError("prompts.json contains no pot prompts")
    names = {frame.filename for frame in frames}
    for prompt in document["prompts"]:
        if prompt.get("filename") not in names:
            raise ValueError(f"Prompt frame is not in the input dataset: {prompt.get('filename')}")
        box = prompt.get("box_xyxy_normalized")
        if box is not None and (len(box) != 4 or any(not 0 <= float(v) <= 1 for v in box)):
            raise ValueError(f"Invalid normalized box for {prompt.get('filename')}")


def upsert_prompt(document: dict[str, Any], prompt: dict[str, Any]) -> None:
    prompts = document.setdefault("prompts", [])
    prompts[:] = [item for item in prompts if item.get("filename") != prompt["filename"]]
    prompts.append(prompt)
    prompts.sort(key=lambda item: natural_sort_key(item["filename"]))


def _link_or_copy(source: Path, target: Path) -> str:
    try:
        os.symlink(source, target)
        return "symlink"
    except OSError:
        pass
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


@contextmanager
def numeric_frame_staging(
    frames: Sequence[FrameInfo], jpeg_quality: int = 100
) -> Iterator[tuple[Path, list[str]]]:
    """Expose frames to SAM 2 as numeric JPEGs without touching the originals."""

    with tempfile.TemporaryDirectory(prefix="pot-mask-sam2-") as directory:
        staging = Path(directory)
        methods: list[str] = []
        digits = max(6, len(str(max(0, len(frames) - 1))))
        for local_index, frame in enumerate(frames):
            target = staging / f"{local_index:0{digits}d}.jpg"
            if frame.path.suffix.lower() in {".jpg", ".jpeg"}:
                methods.append(_link_or_copy(frame.path, target))
            else:
                image = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
                if image is None or not cv2.imwrite(
                    str(target), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                ):
                    raise RuntimeError(f"Could not stage frame as JPEG: {frame.path}")
                methods.append("jpeg-conversion")
        yield staging, methods


def ensure_binary_mask(mask: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(mask)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional mask, got {array.shape}")
    if shape is not None and array.shape != shape:
        array = cv2.resize(
            array.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR
        )
    return np.where(array > 0, 255, 0).astype(np.uint8)


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_fg = np.asarray(first) > 0
    second_fg = np.asarray(second) > 0
    union = np.logical_or(first_fg, second_fg).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first_fg, second_fg).sum() / union)


def _select_component(mask: np.ndarray, previous_mask: np.ndarray | None) -> tuple[np.ndarray, int]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    all_component_count = max(0, count - 1)
    if all_component_count == 0:
        return np.zeros_like(mask), 0

    largest_area = max(int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count))
    significant_area = max(16, round(largest_area * 0.001))
    component_count = sum(
        int(stats[label, cv2.CC_STAT_AREA]) >= significant_area
        for label in range(1, count)
    )

    previous_fg = None
    if previous_mask is not None and np.any(previous_mask):
        previous_fg = cv2.dilate(
            (previous_mask > 0).astype(np.uint8), np.ones((7, 7), np.uint8)
        ) > 0

    height, width = mask.shape
    center = np.asarray([width / 2.0, height / 2.0])
    diagonal = math.hypot(width, height)
    best_label = 1
    best_score = -1.0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        candidate = labels == label
        if previous_fg is not None:
            overlap = int(np.logical_and(candidate, previous_fg).sum())
            score = overlap / max(1, min(area, int(previous_fg.sum())))
            score += min(area / (height * width), 0.25) * 0.02
        else:
            distance = float(np.linalg.norm(centroids[label] - center)) / max(diagonal, 1)
            score = area / (height * width) - distance * 0.01
        if score > best_score:
            best_score = score
            best_label = label
    return np.where(labels == best_label, 255, 0).astype(np.uint8), component_count


def _fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    inverted = cv2.bitwise_not(mask)
    flooded = inverted.copy()
    flood_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flooded, flood_mask, (0, 0), 0)
    return cv2.bitwise_or(mask, flooded)


def postprocess_mask(
    raw_mask: np.ndarray,
    previous_mask: np.ndarray | None = None,
    shape: tuple[int, int] | None = None,
) -> MaskCleanupResult:
    """Keep the tracked pot, close tiny gaps, and fill enclosed false holes."""

    binary = ensure_binary_mask(raw_mask, shape)
    raw_area = int(np.count_nonzero(binary))
    selected, component_count = _select_component(binary, previous_mask)
    if np.any(selected):
        selected = cv2.morphologyEx(
            selected, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
        )
        selected = _fill_enclosed_holes(selected)
    clean_area = int(np.count_nonzero(selected))
    changed = int(np.count_nonzero(cv2.bitwise_xor(binary, selected)))
    cleanup_ratio = changed / max(raw_area, 1)
    return MaskCleanupResult(
        mask=selected,
        raw_area=raw_area,
        clean_area=clean_area,
        cleanup_change_ratio=float(cleanup_ratio),
        raw_component_count=component_count,
    )


def erode_mask(mask: np.ndarray, pixels: int = 3) -> np.ndarray:
    if pixels < 0:
        raise ValueError("Erosion pixels cannot be negative")
    binary = ensure_binary_mask(mask)
    if pixels == 0:
        return binary.copy()
    size = pixels * 2 + 1
    return cv2.erode(binary, np.ones((size, size), np.uint8), iterations=1)


def mask_geometry(mask: np.ndarray) -> dict[str, float | int]:
    foreground = np.asarray(mask) > 0
    height, width = foreground.shape
    ys, xs = np.nonzero(foreground)
    if len(xs) == 0:
        return {
            "mask_area": 0,
            "mask_area_ratio": 0.0,
            "bbox_x": -1,
            "bbox_y": -1,
            "bbox_width": 0,
            "bbox_height": 0,
            "centroid_x_normalized": -1.0,
            "centroid_y_normalized": -1.0,
            "touches_boundary": 0,
        }
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    touches = x0 == 0 or y0 == 0 or x1 == width - 1 or y1 == height - 1
    return {
        "mask_area": int(len(xs)),
        "mask_area_ratio": float(len(xs) / (height * width)),
        "bbox_x": x0,
        "bbox_y": y0,
        "bbox_width": x1 - x0 + 1,
        "bbox_height": y1 - y0 + 1,
        "centroid_x_normalized": float(xs.mean() / width),
        "centroid_y_normalized": float(ys.mean() / height),
        "touches_boundary": int(touches),
    }


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"Could not write PNG: {path}")


def write_rgba(path: Path, bgr: np.ndarray, alpha: np.ndarray) -> None:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("RGBA output requires a three-channel source image")
    binary = ensure_binary_mask(alpha, bgr.shape[:2])
    bgra = np.dstack((bgr, binary))
    write_png(path, bgra)


def make_overlay(
    bgr: np.ndarray, mask: np.ndarray, max_dimension: int = 1280
) -> np.ndarray:
    binary = ensure_binary_mask(mask, bgr.shape[:2])
    overlay = bgr.copy()
    foreground = binary > 0
    tint = np.zeros_like(overlay)
    tint[:, :] = (60, 210, 70)
    overlay[foreground] = cv2.addWeighted(
        overlay[foreground], 0.72, tint[foreground], 0.28, 0
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 3)
    height, width = overlay.shape[:2]
    scale = min(1.0, max_dimension / max(width, height))
    if scale < 1.0:
        overlay = cv2.resize(
            overlay,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return overlay


def write_jpeg(path: Path, image: np.ndarray, quality: int = 88) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f"Could not write JPEG: {path}")


def calculate_sequence_flags(
    records: list[dict[str, Any]],
    material: str,
    area_threshold: float = 0.04,
    centroid_threshold: float = 0.01,
    iou_threshold: float = 0.95,
) -> None:
    """Append deterministic sequence-level QC flags to manifest records."""

    areas = [float(record["mask_area_ratio"]) for record in records]
    cleanup_threshold = 0.005 if material == "glossy" else 0.01
    for index, record in enumerate(records):
        flags = set(filter(None, str(record.get("qc_warnings", "")).split("|")))
        if int(record["mask_area"]) == 0:
            flags.add("empty_mask")
        if int(record.get("touches_boundary", 0)):
            flags.add("touches_boundary")
        if int(record.get("raw_component_count", 0)) > 1:
            flags.add("disconnected_prediction")
        if float(record.get("cleanup_change_ratio", 0.0)) > cleanup_threshold:
            flags.add("large_cleanup_change")
        if index > 0 and float(record.get("previous_iou", 1.0)) < iou_threshold:
            flags.add("low_previous_iou")

        start = max(0, index - 2)
        end = min(len(records), index + 3)
        local_median = float(np.median(areas[start:end]))
        if local_median > 0 and abs(areas[index] - local_median) / local_median > area_threshold:
            flags.add("area_jump")

        if index > 0:
            previous = records[index - 1]
            current_x = float(record["centroid_x_normalized"])
            current_y = float(record["centroid_y_normalized"])
            previous_x = float(previous["centroid_x_normalized"])
            previous_y = float(previous["centroid_y_normalized"])
            if (
                min(current_x, current_y, previous_x, previous_y) >= 0
                and math.hypot(current_x - previous_x, current_y - previous_y)
                > centroid_threshold
            ):
                flags.add("centroid_jump")

        record["qc_warnings"] = "|".join(sorted(flags))
        if record.get("review_status") not in {"accepted", "correction_added", "prompted"}:
            record["review_status"] = "flagged" if flags else "unreviewed"


def write_manifest(path: Path, records: Sequence[dict[str, Any]], source_fields: Sequence[str]) -> None:
    generated_fields = [
        "input_filename",
        "object_mask",
        "colmap_mask",
        "rgba_image",
        "overlay_image",
        "mask_area",
        "mask_area_ratio",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "centroid_x_normalized",
        "centroid_y_normalized",
        "previous_iou",
        "cleanup_change_ratio",
        "raw_component_count",
        "chunk_id",
        "review_status",
        "qc_warnings",
    ]
    fields = list(dict.fromkeys([*source_fields, *generated_fields]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def read_mask_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def build_qc_contact_sheet(
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    destination: Path,
    sample_stride: int = 20,
    columns: int = 6,
) -> int:
    selected = [
        record
        for index, record in enumerate(records)
        if record.get("qc_warnings") or index % sample_stride == 0
    ]
    if not selected:
        return 0

    tile_width, image_height, label_height, gap = 220, 220, 48, 8
    rows = math.ceil(len(selected) / columns)
    canvas = np.full(
        (
            gap + rows * (image_height + label_height + gap),
            gap + columns * (tile_width + gap),
            3,
        ),
        24,
        dtype=np.uint8,
    )
    for index, record in enumerate(selected):
        overlay_path = output_dir / str(record["overlay_image"])
        image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = min(tile_width / image.shape[1], image_height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        row, column = divmod(index, columns)
        x = gap + column * (tile_width + gap)
        y = gap + row * (image_height + label_height + gap)
        x_offset = x + (tile_width - resized.shape[1]) // 2
        y_offset = y + (image_height - resized.shape[0]) // 2
        canvas[y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]] = resized
        warning = str(record.get("qc_warnings") or "sample")
        warning = warning[:30]
        cv2.putText(
            canvas,
            str(record["input_filename"])[:28],
            (x + 3, y + image_height + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            warning,
            (x + 3, y + image_height + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (90, 190, 255) if warning != "sample" else (165, 165, 165),
            1,
            cv2.LINE_AA,
        )
    write_jpeg(destination, canvas, quality=90)
    return len(selected)
