#!/usr/bin/env python3
"""Extract frames from a video and write frame metadata to a CSV manifest.

Example:
    python extract_video_frames.py input.mp4 output_frames

By default, every decoded frame is saved. Use --every-n later when testing
different frame-sampling intervals for COLMAP.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video frames as numbered image files."
    )
    parser.add_argument("video", type=Path, help="Path to the input video.")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="Output directory. Defaults to <video-name>_frames.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        metavar="N",
        help="Save every Nth decoded frame. Default: 1 (save all frames).",
    )
    parser.add_argument(
        "--format",
        choices=("jpg", "png"),
        default="jpg",
        dest="image_format",
        help="Output image format. Default: jpg.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        metavar="1-100",
        help="JPEG quality when --format=jpg. Default: 95.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=3,
        metavar="0-9",
        help="PNG compression when --format=png. Default: 3.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        metavar="INDEX",
        help="First zero-based source frame to consider. Default: 0.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        metavar="INDEX",
        help="Last zero-based source frame to consider, inclusive.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace frame files and the manifest if they already exist.",
    )
    return parser.parse_args()


def format_timestamp(timestamp_ms: float) -> str:
    total_ms = max(0, round(timestamp_ms))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def validate_args(args: argparse.Namespace) -> None:
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if args.every_n < 1:
        raise ValueError("--every-n must be at least 1")
    if args.start_frame < 0:
        raise ValueError("--start-frame cannot be negative")
    if args.end_frame is not None and args.end_frame < args.start_frame:
        raise ValueError("--end-frame cannot be less than --start-frame")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if not 0 <= args.png_compression <= 9:
        raise ValueError("--png-compression must be between 0 and 9")


def image_write_parameters(args: argparse.Namespace) -> list[int]:
    if args.image_format == "jpg":
        return [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    return [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]


def ensure_output_is_safe(
    output_dir: Path, manifest_path: Path, extension: str, overwrite: bool
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = [
        *output_dir.glob("frame_*.jpg"),
        *output_dir.glob("frame_*.png"),
    ]
    if not overwrite and (existing_frames or manifest_path.exists()):
        raise FileExistsError(
            f"Output already exists in {output_dir}. "
            "Use --overwrite or choose another directory."
        )
    if overwrite:
        for frame_path in existing_frames:
            frame_path.unlink()
        manifest_path.unlink(missing_ok=True)


def extract_frames(args: argparse.Namespace) -> tuple[int, int, Path]:
    output_dir = args.output_dir or args.video.with_name(
        f"{args.video.stem}_frames"
    )
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "frames_manifest.csv"
    extension = args.image_format

    ensure_output_is_safe(output_dir, manifest_path, extension, args.overwrite)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {args.video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_digits = max(6, len(str(max(0, reported_frame_count - 1))))
    write_parameters = image_write_parameters(args)

    decoded_count = 0
    saved_count = 0
    source_frame_index = 0

    try:
        with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "filename",
                "output_index",
                "source_frame_index",
                "timestamp_ms",
                "timestamp",
                "width",
                "height",
                "source_video",
                "source_fps",
                "sampling_interval",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                success, frame = capture.read()
                if not success:
                    break

                decoded_count += 1
                if args.end_frame is not None and source_frame_index > args.end_frame:
                    break

                should_save = (
                    source_frame_index >= args.start_frame
                    and (source_frame_index - args.start_frame) % args.every_n == 0
                )

                if should_save:
                    filename = f"frame_{source_frame_index:0{frame_digits}d}.{extension}"
                    output_path = output_dir / filename
                    if output_path.exists() and not args.overwrite:
                        raise FileExistsError(f"Frame already exists: {output_path}")

                    if not cv2.imwrite(str(output_path), frame, write_parameters):
                        raise RuntimeError(f"Failed to write frame: {output_path}")

                    timestamp_ms = (
                        source_frame_index * 1000.0 / fps
                        if fps > 0
                        else float(capture.get(cv2.CAP_PROP_POS_MSEC))
                    )
                    height, width = frame.shape[:2]
                    writer.writerow(
                        {
                            "filename": filename,
                            "output_index": saved_count,
                            "source_frame_index": source_frame_index,
                            "timestamp_ms": f"{timestamp_ms:.3f}",
                            "timestamp": format_timestamp(timestamp_ms),
                            "width": width,
                            "height": height,
                            "source_video": str(args.video.resolve()),
                            "source_fps": f"{fps:.6f}" if fps > 0 else "unknown",
                            "sampling_interval": args.every_n,
                        }
                    )
                    saved_count += 1

                    if saved_count % 100 == 0:
                        print(
                            f"Saved {saved_count} frames "
                            f"(source frame {source_frame_index})..."
                        )

                source_frame_index += 1
    except Exception:
        if saved_count == 0 and manifest_path.exists():
            manifest_path.unlink()
        raise
    finally:
        capture.release()

    if saved_count == 0:
        if manifest_path.exists():
            manifest_path.unlink()
        raise RuntimeError("No frames were saved. Check the selected frame range.")

    return decoded_count, saved_count, manifest_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    decoded_count, saved_count, manifest_path = extract_frames(args)

    print(f"Decoded frames: {decoded_count}")
    print(f"Saved frames:   {saved_count}")
    print(f"Manifest:       {manifest_path}")


if __name__ == "__main__":
    main()
