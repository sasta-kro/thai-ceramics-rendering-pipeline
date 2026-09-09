#!/usr/bin/env python3
"""
Rotate a single photo by 180 degrees.

Usage:
    python scripts/capture/rotate_photo_180.py /path/to/folder
    python scripts/capture/rotate_photo_180.py --output /path/to/other_folder
    python scripts/capture/rotate_photo_180.py /path/to/folder --in-place
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# Maps JPEG (h,v) sampling factors -> Pillow's subsampling code, mirroring
# the private table Pillow itself uses for quality="keep".
_JPEG_SUBSAMPLING = {
    (1, 1, 1, 1, 1, 1): 0,  # 4:4:4
    (2, 1, 1, 1, 1, 1): 1,  # 4:2:2
    (2, 2, 1, 1, 1, 1): 2,  # 4:2:0
}


def rotate_image_180(src_path: Path, dst_path: Path) -> None:
    """Rotate a single image 180 degrees and save it, preserving quality."""
    with Image.open(src_path) as img:
        img.load()  # force-read pixel data now, in case dst == src (in-place mode)
        fmt = img.format  # e.g. 'JPEG', 'PNG', ...
        icc_profile = img.info.get("icc_profile")
        exif = img.info.get("exif")

        save_kwargs = {}
        if fmt == "JPEG":
            # Reuse the exact original quantization tables and chroma
            # subsampling so re-encoding the rotated pixels doesn't add any
            # extra generation loss (this is what quality="keep" does
            # internally; we replicate it manually since Image.transpose()
            # returns a plain Image that loses the JPEG-specific metadata
            # quality="keep" needs).
            quantization = getattr(img, "quantization", None)
            if quantization:
                save_kwargs["qtables"] = quantization
            layers = getattr(img, "layers", None)
            layer = getattr(img, "layer", None)
            if layer and layers not in (1, 4):
                sampling = layer[0][1:3] + layer[1][1:3] + layer[2][1:3]
                if sampling in _JPEG_SUBSAMPLING:
                    save_kwargs["subsampling"] = _JPEG_SUBSAMPLING[sampling]
        elif fmt == "WEBP":
            save_kwargs["lossless"] = True
            save_kwargs["quality"] = 100
        elif fmt == "TIFF":
            compression = img.info.get("compression")
            if compression:
                save_kwargs["compression"] = compression
        # PNG and BMP are lossless by default already; nothing extra needed.

        # Pure 180-degree turn: exact pixel remap, no interpolation, no flip.
        rotated = img.transpose(Image.ROTATE_180)

    # Source file handle is closed now, so it's safe to write even if
    # dst_path is the same file as src_path.
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    if exif:
        save_kwargs["exif"] = exif

    rotated.save(dst_path, format=fmt, **save_kwargs)


def process_folder(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print(f"No supported image files found in {input_dir}")
        return

    ok, failed = 0, 0
    for path in files:
        dst = output_dir / path.name
        try:
            rotate_image_180(path, dst)
            print(f"Rotated: {path.name}")
            ok += 1
        except Exception as e:
            print(f"Failed on {path.name}: {e}", file=sys.stderr)
            failed += 1

    print(f"\nDone. {ok} rotated, {failed} failed -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Rotate every photo in a folder by 180 degrees (no flipping)."
    )
    parser.add_argument("folder", type=Path, help="Folder containing the photos to rotate")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Destination folder (default: <folder>/rotated_180)"
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite the original files instead of writing to a new folder"
    )
    args = parser.parse_args()

    input_dir = args.folder
    if not input_dir.is_dir():
        sys.exit(f"Not a folder: {input_dir}")

    output_dir = input_dir if args.in_place else (args.output or (input_dir / "rotated_180"))

    process_folder(input_dir, output_dir)


if __name__ == "__main__":
    main()