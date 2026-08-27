#!/usr/bin/env python3
"""Visualize SIFT feature matching between two image frames.

This is intended as a small photogrammetry/SfM progress-report demo:
1. detect SIFT keypoints and descriptors,
2. match descriptors with FLANN,
3. filter ambiguous matches with Lowe's ratio test,
4. optionally use a fundamental-matrix RANSAC check to reject
   geometrically inconsistent matches,
5. save a side-by-side visualization.

Example:
    python demo_feature_matching.py frame_000879.jpg frame_000882.jpg matches.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match SIFT features between two frames and save a visualization."
    )
    parser.add_argument("image1", type=Path, help="First image/frame.")
    parser.add_argument("image2", type=Path, help="Second image/frame.")
    parser.add_argument("output", type=Path, help="Output .jpg or .png visualization.")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe ratio-test threshold. Lower is stricter. Default: 0.75.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.5,
        metavar="PIXELS",
        help="Fundamental-matrix RANSAC reprojection threshold. Default: 1.5.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.99,
        help="RANSAC confidence in the range (0, 1). Default: 0.99.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1600,
        metavar="PIXELS",
        help="Resize each frame so its longest side is at most this size before matching. "
        "Use 0 for original resolution. Default: 1600.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=4000,
        metavar="N",
        help="Maximum SIFT features per image. Use 0 for OpenCV default/unlimited. Default: 4000.",
    )
    parser.add_argument(
        "--draw-limit",
        type=int,
        default=150,
        metavar="N",
        help="Maximum geometrically verified matches to draw. Use 0 to draw all. Default: 150.",
    )
    parser.add_argument(
        "--no-ransac",
        action="store_true",
        help="Skip fundamental-matrix RANSAC and draw ratio-test matches directly.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an OpenCV preview window after saving.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.ratio < 1.0:
        raise ValueError("--ratio must be between 0 and 1")
    if args.ransac_threshold <= 0:
        raise ValueError("--ransac-threshold must be greater than 0")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    if args.max_size < 0:
        raise ValueError("--max-size cannot be negative")
    if args.max_features < 0:
        raise ValueError("--max-features cannot be negative")
    if args.draw_limit < 0:
        raise ValueError("--draw-limit cannot be negative")
    if args.output.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Output filename must end in .jpg, .jpeg, or .png")


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def resize_for_matching(image: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    if max_size == 0:
        return image, 1.0

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_size:
        return image, 1.0

    scale = max_size / longest_side
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def detect_sift(image: np.ndarray, max_features: int):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def ratio_test_matches(descriptors1: np.ndarray, descriptors2: np.ndarray, ratio: float):
    index_params = dict(algorithm=1, trees=5)  # FLANN KD-tree for SIFT descriptors
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        best, second_best = pair
        if best.distance < ratio * second_best.distance:
            good_matches.append(best)

    return good_matches, len(knn_matches)


def fundamental_ransac_inliers(keypoints1, keypoints2, matches, threshold: float, confidence: float):
    if len(matches) < 8:
        return None, []

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    fundamental_matrix, mask = cv2.findFundamentalMat(
        points1,
        points2,
        cv2.FM_RANSAC,
        threshold,
        confidence,
    )

    if fundamental_matrix is None or mask is None:
        return None, []

    mask = mask.ravel().astype(bool)
    inliers = [match for match, keep in zip(matches, mask) if keep]
    return fundamental_matrix, inliers


def add_header(image: np.ndarray, lines: list[str]) -> np.ndarray:
    header_height = 72
    canvas = np.full(
        (image.shape[0] + header_height, image.shape[1], 3),
        20,
        dtype=np.uint8,
    )
    canvas[header_height:] = image

    cv2.putText(
        canvas,
        lines[0],
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    if len(lines) > 1:
        cv2.putText(
            canvas,
            lines[1],
            (12, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (190, 210, 230),
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_output(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        parameters = [cv2.IMWRITE_JPEG_QUALITY, 95]
    else:
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    if not cv2.imwrite(str(path), image, parameters):
        raise RuntimeError(f"Failed to write output image: {path}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    image1_original = load_image(args.image1)
    image2_original = load_image(args.image2)
    image1, scale1 = resize_for_matching(image1_original, args.max_size)
    image2, scale2 = resize_for_matching(image2_original, args.max_size)

    keypoints1, descriptors1 = detect_sift(image1, args.max_features)
    keypoints2, descriptors2 = detect_sift(image2, args.max_features)

    if descriptors1 is None or descriptors2 is None:
        raise RuntimeError("SIFT could not find usable descriptors in one or both images.")

    good_matches, candidate_count = ratio_test_matches(
        descriptors1, descriptors2, args.ratio
    )

    fundamental_matrix = None
    if args.no_ransac:
        verified_matches = good_matches
        verification_name = "RANSAC disabled"
    else:
        fundamental_matrix, verified_matches = fundamental_ransac_inliers(
            keypoints1,
            keypoints2,
            good_matches,
            args.ransac_threshold,
            args.confidence,
        )
        verification_name = "Fundamental matrix RANSAC"

    verified_matches = sorted(verified_matches, key=lambda match: match.distance)
    matches_to_draw = (
        verified_matches
        if args.draw_limit == 0
        else verified_matches[: args.draw_limit]
    )

    visualization = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches_to_draw,
        None,
        matchColor=(0, 220, 0),
        singlePointColor=(80, 80, 80),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    inlier_ratio = (
        100.0 * len(verified_matches) / len(good_matches)
        if good_matches
        else 0.0
    )

    first_line = (
        f"SIFT + FLANN | ratio={args.ratio:.2f} | "
        f"keypoints: {len(keypoints1)} / {len(keypoints2)}"
    )
    if args.no_ransac:
        second_line = (
            f"Ratio-test matches: {len(good_matches)} | "
            f"drawing {len(matches_to_draw)}"
        )
    else:
        second_line = (
            f"{verification_name}: {len(verified_matches)}/{len(good_matches)} inliers "
            f"({inlier_ratio:.1f}%) | drawing {len(matches_to_draw)}"
        )

    visualization = add_header(visualization, [first_line, second_line])
    save_output(args.output, visualization)

    print(f"Image 1:                  {args.image1}")
    print(f"Image 2:                  {args.image2}")
    print(f"Image 1 working size:     {image1.shape[1]}x{image1.shape[0]} (scale {scale1:.3f})")
    print(f"Image 2 working size:     {image2.shape[1]}x{image2.shape[0]} (scale {scale2:.3f})")
    print(f"SIFT keypoints:           {len(keypoints1)} / {len(keypoints2)}")
    print(f"FLANN k-NN candidates:    {candidate_count}")
    print(f"Lowe ratio-test matches:  {len(good_matches)}")

    if args.no_ransac:
        print("Geometric verification:   disabled")
    else:
        print(f"RANSAC inliers:            {len(verified_matches)} ({inlier_ratio:.1f}%)")
        if fundamental_matrix is None:
            print("Fundamental matrix:        estimation failed")
        else:
            print("Fundamental matrix:")
            print(fundamental_matrix)

    print(f"Matches drawn:            {len(matches_to_draw)}")
    print(f"Output:                   {args.output.resolve()}")

    if args.show:
        cv2.imshow("SIFT feature matching", visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
