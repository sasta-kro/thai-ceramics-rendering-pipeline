"""Build a smooth, multi-camera COLMAP pose model without merging point clouds.

The script aligns the combined side/underside model to the original smooth
side-only model through shared side-to-underside 3D observations. It then
aligns the smooth top-only model through shared top-to-side observations.
Only camera poses are transferred. Existing 3D observations are cleared so a
subsequent point_triangulator run reconstructs a fresh, consistent point set.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPAIR_ROOT = (
    REPO_ROOT
    / "data/processed/pot1-unglazed_multiview/"
    "colmap_sparse_masked_multiview_lightglue_repaired"
)
DEFAULT_SIDE_TEXT = DEFAULT_REPAIR_ROOT / "source_text/side"
DEFAULT_TOP_TEXT = DEFAULT_REPAIR_ROOT / "source_text/top"
DEFAULT_COMBINED_TEXT = DEFAULT_REPAIR_ROOT / "source_text/combined"
DEFAULT_OUTPUT_TEXT = DEFAULT_REPAIR_ROOT / "repaired_text"
DEFAULT_REPORT = DEFAULT_REPAIR_ROOT / "reports/pose_alignment.json"

SIDE_CAMERA_ID = 1
TOP_CAMERA_ID = 2
UNDERSIDE_CAMERA_ID = 3
RANSAC_SEED = 4213


class PoseRepairError(RuntimeError):
    """Raised when a safe pose repair cannot be constructed."""


@dataclass(frozen=True)
class Pose:
    image_id: int
    name: str
    qvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    center: np.ndarray


@dataclass(frozen=True)
class PointModel:
    xyz: dict[int, np.ndarray]
    tracks: dict[int, tuple[tuple[str, int], ...]]
    observation_to_point: dict[tuple[str, int], int]
    point_views: dict[int, frozenset[str]]


@dataclass(frozen=True)
class Similarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        return (self.scale * (self.rotation @ points.T)).T + self.translation


@dataclass(frozen=True)
class AlignmentResult:
    similarity: Similarity
    candidate_count: int
    inlier_count: int
    residual_median: float
    residual_rmse: float
    residual_max: float
    threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align smooth side/top COLMAP camera trajectories into the combined "
            "coordinate system and emit a point-free three-camera text model."
        )
    )
    parser.add_argument("--side-text", type=Path, default=DEFAULT_SIDE_TEXT)
    parser.add_argument("--top-text", type=Path, default=DEFAULT_TOP_TEXT)
    parser.add_argument("--combined-text", type=Path, default=DEFAULT_COMBINED_TEXT)
    parser.add_argument("--output-text", type=Path, default=DEFAULT_OUTPUT_TEXT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ransac-iterations", type=int, default=4000)
    parser.add_argument("--side-threshold", type=float, default=0.04)
    parser.add_argument("--top-threshold", type=float, default=0.04)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def view_group(name: str) -> str:
    basename = Path(name).name.lower()
    if basename.startswith("side_"):
        return "side"
    if basename.startswith("top45_"):
        return "top45"
    if basename.startswith("underside_"):
        return "underside"
    return "other"


def normalized_source_name(name: str, group: str) -> str:
    basename = Path(name).name
    if group == "side" and basename.startswith("frame_"):
        return f"side_{basename}"
    return basename


def frame_index(name: str) -> int:
    match = re.search(r"(\d+)(?=\.[^.]+$)", Path(name).name)
    if match is None:
        raise PoseRepairError(f"Could not read frame index from {name!r}.")
    return int(match.group(1))


def qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=float)
    norm = np.linalg.norm(qvec)
    if not np.isfinite(norm) or norm == 0:
        raise PoseRepairError("Invalid zero or non-finite quaternion.")
    qw, qx, qy, qz = qvec / norm
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=float,
    )


def rotation_to_qvec(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float)
    trace = float(np.trace(rotation))
    if trace > 0:
        root = math.sqrt(trace + 1.0) * 2
        qvec = np.array(
            [
                0.25 * root,
                (rotation[2, 1] - rotation[1, 2]) / root,
                (rotation[0, 2] - rotation[2, 0]) / root,
                (rotation[1, 0] - rotation[0, 1]) / root,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            root = math.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2
            qvec = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / root,
                    0.25 * root,
                    (rotation[0, 1] + rotation[1, 0]) / root,
                    (rotation[0, 2] + rotation[2, 0]) / root,
                ]
            )
        elif index == 1:
            root = math.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2
            qvec = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / root,
                    (rotation[0, 1] + rotation[1, 0]) / root,
                    0.25 * root,
                    (rotation[1, 2] + rotation[2, 1]) / root,
                ]
            )
        else:
            root = math.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2
            qvec = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / root,
                    (rotation[0, 2] + rotation[2, 0]) / root,
                    (rotation[1, 2] + rotation[2, 1]) / root,
                    0.25 * root,
                ]
            )
    qvec /= np.linalg.norm(qvec)
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def make_pose(image_id: int, name: str, qvec: np.ndarray, tvec: np.ndarray) -> Pose:
    rotation = qvec_to_rotation(qvec)
    center = -rotation.T @ np.asarray(tvec, dtype=float)
    return Pose(
        image_id=image_id,
        name=name,
        qvec=np.asarray(qvec, dtype=float),
        tvec=np.asarray(tvec, dtype=float),
        rotation=rotation,
        center=center,
    )


def transform_pose(pose: Pose, similarity: Similarity) -> Pose:
    center = (
        similarity.scale * (similarity.rotation @ pose.center)
        + similarity.translation
    )
    rotation = pose.rotation @ similarity.rotation.T
    tvec = -rotation @ center
    return make_pose(pose.image_id, pose.name, rotation_to_qvec(rotation), tvec)


def compose_similarity(outer: Similarity, inner: Similarity) -> Similarity:
    return Similarity(
        scale=outer.scale * inner.scale,
        rotation=outer.rotation @ inner.rotation,
        translation=(
            outer.scale * (outer.rotation @ inner.translation)
            + outer.translation
        ),
    )


def read_image_poses(model_dir: Path, group: str | None = None) -> dict[str, Pose]:
    path = model_dir / "images.txt"
    if not path.is_file():
        raise PoseRepairError(f"Missing COLMAP images file: {path}")
    poses: dict[str, Pose] = {}
    expect_image = True
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            if expect_image and line.strip():
                fields = line.split()
                if len(fields) < 10:
                    raise PoseRepairError(f"Invalid image metadata line in {path}.")
                name = normalized_source_name(fields[9], group or "")
                pose = make_pose(
                    int(fields[0]),
                    name,
                    np.asarray(fields[1:5], dtype=float),
                    np.asarray(fields[5:8], dtype=float),
                )
                if name in poses:
                    raise PoseRepairError(f"Duplicate image name {name!r} in {path}.")
                poses[name] = pose
            expect_image = not expect_image
    return poses


def image_id_names(poses: dict[str, Pose]) -> dict[int, str]:
    result = {pose.image_id: name for name, pose in poses.items()}
    if len(result) != len(poses):
        raise PoseRepairError("Image IDs are not unique.")
    return result


def read_points(model_dir: Path, names: dict[int, str]) -> PointModel:
    path = model_dir / "points3D.txt"
    xyz: dict[int, np.ndarray] = {}
    tracks: dict[int, tuple[tuple[str, int], ...]] = {}
    observation_to_point: dict[tuple[str, int], int] = {}
    point_views: dict[int, frozenset[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            point_id = int(fields[0])
            xyz[point_id] = np.asarray(fields[1:4], dtype=float)
            observations: list[tuple[str, int]] = []
            views: set[str] = set()
            for offset in range(8, len(fields), 2):
                image_id = int(fields[offset])
                if image_id not in names:
                    raise PoseRepairError(
                        f"Point {point_id} refers to unknown image {image_id}."
                    )
                observation = (names[image_id], int(fields[offset + 1]))
                observations.append(observation)
                observation_to_point[observation] = point_id
                views.add(view_group(observation[0]))
            tracks[point_id] = tuple(observations)
            point_views[point_id] = frozenset(views)
    return PointModel(xyz, tracks, observation_to_point, point_views)


def shared_point_pairs(
    source: PointModel,
    target: PointModel,
    required_target_view: str,
) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for source_id, observations in source.tracks.items():
        votes = Counter(
            target.observation_to_point[observation]
            for observation in observations
            if observation in target.observation_to_point
        )
        if not votes:
            continue
        target_id, _ = votes.most_common(1)[0]
        if required_target_view not in target.point_views[target_id]:
            continue
        pairs.add((source_id, target_id))
    return sorted(pairs)


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> Similarity:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise PoseRepairError("Similarity inputs must be equal N-by-3 arrays.")
    if len(source) < 3:
        raise PoseRepairError("At least three correspondences are required.")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_transpose))
    rotation = left @ correction @ right_transpose
    variance = float(np.sum(source_centered * source_centered) / len(source))
    if variance <= np.finfo(float).eps:
        raise PoseRepairError("Degenerate similarity correspondence sample.")
    scale = float(np.trace(np.diag(singular) @ correction) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise PoseRepairError("Similarity fit produced an invalid scale.")
    translation = target_mean - scale * (rotation @ source_mean)
    return Similarity(scale, rotation, translation)


def robust_similarity(
    source: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    iterations: int,
    seed: int = RANSAC_SEED,
) -> tuple[AlignmentResult, np.ndarray]:
    if threshold <= 0 or iterations <= 0:
        raise PoseRepairError("RANSAC threshold and iterations must be positive.")
    if len(source) < 4:
        raise PoseRepairError("At least four correspondences are required for RANSAC.")
    generator = np.random.default_rng(seed)
    evaluation_indices = generator.choice(
        len(source), min(6000, len(source)), replace=False
    )
    best_score: tuple[int, float] | None = None
    best_similarity: Similarity | None = None
    for _ in range(iterations):
        sample = generator.choice(len(source), 4, replace=False)
        try:
            similarity = estimate_similarity(source[sample], target[sample])
        except (PoseRepairError, np.linalg.LinAlgError):
            continue
        residuals = np.linalg.norm(
            similarity.transform_points(source[evaluation_indices])
            - target[evaluation_indices],
            axis=1,
        )
        inliers = residuals < threshold
        if not np.any(inliers):
            continue
        score = (int(inliers.sum()), -float(np.median(residuals[inliers])))
        if best_score is None or score > best_score:
            best_score = score
            best_similarity = similarity
    if best_similarity is None:
        raise PoseRepairError("RANSAC did not find a valid similarity transform.")

    similarity = best_similarity
    residuals = np.linalg.norm(
        similarity.transform_points(source) - target, axis=1
    )
    inliers = residuals < threshold
    for _ in range(10):
        similarity = estimate_similarity(source[inliers], target[inliers])
        residuals = np.linalg.norm(
            similarity.transform_points(source) - target, axis=1
        )
        updated = residuals < threshold
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    inlier_residuals = residuals[inliers]
    result = AlignmentResult(
        similarity=similarity,
        candidate_count=len(source),
        inlier_count=int(inliers.sum()),
        residual_median=float(np.median(inlier_residuals)),
        residual_rmse=float(np.sqrt(np.mean(inlier_residuals**2))),
        residual_max=float(np.max(inlier_residuals)),
        threshold=threshold,
    )
    return result, inliers


def read_camera_line(model_dir: Path) -> list[str]:
    path = model_dir / "cameras.txt"
    lines = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(lines) != 1:
        raise PoseRepairError(f"Expected one source camera in {path}, found {len(lines)}.")
    return lines[0]


def float_text(values: Iterable[float]) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def write_camera_files(
    output: Path, side_text: Path, top_text: Path, combined_text: Path
) -> None:
    definitions = [
        (SIDE_CAMERA_ID, read_camera_line(side_text)),
        (TOP_CAMERA_ID, read_camera_line(top_text)),
        (UNDERSIDE_CAMERA_ID, read_camera_line(combined_text)),
    ]
    with (output / "cameras.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write("# Number of cameras: 3\n")
        for camera_id, fields in definitions:
            handle.write(" ".join([str(camera_id), *fields[1:]]) + "\n")
    with (output / "rigs.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Rig calib list with one line of data per calib:\n")
        handle.write(
            "#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, "
            "SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE, "
            "[QW, QX, QY, QZ, TX, TY, TZ])\n"
        )
        handle.write("# Number of rigs: 3\n")
        for rig_id, camera_id in (
            (SIDE_CAMERA_ID, SIDE_CAMERA_ID),
            (TOP_CAMERA_ID, TOP_CAMERA_ID),
            (UNDERSIDE_CAMERA_ID, UNDERSIDE_CAMERA_ID),
        ):
            handle.write(f"{rig_id} 1 CAMERA {camera_id}\n")


def camera_id_for_name(name: str) -> int:
    group = view_group(name)
    if group == "side":
        return SIDE_CAMERA_ID
    if group == "top45":
        return TOP_CAMERA_ID
    if group == "underside":
        return UNDERSIDE_CAMERA_ID
    raise PoseRepairError(f"Unknown image view group for {name!r}.")


def write_images(
    source_path: Path,
    destination_path: Path,
    repaired_poses: dict[str, Pose],
) -> None:
    expect_image = True
    written = 0
    with source_path.open("r", encoding="utf-8") as source, destination_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        destination.write("# Image list with two lines of data per image:\n")
        destination.write(
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        )
        destination.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        destination.write(f"# Number of images: {len(repaired_poses)}\n")
        for line in source:
            if line.startswith("#"):
                continue
            if expect_image:
                fields = line.split()
                if not fields:
                    continue
                name = fields[9]
                pose = repaired_poses[name]
                destination.write(
                    f"{pose.image_id} {float_text(pose.qvec)} "
                    f"{float_text(pose.tvec)} {camera_id_for_name(name)} {name}\n"
                )
                written += 1
            else:
                fields = line.split()
                if len(fields) % 3 != 0:
                    raise PoseRepairError("Invalid POINTS2D line in combined model.")
                for offset in range(2, len(fields), 3):
                    fields[offset] = "-1"
                destination.write(" ".join(fields) + "\n")
            expect_image = not expect_image
    if written != len(repaired_poses):
        raise PoseRepairError(
            f"Wrote {written} repaired images, expected {len(repaired_poses)}."
        )


def write_frames(
    source_path: Path,
    destination_path: Path,
    repaired_poses: dict[str, Pose],
) -> None:
    poses_by_id = {pose.image_id: pose for pose in repaired_poses.values()}
    with source_path.open("r", encoding="utf-8") as source, destination_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        destination.write("# Frame list with one line of data per frame:\n")
        destination.write(
            "#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, "
            "TZ], NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)\n"
        )
        destination.write(f"# Number of frames: {len(repaired_poses)}\n")
        written = 0
        for line in source:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 13 or fields[10] != "CAMERA":
                raise PoseRepairError("Unexpected non-trivial frame layout.")
            frame_id = int(fields[0])
            image_id = int(fields[12])
            pose = poses_by_id[image_id]
            camera_id = camera_id_for_name(pose.name)
            destination.write(
                f"{frame_id} {camera_id} {float_text(pose.qvec)} "
                f"{float_text(pose.tvec)} 1 CAMERA {camera_id} {image_id}\n"
            )
            written += 1
    if written != len(repaired_poses):
        raise PoseRepairError(
            f"Wrote {written} repaired frames, expected {len(repaired_poses)}."
        )


def write_empty_points(path: Path) -> None:
    path.write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
        "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0, mean track length: 0\n",
        encoding="utf-8",
        newline="\n",
    )


def continuity_report(poses: dict[str, Pose]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for group in ("side", "top45", "underside"):
        sequence = sorted(
            (pose for pose in poses.values() if view_group(pose.name) == group),
            key=lambda pose: frame_index(pose.name),
        )
        distances = np.asarray(
            [
                np.linalg.norm(second.center - first.center)
                for first, second in zip(sequence, sequence[1:])
            ]
        )
        median = float(np.median(distances))
        maximum = float(np.max(distances))
        result[group] = {
            "images": len(sequence),
            "median_adjacent_distance": median,
            "max_adjacent_distance": maximum,
            "max_over_median": maximum / median,
            "closure_distance": float(
                np.linalg.norm(sequence[-1].center - sequence[0].center)
            ),
        }
    return result


def similarity_json(similarity: Similarity) -> dict[str, object]:
    return {
        "scale": similarity.scale,
        "rotation": similarity.rotation.tolist(),
        "translation": similarity.translation.tolist(),
    }


def alignment_json(result: AlignmentResult) -> dict[str, object]:
    return {
        "candidate_correspondences": result.candidate_count,
        "inliers": result.inlier_count,
        "threshold": result.threshold,
        "residual_median": result.residual_median,
        "residual_rmse": result.residual_rmse,
        "residual_max": result.residual_max,
        "similarity": similarity_json(result.similarity),
    }


def ensure_text_model(model_dir: Path) -> None:
    required = ("cameras.txt", "images.txt", "points3D.txt", "rigs.txt", "frames.txt")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise PoseRepairError(
            f"Text model {model_dir} is missing: {', '.join(missing)}"
        )


def build_repaired_model(args: argparse.Namespace) -> dict[str, object]:
    side_text = args.side_text.resolve()
    top_text = args.top_text.resolve()
    combined_text = args.combined_text.resolve()
    output_text = args.output_text.resolve()
    report_path = args.report.resolve()
    for model_dir in (side_text, top_text, combined_text):
        ensure_text_model(model_dir)

    side_poses = read_image_poses(side_text, "side")
    top_poses = read_image_poses(top_text, "top45")
    combined_poses = read_image_poses(combined_text)
    if (len(side_poses), len(top_poses), len(combined_poses)) != (273, 233, 728):
        raise PoseRepairError(
            "Unexpected source model sizes: "
            f"side={len(side_poses)}, top={len(top_poses)}, "
            f"combined={len(combined_poses)}."
        )

    side_points = read_points(side_text, image_id_names(side_poses))
    top_points = read_points(top_text, image_id_names(top_poses))
    combined_points = read_points(combined_text, image_id_names(combined_poses))

    side_pairs = shared_point_pairs(side_points, combined_points, "underside")
    top_pairs = shared_point_pairs(top_points, combined_points, "side")
    side_source = np.asarray(
        [combined_points.xyz[combined_id] for _, combined_id in side_pairs]
    )
    side_target = np.asarray(
        [side_points.xyz[side_id] for side_id, _ in side_pairs]
    )
    top_source = np.asarray([top_points.xyz[top_id] for top_id, _ in top_pairs])
    top_target = np.asarray(
        [combined_points.xyz[combined_id] for _, combined_id in top_pairs]
    )

    side_alignment, _ = robust_similarity(
        side_source,
        side_target,
        threshold=args.side_threshold,
        iterations=args.ransac_iterations,
    )
    top_alignment, _ = robust_similarity(
        top_source,
        top_target,
        threshold=args.top_threshold,
        iterations=args.ransac_iterations,
    )
    if side_alignment.inlier_count < 40:
        raise PoseRepairError(
            f"Only {side_alignment.inlier_count} side-under alignment inliers."
        )
    if top_alignment.inlier_count < 1000:
        raise PoseRepairError(
            f"Only {top_alignment.inlier_count} top-side alignment inliers."
        )

    combined_to_side = side_alignment.similarity
    top_to_side = compose_similarity(combined_to_side, top_alignment.similarity)
    repaired: dict[str, Pose] = {}
    for name, pose in side_poses.items():
        repaired[name] = pose
    for name, pose in top_poses.items():
        repaired[name] = transform_pose(pose, top_to_side)
    for name, pose in combined_poses.items():
        if view_group(name) == "underside":
            repaired[name] = transform_pose(pose, combined_to_side)
    if len(repaired) != 728 or set(repaired) != set(combined_poses):
        raise PoseRepairError("Repaired pose set does not contain exactly 728 images.")

    continuity = continuity_report(repaired)
    for group, values in continuity.items():
        if float(values["max_over_median"]) > 2.0:
            raise PoseRepairError(
                f"Repaired {group} trajectory remains discontinuous: "
                f"{values['max_over_median']:.3f}x median spacing."
            )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "shared-observation robust Sim(3) pose transfer",
        "point_clouds_merged": False,
        "source_models": {
            "side": str(side_text),
            "top": str(top_text),
            "combined": str(combined_text),
        },
        "camera_assignments": {
            "side": SIDE_CAMERA_ID,
            "top45": TOP_CAMERA_ID,
            "underside": UNDERSIDE_CAMERA_ID,
        },
        "side_under_alignment": alignment_json(side_alignment),
        "top_side_alignment": alignment_json(top_alignment),
        "top_to_side_similarity": similarity_json(top_to_side),
        "continuity": continuity,
    }
    print(
        "Side-under alignment: "
        f"{side_alignment.inlier_count}/{side_alignment.candidate_count} inliers, "
        f"RMSE {side_alignment.residual_rmse:.6f}"
    )
    print(
        "Top-side alignment: "
        f"{top_alignment.inlier_count}/{top_alignment.candidate_count} inliers, "
        f"RMSE {top_alignment.residual_rmse:.6f}"
    )
    for group, values in continuity.items():
        print(
            f"{group}: {values['images']} images, max/median adjacent spacing "
            f"{values['max_over_median']:.3f}"
        )
    if args.dry_run:
        print("Dry run only: no repaired text model was written.")
        return report

    if any(output_text.iterdir()):
        raise PoseRepairError(f"Output text directory is not empty: {output_text}")
    staging = output_text.parent / f".{output_text.name}-staging"
    if staging.exists():
        raise PoseRepairError(f"Staging directory already exists: {staging}")
    staging.mkdir()
    try:
        write_camera_files(staging, side_text, top_text, combined_text)
        write_images(
            combined_text / "images.txt", staging / "images.txt", repaired
        )
        write_frames(
            combined_text / "frames.txt", staging / "frames.txt", repaired
        )
        write_empty_points(staging / "points3D.txt")
        for generated in staging.iterdir():
            generated.replace(output_text / generated.name)
        staging.rmdir()
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Repaired text model: {output_text}")
    print(f"Alignment report: {report_path}")
    return report


def main() -> int:
    args = parse_args()
    try:
        build_repaired_model(args)
    except (OSError, ValueError, PoseRepairError, np.linalg.LinAlgError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
