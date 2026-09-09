"""COLMAP and prepared-cache loader for the masked 3DGS branch."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.common import GaussianSplattingPaths, PipelineError


@dataclass(frozen=True)
class CameraRecord:
    name: str
    image_path: Path
    mask_path: Path
    K: np.ndarray
    camtoworld: np.ndarray


@dataclass(frozen=True)
class SceneData:
    factor: int
    width: int
    height: int
    resolutions: tuple[tuple[int, int], ...]
    train_records: tuple[CameraRecord, ...]
    test_records: tuple[CameraRecord, ...]
    points: np.ndarray
    point_colors: np.ndarray
    normalization_center: np.ndarray
    normalization_scale: float


def camera_model_name(camera: Any) -> str:
    """Return a stable COLMAP camera-model name across pycolmap versions."""

    return getattr(camera.model, "name", str(camera.model))


def validate_undistorted_cameras(cameras: Any) -> None:
    """Accept one or more cameras when every camera is undistorted PINHOLE."""

    values = list(cameras.values()) if hasattr(cameras, "values") else list(cameras)
    if not values:
        raise PipelineError("The undistorted COLMAP model contains no cameras.")
    unsupported = sorted(
        {camera_model_name(camera) for camera in values if camera_model_name(camera) != "PINHOLE"}
    )
    if unsupported:
        raise PipelineError(
            "Undistorted 3DGS input must use only PINHOLE cameras, found: "
            + ", ".join(unsupported)
        )


def resolution_text(resolutions: tuple[tuple[int, int], ...]) -> str:
    return ", ".join(f"{width}x{height}" for width, height in resolutions)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PipelineError(f"Required prepared-data manifest was not found: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"Could not read prepared-data manifest {path}: {error}") from error
    if not isinstance(data, dict):
        raise PipelineError(f"Manifest root must be an object: {path}")
    return data


def normalize_scene(
    camtoworlds: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Center at the camera orbit and scale its maximum radius to one."""

    if camtoworlds.ndim != 3 or camtoworlds.shape[1:] != (4, 4):
        raise PipelineError("Camera-to-world matrices must have shape [N, 4, 4].")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise PipelineError("COLMAP initialization points must have shape [N, 3].")

    camera_centers = camtoworlds[:, :3, 3]
    center = camera_centers.mean(axis=0)
    radius = np.linalg.norm(camera_centers - center, axis=1).max()
    if not np.isfinite(radius) or radius <= 0:
        raise PipelineError("Could not determine a valid camera-orbit scale.")
    scale = float(1.0 / radius)

    normalized_cameras = camtoworlds.copy()
    normalized_cameras[:, :3, 3] = (camera_centers - center) * scale
    normalized_points = (points - center) * scale
    return normalized_cameras, normalized_points, center.astype(np.float32), scale


def load_scene(paths: GaussianSplattingPaths, factor: int) -> SceneData:
    try:
        import pycolmap
        from PIL import Image
    except ModuleNotFoundError as error:
        raise PipelineError("pycolmap and Pillow are required to load the 3DGS dataset.") from error

    factor_root = paths.cache / f"factor_{factor}"
    cache_manifest = read_json(factor_root / "manifest.json")
    split_manifest = read_json(paths.holdout_split)
    if cache_manifest.get("factor") != factor:
        raise PipelineError(
            f"Cache manifest factor is {cache_manifest.get('factor')}, expected {factor}."
        )

    entries = cache_manifest.get("images")
    if not isinstance(entries, list) or not entries:
        raise PipelineError("Prepared cache manifest contains no image entries.")
    entry_by_name: dict[str, dict[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise PipelineError("Prepared cache image entry must be an object.")
        source_name = entry.get("source_name")
        image_name = entry.get("image")
        mask_name = entry.get("mask")
        if not all(isinstance(value, str) and value for value in (source_name, image_name, mask_name)):
            raise PipelineError("Prepared cache image entry is missing a path.")
        entry_by_name[source_name.replace("\\", "/")] = {
            "image": image_name,
            "mask": mask_name,
        }

    reconstruction = pycolmap.Reconstruction(paths.sparse_model)
    validate_undistorted_cameras(reconstruction.cameras)

    images = sorted(
        (image for image in reconstruction.images.values() if image.has_pose),
        key=lambda image: image.name,
    )
    if len(images) != len(entry_by_name):
        raise PipelineError(
            f"Registered COLMAP image count {len(images)} does not match prepared "
            f"cache count {len(entry_by_name)}."
        )

    raw_camtoworlds: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    cached_paths: list[tuple[Path, Path]] = []
    source_names: list[str] = []
    resolutions: set[tuple[int, int]] = set()

    for image in images:
        normalized_name = image.name.replace("\\", "/")
        entry = entry_by_name.get(normalized_name)
        if entry is None:
            raise PipelineError(
                f"Registered image is missing from factor-{factor} cache: {normalized_name}"
            )
        cached_image = factor_root / entry["image"]
        cached_mask = factor_root / entry["mask"]
        if not cached_image.is_file() or not cached_mask.is_file():
            raise PipelineError(f"Prepared cache pair is missing for: {normalized_name}")

        with Image.open(cached_image) as prepared_image, Image.open(cached_mask) as prepared_mask:
            if prepared_image.size != prepared_mask.size:
                raise PipelineError(f"Prepared image/mask dimensions differ: {normalized_name}")
            current_width, current_height = prepared_image.size
        resolutions.add((current_width, current_height))

        camera = reconstruction.cameras[image.camera_id]
        K = np.asarray(camera.calibration_matrix(), dtype=np.float64)
        K[0, :] *= current_width / camera.width
        K[1, :] *= current_height / camera.height

        worldtocamera = np.eye(4, dtype=np.float64)
        worldtocamera[:3, :4] = np.asarray(image.cam_from_world().matrix())
        raw_camtoworlds.append(np.linalg.inv(worldtocamera))
        intrinsics.append(K)
        cached_paths.append((cached_image, cached_mask))
        source_names.append(normalized_name)

    point_ids = sorted(reconstruction.points3D)
    points = np.asarray(
        [reconstruction.points3D[point_id].xyz for point_id in point_ids],
        dtype=np.float32,
    )
    point_colors = np.asarray(
        [reconstruction.points3D[point_id].color for point_id in point_ids],
        dtype=np.float32,
    ) / 255.0
    camera_array = np.asarray(raw_camtoworlds, dtype=np.float32)
    normalized_cameras, normalized_points, center, scale = normalize_scene(
        camera_array, points
    )

    train_names = split_manifest.get("train")
    test_names = split_manifest.get("test")
    if not isinstance(train_names, list) or not isinstance(test_names, list):
        raise PipelineError("Holdout split manifest is missing train/test name lists.")
    train_set = {str(name).replace("\\", "/") for name in train_names}
    test_set = {str(name).replace("\\", "/") for name in test_names}
    if train_set & test_set or train_set | test_set != set(source_names):
        raise PipelineError("Holdout split does not exactly partition the prepared images.")

    records: list[CameraRecord] = []
    for index, name in enumerate(source_names):
        records.append(
            CameraRecord(
                name=name,
                image_path=cached_paths[index][0],
                mask_path=cached_paths[index][1],
                K=np.asarray(intrinsics[index], dtype=np.float32),
                camtoworld=np.asarray(normalized_cameras[index], dtype=np.float32),
            )
        )

    train_records = tuple(record for record in records if record.name in train_set)
    test_records = tuple(record for record in records if record.name in test_set)
    sorted_resolutions = tuple(sorted(resolutions))
    if not sorted_resolutions:
        raise PipelineError("Prepared cache contains no image resolutions.")
    width, height = sorted_resolutions[0]
    return SceneData(
        factor=factor,
        width=width,
        height=height,
        resolutions=sorted_resolutions,
        train_records=train_records,
        test_records=test_records,
        points=np.asarray(normalized_points, dtype=np.float32),
        point_colors=point_colors,
        normalization_center=center,
        normalization_scale=scale,
    )


def load_view(record: CameraRecord) -> dict[str, Any]:
    try:
        import torch
        from PIL import Image
    except ModuleNotFoundError as error:
        raise PipelineError("PyTorch and Pillow are required to load training views.") from error

    with Image.open(record.image_path) as image, Image.open(record.mask_path) as mask:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        alpha = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return {
        "name": record.name,
        "image": torch.from_numpy(rgb.copy()),
        "alpha": torch.from_numpy(alpha.copy()),
        "K": torch.from_numpy(record.K.copy()),
        "camtoworld": torch.from_numpy(record.camtoworld.copy()),
    }
