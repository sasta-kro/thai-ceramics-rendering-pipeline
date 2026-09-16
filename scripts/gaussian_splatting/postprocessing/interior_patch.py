#!/usr/bin/env python3
"""Create a conservative cosmetic interior for an exported 3DGS PLY.

This is deliberately an export-only operation: the trained checkpoint and its
original exports are never modified.  The pot axis is inferred from the
mid25/underside COLMAP camera rings, then a shallow Gaussian bowl is placed
inside the opening while unstable central splats in front of it are removed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import sys
from typing import BinaryIO

import numpy as np

from ..core.common import (
    DEFAULT_CONFIG,
    PipelineError,
    load_yaml,
    resolve_config_path,
    resolve_paths,
)
from .checkpoint import export_path, resolve_complete_run, sha256_file


SH_C0 = 0.28209479177387814
PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass(frozen=True)
class PlyData:
    header_lines: tuple[str, ...]
    vertex_count: int
    vertices: np.ndarray


@dataclass(frozen=True)
class InteriorGeometry:
    axis: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray
    transverse_center: np.ndarray
    axial_bottom: float
    axial_rim: float
    rim_outer_radius: float
    opening_radius: float
    patch_radius: float
    edge_axial: float
    center_axial: float
    source_rgb: np.ndarray


@dataclass(frozen=True)
class PatchPlan:
    source_ply: str
    destination_ply: str
    source_gaussians: int
    removed_gaussians: int
    added_gaussians: int
    output_gaussians: int
    axis: list[float]
    rim_axial: float
    rim_outer_radius: float
    opening_radius: float
    patch_radius: float
    edge_axial: float
    center_axial: float
    source_rgb: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a conservative cosmetic interior to a completed 3DGS PLY."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--patch-name",
        default="interior_patched_v1",
        help="Subdirectory/name for the new export; the original export is untouched.",
    )
    parser.add_argument("--patch-gaussians", type=int, default=30000)
    parser.add_argument(
        "--opening-radius-scale",
        type=float,
        default=0.72,
        help="Opening radius as a fraction of the inferred outer rim radius.",
    )
    parser.add_argument(
        "--patch-radius-scale",
        type=float,
        default=0.90,
        help="Patched radius as a fraction of the inferred opening radius.",
    )
    parser.add_argument(
        "--edge-drop-scale",
        type=float,
        default=0.16,
        help="Patch edge depth below the rim, relative to outer rim radius.",
    )
    parser.add_argument(
        "--bowl-depth-scale",
        type=float,
        default=0.48,
        help="Additional center depth, relative to outer rim radius.",
    )
    parser.add_argument("--color-scale", type=float, default=0.68)
    parser.add_argument("--opacity", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    safe_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if not args.patch_name or any(
        char not in safe_characters for char in args.patch_name
    ):
        raise PipelineError("patch-name may contain only letters, numbers, '.', '_' and '-'.")
    if args.patch_gaussians < 1000:
        raise PipelineError("patch-gaussians must be at least 1000.")
    for label in (
        "opening_radius_scale",
        "patch_radius_scale",
        "edge_drop_scale",
        "bowl_depth_scale",
        "color_scale",
    ):
        value = float(getattr(args, label))
        if not math.isfinite(value) or value <= 0:
            raise PipelineError(f"{label.replace('_', '-')} must be positive.")
    if not 0 < args.opening_radius_scale < 1:
        raise PipelineError("opening-radius-scale must be between 0 and 1.")
    if not 0 < args.patch_radius_scale <= 1:
        raise PipelineError("patch-radius-scale must be in (0, 1].")
    if not 0 < args.color_scale <= 1:
        raise PipelineError("color-scale must be in (0, 1].")
    if not 0 < args.opacity < 1:
        raise PipelineError("opacity must be between 0 and 1.")


def read_exact(source: BinaryIO, size: int) -> bytes:
    value = source.read(size)
    if len(value) != size:
        raise PipelineError("COLMAP images.bin ended unexpectedly.")
    return value


def read_colmap_camera_centers(images_bin: Path) -> dict[str, np.ndarray]:
    """Read image names and camera centers from a COLMAP images.bin file."""

    try:
        source = images_bin.open("rb")
    except OSError as error:
        raise PipelineError(f"Could not open COLMAP image poses: {images_bin}") from error
    images: dict[str, np.ndarray] = {}
    with source:
        image_count = struct.unpack("<Q", read_exact(source, 8))[0]
        for _ in range(image_count):
            _image_id = struct.unpack("<i", read_exact(source, 4))[0]
            qvec = np.asarray(struct.unpack("<4d", read_exact(source, 32)), dtype=np.float64)
            tvec = np.asarray(struct.unpack("<3d", read_exact(source, 24)), dtype=np.float64)
            _camera_id = struct.unpack("<i", read_exact(source, 4))[0]
            name_bytes = bytearray()
            while True:
                byte = read_exact(source, 1)
                if byte == b"\x00":
                    break
                name_bytes.extend(byte)
            name = name_bytes.decode("utf-8")
            point_count = struct.unpack("<Q", read_exact(source, 8))[0]
            source.seek(point_count * 24, 1)

            norm = float(np.linalg.norm(qvec))
            if not math.isfinite(norm) or norm <= 0:
                raise PipelineError(f"Invalid camera quaternion for {name}.")
            qw, qx, qy, qz = qvec / norm
            rotation = np.asarray(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ],
                dtype=np.float64,
            )
            images[name.replace("\\", "/")] = -(rotation.T @ tvec)
    if not images:
        raise PipelineError("COLMAP images.bin contains no registered images.")
    return images


def read_ply(path: Path) -> PlyData:
    try:
        source = path.open("rb")
    except OSError as error:
        raise PipelineError(f"Could not open source PLY: {path}") from error
    with source:
        header: list[str] = []
        while True:
            raw = source.readline()
            if not raw:
                raise PipelineError("PLY header is missing end_header.")
            try:
                line = raw.decode("ascii").rstrip("\r\n")
            except UnicodeDecodeError as error:
                raise PipelineError("PLY header is not ASCII.") from error
            header.append(line)
            if line == "end_header":
                break
            if len(header) > 512:
                raise PipelineError("PLY header is unreasonably long.")

        if not header or header[0] != "ply" or "format binary_little_endian 1.0" not in header:
            raise PipelineError("Only binary little-endian PLY exports are supported.")
        vertex_count: int | None = None
        properties: list[tuple[str, str]] = []
        in_vertices = False
        other_elements: list[str] = []
        for line in header:
            fields = line.split()
            if fields[:2] == ["element", "vertex"] and len(fields) == 3:
                vertex_count = int(fields[2])
                in_vertices = True
            elif fields[:1] == ["element"] and fields[1:2] != ["vertex"]:
                in_vertices = False
                other_elements.append(fields[1])
            elif fields[:1] == ["property"] and in_vertices:
                if len(fields) != 3 or fields[1] == "list" or fields[1] not in PLY_TYPES:
                    raise PipelineError(f"Unsupported PLY vertex property: {line}")
                properties.append((fields[2], PLY_TYPES[fields[1]]))
        if vertex_count is None or vertex_count <= 0 or not properties:
            raise PipelineError("PLY does not contain a valid vertex element.")
        if other_elements:
            raise PipelineError("PLY contains unsupported non-vertex elements.")
        dtype = np.dtype(properties)
        vertices = np.fromfile(source, dtype=dtype, count=vertex_count)
        if len(vertices) != vertex_count:
            raise PipelineError("PLY vertex payload is truncated.")
        if source.read(1):
            raise PipelineError("PLY contains unexpected data after its vertices.")
    required = {
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = sorted(required - set(vertices.dtype.names or ()))
    if missing:
        raise PipelineError("PLY is missing 3DGS properties: " + ", ".join(missing))
    return PlyData(tuple(header), vertex_count, vertices)


def normalized_axis(
    centers: dict[str, np.ndarray], normalization_center: np.ndarray, scale: float
) -> np.ndarray:
    groups: dict[str, list[np.ndarray]] = {"mid25": [], "underside": []}
    for name, center in centers.items():
        basename = Path(name).name.lower()
        for group in groups:
            if basename.startswith(group + "_"):
                groups[group].append((center - normalization_center) * scale)
    if any(len(values) < 3 for values in groups.values()):
        counts = ", ".join(f"{name}={len(values)}" for name, values in groups.items())
        raise PipelineError(f"Could not infer top direction from camera rings ({counts}).")
    top = np.mean(groups["mid25"], axis=0)
    bottom = np.mean(groups["underside"], axis=0)
    axis = top - bottom
    length = float(np.linalg.norm(axis))
    if not math.isfinite(length) or length <= 1e-8:
        raise PipelineError("Camera-ring means do not define a valid pot axis.")
    return axis / length


def orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    basis_u = np.cross(reference, axis)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    basis_v /= np.linalg.norm(basis_v)
    return basis_u, basis_v


def vertex_xyz(vertices: np.ndarray) -> np.ndarray:
    return np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
        np.float64, copy=False
    )


def fit_rim_circle_center(points: np.ndarray) -> np.ndarray:
    """Fit a robust 2D center to a mostly circular upper-rim point band."""

    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 100:
        raise PipelineError("At least 100 projected rim points are required.")
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center[None, :], axis=1)
    lower, upper = np.quantile(distances, [0.55, 0.985])
    selected = points[(distances >= lower) & (distances <= upper)]
    if len(selected) < 50:
        raise PipelineError("Could not isolate enough outer-rim points for circle fitting.")

    for _ in range(3):
        matrix = np.column_stack(
            (2.0 * selected[:, 0], 2.0 * selected[:, 1], np.ones(len(selected)))
        )
        target = np.sum(selected * selected, axis=1)
        solution, _residuals, rank, _singular = np.linalg.lstsq(
            matrix, target, rcond=None
        )
        if rank < 3 or not np.all(np.isfinite(solution)):
            raise PipelineError("Upper-rim circle fit is degenerate.")
        center = solution[:2]
        all_distances = np.linalg.norm(points - center[None, :], axis=1)
        fitted_radius = float(np.median(np.linalg.norm(selected - center[None, :], axis=1)))
        residual = np.abs(all_distances - fitted_radius)
        cutoff = float(np.quantile(residual, 0.55))
        selected = points[residual <= max(cutoff, 1e-8)]
    return center


def infer_geometry(
    vertices: np.ndarray,
    axis: np.ndarray,
    opening_radius_scale: float,
    patch_radius_scale: float,
    edge_drop_scale: float,
    bowl_depth_scale: float,
) -> InteriorGeometry:
    xyz = vertex_xyz(vertices)
    logits = np.asarray(vertices["opacity"], dtype=np.float64)
    opacity = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    opaque = opacity >= 0.05
    if int(np.count_nonzero(opaque)) < 1000:
        raise PipelineError("Too few visible Gaussians remain to infer the pot opening.")
    visible = xyz[opaque]
    axial_visible = visible @ axis
    axial_bottom, axial_rim = np.quantile(axial_visible, [0.005, 0.995])
    height = float(axial_rim - axial_bottom)
    if not math.isfinite(height) or height <= 0:
        raise PipelineError("Could not infer a valid pot height.")

    basis_u, basis_v = orthonormal_basis(axis)
    upper_cut = float(np.quantile(axial_visible, 0.94))
    upper = visible[axial_visible >= upper_cut]
    upper_axial = upper @ axis
    transverse = upper - upper_axial[:, None] * axis[None, :]
    transverse_center = np.median(transverse, axis=0)

    axial = xyz @ axis
    transverse_all = xyz - axial[:, None] * axis[None, :]
    radial = np.linalg.norm(transverse_all - transverse_center[None, :], axis=1)
    rim_low = float(np.quantile(axial_visible, 0.972))
    rim_high = float(np.quantile(axial_visible, 0.999))
    rim_band = opaque & (axial >= rim_low) & (axial <= rim_high)
    if int(np.count_nonzero(rim_band)) < 100:
        raise PipelineError("Too few upper-rim Gaussians to infer the opening radius.")
    rim_points = xyz[rim_band]
    projected_rim = np.column_stack((rim_points @ basis_u, rim_points @ basis_v))
    fitted_center = fit_rim_circle_center(projected_rim)
    transverse_center = fitted_center[0] * basis_u + fitted_center[1] * basis_v
    transverse_all = xyz - axial[:, None] * axis[None, :]
    radial = np.linalg.norm(transverse_all - transverse_center[None, :], axis=1)
    rim_outer_radius = float(np.quantile(radial[rim_band], 0.82))
    if not math.isfinite(rim_outer_radius) or not 0.02 * height < rim_outer_radius < 2.0 * height:
        raise PipelineError("Inferred rim radius is implausible; refusing to patch.")

    opening_radius = rim_outer_radius * opening_radius_scale
    patch_radius = opening_radius * patch_radius_scale
    edge_axial = float(axial_rim - edge_drop_scale * rim_outer_radius)
    center_axial = float(edge_axial - bowl_depth_scale * rim_outer_radius)

    color_band = (
        opaque
        & (radial >= patch_radius * 0.92)
        & (radial <= rim_outer_radius * 1.08)
        & (axial >= center_axial - 0.15 * rim_outer_radius)
        & (axial <= axial_rim + 0.05 * rim_outer_radius)
    )
    if int(np.count_nonzero(color_band)) < 100:
        color_band = rim_band
    dc = np.column_stack(
        (vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"])
    ).astype(np.float64, copy=False)
    rgb = np.clip(dc * SH_C0 + 0.5, 0.0, 1.0)
    source_rgb = np.median(rgb[color_band], axis=0)
    return InteriorGeometry(
        axis=axis,
        basis_u=basis_u,
        basis_v=basis_v,
        transverse_center=transverse_center,
        axial_bottom=float(axial_bottom),
        axial_rim=float(axial_rim),
        rim_outer_radius=rim_outer_radius,
        opening_radius=opening_radius,
        patch_radius=patch_radius,
        edge_axial=edge_axial,
        center_axial=center_axial,
        source_rgb=source_rgb,
    )


def patch_surface_axial(radial: np.ndarray, geometry: InteriorGeometry) -> np.ndarray:
    ratio = np.clip(radial / geometry.patch_radius, 0.0, 1.0)
    return geometry.center_axial + (geometry.edge_axial - geometry.center_axial) * ratio**2


def removal_mask(vertices: np.ndarray, geometry: InteriorGeometry) -> np.ndarray:
    xyz = vertex_xyz(vertices)
    axial = xyz @ geometry.axis
    transverse = xyz - axial[:, None] * geometry.axis[None, :]
    radial = np.linalg.norm(transverse - geometry.transverse_center[None, :], axis=1)
    surface = patch_surface_axial(radial, geometry)
    tolerance = 0.06 * geometry.rim_outer_radius
    return (
        (radial <= geometry.patch_radius * 1.03)
        & (axial >= surface - tolerance)
        & (axial <= geometry.axial_rim + 0.20 * geometry.rim_outer_radius)
    )


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert one proper 3x3 rotation matrix to a normalized wxyz quaternion."""

    trace = float(np.trace(rotation))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        quaternion = np.asarray(
            [0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s,
             (rotation[0, 2] - rotation[2, 0]) / s,
             (rotation[1, 0] - rotation[0, 1]) / s]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            quaternion = np.asarray(
                [(rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s,
                 (rotation[0, 1] + rotation[1, 0]) / s,
                 (rotation[0, 2] + rotation[2, 0]) / s]
            )
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            quaternion = np.asarray(
                [(rotation[0, 2] - rotation[2, 0]) / s,
                 (rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s,
                 (rotation[1, 2] + rotation[2, 1]) / s]
            )
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            quaternion = np.asarray(
                [(rotation[1, 0] - rotation[0, 1]) / s,
                 (rotation[0, 2] + rotation[2, 0]) / s,
                 (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    return quaternion


def create_patch_vertices(
    dtype: np.dtype,
    geometry: InteriorGeometry,
    count: int,
    opacity: float,
    color_scale: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    index = np.arange(count, dtype=np.float64)
    radius = geometry.patch_radius * np.sqrt((index + 0.5) / count)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    angle = index * golden_angle
    radial_vectors = (
        np.cos(angle)[:, None] * geometry.basis_u[None, :]
        + np.sin(angle)[:, None] * geometry.basis_v[None, :]
    )
    axial = patch_surface_axial(radius, geometry)
    xyz = (
        geometry.transverse_center[None, :]
        + radius[:, None] * radial_vectors
        + axial[:, None] * geometry.axis[None, :]
    )

    patch = np.zeros(count, dtype=dtype)
    patch["x"], patch["y"], patch["z"] = xyz.T.astype(np.float32)
    edge_ratio = radius / geometry.patch_radius
    brightness = color_scale * (0.78 + 0.22 * edge_ratio**1.5)
    brightness *= np.clip(1.0 + rng.normal(0.0, 0.018, count), 0.92, 1.08)
    colors = np.clip(geometry.source_rgb[None, :] * brightness[:, None], 0.02, 0.95)
    dc = (colors - 0.5) / SH_C0
    patch["f_dc_0"], patch["f_dc_1"], patch["f_dc_2"] = dc.T.astype(np.float32)

    area_per_point = math.pi * geometry.patch_radius**2 / count
    spacing = math.sqrt(area_per_point)
    tangent_scale = max(spacing * 1.35, 1e-7)
    normal_scale = max(spacing * 0.28, 1e-7)
    patch["scale_0"] = math.log(tangent_scale)
    patch["scale_1"] = math.log(tangent_scale)
    patch["scale_2"] = math.log(normal_scale)
    rotation = np.column_stack((geometry.basis_u, geometry.basis_v, geometry.axis))
    quaternion = rotation_matrix_to_quaternion(rotation)
    for offset in range(4):
        patch[f"rot_{offset}"] = quaternion[offset]
    patch["opacity"] = math.log(opacity / (1.0 - opacity))
    return patch


def updated_header(lines: tuple[str, ...], vertex_count: int) -> bytes:
    output: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith("element vertex "):
            output.append(f"element vertex {vertex_count}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        raise PipelineError("Could not update PLY vertex count.")
    return ("\n".join(output) + "\n").encode("ascii")


def write_ply(path: Path, ply: PlyData, vertices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("wb") as destination:
            destination.write(updated_header(ply.header_lines, len(vertices)))
            vertices.tofile(destination)
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def read_export_normalization(manifest_path: Path) -> tuple[np.ndarray, float]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"Could not read export manifest: {manifest_path}") from error
    center = np.asarray(manifest.get("normalization_center"), dtype=np.float64)
    scale = manifest.get("normalization_scale")
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise PipelineError("Export manifest contains an invalid normalization center.")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise PipelineError("Export manifest contains an invalid normalization scale.")
    return center, float(scale)


def print_plan(plan: PatchPlan, dry_run: bool) -> None:
    print("3DGS cosmetic interior patch " + ("dry run" if dry_run else "complete"))
    print(f"Source Gaussians: {plan.source_gaussians}")
    print(f"Remove/add/output: {plan.removed_gaussians}/{plan.added_gaussians}/{plan.output_gaussians}")
    print("Top axis: " + ", ".join(f"{value:.6f}" for value in plan.axis))
    print(f"Outer rim/opening/patch radius: {plan.rim_outer_radius:.6f}/{plan.opening_radius:.6f}/{plan.patch_radius:.6f}")
    print(f"Rim/edge/center axial: {plan.rim_axial:.6f}/{plan.edge_axial:.6f}/{plan.center_axial:.6f}")
    print("Sampled inner color: " + ", ".join(f"{value:.4f}" for value in plan.source_rgb))
    print(f"Output: {plan.destination_ply}")
    if dry_run:
        print("Dry run passed. No patched export was created.")


def patch(args: argparse.Namespace) -> int:
    validate_args(args)
    config_path = resolve_config_path(args.config)
    paths = resolve_paths(config_path, load_yaml(config_path))
    run = resolve_complete_run(paths, args.run_name)
    source_ply = export_path(run, "ply")
    if not source_ply.is_file():
        raise PipelineError(f"Full PLY export was not found: {source_ply}")
    export_manifest = source_ply.parent / "export_manifest.json"
    normalization_center, normalization_scale = read_export_normalization(export_manifest)
    destination_dir = run.run_dir / "exports" / args.patch_name
    destination_ply = destination_dir / f"thai_ceramics_{run.run_name}_{args.patch_name}.ply"
    destination_manifest = destination_dir / "interior_patch_manifest.json"
    if destination_ply.exists() or destination_manifest.exists():
        raise PipelineError(f"Refusing to overwrite existing interior patch: {destination_dir}")

    ply = read_ply(source_ply)
    camera_centers = read_colmap_camera_centers(paths.sparse_model / "images.bin")
    axis = normalized_axis(camera_centers, normalization_center, normalization_scale)
    geometry = infer_geometry(
        ply.vertices,
        axis,
        args.opening_radius_scale,
        args.patch_radius_scale,
        args.edge_drop_scale,
        args.bowl_depth_scale,
    )
    remove = removal_mask(ply.vertices, geometry)
    removed = int(np.count_nonzero(remove))
    if removed < 10:
        raise PipelineError("Automatic interior selection removed too few Gaussians; refusing to patch.")
    if removed > int(0.20 * ply.vertex_count):
        raise PipelineError("Automatic interior selection is too broad; refusing to patch.")
    output_count = ply.vertex_count - removed + args.patch_gaussians
    plan = PatchPlan(
        source_ply=str(source_ply),
        destination_ply=str(destination_ply),
        source_gaussians=ply.vertex_count,
        removed_gaussians=removed,
        added_gaussians=args.patch_gaussians,
        output_gaussians=output_count,
        axis=[float(value) for value in geometry.axis],
        rim_axial=geometry.axial_rim,
        rim_outer_radius=geometry.rim_outer_radius,
        opening_radius=geometry.opening_radius,
        patch_radius=geometry.patch_radius,
        edge_axial=geometry.edge_axial,
        center_axial=geometry.center_axial,
        source_rgb=[float(value) for value in geometry.source_rgb],
    )
    print_plan(plan, args.dry_run)
    if args.dry_run:
        return 0

    patch_vertices = create_patch_vertices(
        ply.vertices.dtype,
        geometry,
        args.patch_gaussians,
        args.opacity,
        args.color_scale,
        args.seed,
    )
    output_vertices = np.concatenate((ply.vertices[~remove], patch_vertices))
    write_ply(destination_ply, ply, output_vertices)
    if not destination_ply.is_file() or destination_ply.stat().st_size == 0:
        raise PipelineError("Patched PLY was not created correctly.")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation": "cosmetic_interior_patch",
        "warning": "The added interior is a visual approximation, not reconstructed geometry.",
        "run_name": run.run_name,
        "source_ply": str(source_ply),
        "source_sha256": sha256_file(source_ply),
        "destination_ply": str(destination_ply),
        "destination_bytes": destination_ply.stat().st_size,
        "destination_sha256": sha256_file(destination_ply),
        "settings": {
            "patch_gaussians": args.patch_gaussians,
            "opening_radius_scale": args.opening_radius_scale,
            "patch_radius_scale": args.patch_radius_scale,
            "edge_drop_scale": args.edge_drop_scale,
            "bowl_depth_scale": args.bowl_depth_scale,
            "color_scale": args.color_scale,
            "opacity": args.opacity,
            "seed": args.seed,
        },
        "plan": asdict(plan),
    }
    destination_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {destination_manifest}")
    return 0


def main() -> int:
    try:
        return patch(parse_args())
    except (PipelineError, OSError, ValueError, struct.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
