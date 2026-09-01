"""Shared validation and conversion helpers for trained 3DGS checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..core.common import GaussianSplattingPaths, PipelineError, resolve_child
from ..data.scene import read_json
from ..training.runner import SH_C0, TrainingSettings, validated_run_name


REQUIRED_SPLAT_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")
EXPORT_SUFFIXES = {
    "ply": ".ply",
    "splat": ".splat",
    "ply_compressed": ".compressed.ply",
}


@dataclass(frozen=True)
class CompleteRun:
    run_name: str
    run_dir: Path
    checkpoint: Path
    step: int
    profile_name: str
    summary: Mapping[str, Any]


def resolve_complete_run(
    paths: GaussianSplattingPaths, run_name: str
) -> CompleteRun:
    """Resolve a complete run and its exact final checkpoint."""

    safe_name = validated_run_name(run_name)
    run_dir = resolve_child(paths.runs, safe_name, "run_name")
    if not run_dir.is_dir():
        raise PipelineError(f"Completed run directory was not found: {run_dir}")

    summary = read_json(run_dir / "training_summary.json")
    if summary.get("status") != "complete":
        raise PipelineError(f"Run is not marked complete: {run_dir}")
    step = summary.get("steps")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise PipelineError("Training summary contains an invalid final step.")

    manifest = read_json(run_dir / "run_manifest.json")
    profile = manifest.get("profile")
    if not isinstance(profile, Mapping):
        raise PipelineError("Run manifest is missing its training profile.")
    profile_name = profile.get("profile_name")
    if not isinstance(profile_name, str) or not profile_name:
        raise PipelineError("Run manifest contains an invalid profile name.")

    checkpoint = run_dir / "checkpoints" / f"step_{step:06d}.pt"
    if not checkpoint.is_file():
        raise PipelineError(f"Final checkpoint was not found: {checkpoint}")
    return CompleteRun(
        run_name=safe_name,
        run_dir=run_dir,
        checkpoint=checkpoint,
        step=step,
        profile_name=profile_name,
        summary=summary,
    )


def load_checkpoint_cpu(checkpoint_path: Path) -> dict[str, Any]:
    """Load and structurally validate one locally generated checkpoint."""

    try:
        import torch
    except ModuleNotFoundError as error:
        raise PipelineError("PyTorch is required to load a 3DGS checkpoint.") from error

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise PipelineError(f"Could not load checkpoint {checkpoint_path}: {error}") from error
    if not isinstance(checkpoint, dict):
        raise PipelineError("Checkpoint root must be a dictionary.")
    splats = checkpoint.get("splats")
    if not isinstance(splats, Mapping):
        raise PipelineError("Checkpoint is missing its splat tensors.")
    missing = [key for key in REQUIRED_SPLAT_KEYS if key not in splats]
    if missing:
        raise PipelineError("Checkpoint is missing splat tensors: " + ", ".join(missing))

    means = splats["means"]
    if not isinstance(means, torch.Tensor) or means.ndim != 2 or means.shape[1] != 3:
        raise PipelineError("Checkpoint means must have shape [N, 3].")
    count = means.shape[0]
    expected_shapes = {
        "scales": (count, 3),
        "quats": (count, 4),
        "opacities": (count,),
        "sh0": (count, 1, 3),
    }
    for key, expected in expected_shapes.items():
        value = splats[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise PipelineError(
                f"Checkpoint {key} has shape {getattr(value, 'shape', None)}, "
                f"expected {expected}."
            )
    shn = splats["shN"]
    if (
        not isinstance(shn, torch.Tensor)
        or shn.ndim != 3
        or shn.shape[0] != count
        or shn.shape[2] != 3
    ):
        raise PipelineError("Checkpoint shN must have shape [N, K, 3].")
    if count == 0:
        raise PipelineError("Checkpoint contains no Gaussians.")
    if not all(torch.isfinite(splats[key]).all().item() for key in REQUIRED_SPLAT_KEYS):
        raise PipelineError("Checkpoint contains NaN or infinite splat values.")

    profile = checkpoint.get("profile")
    if not isinstance(profile, Mapping):
        raise PipelineError("Checkpoint is missing its training profile.")
    try:
        TrainingSettings(**dict(profile))
    except TypeError as error:
        raise PipelineError(f"Checkpoint training profile is incompatible: {error}") from error
    return checkpoint


def checkpoint_settings(checkpoint: Mapping[str, Any]) -> TrainingSettings:
    return TrainingSettings(**dict(checkpoint["profile"]))


def export_path(run: CompleteRun, export_format: str) -> Path:
    suffix = EXPORT_SUFFIXES.get(export_format)
    if suffix is None:
        choices = ", ".join(EXPORT_SUFFIXES)
        raise PipelineError(f"Unknown export format '{export_format}'. Choose: {choices}.")
    return run.run_dir / "exports" / f"thai_ceramics_{run.run_name}{suffix}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quaternion_covariances(quats, log_scales):
    """Convert wxyz quaternions and log standard deviations to covariances."""

    import torch

    quats = torch.nn.functional.normalize(quats, dim=-1)
    w, x, y, z = quats.unbind(dim=-1)
    rotations = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    scaled_rotations = rotations * torch.exp(log_scales)[:, None, :]
    return scaled_rotations @ scaled_rotations.transpose(1, 2)


def viewer_arrays(
    checkpoint: Mapping[str, Any], opacity_threshold: float, max_gaussians: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare a bounded set of base-color Gaussians for viser's WebGL viewer."""

    import torch

    if not 0.0 <= opacity_threshold < 1.0:
        raise PipelineError("Viewer opacity threshold must be in [0, 1).")
    if max_gaussians <= 0:
        raise PipelineError("Viewer Gaussian limit must be positive.")
    splats = checkpoint["splats"]
    opacities = torch.sigmoid(splats["opacities"])
    selected = torch.nonzero(opacities >= opacity_threshold, as_tuple=False).flatten()
    if len(selected) == 0:
        raise PipelineError("Viewer opacity threshold removed every Gaussian.")
    if len(selected) > max_gaussians:
        ranked = torch.topk(opacities[selected], k=max_gaussians, sorted=False).indices
        selected = selected[ranked]

    centers = splats["means"][selected].float()
    covariances = quaternion_covariances(
        splats["quats"][selected].float(), splats["scales"][selected].float()
    )
    colors = (splats["sh0"][selected, 0, :].float() * SH_C0 + 0.5).clamp(0, 1)
    alpha = opacities[selected, None].float()
    return tuple(
        value.contiguous().numpy()
        for value in (centers, covariances, colors, alpha)
    )
