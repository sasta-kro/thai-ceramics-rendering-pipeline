#!/usr/bin/env python3
"""Run a guarded low-memory gsplat training profile."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import time
from typing import Any, Mapping

import numpy as np

from ..core.common import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    PipelineError,
    load_yaml,
    profile,
    required_bool,
    required_positive_int,
    required_positive_number,
    required_string,
    resolve_child,
    resolve_config_path,
    resolve_paths,
    section,
)
from ..data.scene import SceneData, load_scene, load_view


SH_C0 = 0.28209479177387814
SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class TrainingSettings:
    profile_name: str
    image_factor: int
    max_steps: int
    max_gaussians: int
    sh_degree: int
    sh_degree_interval: int
    packed: bool
    sparse_grad: bool
    disable_viewer: bool
    refine_start_iter: int
    refine_stop_iter: int
    refine_every: int
    seed: int
    init_opacity: float
    init_scale: float
    means_lr: float
    scales_lr: float
    opacities_lr: float
    quats_lr: float
    sh0_lr: float
    shn_lr: float
    ssim_lambda: float
    near_plane: float
    far_plane: float
    random_background: bool
    log_every: int
    preview_views: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one guarded 3DGS training profile."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--run-name",
        help="Optional output directory name; defaults to the profile name.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs/settings and print the plan without creating outputs.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Acknowledge and start the configured training job.",
    )
    return parser.parse_args()


def nonnegative_int(values: Mapping[str, Any], key: str, section_name: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PipelineError(f"'{section_name}.{key}' must be a non-negative integer.")
    return value


def load_training_settings(
    config: Mapping[str, Any], profile_name: str
) -> TrainingSettings:
    values = profile(config, profile_name)
    training = section(config, "training")
    if required_string(training, "strategy", "training").lower() != "mcmc":
        raise PipelineError("Only the capped MCMC strategy is supported by this runner.")

    packed = required_bool(values, "packed", f"profiles.{profile_name}")
    sparse_grad = required_bool(values, "sparse_grad", f"profiles.{profile_name}")
    disable_viewer = required_bool(
        values, "disable_viewer", f"profiles.{profile_name}"
    )
    if sparse_grad and not packed:
        raise PipelineError("Sparse gradients require packed rasterization.")
    if not disable_viewer:
        raise PipelineError("Training-time viewer must remain disabled on the 4 GB GPU.")

    ssim_lambda = required_positive_number(training, "ssim_lambda", "training")
    if not 0 < ssim_lambda < 1:
        raise PipelineError("training.ssim_lambda must be between zero and one.")

    return TrainingSettings(
        profile_name=profile_name,
        image_factor=required_positive_int(
            values, "image_factor", f"profiles.{profile_name}"
        ),
        max_steps=required_positive_int(values, "max_steps", f"profiles.{profile_name}"),
        max_gaussians=required_positive_int(
            values, "max_gaussians", f"profiles.{profile_name}"
        ),
        sh_degree=nonnegative_int(values, "sh_degree", f"profiles.{profile_name}"),
        sh_degree_interval=required_positive_int(
            values, "sh_degree_interval", f"profiles.{profile_name}"
        ),
        packed=packed,
        sparse_grad=sparse_grad,
        disable_viewer=disable_viewer,
        refine_start_iter=nonnegative_int(
            values, "refine_start_iter", f"profiles.{profile_name}"
        ),
        refine_stop_iter=nonnegative_int(
            values, "refine_stop_iter", f"profiles.{profile_name}"
        ),
        refine_every=required_positive_int(
            values, "refine_every", f"profiles.{profile_name}"
        ),
        seed=nonnegative_int(training, "seed", "training"),
        init_opacity=required_positive_number(training, "init_opacity", "training"),
        init_scale=required_positive_number(training, "init_scale", "training"),
        means_lr=required_positive_number(training, "means_lr", "training"),
        scales_lr=required_positive_number(training, "scales_lr", "training"),
        opacities_lr=required_positive_number(training, "opacities_lr", "training"),
        quats_lr=required_positive_number(training, "quats_lr", "training"),
        sh0_lr=required_positive_number(training, "sh0_lr", "training"),
        shn_lr=required_positive_number(training, "shn_lr", "training"),
        ssim_lambda=ssim_lambda,
        near_plane=required_positive_number(training, "near_plane", "training"),
        far_plane=required_positive_number(training, "far_plane", "training"),
        random_background=required_bool(
            training, "random_background", "training"
        ),
        log_every=required_positive_int(training, "log_every", "training"),
        preview_views=required_positive_int(
            training, "preview_views", "training"
        ),
    )


def validated_run_name(raw_name: str) -> str:
    if not SAFE_RUN_NAME.fullmatch(raw_name):
        raise PipelineError(
            "Run name may contain only letters, numbers, dot, underscore, and hyphen."
        )
    return raw_name


def reset_cuda_peak_memory_stats(torch_module: Any) -> None:
    """Reset counters for the current CUDA device.

    The Windows PyTorch 2.4 build used by this project rejects an explicit device
    argument here, even though its public API documents that form.
    """
    torch_module.cuda.reset_peak_memory_stats()


def peak_cuda_memory_gb(torch_module: Any) -> float:
    """Return peak allocated memory for the current CUDA device in GiB."""
    return torch_module.cuda.max_memory_allocated() / 1024**3


def atomic_write_json(data: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def create_splats(scene: SceneData, settings: TrainingSettings, device: str):
    import torch
    from scipy.spatial import cKDTree

    points = torch.from_numpy(scene.points).float()
    colors_rgb = torch.from_numpy(scene.point_colors).float()
    distances, _ = cKDTree(scene.points).query(scene.points, k=4)
    rms_distance = np.sqrt(np.mean(np.square(distances[:, 1:]), axis=1))
    rms_distance = np.clip(rms_distance, 1e-6, None)
    scales = torch.from_numpy(np.log(rms_distance * settings.init_scale)).float()
    scales = scales[:, None].repeat(1, 3)

    generator = torch.Generator(device="cpu").manual_seed(settings.seed)
    quats = torch.rand((len(points), 4), generator=generator)
    opacities = torch.logit(
        torch.full((len(points),), settings.init_opacity, dtype=torch.float32)
    )
    sh = torch.zeros((len(points), (settings.sh_degree + 1) ** 2, 3))
    sh[:, 0, :] = (colors_rgb - 0.5) / SH_C0

    parameter_values = [
        ("means", points, settings.means_lr),
        ("scales", scales, settings.scales_lr),
        ("quats", quats, settings.quats_lr),
        ("opacities", opacities, settings.opacities_lr),
        ("sh0", sh[:, :1, :], settings.sh0_lr),
        ("shN", sh[:, 1:, :], settings.shn_lr),
    ]
    splats = torch.nn.ParameterDict(
        {name: torch.nn.Parameter(value) for name, value, _ in parameter_values}
    ).to(device)
    optimizer_class = torch.optim.SparseAdam if settings.sparse_grad else torch.optim.Adam
    optimizers = {
        name: optimizer_class([{"params": splats[name], "lr": lr, "name": name}], eps=1e-15)
        for name, _, lr in parameter_values
    }
    return splats, optimizers


def rasterize_view(splats, view: Mapping[str, Any], settings: TrainingSettings):
    import torch
    from gsplat import rasterization

    camtoworld = view["camtoworld"].to(splats["means"].device).unsqueeze(0)
    K = view["K"].to(splats["means"].device).unsqueeze(0)
    image = view["image"].to(splats["means"].device)
    height, width = image.shape[:2]
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    rendered, alphas, info = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=colors,
        viewmats=torch.linalg.inv(camtoworld),
        Ks=K,
        width=width,
        height=height,
        near_plane=settings.near_plane,
        far_plane=settings.far_plane,
        sh_degree=settings.sh_degree,
        packed=settings.packed,
        sparse_grad=settings.sparse_grad,
        # Composite backgrounds after rasterization. gsplat 1.5.3's packed CUDA
        # wrapper rejects the documented batched background shape.
        backgrounds=None,
        render_mode="RGB",
        camera_model="pinhole",
    )
    return rendered, alphas, info


def composite_background(rendered, alpha, background):
    """Composite a premultiplied raster over one RGB background color."""
    return rendered + background.view(1, 1, 3) * (1.0 - alpha)


def sparse_parameter_gradients(splats, info: Mapping[str, Any]) -> None:
    import torch

    gaussian_ids = info["gaussian_ids"]
    for parameter in splats.values():
        gradient = parameter.grad
        if gradient is None or gradient.is_sparse:
            continue
        parameter.grad = torch.sparse_coo_tensor(
            indices=gaussian_ids[None],
            values=gradient[gaussian_ids],
            size=parameter.size(),
            is_coalesced=True,
        )


def save_preview(run_dir: Path, scene: SceneData, splats, settings: TrainingSettings) -> None:
    import imageio.v2 as imageio
    import torch

    preview_dir = run_dir / "renders" / "smoke_test"
    preview_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, record in enumerate(scene.test_records[: settings.preview_views]):
            view = load_view(record)
            rendered, _, _ = rasterize_view(splats, view, settings)
            target = view["image"] * view["alpha"][..., None]
            comparison = torch.cat(
                [target, rendered[0].detach().cpu().clamp(0, 1)], dim=1
            )
            imageio.imwrite(
                preview_dir / f"{index:02d}_{Path(record.name).stem}.png",
                np.rint(comparison.numpy() * 255.0).astype(np.uint8),
            )


def train(
    run_dir: Path,
    scene: SceneData,
    settings: TrainingSettings,
    config_path: Path,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from gsplat.strategy import MCMCStrategy
    from torchmetrics.functional.image import structural_similarity_index_measure
    from tqdm import trange

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    cuda_device_index = 0
    torch.cuda.set_device(cuda_device_index)
    device = f"cuda:{cuda_device_index}"
    reset_cuda_peak_memory_stats(torch)

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "logs").mkdir()
    shutil.copyfile(config_path, run_dir / "config_snapshot.yml")
    atomic_write_json(
        {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "profile": asdict(settings),
            "train_images": len(scene.train_records),
            "test_images": len(scene.test_records),
            "initial_points": len(scene.points),
            "normalization_center": scene.normalization_center.tolist(),
            "normalization_scale": scene.normalization_scale,
        },
        run_dir / "run_manifest.json",
    )

    splats, optimizers = create_splats(scene, settings, device)
    strategy = MCMCStrategy(
        cap_max=settings.max_gaussians,
        refine_start_iter=settings.refine_start_iter,
        refine_stop_iter=settings.refine_stop_iter,
        refine_every=settings.refine_every,
        verbose=True,
    )
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / settings.max_steps)
    )

    log_path = run_dir / "logs" / "training.csv"
    log_path.write_text("step,loss,l1,ssim,gaussians,vram_gb\n", encoding="utf-8")
    start = time.time()
    final_loss = math.nan

    try:
        progress = trange(settings.max_steps, desc=settings.profile_name)
        for step in progress:
            record = scene.train_records[random.randrange(len(scene.train_records))]
            view = load_view(record)
            image = view["image"].to(device)
            alpha = view["alpha"].to(device)[..., None]
            background = (
                torch.rand((3,), device=device)
                if settings.random_background
                else torch.zeros((3,), device=device)
            )
            target = image * alpha + background.view(1, 1, 3) * (1.0 - alpha)

            sh_degree = min(step // settings.sh_degree_interval, settings.sh_degree)
            step_settings = TrainingSettings(
                **{**asdict(settings), "sh_degree": sh_degree}
            )
            rendered, rendered_alpha, info = rasterize_view(
                splats, view, step_settings
            )
            rendered = composite_background(
                rendered[0], rendered_alpha[0], background
            ).clamp(0.0, 1.0)
            l1 = F.l1_loss(rendered, target)
            ssim = structural_similarity_index_measure(
                rendered.permute(2, 0, 1).unsqueeze(0),
                target.permute(2, 0, 1).unsqueeze(0),
                data_range=1.0,
            )
            loss = (1.0 - settings.ssim_lambda) * l1 + settings.ssim_lambda * (1.0 - ssim)
            loss.backward()

            if settings.sparse_grad:
                sparse_parameter_gradients(splats, info)
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            strategy.step_post_backward(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                step=step,
                info=info,
                lr=scheduler.get_last_lr()[0],
            )

            final_loss = float(loss.detach().cpu())
            if step % settings.log_every == 0 or step == settings.max_steps - 1:
                vram = peak_cuda_memory_gb(torch)
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"{step + 1},{final_loss:.8f},{float(l1):.8f},"
                        f"{float(ssim):.8f},{len(splats['means'])},{vram:.4f}\n"
                    )
                progress.set_postfix(loss=f"{final_loss:.4f}", vram=f"{vram:.2f}GB")
    except torch.cuda.OutOfMemoryError as error:
        atomic_write_json(
            {
                "status": "failed_out_of_memory",
                "peak_vram_gb": peak_cuda_memory_gb(torch),
                "recommendation": "Use factor 4 and a lower Gaussian cap.",
            },
            run_dir / "failure.json",
        )
        raise PipelineError(f"CUDA ran out of memory: {error}") from error

    checkpoint = {
        "schema_version": 1,
        "step": settings.max_steps,
        "profile": asdict(settings),
        "splats": {name: value.detach().cpu() for name, value in splats.items()},
        "normalization_center": scene.normalization_center,
        "normalization_scale": scene.normalization_scale,
    }
    torch.save(
        checkpoint,
        run_dir / "checkpoints" / f"step_{settings.max_steps:06d}.pt",
    )
    save_preview(run_dir, scene, splats, settings)
    elapsed = time.time() - start
    summary = {
        "status": "complete",
        "steps": settings.max_steps,
        "elapsed_seconds": elapsed,
        "final_loss": final_loss,
        "gaussians": len(splats["means"]),
        "peak_vram_gb": peak_cuda_memory_gb(torch),
    }
    atomic_write_json(summary, run_dir / "training_summary.json")
    return summary


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_config_path(args.config)
        config = load_yaml(config_path)
        paths = resolve_paths(config_path, config)
        settings = load_training_settings(config, args.profile)
        run_name = validated_run_name(args.run_name or args.profile)
        run_dir = resolve_child(paths.runs, run_name, "run_name")
        scene = load_scene(paths, settings.image_factor)
        if run_dir.exists():
            raise PipelineError(
                f"Run directory already exists; choose a new --run-name: {run_dir}"
            )

        print("3DGS training plan")
        print(f"Profile: {settings.profile_name}")
        print(f"Prepared factor: {settings.image_factor}")
        print(f"Resolution: {scene.width}x{scene.height}")
        print(f"Train/test images: {len(scene.train_records)}/{len(scene.test_records)}")
        print(f"COLMAP initialization points: {len(scene.points)}")
        print(f"Steps: {settings.max_steps}")
        print(f"Gaussian cap: {settings.max_gaussians}")
        print(f"Packed/sparse gradients: {settings.packed}/{settings.sparse_grad}")
        print(f"Output: {run_dir}")
        if args.dry_run:
            print("Dry run passed. No training was started and no outputs were created.")
            return 0

        summary = train(run_dir, scene, settings, config_path)
        print("3DGS training complete")
        print(f"Steps: {summary['steps']}")
        print(f"Final loss: {summary['final_loss']:.6f}")
        print(f"Gaussians: {summary['gaussians']}")
        print(f"Peak VRAM: {summary['peak_vram_gb']:.3f} GB")
        print(f"Elapsed: {summary['elapsed_seconds'] / 60:.2f} minutes")
        print(f"Results: {run_dir}")
        return 0
    except PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
