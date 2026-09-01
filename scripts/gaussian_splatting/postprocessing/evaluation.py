#!/usr/bin/env python3
"""Evaluate a complete 3DGS checkpoint on the deterministic holdout split."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from ..core.common import (
    DEFAULT_CONFIG,
    PipelineError,
    load_yaml,
    resolve_config_path,
    resolve_paths,
    section,
)
from ..data.scene import load_scene, load_view
from .checkpoint import (
    checkpoint_settings,
    load_checkpoint_cpu,
    resolve_complete_run,
)
from ..training.runner import atomic_write_json, rasterize_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one complete 3DGS run on all held-out images."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default="baseline_7k")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    if not values:
        raise PipelineError("Cannot summarize an empty metric list.")
    return float(sum(values) / len(values))


def evaluate(args: argparse.Namespace) -> int:
    import imageio.v2 as imageio
    import torch
    from torchmetrics.functional.image import structural_similarity_index_measure
    from tqdm import tqdm

    config_path = resolve_config_path(args.config)
    config = load_yaml(config_path)
    paths = resolve_paths(config_path, config)
    run = resolve_complete_run(paths, args.run_name)
    checkpoint = load_checkpoint_cpu(run.checkpoint)
    settings = checkpoint_settings(checkpoint)
    scene = load_scene(paths, settings.image_factor)
    evaluation_config = section(config, "evaluation")
    alpha_threshold = float(evaluation_config.get("alpha_threshold", 1.0 / 255.0))
    if not 0.0 < alpha_threshold < 1.0:
        raise PipelineError("evaluation.alpha_threshold must be between zero and one.")
    output_dir = run.run_dir / "evaluation" / "holdout_black"
    if output_dir.exists():
        raise PipelineError(f"Evaluation output already exists: {output_dir}")

    gaussian_count = int(checkpoint["splats"]["means"].shape[0])
    print("3DGS held-out evaluation plan")
    print(f"Run: {run.run_name}")
    print(f"Checkpoint: {run.checkpoint}")
    print(f"Resolution: {scene.width}x{scene.height}")
    print(f"Held-out images: {len(scene.test_records)}")
    print(f"Gaussians: {gaussian_count}")
    print(f"Output: {output_dir}")
    if args.dry_run:
        print("Dry run passed. No evaluation renders or outputs were created.")
        return 0

    torch.cuda.set_device(0)
    device = "cuda:0"
    torch.cuda.reset_peak_memory_stats()
    splats = {
        key: value.to(device)
        for key, value in checkpoint["splats"].items()
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    comparisons = output_dir / "comparisons"
    comparisons.mkdir()
    shutil.copyfile(config_path, output_dir / "config_snapshot.yml")

    rows: list[dict[str, float | str]] = []
    started = time.time()
    try:
        with torch.no_grad():
            for index, record in enumerate(
                tqdm(scene.test_records, desc="held_out_evaluation")
            ):
                view = load_view(record)
                target = (
                    view["image"].to(device)
                    * view["alpha"].to(device)[..., None]
                )
                rendered, rendered_alpha, _ = rasterize_view(
                    splats, view, settings
                )
                rendered = rendered[0].clamp(0.0, 1.0)
                rendered_alpha = rendered_alpha[0, ..., 0].clamp(0.0, 1.0)
                difference = rendered - target
                mse = difference.square().mean().clamp_min(1e-12)
                psnr = -10.0 * torch.log10(mse)
                ssim = structural_similarity_index_measure(
                    rendered.permute(2, 0, 1).unsqueeze(0),
                    target.permute(2, 0, 1).unsqueeze(0),
                    data_range=1.0,
                )

                target_alpha = view["alpha"].to(device)
                foreground = target_alpha >= alpha_threshold
                foreground_count = foreground.sum().clamp_min(1)
                foreground_mse = (
                    difference.square() * foreground[..., None]
                ).sum() / (foreground_count * 3)
                foreground_psnr = -10.0 * torch.log10(
                    foreground_mse.clamp_min(1e-12)
                )
                foreground_l1 = (
                    difference.abs() * foreground[..., None]
                ).sum() / (foreground_count * 3)
                predicted_foreground = rendered_alpha >= alpha_threshold
                intersection = (foreground & predicted_foreground).sum()
                union = (foreground | predicted_foreground).sum().clamp_min(1)
                alpha_iou = intersection / union

                row = {
                    "image": record.name,
                    "psnr": float(psnr.cpu()),
                    "ssim": float(ssim.cpu()),
                    "foreground_psnr": float(foreground_psnr.cpu()),
                    "foreground_l1": float(foreground_l1.cpu()),
                    "alpha_iou": float(alpha_iou.cpu()),
                }
                rows.append(row)
                comparison = torch.cat([target, rendered], dim=1).cpu().numpy()
                imageio.imwrite(
                    comparisons / f"{index:02d}_{Path(record.name).stem}.png",
                    np.rint(comparison * 255.0).astype(np.uint8),
                )
    except torch.cuda.OutOfMemoryError as error:
        atomic_write_json(
            {
                "status": "failed_out_of_memory",
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            },
            output_dir / "failure.json",
        )
        raise PipelineError(f"CUDA ran out of memory during evaluation: {error}") from error

    metric_path = output_dir / "metrics.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as target_file:
        writer = csv.DictWriter(target_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run.run_name,
        "checkpoint": str(run.checkpoint),
        "profile": asdict(settings),
        "held_out_images": len(rows),
        "resolution": [scene.width, scene.height],
        "gaussians": gaussian_count,
        "mean_psnr": mean([float(row["psnr"]) for row in rows]),
        "mean_ssim": mean([float(row["ssim"]) for row in rows]),
        "mean_foreground_psnr": mean(
            [float(row["foreground_psnr"]) for row in rows]
        ),
        "mean_foreground_l1": mean(
            [float(row["foreground_l1"]) for row in rows]
        ),
        "mean_alpha_iou": mean([float(row["alpha_iou"]) for row in rows]),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(summary, output_dir / "evaluation_summary.json")
    print("3DGS held-out evaluation complete")
    print(f"Mean PSNR: {summary['mean_psnr']:.3f} dB")
    print(f"Mean SSIM: {summary['mean_ssim']:.5f}")
    print(f"Foreground PSNR: {summary['mean_foreground_psnr']:.3f} dB")
    print(f"Alpha IoU: {summary['mean_alpha_iou']:.5f}")
    print(f"Peak VRAM: {summary['peak_vram_gb']:.3f} GB")
    print(f"Results: {output_dir}")
    return 0


def main() -> int:
    try:
        return evaluate(parse_args())
    except PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
