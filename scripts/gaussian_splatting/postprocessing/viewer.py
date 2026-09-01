#!/usr/bin/env python3
"""Launch a local, checkpoint-backed browser viewer for one 3DGS run."""

from __future__ import annotations

import argparse
from pathlib import Path
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
from ..data.scene import load_scene
from .checkpoint import (
    checkpoint_settings,
    load_checkpoint_cpu,
    resolve_complete_run,
    viewer_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View one complete 3DGS checkpoint in a local browser."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default="baseline_7k")
    parser.add_argument("--port", type=int)
    parser.add_argument("--max-gaussians", type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def view(args: argparse.Namespace) -> int:
    config_path = resolve_config_path(args.config)
    config = load_yaml(config_path)
    paths = resolve_paths(config_path, config)
    run = resolve_complete_run(paths, args.run_name)
    checkpoint = load_checkpoint_cpu(run.checkpoint)
    settings = checkpoint_settings(checkpoint)
    scene = load_scene(paths, settings.image_factor)
    viewer_config = section(config, "viewer")
    port = args.port if args.port is not None else int(viewer_config.get("port", 8080))
    max_gaussians = (
        args.max_gaussians
        if args.max_gaussians is not None
        else int(viewer_config.get("max_gaussians", 500000))
    )
    opacity_threshold = float(viewer_config.get("opacity_threshold", 0.005))
    if not 1 <= port <= 65535:
        raise PipelineError("Viewer port must be between 1 and 65535.")
    if max_gaussians <= 0:
        raise PipelineError("Viewer Gaussian limit must be positive.")

    total_gaussians = int(checkpoint["splats"]["means"].shape[0])
    print("Local 3DGS viewer plan")
    print(f"Run: {run.run_name}")
    print(f"Checkpoint: {run.checkpoint}")
    print(f"Available Gaussians: {total_gaussians}")
    print(f"Viewer limit: {max_gaussians}")
    print(f"Opacity threshold: {opacity_threshold}")
    print(f"Address: http://localhost:{port}")
    if args.dry_run:
        print("Dry run passed. No viewer server was started.")
        return 0

    try:
        import viser
    except (ImportError, ModuleNotFoundError) as error:
        raise PipelineError("viser is unavailable in this environment.") from error

    centers, covariances, colors, opacities = viewer_arrays(
        checkpoint, opacity_threshold, max_gaussians
    )
    first_camera = scene.test_records[0].camtoworld
    camera_position = first_camera[:3, 3].astype(np.float64)
    camera_up = (-first_camera[:3, 1]).astype(np.float64)
    visible_center = np.median(centers, axis=0).astype(np.float64)

    server = viser.ViserServer(
        host="127.0.0.1",
        port=port,
        label="Thai Ceramics 3D Gaussian Splatting",
    )
    server.scene.set_up_direction(camera_up)
    server.scene.add_gaussian_splats(
        "/thai_ceramics",
        centers=centers,
        covariances=covariances,
        rgbs=colors,
        opacities=opacities,
    )

    @server.on_client_connect
    def initialize_camera(client) -> None:
        client.camera.position = camera_position
        client.camera.look_at = visible_center
        client.camera.up_direction = camera_up
        client.camera.near = 0.01
        client.camera.far = 100.0

    print(f"Loaded {len(centers)} Gaussians.")
    print(f"Open http://localhost:{port} and orbit with the mouse.")
    print("Press Ctrl+C in this terminal to stop the viewer.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Viewer stopped.")
        return 0


def main() -> int:
    try:
        return view(parse_args())
    except PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
