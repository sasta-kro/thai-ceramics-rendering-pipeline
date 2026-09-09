#!/usr/bin/env python3
"""Prepare the repaired multiview COLMAP model for masked 3DGS.

The runner deliberately exposes one stage at a time. It reuses the guarded
masked-input and RGB-undistortion implementation from run_colmap_dense.py, then
undistorts the binary mask images with the identical COLMAP model/settings and
converts them into pixel-aligned masks for the 3DGS dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import prepare_undistorted_masks as mask_tools  # noqa: E402
import run_colmap_dense as dense  # noqa: E402


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/colmap_undistort_pot1_unglazed_multiview_lightglue_repaired.yml"
)
STAGES = (
    "validate",
    "prepare",
    "undistort-rgb",
    "undistort-masks",
    "finalize-masks",
)
MAX_COLMAP_WINDOWS_PATH = 259


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare repaired LightGlue multiview inputs for masked 3DGS."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--dry-run", action="store_true")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true")
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def mask_workspace(paths: dense.DensePaths, config) -> Path:
    output = dense.section(config, "output")
    return dense.resolve_output_child(
        paths.workspace,
        dense.required_string(
            output, "mask_undistortion_workspace", "output"
        ),
        "output.mask_undistortion_workspace",
    )


def undistortion_arguments(
    paths: dense.DensePaths,
    settings: dense.UndistortionSettings,
    *,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    return [
        str(paths.colmap),
        "image_undistorter",
        "--image_path",
        str(image_path),
        "--input_path",
        str(paths.sparse_model),
        "--output_path",
        str(output_path),
        "--output_type",
        settings.output_type,
        "--max_image_size",
        str(settings.max_image_size),
        "--jpeg_quality",
        str(settings.jpeg_quality),
        "--num_threads",
        str(settings.num_threads),
    ]


def undistortion_output_complete(workspace: Path, expected_count: int) -> bool:
    try:
        dense.validate_sparse_components(workspace / "sparse")
    except dense.PipelineError:
        return False
    return len(dense.image_files(workspace / "images")) == expected_count


def validate_colmap_image_path_lengths(
    input_root: Path,
    output_workspace: Path,
    relative_images: list[Path],
    *,
    max_length: int = MAX_COLMAP_WINDOWS_PATH,
) -> None:
    """Reject image paths that native COLMAP cannot reliably open on Windows."""

    offenders: list[tuple[int, Path]] = []
    for relative in relative_images:
        for candidate in (input_root / relative, output_workspace / "images" / relative):
            resolved = candidate.resolve()
            if len(str(resolved)) > max_length:
                offenders.append((len(str(resolved)), resolved))
    if offenders:
        offenders.sort(reverse=True, key=lambda value: value[0])
        length, path = offenders[0]
        raise dense.PipelineError(
            "COLMAP image path exceeds the safe Windows path limit: "
            f"{length} characters (maximum {max_length}). Shorten the configured "
            f"input/output folder names. Longest path: {path}"
        )


def run_mask_undistortion(
    paths: dense.DensePaths,
    workspace: Path,
    settings: dense.UndistortionSettings,
    expected_count: int,
    resume: bool,
) -> dense.UndistortionReport:
    if resume and undistortion_output_complete(workspace, expected_count):
        return dense.UndistortionReport(expected_count, 0.0, True)
    existing = [path for path in workspace.rglob("*") if path.is_file()]
    if existing:
        raise dense.PipelineError(
            "The mask-undistortion workspace is incomplete but contains files:\n"
            + dense.format_examples(str(path) for path in existing)
        )
    workspace.mkdir(parents=True, exist_ok=True)
    elapsed = dense.run_colmap_logged(
        undistortion_arguments(
            paths,
            settings,
            image_path=paths.mask_undistortion_images,
            output_path=workspace,
        ),
        "Undistort binary mask images",
        paths.logs / "03_undistort_masks.log",
    )
    if not undistortion_output_complete(workspace, expected_count):
        raise dense.PipelineError(
            "COLMAP returned success, but the mask-undistortion workspace is "
            f"incomplete. Inspect {paths.logs / '03_undistort_masks.log'}."
        )
    return dense.UndistortionReport(expected_count, elapsed, False)


def print_source_report(report: dense.ValidationReport) -> None:
    resolutions = ", ".join(
        f"{width}x{height}" for width, height in report.resolutions
    )
    print(f"RGB images: {report.image_count}")
    print(f"Masks: {report.mask_count}")
    print(f"Registered images: {report.registered_image_count}")
    print(f"Resolution(s): {resolutions}")


def main() -> int:
    args = parse_args()
    try:
        config_path = dense.resolve_config_path(args.config)
        config = dense.load_yaml(config_path)
        dense.validate_config_schema(config)
        paths = dense.resolve_paths(config_path, config)
        report = dense.validate_inputs(paths)
        settings = dense.undistortion_settings(config)
        mask_output_workspace = mask_workspace(paths, config)
        print(f"Configuration: {config_path}")
        print(f"Stage: {args.stage}")
        print(f"Source sparse model: {paths.sparse_model}")
        print(f"Undistorted 3DGS dataset: {paths.workspace}")

        if args.stage == "validate":
            print_source_report(report)
            print("Validation passed. No files were created.")
            return 0

        if args.stage == "prepare":
            print_source_report(report)
            if args.dry_run:
                print(f"Masked RGB output: {paths.masked_images}")
                print(f"Mask-image output: {paths.mask_undistortion_images}")
                print("Dry run passed. No files were created.")
                return 0
            prepared = dense.prepare_masked_inputs(
                paths,
                dense.mask_preparation_settings(config),
                resume=args.resume,
            )
            print(
                f"Prepared pairs: {prepared.total}; written {prepared.written}; "
                f"resumed {prepared.resumed}"
            )
            return 0

        prepared_count = dense.validate_prepared_inputs(paths)
        relative_images = [
            path.relative_to(paths.source_images)
            for path in dense.image_files(paths.source_images)
        ]
        if args.stage == "undistort-rgb":
            validate_colmap_image_path_lengths(
                paths.masked_images, paths.workspace, relative_images
            )
            arguments = undistortion_arguments(
                paths,
                settings,
                image_path=paths.masked_images,
                output_path=paths.workspace,
            )
            if args.dry_run:
                print(dense.display_command(arguments))
                print("Dry run passed. COLMAP was not run.")
                return 0
            result = dense.run_rgb_undistortion(
                paths, settings, prepared_count, resume=args.resume
            )
            print(
                f"Undistorted RGB images: {result.image_count}; "
                f"resumed={result.resumed}; elapsed={result.elapsed_seconds:.2f}s"
            )
            return 0

        if args.stage == "undistort-masks":
            validate_colmap_image_path_lengths(
                paths.mask_undistortion_images,
                mask_output_workspace,
                relative_images,
            )
            arguments = undistortion_arguments(
                paths,
                settings,
                image_path=paths.mask_undistortion_images,
                output_path=mask_output_workspace,
            )
            if args.dry_run:
                print(dense.display_command(arguments))
                print("Dry run passed. COLMAP was not run.")
                return 0
            result = run_mask_undistortion(
                paths,
                mask_output_workspace,
                settings,
                prepared_count,
                resume=args.resume,
            )
            print(
                f"Undistorted mask images: {result.image_count}; "
                f"resumed={result.resumed}; elapsed={result.elapsed_seconds:.2f}s"
            )
            return 0

        if not dense.undistorted_workspace_is_complete(paths, prepared_count):
            raise dense.PipelineError(
                "RGB undistortion is incomplete. Run --stage undistort-rgb first."
            )
        if not undistortion_output_complete(mask_output_workspace, prepared_count):
            raise dense.PipelineError(
                "Mask undistortion is incomplete. Run --stage undistort-masks first."
            )
        source_masks = mask_output_workspace / "images"
        if args.dry_run:
            print(f"Undistorted mask-image source: {source_masks}")
            print(f"Aligned 3DGS mask output: {paths.undistorted_masks}")
            print("Dry run passed. No masks were created.")
            return 0
        masking = dense.section(config, "masking")
        threshold = dense.optional_positive_int(
            masking, "threshold", "masking", 128
        )
        erosion = dense.optional_nonnegative_int(
            masking, "fusion_erosion_pixels", "masking", 0
        )
        expected, written, resumed = mask_tools.convert_masks(
            source_masks,
            paths.undistorted_images,
            paths.undistorted_masks,
            threshold,
            erosion,
            args.resume,
        )
        validated = mask_tools.validate_masks(
            paths.undistorted_images, paths.undistorted_masks
        )
        print(
            f"Aligned masks: expected {expected}; written {written}; "
            f"resumed {resumed}; validated {validated}"
        )
        return 0
    except (dense.PipelineError, mask_tools.MaskPreparationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
