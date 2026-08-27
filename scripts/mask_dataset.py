#!/usr/bin/env python3
"""Create pot-only masks, transparent PNGs, COLMAP masks, and QC reports.

The SAM 2 dependency is imported only by ``process`` and the optional full
``doctor`` check.  Annotation and review therefore remain usable in the
project's ordinary OpenCV environment.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "sam2.1_hiera_tiny.pt"
DEFAULT_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
GENERATED_DIRECTORIES = ("masks_object", "masks_colmap", "rgba", "overlays")
GENERATED_FILES = ("run_metadata.json", "mask_manifest.csv", "qc_contact_sheet.jpg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pot-only masks and transparent images from extracted frames."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check the masking runtime and hardware.")
    doctor.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("SAM2_CHECKPOINT", DEFAULT_CHECKPOINT)),
        help="SAM 2.1 Hiera Tiny checkpoint path.",
    )
    doctor.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    doctor.add_argument(
        "--full",
        action="store_true",
        help="Load the checkpoint and run one synthetic-image inference.",
    )

    annotate = subparsers.add_parser(
        "annotate", help="Create or add a pot prompt in prompts.json."
    )
    annotate.add_argument("input_frames", type=Path)
    annotate.add_argument("output", type=Path)
    annotate.add_argument("--material", choices=("matte", "glossy"), required=True)
    annotate.add_argument(
        "--frame",
        default="0",
        help="Zero-based frame index or exact filename. Default: first frame.",
    )
    annotate.add_argument(
        "--box",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Prompt box in original-image pixels. Omit to draw it interactively.",
    )
    annotate.add_argument(
        "--positive",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("X", "Y"),
        help="Optional foreground point in original-image pixels. Repeat as needed.",
    )
    annotate.add_argument(
        "--negative",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("X", "Y"),
        help="Optional background point, especially on the turntable. Repeat as needed.",
    )
    annotate.add_argument("--overwrite", action="store_true")

    process = subparsers.add_parser("process", help="Run SAM 2 and write mask products.")
    process.add_argument("input_frames", type=Path)
    process.add_argument("output", type=Path)
    process.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    process.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("SAM2_CHECKPOINT", DEFAULT_CHECKPOINT)),
    )
    process.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    process.add_argument("--chunk-size", type=int, default=120)
    process.add_argument("--chunk-overlap", type=int, default=8)
    process.add_argument("--colmap-erosion", type=int, default=3, metavar="PIXELS")
    process.add_argument("--overlay-max-dimension", type=int, default=1280)
    process.add_argument("--qc-sample-stride", type=int, default=20)
    process.add_argument("--overwrite", action="store_true")

    review = subparsers.add_parser("review", help="Review sampled and flagged masks.")
    review.add_argument("input_frames", type=Path)
    review.add_argument("output", type=Path)
    review.add_argument("--sample-stride", type=int, default=20)
    review.add_argument(
        "--summary",
        action="store_true",
        help="Print the QC summary without opening a window.",
    )
    return parser.parse_args()


def import_core():
    try:
        return importlib.import_module("masking_core")
    except ModuleNotFoundError as error:
        if error.name in {"cv2", "numpy"}:
            raise RuntimeError(
                "OpenCV and NumPy are required. Activate the masking environment first."
            ) from error
        raise


def import_optional(name: str) -> tuple[Any | None, str | None]:
    try:
        module = importlib.import_module(name)
        return module, None
    except Exception as error:  # diagnostics must report broken installs too
        return None, f"{type(error).__name__}: {error}"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def choose_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but PyTorch cannot access Apple MPS")
    return torch.device(requested)


def inference_context(torch: Any, device: Any):
    if device.type == "cuda":
        major = int(torch.cuda.get_device_properties(device).major)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def device_policy(torch: Any, device: Any) -> dict[str, Any]:
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        low_memory = int(properties.total_memory) < 8 * 1024**3
        return {
            "device": "cuda",
            "device_name": properties.name,
            "dtype": "bfloat16" if properties.major >= 8 else "float16",
            "offload_video_to_cpu": low_memory,
            "offload_state_to_cpu": low_memory,
            "total_memory_bytes": int(properties.total_memory),
        }
    if device.type == "mps":
        return {
            "device": "mps",
            "device_name": "Apple Metal Performance Shaders",
            "dtype": "float32",
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": False,
            "total_memory_bytes": None,
        }
    return {
        "device": "cpu",
        "device_name": platform.processor() or "CPU",
        "dtype": "float32",
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
        "total_memory_bytes": None,
    }


def find_sam_commit(sam2_module: Any) -> str:
    module_path = Path(sam2_module.__file__).resolve()
    for parent in module_path.parents:
        if (parent / ".git").exists():
            try:
                result = subprocess.run(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.stdout.strip()
            except Exception:
                break
    return "unknown"


def command_doctor(args: argparse.Namespace) -> int:
    print(f"Python:       {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform:     {platform.platform()}")
    failures: list[str] = []
    modules: dict[str, Any] = {}
    for import_name, display_name in (
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("torch", "PyTorch"),
        ("sam2", "SAM 2"),
    ):
        module, error = import_optional(import_name)
        modules[import_name] = module
        if module is None:
            version = "missing"
        elif import_name == "sam2":
            version = package_version("SAM-2")
        else:
            version = getattr(module, "__version__", package_version(import_name))
        status = f"{display_name:<12} {version}"
        if error:
            status += f" ({error})"
            failures.append(display_name)
        print(status)

    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_exists = checkpoint.is_file()
    print(f"Checkpoint:   {checkpoint} ({'found' if checkpoint_exists else 'missing'})")
    if not checkpoint_exists:
        failures.append("checkpoint")

    torch = modules.get("torch")
    device = None
    if torch is not None:
        try:
            device = choose_device(torch, args.device)
            policy = device_policy(torch, device)
            print(f"Device:       {policy['device']} ({policy['device_name']})")
            print(f"Precision:    {policy['dtype']}")
            print(
                "CPU offload:  "
                f"video={policy['offload_video_to_cpu']}, "
                f"state={policy['offload_state_to_cpu']}"
            )
        except Exception as error:
            failures.append("device")
            print(f"Device:       unavailable ({error})")

    if args.full and not failures and device is not None:
        print("Smoke test:   loading SAM 2.1 Tiny...")
        try:
            import numpy as np
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            started = time.perf_counter()
            model = build_sam2(
                DEFAULT_MODEL_CONFIG, str(checkpoint), device=device, apply_postprocessing=False
            )
            predictor = SAM2ImagePredictor(model)
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            image[128:384, 160:352] = (150, 95, 55)
            with torch.inference_mode(), inference_context(torch, device):
                predictor.set_image(image)
                masks, _, _ = predictor.predict(
                    box=np.asarray([150, 118, 362, 394], dtype=np.float32),
                    multimask_output=False,
                )
            if not np.asarray(masks).size:
                raise RuntimeError("SAM 2 returned no mask")
            print(f"Smoke test:   passed in {time.perf_counter() - started:.2f} seconds")
        except Exception as error:
            failures.append("smoke test")
            print(f"Smoke test:   failed ({type(error).__name__}: {error})")
    elif args.full:
        print("Smoke test:   skipped because required checks failed")
    else:
        print("Smoke test:   skipped (use --full to load the model)")

    if failures:
        print("Result:       NOT READY - " + ", ".join(dict.fromkeys(failures)))
        return 1
    print("Result:       READY")
    return 0


def resolve_frame(frames: Sequence[Any], selector: str) -> Any:
    try:
        index = int(selector)
    except ValueError:
        matches = [frame for frame in frames if frame.filename == selector]
        if not matches:
            raise ValueError(f"No frame named {selector!r}")
        return matches[0]
    if not 0 <= index < len(frames):
        raise IndexError(f"Frame index must be between 0 and {len(frames) - 1}")
    return frames[index]


def interactive_prompt(image: Any, title: str) -> tuple[list[float], list[tuple[float, float, int]]]:
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    scale = min(1.0, 1280 / max(width, height))
    display = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    print("Draw a tight box around only the pot, then press Enter or Space.")
    roi = cv2.selectROI(title, display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    x, y, box_width, box_height = [float(value) for value in roi]
    if box_width <= 0 or box_height <= 0:
        raise RuntimeError("Annotation was cancelled before a pot box was selected")
    box = [x / scale, y / scale, (x + box_width) / scale, (y + box_height) / scale]

    points: list[tuple[float, float, int]] = []
    window = f"{title} points"

    def on_mouse(event: int, mouse_x: int, mouse_y: int, _flags: int, _data: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((mouse_x / scale, mouse_y / scale, 1))
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((mouse_x / scale, mouse_y / scale, 0))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print(
        "Optional points: left-click the pot, right-click the turntable/background. "
        "Press Enter to accept, R to clear points, or Esc to cancel."
    )
    while True:
        canvas = display.copy()
        scaled_box = [round(value * scale) for value in box]
        cv2.rectangle(
            canvas,
            (scaled_box[0], scaled_box[1]),
            (scaled_box[2], scaled_box[3]),
            (0, 255, 255),
            2,
        )
        for px, py, label in points:
            color = (40, 220, 40) if label == 1 else (40, 40, 240)
            cv2.circle(canvas, (round(px * scale), round(py * scale)), 7, color, -1)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in {10, 13, 32}:
            break
        if key in {ord("r"), ord("R")}:
            points.clear()
        if key == 27:
            cv2.destroyWindow(window)
            raise RuntimeError("Annotation was cancelled")
    cv2.destroyWindow(window)
    return box, points


def create_prompt(
    core: Any,
    frame: Any,
    box: Sequence[float],
    points: Sequence[tuple[float, float, int]],
) -> dict[str, Any]:
    return {
        "filename": frame.filename,
        "frame_index": frame.index,
        "source_width": frame.width,
        "source_height": frame.height,
        "box_xyxy_normalized": core.normalized_box(box, frame.width, frame.height),
        "points_normalized": core.normalized_points(points, frame.width, frame.height),
    }


def command_annotate(args: argparse.Namespace) -> int:
    core = import_core()
    import cv2

    frames, _ = core.discover_frames(args.input_frames)
    frame = resolve_frame(frames, args.frame)
    output = args.output.resolve()
    prompt_path = output / "prompts.json"
    existing = prompt_path.is_file()
    if existing:
        document = core.read_json(prompt_path)
        core.validate_prompt_document(document, frames)
        if document["material"] != args.material:
            raise ValueError(
                f"Existing prompts use material {document['material']!r}, not {args.material!r}"
            )
        if not args.overwrite and any(
            item.get("filename") == frame.filename for item in document["prompts"]
        ):
            raise FileExistsError(
                f"A prompt already exists for {frame.filename}. Use --overwrite to replace it."
            )
    else:
        if frame.index != 0:
            raise ValueError("The first annotation must use frame index 0")
        document = {
            "version": core.PROMPT_VERSION,
            "material": args.material,
            "input_directory": str(args.input_frames.resolve()),
            "input_fingerprint": core.image_directory_fingerprint(frames),
            "source_width": frame.width,
            "source_height": frame.height,
            "prompts": [],
        }

    image = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {frame.path}")
    points = [(*point, 1) for point in args.positive] + [
        (*point, 0) for point in args.negative
    ]
    if args.box:
        box = list(args.box)
    else:
        box, gui_points = interactive_prompt(image, f"Pot annotation - {frame.filename}")
        points.extend(gui_points)

    prompt = create_prompt(core, frame, box, points)
    core.upsert_prompt(document, prompt)
    output.mkdir(parents=True, exist_ok=True)
    core.write_json(prompt_path, document)
    print(f"Prompt saved: {prompt_path}")
    print(f"Frame:        {frame.filename}")
    print(f"Points:       {len(points)}")
    return 0


def validate_process_args(args: argparse.Namespace) -> None:
    if args.chunk_size < 2:
        raise ValueError("--chunk-size must be at least 2")
    if args.chunk_overlap < 1 or args.chunk_overlap >= args.chunk_size:
        raise ValueError("--chunk-overlap must be at least 1 and smaller than --chunk-size")
    if args.colmap_erosion < 0:
        raise ValueError("--colmap-erosion cannot be negative")
    if args.overlay_max_dimension < 100:
        raise ValueError("--overlay-max-dimension must be at least 100")
    if args.qc_sample_stride < 1:
        raise ValueError("--qc-sample-stride must be at least 1")


def prepare_output(output: Path, overwrite: bool) -> None:
    existing = [
        path
        for path in [
            *(output / name for name in GENERATED_DIRECTORIES),
            *(output / name for name in GENERATED_FILES),
        ]
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Generated output already exists in {output}. Use --overwrite to replace it."
        )
    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    for name in GENERATED_DIRECTORIES:
        (output / name).mkdir(parents=True, exist_ok=True)


def chunk_ranges(total: int, size: int, overlap: int) -> Iterable[tuple[int, int, int]]:
    chunk_id = 0
    start = 0
    while start < total:
        end = min(total, start + size)
        yield chunk_id, start, end
        if end == total:
            break
        start = end - overlap
        chunk_id += 1


def prompt_for_sam(core: Any, prompt: dict[str, Any], frame: Any) -> tuple[Any, Any, Any]:
    box = core.denormalized_box(prompt["box_xyxy_normalized"], frame.width, frame.height)
    points, labels = core.denormalized_points(
        prompt.get("points_normalized", []), frame.width, frame.height
    )
    return box, points, labels


def append_warning(record: dict[str, Any], warning: str) -> None:
    warnings = set(filter(None, str(record.get("qc_warnings", "")).split("|")))
    warnings.add(warning)
    record["qc_warnings"] = "|".join(sorted(warnings))


def save_products(
    core: Any,
    frame: Any,
    mask: Any,
    output: Path,
    colmap_erosion: int,
    overlay_max_dimension: int,
) -> dict[str, str]:
    import cv2

    source = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"Could not reread source frame: {frame.path}")
    stem = frame.path.stem
    object_relative = Path("masks_object") / f"{stem}.png"
    colmap_relative = Path("masks_colmap") / f"{frame.filename}.png"
    rgba_relative = Path("rgba") / f"{stem}.png"
    overlay_relative = Path("overlays") / f"{stem}.jpg"

    core.write_png(output / object_relative, mask)
    core.write_png(output / colmap_relative, core.erode_mask(mask, colmap_erosion))
    core.write_rgba(output / rgba_relative, source, mask)
    core.write_jpeg(
        output / overlay_relative,
        core.make_overlay(source, mask, max_dimension=overlay_max_dimension),
    )
    return {
        "object_mask": object_relative.as_posix(),
        "colmap_mask": colmap_relative.as_posix(),
        "rgba_image": rgba_relative.as_posix(),
        "overlay_image": overlay_relative.as_posix(),
    }


def command_process(args: argparse.Namespace) -> int:
    validate_process_args(args)
    core = import_core()
    import cv2
    import numpy as np

    frames, source_fields = core.discover_frames(args.input_frames)
    if len({frame.path.stem for frame in frames}) != len(frames):
        raise ValueError("Input frames must have unique filename stems for PNG outputs")
    output = args.output.resolve()
    prompt_path = output / "prompts.json"
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"No prompts.json found at {prompt_path}. Run the annotate command first."
        )
    prompt_document = core.read_json(prompt_path)
    core.validate_prompt_document(prompt_document, frames)
    if prompt_document.get("input_fingerprint") != core.image_directory_fingerprint(frames):
        raise ValueError("The input frames changed after prompts.json was created")
    if not any(item["filename"] == frames[0].filename for item in prompt_document["prompts"]):
        raise ValueError("A prompt for the first frame is required")
    prepare_output(output, args.overwrite)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"SAM 2 checkpoint not found: {checkpoint}. See scripts/masking_readme.md."
        )
    try:
        import torch
        import sam2
        from sam2.build_sam import build_sam2_video_predictor
    except Exception as error:
        raise RuntimeError(
            "SAM 2 and PyTorch are required for process. Run `mask_dataset.py doctor`."
        ) from error

    device = choose_device(torch, args.device)
    policy = device_policy(torch, device)
    print(f"Device: {policy['device']} ({policy['device_name']})")
    print(f"Precision: {policy['dtype']}")
    print(f"Frames: {len(frames)}")
    print(f"Material: {prompt_document['material']}")
    started = time.perf_counter()
    predictor = build_sam2_video_predictor(
        args.model_config,
        str(checkpoint),
        device=device,
        apply_postprocessing=False,
    )

    prompts_by_index = {
        next(frame.index for frame in frames if frame.filename == prompt["filename"]): prompt
        for prompt in prompt_document["prompts"]
    }
    records: list[dict[str, Any]] = []
    records_by_index: dict[int, dict[str, Any]] = {}
    prompted_names = {prompt["filename"] for prompt in prompt_document["prompts"]}
    last_new_mask = None

    all_staging_methods: set[str] = set()
    for chunk_id, start, end in chunk_ranges(
        len(frames), args.chunk_size, args.chunk_overlap
    ):
        chunk = frames[start:end]
        print(f"Chunk {chunk_id + 1}: frames {start}..{end - 1}")
        inherited_mask = None
        if start > 0:
            inherited_path = output / "masks_object" / f"{frames[start].path.stem}.png"
            inherited_mask = cv2.imread(str(inherited_path), cv2.IMREAD_GRAYSCALE)
            if inherited_mask is None:
                raise RuntimeError(f"Missing overlap seed mask: {inherited_path}")

        with core.numeric_frame_staging(chunk) as (staging, staging_methods):
            all_staging_methods.update(staging_methods)
            state = predictor.init_state(
                video_path=str(staging),
                offload_video_to_cpu=policy["offload_video_to_cpu"],
                offload_state_to_cpu=policy["offload_state_to_cpu"],
                async_loading_frames=False,
            )
            predictor.reset_state(state)
            has_start_correction = start in prompts_by_index
            if inherited_mask is not None and not has_start_correction:
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    mask=inherited_mask > 0,
                )
            for global_index, prompt in prompts_by_index.items():
                if not start <= global_index < end:
                    continue
                local_index = global_index - start
                frame = frames[global_index]
                box, points, labels = prompt_for_sam(core, prompt, frame)
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=local_index,
                    obj_id=1,
                    points=points,
                    labels=labels,
                    box=box,
                    clear_old_points=True,
                )

            raw_by_local: dict[int, Any] = {}
            with torch.inference_mode(), inference_context(torch, device):
                for local_index, object_ids, mask_logits in predictor.propagate_in_video(state):
                    if len(object_ids) == 0:
                        continue
                    raw_by_local[int(local_index)] = (
                        mask_logits[0] > 0.0
                    ).detach().cpu().numpy()

            previous_clean = inherited_mask if inherited_mask is not None else last_new_mask
            for local_index, frame in enumerate(chunk):
                global_index = start + local_index
                if local_index not in raw_by_local:
                    raise RuntimeError(
                        f"SAM 2 returned no mask for frame {global_index} ({frame.filename})"
                    )
                cleanup = core.postprocess_mask(
                    raw_by_local[local_index], previous_clean, (frame.height, frame.width)
                )
                clean_mask = cleanup.mask
                is_overlap = global_index in records_by_index
                if is_overlap:
                    canonical_path = output / records_by_index[global_index]["object_mask"]
                    canonical = cv2.imread(str(canonical_path), cv2.IMREAD_GRAYSCALE)
                    if canonical is None:
                        raise RuntimeError(f"Could not read overlap mask: {canonical_path}")
                    seam_iou = core.mask_iou(canonical, clean_mask)
                    if seam_iou < 0.98:
                        append_warning(records_by_index[global_index], "low_chunk_overlap_iou")
                    clean_mask = canonical
                    previous_clean = canonical
                    continue

                previous_iou = (
                    core.mask_iou(previous_clean, clean_mask)
                    if previous_clean is not None
                    else 1.0
                )
                product_paths = save_products(
                    core,
                    frame,
                    clean_mask,
                    output,
                    args.colmap_erosion,
                    args.overlay_max_dimension,
                )
                geometry = core.mask_geometry(clean_mask)
                record: dict[str, Any] = dict(frame.manifest)
                record.update(
                    {
                        "input_filename": frame.filename,
                        **product_paths,
                        **geometry,
                        "previous_iou": previous_iou,
                        "cleanup_change_ratio": cleanup.cleanup_change_ratio,
                        "raw_component_count": cleanup.raw_component_count,
                        "chunk_id": chunk_id,
                        "review_status": (
                            "prompted" if frame.filename in prompted_names else "unreviewed"
                        ),
                        "qc_warnings": "",
                    }
                )
                records.append(record)
                records_by_index[global_index] = record
                previous_clean = clean_mask
                last_new_mask = clean_mask
                if len(records) % 25 == 0 or len(records) == len(frames):
                    print(f"Wrote {len(records)}/{len(frames)} frame products...")

            predictor.reset_state(state)
            del state, raw_by_local
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if len(records) != len(frames):
        raise RuntimeError(f"Produced {len(records)} records for {len(frames)} frames")

    first_mask = cv2.imread(
        str(output / records[0]["object_mask"]), cv2.IMREAD_GRAYSCALE
    )
    last_mask = cv2.imread(
        str(output / records[-1]["object_mask"]), cv2.IMREAD_GRAYSCALE
    )
    loop_iou = core.mask_iou(first_mask, last_mask)
    if loop_iou < 0.95:
        append_warning(records[0], "low_rotation_loop_iou")
        append_warning(records[-1], "low_rotation_loop_iou")

    core.calculate_sequence_flags(records, prompt_document["material"])
    manifest_path = output / "mask_manifest.csv"
    core.write_manifest(manifest_path, records, source_fields)
    contact_count = core.build_qc_contact_sheet(
        records,
        output,
        output / "qc_contact_sheet.jpg",
        sample_stride=args.qc_sample_stride,
    )
    elapsed = time.perf_counter() - started
    metadata = {
        "version": 1,
        "input_directory": str(args.input_frames.resolve()),
        "input_fingerprint": core.image_directory_fingerprint(frames),
        "frame_count": len(frames),
        "source_width": frames[0].width,
        "source_height": frames[0].height,
        "material": prompt_document["material"],
        "model": "sam2.1_hiera_tiny",
        "model_config": args.model_config,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": core.sha256_file(checkpoint),
        "sam2_commit": find_sam_commit(sam2),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": package_version("torchvision"),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "pillow_version": package_version("Pillow"),
        "device_policy": policy,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "colmap_erosion_pixels": args.colmap_erosion,
        "staging_methods": sorted(all_staging_methods),
        "rotation_loop_iou": loop_iou,
        "qc_contact_sheet_frames": contact_count,
        "elapsed_seconds": elapsed,
        "color_correction": "none",
        "foreground": "pot_only",
    }
    core.write_json(output / "run_metadata.json", metadata)
    flagged = sum(bool(record["qc_warnings"]) for record in records)
    print(f"Completed in: {elapsed:.1f} seconds")
    print(f"Manifest:     {manifest_path}")
    print(f"QC flagged:   {flagged}/{len(records)}")
    print(f"Contact sheet:{output / 'qc_contact_sheet.jpg'}")
    return 0


def command_review(args: argparse.Namespace) -> int:
    core = import_core()
    import cv2

    frames, _ = core.discover_frames(args.input_frames)
    output = args.output.resolve()
    manifest_path = output / "mask_manifest.csv"
    prompt_path = output / "prompts.json"
    if not manifest_path.is_file() or not prompt_path.is_file():
        raise FileNotFoundError("Review requires mask_manifest.csv and prompts.json")
    fields, records = core.read_mask_manifest(manifest_path)
    if len(records) != len(frames):
        raise ValueError("Mask manifest row count does not match the input frames")
    selected_indices = [
        index
        for index, record in enumerate(records)
        if record.get("qc_warnings") or index % args.sample_stride == 0
    ]
    flagged = [record for record in records if record.get("qc_warnings")]
    print(f"Frames:       {len(records)}")
    print(f"Flagged:      {len(flagged)}")
    print(f"Review set:   {len(selected_indices)}")
    if args.summary:
        for record in flagged:
            print(f"{record['input_filename']}: {record['qc_warnings']}")
        return 0

    document = core.read_json(prompt_path)
    core.validate_prompt_document(document, frames)
    position = 0
    changed_manifest = False
    changed_prompts = False
    print("Review keys: N/Space next, P previous, A accept, C add correction, Q save/quit")
    while selected_indices:
        index = selected_indices[position]
        record = records[index]
        overlay = cv2.imread(str(output / record["overlay_image"]), cv2.IMREAD_COLOR)
        if overlay is None:
            raise RuntimeError(f"Could not read overlay for {record['input_filename']}")
        canvas = overlay.copy()
        label = (
            f"{position + 1}/{len(selected_indices)} {record['input_filename']} "
            f"{record.get('qc_warnings') or 'sample'}"
        )
        cv2.putText(
            canvas,
            label[:110],
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Pot mask review", canvas)
        key = cv2.waitKey(0) & 0xFF
        if key in {ord("q"), ord("Q"), 27}:
            break
        if key in {ord("p"), ord("P")}:
            position = (position - 1) % len(selected_indices)
            continue
        if key in {ord("a"), ord("A")}:
            record["review_status"] = "accepted"
            changed_manifest = True
            if position == len(selected_indices) - 1:
                break
            position += 1
            continue
        if key in {ord("c"), ord("C")}:
            source = cv2.imread(str(frames[index].path), cv2.IMREAD_COLOR)
            box, points = interactive_prompt(
                source, f"Correction - {frames[index].filename}"
            )
            core.upsert_prompt(document, create_prompt(core, frames[index], box, points))
            record["review_status"] = "correction_added"
            changed_manifest = True
            changed_prompts = True
            if position == len(selected_indices) - 1:
                break
            position += 1
            continue
        if key in {ord("n"), ord("N"), 13, 32}:
            if position == len(selected_indices) - 1:
                break
            position += 1

    cv2.destroyAllWindows()
    if changed_manifest:
        core.write_manifest(manifest_path, records, fields)
    if changed_prompts:
        core.write_json(prompt_path, document)
        print("Corrections saved. Rerun process with --overwrite.")
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.command == "doctor":
            return command_doctor(args)
        if args.command == "annotate":
            return command_annotate(args)
        if args.command == "process":
            return command_process(args)
        if args.command == "review":
            return command_review(args)
        raise AssertionError(f"Unhandled command: {args.command}")
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
