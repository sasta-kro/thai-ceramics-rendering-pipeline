#!/usr/bin/env python3
"""Check the pinned Windows gsplat environment without starting training."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import platform
import sys
from typing import Iterable


EXPECTED_PYTHON = (3, 10)
EXPECTED_TORCH = "2.4.1"
EXPECTED_CUDA = "12.4"
EXPECTED_GSPLAT = "1.5.3"
MINIMUM_COMPUTE_CAPABILITY = (7, 0)
EXPECTED_GPU_MEMORY_GB = 4.0

REQUIRED_IMPORTS = {
    "Pillow": "PIL",
    "PyYAML": "yaml",
    "pycolmap": "pycolmap",
    "SciPy": "scipy",
    "scikit-learn": "sklearn",
    "tqdm": "tqdm",
    "tyro": "tyro",
    "imageio": "imageio",
    "imageio-ffmpeg": "imageio_ffmpeg",
    "torchmetrics": "torchmetrics",
    "tensorboard": "tensorboard",
    "viser": "viser",
    "OpenCV": "cv2",
}


def normalized_public_version(raw_version: str) -> str:
    """Return the version before a local wheel suffix such as '+cu124'."""

    return raw_version.split("+", maxsplit=1)[0]


def compatibility_problems(
    python_version: tuple[int, int],
    torch_version: str,
    cuda_version: str | None,
    gsplat_version: str,
) -> list[str]:
    """Describe deviations from the pinned, precompiled Windows wheel matrix."""

    problems: list[str] = []
    if python_version != EXPECTED_PYTHON:
        problems.append(
            f"Python {python_version[0]}.{python_version[1]} detected; "
            f"expected {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}."
        )
    if normalized_public_version(torch_version) != EXPECTED_TORCH:
        problems.append(
            f"PyTorch {torch_version} detected; expected {EXPECTED_TORCH}+cu124."
        )
    if cuda_version != EXPECTED_CUDA:
        problems.append(
            f"PyTorch CUDA {cuda_version or 'none'} detected; expected {EXPECTED_CUDA}."
        )
    if normalized_public_version(gsplat_version) != EXPECTED_GSPLAT:
        problems.append(
            f"gsplat {gsplat_version} detected; expected {EXPECTED_GSPLAT}."
        )
    return problems


def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def missing_imports(imports: Iterable[tuple[str, str]]) -> list[str]:
    missing: list[str] = []
    for label, module_name in imports:
        try:
            import_module(module_name)
        except (ImportError, OSError) as error:
            missing.append(f"{label} ({module_name}): {error}")
    return missing


def main() -> int:
    print("Thai ceramics 3DGS environment doctor")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")

    try:
        import torch
    except (ImportError, OSError) as error:
        print(f"ERROR: PyTorch could not be imported: {error}", file=sys.stderr)
        return 1

    try:
        import gsplat
        from gsplat import rasterization  # noqa: F401
    except (ImportError, OSError) as error:
        print(f"ERROR: gsplat could not be imported: {error}", file=sys.stderr)
        return 1

    gsplat_version = getattr(gsplat, "__version__", package_version("gsplat"))
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"gsplat: {gsplat_version}")

    problems = compatibility_problems(
        (sys.version_info.major, sys.version_info.minor),
        torch.__version__,
        torch.version.cuda,
        gsplat_version,
    )

    if not torch.cuda.is_available():
        problems.append("PyTorch cannot access a CUDA GPU.")
    else:
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        capability = torch.cuda.get_device_capability(device_index)
        memory_gb = properties.total_memory / 1024**3
        print(f"GPU: {properties.name}")
        print(f"Compute capability: {capability[0]}.{capability[1]}")
        print(f"GPU memory: {memory_gb:.2f} GB")
        if capability < MINIMUM_COMPUTE_CAPABILITY:
            problems.append(
                "GPU compute capability is below the gsplat requirement of 7.0."
            )
        if memory_gb < EXPECTED_GPU_MEMORY_GB * 0.9:
            problems.append(
                f"Less than the expected {EXPECTED_GPU_MEMORY_GB:.0f} GB GPU memory "
                "was detected."
            )

    unavailable = missing_imports(REQUIRED_IMPORTS.items())
    if unavailable:
        problems.append("Missing or unloadable dependencies:\n  " + "\n  ".join(unavailable))

    if problems:
        print("\nEnvironment check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1

    print("\nEnvironment check passed.")
    print("No rasterization, image preparation, or training was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
