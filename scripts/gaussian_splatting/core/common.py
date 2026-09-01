"""Shared configuration and validation helpers for the 3DGS pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "gaussian_splatting_pot1_unglazed_every6.yml"
)
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
SPARSE_MODEL_COMPONENTS = ("cameras", "images", "points3D")


class PipelineError(RuntimeError):
    """A user-correctable 3DGS configuration or input error."""


@dataclass(frozen=True)
class GaussianSplattingPaths:
    """Resolved input and output paths for the 3DGS branch."""

    config: Path
    colmap: Path
    dataset: Path
    images: Path
    masks: Path
    sparse_model: Path
    workspace: Path
    cache: Path
    runs: Path
    dataset_manifest: Path
    holdout_split: Path


@dataclass(frozen=True)
class ValidationReport:
    """Summary of a successfully validated 3DGS input dataset."""

    image_count: int
    mask_count: int
    registered_image_count: int
    resolutions: tuple[tuple[int, int], ...]
    sparse_format: str


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise PipelineError(
            "PyYAML is required. Update the pot-masking environment from "
            "environment-masking.yml before running preflight validation."
        ) from error

    try:
        with path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file)
    except FileNotFoundError as error:
        raise PipelineError(f"Configuration file not found: {path}") from error
    except yaml.YAMLError as error:
        raise PipelineError(f"Invalid YAML in {path}: {error}") from error

    if not isinstance(data, Mapping):
        raise PipelineError(f"YAML root must be a mapping: {path}")
    return data


def section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise PipelineError(f"Missing or invalid '{name}' section in the YAML config.")
    return value


def required_string(values: Mapping[str, Any], key: str, section_name: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(
            f"'{section_name}.{key}' must be a non-empty string in the YAML config."
        )
    return value.strip()


def required_positive_int(
    values: Mapping[str, Any], key: str, section_name: str
) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PipelineError(f"'{section_name}.{key}' must be a positive integer.")
    return value


def required_positive_number(
    values: Mapping[str, Any], key: str, section_name: str
) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise PipelineError(f"'{section_name}.{key}' must be a positive number.")
    return float(value)


def required_bool(values: Mapping[str, Any], key: str, section_name: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise PipelineError(f"'{section_name}.{key}' must be true or false.")
    return value


def profile(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    profiles = section(config, "profiles")
    value = profiles.get(name)
    if not isinstance(value, Mapping):
        choices = ", ".join(sorted(str(key) for key in profiles))
        raise PipelineError(f"Unknown training profile '{name}'. Choose: {choices}.")
    return value


def preparation_factors(config: Mapping[str, Any]) -> tuple[int, ...]:
    profiles = section(config, "profiles")
    factors: set[int] = set()
    for name, values in profiles.items():
        if not isinstance(values, Mapping):
            raise PipelineError(f"Profile '{name}' must be a mapping.")
        factors.add(required_positive_int(values, "image_factor", f"profiles.{name}"))
    return tuple(sorted(factors))


def resolve_config_path(raw_path: Path) -> Path:
    candidate = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
    candidate = candidate.resolve()
    _require_within(candidate, PROJECT_ROOT, "configuration")
    return candidate


def resolve_project_path(raw_path: str, label: str) -> Path:
    candidate = Path(os.path.expandvars(raw_path)).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    _require_within(candidate, PROJECT_ROOT, label)
    return candidate


def resolve_child(parent: Path, raw_path: str, label: str) -> Path:
    configured = Path(raw_path)
    candidate = configured if configured.is_absolute() else parent / configured
    candidate = candidate.resolve()
    _require_within(candidate, parent, label)
    return candidate


def _require_within(candidate: Path, parent: Path, label: str) -> None:
    try:
        candidate.relative_to(parent.resolve())
    except ValueError as error:
        raise PipelineError(
            f"'{label}' must stay inside {parent.resolve()}. Resolved path: {candidate}"
        ) from error


def resolve_paths(config_path: Path, config: Mapping[str, Any]) -> GaussianSplattingPaths:
    colmap_config = section(config, "colmap")
    input_config = section(config, "input")
    output_config = section(config, "output")

    dataset = resolve_project_path(
        required_string(input_config, "dataset", "input"), "input.dataset"
    )
    workspace = resolve_project_path(
        required_string(output_config, "workspace", "output"), "output.workspace"
    )

    return GaussianSplattingPaths(
        config=config_path,
        colmap=resolve_project_path(
            required_string(colmap_config, "executable", "colmap"),
            "colmap.executable",
        )
        if not Path(required_string(colmap_config, "executable", "colmap")).is_absolute()
        else Path(required_string(colmap_config, "executable", "colmap")).resolve(),
        dataset=dataset,
        images=resolve_child(
            dataset, required_string(input_config, "images", "input"), "input.images"
        ),
        masks=resolve_child(
            dataset, required_string(input_config, "masks", "input"), "input.masks"
        ),
        sparse_model=resolve_child(
            dataset,
            required_string(input_config, "sparse_model", "input"),
            "input.sparse_model",
        ),
        workspace=workspace,
        cache=resolve_child(
            workspace, required_string(output_config, "cache", "output"), "output.cache"
        ),
        runs=resolve_child(
            workspace, required_string(output_config, "runs", "output"), "output.runs"
        ),
        dataset_manifest=resolve_child(
            workspace,
            required_string(output_config, "dataset_manifest", "output"),
            "output.dataset_manifest",
        ),
        holdout_split=resolve_child(
            workspace,
            required_string(output_config, "holdout_split", "output"),
            "output.holdout_split",
        ),
    )


def image_files(image_root: Path) -> list[Path]:
    return sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def expected_mask_path(image_path: Path, image_root: Path, mask_root: Path) -> Path:
    relative_image = image_path.relative_to(image_root)
    return mask_root / Path(f"{relative_image}.png")


def validate_sparse_components(sparse_model: Path) -> str:
    if not sparse_model.is_dir():
        raise PipelineError(f"Sparse model directory was not found: {sparse_model}")

    formats: list[str] = []
    missing: list[str] = []
    for component in SPARSE_MODEL_COMPONENTS:
        found = next(
            (
                extension
                for extension in ("bin", "txt")
                if (sparse_model / f"{component}.{extension}").is_file()
            ),
            None,
        )
        if found is None:
            missing.append(component)
        else:
            formats.append(found)
    if missing:
        raise PipelineError(
            "Sparse model is missing required component(s): " + ", ".join(missing)
        )
    if len(set(formats)) != 1:
        raise PipelineError("Sparse model components mix binary and text formats.")
    return formats[0]


def read_text_model_image_names(images_txt: Path) -> list[str]:
    names: list[str] = []
    expecting_image_line = True
    for raw_line in images_txt.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            continue
        if expecting_image_line:
            if not stripped:
                continue
            fields = stripped.split(maxsplit=9)
            if len(fields) < 10:
                raise PipelineError(
                    f"Malformed image record in converted model: {raw_line}"
                )
            names.append(fields[9].replace("\\", "/"))
            expecting_image_line = False
        else:
            expecting_image_line = True
    if not expecting_image_line:
        raise PipelineError("Converted images.txt ended before its points2D record.")
    if not names:
        raise PipelineError("The sparse model contains no registered images.")
    if len(names) != len(set(names)):
        raise PipelineError("The sparse model contains duplicate registered image names.")
    return names


def registered_image_names(colmap: Path, sparse_model: Path) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="thai_ceramics_3dgs_validate_") as directory:
        text_model = Path(directory)
        arguments = [
            str(colmap),
            "model_converter",
            "--input_path",
            str(sparse_model),
            "--output_path",
            str(text_model),
            "--output_type",
            "TXT",
        ]
        try:
            completed = subprocess.run(
                arguments,
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise PipelineError(
                f"Could not launch COLMAP while validating the sparse model: {error}"
            ) from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            if detail:
                detail = f"\n{detail[-2000:]}"
            raise PipelineError(
                "COLMAP could not convert the sparse model for validation "
                f"(exit code {completed.returncode}).{detail}"
            )
        return read_text_model_image_names(text_model / "images.txt")


def format_examples(values: Iterable[str], limit: int = 10) -> str:
    examples = list(values)
    shown = "\n".join(f"  {value}" for value in examples[:limit])
    if len(examples) > limit:
        shown += f"\n  ...and {len(examples) - limit} more"
    return shown
