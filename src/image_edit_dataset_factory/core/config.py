from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from image_edit_dataset_factory.core.enums import Category


class PathsConfig(BaseModel):
    root: str = "."
    data_dir: str = "data"
    outputs_dir: str = "outputs"
    logs_dir: str = "logs"


class IngestConfig(BaseModel):
    source_dir: str = "data/input"
    manifest_path: str | None = None
    symlink: bool = True
    recursive: bool = True


class FilterConfig(BaseModel):
    min_width: int = 3840
    min_height: int = 2160
    reject_grayscale: bool = True
    reject_borders: bool = True
    reject_text_like: bool = True
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"]
    )


class DecomposeConfig(BaseModel):
    backend: str = "mock"
    max_layers: int = 8
    num_workers: int = 2
    overwrite: bool = False


class EditBackendConfig(BaseModel):
    backend: str = "opencv"


class GenerationCategoryConfig(BaseModel):
    enabled: bool = True
    subtypes: list[str] = Field(default_factory=list)
    per_source: int = 1


class GenerateConfig(BaseModel):
    dry_run: bool = False
    num_workers: int = 2
    categories: dict[str, GenerationCategoryConfig] = Field(default_factory=dict)


class QuotasConfig(BaseModel):
    target_total: int = 50000
    per_category: dict[str, int] = Field(default_factory=dict)
    per_scene: dict[str, int] = Field(default_factory=dict)


class QAConfig(BaseModel):
    enabled: bool = True
    allowed_region_dilation_px: int = 7
    max_mse_outside_region: float = 2.0
    min_ssim_outside_region: float = 0.995
    max_changed_pixel_ratio_outside_region: float = 0.005


class PipelineSwitches(BaseModel):
    resume: bool = True
    ingest: bool = True
    filter: bool = True
    decompose: bool = True
    generate: bool = True
    export: bool = True
    lint: bool = True
    qa: bool = True


class BackendRuntimeConfig(BaseModel):
    device: str = "cpu"
    use_half: bool = False


class AppConfig(BaseModel):
    paths: PathsConfig = PathsConfig()
    ingest: IngestConfig = IngestConfig()
    filter: FilterConfig = FilterConfig()
    decompose: DecomposeConfig = DecomposeConfig()
    edit: EditBackendConfig = EditBackendConfig()
    generate: GenerateConfig = GenerateConfig()
    quotas: QuotasConfig = QuotasConfig()
    qa: QAConfig = QAConfig()
    pipeline: PipelineSwitches = PipelineSwitches()
    backend_runtime: BackendRuntimeConfig = BackendRuntimeConfig()
    categories: list[Category] = Field(
        default_factory=lambda: [
            Category.PORTRAIT_ATTRIBUTE,
            Category.SEMANTIC_EDIT,
            Category.STYLE_EDIT,
            Category.STRUCTURAL_EDIT,
            Category.TEXT_EDIT,
        ]
    )
    scenes: list[str] = Field(default_factory=lambda: ["mixed"])
    json_logs: bool = True


def _deep_set(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = data
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        msg = f"Override must be key=value format: {raw}"
        raise ValueError(msg)
    key, value_str = raw.split("=", 1)
    value = yaml.safe_load(value_str)
    return key.strip(), value


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    for item in overrides or []:
        key, value = _parse_override(item)
        _deep_set(payload, key, value)

    return AppConfig.model_validate(payload)


def dump_config(config: AppConfig, path: str | Path) -> None:
    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False, allow_unicode=True)
