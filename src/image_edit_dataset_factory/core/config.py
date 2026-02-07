from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from image_edit_dataset_factory.core.enums import DEFAULT_CATEGORY_TO_TASK


class PathsConfig(BaseModel):
    project_root: str = "."
    data_root: str = "./data"
    output_root: str = "./outputs"
    logs_root: str = "./logs"


class IngestConfig(BaseModel):
    include_categories: list[str] = Field(
        default_factory=lambda: [
            "人物物体一致性",
            "物体一致性",
            "物理变化",
        ]
    )
    recursive: bool = True
    max_images_per_category: int = 50


class FilterConfig(BaseModel):
    enabled: bool = True
    min_width: int = 256
    min_height: int = 256
    reject_grayscale: bool = False
    reject_borders: bool = False


class ModelScopeConfig(BaseModel):
    enabled: bool = False
    cache_dir: str | None = None
    qwen_layered_model_dir: str | None = "qwen/Qwen-Image-Layered"
    qwen_edit_model_dir: str | None = "Qwen/Qwen-Image-Edit"


class BackendConfig(BaseModel):
    layered_backend: str = "mock"
    edit_backend: str = "opencv"
    use_modelscope: bool = False
    device: str = "cpu"


class ServiceEndpointConfig(BaseModel):
    enabled: bool = False
    endpoint: str
    timeout_sec: float = 120.0
    max_retries: int = 2
    backoff_sec: float = 1.0
    send_mode: str = "base64"
    fallback_to_mock: bool = True

    @field_validator("send_mode")
    @classmethod
    def _validate_send_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"path", "base64"}:
            msg = f"send_mode must be one of path/base64, got: {value}"
            raise ValueError(msg)
        return normalized


class ServicesConfig(BaseModel):
    api_mode: bool = False
    request_batch_size: int = 1
    layered: ServiceEndpointConfig = ServiceEndpointConfig(
        enabled=False,
        endpoint="http://127.0.0.1:8101",
    )
    edit: ServiceEndpointConfig = ServiceEndpointConfig(
        enabled=False,
        endpoint="http://127.0.0.1:8102",
    )


class GenerateConfig(BaseModel):
    dry_run: bool = False
    category_to_task: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_CATEGORY_TO_TASK))
    subtypes: dict[str, str] = Field(
        default_factory=lambda: {
            "structural_edit": "move",
            "semantic_edit": "delete",
            "consistency_edit": "identity",
        }
    )


class QAConfig(BaseModel):
    allowed_region_dilation_px: int = 7
    max_mse_outside_region: float = 4.0
    min_ssim_outside_region: float = 0.98
    max_changed_pixel_ratio_outside_region: float = 0.02


class PipelineConfig(BaseModel):
    ingest: bool = True
    decompose: bool = True
    generate: bool = True
    export: bool = True
    qa: bool = True
    resume: bool = False


class AppConfig(BaseModel):
    paths: PathsConfig = PathsConfig()
    ingest: IngestConfig = IngestConfig()
    filter: FilterConfig = FilterConfig()
    backends: BackendConfig = BackendConfig()
    modelscope: ModelScopeConfig = ModelScopeConfig()
    services: ServicesConfig = ServicesConfig()
    generate: GenerateConfig = GenerateConfig()
    qa: QAConfig = QAConfig()
    pipeline: PipelineConfig = PipelineConfig()
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
