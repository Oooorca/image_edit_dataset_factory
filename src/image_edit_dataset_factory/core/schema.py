from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from image_edit_dataset_factory.core.enums import Category


class SourceMetadata(BaseModel):
    source_id: str
    image_path: str
    width: int
    height: int
    source: str = "local"
    scene: str = "mixed"
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("image_path")
    @classmethod
    def _normalize_image_path(cls, value: str) -> str:
        return str(Path(value))


class SampleModel(BaseModel):
    sample_id: str
    category: Category
    subtype: str
    scene: str
    src_image_path: str
    result_image_path: str
    mask_paths: list[str] = Field(default_factory=list)
    instruction_ch: str
    instruction_en: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QAScore(BaseModel):
    sample_id: str
    passed: bool
    mse_outside_region: float
    ssim_outside_region: float
    changed_pixel_ratio_outside_region: float
    details: dict[str, Any] = Field(default_factory=dict)


class LintIssue(BaseModel):
    path: str
    code: str
    message: str
