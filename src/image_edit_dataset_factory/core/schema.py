from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from image_edit_dataset_factory.core.enums import EditTask


class SourceSample(BaseModel):
    source_id: str
    dataset_category: str
    image_path: str
    width: int
    height: int
    scene: str = "mixed"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("image_path")
    @classmethod
    def _normalize_path(cls, value: str) -> str:
        return str(Path(value))


class DecomposeRecord(BaseModel):
    source_id: str
    image_path: str
    mask_path: str
    layer_paths: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SampleRecord(BaseModel):
    sample_id: str
    dataset_category: str
    edit_task: EditTask
    subtype: str
    scene: str
    source_id: str
    src_image_path: str
    result_image_path: str
    mask_paths: list[str] = Field(default_factory=list)
    instruction_ch: str
    instruction_en: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LintIssue(BaseModel):
    path: str
    code: str
    message: str


class QAScore(BaseModel):
    sample_id: str
    passed: bool
    mse_outside_region: float
    ssim_outside_region: float
    changed_pixel_ratio_outside_region: float
    details: dict[str, Any] = Field(default_factory=dict)
