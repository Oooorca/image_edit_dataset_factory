from __future__ import annotations

from pydantic import BaseModel, field_validator


class ImageInput(BaseModel):
    image_path: str | None = None
    image_b64: str | None = None

    @field_validator("image_b64", mode="after")
    @classmethod
    def _validate_any_image_input(cls, value: str | None, info: object) -> str | None:
        data = getattr(info, "data", {})
        image_path = data.get("image_path") if isinstance(data, dict) else None
        if not image_path and not value:
            msg = "either image_path or image_b64 must be provided"
            raise ValueError(msg)
        return value


class LayeredInferRequest(ImageInput):
    request_id: str
    sample_id: str | None = None
    return_b64: bool = True
    save_cache: bool = True


class LayerInfo(BaseModel):
    layer_id: int
    rgba_b64: str | None = None
    alpha_b64: str | None = None
    rgba_path: str | None = None
    alpha_path: str | None = None


class LayeredInferResponse(BaseModel):
    request_id: str
    runtime: str
    width: int
    height: int
    layers: list[LayerInfo]
    cache_dir: str | None = None


class EditInferRequest(ImageInput):
    request_id: str
    sample_id: str | None = None
    mask_path: str | None = None
    mask_b64: str | None = None
    prompt: str | None = None
    return_b64: bool = True
    save_cache: bool = True


class EditInferResponse(BaseModel):
    request_id: str
    runtime: str
    width: int
    height: int
    result_image_b64: str | None = None
    result_image_path: str | None = None
    cache_dir: str | None = None
