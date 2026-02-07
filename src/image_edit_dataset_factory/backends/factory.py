from __future__ import annotations

from image_edit_dataset_factory.backends.api_service_backends import (
    ApiEditorBackend,
    ApiLayeredDecomposer,
)
from image_edit_dataset_factory.backends.edit_base import EditorBackend
from image_edit_dataset_factory.backends.layered_base import LayeredDecomposer
from image_edit_dataset_factory.backends.mock_backend import (
    MockEditorBackend,
    MockLayeredDecomposer,
)
from image_edit_dataset_factory.backends.opencv_fallback import OpenCVFallbackBackend
from image_edit_dataset_factory.backends.qwen_image_edit_modelscope import (
    QwenImageEditModelScopeBackend,
)
from image_edit_dataset_factory.backends.qwen_layered_modelscope import QwenLayeredModelScopeBackend
from image_edit_dataset_factory.core.config import AppConfig


def build_layered_backend(cfg: AppConfig) -> LayeredDecomposer:
    key = cfg.backends.layered_backend.lower()
    if cfg.services.api_mode:
        key = "api"
    if key == "mock":
        return MockLayeredDecomposer()
    if key == "api":
        fallback = MockLayeredDecomposer() if cfg.services.layered.fallback_to_mock else None
        return ApiLayeredDecomposer(endpoint_cfg=cfg.services.layered, fallback=fallback)
    if key == "qwen":
        return QwenLayeredModelScopeBackend(
            model_dir=cfg.modelscope.qwen_layered_model_dir,
            device=cfg.backends.device,
        )
    raise ValueError(f"Unsupported layered backend: {cfg.backends.layered_backend}")


def build_edit_backend(cfg: AppConfig) -> EditorBackend:
    key = cfg.backends.edit_backend.lower()
    if cfg.services.api_mode:
        key = "api"
    if key == "mock":
        return MockEditorBackend()
    if key == "api":
        fallback: EditorBackend | None = None
        if cfg.services.edit.fallback_to_mock:
            fallback = MockEditorBackend()
        return ApiEditorBackend(endpoint_cfg=cfg.services.edit, fallback=fallback)
    if key == "opencv":
        return OpenCVFallbackBackend()
    if key == "qwen":
        return QwenImageEditModelScopeBackend(
            model_dir=cfg.modelscope.qwen_edit_model_dir,
            device=cfg.backends.device,
        )
    raise ValueError(f"Unsupported edit backend: {cfg.backends.edit_backend}")
