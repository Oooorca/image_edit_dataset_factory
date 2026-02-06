from __future__ import annotations

from image_edit_dataset_factory.backends.edit_base import EditorBackend
from image_edit_dataset_factory.backends.layered_base import LayeredDecomposer
from image_edit_dataset_factory.backends.mock_backends import (
    MockEditorBackend,
    MockLayeredDecomposer,
)
from image_edit_dataset_factory.backends.opencv_inpaint import OpenCVInpaintBackend
from image_edit_dataset_factory.backends.qwen_image_edit import QwenImageEditBackend
from image_edit_dataset_factory.backends.qwen_layered import QwenLayeredDecomposer


def build_layered_backend(name: str, device: str = "cpu") -> LayeredDecomposer:
    key = name.lower()
    if key == "mock":
        return MockLayeredDecomposer()
    if key == "qwen":
        return QwenLayeredDecomposer(device=device)
    msg = f"Unknown layered backend: {name}"
    raise ValueError(msg)


def build_edit_backend(name: str, device: str = "cpu") -> EditorBackend:
    key = name.lower()
    if key == "mock":
        return MockEditorBackend()
    if key == "opencv":
        return OpenCVInpaintBackend()
    if key == "qwen":
        return QwenImageEditBackend(device=device)
    msg = f"Unknown edit backend: {name}"
    raise ValueError(msg)
