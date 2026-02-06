from __future__ import annotations

from typing import Any

import numpy as np

from image_edit_dataset_factory.backends.edit_base import EditorBackend


class QwenImageEditBackend(EditorBackend):
    """Skeleton for Qwen-Image-Edit integration with guarded imports."""

    def __init__(self, model_id: str = "Qwen/Qwen-Image-Edit", device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device
        self._pipeline: Any | None = None

    def _lazy_init(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from diffusers import AutoPipelineForImage2Image  # type: ignore
        except ImportError as exc:  # pragma: no cover
            msg = (
                "diffusers is required for qwen image edit backend. "
                "Install optional gpu deps and model weights, or use backend=opencv/mock."
            )
            raise RuntimeError(msg) from exc

        self._pipeline = AutoPipelineForImage2Image.from_pretrained(self.model_id)
        self._pipeline = self._pipeline.to(self.device)

    def inpaint(
        self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str | None = None
    ) -> np.ndarray:
        self._lazy_init()
        # Placeholder behavior. Replace with actual Qwen inpaint/edit call.
        output = image_rgb.copy()
        output[mask > 0] = np.array([128, 128, 128], dtype=np.uint8)
        return output
