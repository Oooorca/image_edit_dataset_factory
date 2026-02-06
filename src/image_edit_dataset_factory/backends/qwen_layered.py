from __future__ import annotations

from typing import Any

import numpy as np

from image_edit_dataset_factory.backends.layered_base import LayerData, LayeredDecomposer


class QwenLayeredDecomposer(LayeredDecomposer):
    """Skeleton for Qwen-Image-Layered integration with guarded imports."""

    def __init__(self, model_id: str = "Qwen/Qwen-Image-Layered", device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device
        self._pipeline: Any | None = None

    def _lazy_init(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from diffusers import QwenImageLayeredPipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover
            msg = (
                "diffusers with QwenImageLayeredPipeline is required for qwen backend. "
                "Install optional gpu deps and model weights, or use backend=mock."
            )
            raise RuntimeError(msg) from exc

        self._pipeline = QwenImageLayeredPipeline.from_pretrained(self.model_id)
        self._pipeline = self._pipeline.to(self.device)

    def decompose(self, image_rgb: np.ndarray) -> list[LayerData]:
        self._lazy_init()
        # Placeholder behavior until model output format is finalized.
        h, w = image_rgb.shape[:2]
        alpha = np.full((h, w), 255, dtype=np.uint8)
        rgba = np.dstack([image_rgb, alpha])
        return [LayerData(layer_id=1, rgba=rgba, alpha=alpha)]
