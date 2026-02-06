from __future__ import annotations

import numpy as np

from image_edit_dataset_factory.backends.edit_base import EditorBackend
from image_edit_dataset_factory.backends.layered_base import LayerData, LayeredDecomposer


class MockLayeredDecomposer(LayeredDecomposer):
    def decompose(self, image_rgb: np.ndarray) -> list[LayerData]:
        h, w = image_rgb.shape[:2]
        alpha = np.full((h, w), 255, dtype=np.uint8)
        rgba = np.dstack([image_rgb, alpha])
        return [LayerData(layer_id=1, rgba=rgba, alpha=alpha)]


class MockEditorBackend(EditorBackend):
    def inpaint(
        self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str | None = None
    ) -> np.ndarray:
        output = image_rgb.copy()
        fill_value = 127 if not prompt else (sum(ord(ch) for ch in prompt) % 200)
        output[mask > 0] = np.array([fill_value, fill_value, fill_value], dtype=np.uint8)
        return output
