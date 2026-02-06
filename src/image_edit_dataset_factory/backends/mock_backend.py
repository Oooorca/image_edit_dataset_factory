from __future__ import annotations

import numpy as np

from image_edit_dataset_factory.backends.edit_base import EditorBackend
from image_edit_dataset_factory.backends.layered_base import LayeredDecomposer, LayerOutput


class MockLayeredDecomposer(LayeredDecomposer):
    def decompose(self, image_rgb: np.ndarray) -> list[LayerOutput]:
        h, w = image_rgb.shape[:2]
        alpha_bg = np.full((h, w), 255, dtype=np.uint8)
        rgba_bg = np.dstack([image_rgb, alpha_bg])

        mask = np.zeros((h, w), dtype=np.uint8)
        y0, y1 = h // 4, (3 * h) // 4
        x0, x1 = w // 4, (3 * w) // 4
        mask[y0:y1, x0:x1] = 255

        rgba_obj = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_obj[:, :, :3] = image_rgb
        rgba_obj[:, :, 3] = mask

        return [
            LayerOutput(layer_id=0, rgba=rgba_bg, alpha=alpha_bg),
            LayerOutput(layer_id=1, rgba=rgba_obj, alpha=mask),
        ]


class MockEditorBackend(EditorBackend):
    def inpaint(
        self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str | None = None
    ) -> np.ndarray:
        output = image_rgb.copy()
        token = 128 if prompt is None else (sum(ord(ch) for ch in prompt) % 255)
        output[mask > 0] = np.array([token, token // 2, 255 - token], dtype=np.uint8)
        return output
