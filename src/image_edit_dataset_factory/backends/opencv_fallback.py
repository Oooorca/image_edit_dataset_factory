from __future__ import annotations

import cv2
import numpy as np

from image_edit_dataset_factory.backends.edit_base import EditorBackend


class OpenCVFallbackBackend(EditorBackend):
    def __init__(self, radius: float = 3.0, method: int = cv2.INPAINT_TELEA) -> None:
        self.radius = radius
        self.method = method

    def inpaint(
        self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str | None = None
    ) -> np.ndarray:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(image_bgr, (mask > 0).astype(np.uint8), self.radius, self.method)
        return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
