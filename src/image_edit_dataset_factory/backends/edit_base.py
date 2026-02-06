from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EditorBackend(ABC):
    @abstractmethod
    def inpaint(
        self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str | None = None
    ) -> np.ndarray:
        """Inpaint masked region and return an RGB image."""

    def edit(self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
        """Default edit falls back to inpainting behavior."""
        return self.inpaint(image_rgb=image_rgb, mask=mask, prompt=prompt)
