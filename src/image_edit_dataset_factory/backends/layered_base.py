from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class LayerData:
    layer_id: int
    rgba: np.ndarray
    alpha: np.ndarray


class LayeredDecomposer(ABC):
    @abstractmethod
    def decompose(self, image_rgb: np.ndarray) -> list[LayerData]:
        """Decompose an RGB image into RGBA layers."""
