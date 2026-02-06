from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from image_edit_dataset_factory.backends.layered_base import LayeredDecomposer, LayerOutput


class QwenLayeredModelScopeBackend(LayeredDecomposer):
    """ModelScope backend for qwen/Qwen-Image-Layered.

    This class never downloads weights automatically. `model_dir` must point
    to a pre-downloaded local model path.
    """

    def __init__(self, model_dir: str | None, device: str = "cpu") -> None:
        self.model_dir = model_dir
        self.device = device
        self._pipeline: Any | None = None

    def _lazy_init(self) -> None:
        if self._pipeline is not None:
            return
        if not self.model_dir:
            raise RuntimeError(
                "Qwen layered backend requires modelscope.qwen_layered_model_dir. "
                "No auto-download is performed."
            )

        model_path = Path(self.model_dir)
        if not model_path.exists():
            raise RuntimeError(
                f"Qwen layered model dir not found: {model_path}. "
                "Please pre-download using `modelscope download --model qwen/Qwen-Image-Layered`."
            )

        try:
            from modelscope.pipelines import pipeline  # type: ignore
            from modelscope.utils.constant import Tasks  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("modelscope is not installed; install dependencies first.") from exc

        # Real task name can vary by ModelScope release; keep lazy and guarded.
        self._pipeline = pipeline(
            task=Tasks.image_editing, model=str(model_path), device=self.device
        )

    def decompose(self, image_rgb: np.ndarray) -> list[LayerOutput]:
        self._lazy_init()
        # Placeholder adapter until exact ModelScope output format is integrated.
        h, w = image_rgb.shape[:2]
        alpha = np.full((h, w), 255, dtype=np.uint8)
        rgba = np.dstack([image_rgb, alpha])
        return [LayerOutput(layer_id=0, rgba=rgba, alpha=alpha)]
