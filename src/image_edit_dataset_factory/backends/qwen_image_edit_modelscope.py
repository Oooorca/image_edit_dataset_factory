from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from image_edit_dataset_factory.backends.edit_base import EditorBackend


class QwenImageEditModelScopeBackend(EditorBackend):
    """ModelScope backend for Qwen/Qwen-Image-Edit.

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
                "Qwen image edit backend requires modelscope.qwen_edit_model_dir. "
                "No auto-download is performed."
            )

        model_path = Path(self.model_dir)
        if not model_path.exists():
            raise RuntimeError(
                f"Qwen image edit model dir not found: {model_path}. "
                "Please pre-download using `modelscope download --model Qwen/Qwen-Image-Edit`."
            )

        try:
            from modelscope.pipelines import pipeline  # type: ignore
            from modelscope.utils.constant import Tasks  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("modelscope is not installed; install dependencies first.") from exc

        self._pipeline = pipeline(
            task=Tasks.image_editing, model=str(model_path), device=self.device
        )

    def inpaint(
        self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str | None = None
    ) -> np.ndarray:
        self._lazy_init()
        # Placeholder until exact ModelScope Qwen-Image-Edit inputs/outputs are finalized.
        output = image_rgb.copy()
        output[mask > 0] = np.array([96, 96, 96], dtype=np.uint8)
        return output
