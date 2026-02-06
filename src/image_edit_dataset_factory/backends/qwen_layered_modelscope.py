from __future__ import annotations

from typing import Any

import numpy as np

from image_edit_dataset_factory.backends.layered_base import LayeredDecomposer, LayerOutput
from image_edit_dataset_factory.backends.modelscope_utils import (
    pil_to_rgb_array,
    resolve_local_model_dir,
    to_pil,
)
from image_edit_dataset_factory.utils.mask_ops import alpha_to_mask


class QwenLayeredModelScopeBackend(LayeredDecomposer):
    """ModelScope backend for qwen/Qwen-Image-Layered.

    The backend only uses locally downloaded weights. It never triggers downloads.
    """

    MODEL_ID = "qwen/Qwen-Image-Layered"

    def __init__(self, model_dir: str | None, device: str = "cpu") -> None:
        self.model_dir = model_dir
        self.device = device
        self._pipeline: Any | None = None

    def _lazy_init(self) -> None:
        if self._pipeline is not None:
            return

        local_dir = resolve_local_model_dir(self.model_dir, self.MODEL_ID)

        try:
            from modelscope.pipelines import pipeline  # type: ignore
            from modelscope.utils.constant import Tasks  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("modelscope is not installed; install dependencies first.") from exc

        device = "gpu" if self.device.lower().startswith("cuda") else self.device
        task_candidates = [
            getattr(Tasks, "image_editing", None),
            "image_editing",
            "image-editing",
            "image_to_image_generation",
        ]

        errors: list[str] = []
        for task in task_candidates:
            if task is None:
                continue
            try:
                self._pipeline = pipeline(task=task, model=str(local_dir), device=device)
                return
            except Exception as exc:  # pragma: no cover
                errors.append(f"task={task!r} error={exc}")

        raise RuntimeError(
            "Failed to initialize ModelScope layered pipeline. "
            f"model_dir={local_dir}. Attempts: {' | '.join(errors)}"
        )

    @staticmethod
    def _normalize_alpha(alpha: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        arr = alpha.squeeze()
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = arr.astype(np.uint8)
        if arr.shape != shape:
            arr = np.array(to_pil(np.dstack([arr, arr, arr])).resize(shape[::-1]).convert("L"))
        return arr

    def _extract_layers(self, output: Any, source: np.ndarray) -> list[LayerOutput]:
        h, w = source.shape[:2]

        def _to_rgba_and_alpha(item: Any) -> tuple[np.ndarray, np.ndarray] | None:
            if hasattr(item, "mode"):
                # PIL Image
                try:
                    rgb = pil_to_rgb_array(item)
                    if item.mode == "RGBA":
                        alpha = np.asarray(item.split()[-1], dtype=np.uint8)
                    else:
                        alpha = np.full((h, w), 255, dtype=np.uint8)
                    return np.dstack([rgb, alpha]), alpha
                except Exception:
                    return None

            if isinstance(item, np.ndarray):
                arr = item
                if arr.ndim == 3 and arr.shape[2] == 4:
                    alpha = arr[:, :, 3].astype(np.uint8)
                    return arr.astype(np.uint8), alpha
                if arr.ndim == 3 and arr.shape[2] == 3:
                    alpha = np.full((arr.shape[0], arr.shape[1]), 255, dtype=np.uint8)
                    return np.dstack([arr.astype(np.uint8), alpha]), alpha
                if arr.ndim == 2:
                    alpha = alpha_to_mask(arr.astype(np.uint8))
                    rgb = source.copy()
                    return np.dstack([rgb, alpha]), alpha
            return None

        layers: list[LayerOutput] = []

        if isinstance(output, dict):
            layer_items = (
                output.get("layers")
                or output.get("output_layers")
                or output.get("output_imgs")
            )
            masks = output.get("masks") or output.get("alphas") or []

            if isinstance(layer_items, list):
                for idx, item in enumerate(layer_items):
                    parsed = _to_rgba_and_alpha(item)
                    if parsed is None:
                        continue
                    rgba, alpha = parsed
                    if idx < len(masks):
                        mask_arr = masks[idx]
                        if hasattr(mask_arr, "mode"):
                            alpha = np.asarray(mask_arr.convert("L"), dtype=np.uint8)
                        elif isinstance(mask_arr, np.ndarray):
                            alpha = alpha_to_mask(mask_arr.astype(np.uint8))
                        rgba = np.dstack([rgba[:, :, :3], alpha])
                    layers.append(LayerOutput(layer_id=idx, rgba=rgba, alpha=alpha))

            if not layers:
                candidate = output.get("output_img") or output.get("image")
                parsed = _to_rgba_and_alpha(candidate) if candidate is not None else None
                if parsed is not None:
                    rgba, alpha = parsed
                    layers.append(LayerOutput(layer_id=0, rgba=rgba, alpha=alpha))

        if not layers:
            alpha = np.full((h, w), 255, dtype=np.uint8)
            layers.append(LayerOutput(layer_id=0, rgba=np.dstack([source, alpha]), alpha=alpha))

        normalized: list[LayerOutput] = []
        for idx, layer in enumerate(layers):
            rgb = layer.rgba[:, :, :3]
            if rgb.shape[:2] != (h, w):
                rgb = np.asarray(to_pil(rgb).resize((w, h)), dtype=np.uint8)
            alpha = self._normalize_alpha(layer.alpha, (h, w))
            normalized.append(LayerOutput(layer_id=idx, rgba=np.dstack([rgb, alpha]), alpha=alpha))

        return normalized

    def decompose(self, image_rgb: np.ndarray) -> list[LayerOutput]:
        self._lazy_init()
        image_pil = to_pil(image_rgb)

        attempts = [
            lambda: self._pipeline(image=image_pil),
            lambda: self._pipeline({"image": image_pil}),
            lambda: self._pipeline(image_pil),
        ]

        output: Any | None = None
        errors: list[str] = []
        for fn in attempts:
            try:
                output = fn()
                break
            except Exception as exc:  # pragma: no cover
                errors.append(str(exc))

        if output is None:
            raise RuntimeError(
                "Qwen layered inference failed for all input call patterns. "
                f"Errors: {' | '.join(errors)}"
            )

        return self._extract_layers(output, image_rgb)
