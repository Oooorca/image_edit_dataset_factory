from __future__ import annotations

from typing import Any

import numpy as np

from image_edit_dataset_factory.backends.edit_base import EditorBackend
from image_edit_dataset_factory.backends.modelscope_utils import (
    pil_to_rgb_array,
    resolve_local_model_dir,
    to_pil,
)


class QwenImageEditModelScopeBackend(EditorBackend):
    """ModelScope backend for Qwen/Qwen-Image-Edit.

    The backend only uses locally downloaded weights. It never triggers downloads.
    """

    MODEL_ID = "Qwen/Qwen-Image-Edit"

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
            "Failed to initialize ModelScope edit pipeline. "
            f"model_dir={local_dir}. Attempts: {' | '.join(errors)}"
        )

    @staticmethod
    def _extract_output_image(output: Any) -> np.ndarray | None:
        if output is None:
            return None

        if isinstance(output, dict):
            candidates = [
                output.get("output_img"),
                output.get("output_image"),
                output.get("result"),
                output.get("image"),
            ]
            imgs = output.get("output_imgs")
            if isinstance(imgs, list) and imgs:
                candidates.insert(0, imgs[0])

            for item in candidates:
                image = QwenImageEditModelScopeBackend._extract_output_image(item)
                if image is not None:
                    return image
            return None

        if isinstance(output, list) and output:
            return QwenImageEditModelScopeBackend._extract_output_image(output[0])

        if hasattr(output, "mode"):
            try:
                return pil_to_rgb_array(output)
            except Exception:
                return None

        if isinstance(output, np.ndarray):
            arr = output
            if arr.ndim == 3 and arr.shape[2] >= 3:
                return arr[:, :, :3].astype(np.uint8)

        return None

    def inpaint(
        self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str | None = None
    ) -> np.ndarray:
        self._lazy_init()

        from PIL import Image

        image_pil = to_pil(image_rgb)
        mask_pil = Image.fromarray(mask.astype(np.uint8), mode="L")
        text = prompt or "remove object and naturally repair the background"

        attempts = [
            lambda: self._pipeline(image=image_pil, mask=mask_pil, prompt=text),
            lambda: self._pipeline({"image": image_pil, "mask": mask_pil, "prompt": text}),
            lambda: self._pipeline({"img": image_pil, "mask": mask_pil, "text": text}),
        ]

        output: Any | None = None
        errors: list[str] = []
        for fn in attempts:
            try:
                output = fn()
                result = self._extract_output_image(output)
                if result is not None:
                    return result
            except Exception as exc:  # pragma: no cover
                errors.append(str(exc))

        raise RuntimeError(
            "Qwen image edit inference failed or produced no valid output image. "
            f"Errors: {' | '.join(errors)}"
        )
