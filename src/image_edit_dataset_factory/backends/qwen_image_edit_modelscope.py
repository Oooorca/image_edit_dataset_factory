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
    """Edit backend for Qwen/Qwen-Image-Edit using local model files.

    Loading strategy:
    1) try ModelScope pipeline
    2) fallback to diffusers pipeline from local dir

    No download is triggered by this class.
    """

    MODEL_ID = "Qwen/Qwen-Image-Edit"

    def __init__(self, model_dir: str | None, device: str = "cpu") -> None:
        self.model_dir = model_dir
        self.device = device
        self._pipeline: Any | None = None
        self._runtime: str | None = None

    def _lazy_init(self) -> None:
        if self._pipeline is not None:
            return

        local_dir = resolve_local_model_dir(self.model_dir, self.MODEL_ID)

        ms_errors: list[str] = []
        try:
            from modelscope.pipelines import pipeline  # type: ignore
            from modelscope.utils.constant import Tasks  # type: ignore

            device = "gpu" if self.device.lower().startswith("cuda") else self.device
            task_candidates = [
                getattr(Tasks, "image_to_image", None),
                getattr(Tasks, "image_editing", None),
                "image-to-image",
                "image_editing",
                "image-editing",
            ]
            for task in task_candidates:
                if task is None:
                    continue
                try:
                    self._pipeline = pipeline(
                        task=task,
                        model=str(local_dir),
                        device=device,
                        model_revision=None,
                        third_party=None,
                        external_engine_for_llm=True,
                    )
                    self._runtime = "modelscope"
                    return
                except Exception as exc:  # pragma: no cover
                    ms_errors.append(f"task={task!r}: {exc}")
        except Exception as exc:  # pragma: no cover
            ms_errors.append(f"import_modelscope_failed: {exc}")

        try:
            import torch
            from diffusers import DiffusionPipeline

            torch_dtype = (
                torch.bfloat16
                if self.device.lower().startswith("cuda")
                else torch.float32
            )
            pipe = DiffusionPipeline.from_pretrained(
                str(local_dir),
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
            if self.device.lower().startswith("cuda"):
                pipe = pipe.to("cuda")
            else:
                pipe = pipe.to("cpu")
            self._pipeline = pipe
            self._runtime = "diffusers"
            return
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to initialize edit backend from local model directory. "
                f"model_dir={local_dir}. ModelScope attempts: {' | '.join(ms_errors)}. "
                f"Diffusers fallback error: {exc}."
            ) from exc

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
            imgs = output.get("output_imgs") or output.get("images")
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
            lambda: self._pipeline(
                {"image": image_pil, "mask": mask_pil, "prompt": text}
            ),
            lambda: self._pipeline({"img": image_pil, "mask": mask_pil, "text": text}),
            lambda: self._pipeline(prompt=text, image=image_pil, mask_image=mask_pil),
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
            f"runtime={self._runtime}, errors={' | '.join(errors)}"
        )
