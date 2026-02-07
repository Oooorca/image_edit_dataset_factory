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

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        text = str(exc).lower()
        return "cuda out of memory" in text or ("out of memory" in text and "cuda" in text)

    @staticmethod
    def _resize_image_and_mask(
        image_rgb: np.ndarray,
        mask: np.ndarray,
        max_side: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        from PIL import Image

        h, w = image_rgb.shape[:2]
        if max(h, w) <= max_side:
            return image_rgb, mask

        scale = max_side / float(max(h, w))
        new_w = max(64, int(w * scale))
        new_h = max(64, int(h * scale))
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        new_w = max(64, new_w)
        new_h = max(64, new_h)

        image_pil = Image.fromarray(image_rgb, mode="RGB").resize(
            (new_w, new_h), Image.Resampling.LANCZOS
        )
        mask_pil = Image.fromarray(mask.astype(np.uint8), mode="L").resize(
            (new_w, new_h), Image.Resampling.NEAREST
        )
        return np.asarray(image_pil, dtype=np.uint8), np.asarray(mask_pil, dtype=np.uint8)

    def _lazy_init(self) -> None:
        if self._pipeline is not None:
            return

        local_dir = resolve_local_model_dir(self.model_dir, self.MODEL_ID)

        # Prefer diffusers first for local checkpoints. This path avoids
        # ModelScope runtime deps like `swift` and is typically more stable.
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
                torch_dtype=torch_dtype,
            )
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing("max")
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()

            if self.device.lower().startswith("cuda"):
                if hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                else:
                    pipe = pipe.to("cuda")
            else:
                pipe = pipe.to("cpu")
            self._pipeline = pipe
            self._runtime = "diffusers"
            return
        except Exception as diffusers_exc:  # pragma: no cover
            diffusers_error = str(diffusers_exc)

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

        raise RuntimeError(
            "Failed to initialize edit backend from local model directory. "
            f"model_dir={local_dir}. Diffusers error: {diffusers_error}. "
            f"ModelScope attempts: {' | '.join(ms_errors)}."
        )

    @staticmethod
    def _extract_output_image(output: Any) -> np.ndarray | None:
        if output is None:
            return None

        if hasattr(output, "images"):
            images = output.images
            if isinstance(images, list) and images:
                return QwenImageEditModelScopeBackend._extract_output_image(images[0])

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

        import torch
        from PIL import Image

        text = prompt or "remove object and naturally repair the background"
        original_h, original_w = image_rgb.shape[:2]
        max_side_candidates = [1024, 896, 768, 640, 512]
        max_side_candidates = sorted(set(max_side_candidates), reverse=True)

        all_errors: list[str] = []
        for max_side in max_side_candidates:
            resized_image, resized_mask = self._resize_image_and_mask(
                image_rgb=image_rgb,
                mask=mask,
                max_side=max_side,
            )
            image_pil = to_pil(resized_image)

            # QwenImageEditPipeline (diffusers) in this env does not accept `mask`.
            # We run full-image edit first, then blend edited region by mask.
            attempts = [
                lambda image_pil=image_pil: self._pipeline(
                    image=image_pil,
                    prompt=text,
                    num_inference_steps=30,
                    output_type="pil",
                    return_dict=True,
                ),
                lambda image_pil=image_pil: self._pipeline(
                    image=image_pil,
                    prompt=text,
                    num_inference_steps=24,
                    output_type="pil",
                    return_dict=True,
                ),
            ]

            per_scale_errors: list[str] = []
            for fn in attempts:
                try:
                    with torch.inference_mode():
                        output = fn()
                    result = self._extract_output_image(output)
                    if result is None:
                        per_scale_errors.append("pipeline returned no valid output image")
                        continue

                    if result.shape[:2] != resized_image.shape[:2]:
                        result = np.asarray(
                            Image.fromarray(result).resize(
                                (resized_image.shape[1], resized_image.shape[0]),
                                Image.Resampling.LANCZOS,
                            ),
                            dtype=np.uint8,
                        )

                    # Localize edit result with mask to emulate inpaint behavior.
                    mask_f = (resized_mask.astype(np.float32) / 255.0)[..., None]
                    blended = (
                        result.astype(np.float32) * mask_f
                        + resized_image.astype(np.float32) * (1.0 - mask_f)
                    ).clip(0, 255).astype(np.uint8)

                    if blended.shape[:2] != (original_h, original_w):
                        result = np.asarray(
                            Image.fromarray(blended).resize(
                                (original_w, original_h), Image.Resampling.LANCZOS
                            ),
                            dtype=np.uint8,
                        )
                    else:
                        result = blended
                    return result
                except Exception as exc:  # pragma: no cover
                    if self._is_cuda_oom(exc):
                        torch.cuda.empty_cache()
                    per_scale_errors.append(str(exc))

            all_errors.append(
                "max_side="
                f"{max_side}: {' | '.join(per_scale_errors) if per_scale_errors else 'failed'}"
            )
            if self.device.lower().startswith("cuda"):
                torch.cuda.empty_cache()

        raise RuntimeError(
            "Qwen image edit inference failed or produced no valid output image. "
            f"runtime={self._runtime}, attempts={' || '.join(all_errors)}"
        )
