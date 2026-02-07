from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image

from image_edit_dataset_factory.backends.mock_backend import MockEditorBackend
from image_edit_dataset_factory.backends.opencv_fallback import OpenCVFallbackBackend
from image_edit_dataset_factory.backends.qwen_image_edit_modelscope import (
    QwenImageEditModelScopeBackend,
)
from image_edit_dataset_factory.clients.contracts import EditInferRequest, EditInferResponse
from image_edit_dataset_factory.clients.serialization import (
    decode_mask_png_base64,
    decode_rgb_png_base64,
    encode_rgb_png_base64,
)
from image_edit_dataset_factory.utils.image_io import read_image_rgb, read_mask
from image_edit_dataset_factory.utils.mask_ops import mask_from_bbox
from services.common import (
    BackendState,
    RequestLimiter,
    ServiceRuntime,
    env_bool,
    env_float,
    env_int,
    env_str,
    infer_runtime_name,
)

LOGGER = logging.getLogger(__name__)


class EditServiceSettings(ServiceRuntime):
    def __init__(self) -> None:
        super().__init__(
            cache_dir=Path(env_str("EDIT_CACHE_DIR", "outputs/service_cache/edit")).resolve(),
            preload=env_bool("EDIT_PRELOAD", False),
            max_concurrency=env_int("EDIT_MAX_CONCURRENCY", 1),
            max_queue=env_int("EDIT_MAX_QUEUE", 16),
            infer_timeout_sec=env_float("EDIT_INFER_TIMEOUT_SEC", 600.0),
        )
        self.backend = env_str("EDIT_BACKEND", "mock").strip().lower()
        self.model_dir = env_str("EDIT_MODEL_DIR", "Qwen/Qwen-Image-Edit")
        self.device = env_str("EDIT_DEVICE", "cuda")


def _build_backend(settings: EditServiceSettings) -> Any:
    if settings.backend == "qwen":
        return QwenImageEditModelScopeBackend(model_dir=settings.model_dir, device=settings.device)
    if settings.backend == "opencv":
        return OpenCVFallbackBackend()
    if settings.backend == "mock":
        return MockEditorBackend()
    raise RuntimeError(f"unsupported edit backend: {settings.backend}")


def _build_input_image(req: EditInferRequest) -> np.ndarray:
    if req.image_path:
        return read_image_rgb(req.image_path)
    if req.image_b64:
        return decode_rgb_png_base64(req.image_b64)
    raise HTTPException(
        status_code=422,
        detail={"code": "invalid_input", "message": "image_path or image_b64 required"},
    )


def _build_input_mask(req: EditInferRequest, image_shape: tuple[int, int]) -> np.ndarray:
    if req.mask_path:
        return read_mask(req.mask_path)
    if req.mask_b64:
        return decode_mask_png_base64(req.mask_b64)

    h, w = image_shape
    return mask_from_bbox((h, w), (w // 4, h // 4, (3 * w) // 4, (3 * h) // 4))


def create_app(settings: EditServiceSettings | None = None) -> FastAPI:
    cfg = settings or EditServiceSettings()
    backend = _build_backend(cfg)
    state = BackendState(backend)
    limiter = RequestLimiter(max_concurrency=cfg.max_concurrency, max_queue=cfg.max_queue)

    app = FastAPI(title="IEDF Edit Service", version="1.0.0")

    @app.on_event("startup")
    async def _startup() -> None:
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        if cfg.preload:
            state.try_preload()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        return {
            "ready": state.is_ready(),
            "backend": cfg.backend,
            "last_error": state.last_error,
        }

    @app.post("/infer", response_model=EditInferResponse)
    async def infer(req: EditInferRequest) -> EditInferResponse:
        sample_id = req.sample_id or "unknown"
        LOGGER.info("edit_infer_start request_id=%s sample_id=%s", req.request_id, sample_id)

        def _run() -> EditInferResponse:
            image = _build_input_image(req)
            mask = _build_input_mask(req, image_shape=image.shape[:2])
            result = backend.inpaint(image_rgb=image, mask=mask, prompt=req.prompt)

            cache_dir = cfg.cache_dir / req.request_id
            result_path: str | None = None
            if req.save_cache:
                cache_dir.mkdir(parents=True, exist_ok=True)
                output_file = cache_dir / "result.png"
                Image.fromarray(result.astype(np.uint8), mode="RGB").save(output_file)
                result_path = str(output_file)

            runtime = infer_runtime_name(backend, default=cfg.backend)
            return EditInferResponse(
                request_id=req.request_id,
                runtime=runtime,
                width=result.shape[1],
                height=result.shape[0],
                result_image_b64=encode_rgb_png_base64(result) if req.return_b64 else None,
                result_image_path=result_path,
                cache_dir=str(cache_dir) if req.save_cache else None,
            )

        try:
            result = await limiter.run(_run, timeout_sec=cfg.infer_timeout_sec)
        except HTTPException:
            raise
        except Exception as exc:
            state.last_error = str(exc)
            LOGGER.exception(
                "edit_infer_failed request_id=%s sample_id=%s error=%s",
                req.request_id,
                sample_id,
                exc,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "infer_failed",
                    "message": str(exc),
                    "request_id": req.request_id,
                    "sample_id": sample_id,
                },
            ) from exc

        LOGGER.info("edit_infer_done request_id=%s sample_id=%s", req.request_id, sample_id)
        return result

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.edit_service.app:app", host="0.0.0.0", port=8102, workers=1)
