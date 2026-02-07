from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image

from image_edit_dataset_factory.backends.mock_backend import MockLayeredDecomposer
from image_edit_dataset_factory.backends.qwen_layered_modelscope import (
    QwenLayeredModelScopeBackend,
)
from image_edit_dataset_factory.clients.contracts import (
    LayeredInferRequest,
    LayeredInferResponse,
    LayerInfo,
)
from image_edit_dataset_factory.clients.serialization import (
    decode_rgb_png_base64,
    encode_mask_png_base64,
    encode_rgba_png_base64,
)
from image_edit_dataset_factory.utils.image_io import read_image_rgb
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


class LayeredServiceSettings(ServiceRuntime):
    def __init__(self) -> None:
        super().__init__(
            cache_dir=Path(env_str("LAYERED_CACHE_DIR", "outputs/service_cache/layered")).resolve(),
            preload=env_bool("LAYERED_PRELOAD", False),
            max_concurrency=env_int("LAYERED_MAX_CONCURRENCY", 1),
            max_queue=env_int("LAYERED_MAX_QUEUE", 16),
            infer_timeout_sec=env_float("LAYERED_INFER_TIMEOUT_SEC", 300.0),
        )
        self.backend = env_str("LAYERED_BACKEND", "mock").strip().lower()
        self.model_dir = env_str("LAYERED_MODEL_DIR", "qwen/Qwen-Image-Layered")
        self.device = env_str("LAYERED_DEVICE", "cuda")


def _build_backend(settings: LayeredServiceSettings) -> Any:
    if settings.backend == "qwen":
        return QwenLayeredModelScopeBackend(model_dir=settings.model_dir, device=settings.device)
    if settings.backend == "mock":
        return MockLayeredDecomposer()
    raise RuntimeError(f"unsupported layered backend: {settings.backend}")


def _save_rgba(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba.astype(np.uint8), mode="RGBA").save(path)


def _build_input_image(req: LayeredInferRequest) -> np.ndarray:
    if req.image_path:
        return read_image_rgb(req.image_path)
    if req.image_b64:
        return decode_rgb_png_base64(req.image_b64)
    raise HTTPException(
        status_code=422,
        detail={"code": "invalid_input", "message": "image_path or image_b64 required"},
    )


def create_app(settings: LayeredServiceSettings | None = None) -> FastAPI:
    cfg = settings or LayeredServiceSettings()
    backend = _build_backend(cfg)
    state = BackendState(backend)
    limiter = RequestLimiter(max_concurrency=cfg.max_concurrency, max_queue=cfg.max_queue)

    app = FastAPI(title="IEDF Layered Service", version="1.0.0")

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

    @app.post("/infer", response_model=LayeredInferResponse)
    async def infer(req: LayeredInferRequest) -> LayeredInferResponse:
        sample_id = req.sample_id or "unknown"
        LOGGER.info("layered_infer_start request_id=%s sample_id=%s", req.request_id, sample_id)

        def _run() -> LayeredInferResponse:
            image = _build_input_image(req)
            layers = backend.decompose(image)
            cache_dir = cfg.cache_dir / req.request_id
            if req.save_cache:
                cache_dir.mkdir(parents=True, exist_ok=True)

            items: list[LayerInfo] = []
            for layer in layers:
                rgba_path: str | None = None
                alpha_path: str | None = None
                if req.save_cache:
                    rgba_file = cache_dir / f"layer_{layer.layer_id:02d}.png"
                    alpha_file = cache_dir / f"layer_{layer.layer_id:02d}_alpha.png"
                    _save_rgba(rgba_file, layer.rgba)
                    Image.fromarray(layer.alpha.astype(np.uint8), mode="L").save(alpha_file)
                    rgba_path = str(rgba_file)
                    alpha_path = str(alpha_file)

                items.append(
                    LayerInfo(
                        layer_id=layer.layer_id,
                        rgba_b64=encode_rgba_png_base64(layer.rgba) if req.return_b64 else None,
                        alpha_b64=encode_mask_png_base64(layer.alpha) if req.return_b64 else None,
                        rgba_path=rgba_path,
                        alpha_path=alpha_path,
                    )
                )

            runtime = infer_runtime_name(backend, default=cfg.backend)
            return LayeredInferResponse(
                request_id=req.request_id,
                runtime=runtime,
                width=image.shape[1],
                height=image.shape[0],
                layers=items,
                cache_dir=str(cache_dir) if req.save_cache else None,
            )

        try:
            result = await limiter.run(_run, timeout_sec=cfg.infer_timeout_sec)
        except HTTPException:
            raise
        except Exception as exc:
            state.last_error = str(exc)
            LOGGER.exception(
                "layered_infer_failed request_id=%s sample_id=%s error=%s",
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

        LOGGER.info("layered_infer_done request_id=%s sample_id=%s", req.request_id, sample_id)
        return result

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.layered_service.app:app", host="0.0.0.0", port=8101, workers=1)
