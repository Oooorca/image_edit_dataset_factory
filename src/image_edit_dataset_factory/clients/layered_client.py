from __future__ import annotations

import uuid
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from image_edit_dataset_factory.backends.layered_base import LayerOutput
from image_edit_dataset_factory.clients.contracts import (
    LayeredInferRequest,
    LayeredInferResponse,
    LayerInfo,
)
from image_edit_dataset_factory.clients.http_client import RetryingJsonHttpClient
from image_edit_dataset_factory.clients.serialization import (
    decode_mask_png_base64,
    decode_rgba_png_base64,
    encode_rgb_png_base64,
)
from image_edit_dataset_factory.core.config import ServiceEndpointConfig


class LayeredServiceClient:
    def __init__(
        self,
        endpoint_cfg: ServiceEndpointConfig,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.endpoint_cfg = endpoint_cfg
        self.http = RetryingJsonHttpClient(
            endpoint=endpoint_cfg.endpoint,
            timeout_sec=endpoint_cfg.timeout_sec,
            max_retries=endpoint_cfg.max_retries,
            backoff_sec=endpoint_cfg.backoff_sec,
            transport=transport,
        )

    @staticmethod
    def _layer_from_info(info: LayerInfo) -> LayerOutput:
        if info.rgba_b64:
            rgba = decode_rgba_png_base64(info.rgba_b64).copy()
        elif info.rgba_path:
            with Image.open(info.rgba_path) as img:
                rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8).copy()
        else:
            msg = f"layer {info.layer_id} has neither rgba_b64 nor rgba_path"
            raise RuntimeError(msg)

        if info.alpha_b64:
            alpha = decode_mask_png_base64(info.alpha_b64)
        elif info.alpha_path:
            with Image.open(info.alpha_path) as img:
                alpha = np.asarray(img.convert("L"), dtype=np.uint8)
        else:
            alpha = rgba[:, :, 3].astype(np.uint8)

        if rgba.shape[:2] != alpha.shape[:2]:
            msg = (
                f"layer shape mismatch layer_id={info.layer_id}, "
                f"rgba_shape={rgba.shape}, alpha_shape={alpha.shape}"
            )
            raise RuntimeError(msg)

        rgba[:, :, 3] = alpha
        return LayerOutput(layer_id=info.layer_id, rgba=rgba, alpha=alpha)

    def decompose(
        self, image_rgb: np.ndarray, sample_id: str | None = None
    ) -> list[LayerOutput]:
        payload = LayeredInferRequest(
            request_id=str(uuid.uuid4()),
            sample_id=sample_id,
            image_b64=encode_rgb_png_base64(image_rgb),
            return_b64=True,
            save_cache=True,
        ).model_dump(mode="json")

        data = self.http.post_json("/infer", payload)
        response = LayeredInferResponse.model_validate(data)
        return [self._layer_from_info(item) for item in response.layers]

    def decompose_from_path(
        self, image_path: str | Path, sample_id: str | None = None
    ) -> list[LayerOutput]:
        if self.endpoint_cfg.send_mode != "path":
            with Image.open(image_path) as img:
                image = np.asarray(img.convert("RGB"), dtype=np.uint8)
            return self.decompose(image, sample_id=sample_id)

        payload = LayeredInferRequest(
            request_id=str(uuid.uuid4()),
            sample_id=sample_id,
            image_path=str(image_path),
            return_b64=True,
            save_cache=True,
        ).model_dump(mode="json")

        data = self.http.post_json("/infer", payload)
        response = LayeredInferResponse.model_validate(data)
        return [self._layer_from_info(item) for item in response.layers]
