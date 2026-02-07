from __future__ import annotations

import uuid
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from image_edit_dataset_factory.clients.contracts import EditInferRequest, EditInferResponse
from image_edit_dataset_factory.clients.http_client import RetryingJsonHttpClient
from image_edit_dataset_factory.clients.serialization import (
    decode_rgb_png_base64,
    encode_mask_png_base64,
    encode_rgb_png_base64,
)
from image_edit_dataset_factory.core.config import ServiceEndpointConfig


class EditServiceClient:
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

    def inpaint(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        prompt: str | None = None,
        sample_id: str | None = None,
    ) -> np.ndarray:
        payload = EditInferRequest(
            request_id=str(uuid.uuid4()),
            sample_id=sample_id,
            image_b64=encode_rgb_png_base64(image_rgb),
            mask_b64=encode_mask_png_base64(mask),
            prompt=prompt,
            return_b64=True,
            save_cache=True,
        ).model_dump(mode="json")

        data = self.http.post_json("/infer", payload)
        response = EditInferResponse.model_validate(data)

        if response.result_image_b64:
            return decode_rgb_png_base64(response.result_image_b64)

        if response.result_image_path:
            with Image.open(response.result_image_path) as img:
                return np.asarray(img.convert("RGB"), dtype=np.uint8)

        msg = "edit service returned neither result_image_b64 nor result_image_path"
        raise RuntimeError(msg)

    def inpaint_from_path(
        self,
        image_path: str | Path,
        mask_path: str | Path,
        prompt: str | None = None,
        sample_id: str | None = None,
    ) -> np.ndarray:
        if self.endpoint_cfg.send_mode != "path":
            with Image.open(image_path) as img:
                image = np.asarray(img.convert("RGB"), dtype=np.uint8)
            with Image.open(mask_path) as mask_img:
                mask = np.asarray(mask_img.convert("L"), dtype=np.uint8)
            return self.inpaint(image, mask, prompt=prompt, sample_id=sample_id)

        payload = EditInferRequest(
            request_id=str(uuid.uuid4()),
            sample_id=sample_id,
            image_path=str(image_path),
            mask_path=str(mask_path),
            prompt=prompt,
            return_b64=True,
            save_cache=True,
        ).model_dump(mode="json")

        data = self.http.post_json("/infer", payload)
        response = EditInferResponse.model_validate(data)

        if response.result_image_b64:
            return decode_rgb_png_base64(response.result_image_b64)

        if response.result_image_path:
            with Image.open(response.result_image_path) as img:
                return np.asarray(img.convert("RGB"), dtype=np.uint8)

        msg = "edit service returned neither result_image_b64 nor result_image_path"
        raise RuntimeError(msg)
