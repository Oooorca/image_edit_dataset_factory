from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from image_edit_dataset_factory.backends.edit_base import EditorBackend
from image_edit_dataset_factory.backends.layered_base import LayeredDecomposer, LayerOutput
from image_edit_dataset_factory.clients.edit_client import EditServiceClient
from image_edit_dataset_factory.clients.layered_client import LayeredServiceClient
from image_edit_dataset_factory.core.config import ServiceEndpointConfig

LOGGER = logging.getLogger(__name__)


class ApiLayeredDecomposer(LayeredDecomposer):
    def __init__(
        self,
        endpoint_cfg: ServiceEndpointConfig,
        fallback: LayeredDecomposer | None = None,
    ) -> None:
        self.client = LayeredServiceClient(endpoint_cfg)
        self.fallback = fallback

    def decompose(self, image_rgb: np.ndarray) -> list[LayerOutput]:
        try:
            return self.client.decompose(image_rgb)
        except Exception as exc:
            if self.fallback is None:
                raise
            LOGGER.warning("layered_api_failed_fallback_to_mock error=%s", exc)
            return self.fallback.decompose(image_rgb)

    def decompose_from_path(
        self, image_path: str | Path, sample_id: str | None = None
    ) -> list[LayerOutput]:
        try:
            return self.client.decompose_from_path(image_path=image_path, sample_id=sample_id)
        except Exception as exc:
            if self.fallback is None:
                raise
            LOGGER.warning("layered_api_from_path_failed_fallback_to_mock error=%s", exc)
            with Image.open(image_path) as img:
                image = np.asarray(img.convert("RGB"), dtype=np.uint8)
            return self.fallback.decompose(image)


class ApiEditorBackend(EditorBackend):
    def __init__(
        self,
        endpoint_cfg: ServiceEndpointConfig,
        fallback: EditorBackend | None = None,
    ) -> None:
        self.client = EditServiceClient(endpoint_cfg)
        self.fallback = fallback

    def inpaint(
        self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str | None = None
    ) -> np.ndarray:
        try:
            return self.client.inpaint(image_rgb=image_rgb, mask=mask, prompt=prompt)
        except Exception as exc:
            if self.fallback is None:
                raise
            LOGGER.warning("edit_api_failed_fallback_to_mock error=%s", exc)
            return self.fallback.inpaint(image_rgb=image_rgb, mask=mask, prompt=prompt)

    def inpaint_from_path(
        self,
        image_path: str | Path,
        mask_path: str | Path,
        prompt: str | None = None,
        sample_id: str | None = None,
    ) -> np.ndarray:
        try:
            return self.client.inpaint_from_path(
                image_path=image_path,
                mask_path=mask_path,
                prompt=prompt,
                sample_id=sample_id,
            )
        except Exception as exc:
            if self.fallback is None:
                raise
            LOGGER.warning("edit_api_from_path_failed_fallback_to_mock error=%s", exc)
            with Image.open(image_path) as img:
                image = np.asarray(img.convert("RGB"), dtype=np.uint8)
            with Image.open(mask_path) as mask_img:
                mask = np.asarray(mask_img.convert("L"), dtype=np.uint8)
            return self.fallback.inpaint(image_rgb=image, mask=mask, prompt=prompt)
