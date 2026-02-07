from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from image_edit_dataset_factory.clients.serialization import (
    decode_rgb_png_base64,
    encode_mask_png_base64,
    encode_rgb_png_base64,
)
from services.edit_service.app import EditServiceSettings, create_app


def test_edit_service_infer_contract(tmp_path: Path) -> None:
    settings = EditServiceSettings()
    settings.backend = "mock"
    settings.cache_dir = tmp_path / "cache"
    settings.preload = False
    settings.max_concurrency = 1
    settings.max_queue = 2
    settings.infer_timeout_sec = 30

    app = create_app(settings)
    client = TestClient(app)

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :] = [60, 80, 100]
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255

    payload = {
        "request_id": "req-edit-1",
        "sample_id": "sample-1",
        "image_b64": encode_rgb_png_base64(image),
        "mask_b64": encode_mask_png_base64(mask),
        "prompt": "delete object",
        "return_b64": True,
        "save_cache": True,
    }

    response = client.post("/infer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == "req-edit-1"
    assert data["width"] == 64
    assert data["height"] == 64
    result = decode_rgb_png_base64(data["result_image_b64"])
    assert result.shape == (64, 64, 3)
