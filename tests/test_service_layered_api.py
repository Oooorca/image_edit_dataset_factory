from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from image_edit_dataset_factory.clients.serialization import (
    decode_mask_png_base64,
    encode_rgb_png_base64,
)
from services.layered_service.app import LayeredServiceSettings, create_app


def test_layered_service_infer_contract(tmp_path: Path) -> None:
    settings = LayeredServiceSettings()
    settings.backend = "mock"
    settings.cache_dir = tmp_path / "cache"
    settings.preload = False
    settings.max_concurrency = 1
    settings.max_queue = 2
    settings.infer_timeout_sec = 30

    app = create_app(settings)
    client = TestClient(app)

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :] = [20, 30, 40]
    payload = {
        "request_id": "req-layered-1",
        "sample_id": "sample-1",
        "image_b64": encode_rgb_png_base64(image),
        "return_b64": True,
        "save_cache": True,
    }

    response = client.post("/infer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == "req-layered-1"
    assert data["width"] == 64
    assert data["height"] == 64
    assert len(data["layers"]) >= 1

    alpha_b64 = data["layers"][0]["alpha_b64"]
    alpha = decode_mask_png_base64(alpha_b64)
    assert alpha.shape == (64, 64)
