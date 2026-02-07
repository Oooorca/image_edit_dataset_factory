import json

import httpx
import numpy as np

from image_edit_dataset_factory.clients.edit_client import EditServiceClient
from image_edit_dataset_factory.clients.layered_client import LayeredServiceClient
from image_edit_dataset_factory.clients.serialization import (
    encode_mask_png_base64,
    encode_rgb_png_base64,
    encode_rgba_png_base64,
)
from image_edit_dataset_factory.core.config import ServiceEndpointConfig


def test_layered_client_retry_success() -> None:
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[:, :, :3] = [10, 20, 30]
    rgba[:, :, 3] = 255
    alpha = np.full((16, 16), 255, dtype=np.uint8)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(status_code=500, json={"error": "temporary"})

        body = {
            "request_id": "r1",
            "runtime": "mock",
            "width": 16,
            "height": 16,
            "layers": [
                {
                    "layer_id": 0,
                    "rgba_b64": encode_rgba_png_base64(rgba),
                    "alpha_b64": encode_mask_png_base64(alpha),
                }
            ],
        }
        return httpx.Response(status_code=200, json=body)

    transport = httpx.MockTransport(handler)
    cfg = ServiceEndpointConfig(endpoint="http://test-layered", max_retries=1, backoff_sec=0)
    client = LayeredServiceClient(cfg, transport=transport)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    layers = client.decompose(image)
    assert calls["n"] == 2
    assert len(layers) == 1
    assert layers[0].rgba.shape == (16, 16, 4)


def test_edit_client_unavailable_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(handler)
    cfg = ServiceEndpointConfig(endpoint="http://test-edit", max_retries=1, backoff_sec=0)
    client = EditServiceClient(cfg, transport=transport)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)

    try:
        _ = client.inpaint(image, mask, prompt="edit")
    except RuntimeError as exc:
        message = str(exc)
        assert "service request failed" in message
    else:
        raise AssertionError("expected RuntimeError")


def test_edit_client_success_from_mock_server() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, :] = [100, 120, 140]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["prompt"] == "edit"
        body = {
            "request_id": payload["request_id"],
            "runtime": "mock",
            "width": 16,
            "height": 16,
            "result_image_b64": encode_rgb_png_base64(image),
        }
        return httpx.Response(status_code=200, json=body)

    transport = httpx.MockTransport(handler)
    cfg = ServiceEndpointConfig(endpoint="http://test-edit", max_retries=0)
    client = EditServiceClient(cfg, transport=transport)

    mask = np.zeros((16, 16), dtype=np.uint8)
    result = client.inpaint(image, mask, prompt="edit")
    assert result.shape == (16, 16, 3)


def test_client_timeout_retries_then_fail() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        calls["n"] += 1
        raise httpx.ReadTimeout("timeout")

    transport = httpx.MockTransport(handler)
    cfg = ServiceEndpointConfig(endpoint="http://test-timeout", max_retries=2, backoff_sec=0)
    client = LayeredServiceClient(cfg, transport=transport)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        _ = client.decompose(image)
    except RuntimeError as exc:
        assert "service request failed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert calls["n"] == 3
