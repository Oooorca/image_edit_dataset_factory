from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.pipeline.orchestrator import PipelineOrchestrator
from services.edit_service.app import EditServiceSettings
from services.edit_service.app import create_app as create_edit_app
from services.layered_service.app import (
    LayeredServiceSettings,
)
from services.layered_service.app import (
    create_app as create_layered_app,
)

CATEGORIES = ["人物物体一致性", "物体一致性", "物理变化"]


def _create_images(root: Path) -> None:
    for category in CATEGORIES:
        folder = root / category / "case_001"
        folder.mkdir(parents=True, exist_ok=True)
        arr = np.zeros((320, 320, 3), dtype=np.uint8)
        arr[:, :, :] = [80, 120, 160]
        arr[96:224, 96:224] = [220, 100, 90]
        Image.fromarray(arr).save(folder / "img.jpg", quality=95)


def test_pipeline_api_mode_with_mock_services(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    _create_images(data_root)

    layered_settings = LayeredServiceSettings()
    layered_settings.backend = "mock"
    layered_settings.cache_dir = tmp_path / "service_cache" / "layered"
    edit_settings = EditServiceSettings()
    edit_settings.backend = "mock"
    edit_settings.cache_dir = tmp_path / "service_cache" / "edit"

    layered_client = TestClient(create_layered_app(layered_settings))
    edit_client = TestClient(create_edit_app(edit_settings))

    from image_edit_dataset_factory.clients.http_client import RetryingJsonHttpClient

    def fake_post_json(self: RetryingJsonHttpClient, path: str, payload: dict):
        if "layered" in self.endpoint:
            resp = layered_client.post(path, json=payload)
        elif "edit" in self.endpoint:
            resp = edit_client.post(path, json=payload)
        else:
            raise RuntimeError(f"unknown endpoint: {self.endpoint}")
        resp.raise_for_status()
        return resp.json()

    monkeypatch.setattr(RetryingJsonHttpClient, "post_json", fake_post_json)

    cfg = AppConfig.model_validate(
        {
            "paths": {
                "project_root": str(tmp_path),
                "data_root": "./data",
                "output_root": "./outputs",
                "logs_root": "./logs",
            },
            "ingest": {
                "include_categories": CATEGORIES,
                "recursive": True,
                "max_images_per_category": 1,
            },
            "backends": {
                "layered_backend": "api",
                "edit_backend": "api",
                "device": "cpu",
            },
            "services": {
                "api_mode": True,
                "layered": {
                    "enabled": True,
                    "endpoint": "http://layered.local",
                    "send_mode": "base64",
                    "fallback_to_mock": False,
                },
                "edit": {
                    "enabled": True,
                    "endpoint": "http://edit.local",
                    "send_mode": "base64",
                    "fallback_to_mock": False,
                },
            },
            "pipeline": {
                "ingest": True,
                "decompose": True,
                "generate": True,
                "export": True,
                "qa": True,
                "resume": False,
            },
        }
    )

    summary = PipelineOrchestrator(cfg).run()
    dataset_root = tmp_path / "outputs" / "dataset"

    assert dataset_root.exists()
    assert len(list(dataset_root.rglob("*_result.jpg"))) >= 3
    assert int(summary["lint_issue_count"]) == 0
