from pathlib import Path

import numpy as np
from PIL import Image

from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.pipeline.orchestrator import PipelineOrchestrator

CATEGORIES = ["人物物体一致性", "物体一致性", "物理变化"]


def _create_images(root: Path) -> None:
    for category in CATEGORIES:
        folder = root / category / "case_001"
        folder.mkdir(parents=True, exist_ok=True)
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        arr[:, :, 0] = 70
        arr[:, :, 1] = 120
        arr[:, :, 2] = 180
        arr[80:180, 80:180] = [220, 80, 90]
        Image.fromarray(arr).save(folder / "img.jpg", quality=95)


def test_mock_pipeline_end_to_end(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _create_images(data_root)

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
                "max_images_per_category": 2,
            },
            "filter": {
                "enabled": True,
                "min_width": 64,
                "min_height": 64,
                "reject_grayscale": False,
                "reject_borders": False,
            },
            "backends": {
                "layered_backend": "mock",
                "edit_backend": "opencv",
                "use_modelscope": False,
                "device": "cpu",
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
