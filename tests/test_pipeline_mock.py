from pathlib import Path

import numpy as np
from PIL import Image

from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.pipeline.orchestrator import PipelineOrchestrator


def _create_input_images(path: Path, count: int = 3) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        arr[:, :, 0] = 60 + i * 10
        arr[:, :, 1] = 120
        arr[:, :, 2] = 180
        arr[80:180, 80:180] = [220, 80, 90]
        Image.fromarray(arr).save(path / f"img_{i:02d}.jpg", quality=95)


def test_mock_pipeline_end_to_end(tmp_path: Path) -> None:
    input_dir = tmp_path / "data" / "input"
    _create_input_images(input_dir)

    cfg = AppConfig.model_validate(
        {
            "paths": {
                "root": str(tmp_path),
                "data_dir": "data",
                "outputs_dir": "outputs",
                "logs_dir": "logs",
            },
            "ingest": {"source_dir": str(input_dir), "symlink": False, "recursive": True},
            "filter": {
                "min_width": 128,
                "min_height": 128,
                "reject_grayscale": False,
                "reject_borders": False,
                "reject_text_like": False,
            },
            "decompose": {"backend": "mock", "overwrite": True},
            "edit": {"backend": "opencv"},
            "generate": {
                "dry_run": False,
                "categories": {
                    "semantic_edit": {"enabled": True, "subtypes": ["delete"], "per_source": 1},
                    "style_edit": {
                        "enabled": True,
                        "subtypes": ["contrast", "brightness", "color_tone"],
                        "per_source": 1,
                    },
                    "structural_edit": {
                        "enabled": True,
                        "subtypes": ["move", "scale"],
                        "per_source": 1,
                    },
                    "portrait_attribute": {"enabled": False, "subtypes": [], "per_source": 0},
                    "text_edit": {"enabled": False, "subtypes": [], "per_source": 0},
                },
            },
            "quotas": {"target_total": 20},
            "qa": {
                "enabled": True,
                "allowed_region_dilation_px": 9,
                "max_mse_outside_region": 8.0,
                "min_ssim_outside_region": 0.95,
                "max_changed_pixel_ratio_outside_region": 0.05,
            },
            "pipeline": {
                "resume": False,
                "ingest": True,
                "filter": True,
                "decompose": True,
                "generate": True,
                "export": True,
                "lint": True,
                "qa": True,
            },
        }
    )

    summary = PipelineOrchestrator(cfg).run()
    dataset_root = tmp_path / "outputs" / "dataset"

    assert dataset_root.exists()
    assert any(dataset_root.rglob("*.jpg"))
    assert (tmp_path / "outputs" / "reports" / "index.csv").exists()
    assert "lint_issue_count" in summary
