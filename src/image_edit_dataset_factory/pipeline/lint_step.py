from __future__ import annotations

import json
import logging
from pathlib import Path

from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.qa.linter import lint_dataset

LOGGER = logging.getLogger(__name__)


def run_lint(cfg: AppConfig) -> tuple[Path, int]:
    root = Path(cfg.paths.root)
    dataset_root = root / cfg.paths.outputs_dir / "dataset"
    reports_dir = root / cfg.paths.outputs_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    issues = lint_dataset(dataset_root)
    output = reports_dir / "lint_issues.json"
    output.write_text(
        json.dumps(
            [issue.model_dump(mode="json") for issue in issues], ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    LOGGER.info("lint_done issues=%s report=%s", len(issues), output)
    return output, len(issues)
