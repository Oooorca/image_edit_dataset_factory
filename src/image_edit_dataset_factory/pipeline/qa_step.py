from __future__ import annotations

import logging
from pathlib import Path

from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.core.schema import SampleModel
from image_edit_dataset_factory.qa.consistency import run_consistency
from image_edit_dataset_factory.qa.report import write_qa_report
from image_edit_dataset_factory.utils.jsonl import read_jsonl

LOGGER = logging.getLogger(__name__)


def run_qa(cfg: AppConfig) -> tuple[Path, Path, int]:
    root = Path(cfg.paths.root)
    index_path = root / cfg.paths.outputs_dir / "reports" / "index.jsonl"
    qa_dir = root / cfg.paths.outputs_dir / "reports" / "qa"

    samples = [SampleModel.model_validate(row) for row in read_jsonl(index_path)]
    scores = run_consistency(samples, cfg.qa)
    csv_path, json_path = write_qa_report(scores, qa_dir)

    fail_count = sum(1 for item in scores if not item.passed)
    LOGGER.info("qa_done total=%s failed=%s", len(scores), fail_count)
    return csv_path, json_path, fail_count
