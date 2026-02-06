from __future__ import annotations

import json
import logging

from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.core.paths import resolve_paths
from image_edit_dataset_factory.core.schema import SampleRecord
from image_edit_dataset_factory.qa.consistency import run_consistency
from image_edit_dataset_factory.qa.linter import lint_dataset
from image_edit_dataset_factory.qa.report import write_lint_report, write_qa_report

LOGGER = logging.getLogger(__name__)


def run_qa(cfg: AppConfig) -> dict[str, object]:
    paths = resolve_paths(cfg)
    paths.ensure_runtime_dirs()

    index_path = paths.reports_dir / "index.jsonl"
    if index_path.exists():
        rows = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        samples = [SampleRecord.model_validate(item) for item in rows]
    else:
        samples = []

    lint_issues = lint_dataset(paths.dataset_dir)
    lint_report = write_lint_report(lint_issues, paths.reports_dir)

    qa_scores = run_consistency(samples, cfg.qa)
    qa_csv, qa_summary = write_qa_report(qa_scores, paths.reports_dir / "qa")
    qa_fail_count = sum(1 for item in qa_scores if not item.passed)

    report = {
        "lint_report": str(lint_report),
        "lint_issue_count": len(lint_issues),
        "qa_csv": str(qa_csv),
        "qa_summary": str(qa_summary),
        "qa_fail_count": qa_fail_count,
    }
    LOGGER.info("qa_done summary=%s", report)
    return report
