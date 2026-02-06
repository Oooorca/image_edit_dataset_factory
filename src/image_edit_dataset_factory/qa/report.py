from __future__ import annotations

import csv
import json
from pathlib import Path

from image_edit_dataset_factory.core.schema import LintIssue, QAScore


def write_lint_report(issues: list[LintIssue], report_dir: str | Path) -> Path:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lint_issues.json"
    path.write_text(
        json.dumps([item.model_dump(mode="json") for item in issues], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_qa_report(scores: list[QAScore], report_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "qa_scores.csv"
    summary_path = out_dir / "qa_summary.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "passed",
                "mse_outside_region",
                "ssim_outside_region",
                "changed_pixel_ratio_outside_region",
                "details",
            ]
        )
        for item in scores:
            writer.writerow(
                [
                    item.sample_id,
                    item.passed,
                    item.mse_outside_region,
                    item.ssim_outside_region,
                    item.changed_pixel_ratio_outside_region,
                    json.dumps(item.details, ensure_ascii=False),
                ]
            )

    total = len(scores)
    passed = sum(1 for item in scores if item.passed)
    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (passed / total) if total else 0.0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, summary_path
