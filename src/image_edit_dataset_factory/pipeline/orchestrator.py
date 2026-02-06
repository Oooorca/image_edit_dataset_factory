from __future__ import annotations

import logging
from pathlib import Path

from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.pipeline.decompose import run_decompose
from image_edit_dataset_factory.pipeline.export_step import run_export
from image_edit_dataset_factory.pipeline.filter import run_filter
from image_edit_dataset_factory.pipeline.generate_samples import run_generate
from image_edit_dataset_factory.pipeline.ingest import run_ingest
from image_edit_dataset_factory.pipeline.lint_step import run_lint
from image_edit_dataset_factory.pipeline.qa_step import run_qa

LOGGER = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg.paths.root)

    def _stage_output(self, name: str) -> Path:
        data_dir = self.root / self.cfg.paths.data_dir
        outputs_dir = self.root / self.cfg.paths.outputs_dir

        mapping = {
            "ingest": data_dir / "raw" / "source_meta.jsonl",
            "filter": data_dir / "filtered" / "filtered_meta.jsonl",
            "decompose": outputs_dir / "layers_cache" / "decompose_index.jsonl",
            "generate": outputs_dir / "generated" / "generated_samples.jsonl",
            "export": outputs_dir / "reports" / "index.jsonl",
            "lint": outputs_dir / "reports" / "lint_issues.json",
            "qa": outputs_dir / "reports" / "qa" / "qa_summary.json",
        }
        return mapping[name]

    def _should_skip(self, stage: str) -> bool:
        if not self.cfg.pipeline.resume:
            return False
        return self._stage_output(stage).exists()

    def run(self) -> dict[str, object]:
        summary: dict[str, object] = {}

        if self.cfg.pipeline.ingest:
            if self._should_skip("ingest"):
                LOGGER.info("stage_skipped stage=ingest reason=resume")
            else:
                summary["ingest"] = str(run_ingest(self.cfg))

        if self.cfg.pipeline.filter:
            if self._should_skip("filter"):
                LOGGER.info("stage_skipped stage=filter reason=resume")
            else:
                summary["filter"] = str(run_filter(self.cfg))

        if self.cfg.pipeline.decompose:
            if self._should_skip("decompose") and not self.cfg.decompose.overwrite:
                LOGGER.info("stage_skipped stage=decompose reason=resume")
            else:
                summary["decompose"] = str(run_decompose(self.cfg))

        if self.cfg.pipeline.generate:
            if self._should_skip("generate"):
                LOGGER.info("stage_skipped stage=generate reason=resume")
            else:
                summary["generate"] = str(run_generate(self.cfg))

        if self.cfg.pipeline.export:
            if self._should_skip("export"):
                LOGGER.info("stage_skipped stage=export reason=resume")
            else:
                summary["export"] = str(run_export(self.cfg))

        if self.cfg.pipeline.lint:
            lint_report, lint_issue_count = run_lint(self.cfg)
            summary["lint_report"] = str(lint_report)
            summary["lint_issue_count"] = lint_issue_count

        if self.cfg.pipeline.qa and self.cfg.qa.enabled:
            qa_csv, qa_summary, qa_fail_count = run_qa(self.cfg)
            summary["qa_csv"] = str(qa_csv)
            summary["qa_summary"] = str(qa_summary)
            summary["qa_fail_count"] = qa_fail_count

        LOGGER.info("pipeline_done summary=%s", summary)
        return summary
