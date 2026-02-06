from __future__ import annotations

import logging

from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.pipeline.decompose import run_decompose
from image_edit_dataset_factory.pipeline.export import run_export
from image_edit_dataset_factory.pipeline.generate_samples import run_generate
from image_edit_dataset_factory.pipeline.ingest import run_ingest
from image_edit_dataset_factory.pipeline.qa_step import run_qa

LOGGER = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    def run(self) -> dict[str, object]:
        summary: dict[str, object] = {}

        if self.cfg.pipeline.ingest:
            summary["ingest_manifest"] = str(run_ingest(self.cfg))

        if self.cfg.pipeline.decompose:
            summary["decompose_manifest"] = str(run_decompose(self.cfg))

        if self.cfg.pipeline.generate:
            summary["generated_manifest"] = str(run_generate(self.cfg))

        if self.cfg.pipeline.export:
            summary["index_jsonl"] = str(run_export(self.cfg))

        if self.cfg.pipeline.qa:
            summary.update(run_qa(self.cfg))

        LOGGER.info("pipeline_done summary=%s", summary)
        return summary
