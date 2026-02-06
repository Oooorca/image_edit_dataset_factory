from __future__ import annotations

import logging
import shutil
from pathlib import Path

from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.core.schema import SampleModel
from image_edit_dataset_factory.pipeline.export import export_samples
from image_edit_dataset_factory.pipeline.manifest import write_global_index
from image_edit_dataset_factory.utils.jsonl import read_jsonl

LOGGER = logging.getLogger(__name__)


def run_export(cfg: AppConfig) -> Path:
    root = Path(cfg.paths.root)
    generated_path = root / cfg.paths.outputs_dir / "generated" / "generated_samples.jsonl"
    dataset_root = root / cfg.paths.outputs_dir / "dataset"
    reports_root = root / cfg.paths.outputs_dir / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    if not cfg.pipeline.resume and dataset_root.exists():
        shutil.rmtree(dataset_root)

    rows = read_jsonl(generated_path)
    samples = [SampleModel.model_validate(row) for row in rows]
    exported = export_samples(samples, dataset_root)

    csv_path = reports_root / "index.csv"
    jsonl_path = reports_root / "index.jsonl"
    write_global_index(exported, csv_path, jsonl_path)
    LOGGER.info("export_index_written csv=%s jsonl=%s", csv_path, jsonl_path)
    return jsonl_path
