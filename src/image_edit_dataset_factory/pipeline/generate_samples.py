from __future__ import annotations

import logging
from pathlib import Path

from image_edit_dataset_factory.backends.factory import build_edit_backend
from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.core.enums import EditTask
from image_edit_dataset_factory.core.paths import resolve_paths
from image_edit_dataset_factory.core.schema import DecomposeRecord, SampleRecord, SourceSample
from image_edit_dataset_factory.pipeline.generate.base import BaseGenerator, GenerationContext
from image_edit_dataset_factory.pipeline.generate.consistency import ConsistencyGenerator
from image_edit_dataset_factory.pipeline.generate.semantic import SemanticGenerator
from image_edit_dataset_factory.pipeline.generate.structural import StructuralGenerator
from image_edit_dataset_factory.utils.jsonl import read_jsonl, write_jsonl

LOGGER = logging.getLogger(__name__)


GENERATOR_MAP: dict[str, type[BaseGenerator]] = {
    EditTask.SEMANTIC.value: SemanticGenerator,
    EditTask.STRUCTURAL.value: StructuralGenerator,
    EditTask.CONSISTENCY.value: ConsistencyGenerator,
}


def run_generate(cfg: AppConfig) -> Path:
    paths = resolve_paths(cfg)
    paths.ensure_runtime_dirs()

    source_rows = [
        SourceSample.model_validate(item)
        for item in read_jsonl(paths.manifests_dir / "source_manifest.jsonl")
    ]
    decompose_rows = [
        DecomposeRecord.model_validate(item)
        for item in read_jsonl(paths.manifests_dir / "decompose_manifest.jsonl")
    ]
    decompose_map = {item.source_id: item for item in decompose_rows}

    context = GenerationContext(
        cfg=cfg,
        staging_dir=paths.staging_dir / "generated",
        edit_backend=build_edit_backend(cfg),
    )

    generated: list[SampleRecord] = []
    for source in source_rows:
        decompose = decompose_map.get(source.source_id)
        if decompose is None:
            LOGGER.warning("generate_skip_no_decompose source_id=%s", source.source_id)
            continue

        task_name = cfg.generate.category_to_task.get(
            source.dataset_category, EditTask.SEMANTIC.value
        )
        generator_cls = GENERATOR_MAP.get(task_name)
        if generator_cls is None:
            LOGGER.warning(
                "generate_skip_unknown_task category=%s task=%s", source.dataset_category, task_name
            )
            continue

        generator = generator_cls(context)
        generated.append(generator.generate(source, decompose))

    out_path = paths.manifests_dir / "generated_manifest.jsonl"
    write_jsonl(out_path, [item.model_dump(mode="json") for item in generated])
    LOGGER.info("generate_done count=%s manifest=%s", len(generated), out_path)
    return out_path
