from __future__ import annotations

import json
import logging
from pathlib import Path

from image_edit_dataset_factory.backends.factory import build_edit_backend
from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.core.schema import SampleModel, SourceMetadata
from image_edit_dataset_factory.pipeline.generate.base import GenerationContext, SampleGenerator
from image_edit_dataset_factory.pipeline.generate.portrait import PortraitGenerator
from image_edit_dataset_factory.pipeline.generate.semantic import SemanticGenerator
from image_edit_dataset_factory.pipeline.generate.structural import StructuralGenerator
from image_edit_dataset_factory.pipeline.generate.style import StyleGenerator
from image_edit_dataset_factory.pipeline.generate.text import TextGenerator
from image_edit_dataset_factory.pipeline.quotas import enforce_quotas
from image_edit_dataset_factory.utils.jsonl import read_jsonl, write_jsonl

LOGGER = logging.getLogger(__name__)


GENERATOR_REGISTRY: dict[str, type[SampleGenerator]] = {
    "semantic_edit": SemanticGenerator,
    "style_edit": StyleGenerator,
    "structural_edit": StructuralGenerator,
    "portrait_attribute": PortraitGenerator,
    "text_edit": TextGenerator,
}


def _load_decompose_manifest_map(index_path: Path) -> dict[str, dict[str, object]]:
    rows = read_jsonl(index_path)
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        manifest_path = Path(str(row["manifest_path"]))
        if manifest_path.exists():
            output[str(row["source_id"])] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return output


def run_generate(cfg: AppConfig) -> Path:
    root = Path(cfg.paths.root)
    filtered_meta_path = root / cfg.paths.data_dir / "filtered" / "filtered_meta.jsonl"
    decompose_index_path = root / cfg.paths.outputs_dir / "layers_cache" / "decompose_index.jsonl"
    generated_dir = root / cfg.paths.outputs_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    edit_backend = build_edit_backend(cfg.edit.backend, device=cfg.backend_runtime.device)
    context = GenerationContext(cfg=cfg, intermediate_dir=generated_dir, edit_backend=edit_backend)

    source_rows = [SourceMetadata.model_validate(row) for row in read_jsonl(filtered_meta_path)]
    decompose_map = _load_decompose_manifest_map(decompose_index_path)

    all_samples: list[SampleModel] = []
    for seed_index, source_meta in enumerate(source_rows):
        source_manifest = decompose_map.get(source_meta.source_id)
        if source_manifest is None:
            LOGGER.warning("generate_skip_missing_manifest source_id=%s", source_meta.source_id)
            continue

        for category_name, generator_cls in GENERATOR_REGISTRY.items():
            category_cfg = cfg.generate.categories.get(category_name)
            if category_cfg is not None and not category_cfg.enabled:
                continue
            generator = generator_cls(context)
            generated = generator.generate_for_source(
                source_meta, source_manifest, seed_index=seed_index
            )
            all_samples.extend(generated)

    selected = enforce_quotas(all_samples, cfg.quotas)
    output_path = generated_dir / "generated_samples.jsonl"
    write_jsonl(output_path, [sample.model_dump(mode="json") for sample in selected])
    LOGGER.info("generate_done total=%s selected=%s", len(all_samples), len(selected))
    return output_path
