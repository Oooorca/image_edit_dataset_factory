from __future__ import annotations

import logging
from pathlib import Path

from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.core.paths import resolve_paths
from image_edit_dataset_factory.core.schema import SourceSample
from image_edit_dataset_factory.utils.image_io import image_shape, is_image_file
from image_edit_dataset_factory.utils.jsonl import write_jsonl
from image_edit_dataset_factory.utils.validators import validate_image

LOGGER = logging.getLogger(__name__)


def _iter_images(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in root.glob(pattern) if path.is_file() and is_image_file(path))


def run_ingest(cfg: AppConfig) -> Path:
    paths = resolve_paths(cfg)
    paths.ensure_runtime_dirs()

    rows: list[dict[str, object]] = []
    source_counter = 0

    for category in cfg.ingest.include_categories:
        category_dir = paths.data_root / category
        if not category_dir.exists():
            LOGGER.warning("ingest_missing_category category=%s path=%s", category, category_dir)
            continue

        images = _iter_images(category_dir, recursive=cfg.ingest.recursive)
        if cfg.ingest.max_images_per_category > 0:
            images = images[: cfg.ingest.max_images_per_category]

        for image_path in images:
            if cfg.filter.enabled:
                result = validate_image(image_path, cfg.filter)
                if not result.passed:
                    LOGGER.info(
                        "ingest_filtered_out path=%s reasons=%s",
                        image_path,
                        ",".join(result.reasons),
                    )
                    continue

            source_counter += 1
            width, height = image_shape(image_path)
            source = SourceSample(
                source_id=f"src_{source_counter:06d}",
                dataset_category=category,
                image_path=str(image_path),
                width=width,
                height=height,
                scene="mixed",
                metadata={"relative_path": str(image_path.relative_to(paths.data_root))},
            )
            rows.append(source.model_dump(mode="json"))

    manifest_path = paths.manifests_dir / "source_manifest.jsonl"
    write_jsonl(manifest_path, rows)
    LOGGER.info("ingest_done count=%s manifest=%s", len(rows), manifest_path)
    return manifest_path
