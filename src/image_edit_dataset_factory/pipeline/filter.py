from __future__ import annotations

import logging
import shutil
from pathlib import Path

from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.core.schema import SourceMetadata
from image_edit_dataset_factory.utils.jsonl import read_jsonl, write_jsonl
from image_edit_dataset_factory.utils.validators import validate_image

LOGGER = logging.getLogger(__name__)


def run_filter(cfg: AppConfig) -> Path:
    root = Path(cfg.paths.root)
    input_meta = root / cfg.paths.data_dir / "raw" / "source_meta.jsonl"
    filtered_dir = root / cfg.paths.data_dir / "filtered" / "images"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_meta)
    kept: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []

    for row in rows:
        meta = SourceMetadata.model_validate(row)
        src = Path(meta.image_path)
        result = validate_image(src, cfg.filter)
        if result.passed:
            dest = filtered_dir / src.name
            if not dest.exists():
                if src.is_symlink():
                    try:
                        dest.symlink_to(src.resolve())
                    except OSError:
                        shutil.copy2(src, dest)
                else:
                    shutil.copy2(src, dest)
            updated = meta.model_copy(update={"image_path": str(dest)})
            kept.append(updated.model_dump(mode="json"))
        else:
            rejected.append(
                {
                    "source_id": meta.source_id,
                    "image_path": meta.image_path,
                    "reasons": result.reasons,
                }
            )

    kept_path = filtered_dir.parent / "filtered_meta.jsonl"
    rejected_path = filtered_dir.parent / "rejected_meta.jsonl"
    write_jsonl(kept_path, kept)
    write_jsonl(rejected_path, rejected)
    LOGGER.info("filter_done kept=%s rejected=%s", len(kept), len(rejected))
    return kept_path
