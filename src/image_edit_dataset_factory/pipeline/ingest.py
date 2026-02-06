from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path

from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.core.schema import SourceMetadata
from image_edit_dataset_factory.utils.image_io import image_shape, is_image_file
from image_edit_dataset_factory.utils.jsonl import write_jsonl

LOGGER = logging.getLogger(__name__)


def _load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if manifest_path.suffix.lower() == ".jsonl":
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if manifest_path.suffix.lower() == ".csv":
        with manifest_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows.extend(reader)
        return rows

    msg = f"Unsupported manifest format: {manifest_path}"
    raise ValueError(msg)


def _iter_source_files(cfg: AppConfig, root: Path) -> list[Path]:
    if cfg.ingest.manifest_path:
        manifest_rows = _load_manifest(Path(cfg.ingest.manifest_path))
        return [Path(item["image_path"]) for item in manifest_rows if "image_path" in item]

    pattern = "**/*" if cfg.ingest.recursive else "*"
    return sorted([p for p in root.glob(pattern) if p.is_file() and is_image_file(p)])


def run_ingest(cfg: AppConfig) -> Path:
    root = Path(cfg.paths.root)
    raw_images_dir = root / cfg.paths.data_dir / "raw" / "images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    source_root = Path(cfg.ingest.source_dir)
    source_files = _iter_source_files(cfg, source_root)
    LOGGER.info("ingest_found_sources count=%s", len(source_files))

    records: list[dict[str, object]] = []
    for idx, src in enumerate(source_files, start=1):
        source_id = f"src_{idx:06d}"
        ext = src.suffix.lower() if src.suffix else ".jpg"
        dest = raw_images_dir / f"{source_id}{ext}"

        if not dest.exists():
            if cfg.ingest.symlink:
                try:
                    dest.symlink_to(src.resolve())
                except FileExistsError:
                    pass
                except OSError:
                    shutil.copy2(src, dest)
            else:
                shutil.copy2(src, dest)

        width, height = image_shape(dest)
        source_meta = SourceMetadata(
            source_id=source_id,
            image_path=str(dest),
            width=width,
            height=height,
            scene="mixed",
        )
        records.append(source_meta.model_dump(mode="json"))

    source_meta_path = raw_images_dir.parent / "source_meta.jsonl"
    write_jsonl(source_meta_path, records)
    LOGGER.info("ingest_written_metadata path=%s count=%s", source_meta_path, len(records))
    return source_meta_path
