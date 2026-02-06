from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path

from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.core.paths import resolve_paths
from image_edit_dataset_factory.core.schema import SampleRecord
from image_edit_dataset_factory.utils.image_io import (
    read_image_rgb,
    read_mask,
    write_image_rgb,
    write_mask,
)
from image_edit_dataset_factory.utils.mask_ops import ensure_binary, invert_mask
from image_edit_dataset_factory.utils.naming import (
    format_sample_id,
    instruction_ch_name,
    instruction_en_name,
    mask_name,
    result_image_name,
    source_image_name,
)
from image_edit_dataset_factory.utils.text_ops import write_utf8_text

LOGGER = logging.getLogger(__name__)


def _write_index(samples: list[SampleRecord], reports_dir: Path) -> tuple[Path, Path]:
    csv_path = reports_dir / "index.csv"
    jsonl_path = reports_dir / "index.jsonl"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "dataset_category",
                "edit_task",
                "subtype",
                "scene",
                "source_id",
                "src_image_path",
                "result_image_path",
                "mask_paths",
                "instruction_ch",
                "instruction_en",
                "metadata",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.sample_id,
                    sample.dataset_category,
                    sample.edit_task,
                    sample.subtype,
                    sample.scene,
                    sample.source_id,
                    sample.src_image_path,
                    sample.result_image_path,
                    "|".join(sample.mask_paths),
                    sample.instruction_ch,
                    sample.instruction_en,
                    json.dumps(sample.metadata, ensure_ascii=False),
                ]
            )

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")

    return csv_path, jsonl_path


def run_export(cfg: AppConfig) -> Path:
    paths = resolve_paths(cfg)
    paths.ensure_runtime_dirs()

    generated_manifest = paths.manifests_dir / "generated_manifest.jsonl"
    rows = [
        json.loads(line)
        for line in generated_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generated = [SampleRecord.model_validate(item) for item in rows]

    if not cfg.pipeline.resume and paths.dataset_dir.exists():
        shutil.rmtree(paths.dataset_dir)
        paths.dataset_dir.mkdir(parents=True, exist_ok=True)

    exported: list[SampleRecord] = []
    next_idx = 1
    for sample in generated:
        sid = format_sample_id(next_idx)
        next_idx += 1

        scene_dir = paths.dataset_dir / sample.edit_task.value / sample.subtype / sample.scene
        scene_dir.mkdir(parents=True, exist_ok=True)

        src_out = scene_dir / source_image_name(sid)
        result_out = scene_dir / result_image_name(sid)
        ch_out = scene_dir / instruction_ch_name(sid)
        en_out = scene_dir / instruction_en_name(sid)

        write_image_rgb(src_out, read_image_rgb(sample.src_image_path))
        write_image_rgb(result_out, read_image_rgb(sample.result_image_path))
        write_utf8_text(ch_out, sample.instruction_ch)
        write_utf8_text(en_out, sample.instruction_en)

        out_masks: list[str] = []
        if sample.mask_paths:
            mask0 = ensure_binary(read_mask(sample.mask_paths[0]))
            mask0_out = scene_dir / mask_name(sid)
            write_mask(mask0_out, mask0)
            out_masks.append(str(mask0_out))

            mask1_out = scene_dir / mask_name(sid, index=1)
            if len(sample.mask_paths) > 1:
                mask1 = ensure_binary(read_mask(sample.mask_paths[1]))
            else:
                mask1 = invert_mask(mask0)
            write_mask(mask1_out, mask1)
            out_masks.append(str(mask1_out))

        exported.append(
            sample.model_copy(
                update={
                    "sample_id": sid,
                    "src_image_path": str(src_out),
                    "result_image_path": str(result_out),
                    "mask_paths": out_masks,
                }
            )
        )

    _, index_jsonl = _write_index(exported, paths.reports_dir)
    LOGGER.info("export_done count=%s index=%s", len(exported), index_jsonl)
    return index_jsonl
