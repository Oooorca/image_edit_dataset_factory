from __future__ import annotations

import csv
import json
from pathlib import Path

from image_edit_dataset_factory.core.schema import SampleModel
from image_edit_dataset_factory.utils.jsonl import write_jsonl


def write_sample_manifest(sample: SampleModel, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(sample.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")


def write_global_index(
    samples: list[SampleModel], csv_path: str | Path, jsonl_path: str | Path
) -> None:
    csv_target = Path(csv_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)

    rows = [sample.model_dump(mode="json") for sample in samples]
    write_jsonl(jsonl_path, rows)

    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "category",
                "subtype",
                "scene",
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
                    sample.category,
                    sample.subtype,
                    sample.scene,
                    sample.src_image_path,
                    sample.result_image_path,
                    "|".join(sample.mask_paths),
                    sample.instruction_ch,
                    sample.instruction_en,
                    json.dumps(sample.metadata, ensure_ascii=False),
                ]
            )
