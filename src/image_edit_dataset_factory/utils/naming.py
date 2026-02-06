from __future__ import annotations

import re
from pathlib import Path

ID_PATTERN = re.compile(r"^\d{5}$")


def format_sample_id(index: int) -> str:
    if index < 0:
        msg = "index must be >= 0"
        raise ValueError(msg)
    return f"{index:05d}"


def validate_sample_id(sample_id: str) -> bool:
    return bool(ID_PATTERN.match(sample_id))


def source_image_name(sample_id: str) -> str:
    return f"{sample_id}.jpg"


def result_image_name(sample_id: str) -> str:
    return f"{sample_id}_result.jpg"


def instruction_ch_name(sample_id: str) -> str:
    return f"{sample_id}_CH.txt"


def instruction_en_name(sample_id: str) -> str:
    return f"{sample_id}_EN.txt"


def mask_name(sample_id: str, index: int | None = None) -> str:
    if index is None:
        return f"{sample_id}_mask.png"
    return f"{sample_id}_mask-{index}.png"


def next_id_from_dataset_root(dataset_root: str | Path) -> int:
    root = Path(dataset_root)
    max_id = 0
    for path in root.rglob("*.jpg"):
        stem = path.stem
        if stem.endswith("_result"):
            stem = stem[: -len("_result")]
        if validate_sample_id(stem):
            max_id = max(max_id, int(stem))
    return max_id + 1
