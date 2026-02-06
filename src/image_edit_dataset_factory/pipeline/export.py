from __future__ import annotations

import logging
from pathlib import Path

from image_edit_dataset_factory.core.schema import SampleModel
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
    next_id_from_dataset_root,
    result_image_name,
    source_image_name,
)
from image_edit_dataset_factory.utils.text_ops import write_utf8_text

LOGGER = logging.getLogger(__name__)


def _category_dir(dataset_root: Path, sample: SampleModel) -> Path:
    return dataset_root / str(sample.category) / sample.subtype / sample.scene


def export_samples(samples: list[SampleModel], dataset_root: str | Path) -> list[SampleModel]:
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)

    next_index = next_id_from_dataset_root(root)
    exported: list[SampleModel] = []

    for sample in samples:
        sample_id = format_sample_id(next_index)
        next_index += 1

        out_dir = _category_dir(root, sample)
        out_dir.mkdir(parents=True, exist_ok=True)

        src_out = out_dir / source_image_name(sample_id)
        result_out = out_dir / result_image_name(sample_id)
        ch_out = out_dir / instruction_ch_name(sample_id)
        en_out = out_dir / instruction_en_name(sample_id)

        write_image_rgb(src_out, read_image_rgb(sample.src_image_path))
        write_image_rgb(result_out, read_image_rgb(sample.result_image_path))
        write_utf8_text(ch_out, sample.instruction_ch)
        write_utf8_text(en_out, sample.instruction_en)

        exported_mask_paths: list[str] = []
        if sample.mask_paths:
            primary = ensure_binary(read_mask(sample.mask_paths[0]))
            mask_out = out_dir / mask_name(sample_id)
            write_mask(mask_out, primary)
            exported_mask_paths.append(str(mask_out))

            if len(sample.mask_paths) >= 2:
                secondary = ensure_binary(read_mask(sample.mask_paths[1]))
            else:
                secondary = invert_mask(primary)
            mask1_out = out_dir / mask_name(sample_id, index=1)
            write_mask(mask1_out, secondary)
            exported_mask_paths.append(str(mask1_out))

        updated = sample.model_copy(
            update={
                "sample_id": sample_id,
                "src_image_path": str(src_out),
                "result_image_path": str(result_out),
                "mask_paths": exported_mask_paths,
            }
        )
        exported.append(updated)

    LOGGER.info("export_done count=%s dataset_root=%s", len(exported), root)
    return exported
