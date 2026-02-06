from __future__ import annotations

import numpy as np

from image_edit_dataset_factory.core.enums import EditTask
from image_edit_dataset_factory.core.schema import DecomposeRecord, SampleRecord, SourceSample
from image_edit_dataset_factory.pipeline.generate.base import BaseGenerator
from image_edit_dataset_factory.utils.image_io import (
    read_image_rgb,
    read_mask,
    write_image_rgb,
    write_mask,
)
from image_edit_dataset_factory.utils.mask_ops import dilate_mask, ensure_binary


class ConsistencyGenerator(BaseGenerator):
    edit_task = EditTask.CONSISTENCY.value

    def generate(self, source: SourceSample, decompose: DecomposeRecord) -> SampleRecord:
        image = read_image_rgb(source.image_path)
        mask = ensure_binary(read_mask(decompose.mask_path))

        out_dir = self.context.staging_dir / self.edit_task / source.source_id
        out_dir.mkdir(parents=True, exist_ok=True)
        src_path = out_dir / "source.jpg"
        result_path = out_dir / "result.jpg"
        mask_path = out_dir / "mask.png"
        allowed_path = out_dir / "allowed_mask.png"

        write_image_rgb(src_path, image)
        write_mask(mask_path, mask)
        write_mask(
            allowed_path, dilate_mask(mask, pixels=self.context.cfg.qa.allowed_region_dilation_px)
        )

        if self.context.cfg.generate.dry_run:
            edited = image.copy()
        else:
            edited = image.copy()
            region = mask > 0
            # Placeholder consistency-style edit: subtle color-temperature shift in masked region.
            temp = edited.astype(np.float32)
            temp[region, 0] *= 1.03
            temp[region, 2] *= 0.97
            edited = np.clip(temp, 0, 255).astype(np.uint8)

        write_image_rgb(result_path, edited)

        return SampleRecord(
            sample_id=source.source_id,
            dataset_category=source.dataset_category,
            edit_task=EditTask.CONSISTENCY,
            subtype=self.context.cfg.generate.subtypes.get(self.edit_task, "identity"),
            scene=source.scene,
            source_id=source.source_id,
            src_image_path=str(src_path),
            result_image_path=str(result_path),
            mask_paths=[str(mask_path)],
            instruction_ch="保持主体一致性并进行轻微一致性编辑",
            instruction_en="Preserve subject consistency with a mild consistency edit",
            metadata={"allowed_region_mask_path": str(allowed_path)},
        )
