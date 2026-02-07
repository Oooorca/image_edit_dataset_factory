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
from image_edit_dataset_factory.utils.mask_ops import bbox_from_mask, dilate_mask, ensure_binary


class StructuralGenerator(BaseGenerator):
    edit_task = EditTask.STRUCTURAL.value

    def generate(self, source: SourceSample, decompose: DecomposeRecord) -> SampleRecord:
        image = read_image_rgb(source.image_path)
        mask = ensure_binary(read_mask(decompose.mask_path))
        bbox = bbox_from_mask(mask)

        out_dir = self.context.staging_dir / self.edit_task / source.source_id
        out_dir.mkdir(parents=True, exist_ok=True)
        src_path = out_dir / "source.jpg"
        result_path = out_dir / "result.jpg"
        mask_path = out_dir / "mask.png"
        allowed_path = out_dir / "allowed_mask.png"

        write_image_rgb(src_path, image)
        write_mask(mask_path, mask)

        allowed_base = mask.copy()

        if bbox is None or self.context.cfg.generate.dry_run:
            edited = image.copy()
        else:
            x0, y0, x1, y1 = bbox
            roi = image[y0 : y1 + 1, x0 : x1 + 1]
            roi_mask = mask[y0 : y1 + 1, x0 : x1 + 1]

            # Inpaint old location
            if hasattr(self.context.edit_backend, "inpaint_from_path"):
                base = self.context.edit_backend.inpaint_from_path(
                    image_path=src_path,
                    mask_path=mask_path,
                    prompt="repair hole",
                    sample_id=source.source_id,
                )
            else:
                base = self.context.edit_backend.inpaint(image, mask, prompt="repair hole")

            # Move region slightly to simulate structural edit
            dx = max(5, int(image.shape[1] * 0.06))
            dy = max(5, int(image.shape[0] * 0.03))
            nx0 = min(max(0, x0 + dx), image.shape[1] - 1)
            ny0 = min(max(0, y0 + dy), image.shape[0] - 1)
            nx1 = min(image.shape[1], nx0 + roi.shape[1])
            ny1 = min(image.shape[0], ny0 + roi.shape[0])

            edited = base.copy()
            paste_roi = roi[: ny1 - ny0, : nx1 - nx0]
            paste_mask = roi_mask[: ny1 - ny0, : nx1 - nx0] > 0
            view = edited[ny0:ny1, nx0:nx1]
            view[paste_mask] = paste_roi[paste_mask]

            moved_mask = np.zeros_like(mask)
            moved_view = moved_mask[ny0:ny1, nx0:nx1]
            moved_view[paste_mask] = 255
            allowed_base = np.maximum(mask, moved_mask)

        write_image_rgb(result_path, edited)
        write_mask(
            allowed_path,
            dilate_mask(
                allowed_base,
                pixels=self.context.cfg.qa.allowed_region_dilation_px,
            ),
        )

        return SampleRecord(
            sample_id=source.source_id,
            dataset_category=source.dataset_category,
            edit_task=EditTask.STRUCTURAL,
            subtype=self.context.cfg.generate.subtypes.get(self.edit_task, "move"),
            scene=source.scene,
            source_id=source.source_id,
            src_image_path=str(src_path),
            result_image_path=str(result_path),
            mask_paths=[str(mask_path)],
            instruction_ch="调整目标结构位置并修复背景",
            instruction_en="Move or scale the target structure and repair background",
            metadata={"allowed_region_mask_path": str(allowed_path)},
        )
