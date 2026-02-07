from __future__ import annotations

from image_edit_dataset_factory.core.enums import EditTask
from image_edit_dataset_factory.core.schema import DecomposeRecord, SampleRecord, SourceSample
from image_edit_dataset_factory.pipeline.generate.base import BaseGenerator
from image_edit_dataset_factory.utils.image_io import (
    read_image_rgb,
    read_mask,
    write_image_rgb,
    write_mask,
)
from image_edit_dataset_factory.utils.mask_ops import dilate_mask, ensure_binary, invert_mask


class SemanticGenerator(BaseGenerator):
    edit_task = EditTask.SEMANTIC.value

    def generate(self, source: SourceSample, decompose: DecomposeRecord) -> SampleRecord:
        image = read_image_rgb(source.image_path)
        mask = ensure_binary(read_mask(decompose.mask_path))

        out_dir = self.context.staging_dir / self.edit_task / source.source_id
        out_dir.mkdir(parents=True, exist_ok=True)

        src_path = out_dir / "source.jpg"
        result_path = out_dir / "result.jpg"
        mask_path = out_dir / "mask.png"
        mask1_path = out_dir / "mask-1.png"
        allowed_path = out_dir / "allowed_mask.png"

        write_image_rgb(src_path, image)
        write_mask(mask_path, mask)
        write_mask(mask1_path, invert_mask(mask))
        write_mask(
            allowed_path, dilate_mask(mask, pixels=self.context.cfg.qa.allowed_region_dilation_px)
        )

        if self.context.cfg.generate.dry_run:
            edited = image.copy()
        else:
            if hasattr(self.context.edit_backend, "inpaint_from_path"):
                edited = self.context.edit_backend.inpaint_from_path(
                    image_path=src_path,
                    mask_path=mask_path,
                    prompt="delete object",
                    sample_id=source.source_id,
                )
            else:
                edited = self.context.edit_backend.inpaint(image, mask, prompt="delete object")
        write_image_rgb(result_path, edited)

        return SampleRecord(
            sample_id=source.source_id,
            dataset_category=source.dataset_category,
            edit_task=EditTask.SEMANTIC,
            subtype=self.context.cfg.generate.subtypes.get(self.edit_task, "delete"),
            scene=source.scene,
            source_id=source.source_id,
            src_image_path=str(src_path),
            result_image_path=str(result_path),
            mask_paths=[str(mask_path), str(mask1_path)],
            instruction_ch="删除目标并修复背景",
            instruction_en="Delete the target object and repair background",
            metadata={"allowed_region_mask_path": str(allowed_path)},
        )
