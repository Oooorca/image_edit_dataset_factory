from __future__ import annotations

import numpy as np

from image_edit_dataset_factory.core.schema import SampleModel, SourceMetadata
from image_edit_dataset_factory.pipeline.generate.base import SampleGenerator
from image_edit_dataset_factory.utils.image_io import (
    read_image_rgb,
    read_mask,
    write_image_rgb,
    write_mask,
)
from image_edit_dataset_factory.utils.mask_ops import (
    bbox_from_mask,
    dilate_mask,
    ensure_binary,
    mask_from_bbox,
)


class SemanticGenerator(SampleGenerator):
    category = "semantic_edit"

    def _load_or_build_mask(
        self, decompose_manifest: dict[str, object], image_shape: tuple[int, int]
    ) -> np.ndarray:
        layers = decompose_manifest.get("layers", [])
        if isinstance(layers, list) and layers:
            first = layers[0]
            if isinstance(first, dict) and "mask_path" in first:
                try:
                    mask = read_mask(str(first["mask_path"]))
                    mask = ensure_binary(mask)
                    coverage = float((mask > 0).mean())
                    if 0.01 <= coverage <= 0.7:
                        return mask
                except Exception:
                    pass

        h, w = image_shape
        box = (w // 4, h // 4, (3 * w) // 4, (3 * h) // 4)
        return mask_from_bbox((h, w), box)

    def generate_for_source(
        self,
        source_meta: SourceMetadata,
        decompose_manifest: dict[str, object],
        seed_index: int,
    ) -> list[SampleModel]:
        category_cfg = self.context.cfg.generate.categories.get(self.category)
        if category_cfg and category_cfg.per_source <= 0:
            return []

        image = read_image_rgb(source_meta.image_path)
        h, w = image.shape[:2]
        mask = self._load_or_build_mask(decompose_manifest, (h, w))
        bbox = bbox_from_mask(mask)
        if bbox is None:
            return []

        work_dir = self.context.intermediate_dir / self.category / "delete" / source_meta.scene
        work_dir.mkdir(parents=True, exist_ok=True)
        base_id = f"{source_meta.source_id}_delete_{seed_index}"
        src_path = work_dir / f"{base_id}.jpg"
        result_path = work_dir / f"{base_id}_result.jpg"
        mask_path = work_dir / f"{base_id}_mask.png"
        mask1_path = work_dir / f"{base_id}_mask-1.png"
        allowed_mask_path = work_dir / f"{base_id}_allowed.png"

        if not self.context.cfg.generate.dry_run:
            write_image_rgb(src_path, image)
            write_mask(mask_path, mask)
            write_mask(mask1_path, 255 - mask)
            allowed = dilate_mask(mask, pixels=self.context.cfg.qa.allowed_region_dilation_px)
            write_mask(allowed_mask_path, allowed)
            result = self.context.edit_backend.inpaint(image, mask, prompt="remove object")
            write_image_rgb(result_path, result)
        else:
            write_image_rgb(src_path, image)
            write_image_rgb(result_path, image)
            write_mask(mask_path, mask)
            write_mask(mask1_path, 255 - mask)
            write_mask(
                allowed_mask_path,
                dilate_mask(mask, pixels=self.context.cfg.qa.allowed_region_dilation_px),
            )

        return [
            SampleModel(
                sample_id=base_id,
                category=self.category,
                subtype="delete",
                scene=source_meta.scene,
                src_image_path=str(src_path),
                result_image_path=str(result_path),
                mask_paths=[str(mask_path), str(mask1_path)],
                instruction_ch="请删除图中目标对象并自然修复背景。",
                instruction_en="Delete the target object and naturally repair the background.",
                metadata={
                    "source_id": source_meta.source_id,
                    "bbox": bbox,
                    "allowed_region_mask_path": str(allowed_mask_path),
                },
            )
        ]
