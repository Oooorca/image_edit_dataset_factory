from __future__ import annotations

import cv2
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


class StructuralGenerator(SampleGenerator):
    category = "structural_edit"

    def _base_mask(
        self, decompose_manifest: dict[str, object], shape: tuple[int, int]
    ) -> np.ndarray:
        layers = decompose_manifest.get("layers", [])
        if isinstance(layers, list) and layers:
            first = layers[0]
            if isinstance(first, dict) and "mask_path" in first:
                try:
                    mask = ensure_binary(read_mask(str(first["mask_path"])))
                    cov = float((mask > 0).mean())
                    if 0.01 <= cov <= 0.5:
                        return mask
                except Exception:
                    pass
        h, w = shape
        return mask_from_bbox((h, w), (w // 3, h // 3, (2 * w) // 3, (2 * h) // 3))

    @staticmethod
    def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
        h, w = mask.shape
        mat = np.float32([[1, 0, dx], [0, 1, dy]])
        moved = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        return ensure_binary(moved)

    @staticmethod
    def _blend_object(
        base_image: np.ndarray,
        obj_image: np.ndarray,
        obj_mask: np.ndarray,
    ) -> np.ndarray:
        out = base_image.copy()
        selector = obj_mask > 0
        out[selector] = obj_image[selector]
        return out

    def _generate_variant(
        self,
        subtype: str,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        bbox = bbox_from_mask(mask)
        if bbox is None:
            return (
                image.copy(),
                mask,
                dilate_mask(mask, pixels=self.context.cfg.qa.allowed_region_dilation_px),
            )

        x0, y0, x1, y1 = bbox
        obj = np.zeros_like(image)
        obj[mask > 0] = image[mask > 0]

        if subtype == "move":
            dx = max(5, int(w * 0.08))
            dy = max(5, int(h * 0.03))
            moved_obj = cv2.warpAffine(
                obj,
                np.float32([[1, 0, dx], [0, 1, dy]]),
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderValue=(0, 0, 0),
            )
            moved_mask = self._shift_mask(mask, dx, dy)
        elif subtype == "scale":
            roi = image[y0 : y1 + 1, x0 : x1 + 1]
            roi_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
            scale = 1.15
            new_w = max(1, int(roi.shape[1] * scale))
            new_h = max(1, int(roi.shape[0] * scale))
            roi_scaled = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_scaled = cv2.resize(roi_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            moved_obj = np.zeros_like(image)
            moved_mask = np.zeros(mask.shape, dtype=np.uint8)
            nx0 = max(0, x0 - (new_w - roi.shape[1]) // 2)
            ny0 = max(0, y0 - (new_h - roi.shape[0]) // 2)
            nx1 = min(w, nx0 + new_w)
            ny1 = min(h, ny0 + new_h)
            src_w = nx1 - nx0
            src_h = ny1 - ny0
            moved_obj[ny0:ny1, nx0:nx1] = roi_scaled[:src_h, :src_w]
            moved_mask[ny0:ny1, nx0:nx1] = mask_scaled[:src_h, :src_w]
            moved_mask = ensure_binary(moved_mask)
        else:  # warp fallback
            moved_obj = cv2.GaussianBlur(obj, (0, 0), sigmaX=1.2)
            moved_mask = mask

        hole_mask = ensure_binary(mask)
        repaired = self.context.edit_backend.inpaint(image, hole_mask, prompt="repair background")
        result = self._blend_object(repaired, moved_obj, moved_mask)
        allowed = dilate_mask(
            np.maximum(mask, moved_mask), pixels=self.context.cfg.qa.allowed_region_dilation_px
        )
        return result, hole_mask, allowed

    def generate_for_source(
        self,
        source_meta: SourceMetadata,
        decompose_manifest: dict[str, object],
        seed_index: int,
    ) -> list[SampleModel]:
        image = read_image_rgb(source_meta.image_path)
        mask = self._base_mask(decompose_manifest, image.shape[:2])
        samples: list[SampleModel] = []

        category_cfg = self.context.cfg.generate.categories.get(self.category)
        subtypes_all = (
            category_cfg.subtypes if category_cfg and category_cfg.subtypes else ["move", "scale"]
        )
        per_source = category_cfg.per_source if category_cfg else len(subtypes_all)
        if per_source <= 0:
            return []

        count = min(per_source, len(subtypes_all))
        offset = seed_index % len(subtypes_all)
        subtypes = [subtypes_all[(offset + i) % len(subtypes_all)] for i in range(count)]

        for idx, subtype in enumerate(subtypes):
            work_dir = self.context.intermediate_dir / self.category / subtype / source_meta.scene
            work_dir.mkdir(parents=True, exist_ok=True)
            base_id = f"{source_meta.source_id}_{subtype}_{seed_index}_{idx}"

            src_path = work_dir / f"{base_id}.jpg"
            result_path = work_dir / f"{base_id}_result.jpg"
            mask_path = work_dir / f"{base_id}_mask.png"
            allowed_path = work_dir / f"{base_id}_allowed.png"

            write_image_rgb(src_path, image)
            result, mask_used, allowed = self._generate_variant(subtype, image, mask)
            write_image_rgb(result_path, result)
            write_mask(mask_path, mask_used)
            write_mask(allowed_path, allowed)

            samples.append(
                SampleModel(
                    sample_id=base_id,
                    category=self.category,
                    subtype=subtype,
                    scene=source_meta.scene,
                    src_image_path=str(src_path),
                    result_image_path=str(result_path),
                    mask_paths=[str(mask_path)],
                    instruction_ch=f"请对目标进行{subtype}结构调整并补全背景。",
                    instruction_en=(
                        f"Apply a {subtype} structural change to the target "
                        "and repair the background."
                    ),
                    metadata={
                        "source_id": source_meta.source_id,
                        "allowed_region_mask_path": str(allowed_path),
                    },
                )
            )

        return samples
