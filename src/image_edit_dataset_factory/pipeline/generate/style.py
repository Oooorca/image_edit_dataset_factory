from __future__ import annotations

import numpy as np

from image_edit_dataset_factory.core.schema import SampleModel, SourceMetadata
from image_edit_dataset_factory.pipeline.generate.base import GenerationContext, SampleGenerator
from image_edit_dataset_factory.utils.image_io import read_image_rgb, write_image_rgb, write_mask


class StyleGenerator(SampleGenerator):
    category = "style_edit"

    def __init__(self, context: GenerationContext) -> None:
        super().__init__(context)
        self.subtypes = context.cfg.generate.categories.get(self.category)

    @staticmethod
    def _apply_contrast(image: np.ndarray, factor: float = 1.25) -> np.ndarray:
        centered = image.astype(np.float32) - 127.5
        out = centered * factor + 127.5
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_brightness(image: np.ndarray, delta: int = 20) -> np.ndarray:
        out = image.astype(np.int16) + delta
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_color_tone(image: np.ndarray) -> np.ndarray:
        out = image.astype(np.float32).copy()
        out[:, :, 0] *= 1.08
        out[:, :, 1] *= 1.00
        out[:, :, 2] *= 0.92
        return np.clip(out, 0, 255).astype(np.uint8)

    def _transform(self, subtype: str, image: np.ndarray) -> np.ndarray:
        if subtype == "contrast":
            return self._apply_contrast(image)
        if subtype == "brightness":
            return self._apply_brightness(image)
        if subtype == "color_tone":
            return self._apply_color_tone(image)
        return image

    def generate_for_source(
        self,
        source_meta: SourceMetadata,
        decompose_manifest: dict[str, object],
        seed_index: int,
    ) -> list[SampleModel]:
        image = read_image_rgb(source_meta.image_path)
        subtypes_all = (
            self.subtypes.subtypes if self.subtypes else ["contrast", "brightness", "color_tone"]
        )
        per_source = self.subtypes.per_source if self.subtypes else len(subtypes_all)
        if per_source <= 0:
            return []

        count = min(per_source, len(subtypes_all))
        offset = seed_index % len(subtypes_all)
        subtypes = [subtypes_all[(offset + i) % len(subtypes_all)] for i in range(count)]
        samples: list[SampleModel] = []

        for idx, subtype in enumerate(subtypes):
            src_dir = self.context.intermediate_dir / self.category / subtype / source_meta.scene
            src_dir.mkdir(parents=True, exist_ok=True)
            base_id = f"{source_meta.source_id}_{idx}_{seed_index}"

            src_path = src_dir / f"{base_id}.jpg"
            result_path = src_dir / f"{base_id}_result.jpg"
            allowed_mask_path = src_dir / f"{base_id}_allowed.png"

            write_image_rgb(src_path, image)
            transformed = self._transform(subtype, image)
            write_image_rgb(result_path, transformed)
            write_mask(allowed_mask_path, np.full(image.shape[:2], 255, dtype=np.uint8))

            instruction_en = f"Apply {subtype.replace('_', ' ')} style to the image."
            instruction_ch = f"请将图像进行{subtype}风格编辑。"
            samples.append(
                SampleModel(
                    sample_id=base_id,
                    category=self.category,
                    subtype=subtype,
                    scene=source_meta.scene,
                    src_image_path=str(src_path),
                    result_image_path=str(result_path),
                    instruction_ch=instruction_ch,
                    instruction_en=instruction_en,
                    metadata={
                        "allowed_region_mask_path": str(allowed_mask_path),
                        "source_id": source_meta.source_id,
                    },
                )
            )
        return samples
