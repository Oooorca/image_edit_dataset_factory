from __future__ import annotations

from image_edit_dataset_factory.core.schema import SampleModel, SourceMetadata
from image_edit_dataset_factory.pipeline.generate.base import SampleGenerator


class TextGenerator(SampleGenerator):
    category = "text_edit"

    def generate_for_source(
        self,
        source_meta: SourceMetadata,
        decompose_manifest: dict[str, object],
        seed_index: int,
    ) -> list[SampleModel]:
        # Placeholder: OCR detection + text replacement/removal can be integrated later.
        return []
