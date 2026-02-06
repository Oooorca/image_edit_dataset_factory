from __future__ import annotations

from image_edit_dataset_factory.core.schema import SampleModel, SourceMetadata
from image_edit_dataset_factory.pipeline.generate.base import SampleGenerator


class PortraitGenerator(SampleGenerator):
    category = "portrait_attribute"

    def generate_for_source(
        self,
        source_meta: SourceMetadata,
        decompose_manifest: dict[str, object],
        seed_index: int,
    ) -> list[SampleModel]:
        # Placeholder: requires dedicated portrait editing backends.
        return []
