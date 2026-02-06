from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from image_edit_dataset_factory.backends.edit_base import EditorBackend
from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.core.schema import SampleModel, SourceMetadata


@dataclass
class GenerationContext:
    cfg: AppConfig
    intermediate_dir: Path
    edit_backend: EditorBackend


class SampleGenerator(ABC):
    category: str

    def __init__(self, context: GenerationContext) -> None:
        self.context = context

    @abstractmethod
    def generate_for_source(
        self,
        source_meta: SourceMetadata,
        decompose_manifest: dict[str, object],
        seed_index: int,
    ) -> list[SampleModel]: ...
