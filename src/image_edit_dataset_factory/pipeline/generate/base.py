from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from image_edit_dataset_factory.backends.edit_base import EditorBackend
from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.core.schema import DecomposeRecord, SampleRecord, SourceSample


@dataclass
class GenerationContext:
    cfg: AppConfig
    staging_dir: Path
    edit_backend: EditorBackend


class BaseGenerator(ABC):
    edit_task: str

    def __init__(self, context: GenerationContext) -> None:
        self.context = context

    @abstractmethod
    def generate(
        self,
        source: SourceSample,
        decompose: DecomposeRecord,
    ) -> SampleRecord: ...
