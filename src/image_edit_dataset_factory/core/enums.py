from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class DatasetCategory(StrEnum):
    PERSON_OBJECT_CONSISTENCY = "人物物体一致性"
    OBJECT_CONSISTENCY = "物体一致性"
    PHYSICAL_CHANGE = "物理变化"


class EditTask(StrEnum):
    STRUCTURAL = "structural_edit"
    SEMANTIC = "semantic_edit"
    CONSISTENCY = "consistency_edit"


class Scene(StrEnum):
    MIXED = "mixed"


DEFAULT_CATEGORY_TO_TASK = {
    DatasetCategory.PERSON_OBJECT_CONSISTENCY.value: EditTask.CONSISTENCY.value,
    DatasetCategory.OBJECT_CONSISTENCY.value: EditTask.SEMANTIC.value,
    DatasetCategory.PHYSICAL_CHANGE.value: EditTask.STRUCTURAL.value,
}
