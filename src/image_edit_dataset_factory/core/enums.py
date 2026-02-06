from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Category(StrEnum):
    PORTRAIT_ATTRIBUTE = "portrait_attribute"
    SEMANTIC_EDIT = "semantic_edit"
    STYLE_EDIT = "style_edit"
    STRUCTURAL_EDIT = "structural_edit"
    TEXT_EDIT = "text_edit"


class Scene(StrEnum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    STUDIO = "studio"
    STREET = "street"
    NATURE = "nature"
    URBAN = "urban"
    MIXED = "mixed"


SEMANTIC_SUBTYPES = ("delete", "add", "replace", "attribute", "background")
STYLE_SUBTYPES = ("contrast", "brightness", "color_tone")
STRUCTURAL_SUBTYPES = ("move", "scale", "warp")
PORTRAIT_SUBTYPES = ("hair", "age", "makeup", "expression")
TEXT_SUBTYPES = ("replace_text", "remove_text")

ALL_SUBTYPES = {
    Category.PORTRAIT_ATTRIBUTE: PORTRAIT_SUBTYPES,
    Category.SEMANTIC_EDIT: SEMANTIC_SUBTYPES,
    Category.STYLE_EDIT: STYLE_SUBTYPES,
    Category.STRUCTURAL_EDIT: STRUCTURAL_SUBTYPES,
    Category.TEXT_EDIT: TEXT_SUBTYPES,
}
