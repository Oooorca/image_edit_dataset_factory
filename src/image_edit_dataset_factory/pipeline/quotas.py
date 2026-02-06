from __future__ import annotations

from collections import defaultdict

from image_edit_dataset_factory.core.config import QuotasConfig
from image_edit_dataset_factory.core.schema import SampleModel


def enforce_quotas(samples: list[SampleModel], quotas: QuotasConfig) -> list[SampleModel]:
    selected: list[SampleModel] = []
    per_category = defaultdict(int)
    per_scene = defaultdict(int)

    for sample in samples:
        category_limit = quotas.per_category.get(str(sample.category), quotas.target_total)
        scene_limit = quotas.per_scene.get(sample.scene, quotas.target_total)

        if per_category[str(sample.category)] >= category_limit:
            continue
        if per_scene[sample.scene] >= scene_limit:
            continue
        if len(selected) >= quotas.target_total:
            break

        selected.append(sample)
        per_category[str(sample.category)] += 1
        per_scene[sample.scene] += 1

    return selected
