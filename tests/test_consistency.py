from pathlib import Path

import numpy as np

from image_edit_dataset_factory.core.config import QAConfig
from image_edit_dataset_factory.core.enums import EditTask
from image_edit_dataset_factory.core.schema import SampleRecord
from image_edit_dataset_factory.qa.consistency import check_non_edit_region
from image_edit_dataset_factory.utils.image_io import write_image_rgb, write_mask


def test_non_edit_region_check_passes_when_change_is_inside_mask(tmp_path: Path) -> None:
    src = np.zeros((64, 64, 3), dtype=np.uint8)
    result = src.copy()
    result[20:40, 20:40] = [255, 0, 0]

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255

    src_path = tmp_path / "src.jpg"
    result_path = tmp_path / "result.jpg"
    mask_path = tmp_path / "mask.png"
    write_image_rgb(src_path, src)
    write_image_rgb(result_path, result)
    write_mask(mask_path, mask)

    sample = SampleRecord(
        sample_id="00001",
        dataset_category="物体一致性",
        edit_task=EditTask.SEMANTIC,
        subtype="delete",
        scene="mixed",
        source_id="src_000001",
        src_image_path=str(src_path),
        result_image_path=str(result_path),
        mask_paths=[str(mask_path)],
        instruction_ch="",
        instruction_en="",
        metadata={},
    )

    score = check_non_edit_region(sample, QAConfig())
    assert score.passed
