from pathlib import Path

import numpy as np
from PIL import Image

from image_edit_dataset_factory.qa.linter import lint_dataset


def _write_rgb(path: Path) -> None:
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :, 0] = 100
    Image.fromarray(arr).save(path)


def _write_mask(path: Path) -> None:
    arr = np.zeros((32, 32), dtype=np.uint8)
    arr[10:20, 10:20] = 255
    Image.fromarray(arr).save(path)


def test_linter_catches_missing_required_files(tmp_path: Path) -> None:
    scene = tmp_path / "semantic_edit" / "delete" / "mixed"
    scene.mkdir(parents=True)

    _write_rgb(scene / "00001.jpg")
    _write_rgb(scene / "00001_result.jpg")
    (scene / "00001_EN.txt").write_text("remove object", encoding="utf-8")
    _write_mask(scene / "00001_mask.png")

    issues = lint_dataset(tmp_path)
    codes = {issue.code for issue in issues}
    assert "missing_required" in codes
    assert "semantic_mask_required" in codes
