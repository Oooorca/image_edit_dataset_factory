import numpy as np

from image_edit_dataset_factory.utils.mask_ops import (
    alpha_to_mask,
    bbox_from_mask,
    dilate_mask,
    edge_error_px,
    mask_from_bbox,
    refine_mask,
)


def test_alpha_to_mask_and_bbox() -> None:
    alpha = np.zeros((10, 10), dtype=np.uint8)
    alpha[2:7, 3:8] = 200
    mask = alpha_to_mask(alpha)
    assert mask.max() == 255
    assert bbox_from_mask(mask) == (3, 2, 7, 6)


def test_refine_and_dilate() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 255
    refined = refine_mask(mask, kernel_size=3)
    dilated = dilate_mask(refined, pixels=2)
    assert dilated.sum() >= refined.sum()


def test_edge_error() -> None:
    m1 = mask_from_bbox((32, 32), (8, 8, 20, 20))
    m2 = mask_from_bbox((32, 32), (9, 8, 21, 20))
    err = edge_error_px(m1, m2)
    assert err <= 2.0
