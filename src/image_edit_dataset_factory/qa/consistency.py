from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from image_edit_dataset_factory.core.config import QAConfig
from image_edit_dataset_factory.core.schema import QAScore, SampleModel
from image_edit_dataset_factory.utils.image_io import read_image_rgb, read_mask
from image_edit_dataset_factory.utils.mask_ops import (
    dilate_mask,
    edge_error_px,
    ensure_binary,
    refine_mask,
)
from image_edit_dataset_factory.utils.metrics import ssim_rgb

LOGGER = logging.getLogger(__name__)


def _allowed_region_mask(
    sample: SampleModel, qa_cfg: QAConfig, shape: tuple[int, int]
) -> np.ndarray:
    meta_path = (
        sample.metadata.get("allowed_region_mask_path")
        if isinstance(sample.metadata, dict)
        else None
    )
    if isinstance(meta_path, str) and Path(meta_path).exists():
        return ensure_binary(read_mask(meta_path))

    if sample.mask_paths:
        base_mask = ensure_binary(read_mask(sample.mask_paths[0]))
        return dilate_mask(base_mask, pixels=qa_cfg.allowed_region_dilation_px)

    return np.full(shape, 255, dtype=np.uint8)


def check_non_edit_region(sample: SampleModel, qa_cfg: QAConfig) -> QAScore:
    src = read_image_rgb(sample.src_image_path)
    result = read_image_rgb(sample.result_image_path)
    if src.shape != result.shape:
        return QAScore(
            sample_id=sample.sample_id,
            passed=False,
            mse_outside_region=1e9,
            ssim_outside_region=0.0,
            changed_pixel_ratio_outside_region=1.0,
            details={"error": "shape_mismatch"},
        )

    allowed = _allowed_region_mask(sample, qa_cfg, src.shape[:2])
    outside = allowed == 0

    if np.count_nonzero(outside) == 0:
        return QAScore(
            sample_id=sample.sample_id,
            passed=True,
            mse_outside_region=0.0,
            ssim_outside_region=1.0,
            changed_pixel_ratio_outside_region=0.0,
            details={"outside_region_empty": True},
        )

    src_outside = src[outside]
    result_outside = result[outside]

    mse_value = float(
        np.mean((src_outside.astype(np.float32) - result_outside.astype(np.float32)) ** 2)
    )
    changed_ratio = float(
        np.mean(
            np.any(
                np.abs(src_outside.astype(np.int16) - result_outside.astype(np.int16)) > 2, axis=1
            )
        )
    )

    src_masked = src.copy()
    result_masked = result.copy()
    src_masked[allowed > 0] = result_masked[allowed > 0]
    ssim_value = ssim_rgb(src_masked, result_masked)

    passed = (
        mse_value <= qa_cfg.max_mse_outside_region
        and ssim_value >= qa_cfg.min_ssim_outside_region
        and changed_ratio <= qa_cfg.max_changed_pixel_ratio_outside_region
    )

    details: dict[str, float] = {}
    if str(sample.category) == "semantic_edit" and sample.mask_paths:
        semantic_mask = ensure_binary(read_mask(sample.mask_paths[0]))
        refined = refine_mask(semantic_mask, kernel_size=3, iterations=1)
        edge_err = edge_error_px(semantic_mask, refined)
        details["mask_edge_error_px"] = edge_err
        passed = passed and edge_err <= 5.0

    return QAScore(
        sample_id=sample.sample_id,
        passed=passed,
        mse_outside_region=mse_value,
        ssim_outside_region=ssim_value,
        changed_pixel_ratio_outside_region=changed_ratio,
        details=details,
    )


def run_consistency(samples: list[SampleModel], qa_cfg: QAConfig) -> list[QAScore]:
    scores = [check_non_edit_region(sample, qa_cfg) for sample in samples]
    failed = sum(1 for item in scores if not item.passed)
    LOGGER.info("qa_consistency_done total=%s failed=%s", len(scores), failed)
    return scores
