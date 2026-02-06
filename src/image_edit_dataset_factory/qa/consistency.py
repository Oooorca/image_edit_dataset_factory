from __future__ import annotations

from pathlib import Path

import numpy as np

from image_edit_dataset_factory.core.config import QAConfig
from image_edit_dataset_factory.core.schema import QAScore, SampleRecord
from image_edit_dataset_factory.utils.image_io import read_image_rgb, read_mask
from image_edit_dataset_factory.utils.mask_ops import dilate_mask, ensure_binary
from image_edit_dataset_factory.utils.metrics import ssim_rgb


def _allowed_mask(sample: SampleRecord, qa_cfg: QAConfig, shape: tuple[int, int]) -> np.ndarray:
    explicit = sample.metadata.get("allowed_region_mask_path")
    if isinstance(explicit, str) and Path(explicit).exists():
        return ensure_binary(read_mask(explicit))
    if sample.mask_paths:
        return dilate_mask(
            ensure_binary(read_mask(sample.mask_paths[0])), pixels=qa_cfg.allowed_region_dilation_px
        )
    return np.full(shape, 255, dtype=np.uint8)


def check_non_edit_region(sample: SampleRecord, qa_cfg: QAConfig) -> QAScore:
    src = read_image_rgb(sample.src_image_path)
    res = read_image_rgb(sample.result_image_path)

    if src.shape != res.shape:
        return QAScore(
            sample_id=sample.sample_id,
            passed=False,
            mse_outside_region=1e9,
            ssim_outside_region=0.0,
            changed_pixel_ratio_outside_region=1.0,
            details={"error": "shape_mismatch"},
        )

    allowed = _allowed_mask(sample, qa_cfg, src.shape[:2])
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

    src_out = src[outside]
    res_out = res[outside]
    mse_value = float(np.mean((src_out.astype(np.float32) - res_out.astype(np.float32)) ** 2))
    changed_ratio = float(
        np.mean(np.any(np.abs(src_out.astype(np.int16) - res_out.astype(np.int16)) > 2, axis=1))
    )

    src_masked = src.copy()
    res_masked = res.copy()
    src_masked[allowed > 0] = res_masked[allowed > 0]
    ssim_value = ssim_rgb(src_masked, res_masked)

    passed = (
        mse_value <= qa_cfg.max_mse_outside_region
        and ssim_value >= qa_cfg.min_ssim_outside_region
        and changed_ratio <= qa_cfg.max_changed_pixel_ratio_outside_region
    )

    return QAScore(
        sample_id=sample.sample_id,
        passed=passed,
        mse_outside_region=mse_value,
        ssim_outside_region=ssim_value,
        changed_pixel_ratio_outside_region=changed_ratio,
    )


def run_consistency(samples: list[SampleRecord], qa_cfg: QAConfig) -> list[QAScore]:
    return [check_non_edit_region(item, qa_cfg) for item in samples]
