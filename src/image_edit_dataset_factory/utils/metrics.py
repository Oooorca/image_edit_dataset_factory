from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def mse(image_a: np.ndarray, image_b: np.ndarray) -> float:
    diff = image_a.astype(np.float32) - image_b.astype(np.float32)
    return float(np.mean(np.square(diff)))


def pixel_diff_ratio(image_a: np.ndarray, image_b: np.ndarray, threshold: int = 2) -> float:
    diff = np.abs(image_a.astype(np.int16) - image_b.astype(np.int16))
    changed = np.any(diff > threshold, axis=-1)
    return float(changed.mean())


def ssim_rgb(image_a: np.ndarray, image_b: np.ndarray) -> float:
    if image_a.shape != image_b.shape:
        msg = "Images must have same shape"
        raise ValueError(msg)
    return float(
        structural_similarity(
            image_a,
            image_b,
            channel_axis=2,
            data_range=255,
        )
    )
