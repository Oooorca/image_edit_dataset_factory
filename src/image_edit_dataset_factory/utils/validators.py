from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from image_edit_dataset_factory.core.config import FilterConfig
from image_edit_dataset_factory.utils.image_io import read_image_rgb


@dataclass
class ValidationResult:
    passed: bool
    reasons: list[str]


def check_resolution(image: np.ndarray, min_width: int, min_height: int) -> bool:
    h, w = image.shape[:2]
    return w >= min_width and h >= min_height


def is_grayscale(image: np.ndarray, tolerance: float = 2.0) -> bool:
    if image.ndim != 3 or image.shape[2] < 3:
        return True
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    mean_delta = float(
        (
            np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32)))
            + np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32)))
            + np.mean(np.abs(r.astype(np.float32) - b.astype(np.float32)))
        )
        / 3.0
    )
    return mean_delta < tolerance


def has_uniform_border(image: np.ndarray, border_px: int = 10, std_threshold: float = 4.0) -> bool:
    h, w = image.shape[:2]
    if h < border_px * 2 or w < border_px * 2:
        return False

    top = image[:border_px, :, :]
    bottom = image[-border_px:, :, :]
    left = image[:, :border_px, :]
    right = image[:, -border_px:, :]
    edge_pixels = np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    )
    return float(np.std(edge_pixels)) < std_threshold


def validate_image(path: str | Path, cfg: FilterConfig) -> ValidationResult:
    image = read_image_rgb(path)
    reasons: list[str] = []

    if not check_resolution(image, cfg.min_width, cfg.min_height):
        reasons.append("resolution_too_small")
    if cfg.reject_grayscale and is_grayscale(image):
        reasons.append("grayscale_detected")
    if cfg.reject_borders and has_uniform_border(image):
        reasons.append("uniform_border_detected")

    return ValidationResult(passed=not reasons, reasons=reasons)
