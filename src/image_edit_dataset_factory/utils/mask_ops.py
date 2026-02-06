from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np


def alpha_to_mask(alpha: np.ndarray, threshold: int = 1) -> np.ndarray:
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    mask = (alpha >= threshold).astype(np.uint8) * 255
    return mask


def ensure_binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 127).astype(np.uint8) * 255


def refine_mask(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    binary = ensure_binary(mask)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


def dilate_mask(mask: np.ndarray, pixels: int = 5) -> np.ndarray:
    if pixels <= 0:
        return ensure_binary(mask)
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
    return cv2.dilate(ensure_binary(mask), kernel, iterations=1)


def erode_mask(mask: np.ndarray, pixels: int = 3) -> np.ndarray:
    if pixels <= 0:
        return ensure_binary(mask)
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
    return cv2.erode(ensure_binary(mask), kernel, iterations=1)


def invert_mask(mask: np.ndarray) -> np.ndarray:
    return 255 - ensure_binary(mask)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = np.where(mask > 0)
    if points[0].size == 0:
        return None
    y_min, y_max = int(points[0].min()), int(points[0].max())
    x_min, x_max = int(points[1].min()), int(points[1].max())
    return x_min, y_min, x_max, y_max


def mask_from_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y_min : y_max + 1, x_min : x_max + 1] = 255
    return mask


def combine_masks(masks: Iterable[np.ndarray]) -> np.ndarray:
    masks = list(masks)
    if not masks:
        msg = "At least one mask is required"
        raise ValueError(msg)
    output = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        output = np.maximum(output, ensure_binary(mask))
    return output


def edge_error_px(pred_mask: np.ndarray, ref_mask: np.ndarray) -> float:
    pred = ensure_binary(pred_mask)
    ref = ensure_binary(ref_mask)
    pred_edges = cv2.Canny(pred, 100, 200)
    ref_edges = cv2.Canny(ref, 100, 200)
    if pred_edges.sum() == 0 and ref_edges.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(255 - ref_edges, cv2.DIST_L2, 3)
    values = dist[pred_edges > 0]
    if values.size == 0:
        return float(max(pred.shape))
    return float(np.mean(values))
