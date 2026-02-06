from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from image_edit_dataset_factory.backends.factory import build_layered_backend
from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.core.paths import resolve_paths
from image_edit_dataset_factory.core.schema import DecomposeRecord, SourceSample
from image_edit_dataset_factory.utils.image_io import read_image_rgb, write_image_rgb, write_mask
from image_edit_dataset_factory.utils.jsonl import read_jsonl, write_jsonl
from image_edit_dataset_factory.utils.mask_ops import alpha_to_mask, mask_from_bbox, refine_mask

LOGGER = logging.getLogger(__name__)


def _select_primary_mask(image_rgb: np.ndarray, alphas: list[np.ndarray]) -> np.ndarray:
    if alphas:
        candidates: list[tuple[float, np.ndarray]] = []
        total = image_rgb.shape[0] * image_rgb.shape[1]
        for alpha in alphas:
            mask = refine_mask(alpha_to_mask(alpha), kernel_size=3, iterations=1)
            ratio = float((mask > 0).sum() / total)
            candidates.append((ratio, mask))

        candidates = [item for item in candidates if 0.01 <= item[0] <= 0.9]
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

    h, w = image_rgb.shape[:2]
    return mask_from_bbox((h, w), (w // 4, h // 4, (3 * w) // 4, (3 * h) // 4))


def run_decompose(cfg: AppConfig) -> Path:
    paths = resolve_paths(cfg)
    paths.ensure_runtime_dirs()

    source_manifest = paths.manifests_dir / "source_manifest.jsonl"
    rows = [SourceSample.model_validate(item) for item in read_jsonl(source_manifest)]

    backend = build_layered_backend(cfg)
    out_dir = paths.cache_dir / "decompose"
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for source in rows:
        image = read_image_rgb(source.image_path)
        layers = backend.decompose(image)

        source_dir = out_dir / source.source_id
        source_dir.mkdir(parents=True, exist_ok=True)

        alpha_list: list[np.ndarray] = []
        layer_paths: list[str] = []
        for idx, layer in enumerate(layers):
            rgba_path = source_dir / f"layer_{idx:02d}.png"
            alpha_path = source_dir / f"layer_{idx:02d}_alpha.png"
            write_image_rgb(rgba_path, layer.rgba[:, :, :3])
            write_mask(alpha_path, layer.alpha)
            alpha_list.append(layer.alpha)
            layer_paths.append(str(rgba_path))

        mask = _select_primary_mask(image, alpha_list)
        mask_path = source_dir / "primary_mask.png"
        write_mask(mask_path, mask)

        record = DecomposeRecord(
            source_id=source.source_id,
            image_path=source.image_path,
            mask_path=str(mask_path),
            layer_paths=layer_paths,
            metadata={"dataset_category": source.dataset_category},
        )
        records.append(record.model_dump(mode="json"))

    manifest_path = paths.manifests_dir / "decompose_manifest.jsonl"
    write_jsonl(manifest_path, records)
    LOGGER.info("decompose_done count=%s manifest=%s", len(records), manifest_path)
    return manifest_path
