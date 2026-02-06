from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from image_edit_dataset_factory.backends.factory import build_layered_backend
from image_edit_dataset_factory.core.config import AppConfig
from image_edit_dataset_factory.core.schema import SourceMetadata
from image_edit_dataset_factory.utils.image_io import read_image_rgb, write_image_rgb, write_mask
from image_edit_dataset_factory.utils.jsonl import read_jsonl, write_jsonl
from image_edit_dataset_factory.utils.mask_ops import alpha_to_mask, refine_mask

LOGGER = logging.getLogger(__name__)


def _save_rgba_as_rgb(path: Path, rgba: np.ndarray) -> None:
    rgb = rgba[:, :, :3]
    write_image_rgb(path, rgb)


def run_decompose(cfg: AppConfig) -> Path:
    root = Path(cfg.paths.root)
    filtered_meta_path = root / cfg.paths.data_dir / "filtered" / "filtered_meta.jsonl"
    cache_dir = root / cfg.paths.outputs_dir / "layers_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    backend = build_layered_backend(cfg.decompose.backend, device=cfg.backend_runtime.device)
    rows = read_jsonl(filtered_meta_path)
    manifests: list[dict[str, object]] = []

    for row in rows:
        meta = SourceMetadata.model_validate(row)
        sample_dir = cache_dir / meta.source_id
        manifest_path = sample_dir / "manifest.json"
        if manifest_path.exists() and not cfg.decompose.overwrite:
            manifests.append({"source_id": meta.source_id, "manifest_path": str(manifest_path)})
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)
        image = read_image_rgb(meta.image_path)
        layers = backend.decompose(image)

        layer_entries: list[dict[str, str | int]] = []
        for idx, layer in enumerate(layers, start=1):
            layer_path = sample_dir / f"layer-{idx}.png"
            alpha_path = sample_dir / f"layer-{idx}_alpha.png"
            mask_path = sample_dir / f"layer-{idx}_mask.png"
            _save_rgba_as_rgb(layer_path, layer.rgba)
            write_mask(alpha_path, layer.alpha)
            refined = refine_mask(alpha_to_mask(layer.alpha), kernel_size=3, iterations=1)
            write_mask(mask_path, refined)
            layer_entries.append(
                {
                    "layer_id": layer.layer_id,
                    "layer_path": str(layer_path),
                    "alpha_path": str(alpha_path),
                    "mask_path": str(mask_path),
                }
            )

        payload = {
            "source_id": meta.source_id,
            "source_image_path": meta.image_path,
            "layers": layer_entries,
        }
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifests.append({"source_id": meta.source_id, "manifest_path": str(manifest_path)})

    output_manifest = cache_dir / "decompose_index.jsonl"
    write_jsonl(output_manifest, manifests)
    LOGGER.info("decompose_done count=%s", len(manifests))
    return output_manifest
