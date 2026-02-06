from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image


def _cache_roots() -> list[Path]:
    roots: list[Path] = []
    env_cache = os.getenv("MODELSCOPE_CACHE")
    if env_cache:
        roots.append(Path(env_cache).expanduser())

    roots.extend(
        [
            Path("~/.cache/modelscope").expanduser(),
            Path("~/.cache/modelscope/hub").expanduser(),
            Path("~/modelscope").expanduser(),
        ]
    )

    dedup: list[Path] = []
    seen: set[str] = set()
    for item in roots:
        key = str(item)
        if key not in seen:
            seen.add(key)
            dedup.append(item)
    return dedup


def _candidate_dirs(model_id: str) -> list[Path]:
    org, name = model_id.split("/", 1)
    candidates: list[Path] = []

    for root in _cache_roots():
        candidates.extend(
            [
                root / org / name,
                root / "hub" / org / name,
                root / "models" / org / name,
                root / "hub" / "models" / org / name,
            ]
        )
    return candidates


def resolve_local_model_dir(model_ref: str | None, model_id: str) -> Path:
    """Resolve local model path without triggering downloads.

    `model_ref` may be an explicit directory or the model id itself.
    """

    if model_ref:
        explicit = Path(model_ref).expanduser()
        if explicit.exists() and explicit.is_dir():
            return explicit

    search_id = model_ref if model_ref and "/" in model_ref else model_id

    for candidate in _candidate_dirs(search_id):
        if candidate.exists() and candidate.is_dir():
            return candidate

    org, name = search_id.split("/", 1)
    for root in _cache_roots():
        if not root.exists():
            continue

        direct = list(root.glob(f"**/{org}/{name}"))
        for path in direct:
            if path.is_dir():
                return path

        tail = list(root.glob(f"**/{name}"))
        for path in tail:
            if path.is_dir():
                return path

    searched = [str(path) for path in _cache_roots()]
    raise RuntimeError(
        "Model directory not found locally. "
        f"model_ref={model_ref!r}, model_id={model_id!r}, searched_roots={searched}. "
        "Please run: `modelscope download --model "
        f"{model_id}` first."
    )


def to_pil(image_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(image_rgb.astype("uint8"), mode="RGB")


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)
