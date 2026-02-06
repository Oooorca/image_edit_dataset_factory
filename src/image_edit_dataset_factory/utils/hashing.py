from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import imagehash  # type: ignore
except ImportError:  # pragma: no cover
    imagehash = None


def perceptual_hash(path: str | Path) -> str:
    img = Image.open(path).convert("L")
    if imagehash is not None:
        return str(imagehash.phash(img))

    resized = img.resize((8, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32)
    mean = float(arr.mean())
    bits = (arr > mean).astype(np.uint8).flatten()
    return "".join(str(x) for x in bits)


def hamming_distance(hash_a: str, hash_b: str) -> int:
    if len(hash_a) != len(hash_b):
        msg = "Hashes must have same length"
        raise ValueError(msg)
    return sum(ch1 != ch2 for ch1, ch2 in zip(hash_a, hash_b, strict=True))
