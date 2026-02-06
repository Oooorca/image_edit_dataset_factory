#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def create_demo_image(size: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.zeros((size, size, 3), dtype=np.uint8)
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)

    base[:, :, 0] = np.clip(80 + 100 * xx + rng.integers(0, 20), 0, 255)
    base[:, :, 1] = np.clip(60 + 120 * yy + rng.integers(0, 20), 0, 255)
    base[:, :, 2] = np.clip(120 + 90 * (1 - xx) + rng.integers(0, 20), 0, 255)

    img = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(img)
    pad = size // 8
    draw.ellipse((pad, pad, size // 2, size // 2), fill=(220, 80, 90))
    draw.rectangle((size // 2, size // 3, size - pad, size - pad), fill=(50, 180, 110))
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Create synthetic demo images")
    parser.add_argument("--out", default="data/demo_input")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        img = create_demo_image(args.size, seed=i)
        img.save(out_dir / f"demo_{i+1:03d}.jpg", quality=95)

    print(f"wrote {args.count} demo images to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
