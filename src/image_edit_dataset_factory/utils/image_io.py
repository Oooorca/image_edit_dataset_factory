from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def read_image_pil(path: str | Path, mode: str = "RGB") -> Image.Image:
    with Image.open(path) as img:
        fixed = ImageOps.exif_transpose(img)
        return fixed.convert(mode)


def read_image_rgb(path: str | Path) -> np.ndarray:
    img = read_image_pil(path, mode="RGB")
    return np.asarray(img)


def read_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"))


def write_image_rgb(path: str | Path, image: np.ndarray, quality: int = 95) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    if target.suffix.lower() in {".jpg", ".jpeg"}:
        pil_image.save(target, quality=quality, optimize=True)
    else:
        pil_image.save(target)


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray(mask.astype(np.uint8), mode="L")
    mask_img.save(target)


def convert_image(path: str | Path, output_path: str | Path, mode: str = "RGB") -> None:
    img = read_image_pil(path, mode=mode)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def image_shape(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as img:
        fixed = ImageOps.exif_transpose(img)
        return fixed.width, fixed.height


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}


def is_corrupted(path: str | Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        _ = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return _ is None
    except Exception:
        return True


def ensure_jpeg(path: str | Path, output_path: str | Path) -> Path:
    img = read_image_pil(path, mode="RGB")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    img.save(target, quality=95, optimize=True)
    return target
