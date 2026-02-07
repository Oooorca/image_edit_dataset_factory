from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def encode_rgb_png_base64(image_rgb: np.ndarray) -> str:
    image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_rgb_png_base64(data: str) -> np.ndarray:
    raw = base64.b64decode(data.encode("ascii"))
    with Image.open(io.BytesIO(raw)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def encode_mask_png_base64(mask: np.ndarray) -> str:
    image = Image.fromarray(mask.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_mask_png_base64(data: str) -> np.ndarray:
    raw = base64.b64decode(data.encode("ascii"))
    with Image.open(io.BytesIO(raw)) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def encode_rgba_png_base64(image_rgba: np.ndarray) -> str:
    image = Image.fromarray(image_rgba.astype(np.uint8), mode="RGBA")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_rgba_png_base64(data: str) -> np.ndarray:
    raw = base64.b64decode(data.encode("ascii"))
    with Image.open(io.BytesIO(raw)) as img:
        return np.asarray(img.convert("RGBA"), dtype=np.uint8)
