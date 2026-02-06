from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from image_edit_dataset_factory.backends.qwen_image_edit_modelscope import (
    QwenImageEditModelScopeBackend,
)
from image_edit_dataset_factory.backends.qwen_layered_modelscope import (
    QwenLayeredModelScopeBackend,
)
from image_edit_dataset_factory.utils.image_io import read_image_rgb, write_image_rgb, write_mask
from image_edit_dataset_factory.utils.mask_ops import mask_from_bbox


def _pick_mask(alphas: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    best: np.ndarray | None = None
    best_ratio = 1.0
    total = float(h * w)

    for alpha in alphas:
        mask = (alpha > 127).astype(np.uint8) * 255
        ratio = float(mask.sum() / 255.0 / total)
        if 0.01 <= ratio <= 0.7 and ratio < best_ratio:
            best = mask
            best_ratio = ratio

    if best is not None:
        return best

    return mask_from_bbox((h, w), (w // 4, h // 4, (3 * w) // 4, (3 * h) // 4))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen layered+edit on one sample image")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output-dir", default="outputs/qwen_single", help="Output directory")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument(
        "--layered-model-dir",
        default="qwen/Qwen-Image-Layered",
        help="Local model dir or ModelScope model id",
    )
    parser.add_argument(
        "--edit-model-dir",
        default="Qwen/Qwen-Image-Edit",
        help="Local model dir or ModelScope model id",
    )
    parser.add_argument(
        "--prompt",
        default="删除主体区域并自然补全背景",
        help="Edit prompt",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = read_image_rgb(args.image)
    write_image_rgb(out_dir / "input.jpg", image)

    layered = QwenLayeredModelScopeBackend(model_dir=args.layered_model_dir, device=args.device)
    layers = layered.decompose(image)

    alphas: list[np.ndarray] = []
    for idx, layer in enumerate(layers):
        write_image_rgb(out_dir / f"layer_{idx:02d}.jpg", layer.rgba[:, :, :3])
        write_mask(out_dir / f"layer_{idx:02d}_alpha.png", layer.alpha)
        alphas.append(layer.alpha)

    mask = _pick_mask(alphas, image.shape[:2])
    write_mask(out_dir / "edit_mask.png", mask)

    editor = QwenImageEditModelScopeBackend(model_dir=args.edit_model_dir, device=args.device)
    result = editor.inpaint(image, mask, prompt=args.prompt)
    write_image_rgb(out_dir / "result.jpg", result)

    print(f"saved test outputs to: {out_dir}")
    print("files: input.jpg, layer_*.jpg, layer_*_alpha.png, edit_mask.png, result.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
