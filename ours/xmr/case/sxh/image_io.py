"""PNG export helpers for SXH web DRR visualization."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def to_zero_one(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + eps)


def write_gray_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = (to_zero_one(image) * 255).astype(np.uint8)
    if not cv2.imwrite(str(path), image_u8):
        raise IOError(f"Failed to write image: {path}")


def write_overlay_png(path: Path, background: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    background_u8 = (to_zero_one(background) * 255).astype(np.uint8)
    mask_u8 = (to_zero_one(mask) * 255).astype(np.uint8)

    bg_color = cv2.cvtColor(background_u8, cv2.COLOR_GRAY2BGR)
    red_mask = np.zeros_like(bg_color)
    red_mask[:, :, 2] = 255
    alpha_mask = (mask_u8 * alpha / 255.0).astype(np.float32)
    result = bg_color.astype(np.float32) * (1 - alpha_mask[:, :, None]) + red_mask.astype(np.float32) * alpha_mask[:, :, None]
    result = np.clip(result, 0, 255).astype(np.uint8)

    if not cv2.imwrite(str(path), result):
        raise IOError(f"Failed to write image: {path}")
