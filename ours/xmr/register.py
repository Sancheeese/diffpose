"""Deterministic 2D initialization for MRCP/X-ray projection registration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .mind import mind_ssd


@dataclass(frozen=True)
class TranslationSearchResult:
    """Translation applied to the moving projection, measured in pixels."""

    dy: int
    dx: int
    loss: float


def translate_2d(image: torch.Tensor, *, dy: int, dx: int) -> torch.Tensor:
    """Translate a 2D image with zero padding instead of wraparound."""
    if image.ndim != 2:
        raise ValueError("image must have shape (H, W)")

    result = torch.zeros_like(image)
    height, width = image.shape
    source_y_start = max(0, -dy)
    source_y_end = min(height, height - dy)
    source_x_start = max(0, -dx)
    source_x_end = min(width, width - dx)
    if source_y_start >= source_y_end or source_x_start >= source_x_end:
        return result

    target_y_start = source_y_start + dy
    target_y_end = source_y_end + dy
    target_x_start = source_x_start + dx
    target_x_end = source_x_end + dx
    result[target_y_start:target_y_end, target_x_start:target_x_end] = image[
        source_y_start:source_y_end, source_x_start:source_x_end
    ]
    return result


def search_translation_mind(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    max_shift: int,
    mask: torch.Tensor | None = None,
    patch_radius: int = 1,
) -> TranslationSearchResult:
    """Find the integer-pixel translation minimizing masked 2D MIND-SSD.

    This is a coarse image-space initializer. It does not estimate camera pose;
    the next stage should optimize a 3D MRCP projection under calibrated X-ray
    geometry.
    """
    if fixed.ndim != 2 or moving.ndim != 2:
        raise ValueError("fixed and moving must both have shape (H, W)")
    if fixed.shape != moving.shape:
        raise ValueError("fixed and moving must have the same shape")
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")

    best: TranslationSearchResult | None = None
    with torch.no_grad():
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                loss = mind_ssd(
                    fixed,
                    translate_2d(moving, dy=dy, dx=dx),
                    mask=mask,
                    patch_radius=patch_radius,
                ).item()
                candidate = TranslationSearchResult(dy=dy, dx=dx, loss=loss)
                if best is None or candidate.loss < best.loss:
                    best = candidate

    assert best is not None
    return best
