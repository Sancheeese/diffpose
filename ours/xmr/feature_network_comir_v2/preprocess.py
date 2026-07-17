"""Preprocessing for CoMIR anti-shortcut training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SquareCropBounds:
    start_y: int
    end_y: int
    start_x: int
    end_x: int
    side: int


def canonical_square_bounds(image_size: int = 256, radius: int = 119) -> SquareCropBounds:
    """Return the centered largest square fully inside the circular FOV."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    side = int(math.floor(radius * math.sqrt(2.0)))
    # Keep an even side so the crop is symmetric around the integer center.
    if side % 2:
        side -= 1
    center = image_size // 2
    start = center - side // 2
    end = start + side
    if start < 0 or end > image_size:
        raise ValueError("Computed square crop is outside the image")
    return SquareCropBounds(start_y=start, end_y=end, start_x=start, end_x=end, side=side)


class CanonicalSquareCrop(nn.Module):
    """Crop the circle's largest inscribed square and resize back to 256."""

    def __init__(self, image_size: int = 256, radius: int = 119, output_size: int = 256) -> None:
        super().__init__()
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        self.image_size = image_size
        self.radius = radius
        self.output_size = output_size
        self.bounds = canonical_square_bounds(image_size=image_size, radius=radius)

    def forward(self, images: Tensor, mode: str = "bilinear") -> Tensor:
        if images.ndim not in (3, 4):
            raise ValueError(f"Expected [C, H, W] or [B, C, H, W], got {tuple(images.shape)}")
        was_unbatched = images.ndim == 3
        if was_unbatched:
            images = images.unsqueeze(0)
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size}, got {tuple(images.shape[-2:])}")

        b = self.bounds
        cropped = images[..., b.start_y : b.end_y, b.start_x : b.end_x]
        if mode == "nearest":
            resized = F.interpolate(cropped, size=(self.output_size, self.output_size), mode=mode)
        else:
            resized = F.interpolate(cropped, size=(self.output_size, self.output_size), mode=mode, align_corners=False)
        return resized.squeeze(0) if was_unbatched else resized
