"""2D MIND features and a masked MIND-SSD similarity measure."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional


_DEFAULT_OFFSETS: tuple[tuple[int, int], ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))


def _as_image_batch(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        return image.unsqueeze(0).unsqueeze(0)
    if image.ndim == 3:
        return image.unsqueeze(1)
    if image.ndim == 4 and image.shape[1] == 1:
        return image
    raise ValueError("image must have shape (H, W), (N, H, W), or (N, 1, H, W)")


def _shift(image: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    pad_y = abs(dy)
    pad_x = abs(dx)
    padded = functional.pad(image, (pad_x, pad_x, pad_y, pad_y), mode="replicate")
    return padded[..., pad_y + dy : pad_y + dy + height, pad_x + dx : pad_x + dx + width]


def mind_descriptor(
    image: torch.Tensor,
    *,
    patch_radius: int = 1,
    offsets: Sequence[tuple[int, int]] = _DEFAULT_OFFSETS,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return a 2D modality-independent local self-similarity descriptor.

    The output has shape ``(N, len(offsets), H, W)``. It is a compact 2D
    MIND-style descriptor for matching projections, not the 3D six-neighbour
    MIND-SSC variant used by ConvexAdam.
    """
    if patch_radius < 0:
        raise ValueError("patch_radius must be non-negative")
    if not offsets:
        raise ValueError("offsets must not be empty")

    batch = _as_image_batch(image).float()
    kernel_size = 2 * patch_radius + 1
    distances = []
    for dy, dx in offsets:
        squared_difference = (batch - _shift(batch, dy, dx)).square()
        distances.append(
            functional.avg_pool2d(
                squared_difference,
                kernel_size=kernel_size,
                stride=1,
                padding=patch_radius,
            )
        )

    distance_stack = torch.cat(distances, dim=1)
    local_variance = distance_stack.mean(dim=1, keepdim=True).clamp_min(epsilon)
    return torch.exp(-distance_stack / local_variance)


def mind_ssd(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    patch_radius: int = 1,
) -> torch.Tensor:
    """Return mean squared MIND-feature error, optionally restricted to a ROI."""
    fixed_features = mind_descriptor(fixed, patch_radius=patch_radius)
    moving_features = mind_descriptor(moving, patch_radius=patch_radius)
    squared_error = (fixed_features - moving_features).square()

    if mask is None:
        return squared_error.mean()

    mask_batch = _as_image_batch(mask).to(dtype=squared_error.dtype)
    if mask_batch.shape[0] != squared_error.shape[0] or mask_batch.shape[-2:] != squared_error.shape[-2:]:
        raise ValueError("mask batch and spatial dimensions must match the images")
    weighted_error = squared_error * mask_batch
    denominator = mask_batch.sum() * squared_error.shape[1]
    if denominator <= 0:
        raise ValueError("mask must include at least one pixel")
    return weighted_error.sum() / denominator
