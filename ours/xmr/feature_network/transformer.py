"""Image normalization for paired CT-DRR and MRCP feature training."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.transforms import Resize


class PerImageLegacyTransform(nn.Module):
    """Apply the legacy SXH image transform with independent per-image min-max.

    The processing order intentionally matches ``ours.utils.CT_dataset.Transforms``:
    resize, min-max normalization, inversion, circular field-of-view masking,
    then fixed standardization. Unlike the legacy implementation, each batch
    item obtains its own min/max so one rendered image cannot affect another.
    """

    def __init__(
        self,
        size: int = 256,
        eps: float = 1e-6,
        radius: int = 119,
        mean: float = 0.3080,
        std: float = 0.1494,
    ) -> None:
        super().__init__()
        if size <= 0:
            raise ValueError("size must be positive")
        if radius <= 0:
            raise ValueError("radius must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if std <= 0:
            raise ValueError("std must be positive")

        self.resize = Resize((size, size), antialias=True)
        self.eps = eps
        self.mean = mean
        self.std = std

        coordinates = torch.arange(size, dtype=torch.float32) - size // 2
        y_coord, x_coord = torch.meshgrid(coordinates, coordinates, indexing="ij")
        mask = (x_coord.square() + y_coord.square() <= radius**2).to(torch.float32)
        self.register_buffer("fov_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, images: Tensor, invert: bool = True) -> Tensor:
        """Normalize a ``[B, C, H, W]`` or ``[C, H, W]`` tensor.

        Per-image extrema are computed across every channel and pixel within
        each batch item. The result preserves the input batch convention.
        """
        if images.ndim not in (3, 4):
            raise ValueError(f"Expected [C, H, W] or [B, C, H, W], got {tuple(images.shape)}")
        if not torch.is_floating_point(images):
            images = images.to(torch.float32)

        was_unbatched = images.ndim == 3
        if was_unbatched:
            images = images.unsqueeze(0)

        images = self.resize(images)
        image_min = images.amin(dim=(1, 2, 3), keepdim=True)
        image_max = images.amax(dim=(1, 2, 3), keepdim=True)
        images = (images - image_min) / (image_max - image_min + self.eps)
        if invert:
            images = 1.0 - images

        mask = self.fov_mask.to(device=images.device, dtype=images.dtype)
        images = images * mask
        images = (images - self.mean) / self.std
        return images.squeeze(0) if was_unbatched else images
