"""Padding-free crop and C4 augmentations with exact canonical coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CropC4Parameters:
    """Per-image maps from canonical-square pixels to each transformed image."""

    crop_size: Tensor
    xray_origin_yx: Tensor
    mrcp_origin_yx: Tensor
    xray_rotation_k: Tensor
    mrcp_rotation_k: Tensor


@dataclass(frozen=True)
class AugmentedPair:
    xray: Tensor
    mrcp: Tensor
    parameters: CropC4Parameters


class IndependentCropC4Augment(nn.Module):
    """Use independently shifted crops and independent 90-degree rotations.

    Both modalities use the same crop size so a matching output patch has the
    same physical scale. Their origins and C4 rotations remain independent.
    """

    def __init__(
        self,
        image_size: int = 256,
        output_size: int = 256,
        crop_sizes: tuple[int, ...] = (160, 176, 192, 208, 224),
        max_relative_shift: int = 16,
    ) -> None:
        super().__init__()
        if not crop_sizes or any(size <= 0 or size > image_size for size in crop_sizes):
            raise ValueError("crop_sizes must contain valid crop sizes")
        self.image_size = image_size
        self.output_size = output_size
        self.crop_sizes = tuple(crop_sizes)
        self.max_relative_shift = max_relative_shift

    def sample_parameters(self, batch_size: int, device: torch.device) -> CropC4Parameters:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        size_choices = torch.tensor(self.crop_sizes, device=device)
        crop_size = size_choices[torch.randint(len(size_choices), (batch_size,), device=device)]
        xray_origins: list[Tensor] = []
        mrcp_origins: list[Tensor] = []
        for size in crop_size.tolist():
            max_origin = self.image_size - int(size)
            base = torch.randint(max_origin + 1, (2,), device=device)
            jitter_x = torch.randint(-self.max_relative_shift, self.max_relative_shift + 1, (2,), device=device)
            jitter_m = torch.randint(-self.max_relative_shift, self.max_relative_shift + 1, (2,), device=device)
            xray_origins.append((base + jitter_x).clamp(0, max_origin))
            mrcp_origins.append((base + jitter_m).clamp(0, max_origin))
        return CropC4Parameters(
            crop_size=crop_size,
            xray_origin_yx=torch.stack(xray_origins),
            mrcp_origin_yx=torch.stack(mrcp_origins),
            xray_rotation_k=torch.randint(4, (batch_size,), device=device),
            mrcp_rotation_k=torch.randint(4, (batch_size,), device=device),
        )

    def _crop_resize_rotate(self, images: Tensor, origins_yx: Tensor, crop_sizes: Tensor, rotations: Tensor) -> Tensor:
        outputs = []
        for index in range(images.shape[0]):
            size = int(crop_sizes[index].item())
            y, x = [int(value) for value in origins_yx[index].tolist()]
            crop = images[index : index + 1, :, y : y + size, x : x + size]
            resized = F.interpolate(crop, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
            outputs.append(torch.rot90(resized, int(rotations[index].item()), dims=(-2, -1)))
        return torch.cat(outputs, dim=0)

    def forward(self, xray: Tensor, mrcp: Tensor) -> AugmentedPair:
        if xray.shape != mrcp.shape or xray.ndim != 4 or xray.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError("Expected equal [B, C, image_size, image_size] modality tensors")
        parameters = self.sample_parameters(xray.shape[0], xray.device)
        return AugmentedPair(
            xray=self._crop_resize_rotate(xray, parameters.xray_origin_yx, parameters.crop_size, parameters.xray_rotation_k),
            mrcp=self._crop_resize_rotate(mrcp, parameters.mrcp_origin_yx, parameters.crop_size, parameters.mrcp_rotation_k),
            parameters=parameters,
        )

    def canonical_to_output(self, canonical_yx: Tensor, parameters: CropC4Parameters, modality: str) -> Tensor:
        """Map [..., 2] canonical coordinates to transformed output coordinates."""
        if modality == "xray":
            origins, rotations = parameters.xray_origin_yx, parameters.xray_rotation_k
        elif modality == "mrcp":
            origins, rotations = parameters.mrcp_origin_yx, parameters.mrcp_rotation_k
        else:
            raise ValueError("modality must be 'xray' or 'mrcp'")
        if canonical_yx.shape[0] != parameters.crop_size.shape[0] or canonical_yx.shape[-1] != 2:
            raise ValueError("canonical_yx must start with batch dimension and end in y/x")

        view_shape = (canonical_yx.shape[0],) + (1,) * (canonical_yx.ndim - 2) + (1,)
        scale = (self.output_size / parameters.crop_size.float()).reshape(view_shape)
        origin = origins.float().reshape((origins.shape[0],) + (1,) * (canonical_yx.ndim - 2) + (2,))
        output_yx = (canonical_yx.float() + 0.5 - origin) * scale - 0.5
        y, x = output_yx.unbind(dim=-1)
        size_minus_one = float(self.output_size - 1)
        rotation = rotations.reshape((rotations.shape[0],) + (1,) * (canonical_yx.ndim - 2))
        mapped_y = torch.where(rotation.eq(0), y, torch.where(rotation.eq(1), size_minus_one - x, torch.where(rotation.eq(2), size_minus_one - y, x)))
        mapped_x = torch.where(rotation.eq(0), x, torch.where(rotation.eq(1), y, torch.where(rotation.eq(2), size_minus_one - x, size_minus_one - y)))
        return torch.stack((mapped_y, mapped_x), dim=-1)
