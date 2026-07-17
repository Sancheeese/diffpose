"""Cross-modal 32x32 feature-patch InfoNCE for anti-shortcut CoMIR."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .augment import CropC4Parameters, IndependentCropC4Augment


@dataclass(frozen=True)
class CrossModalPatchLossOutput:
    total: Tensor
    xray_to_mrcp: Tensor
    mrcp_to_xray: Tensor
    canonical_centers_yx: Tensor
    descriptor_shape: tuple[int, ...]


class CrossModalPatchInfoNCE(nn.Module):
    """Per rendered pair: 24 patches, one positive and 23 cross-modal negatives."""

    def __init__(
        self,
        patch_pairs_per_image: int = 24,
        patch_size: int = 32,
        temperature: float = 0.07,
        minimum_patch_gap: float = 2.0,
        output_size: int = 256,
    ) -> None:
        super().__init__()
        if patch_pairs_per_image <= 0 or patch_size <= 0 or temperature <= 0:
            raise ValueError("patch_pairs_per_image, patch_size, and temperature must be positive")
        self.patch_pairs_per_image = patch_pairs_per_image
        self.patch_size = patch_size
        self.temperature = temperature
        self.minimum_patch_gap = minimum_patch_gap
        self.output_size = output_size

    def _candidate_bounds(self, parameters: CropC4Parameters, index: int) -> tuple[int, int, int, int]:
        size = float(parameters.crop_size[index].item())
        xray_origin = parameters.xray_origin_yx[index]
        mrcp_origin = parameters.mrcp_origin_yx[index]
        top = max(float(xray_origin[0]), float(mrcp_origin[0]))
        left = max(float(xray_origin[1]), float(mrcp_origin[1]))
        bottom = min(float(xray_origin[0]) + size, float(mrcp_origin[0]) + size)
        right = min(float(xray_origin[1]) + size, float(mrcp_origin[1]) + size)
        margin = self.patch_size * size / self.output_size / 2.0 + 1.0
        return (
            int(torch.ceil(torch.tensor(top + margin)).item()),
            int(torch.floor(torch.tensor(bottom - margin)).item()),
            int(torch.ceil(torch.tensor(left + margin)).item()),
            int(torch.floor(torch.tensor(right - margin)).item()),
        )

    def sample_canonical_centers(self, parameters: CropC4Parameters) -> Tensor:
        selected_per_image = []
        for index in range(parameters.crop_size.shape[0]):
            y_start, y_end, x_start, x_end = self._candidate_bounds(parameters, index)
            if y_end <= y_start or x_end <= x_start:
                raise ValueError("Crop overlap cannot contain a complete feature patch")
            size = float(parameters.crop_size[index].item())
            stride = int(torch.ceil(torch.tensor(self.patch_size * size / self.output_size + self.minimum_patch_gap)).item())
            candidates = torch.cartesian_prod(
                torch.arange(y_start, y_end, stride, device=parameters.crop_size.device),
                torch.arange(x_start, x_end, stride, device=parameters.crop_size.device),
            )
            if len(candidates) < self.patch_pairs_per_image:
                raise ValueError("Crop overlap cannot provide non-overlapping feature patch pairs")
            indices = torch.randperm(len(candidates), device=candidates.device)[: self.patch_pairs_per_image]
            selected_per_image.append(candidates[indices])
        return torch.stack(selected_per_image)

    def _canonical_patch_grids(self, centers_yx: Tensor, parameters: CropC4Parameters) -> Tensor:
        step = parameters.crop_size.float() / float(self.output_size)
        offsets = torch.arange(
            -self.patch_size // 2,
            self.patch_size - self.patch_size // 2,
            device=centers_yx.device,
            dtype=torch.float32,
        )
        delta_y, delta_x = torch.meshgrid(offsets, offsets, indexing="ij")
        delta = torch.stack((delta_y, delta_x), dim=-1)[None, None]
        return centers_yx.float()[:, :, None, None, :] + delta * step[:, None, None, None, None]

    def _sample_descriptors(
        self,
        features: Tensor,
        grids_yx: Tensor,
        parameters: CropC4Parameters,
        augment: IndependentCropC4Augment,
        modality: str,
    ) -> Tensor:
        batch_size, channels, height, width = features.shape
        mapped = augment.canonical_to_output(grids_yx, parameters, modality)
        if mapped[..., 0].min() < 0 or mapped[..., 1].min() < 0 or mapped[..., 0].max() > height - 1 or mapped[..., 1].max() > width - 1:
            raise ValueError("Mapped feature patch is outside the transformed image")
        grid = torch.stack(
            (mapped[..., 1] * 2.0 / (width - 1) - 1.0, mapped[..., 0] * 2.0 / (height - 1) - 1.0),
            dim=-1,
        )
        count = grids_yx.shape[1]
        sampled = F.grid_sample(
            features[:, None].expand(-1, count, -1, -1, -1).reshape(batch_size * count, channels, height, width).float(),
            grid.reshape(batch_size * count, self.patch_size, self.patch_size, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return F.normalize(sampled.flatten(start_dim=1), p=2, dim=1, eps=1e-6).reshape(batch_size, count, -1)

    def forward(
        self,
        xray_features: Tensor,
        mrcp_features: Tensor,
        parameters: CropC4Parameters,
        augment: IndependentCropC4Augment,
    ) -> CrossModalPatchLossOutput:
        if xray_features.shape != mrcp_features.shape or xray_features.shape[-2:] != (self.output_size, self.output_size):
            raise ValueError("Feature maps must be equal full-resolution tensors")
        centers = self.sample_canonical_centers(parameters)
        grids = self._canonical_patch_grids(centers, parameters)
        xray_descriptors = self._sample_descriptors(xray_features, grids, parameters, augment, "xray")
        mrcp_descriptors = self._sample_descriptors(mrcp_features, grids, parameters, augment, "mrcp")
        logits = torch.bmm(xray_descriptors, mrcp_descriptors.transpose(1, 2)) / self.temperature
        targets = torch.arange(logits.shape[1], device=logits.device).repeat(logits.shape[0])
        xray_to_mrcp = F.cross_entropy(logits.flatten(end_dim=1), targets)
        mrcp_to_xray = F.cross_entropy(logits.transpose(1, 2).flatten(end_dim=1), targets)
        return CrossModalPatchLossOutput(
            total=0.5 * (xray_to_mrcp + mrcp_to_xray),
            xray_to_mrcp=xray_to_mrcp,
            mrcp_to_xray=mrcp_to_xray,
            canonical_centers_yx=centers,
            descriptor_shape=tuple(xray_descriptors.shape),
        )
