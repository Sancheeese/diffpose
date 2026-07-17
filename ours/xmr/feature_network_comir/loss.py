"""Patch-sampled symmetric InfoNCE for CoMIR-style feature maps."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PatchContrastiveLossOutput:
    total: Tensor
    xray_to_mrcp: Tensor
    mrcp_to_xray: Tensor
    centers: Tensor
    patch_descriptors_shape: tuple[int, ...]


class SymmetricPatchInfoNCE(nn.Module):
    """Compare corresponding feature-map patches in both cross-modal directions."""

    def __init__(
        self,
        patches_per_image: int = 32,
        patch_size: int = 32,
        temperature: float = 0.07,
        local_negative_exclusion: int = 32,
    ) -> None:
        super().__init__()
        if patches_per_image <= 0:
            raise ValueError("patches_per_image must be positive")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if local_negative_exclusion < 0:
            raise ValueError("local_negative_exclusion cannot be negative")
        self.patches_per_image = patches_per_image
        self.patch_size = patch_size
        self.temperature = temperature
        self.local_negative_exclusion = local_negative_exclusion

    def _valid_centers(self, valid_mask: Tensor) -> Tensor:
        if valid_mask.ndim == 4:
            if valid_mask.shape[1] != 1:
                raise ValueError("A four-dimensional valid_mask must have one channel")
            valid_mask = valid_mask[:, 0]
        if valid_mask.ndim != 3:
            raise ValueError(f"Expected [B, H, W] mask, got {tuple(valid_mask.shape)}")

        patch = self.patch_size
        kernel = torch.ones((1, 1, patch, patch), dtype=torch.float32, device=valid_mask.device)
        counts = F.conv2d(valid_mask[:, None].float(), kernel)
        valid = counts[:, 0].eq(float(patch * patch))
        # conv2d returns top-left patch positions. Convert them to centers
        # using the same even-patch convention used by _extract_patches:
        # rows center - patch//2 ... center + patch//2 - 1.
        top_left = torch.nonzero(valid, as_tuple=False)
        if top_left.numel() == 0:
            return top_left.reshape(0, 3)
        top_left[:, 1:] += patch // 2
        return top_left

    def sample_centers(self, valid_mask: Tensor) -> Tensor:
        """Draw patch centers whose full patches lie inside the valid mask."""
        valid_centers = self._valid_centers(valid_mask)
        batch_size = valid_mask.shape[0]
        centers = []
        for batch_index in range(batch_size):
            candidates = valid_centers[valid_centers[:, 0].eq(batch_index), 1:]
            if len(candidates) < self.patches_per_image:
                raise ValueError(
                    f"Need {self.patches_per_image} valid patch centers, found only {len(candidates)}"
                )
            selected = torch.randperm(len(candidates), device=candidates.device)[: self.patches_per_image]
            centers.append(candidates[selected])
        return torch.stack(centers, dim=0)

    def _extract_patches(self, features: Tensor, centers: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(features.shape)}")
        batch_size, channels, height, width = features.shape
        if centers.ndim != 3 or centers.shape[0] != batch_size or centers.shape[-1] != 2:
            raise ValueError("Patch center batch size does not match feature batch size")

        half = self.patch_size // 2
        offsets = torch.arange(-half, self.patch_size - half, device=features.device)
        y = centers[..., 0, None, None] + offsets[None, None, :, None]
        x = centers[..., 1, None, None] + offsets[None, None, None, :]
        if y.min() < 0 or x.min() < 0 or y.max() >= height or x.max() >= width:
            raise ValueError("Patch centers produce out-of-bounds patches")

        linear_index = (y * width + x).reshape(batch_size, -1)
        flattened = features.flatten(start_dim=2)
        gathered = torch.gather(flattened, dim=2, index=linear_index[:, None].expand(-1, channels, -1))
        patches = gathered.reshape(batch_size, channels, self.patches_per_image, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(batch_size * self.patches_per_image, -1)
        return F.normalize(patches.float(), p=2, dim=1, eps=1e-6)

    def _directional_loss(self, queries: Tensor, keys: Tensor, centers: Tensor) -> Tensor:
        logits = queries @ keys.transpose(0, 1) / self.temperature
        total = centers.shape[0] * centers.shape[1]
        image_ids = torch.arange(centers.shape[0], device=centers.device).repeat_interleave(centers.shape[1])
        flat_centers = centers.reshape(total, 2)
        same_image = image_ids[:, None].eq(image_ids[None, :])
        center_distance = (flat_centers[:, None] - flat_centers[None, :]).abs().amax(dim=-1)
        remove_local_negative = same_image & center_distance.le(self.local_negative_exclusion)
        valid_candidates = ~remove_local_negative
        targets = torch.arange(total, device=centers.device)
        valid_candidates[targets, targets] = True
        logits = logits.masked_fill(~valid_candidates, torch.finfo(logits.dtype).min)
        return F.cross_entropy(logits, targets)

    def forward(self, xray_features: Tensor, mrcp_features: Tensor, valid_mask: Tensor) -> PatchContrastiveLossOutput:
        if xray_features.shape != mrcp_features.shape:
            raise ValueError("CT-DRR and MRCP feature maps must have identical shapes")
        centers = self.sample_centers(valid_mask)
        xray_patches = self._extract_patches(xray_features, centers)
        mrcp_patches = self._extract_patches(mrcp_features, centers)
        xray_to_mrcp = self._directional_loss(xray_patches, mrcp_patches, centers)
        mrcp_to_xray = self._directional_loss(mrcp_patches, xray_patches, centers)
        return PatchContrastiveLossOutput(
            total=0.5 * (xray_to_mrcp + mrcp_to_xray),
            xray_to_mrcp=xray_to_mrcp,
            mrcp_to_xray=mrcp_to_xray,
            centers=centers,
            patch_descriptors_shape=tuple(xray_patches.shape),
        )
