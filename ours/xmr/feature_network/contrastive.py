"""Coordinate-sampled, symmetric CT-DRR/MRCP contrastive learning loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ContrastiveLossOutput:
    total: Tensor
    xray_to_mrcp: Tensor
    mrcp_to_xray: Tensor
    coordinates: Tensor


class SymmetricCrossModalInfoNCE(nn.Module):
    """Compare same-pose descriptors in both CT-DRR to MRCP directions."""

    def __init__(
        self,
        samples_per_image: int = 256,
        temperature: float = 0.07,
        local_negative_exclusion: int = 2,
    ) -> None:
        super().__init__()
        if samples_per_image <= 0:
            raise ValueError("samples_per_image must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if local_negative_exclusion < 0:
            raise ValueError("local_negative_exclusion cannot be negative")
        self.samples_per_image = samples_per_image
        self.temperature = temperature
        self.local_negative_exclusion = local_negative_exclusion

    def sample_coordinates(self, valid_mask: Tensor) -> Tensor:
        """Draw unique FOV pixel coordinates independently for every batch item."""
        if valid_mask.ndim == 4:
            if valid_mask.shape[1] != 1:
                raise ValueError("A four-dimensional valid_mask must have one channel")
            valid_mask = valid_mask[:, 0]
        if valid_mask.ndim != 3:
            raise ValueError(f"Expected [B, H, W] mask, got {tuple(valid_mask.shape)}")

        coordinates = []
        for image_mask in valid_mask.bool():
            candidates = torch.nonzero(image_mask, as_tuple=False)
            if len(candidates) < self.samples_per_image:
                raise ValueError(
                    f"Need {self.samples_per_image} valid pixels, found only {len(candidates)}"
                )
            selected = torch.randperm(len(candidates), device=candidates.device)[: self.samples_per_image]
            coordinates.append(candidates[selected])
        return torch.stack(coordinates, dim=0)

    @staticmethod
    def _gather(features: Tensor, coordinates: Tensor) -> Tensor:
        batch_size, channels, _, width = features.shape
        if coordinates.ndim != 3 or coordinates.shape[0] != batch_size or coordinates.shape[-1] != 2:
            raise ValueError("Coordinate batch size does not match feature batch size")
        linear_index = coordinates[..., 0] * width + coordinates[..., 1]
        flattened = features.flatten(start_dim=2).transpose(1, 2)
        return torch.gather(flattened, dim=1, index=linear_index.unsqueeze(-1).expand(-1, -1, channels))

    def _directional_loss(self, queries: Tensor, keys: Tensor, coordinates: Tensor) -> Tensor:
        batch_size, samples, channels = queries.shape
        flat_queries = queries.reshape(batch_size * samples, channels)
        flat_keys = keys.reshape(batch_size * samples, channels)
        # Keep the dense similarity matrix in FP32. It is the numerically
        # sensitive part of the loss, especially when the model runs under
        # CUDA autocast.
        logits = flat_queries.float() @ flat_keys.float().transpose(0, 1) / self.temperature

        image_ids = torch.arange(batch_size, device=queries.device).repeat_interleave(samples)
        flat_coordinates = coordinates.reshape(batch_size * samples, 2)
        same_image = image_ids[:, None].eq(image_ids[None, :])
        coordinate_distance = (flat_coordinates[:, None] - flat_coordinates[None, :]).abs().amax(dim=-1)
        remove_local_negative = same_image & coordinate_distance.le(self.local_negative_exclusion)
        valid_candidates = ~remove_local_negative
        targets = torch.arange(batch_size * samples, device=queries.device)
        valid_candidates[targets, targets] = True
        logits = logits.masked_fill(~valid_candidates, torch.finfo(logits.dtype).min)
        return F.cross_entropy(logits, targets)

    def forward(
        self,
        xray_descriptors: Tensor,
        mrcp_descriptors: Tensor,
        valid_mask: Tensor,
    ) -> ContrastiveLossOutput:
        if xray_descriptors.shape != mrcp_descriptors.shape:
            raise ValueError("CT-DRR and MRCP descriptors must have identical shapes")
        if xray_descriptors.ndim != 4:
            raise ValueError("Descriptors must be [B, C, H, W]")
        coordinates = self.sample_coordinates(valid_mask)
        xray_samples = self._gather(xray_descriptors, coordinates)
        mrcp_samples = self._gather(mrcp_descriptors, coordinates)
        xray_to_mrcp = self._directional_loss(xray_samples, mrcp_samples, coordinates)
        mrcp_to_xray = self._directional_loss(mrcp_samples, xray_samples, coordinates)
        return ContrastiveLossOutput(
            total=0.5 * (xray_to_mrcp + mrcp_to_xray),
            xray_to_mrcp=xray_to_mrcp,
            mrcp_to_xray=mrcp_to_xray,
            coordinates=coordinates,
        )
