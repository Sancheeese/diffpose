"""Two-branch CoMIR-style dense feature networks."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ours.xmr.feature_network.model import FullResolutionUNet


class CoMIRPatchFeatureNetwork(nn.Module):
    """One modality branch that directly outputs a normalized CoMIR map."""

    def __init__(self, in_channels: int = 1, feature_channels: int = 32) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.unet = FullResolutionUNet(in_channels=in_channels, feature_channels=feature_channels)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(images.shape)}")
        return F.normalize(self.unet(images), p=2, dim=1, eps=1e-6)


class CoMIRTwoBranchFeatureNetwork(nn.Module):
    """Independent CT-DRR and MRCP branches with identical architecture."""

    def __init__(self, in_channels: int = 1, feature_channels: int = 32) -> None:
        super().__init__()
        self.xray_net = CoMIRPatchFeatureNetwork(in_channels=in_channels, feature_channels=feature_channels)
        self.mrcp_net = CoMIRPatchFeatureNetwork(in_channels=in_channels, feature_channels=feature_channels)

    def forward(self, xray_images: Tensor, mrcp_images: Tensor) -> tuple[Tensor, Tensor]:
        return self.xray_net(xray_images), self.mrcp_net(mrcp_images)
