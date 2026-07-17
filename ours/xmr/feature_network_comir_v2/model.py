"""Independent dense feature branches for anti-shortcut CoMIR training."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from ours.xmr.feature_network.model import FullResolutionUNet


class CoMIRFeatureBranch(nn.Module):
    """Map one modality image to a full-resolution, channel-normalized feature map."""

    def __init__(self, in_channels: int = 1, feature_channels: int = 32) -> None:
        super().__init__()
        self.unet = FullResolutionUNet(in_channels=in_channels, feature_channels=feature_channels)

    def forward(self, images: Tensor) -> Tensor:
        return F.normalize(self.unet(images), p=2, dim=1, eps=1e-6)


class CoMIRTwoBranchFeatureNetwork(nn.Module):
    """Identical U-Net topology with independent X-ray and MRCP weights."""

    def __init__(self, feature_channels: int = 32) -> None:
        super().__init__()
        self.xray_net = CoMIRFeatureBranch(feature_channels=feature_channels)
        self.mrcp_net = CoMIRFeatureBranch(feature_channels=feature_channels)

    def forward(self, xray_images: Tensor, mrcp_images: Tensor) -> tuple[Tensor, Tensor]:
        return self.xray_net(xray_images), self.mrcp_net(mrcp_images)
