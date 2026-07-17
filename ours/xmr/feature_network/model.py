"""Full-resolution shared U-Net, dense-neighbour similarity, and descriptors."""

from __future__ import annotations

from itertools import combinations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):
    """A 3x3 convolution followed by the normalization used throughout the U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(ConvNormAct(in_channels, out_channels), ConvNormAct(out_channels, out_channels))


class BlurPool2d(nn.Module):
    """Fixed anti-aliasing filter applied before the learned stride-two convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel = torch.tensor([1.0, 2.0, 1.0])
        kernel = torch.outer(kernel, kernel) / 16.0
        self.register_buffer("kernel", kernel.expand(channels, 1, 3, 3).contiguous(), persistent=False)
        self.channels = channels

    def forward(self, inputs: Tensor) -> Tensor:
        return F.conv2d(inputs, self.kernel.to(dtype=inputs.dtype), padding=1, groups=self.channels)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.blur = BlurPool2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(self.blur(inputs))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.block = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        inputs = self.reduce(inputs)
        return self.block(torch.cat((inputs, skip), dim=1))


class FullResolutionUNet(nn.Module):
    """Four-level GroupNorm U-Net whose output matches the input resolution."""

    def __init__(self, in_channels: int = 1, feature_channels: int = 64) -> None:
        super().__init__()
        self.enc0 = DoubleConv(in_channels, 32)
        self.down1 = Downsample(32, 64)
        self.enc1 = DoubleConv(64, 64)
        self.down2 = Downsample(64, 128)
        self.enc2 = DoubleConv(128, 128)
        self.down3 = Downsample(128, 256)
        self.enc3 = DoubleConv(256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.up0 = UpBlock(64, 32, 32)
        self.project = nn.Conv2d(32, feature_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(inputs.shape)}")
        if inputs.shape[-2] % 8 or inputs.shape[-1] % 8:
            raise ValueError("Input height and width must be divisible by 8")
        enc0 = self.enc0(inputs)
        enc1 = self.enc1(self.down1(enc0))
        enc2 = self.enc2(self.down2(enc1))
        enc3 = self.enc3(self.down3(enc2))
        decoded = self.up2(enc3, enc2)
        decoded = self.up1(decoded, enc1)
        decoded = self.up0(decoded, enc0)
        return self.project(decoded)


class DenseNeighbourSimilarity(nn.Module):
    """Compute 12 full-resolution self-similarity channels from a feature map."""

    def __init__(self, dilations: tuple[int, int] = (1, 3)) -> None:
        super().__init__()
        self.dilations = dilations

    @staticmethod
    def _neighbours(features: Tensor, dilation: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        height, width = features.shape[-2:]
        padded = F.pad(features, (dilation, dilation, dilation, dilation), mode="reflect")
        return (
            padded[:, :, 0:height, dilation : dilation + width],
            padded[:, :, 2 * dilation : 2 * dilation + height, dilation : dilation + width],
            padded[:, :, dilation : dilation + height, 0:width],
            padded[:, :, dilation : dilation + height, 2 * dilation : 2 * dilation + width],
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(features.shape)}")
        # DNS contains distance, square, and exponential operations. Keeping
        # this block in FP32 avoids underflow/overflow under float16 autocast.
        normalized = F.normalize(features.float(), p=2, dim=1, eps=1e-6)
        similarities: list[Tensor] = []
        for dilation in self.dilations:
            neighbours = self._neighbours(normalized, dilation)
            for first, second in combinations(neighbours, 2):
                squared_distance = (first - second).square().mean(dim=1, keepdim=True)
                similarities.append(torch.exp(-squared_distance))
        return torch.cat(similarities, dim=1)


class DescriptorHead(nn.Module):
    """Compress the 12 DNS channels into a normalized 32-channel descriptor."""

    def __init__(self, in_channels: int = 12, descriptor_channels: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, descriptor_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(descriptor_channels, descriptor_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(descriptor_channels, descriptor_channels, kernel_size=3, padding=1),
        )

    def forward(self, dns: Tensor) -> Tensor:
        return F.normalize(self.layers(dns), p=2, dim=1, eps=1e-6)


class CommonFeatureNetwork(nn.Module):
    """Shared network used independently on CT-DRR and MRCP projection images."""

    def __init__(self) -> None:
        super().__init__()
        self.unet = FullResolutionUNet(in_channels=1, feature_channels=64)
        self.dns = DenseNeighbourSimilarity()
        self.head = DescriptorHead(in_channels=12, descriptor_channels=32)

    def forward(self, images: Tensor) -> Tensor:
        features = self.unet(images)
        return self.head(self.dns(features))
