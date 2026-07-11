import time

from einops import rearrange
from torchvision.models.vision_transformer import Encoder, ConvStemConfig
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation
from torchvision.utils import _log_api_usage_once
import torch.nn.functional as F

from diffpose.calibration import convert


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 1  # 修改为1，因为输入是单通道图像
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=1, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size  # 修改为1
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        # self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # 经过编码器后返回特征，不经过分类头
        x = self.encoder(x)

        # 这里直接返回编码后的特征
        return x

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, out_channels),
            nn.SiLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.GroupNorm(32, out_channels),
            # nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# 下采样
class Down_conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 原论文只是做了一个maxpool，并没有在后边加上卷积，此处加入卷积的目的就是为了更好的融合特征
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            # 原文只有maxpool，我这里加入了卷积，为了能更好的融合maxpool的特征
            # nn.Conv2d(channels, channels, kernel_size=1)
        )

        # 方式二：
        self.down1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GroupNorm(16, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.down1(x)


# 上采样,上采样的时候，先将特征图的大小翻倍，翻倍之后还需要还需要
class Up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 方式一：利用各种插值的方式
        # self.up = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(out_channels),
        #     nn.SiLU(inplace=True),
        # )

        # 方式二，转置卷积
        # print(in_channels, out_channels)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.up1(x)


class Decoder(nn.Module):
    def __init__(self, output_channel):
        super(Decoder, self).__init__()

        self.up1 = Up_conv(512, 256)
        self.conv6 = Conv_block(256, 256)
        self.up2 = Up_conv(256, 128)
        self.conv7 = Conv_block(128, 128)
        self.up3 = Up_conv(128, 64)
        self.conv8 = Conv_block(64, 64)
        self.up4 = Up_conv(64, 16)
        self.conv9 = Conv_block(16, 16)
        self.end = nn.Conv2d(16, output_channel, kernel_size=3, padding=1, stride=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        up1 = self.conv6(self.up1(x))
        up2 = self.conv7(self.up2(up1))
        up3 = self.conv8(self.up3(up2))
        up4 = self.conv9(self.up4(up3))

        return self.act(self.end(up4))

class PoseNet(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0
    ):
        super().__init__()
        self.f_size = image_size // patch_size
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # 定义两个独立的 VisionTransformer 网络，一个用于处理图像，一个用于处理 mask
        self.vit_image = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            # norm_layer=create_group_norm()
        )

        self.vit_mask = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            # norm_layer=create_group_norm()
        )

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
                                                     batch_first=True)

        self.rot_regression = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.decoder = Decoder(output_channel=1)
        self.before_decode = nn.Conv2d(hidden_dim, 512, kernel_size=1, stride=1, bias=True)

    def compute_mask_weights(self, mask: torch.Tensor, patch_size: int) -> torch.Tensor:
        """计算 mask 中每个 patch 的权重，基于 mask 中 1 的占比。

        Args:
            mask (torch.Tensor): 输入 mask 图像，形状为 (batch_size, 1, height, width)
            patch_size (int): 每个 patch 的大小（假设是正方形）

        Returns:
            torch.Tensor: 每个 patch 中 1 的占比，形状为 (batch_size, num_patches)
        """
        batch_size, _, height, width = mask.shape

        # 使用 to_patches 方法将 mask 图像分割成 patches
        patches = mask.unfold(2, patch_size, step=patch_size).unfold(3, patch_size, step=patch_size).contiguous()
        patches = rearrange(patches, "b c p1 p2 h w -> b (c p1 p2) h w")

        # 在每个 patch 内计算 1 的数量
        patch_sum = patches.sum(dim=[-2, -1])  # (batch_size, num_patches_h * num_patches_w)

        # 计算每个 patch 中 1 的占比
        patch_area = patch_size * patch_size
        mask_weights = patch_sum / patch_area  # 得到每个 patch 中 1 的占比

        return mask_weights

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        batch_size = image.shape[0]
        # 通过两个ViT模型提取图像和mask的特征
        image_features = self.vit_image(image)
        mask_features = self.vit_mask(mask)

        attn_output, attn_weights = self.cross_attention(mask_features, image_features, image_features)

        # wei = self.compute_mask_weights(mask, self.patch_size)
        # attn_output = attn_output * wei.unsqueeze(-1)
        attn_output = attn_output.permute(0, 2, 1).reshape(batch_size, self.hidden_dim, self.f_size, self.f_size)

        # pred_mask = self.decoder(self.before_decode(attn_output))

        # 全局平均池化（Global Average Pooling）
        pooled_features = F.adaptive_avg_pool2d(attn_output, (1, 1))  # (batch_size, hidden_dim, 1, 1)

        # 将池化后的特征展平为 (batch_size, hidden_dim)
        x = pooled_features.view(batch_size, -1)

        rot = self.rot_regression(x)
        xyz = self.xyz_regression(x)

        # 后续的处理可以在这里添加，暂时先不处理
        return convert(
            [rot, xyz],
            input_parameterization="se3_log_map",
            output_parameterization="se3_exp_map",
            input_convention=None,
        ), 1

def create_group_norm(num_groups=16):
    """创建 GroupNorm 的工厂函数"""
    def _group_norm_factory(hidden_dim):
        # 计算合适的组数
        groups = min(num_groups, hidden_dim)
        # 确保可整除
        while hidden_dim % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, hidden_dim)
    return _group_norm_factory

if __name__ == "__main__":
    # 测试代码
    image = torch.randn(8, 1, 256, 256)  # 假设输入图像为 8 个批次，1 通道，256x256 图像
    mask = torch.randn(8, 1, 256, 256)  # 假设输入 mask 为 8 个批次，1 通道，256x256 mask

    pose_net = PoseNet(
        image_size=256,
        patch_size=16,
        num_layers=12,
        num_heads=8,
        hidden_dim=768,
        mlp_dim=3072,
        dropout=0.1,
        attention_dropout=0.1
    )
    start = time.time()
    image_features, mask_features = pose_net(image, mask)
    print(f"{time.time() - start}")


