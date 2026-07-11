import time

import math
import matplotlib.pyplot as plt
import torch
import timm
import torch
from einops import rearrange

from diffpose.calibration import RigidTransform, convert
from ours.cnnnet.FRFT import FrFTModule
from ours.dataset.CT_dataset import Transforms
# from diffpose.deepfluoro import Transforms

from torch import nn
import torch.nn.functional as F
from torch_frft.frft_module import frft
from torch_frft.dfrft_module import dfrft

from ours.utils.CT_dataset import toZeroOne
from ours.utils.img_utils import center_crop_and_resize_v2, batch_crop_largest_square_from_circle

N_ANGULAR_COMPONENTS = {
    "axis_angle": 3,
    "euler_angles": 3,
    "se3_log_map": 3,
    "quaternion": 4,
    "rotation_6d": 6,
    "rotation_10d": 10,
    "quaternion_adjugate": 10,
}

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
        self.up4 = Up_conv(64, 32)
        self.conv9 = Conv_block(32, 32)
        self.up5 = Up_conv(32, 16)
        self.conv10 = Conv_block(16, 16)
        self.end = nn.Conv2d(16, output_channel, kernel_size=3, padding=1, stride=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        up1 = self.conv6(self.up1(x))
        up2 = self.conv7(self.up2(up1))
        up3 = self.conv8(self.up3(up2))
        up4 = self.conv9(self.up4(up3))
        up5 = self.conv10(self.up5(up4))

        return self.act(self.end(up5))

class PoseRegressor(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        self.backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )

        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': self.backbone.layer1,
            'layer2': self.backbone.layer2,
            'layer3': self.backbone.layer3,
            'layer4': self.backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x, mask):
        img_feat = self.backbone.forward_features(x)
        mask_feat = self.backbone2.forward_features(mask)

        img_pooled = F.adaptive_avg_pool2d(img_feat, 1).flatten(1)
        mask_pooled = F.adaptive_avg_pool2d(mask_feat, 1).flatten(1)
        all_feat = torch.cat([img_pooled, mask_pooled], dim=1)

        pred_mask = self.decoder(img_feat)

        rot = self.rot_regression(all_feat)
        xyz = self.xyz_regression(all_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class SpatialGate(nn.Module):
    """
    轻量级空间注意力门控模块
    输入: mask_feat (B, in_channels, H, W)
    输出: spatial_weight (B, 1, H, W)，值域 [0, 1]
    """
    def __init__(self, in_channels: int, reduction: int = 2):
        super().__init__()
        mid_channels = in_channels // reduction  # 防止通道太少

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, mid_channels),   # 推荐 GroupNorm，尤其 batch_size 小
            nn.ReLU(),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, mid_channels),
            nn.ReLU(),

            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),  # 降到 1 通道
            nn.Sigmoid()                                           # 输出 0~1 空间权重
        )

    def forward(self, x):
        return self.gate(x)

class EnhancedSemanticAttentionModule(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8):
        super(EnhancedSemanticAttentionModule, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.num_heads = num_heads

        # 线性层，用于调整局部特征维度以匹配全局特征
        self.adjust_local_dim = nn.Linear(local_dim, global_dim)

        # Cross-Attention layers
        self.global_to_local_attention = nn.MultiheadAttention(global_dim, num_heads)
        self.local_to_global_attention = nn.MultiheadAttention(global_dim, num_heads)

        # Self-Attention layer for the concatenated features
        self.self_attention = nn.MultiheadAttention(global_dim * 2, num_heads)

        # Final linear layer to adjust output dimensions
        self.final_linear = nn.Linear(global_dim + local_dim, global_dim + local_dim)

        # Optional: Layer normalization
        self.layer_norm = nn.LayerNorm(global_dim + local_dim)

    def forward(self, global_features, local_features):
        # Cross-attention operations
        global_to_local_attn, _ = self.global_to_local_attention(local_features, global_features, global_features)
        local_to_global_attn, _ = self.local_to_global_attention(global_features, local_features, local_features)

        # Concatenate the cross-attention outputs
        concatenated_features = torch.cat((global_to_local_attn, local_to_global_attn), dim=2)

        # Self-attention to enhance the features further0
        enhanced_features, _ = self.self_attention(concatenated_features, concatenated_features, concatenated_features)

        # Linear layer to adjust final output dimensions
        final_output = self.final_linear(enhanced_features)

        # Optional: Layer normalization
        final_output = self.layer_norm(final_output)
        return final_output

class MultiHeadAttentionWei(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        # Q, K, V 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)  # 缩放因子 1/sqrt(d_k)

    def forward(self, query, key, value,
                attn_mask=None,  # 加性 mask: FloatTensor
                key_padding_mask=None,  # BoolTensor [B, Nk]
                weight_map=None):  # 新增参数：和 attn_weights 形状一样的权重 [B, H, Nq, Nk] 或 [B, Nq, Nk]（会自动广播）
        """
        新增参数:
            weight_map: Tensor, shape 为 [B, num_heads, Nq, Nk] 或 [B, Nq, Nk]（会自动广播到多头）
                        用于在 softmax 后对 attention map 进行逐元素加权（例如 mask-based gating）
                        建议值在 [0, 1] 区间，前景 ≈1，背景 ≈0.01
        """
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        B, Nq, C = query.shape
        _, Nk, _ = key.shape

        # 线性投影 + 分头
        q = self.q_proj(query).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Nq, D]
        k = self.k_proj(key).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Nk, D]
        v = self.v_proj(value).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Nk, D]

        # Scaled dot-product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, Nq, Nk]

        # 应用标准 mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Nq, Nk]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B, 1, Nq, Nk]
            attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, Nk]
            attn_scores = attn_scores.masked_fill(padding_mask, -1e9)

        # 第一次 softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, Nq, Nk]
        attn_weights = self.dropout(attn_weights)

        # ============ 新增：对 attention map 加外部权重 ============
        if weight_map is not None:
            # weight_map 支持两种常见形状：
            # 1. [B, H, Nq, Nk]  — 直接多头独立加权
            # 2. [B, Nq, Nk]     — 对所有 head 共享（会自动广播）
            # 3. [B, 1, Nq, Nk]   — 同上
            attn_weights = attn_weights * weight_map
        # ======================================================

        # 用加权后的 attn_weights 计算输出
        attn_output = torch.matmul(attn_weights, v)  # [B, H, Nq, D]

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Nq, C)

        # 输出投影
        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = attn_output.permute(1, 0, 2)

        final_weights = attn_weights.mean(dim=1)  # [B, Nq, Nk]
        return attn_output, final_weights

class CBAM_SpatialAttention(nn.Module):
    def __init__(self):
        super(CBAM_SpatialAttention, self).__init__()

        # 一个卷积层，输入通道是2（avg_pool + max_pool），输出通道是1（生成空间注意力图）
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)  # 这里可以根据需要调整kernel_size
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Step 1: Channel-wise max pooling and average pooling
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # 进行平均池化
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # 进行最大池化

        # Step 2: Concatenate the average and max pooled features along the channel dimension
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # Step 3: Pass the concatenated features through a single convolution to get attention map
        attention = self.conv(pooled)

        # Step 4: Apply sigmoid to obtain attention weights
        attention = self.sigmoid(attention)

        # Step 5: Apply the attention weights to the input feature map
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        # 一个卷积层，输入通道是2（avg_pool + max_pool），输出通道是1（生成空间注意力图）
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)  # 这里可以根据需要调整kernel_size
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Step 1: Channel-wise max pooling and average pooling
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # 进行平均池化
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # 进行最大池化

        # Step 2: Concatenate the average and max pooled features along the channel dimension
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # Step 3: Pass the concatenated features through a single convolution to get attention map
        attention = self.conv(pooled)

        # Step 4: Apply sigmoid to obtain attention weights
        attention = self.sigmoid(attention)

        # Step 5: Apply the attention weights to the input feature map
        return attention

class MScaleSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,  # 动态确定（见 forward）
            out_channels=1,
            kernel_size=3,
            padding=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_feat, mask_feat, avg_list, max_list):
        B, _, H, W = img_feat.shape
        cur_avg = torch.mean(mask_feat, dim=1, keepdim=True)
        cur_max, _ = torch.max(mask_feat, dim=1, keepdim=True)

        resized_feats = []

        for avg in avg_list:
            resized_feats.append(
                F.interpolate(avg, size=(H, W), mode='bilinear', align_corners=False)
            )
        resized_feats.append(cur_avg)

        for maxv in max_list:
            resized_feats.append(
                F.interpolate(maxv, size=(H, W), mode='bilinear', align_corners=False)
            )
        resized_feats.append(cur_max)
        attn_input = torch.cat(resized_feats, dim=1)

        attention = self.sigmoid(self.conv(attn_input))

        img_feat = img_feat * attention

        # avg_list = avg_list + [cur_avg.detach()]
        # max_list = max_list + [cur_max.detach()]
        avg_list = avg_list + [cur_avg]
        max_list = max_list + [cur_max]
        # plt.figure()
        # plt.imshow(attention[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        return img_feat, attention, avg_list, max_list

class SpatialAttentionWithCat(nn.Module):
    def __init__(self, dim):
        super(SpatialAttentionWithCat, self).__init__()

        # 一个卷积层，输入通道是2（avg_pool + max_pool），输出通道是1（生成空间注意力图）
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)  # 这里可以根据需要调整kernel_size
        self.mix_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, dim),
            nn.ReLU(),

            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, dim),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_feat, mask_feat):
        # Step 1: Channel-wise max pooling and average pooling
        avg_pool = torch.mean(mask_feat, dim=1, keepdim=True)  # 进行平均池化
        max_pool, _ = torch.max(mask_feat, dim=1, keepdim=True)  # 进行最大池化

        # Step 2: Concatenate the average and max pooled features along the channel dimension
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # Step 3: Pass the concatenated features through a single convolution to get attention map
        attention = self.conv(pooled)

        # Step 4: Apply sigmoid to obtain attention weights
        attention = self.sigmoid(attention)

        supplement = self.mix_conv(torch.cat([img_feat, mask_feat], dim=1))

        feat = img_feat * attention + supplement

        # Step 5: Apply the attention weights to the input feature map
        return feat, attention

class SpatialAttentionWithAdd(nn.Module):
    def __init__(self, dim):
        super(SpatialAttentionWithAdd, self).__init__()

        # 一个卷积层，输入通道是2（avg_pool + max_pool），输出通道是1（生成空间注意力图）
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)  # 这里可以根据需要调整kernel_size
        self.proj1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GroupNorm(32, dim),
            nn.ReLU()
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GroupNorm(32, dim),
            nn.ReLU()
        )
        self.mix_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, dim),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_feat, mask_feat):
        # Step 1: Channel-wise max pooling and average pooling
        avg_pool = torch.mean(mask_feat, dim=1, keepdim=True)  # 进行平均池化
        max_pool, _ = torch.max(mask_feat, dim=1, keepdim=True)  # 进行最大池化

        # Step 2: Concatenate the average and max pooled features along the channel dimension
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # Step 3: Pass the concatenated features through a single convolution to get attention map
        attention = self.conv(pooled)

        # Step 4: Apply sigmoid to obtain attention weights
        attention = self.sigmoid(attention)

        mix_feat = self.mix_conv(img_feat + self.proj1(mask_feat))
        supplement = self.proj2(mix_feat)

        feat = img_feat * attention + supplement

        # Step 5: Apply the attention weights to the input feature map
        return feat, attention

class MultiHeadMaskSpatialAttention(nn.Module):
    """
    多头骨骼掩膜引导的空间注意力模块
    输入：
        img_feat: B, C_img, H, W    - 主图像特征
        mask_feat: B, C_mask, H, W  - 骨骼掩膜/增强分支特征
    输出：
        attended_img: B, C_img, H, W   - 经过多头调制的图像特征
        attention_heads: B, num_heads, 1, H, W   - 各头的注意力图（可选，用于可视化）
    """

    def __init__(self, num_heads=8):
        super(MultiHeadMaskSpatialAttention, self).__init__()
        self.num_heads = num_heads

        # 每个头的3x3卷积：输入2通道（mean+max），输出1通道注意力图
        self.head_convs = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
            for _ in range(num_heads)
        ])

        # sigmoid 共享
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_feat, mask_feat):
        # Step 1: 将 mask_feat 按通道分成 num_heads 份（尽量均匀）
        mask_groups = torch.chunk(mask_feat, self.num_heads, dim=1)
        # 如果通道数不能完美整除，最后一组可能稍多，无需强制等分
        img_groups = torch.chunk(img_feat, self.num_heads, dim=1)

        # Step 2: 为每个mask组生成一个独立的注意力图
        attention_heads = []
        modulated_groups = []
        for i in range(self.num_heads):
            group = mask_groups[i]  # B, C_group, H, W

            # channel-wise mean 和 max pooling
            avg_pool = torch.mean(group, dim=1, keepdim=True)  # B,1,H,W
            max_pool, _ = torch.max(group, dim=1, keepdim=True)  # B,1,H,W

            # concat成2通道
            pooled = torch.cat([avg_pool, max_pool], dim=1)  # B,2,H,W

            # 通过该头的3x3卷积
            attn = self.head_convs[i](pooled)  # B,1,H,W

            # sigmoid激活
            attn = self.sigmoid(attn)
            # plt.figure()
            # plt.imshow(attn.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            # plt.show()

            attention_heads.append(attn)
            modulated_groups.append(img_groups[i] * attn)

        # Stack成 B, num_heads, 1, H, W
        # for i in range(self.num_heads):
        #     plt.figure()
        #     plt.imshow(attention_heads[i].squeeze().detach().cpu(), cmap='gray')
        #     plt.show()
        #     feat = modulated_groups[i].mean(dim=1)
        #     plt.figure()
        #     plt.imshow(feat.squeeze().detach().cpu(), cmap='gray')
        #     plt.show()
        attention_heads = torch.cat(attention_heads, dim=1)
        attended_img = torch.cat(modulated_groups, dim=1)

        return attended_img, attention_heads

class MultiHeadMaskSpatialAttentionGlobal(nn.Module):
    """
    多头骨骼掩膜引导的空间注意力模块
    输入：
        img_feat: B, C_img, H, W    - 主图像特征
        mask_feat: B, C_mask, H, W  - 骨骼掩膜/增强分支特征
    输出：
        attended_img: B, C_img, H, W   - 经过多头调制的图像特征
        attention_heads: B, num_heads, 1, H, W   - 各头的注意力图（可选，用于可视化）
    """

    def __init__(self, num_heads=8):
        super(MultiHeadMaskSpatialAttentionGlobal, self).__init__()
        self.num_heads = num_heads

        self.avg_reduce = nn.Conv2d(num_heads, 1, kernel_size=1, bias=False)
        self.max_reduce = nn.Conv2d(num_heads, 1, kernel_size=1, bias=False)
        # 每个头的3x3卷积：输入2通道（mean+max），输出1通道注意力图
        self.head_convs = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
            for _ in range(num_heads)
        ])

        # sigmoid 共享
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_feat, mask_feat):
        # Step 1: 将 mask_feat 和 img_feat 按通道分成 num_heads 组
        mask_groups = torch.chunk(mask_feat, self.num_heads, dim=1)
        img_groups = torch.chunk(img_feat, self.num_heads, dim=1)

        # Step 2: 为所有组分别计算avg和max（每个组的全局通道池化）
        group_avg = []
        group_max = []
        for group in mask_groups:
            avg = torch.mean(group, dim=1, keepdim=True)  # [B, 1, H, W]
            max_p, _ = torch.max(group, dim=1, keepdim=True)  # [B, 1, H, W]
            group_avg.append(avg)
            group_max.append(max_p)

        # Step 3: 所有组的avg和max分别cat起来（全局汇总）
        all_avg = torch.cat(group_avg, dim=1)  # [B, num_heads, H, W]
        all_max = torch.cat(group_max, dim=1)  # [B, num_heads, H, W]

        # Step 4: 用共享的1x1卷积分别浓缩成1通道（全局avg浓缩 & 全局max浓缩）
        global_avg_refined = self.avg_reduce(all_avg)  # [B, 1, H, W]
        global_max_refined = self.max_reduce(all_max)  # [B, 1, H, W]

        # Step 5: 两个浓缩结果cat成2通道，作为每个头的共享输入
        global_pooled = torch.cat([global_avg_refined, global_max_refined], dim=1)  # [B, 2, H, W]

        # Step 6: 每个头用独立conv处理这个共享的2通道输入，生成专属注意力图
        attention_heads = []
        modulated_groups = []
        for i in range(self.num_heads):
            attn = self.head_convs[i](global_pooled)  # [B, 1, H, W]
            attn = self.sigmoid(attn)
            attention_heads.append(attn)

            # 用该头的专属注意力图调制对应组的img特征
            modulated = img_groups[i] * attn
            modulated_groups.append(modulated)

        # 最终输出
        attended_img = torch.cat(modulated_groups, dim=1)  # [B, C, H, W]
        attention_heads = torch.cat(attention_heads, dim=1)  # [B, num_heads, 1, H, W]

        return attended_img, attention_heads

class MultiHeadMaskSpatialAttentionGlobal2(nn.Module):
    """
    多头骨骼掩膜引导的空间注意力模块
    输入：
        img_feat: B, C_img, H, W    - 主图像特征
        mask_feat: B, C_mask, H, W  - 骨骼掩膜/增强分支特征
    输出：
        attended_img: B, C_img, H, W   - 经过多头调制的图像特征
        attention_heads: B, num_heads, 1, H, W   - 各头的注意力图（可选，用于可视化）
    """

    def __init__(self, num_heads=8):
        super(MultiHeadMaskSpatialAttentionGlobal2, self).__init__()
        self.num_heads = num_heads

        # 每个头的3x3卷积：输入2通道（mean+max），输出1通道注意力图
        self.head_convs = nn.ModuleList([
            nn.Conv2d(4, 1, kernel_size=3, padding=1)
            for _ in range(num_heads)
        ])

        # sigmoid 共享
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_feat, mask_feat):
        # Step 1: 将 mask_feat 和 img_feat 按通道分成 num_heads 组
        mask_groups = torch.chunk(mask_feat, self.num_heads, dim=1)
        img_groups = torch.chunk(img_feat, self.num_heads, dim=1)

        global_avg = torch.mean(mask_feat, dim=1, keepdim=True)
        global_max, _ = torch.max(mask_feat, dim=1, keepdim=True)

        # Step 2: 为所有组分别计算 avg 和 max（每组）
        group_avg = []
        group_max = []
        for group in mask_groups:
            avg = torch.mean(group, dim=1, keepdim=True)  # [B,1,H,W]
            max_p, _ = torch.max(group, dim=1, keepdim=True)  # [B,1,H,W]
            group_avg.append(avg)
            group_max.append(max_p)

        # Step 4: 每个头将本头 avg/max 与全局 avg/max concat 作为输入
        attention_heads = []
        modulated_groups = []
        for i in range(self.num_heads):
            # 本头 avg/max
            head_avg = group_avg[i]
            head_max = group_max[i]
            # concat 本头 + 全局
            head_input = torch.cat([head_max, global_max, head_avg, global_avg], dim=1)  # [B,4,H,W]

            # 生成注意力图
            attn = self.head_convs[i](head_input)  # [B,1,H,W]
            attn = self.sigmoid(attn)
            attention_heads.append(attn)

            # 调制对应 img group
            modulated_groups.append(img_groups[i] * attn)

        # Step 5: 输出
        attended_img = torch.cat(modulated_groups, dim=1)  # [B,C,H,W]
        attention_heads = torch.cat(attention_heads, dim=1)  # [B,num_heads,1,H,W]

        return attended_img, attention_heads

class GroupSpatialAttention(nn.Module):
    """
    多头骨骼掩膜引导的空间注意力模块
    输入：
        img_feat: B, C_img, H, W    - 主图像特征
        mask_feat: B, C_mask, H, W  - 骨骼掩膜/增强分支特征
    输出：
        attended_img: B, C_img, H, W   - 经过多头调制的图像特征
        attention_heads: B, num_heads, 1, H, W   - 各头的注意力图（可选，用于可视化）
    """

    def __init__(self, num_heads=8):
        super(GroupSpatialAttention, self).__init__()
        self.num_heads = num_heads

        # 每个头的3x3卷积：输入2通道（mean+max），输出1通道注意力图
        self.head_convs = nn.ModuleList([
            nn.Conv2d(2 * (num_heads + 1), 1, kernel_size=3, padding=1, bias=False)
            for _ in range(num_heads)
        ])

        # sigmoid 共享
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_feat, mask_feat):
        # Step 1: 将 mask_feat 和 img_feat 按通道分成 num_heads 组
        mask_groups = torch.chunk(mask_feat, self.num_heads, dim=1)
        img_groups = torch.chunk(img_feat, self.num_heads, dim=1)

        global_avg = torch.mean(mask_feat, dim=1, keepdim=True)
        global_max, _ = torch.max(mask_feat, dim=1, keepdim=True)

        # Step 2: 为所有组分别计算 avg 和 max（每组）
        group_avg = [global_avg]
        group_max = [global_max]
        for group in mask_groups:
            avg = torch.mean(group, dim=1, keepdim=True)  # [B,1,H,W]
            max_p, _ = torch.max(group, dim=1, keepdim=True)  # [B,1,H,W]
            group_avg.append(avg)
            group_max.append(max_p)

        group_avg = torch.concat(group_avg, dim=1)
        group_max = torch.concat(group_max, dim=1)

        all_status = torch.concat([group_avg, group_max], dim=1)
        attn = self.head_convs(all_status)
        attn = self.sigmoid(attn)

        img_feat = img_feat * attn

        return img_feat, attn

class FCAFeatureExtractor(nn.Module):
    """
    FractMorph-inspired Fractional Cross-Attention (FCA) Feature Extractor
    输入: B, C, H, W (空间域特征)
    输出: B, C_out, H, W (增强后的空间域特征)

    四个并行分支:
    - α=0°: 空间域, 3x3 conv
    - α=45°: 半全局, 3x3 conv
    - α=90°: 频域, 1x1 conv
    - log-magnitude (α=90°): 频域 log 幅度, 1x1 conv
    """

    def __init__(self, in_channels, out_channels=None, alpha_list=[0.0, 0.25, 1.0], use_log_mag=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        # 卷积核大小根据域选择
        self.conv_kernels = nn.ModuleList()
        for alpha in alpha_list:
            if alpha == 1.0:  # 频域用 1x1
                conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
            else:  # 空间/过渡域用 3x3
                conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            self.conv_kernels.append(conv)

        # log-magnitude 分支的卷积
        if use_log_mag:
            self.log_mag_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # 最终融合的 pointwise conv
        total_branches = len(alpha_list) + (1 if use_log_mag else 0)
        self.fusion_conv = nn.Conv2d(in_channels * (total_branches + 1), self.out_channels, kernel_size=1)

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.alpha_list = alpha_list
        self.use_log_mag = use_log_mag

    def forward(self, x):
        B, C, H, W = x.shape
        skip = x  # skip connection

        branch_outputs = []

        # 分支 1-3: 普通 α 分支
        for i, alpha in enumerate(self.alpha_list):
            # FrFT (torch-frft 使用 alpha 在 [0,2]，其中 1.0 = 90°)
            X = frft(x, alpha, dim=(-2, -1))  # 复数域

            # Conv + ReLU (根据 alpha 选择 kernel)
            if alpha == 1.0:
                # 频域: 实部和虚部分别处理，或取幅度/相位（这里简化用实部示例）
                X_real = X.real
                X_conv = self.relu(self.conv_kernels[i](X_real))
                X = X_conv + 1j * X.imag  # 简化处理，实际可更精细
            else:
                # 空间/过渡域: 直接在实部卷积（假设输入实数）
                X_real = X.real if X.is_complex() else X
                X_conv = self.relu(self.conv_kernels[i](X_real))
                X = X_conv

            # 逆 FrFT 回空间域（必须！）
            enhanced = dfrft(X, -alpha, dim=(-2, -1)).real  # 取实部作为输出

            branch_outputs.append(enhanced)

        # log-magnitude 分支（可选）
        if self.use_log_mag:
            X = frft(x, 1.0, dim=(-2, -1))  # α=90°
            mag = torch.abs(X)
            log_mag = torch.log(1 + mag + 1e-8)  # log(1 + |X|)

            log_conv = self.relu(self.log_mag_conv(log_mag))
            log_mag_inv = torch.exp(log_conv) - 1  # 反转回幅度
            phase = torch.angle(X)  # 相位保持
            X_reconstructed = log_mag_inv * torch.exp(1j * phase)

            # 逆 FrFT
            enhanced_log = dfrft(X_reconstructed, -1.0, dim=(-2, -1)).real
            branch_outputs.append(enhanced_log)

        # 所有分支 + skip 拼接
        all_features = [skip] + branch_outputs
        fused = torch.cat(all_features, dim=1)  # B, C*(branches+1), H, W

        # 归一化 + pointwise conv 融合
        fused = self.bn(fused)
        out = self.fusion_conv(fused)

        return out

class FrFTGuidedSpatialAttention(nn.Module):
    """
    简化的 FrFT 多分支空间注意力生成器
    输入: mask_feat (B, C, H, W)
    输出: attention_map (B, 1, H, W)  # 最终融合的空间注意力图
    """

    def __init__(self, size, device='cuda', alpha_list=[0.0, 0.5, 1.0]):
        super().__init__()
        self.alpha_list = alpha_list

        self.frfts = [FrFTModule(order=0.5, device_option=device), FrFTModule(order=1, device_option=device)]

        # 每个分支的 3×3 conv（输入2通道 → 输出1通道）
        self.conv_for_0 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.branch_convs_imag = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=1, bias=False)
            for _ in range(len(alpha_list) - 1)
        ])
        self.branch_convs_real = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=1, bias=False)
            for _ in range(len(alpha_list) - 1)
        ])

        self.final_conv = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm([1, size, size])


    @staticmethod
    def _stack_real_imag(z: torch.Tensor) -> torch.Tensor:
        real_imag = torch.view_as_real(z)
        real_imag = real_imag.permute(0, 1, 4, 2, 3)
        return real_imag.reshape(z.size(0), 2 * z.size(1), *z.shape[2:])

    @staticmethod
    def _unstk_to_complex(pair: torch.Tensor) -> torch.Tensor:
        real, imag = pair.chunk(2, dim=1)
        return torch.complex(real, imag)

    def forward(self, img_feat, mask_feat):
        branch_attns = []

        avg_pool = torch.mean(mask_feat, dim=1, keepdim=True)
        max_pool, _ = torch.max(mask_feat, dim=1, keepdim=True)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv_for_0(pooled)
        branch_attns.append(attention)

        for i in range(2):
            # Step 1: FrFT 到当前域
            X = self.frfts[i].FrFT2D(mask_feat)

            real_part = X.real  # B,C,H,W
            imag_part = X.imag  # B,C,H,W

            # Step 3: 实部的池化统计
            real_avg = torch.mean(real_part, dim=1, keepdim=True)  # B,1,H,W
            real_max, _ = torch.max(real_part, dim=1, keepdim=True)  # B,1,H,W

            # Step 4: 虚部的池化统计
            imag_avg = torch.mean(imag_part, dim=1, keepdim=True)  # B,1,H,W
            imag_max, _ = torch.max(imag_part, dim=1, keepdim=True)  # B,1,H,W

            # Step 5: 拼接特征
            real_pooled = torch.cat([real_avg, real_max], dim=1)  # B,2,H,W
            imag_pooled = torch.cat([imag_avg, imag_max], dim=1)  # B,2,H,W
            # pooled = torch.cat([real_pooled, imag_pooled], dim=1)  # B,4,H,W

            # Step 6: 生成注意力图
            attn_imag = self.branch_convs_imag[i](imag_pooled)  # B,1,H,W
            attn_real = self.branch_convs_real[i](real_pooled)  # B,1,H,W
            # attn = self.sigmoid(attn)

            # Step 6: 逆 FrFT 回空间域（关键！确保位置对齐）
            attn_complex = torch.complex(attn_real, attn_imag)
            attn_spatial = self.frfts[i].IFrFT2D(attn_complex).real
            attn_spatial = self.norm(attn_spatial)

            # plt.figure()
            # plt.imshow(attn_spatial[0].squeeze(0).detach().cpu(), cmap='gray')
            # plt.show()

            branch_attns.append(attn_spatial)

        # Step 7: 融合所有分支的注意力图（简单平均）
        # fused_attn = torch.mean(torch.stack(branch_attns, dim=1), dim=1)  # B,1,H,W
        fused_attn = self.final_conv(torch.concat(branch_attns, dim=1))
        fused_attn = self.sigmoid(fused_attn)
        # plt.figure()
        # plt.imshow(fused_attn[0].squeeze(0).detach().cpu(), cmap='gray')
        # plt.show()

        return fused_attn * img_feat, fused_attn

class FrFTAttention(nn.Module):
    """
    简化的 FrFT 多分支空间注意力生成器
    输入: mask_feat (B, C, H, W)
    输出: attention_map (B, 1, H, W)  # 最终融合的空间注意力图
    """

    def __init__(self, dim, size, device='cuda'):
        super().__init__()
        self.dim = dim

        self.frft45 = FrFTModule(order=0.5, device_option=device)
        self.frft90 = FrFTModule(order=1.0, device_option=device)

        # 每个分支的 3×3 conv（输入2通道 → 输出1通道）
        self.conv_for_0 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_for_log = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.attn0 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.branch_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2, dim * 2, kernel_size=1),
                nn.ReLU(inplace=True)
            )
            for _ in range(2)
        ])
        self.branch_convs_attn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
            for _ in range(3)
        ])

        self.final_conv = nn.Conv2d(4, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        # self.norm = nn.LayerNorm([1, size, size])
        self.norm = nn.GroupNorm(32, dim)


    @staticmethod
    def _stack_real_imag(z: torch.Tensor) -> torch.Tensor:
        real_imag = torch.view_as_real(z)
        real_imag = real_imag.permute(0, 1, 4, 2, 3)
        return real_imag.reshape(z.size(0), 2 * z.size(1), *z.shape[2:])

    @staticmethod
    def _unstk_to_complex(pair: torch.Tensor) -> torch.Tensor:
        real, imag = pair.chunk(2, dim=1)
        return torch.complex(real, imag)

    def get_avg_max(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        return torch.cat([avg_pool, max_pool], dim=1)

    def forward(self, img_feat, mask_feat):
        branch_attns = []

        x0 = self.conv_for_0(mask_feat)
        attn0 = self.attn0(self.get_avg_max(x0))
        branch_attns.append(attn0)

        z45 = self.frft45.FrFT2D(mask_feat)
        x45 = self.branch_convs[0](self._stack_real_imag(z45))
        x45 = self.frft45.IFrFT2D(self._unstk_to_complex(x45)).real
        attn45 = self.branch_convs_attn[0](self.get_avg_max(x45))
        branch_attns.append(attn45)

        z90 = self.frft90.FrFT2D(mask_feat)
        x90 = self.branch_convs[1](self._stack_real_imag(z90))
        x90 = self.frft90.IFrFT2D(self._unstk_to_complex(x90)).real
        attn90 = self.branch_convs_attn[1](self.get_avg_max(x90))
        branch_attns.append(attn90)

        log_mag = self.conv_for_log(torch.log1p(torch.abs(z90)))
        mag = torch.expm1(log_mag)
        z_log = torch.polar(mag, torch.angle(z90))
        log = self.frft90.IFrFT2D(z_log).real
        attn_log = self.branch_convs_attn[2](self.get_avg_max(log))
        branch_attns.append(attn_log)

        # Step 7: 融合所有分支的注意力图（简单平均）
        # fused_attn = torch.mean(torch.stack(branch_attns, dim=1), dim=1)  # B,1,H,W
        fused_attn = self.final_conv(torch.concat(branch_attns, dim=1))
        fused_attn = self.sigmoid(fused_attn)

        plt.figure()
        plt.imshow(fused_attn[0].squeeze(0).cpu(), cmap='gray')
        plt.show()

        return fused_attn * img_feat, fused_attn

class FrFTMaxAttention(nn.Module):
    """
    简化的 FrFT 多分支空间注意力生成器
    输入: mask_feat (B, C, H, W)
    输出: attention_map (B, 1, H, W)  # 最终融合的空间注意力图
    """

    def __init__(self, dim, size, device='cuda'):
        super().__init__()
        self.dim = dim

        self.frft45 = FrFTModule(order=0.5, device_option=device)
        self.frft90 = FrFTModule(order=1.0, device_option=device)

        # 每个分支的 3×3 conv（输入2通道 → 输出1通道）
        self.conv_for_0 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_for_log = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.branch_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2, dim * 2, kernel_size=1),
                nn.ReLU(inplace=True)
            )
            for _ in range(2)
        ])
        self.conv_max = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.conv_avg = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        # self.norm = nn.LayerNorm([dim, size, size])
        self.norm = nn.GroupNorm(32, dim)


    @staticmethod
    def _stack_real_imag(z: torch.Tensor) -> torch.Tensor:
        real_imag = torch.view_as_real(z)
        real_imag = real_imag.permute(0, 1, 4, 2, 3)
        return real_imag.reshape(z.size(0), 2 * z.size(1), *z.shape[2:])

    @staticmethod
    def _unstk_to_complex(pair: torch.Tensor) -> torch.Tensor:
        real, imag = pair.chunk(2, dim=1)
        return torch.complex(real, imag)

    def get_avg_max(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        return torch.cat([avg_pool, max_pool], dim=1)

    def get_avg(self, x):
        return torch.mean(x, dim=1, keepdim=True)

    def get_max(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        return max_pool

    def forward(self, img_feat, mask_feat):
        branch_max = []
        branch_avg = []

        x0 = self.conv_for_0(mask_feat)
        x0 = self.norm(x0)
        branch_max.append(self.get_max(x0))
        branch_avg.append(self.get_avg(x0))

        z45 = self.frft45.FrFT2D(mask_feat)
        x45 = self.branch_convs[0](self._stack_real_imag(z45))
        x45 = self.frft45.IFrFT2D(self._unstk_to_complex(x45)).real
        x45 = self.norm(x45)
        branch_max.append(self.get_max(x45))
        branch_avg.append(self.get_avg(x45))

        z90 = self.frft90.FrFT2D(mask_feat)
        x90 = self.branch_convs[1](self._stack_real_imag(z90))
        x90 = self.frft90.IFrFT2D(self._unstk_to_complex(x90)).real
        x90 = self.norm(x90)
        branch_max.append(self.get_max(x90))
        branch_avg.append(self.get_avg(x90))

        log_mag = self.conv_for_log(torch.log1p(torch.abs(z90)))
        mag = torch.expm1(log_mag)
        z_log = torch.polar(mag, torch.angle(z90))
        log = self.frft90.IFrFT2D(z_log).real
        log = self.norm(log)
        branch_max.append(self.get_max(log))
        branch_avg.append(self.get_avg(log))

        # Step 7: 融合所有分支的注意力图（简单平均）
        # fused_attn = torch.mean(torch.stack(branch_attns, dim=1), dim=1)  # B,1,H,W
        max_feat = self.conv_max(torch.concat(branch_max, dim=1))
        avg_feat = self.conv_avg(torch.concat(branch_avg, dim=1))
        fused_attn = max_feat + avg_feat
        fused_attn = self.sigmoid(fused_attn)

        # plt.figure()
        # plt.imshow(fused_attn[0].squeeze(0).cpu(), cmap='gray')
        # plt.show()

        return fused_attn * img_feat, fused_attn

class MultiFrFTGuidedSpatialAttention(nn.Module):
    """
    多分组 FrFT 引导的空间注意力模块
    - 输入: img_feat (B, C_img, H, W), mask_feat (B, C_mask, H, W)
    - 把 mask_feat 通道分成 num_groups 组
    - 每组独立调用 FrFTGuidedSpatialAttention 生成注意力
    - 融合所有组的注意力图，并调制 img_feat 的对应通道组
    输出: attended_img (B, C_img, H, W), fused_attention (B, 1, H, W)
    """

    def __init__(self, in_channel, size, device='cuda', num_groups=8):
        super().__init__()
        self.num_groups = num_groups

        self.group_size_mask = in_channel // num_groups
        self.group_size_img = in_channel // num_groups

        # 每个分组独立一个 FrFTGuidedSpatialAttention 实例（可共享权重或独立学习）
        self.frft_attns = nn.ModuleList([
            FrFTGuidedSpatialAttention(size, device=device)  # 用你原来的类（或自定义）
            for _ in range(num_groups)
        ])

    def forward(self, img_feat, mask_feat):
        # 把 mask_feat 通道分组
        mask_groups = torch.chunk(mask_feat, self.num_groups, dim=1)
        # 把 img_feat 通道分组（用于后续调制）
        img_groups = torch.chunk(img_feat, self.num_groups, dim=1)
        modulated_groups = []
        for i in range(self.num_groups):
            # Step 1: 每组 mask 调用 FrFTGuidedSpatialAttention
            img_feat, _ = self.frft_attns[i](img_groups[i], mask_groups[i])  # B, 1, H, W
            modulated_groups.append(img_feat)

        # Step 3: 拼接所有调制后的 img 组
        attended_img = torch.cat(modulated_groups, dim=1)  # B, C_img, H, W

        return attended_img, attended_img  # 返回增强特征 + 融合注意力图

class PoseRegressorCat(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = SpatialGate(64)
        self.space_attention2 = SpatialGate(64)
        self.space_attention3 = SpatialGate(128)
        self.space_attention4 = SpatialGate(256)

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        # pred_mask = self.decoder(img_feat)
        img_feat = self.global_pool(img_feat)
        rot = self.rot_regression(img_feat)
        xyz = self.xyz_regression(img_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorCat2(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = SpatialGate(64)
        self.space_attention2 = SpatialGate(64)
        self.space_attention3 = SpatialGate(128)
        self.space_attention4 = SpatialGate(256)

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        img_feat = self.layer1(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)
        mask_feat = self.layer_mask1(mask_feat)

        g2 = self.space_attention2(mask_feat)
        img_feat = img_feat * g2
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)

        g3 = self.space_attention3(mask_feat)
        img_feat = img_feat * g3
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)

        g4 = self.space_attention4(mask_feat)
        img_feat = img_feat * g4
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)

        # pred_mask = self.decoder(img_feat)
        img_feat = self.global_pool(img_feat)
        rot = self.rot_regression(img_feat)
        xyz = self.xyz_regression(img_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorCatCBAM(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = CBAM_SpatialAttention()
        self.space_attention2 = CBAM_SpatialAttention()
        self.space_attention3 = CBAM_SpatialAttention()
        self.space_attention4 = CBAM_SpatialAttention()

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        # pred_mask = self.decoder(img_feat)
        img_feat = self.global_pool(img_feat)
        rot = self.rot_regression(img_feat)
        xyz = self.xyz_regression(img_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorAttn(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        self.backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': self.backbone.layer1,
            'layer2': self.backbone.layer2,
            'layer3': self.backbone.layer3,
            'layer4': self.backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.img_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),  # 推荐 num_groups=32
            nn.ReLU(inplace=True)
        )
        self.mask_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),
            nn.ReLU(inplace=True)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,  # 可根據 feat_dim 調整
            dropout=0.1,
            batch_first=True
        )
        self.pos_embedding = nn.Parameter(torch.empty(1, 64, 512).normal_(std=0.02))

    def forward(self, x, mask):
        img_feat = self.backbone.forward_features(x)
        mask_feat = self.backbone2.forward_features(mask)

        img_feat = self.img_conv_block(img_feat)
        mask_feat = self.mask_conv_block(mask_feat)

        B, C, H, W = img_feat.shape
        img_tokens = img_feat.reshape(B, C, H * W).permute(0, 2, 1)
        mask_tokens = mask_feat.reshape(B, C, H * W).permute(0, 2, 1)
        img_tokens = img_tokens + self.pos_embedding
        mask_tokens = mask_tokens + self.pos_embedding

        attn_output, attn_weights = self.cross_attn(query=mask_tokens, key=img_tokens, value=img_tokens)
        attn_output = attn_output + mask_tokens
        attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W)

        aw = attn_weights[0]
        plt.figure()
        plt.imshow(aw.detach().cpu())
        plt.show()

        cross_feat = F.adaptive_avg_pool2d(attn_output, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorAttnWei(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        self.backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': self.backbone.layer1,
            'layer2': self.backbone.layer2,
            'layer3': self.backbone.layer3,
            'layer4': self.backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.img_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),  # 推荐 num_groups=32
            nn.ReLU(inplace=True)
        )
        self.mask_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),
            nn.ReLU(inplace=True)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,  # 可根據 feat_dim 調整
            dropout=0.1,
            batch_first=True
        )
        # self.pos_embedding = nn.Parameter(torch.empty(1, 64, 512).normal_(std=0.02))
        pos_embed = get_2d_sine_pos_embed(embed_dim=512)
        self.register_buffer('pos_embedding', pos_embed)
        self.norm = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(  # 轻量 FeedForward Network
            nn.Linear(feat_dim, feat_dim * 4),
            nn.ReLU(),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(0.1)
        )

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

    def forward(self, x, mask):
        img_feat = self.backbone.forward_features(x)
        mask_feat = self.backbone2.forward_features(mask)

        img_feat = self.img_conv_block(img_feat)
        mask_feat = self.mask_conv_block(mask_feat)

        B, C, H, W = img_feat.shape
        img_tokens = img_feat.reshape(B, C, H * W).permute(0, 2, 1)
        mask_tokens = mask_feat.reshape(B, C, H * W).permute(0, 2, 1)
        img_tokens = img_tokens + self.pos_embedding
        mask_tokens = mask_tokens + self.pos_embedding

        mask_wei = self.compute_mask_weights(mask, 32)
        mask_wei += 0.1
        attn_output, attn_weights = self.cross_attn(query=mask_tokens, key=img_tokens, value=img_tokens)
        attn_output = attn_output * mask_wei.unsqueeze(-1)
        attn_output = attn_output + mask_tokens
        attn_output = self.norm(attn_output)
        attn_output = attn_output + self.ffn(attn_output)
        attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W)

        # aw = attn_weights[0]
        # plt.figure()
        # plt.imshow(aw.detach().cpu())
        # plt.show()

        cross_feat = F.adaptive_avg_pool2d(attn_output, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorAttnWei2(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        self.backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': self.backbone.layer1,
            'layer2': self.backbone.layer2,
            'layer3': self.backbone.layer3,
            'layer4': self.backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.img_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),  # 推荐 num_groups=32
            nn.ReLU(inplace=True)
        )
        self.mask_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),
            nn.ReLU(inplace=True)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,  # 可根據 feat_dim 調整
            dropout=0.1,
            batch_first=True
        )
        # self.pos_embedding = nn.Parameter(torch.empty(1, 64, 512).normal_(std=0.02))
        pos_embed = get_2d_sine_pos_embed(embed_dim=512)
        self.register_buffer('pos_embedding', pos_embed)
        self.norm = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(  # 轻量 FeedForward Network
            nn.Linear(feat_dim, feat_dim * 4),
            nn.ReLU(),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(0.1)
        )

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

    def forward(self, x, mask):
        img_feat = self.backbone.forward_features(x)
        mask_feat = self.backbone2.forward_features(mask)

        img_feat = self.img_conv_block(img_feat)
        mask_feat = self.mask_conv_block(mask_feat)

        B, C, H, W = img_feat.shape
        img_tokens = img_feat.reshape(B, C, H * W).permute(0, 2, 1)
        mask_tokens = mask_feat.reshape(B, C, H * W).permute(0, 2, 1)
        img_tokens = img_tokens + self.pos_embedding
        mask_tokens = mask_tokens + self.pos_embedding

        # mask_wei = self.compute_mask_weights(mask, 32)
        # mask_wei += 0.1
        attn_output, attn_weights = self.cross_attn(query=mask_tokens, key=img_tokens, value=img_tokens)
        # attn_output = attn_output * mask_wei.unsqueeze(-1)
        # attn_output = attn_output + img_tokens
        # attn_output = self.norm(attn_output)
        attn_output = self.norm(self.ffn(attn_output))
        attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W)

        # aw = attn_weights[0]
        # plt.figure()
        # plt.imshow(aw.detach().cpu())
        # plt.show()

        cross_feat = F.adaptive_avg_pool2d(attn_output, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorAttnNoWei(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = CBAM_SpatialAttention()
        self.space_attention2 = CBAM_SpatialAttention()
        self.space_attention3 = CBAM_SpatialAttention()
        self.space_attention4 = CBAM_SpatialAttention()

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.img_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),  # 推荐 num_groups=32
            nn.ReLU(inplace=True)
        )
        self.mask_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),
            nn.ReLU(inplace=True)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,  # 可根據 feat_dim 調整
            dropout=0.1,
            batch_first=True
        )
        # self.pos_embedding = nn.Parameter(torch.empty(1, 64, 512).normal_(std=0.02))
        pos_embed = get_2d_sine_pos_embed(embed_dim=512)
        self.register_buffer('pos_embedding', pos_embed)
        self.norm = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(  # 轻量 FeedForward Network
            nn.Linear(feat_dim, feat_dim * 4),
            nn.ReLU(),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        img_feat = self.img_conv_block(img_feat)
        mask_feat = self.mask_conv_block(mask_feat)

        B, C, H, W = img_feat.shape
        img_tokens = img_feat.reshape(B, C, H * W).permute(0, 2, 1)
        mask_tokens = mask_feat.reshape(B, C, H * W).permute(0, 2, 1)
        img_tokens = img_tokens + self.pos_embedding
        mask_tokens = mask_tokens + self.pos_embedding

        # mask_wei = self.compute_mask_weights(mask, 32)
        # mask_wei += 0.1
        attn_output, attn_weights = self.cross_attn(query=mask_tokens, key=img_tokens, value=img_tokens)
        # attn_output = attn_output * mask_wei.unsqueeze(-1)
        # attn_output = attn_output + img_tokens
        # attn_output = self.norm(attn_output)
        attn_output = self.norm(self.ffn(attn_output))
        attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W)

        aw = attn_weights[0]
        plt.figure()
        plt.imshow(aw.detach().cpu())
        plt.show()

        cross_feat = F.adaptive_avg_pool2d(attn_output, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorAttnCBAM(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = CBAM_SpatialAttention()
        self.space_attention2 = CBAM_SpatialAttention()
        self.space_attention3 = CBAM_SpatialAttention()
        self.space_attention4 = CBAM_SpatialAttention()

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.img_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),  # 推荐 num_groups=32
            nn.ReLU(inplace=True)
        )
        self.mask_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),
            nn.ReLU(inplace=True)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,  # 可根據 feat_dim 調整
            dropout=0.1,
            batch_first=True
        )
        # self.pos_embedding = nn.Parameter(torch.empty(1, 64, 512).normal_(std=0.02))
        pos_embed = get_2d_sine_pos_embed(embed_dim=512)
        self.register_buffer('pos_embedding', pos_embed)
        self.norm = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(  # 轻量 FeedForward Network
            nn.Linear(feat_dim, feat_dim * 4),
            nn.ReLU(),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        img_feat = self.img_conv_block(img_feat)
        mask_feat = self.mask_conv_block(mask_feat)

        B, C, H, W = img_feat.shape
        img_tokens = img_feat.reshape(B, C, H * W).permute(0, 2, 1)
        mask_tokens = mask_feat.reshape(B, C, H * W).permute(0, 2, 1)
        img_tokens = img_tokens + self.pos_embedding
        mask_tokens = mask_tokens + self.pos_embedding

        mask_wei = compute_mask_weights(mask, 32)
        mask_wei += 0.1
        attn_output, attn_weights = self.cross_attn(query=mask_tokens, key=img_tokens, value=img_tokens)
        attn_output = attn_output * mask_wei.unsqueeze(-1)
        attn_output = attn_output + mask_tokens
        attn_output = self.norm(attn_output)
        attn_output = self.norm(attn_output + self.ffn(attn_output))
        attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W)

        # aw = attn_weights[0]
        # print(aw.max())
        # print(aw.min())
        # plt.figure()
        # plt.imshow(aw.detach().cpu())
        # plt.show()

        cross_feat = F.adaptive_avg_pool2d(attn_output, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorAttnWeiNoAdd(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = CBAM_SpatialAttention()
        self.space_attention2 = CBAM_SpatialAttention()
        self.space_attention3 = CBAM_SpatialAttention()
        self.space_attention4 = CBAM_SpatialAttention()

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.img_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),  # 推荐 num_groups=32
            nn.ReLU(inplace=True)
        )
        self.mask_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),
            nn.ReLU(inplace=True)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,  # 可根據 feat_dim 調整
            dropout=0.1,
            batch_first=True
        )
        # self.pos_embedding = nn.Parameter(torch.empty(1, 64, 512).normal_(std=0.02))
        pos_embed = get_2d_sine_pos_embed(embed_dim=512)
        self.register_buffer('pos_embedding', pos_embed)
        self.norm = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(  # 轻量 FeedForward Network
            nn.Linear(feat_dim, feat_dim * 4),
            nn.ReLU(),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        img_feat = self.img_conv_block(img_feat)
        mask_feat = self.mask_conv_block(mask_feat)

        B, C, H, W = img_feat.shape
        img_tokens = img_feat.reshape(B, C, H * W).permute(0, 2, 1)
        mask_tokens = mask_feat.reshape(B, C, H * W).permute(0, 2, 1)
        img_tokens = img_tokens + self.pos_embedding
        mask_tokens = mask_tokens + self.pos_embedding

        mask_wei = compute_mask_weights(mask, 32)
        mask_wei += 0.1
        attn_output, attn_weights = self.cross_attn(query=mask_tokens, key=img_tokens, value=img_tokens)
        attn_output = attn_output * mask_wei.unsqueeze(-1)
        # attn_output = attn_output + img_tokens
        # attn_output = self.norm(attn_output)
        attn_output = self.norm(attn_output + self.ffn(attn_output))
        attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W)

        # aw = attn_weights[0]
        # plt.figure()
        # plt.imshow(aw.detach().cpu())
        # plt.show()

        cross_feat = F.adaptive_avg_pool2d(attn_output, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorCross(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = CBAM_SpatialAttention()
        self.space_attention2 = CBAM_SpatialAttention()
        self.space_attention3 = CBAM_SpatialAttention()
        self.space_attention4 = CBAM_SpatialAttention()

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.img_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),  # 推荐 num_groups=32
            nn.ReLU(inplace=True)
        )
        self.mask_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),
            nn.ReLU(inplace=True)
        )
        self.cross_attn = EnhancedSemanticAttentionModule(512, 512)

        # self.pos_embedding = nn.Parameter(torch.empty(1, 64, 512).normal_(std=0.02))
        pos_embed = get_2d_sine_pos_embed(embed_dim=512)
        self.register_buffer('pos_embedding', pos_embed)
        self.norm = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(  # 轻量 FeedForward Network
            nn.Linear(feat_dim, feat_dim * 4),
            nn.ReLU(),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        img_feat = self.img_conv_block(img_feat)
        mask_feat = self.mask_conv_block(mask_feat)

        B, C, H, W = img_feat.shape
        img_tokens = img_feat.reshape(B, C, H * W).permute(0, 2, 1)
        mask_tokens = mask_feat.reshape(B, C, H * W).permute(0, 2, 1)
        img_tokens = img_tokens + self.pos_embedding
        mask_tokens = mask_tokens + self.pos_embedding

        attn_output = self.cross_attn(img_tokens, mask_tokens)
        attn_output = attn_output.permute(0, 2, 1).reshape(B, 2 * C, H, W)

        # aw = attn_weights[0]
        # plt.figure()
        # plt.imshow(aw.detach().cpu())
        # plt.show()

        cross_feat = F.adaptive_avg_pool2d(attn_output, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorMapWei(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = SpatialAttention()
        self.space_attention2 = SpatialAttention()
        self.space_attention3 = SpatialAttention()
        self.space_attention4 = SpatialAttention()

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.img_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),  # 推荐 num_groups=32
            nn.ReLU(inplace=True)
        )
        self.mask_conv_block = nn.Sequential(
            nn.Conv2d(
                feat_dim, feat_dim,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=32, num_channels=feat_dim),
            nn.ReLU(inplace=True)
        )
        self.cross_attn = MultiHeadAttentionWei(
            embed_dim=512,
            num_heads=8,  # 可根據 feat_dim 調整
            dropout=0.1,
            batch_first=True
        )
        # self.pos_embedding = nn.Parameter(torch.empty(1, 64, 512).normal_(std=0.02))
        pos_embed = get_2d_sine_pos_embed(embed_dim=512)
        self.register_buffer('pos_embedding', pos_embed)
        self.norm = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(  # 轻量 FeedForward Network
            nn.Linear(feat_dim, feat_dim * 4),
            nn.ReLU(),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        B, C, H, W = img_feat.shape
        img_tokens = img_feat.reshape(B, C, H * W).permute(0, 2, 1)
        mask_tokens = mask_feat.reshape(B, C, H * W).permute(0, 2, 1)
        img_tokens = img_tokens + self.pos_embedding
        mask_tokens = mask_tokens + self.pos_embedding

        mask_wei = compute_mask_weights2d(mask, 32)
        attn_output, attn_weights = self.cross_attn(query=mask_tokens, key=img_tokens, value=img_tokens, weight_map=mask_wei)
        # attn_output = attn_output + mask_tokens
        # attn_output = self.norm(attn_output)
        attn_output = self.norm(attn_output + self.ffn(attn_output))
        attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W)

        # aw = attn_weights[0]
        # print(aw.max())
        # print(aw.min())
        # plt.figure()
        # plt.imshow(aw.detach().cpu())
        # plt.show()

        cross_feat = F.adaptive_avg_pool2d(attn_output, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorCoe(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = CBAM_SpatialAttention()
        self.space_attention2 = CBAM_SpatialAttention()
        self.space_attention3 = CBAM_SpatialAttention()
        self.space_attention4 = CBAM_SpatialAttention()

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        # plt.figure()
        # plt.imshow(g1.mean(dim=1), detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()

        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorCoe2(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = CBAM_SpatialAttention()
        self.space_attention2 = CBAM_SpatialAttention()
        self.space_attention3 = CBAM_SpatialAttention()
        self.space_attention4 = CBAM_SpatialAttention()

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net1 = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

        self.gate_net2 = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g1 = self.gate_net1(feat_cat)
        g2 = self.gate_net2(feat_cat)

        feat_final = g1 * img_feat + g2 * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorCoeDeco(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = CBAM_SpatialAttention()
        self.space_attention2 = CBAM_SpatialAttention()
        self.space_attention3 = CBAM_SpatialAttention()
        self.space_attention4 = CBAM_SpatialAttention()

        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        pred_mask = self.decoder(img_feat)

        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorFieldDeco(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = SpatialAttention()
        self.space_attention2 = SpatialAttention()
        self.space_attention3 = SpatialAttention()
        self.space_attention4 = SpatialAttention()

        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.fusion1 = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, feat_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, feat_dim),
            nn.ReLU(inplace=True)
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, feat_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        pred_mask = self.decoder(img_feat)
        complex_feat = self.fusion1(torch.concat([img_feat, mask_feat], dim=1))
        feat_final = self.fusion2(torch.concat([img_feat, mask_feat + complex_feat], dim=1))

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorField(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = CBAM_SpatialAttention()
        self.space_attention2 = CBAM_SpatialAttention()
        self.space_attention3 = CBAM_SpatialAttention()
        self.space_attention4 = CBAM_SpatialAttention()

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.fusion1 = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, feat_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, feat_dim),
            nn.ReLU(inplace=True)
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, feat_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        complex_feat = self.fusion1(torch.concat([img_feat, mask_feat], dim=1))
        feat_final = self.fusion2(torch.concat([img_feat, mask_feat + complex_feat], dim=1))

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorCatOnly(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = CBAM_SpatialAttention()
        self.space_attention2 = CBAM_SpatialAttention()
        self.space_attention3 = CBAM_SpatialAttention()
        self.space_attention4 = CBAM_SpatialAttention()

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.fusion = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, feat_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        complex_feat = self.fusion(torch.concat([img_feat, mask_feat], dim=1))

        cross_feat = F.adaptive_avg_pool2d(complex_feat, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorCoeSp(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = SpatialAttention()
        self.space_attention2 = SpatialAttention()
        self.space_attention3 = SpatialAttention()
        self.space_attention4 = SpatialAttention()

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorNoAttn(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        # self.space_attention1 = MultiHeadMaskSpatialAttention(8)
        # self.space_attention2 = MultiHeadMaskSpatialAttention(8)
        # self.space_attention3 = MultiHeadMaskSpatialAttention(8)
        # self.space_attention4 = MultiHeadMaskSpatialAttention(8)
        # self.space_attention1 = MultiHeadMaskSpatialAttention2(8)
        # self.space_attention2 = MultiHeadMaskSpatialAttention2(8)
        # self.space_attention3 = MultiHeadMaskSpatialAttention2(8)
        # self.space_attention4 = MultiHeadMaskSpatialAttention2(8)


        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)

        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)

        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)

        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)

        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorMultiSp(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = MultiHeadMaskSpatialAttention(8)
        self.space_attention2 = MultiHeadMaskSpatialAttention(8)
        self.space_attention3 = MultiHeadMaskSpatialAttention(8)
        self.space_attention4 = MultiHeadMaskSpatialAttention(8)
        # self.space_attention1 = MultiHeadMaskSpatialAttention2(8)
        # self.space_attention2 = MultiHeadMaskSpatialAttention2(8)
        # self.space_attention3 = MultiHeadMaskSpatialAttention2(8)
        # self.space_attention4 = MultiHeadMaskSpatialAttention2(8)


        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        img_feat, _ = self.space_attention1(img_feat, mask_feat)
        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)

        img_feat, _ = self.space_attention2(img_feat, mask_feat)
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)

        img_feat, _ = self.space_attention3(img_feat, mask_feat)
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)

        img_feat, _ = self.space_attention4(img_feat, mask_feat)
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)

        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorMultiSpDeco(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = MultiHeadMaskSpatialAttention(8)
        self.space_attention2 = MultiHeadMaskSpatialAttention(8)
        self.space_attention3 = MultiHeadMaskSpatialAttention(8)
        self.space_attention4 = MultiHeadMaskSpatialAttention(8)

        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        img_feat, _ = self.space_attention1(img_feat, mask_feat)
        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)

        img_feat, _ = self.space_attention2(img_feat, mask_feat)
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)

        img_feat, _ = self.space_attention3(img_feat, mask_feat)
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)

        img_feat, _ = self.space_attention4(img_feat, mask_feat)
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)

        pred_mask = self.decoder(img_feat)
        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorMultiSpGlobalDeco(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = MultiHeadMaskSpatialAttentionGlobal(4)
        self.space_attention2 = MultiHeadMaskSpatialAttentionGlobal(4)
        self.space_attention3 = MultiHeadMaskSpatialAttentionGlobal(4)
        self.space_attention4 = MultiHeadMaskSpatialAttentionGlobal(4)

        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        img_feat, _ = self.space_attention1(img_feat, mask_feat)
        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)

        img_feat, _ = self.space_attention2(img_feat, mask_feat)
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)

        img_feat, _ = self.space_attention3(img_feat, mask_feat)
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)

        img_feat, _ = self.space_attention4(img_feat, mask_feat)
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)

        pred_mask = self.decoder(img_feat)
        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorMultiSpGlobalDeco2(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = MultiHeadMaskSpatialAttentionGlobal2(4)
        self.space_attention2 = MultiHeadMaskSpatialAttentionGlobal2(4)
        self.space_attention3 = MultiHeadMaskSpatialAttentionGlobal2(4)
        self.space_attention4 = MultiHeadMaskSpatialAttentionGlobal2(4)

        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        img_feat, _ = self.space_attention1(img_feat, mask_feat)
        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)

        img_feat, _ = self.space_attention2(img_feat, mask_feat)
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)

        img_feat, _ = self.space_attention3(img_feat, mask_feat)
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)

        img_feat, _ = self.space_attention4(img_feat, mask_feat)
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)

        pred_mask = self.decoder(img_feat)
        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorGroupSpDeco(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = MultiHeadMaskSpatialAttentionGlobal2(4)
        self.space_attention2 = MultiHeadMaskSpatialAttentionGlobal2(4)
        self.space_attention3 = MultiHeadMaskSpatialAttentionGlobal2(4)
        self.space_attention4 = MultiHeadMaskSpatialAttentionGlobal2(4)

        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        img_feat, _ = self.space_attention1(img_feat, mask_feat)
        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)

        img_feat, _ = self.space_attention2(img_feat, mask_feat)
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)

        img_feat, _ = self.space_attention3(img_feat, mask_feat)
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)

        img_feat, _ = self.space_attention4(img_feat, mask_feat)
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)

        pred_mask = self.decoder(img_feat)
        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorFrFT(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = FrFTGuidedSpatialAttention(64)
        self.space_attention2 = FrFTGuidedSpatialAttention(64)
        self.space_attention3 = FrFTGuidedSpatialAttention(32)
        self.space_attention4 = FrFTGuidedSpatialAttention(16)
        # self.space_attention1 = MultiHeadMaskSpatialAttention2(8)
        # self.space_attention2 = MultiHeadMaskSpatialAttention2(8)
        # self.space_attention3 = MultiHeadMaskSpatialAttention2(8)
        # self.space_attention4 = MultiHeadMaskSpatialAttention2(8)


        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        img_feat, _ = self.space_attention1(img_feat, mask_feat)
        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)

        img_feat, _ = self.space_attention2(img_feat, mask_feat)
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)

        img_feat, _ = self.space_attention3(img_feat, mask_feat)
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)

        img_feat, _ = self.space_attention4(img_feat, mask_feat)
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)

        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorFrFTDeco(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            device='cuda',
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = FrFTGuidedSpatialAttention(64, device=device)
        self.space_attention2 = FrFTGuidedSpatialAttention(64, device=device)
        self.space_attention3 = FrFTGuidedSpatialAttention(32, device=device)
        self.space_attention4 = FrFTGuidedSpatialAttention(16, device=device)
        # self.space_attention1 = MultiHeadMaskSpatialAttention2(8)
        # self.space_attention2 = MultiHeadMaskSpatialAttention2(8)
        # self.space_attention3 = MultiHeadMaskSpatialAttention2(8)
        # self.space_attention4 = MultiHeadMaskSpatialAttention2(8)


        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        img_feat, _ = self.space_attention1(img_feat, mask_feat)
        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)

        img_feat, _ = self.space_attention2(img_feat, mask_feat)
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)

        img_feat, _ = self.space_attention3(img_feat, mask_feat)
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)

        img_feat, _ = self.space_attention4(img_feat, mask_feat)
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)

        pred_mask = self.decoder(img_feat)
        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorMultiFrFTDeco(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            device='cuda',
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = MultiFrFTGuidedSpatialAttention(64, 64, device=device)
        self.space_attention2 = MultiFrFTGuidedSpatialAttention(64, 64, device=device)
        self.space_attention3 = MultiFrFTGuidedSpatialAttention(128, 32, device=device)
        self.space_attention4 = MultiFrFTGuidedSpatialAttention(256, 16, device=device)


        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        img_feat, _ = self.space_attention1(img_feat, mask_feat)
        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)

        img_feat, _ = self.space_attention2(img_feat, mask_feat)
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)

        img_feat, _ = self.space_attention3(img_feat, mask_feat)
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)

        img_feat, _ = self.space_attention4(img_feat, mask_feat)
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)

        pred_mask = self.decoder(img_feat)
        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorSp(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = SpatialAttention()
        self.space_attention2 = SpatialAttention()
        self.space_attention3 = SpatialAttention()
        self.space_attention4 = SpatialAttention()

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        # feat = torch.abs(mask_feat)
        # feat = feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)
        # plt.figure()
        # plt.imshow(g1[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = torch.abs(mask_feat).mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)
        # plt.figure()
        # plt.imshow(g2[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = torch.abs(mask_feat).mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)
        # plt.figure()
        # plt.imshow(g3[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)
        # plt.figure()
        # plt.imshow(g4[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        # pred_mask = self.decoder(img_feat)
        img_feat = self.global_pool(img_feat)
        rot = self.rot_regression(img_feat)
        xyz = self.xyz_regression(img_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorSpAddDeco(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = SpatialAttentionWithAdd(64)
        self.space_attention2 = SpatialAttentionWithAdd(64)
        self.space_attention3 = SpatialAttentionWithAdd(128)
        self.space_attention4 = SpatialAttentionWithAdd(256)

        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        # feat = torch.abs(mask_feat)
        # feat = feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention1(img_feat, mask_feat)
        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)
        # plt.figure()
        # plt.imshow(g1[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = torch.abs(mask_feat).mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention2(img_feat, mask_feat)
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)
        # plt.figure()
        # plt.imshow(g2[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = torch.abs(mask_feat).mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention3(img_feat, mask_feat)
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)
        # plt.figure()
        # plt.imshow(g3[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention4(img_feat, mask_feat)
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)
        # plt.figure()
        # plt.imshow(g4[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        pred_mask = self.decoder(img_feat)
        img_feat = self.global_pool(img_feat)
        rot = self.rot_regression(img_feat)
        xyz = self.xyz_regression(img_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorSpAdd(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = SpatialAttentionWithAdd(64)
        self.space_attention2 = SpatialAttentionWithAdd(64)
        self.space_attention3 = SpatialAttentionWithAdd(128)
        self.space_attention4 = SpatialAttentionWithAdd(256)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        # feat = torch.abs(mask_feat)
        # feat = feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention1(img_feat, mask_feat)
        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)
        # plt.figure()
        # plt.imshow(g1[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = torch.abs(mask_feat).mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention2(img_feat, mask_feat)
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)
        # plt.figure()
        # plt.imshow(g2[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = torch.abs(mask_feat).mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention3(img_feat, mask_feat)
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)
        # plt.figure()
        # plt.imshow(g3[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention4(img_feat, mask_feat)
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)
        # plt.figure()
        # plt.imshow(g4[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat = self.global_pool(img_feat)
        rot = self.rot_regression(img_feat)
        xyz = self.xyz_regression(img_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorSpCatDeco(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = SpatialAttentionWithCat(64)
        self.space_attention2 = SpatialAttentionWithCat(64)
        self.space_attention3 = SpatialAttentionWithCat(128)
        self.space_attention4 = SpatialAttentionWithCat(256)

        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        # feat = torch.abs(mask_feat)
        # feat = feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention1(img_feat, mask_feat)
        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)
        # plt.figure()
        # plt.imshow(g1[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = torch.abs(mask_feat).mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention2(img_feat, mask_feat)
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)
        # plt.figure()
        # plt.imshow(g2[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = torch.abs(mask_feat).mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention3(img_feat, mask_feat)
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)
        # plt.figure()
        # plt.imshow(g3[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention4(img_feat, mask_feat)
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)
        # plt.figure()
        # plt.imshow(g4[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        pred_mask = self.decoder(img_feat)
        img_feat = self.global_pool(img_feat)
        rot = self.rot_regression(img_feat)
        xyz = self.xyz_regression(img_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorSpCat(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = SpatialAttentionWithCat(64)
        self.space_attention2 = SpatialAttentionWithCat(64)
        self.space_attention3 = SpatialAttentionWithCat(128)
        self.space_attention4 = SpatialAttentionWithCat(256)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 3)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        # feat = torch.abs(mask_feat)
        # feat = feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention1(img_feat, mask_feat)
        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)
        # plt.figure()
        # plt.imshow(g1[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = torch.abs(mask_feat).mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention2(img_feat, mask_feat)
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)
        # plt.figure()
        # plt.imshow(g2[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = torch.abs(mask_feat).mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention3(img_feat, mask_feat)
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)
        # plt.figure()
        # plt.imshow(g3[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention4(img_feat, mask_feat)
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)
        # plt.figure()
        # plt.imshow(g4[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat = self.global_pool(img_feat)
        rot = self.rot_regression(img_feat)
        xyz = self.xyz_regression(img_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

class PoseRegressorCoeSpDeco(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = SpatialAttention()
        self.space_attention2 = SpatialAttention()
        self.space_attention3 = SpatialAttention()
        self.space_attention4 = SpatialAttention()

        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        pred_mask = self.decoder(img_feat)
        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat)
        # g = self.gate_net(feat_cat).unsqueeze(-1).unsqueeze(-1)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorGSpDeco(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = SpatialAttention()
        self.space_attention2 = SpatialAttention()
        self.space_attention3 = SpatialAttention()
        self.space_attention4 = SpatialAttention()

        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        # plt.show()
        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        # plt.show()
        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        # plt.show()
        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        # plt.show()
        pred_mask = self.decoder(img_feat)
        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        # g = self.gate_net(feat_cat)
        g = self.gate_net(feat_cat).unsqueeze(-1).unsqueeze(-1)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorAddSpDeco(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = SpatialAttention()
        self.space_attention2 = SpatialAttention()
        self.space_attention3 = SpatialAttention()
        self.space_attention4 = SpatialAttention()

        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        # feat = g1.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        # plt.show()
        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        # feat = g2.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        # plt.show()
        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        # feat = g3.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        # plt.show()
        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        # feat = g4.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        # plt.show()
        pred_mask = self.decoder(img_feat)

        feat_final = img_feat + mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorAddSp(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = SpatialAttention()
        self.space_attention2 = SpatialAttention()
        self.space_attention3 = SpatialAttention()
        self.space_attention4 = SpatialAttention()
        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        # feat = g1.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        # plt.show()
        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        # feat = g2.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        # plt.show()
        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        # feat = g3.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        # plt.show()
        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        # feat = g4.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        # plt.show()

        feat_final = img_feat + mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorCoeMscaleSpDeco(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = MScaleSpatialAttention(2)
        self.space_attention2 = MScaleSpatialAttention(4)
        self.space_attention3 = MScaleSpatialAttention(6)
        self.space_attention4 = MScaleSpatialAttention(8)

        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        avg_list, max_list = [], []
        img_feat, _, avg_list, max_list = self.space_attention1(img_feat, mask_feat, avg_list, max_list)
        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _, avg_list, max_list = self.space_attention2(img_feat, mask_feat, avg_list, max_list)
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _, avg_list, max_list = self.space_attention3(img_feat, mask_feat, avg_list, max_list)
        # for im in img_feat[0]:
        #     plt.figure()
        #     plt.imshow(im.cpu(), cmap='gray')
        #     plt.show()
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)
        # for im in img_feat[0]:
        #     plt.figure()
        #     plt.imshow(im.cpu(), cmap='gray')
        #     plt.show()

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _, avg_list, max_list = self.space_attention4(img_feat, mask_feat, avg_list, max_list)
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)

        # feat = mask_feat.mean(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        pred_mask = self.decoder(img_feat)
        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat).unsqueeze(-1).unsqueeze(-1)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorFrFTAllDeco(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            device='cuda',
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = FrFTAttention(64,64, device=device)
        self.space_attention2 = FrFTAttention(64,64, device=device)
        self.space_attention3 = FrFTAttention(128,32, device=device)
        self.space_attention4 = FrFTAttention(256, 16, device=device)


        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        # feat, _ = mask_feat.max(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention1(img_feat, mask_feat)
        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)

        # feat, _ = mask_feat.max(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention2(img_feat, mask_feat)
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)

        # feat, _ = mask_feat.max(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention3(img_feat, mask_feat)
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)

        # feat, _ = mask_feat.max(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        img_feat, _ = self.space_attention4(img_feat, mask_feat)
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)

        # feat, _ = mask_feat.max(dim=1)
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        # plt.figure()
        # plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()
        pred_mask = self.decoder(img_feat)
        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorFrFTMaxDeco(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
            self,
            model_name,
            parameterization,
            convention=None,
            pretrained=False,
            device='cuda',
            **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = FrFTMaxAttention(64,64, device=device)
        self.space_attention2 = FrFTMaxAttention(64,64, device=device)
        self.space_attention3 = FrFTMaxAttention(128,32, device=device)
        self.space_attention4 = FrFTMaxAttention(256, 16, device=device)


        self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }

        # 注册前向钩子
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        feat_dim = 512
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 256, kernel_size=1, bias=False),  # 1x1 conv 降维
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128]

            nn.Linear(128, 1),  # 预测一个标量
            nn.Sigmoid()  # 输出 α ∈ [0, 1]
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        img_feat, _ = self.space_attention1(img_feat, mask_feat)
        img_feat = self.layer1(img_feat)
        mask_feat = self.layer_mask1(mask_feat)

        img_feat, _ = self.space_attention2(img_feat, mask_feat)
        img_feat = self.layer2(img_feat)
        mask_feat = self.layer_mask2(mask_feat)

        img_feat, _ = self.space_attention3(img_feat, mask_feat)
        img_feat = self.layer3(img_feat)
        mask_feat = self.layer_mask3(mask_feat)

        img_feat, _ = self.space_attention4(img_feat, mask_feat)
        img_feat = self.layer4(img_feat)
        mask_feat = self.layer_mask4(mask_feat)

        pred_mask = self.decoder(img_feat)
        feat_cat = torch.concat([img_feat, mask_feat], dim=1)
        g = self.gate_net(feat_cat)

        feat_final = g * img_feat + (1 - g) * mask_feat

        cross_feat = F.adaptive_avg_pool2d(feat_final, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ), pred_mask

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）

        return hook

class PoseRegressorCrossAttn(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        backbone2 = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        print(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.global_pool = backbone.global_pool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_mask1 = backbone2.conv1
        self.bn_mask1 = backbone2.bn1
        self.act_mask1 = backbone2.act1
        self.maxpool_mask = backbone2.maxpool
        self.global_pool_mask = backbone2.global_pool
        self.layer_mask1 = backbone2.layer1
        self.layer_mask2 = backbone2.layer2
        self.layer_mask3 = backbone2.layer3
        self.layer_mask4 = backbone2.layer4

        self.space_attention1 = SpatialAttention()
        self.space_attention2 = SpatialAttention()
        self.space_attention3 = SpatialAttention()
        self.space_attention4 = SpatialAttention()

        # self.decoder = Decoder(1)

        self.features = {}
        # 注册钩子的层名列表（ResNet-18的4个基础层）
        self.target_layers = {
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        }
        # self.decoder = Decoder(1)

        # 注册前向钩子
        feat_dim = 512
        cross_dim = 768
        self.handles = []
        for name, layer in self.target_layers.items():
            handle = layer.register_forward_hook(self._save_features(name))
            self.handles.append(handle)

        self.rot_regression = nn.Sequential(
            nn.Linear(cross_dim, 3)
        )

        self.xyz_regression = nn.Sequential(
            nn.Linear(cross_dim, 3)
        )

        self.cross_attn = MultiHeadAttentionWei(
            embed_dim=cross_dim,
            num_heads=8,  # 可根據 feat_dim 調整
            dropout=0.1,
            batch_first=True
        )
        # self.pos_embedding = nn.Parameter(torch.empty(1, 64, 512).normal_(std=0.02))
        self.proj_img = nn.Conv2d(feat_dim, cross_dim, kernel_size=1)
        self.proj_mask = nn.Conv2d(feat_dim, cross_dim, kernel_size=1)
        pos_embed = get_2d_sine_pos_embed(embed_dim=768)
        self.register_buffer('pos_embedding', pos_embed)
        # self.norm = nn.LayerNorm(feat_dim)
        self.norm = nn.GroupNorm(32, cross_dim)
        self.ffn = nn.Sequential(  # 轻量 FeedForward Network
            nn.Linear(cross_dim, cross_dim * 2),
            nn.ReLU(),
            nn.Linear(cross_dim * 2, cross_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, mask):
        img_feat = self.conv1(x)
        img_feat = self.bn1(img_feat)
        img_feat = self.act1(img_feat)
        img_feat = self.maxpool(img_feat)
        mask_feat = self.conv_mask1(mask)
        mask_feat = self.bn_mask1(mask_feat)
        mask_feat = self.act_mask1(mask_feat)
        mask_feat = self.maxpool_mask(mask_feat)

        g1 = self.space_attention1(mask_feat)
        img_feat = self.layer1(img_feat * g1)
        mask_feat = self.layer_mask1(mask_feat)

        g2 = self.space_attention2(mask_feat)
        img_feat = self.layer2(img_feat * g2)
        mask_feat = self.layer_mask2(mask_feat)

        g3 = self.space_attention3(mask_feat)
        img_feat = self.layer3(img_feat * g3)
        mask_feat = self.layer_mask3(mask_feat)

        g4 = self.space_attention4(mask_feat)
        img_feat = self.layer4(img_feat * g4)
        mask_feat = self.layer_mask4(mask_feat)

        img_feat = self.proj_img(img_feat)
        mask_feat = self.proj_mask(mask_feat)
        B, C, H, W = img_feat.shape
        img_tokens = img_feat.reshape(B, C, H * W).permute(0, 2, 1)
        mask_tokens = mask_feat.reshape(B, C, H * W).permute(0, 2, 1)
        img_tokens = img_tokens + self.pos_embedding
        mask_tokens = mask_tokens + self.pos_embedding

        # mask_wei = compute_mask_weights2d(mask, 32)
        # attn_output, attn_weights = self.cross_attn(query=mask_tokens, key=img_tokens, value=img_tokens, weight_map=mask_wei)
        attn_output, attn_weights = self.cross_attn(query=mask_tokens, key=img_tokens, value=img_tokens)
        # attn_output = attn_output + mask_tokens
        # attn_output = self.norm(attn_output)
        attn_output = attn_output + self.ffn(attn_output)
        attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W)
        attn_output = self.norm(attn_output)

        # aw = attn_weights[0]
        # print(aw.max())
        # print(aw.min())
        # plt.figure()
        # plt.imshow(aw.detach().cpu())
        # plt.show()

        cross_feat = F.adaptive_avg_pool2d(attn_output, 1).flatten(1)

        rot = self.rot_regression(cross_feat)
        xyz = self.xyz_regression(cross_feat)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def _save_features(self, name: str):
        def hook(module, input, output):
            self.features[name] = output.detach()  # 保存特征（无梯度）
        return hook

def compute_mask_weights(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
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

def compute_mask_weights2d(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
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

    mask_weights = mask_weights.unsqueeze(2) * mask_weights.unsqueeze(1)
    mask_weights = (mask_weights - 0) / (mask_weights.max() - 0 + 1e-6)

    return mask_weights

def get_2d_sine_pos_embed(embed_dim: int, grid_h: int = 8, grid_w: int = 8):
    """
    生成 2D 正弦位置编码
    返回形状: (1, grid_h * grid_w, embed_dim)
    """
    # 生成行坐标和列坐标网格
    grid_h = torch.arange(grid_h, dtype=torch.float32)
    grid_w = torch.arange(grid_w, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')  # (H, W)

    grid_h = grid_h.flatten()  # (H*W,)
    grid_w = grid_w.flatten()  # (H*W,)

    # 为高度和宽度各分配 embed_dim // 2 维（再分成 sin/cos 各一半）
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (10000 ** omega)  # (pos_dim,)

    # 高度编码
    out_h = grid_h[:, None] * omega[None, :]  # (H*W, pos_dim)
    emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)  # (H*W, pos_dim*2)

    # 宽度编码
    out_w = grid_w[:, None] * omega[None, :]
    emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)  # (H*W, pos_dim*2)

    # 拼接高度和宽度编码
    pos_embed = torch.cat([emb_h, emb_w], dim=1)  # (H*W, embed_dim)
    return pos_embed.unsqueeze(0)  # (1, H*W, embed_dim)

def get_1d_sine_pos_embed(embed_dim: int, seq_len: int = 64):
    """
    最原始的 1D 正弦位置编码（Transformer 原论文版本）
    返回形状: (1, seq_len, embed_dim)
    """
    # position: 0, 1, 2, ..., seq_len-1
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)

    # div_term: 10000^(2k/d) 的倒数
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                         (-math.log(10000.0) / embed_dim))  # (embed_dim//2,)

    # 初始化位置编码
    pe = torch.zeros(1, seq_len, embed_dim)

    # 偶数维度：sin
    pe[0, :, 0::2] = torch.sin(position * div_term)
    # 奇数维度：cos
    pe[0, :, 1::2] = torch.cos(position * div_term)

    return pe  # (1, seq_len, embed_dim)

if __name__ == '__main__':
    import torch

    a = torch.tensor([[1.],
                      [2.]]).unsqueeze(-1).unsqueeze(-1)  # (2, 1)

    b = torch.ones(2, 1, 2, 2)  # (2, 1, 2, 2)

    c = a * b

    print("a.shape:", a.shape)
    print("b.shape:", b.shape)
    print("c.shape:", c.shape)

    print("\nc[0] =\n", c[0])
    print("\nc[1] =\n", c[1])
    print(c)


    device_str = "cuda:1"
    device = device_str
    model_params = {
        "model_name": "resnet18",
        "parameterization": "se3_log_map",
        "convention": None,
        "norm_layer": "groupnorm",
        # "device": device_str
    }
    # model = PoseRegressorAttn(**model_params)
    # model = PoseRegressorCoeFrFT(**model_params)
    model = PoseRegressorCoeMscaleSpDeco(**model_params)
    # model = PoseRegressorSpCatDeco(**model_params)
    model = model.to(device)
    # model = PoseRegressorCat2(**model_params)
    x = torch.randn(8, 1, 256, 256).to(device)
    mask = torch.ones(8, 1, 256, 256).to(device)

    st = time.time()
    model(x, mask)
    print(f"{time.time() - st}")

