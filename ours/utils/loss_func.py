import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import normalized_mutual_information, structural_similarity
from torch import nn
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
from visdom.utils.server_utils import window
import kornia
from packaging import version
from torch import nn

from ours.cut.style_to_drr import StyleChanger
from ours.dataset.CT_dataset import IntubationDataset
from ours.utils.CT_dataset import Transforms


def mutual_information_tensor(img1_tensor, img2_tensor, bins=30, epsilon=1e-10, device='cuda:0'):
    """
    计算两张灰度图像的互信息（PyTorch GPU加速版）

    参数：
        img1_tensor, img2_tensor: 输入图像 (Tensor [H,W] or [C,H,W])
        bins: 直方图的分箱数
        epsilon: 防止log(0)的小值
        device: 计算设备

    返回：
        mi: 互信息值 (标量)
        joint_hist: 联合直方图 (Tensor [bins, bins])
    """
    # 移动到指定设备并展平
    img1_flat = img1_tensor.squeeze().to(device).flatten().float()
    img2_flat = img2_tensor.squeeze().to(device).flatten().float()

    # 计算联合直方图 (PyTorch实现)
    bin_edges = torch.linspace(0, 255, bins + 1, device=device)
    bin_indices_x = torch.bucketize(img1_flat, bin_edges, right=True) - 1
    bin_indices_y = torch.bucketize(img2_flat, bin_edges, right=True) - 1

    # 生成二维直方图
    joint_hist = torch.histc(
        bin_indices_x * bins + bin_indices_y,
        bins=bins * bins,
        min=0,
        max=bins * bins - 1
    ).view(bins, bins).float()

    # 转换为概率分布
    p_xy = joint_hist / (joint_hist.sum() + epsilon)
    p_x = p_xy.sum(dim=1)
    p_y = p_xy.sum(dim=0)

    # 计算熵
    h_x = -torch.sum(p_x * torch.log2(p_x + epsilon))
    h_y = -torch.sum(p_y * torch.log2(p_y + epsilon))
    h_xy = -torch.sum(p_xy * torch.log2(p_xy + epsilon))

    mi = h_x + h_y - h_xy
    return mi, joint_hist.cpu()

class PatchNCE(nn.Module):
    def __init__(self, patch_size=13, num_negatives=128, temperature=0.07):
        super().__init__()
        self.patch_size = patch_size
        self.num_negatives = num_negatives
        self.temperature = temperature

    def extract_patches(self, x):
        """提取并规范化图像块"""
        # 填充图像使其能被patch_size整除
        h_pad = (self.patch_size - x.size(2) % self.patch_size) % self.patch_size
        w_pad = (self.patch_size - x.size(3) % self.patch_size) % self.patch_size
        x = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')

        # 提取图像块 [B, C, H, W] -> [B, num_patches, patch_dim]
        patches = x.unfold(2, self.patch_size, self.patch_size) \
            .unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(x.size(0), -1, self.patch_size ** 2)

        # 零均值归一化 (ZNCC风格)
        patches = (patches - patches.mean(dim=-1, keepdim=True)) / \
                  (patches.std(dim=-1, keepdim=True) + 1e-5)
        return patches

    def forward(self, x1, x2):
        """
        输入：x1, x2 - 形状为[B, C, H, W]的图像张量
        输出：NCE损失值
        """
        # 提取并规范化图像块 [B, num_patches, patch_dim]
        patches1 = self.extract_patches(x1)  # 查询块
        patches2 = self.extract_patches(x2)  # 关键块

        patches1 = F.normalize(patches1, p=2, dim=-1)
        patches2 = F.normalize(patches2, p=2, dim=-1)
        # 计算余弦相似度矩阵 [B, num_patches, num_patches]
        sim_matrix = torch.bmm(patches1, patches2.permute(0, 2, 1))  # 批次矩阵乘法

        # 构建正样本和负样本
        batch_size, num_patches = sim_matrix.shape[:2]
        diag_mask = torch.eye(num_patches, device=x1.device).bool()  # 正样本掩码

        # 正样本相似度 [B, num_patches]
        pos_sim = sim_matrix[:, diag_mask].view(batch_size, num_patches)

        # 随机采样负样本 [B, num_patches, num_negatives]
        neg_indices = get_neg_indices(batch_size, num_patches, self.num_negatives, device=x1.device)
        neg_sim = torch.gather(sim_matrix, 2, neg_indices)
        for i in range(len(neg_indices[0])):
            for j in neg_indices[0][i]:
                if j == i:
                    print(i)
                    print("yes")

        # 合并相似度 [B, num_patches, 1 + num_negatives]
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1) / self.temperature

        # 交叉熵损失（正样本位于索引0）
        labels = torch.zeros((batch_size, num_patches), dtype=torch.long, device=x1.device)
        loss = F.cross_entropy(logits.view(-1, self.num_negatives + 1), labels.view(-1))

        return loss


def get_neg_indices(batch_size, num_patches, num_negatives, device):
    # 生成候选索引 [B, N, N-1]
    all_indices = torch.arange(num_patches, device=device).repeat(batch_size, num_patches, 1)

    # 创建掩码排除正样本（对角线位置）
    mask = ~torch.eye(num_patches, dtype=torch.bool, device=device).unsqueeze(0)
    candidate_indices = all_indices[mask].view(batch_size, num_patches, num_patches - 1)

    # 随机选择负样本 [B, N, K]
    rand_idx = torch.randint(0, num_patches - 1, (batch_size, num_patches, num_negatives), device=device)
    neg_indices = torch.gather(candidate_indices, 2, rand_idx)

    return neg_indices

class PatchNCELoss(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        # if self.opt.nce_includes_all_negatives_from_minibatch:
        #     # reshape features as if they are all negatives of minibatch of size 1.
        #     batch_dim_for_bmm = 1
        # else:
        #     batch_dim_for_bmm = self.opt.batch_size

        batch_dim_for_bmm = self.batch_size
        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss

def masked_ssim(
        img1: torch.Tensor,
        img2: torch.Tensor,
        mask: torch.Tensor = None,
        data_range: float = 1.0,
        window_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
        eps: float = 1e-6
) -> torch.Tensor:
    """
    计算带掩码的SSIM（结构相似性指数）

    参数:
        img1 (torch.Tensor): 输入图像1，形状为 (B, C, H, W)
        img2 (torch.Tensor): 输入图像2，形状与img1相同
        mask (torch.Tensor, optional): 掩码张量，形状与img1相同。值为1表示有效区域，0表示忽略。默认为全1掩码
        data_range (float): 图像数据的动态范围（如[0,1]范围=1，[0,255]范围=255）
        window_size (int): 高斯窗口大小
        k1, k2 (float): SSIM算法稳定性常数
        eps (float): 防止除零的小常数

    返回:
        ssim_per_image (torch.Tensor): 每个图像的SSIM值，形状为 (B,)
    """
    # 输入验证
    assert img1.shape == img2.shape, "Input images must have the same shape"
    assert img1.dim() == 4, "Input must be 4D tensor (B, C, H, W)"

    # 如果没有提供掩码，创建全1掩码
    if mask is None:
        mask = torch.ones_like(img1)
    else:
        assert mask.shape == img1.shape, "Mask must have same shape as images"

    # 确保掩码是二值化的
    mask = (mask > 0.5).float()

    # 计算SSIM常数
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    # 创建高斯窗口
    sigma = 1.5  # 用户指定的高斯核标准差
    x = torch.arange(window_size, device=img1.device, dtype=torch.float) - window_size // 2
    window_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    window_1d = window_1d / window_1d.sum()  # 归一化

    # 创建二维高斯窗口
    window_2d = torch.outer(window_1d, window_1d)
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    window_2d = window_2d / window_2d.sum()

    # 使用卷积计算局部统计量
    def gaussian_convolve(x, window):
        return F.conv2d(x, window, padding=window_size // 2, groups=x.shape[1])

    # 计算加权均值
    mu1 = gaussian_convolve(img1 * mask, window_2d)
    mu2 = gaussian_convolve(img2 * mask, window_2d)

    # 计算加权平方和
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算加权方差和协方差
    sigma1_sq = gaussian_convolve((img1 * mask).pow(2), window_2d) - mu1_sq
    sigma2_sq = gaussian_convolve((img2 * mask).pow(2), window_2d) - mu2_sq
    sigma12 = gaussian_convolve((img1 * mask) * (img2 * mask), window_2d) - mu1_mu2

    # 计算掩码的权重
    weight_mask = gaussian_convolve(mask, window_2d)
    weight_mask = torch.clamp(weight_mask, min=eps)  # 防止除零

    # 应用掩码调整统计量
    mu1 = mu1 / weight_mask
    mu2 = mu2 / weight_mask
    mu1_sq = mu1_sq / weight_mask
    mu2_sq = mu2_sq / weight_mask
    mu1_mu2 = mu1_mu2 / weight_mask
    sigma1_sq = sigma1_sq / weight_mask
    sigma2_sq = sigma2_sq / weight_mask
    sigma12 = sigma12 / weight_mask

    # 计算SSIM图
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + eps)

    # 仅考虑有效区域
    valid_mask = (weight_mask > 0.01).float()
    ssim_map = ssim_map * valid_mask

    # 计算每张图的平均SSIM
    batch_size = img1.size(0)
    ssim_per_image = torch.zeros(batch_size, device=img1.device)

    for i in range(batch_size):
        valid_pixels = valid_mask[i].sum()
        if valid_pixels > 0:
            ssim_per_image[i] = ssim_map[i][valid_mask[i] > 0].mean()
        else:
            ssim_per_image[i] = torch.tensor(0.0, device=img1.device)

    return ssim_per_image


def masked_ssim2(
        img1: torch.Tensor,
        img2: torch.Tensor,
        mask: torch.Tensor = None,
        data_range: float = 1.0,
        window_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    计算带掩码的SSIM（结构相似性指数）
    """
    # 输入验证
    assert img1.shape == img2.shape, "输入图像必须形状相同"
    assert img1.dim() == 4, "输入必须为4D张量 (B, C, H, W)"
    assert img1.size(1) == 1, "本实现仅支持单通道输入"

    # 处理默认掩码
    if mask is None:
        mask = torch.ones_like(img1)
    else:
        mask = (mask > 0.5).float()

    # SSIM常数计算
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    # 高斯窗口生成（优化版）
    sigma = 1.5
    x = torch.linspace(-window_size // 2, window_size // 2, window_size, device=img1.device)
    window_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    window_1d /= window_1d.sum()
    window_2d = torch.outer(window_1d, window_1d).view(1, 1, window_size, window_size)  # (1,1,H,W)

    # 卷积辅助函数（针对单通道优化）
    def masked_conv(input_tensor):
        """带掩码的卷积操作"""
        # 计算加权和
        weighted_sum = F.conv2d(
            input=input_tensor * mask,
            weight=window_2d,
            padding=window_size // 2,
            groups=1
        )
        # 计算有效权重和
        sum_weights = F.conv2d(
            input=mask,
            weight=window_2d,
            padding=window_size // 2,
            groups=1
        )
        return weighted_sum / (sum_weights + eps)

    # 计算均值
    mu1 = masked_conv(img1)
    mu2 = masked_conv(img2)

    # 计算协方差项
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算方差（数值稳定版）
    sigma1_sq = masked_conv(img1.pow(2)) - mu1_sq
    sigma2_sq = masked_conv(img2.pow(2)) - mu2_sq
    sigma12 = masked_conv(img1 * img2) - mu1_mu2

    # 数值稳定性处理
    sigma1_sq = torch.clamp(sigma1_sq, min=0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0)

    # 计算SSIM分子分母
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    # 逐像素SSIM计算
    ssim_map = numerator / (denominator + eps)

    # 有效区域加权平均
    valid_mask = mask * F.conv2d(mask, window_2d, padding=window_size // 2, groups=1)
    valid_mask = (valid_mask > 1e-3).float()  # 避免极小权重影响

    # 加权聚合
    weighted_ssim = ssim_map * valid_mask
    return (weighted_ssim.sum(dim=(1, 2, 3)) / (valid_mask.sum(dim=(1, 2, 3)) + eps)).squeeze()

class MaskedLuminanceDistributionLoss(nn.Module):
    def __init__(self, bins=64, dark_weight_factor=5.0, weight=1.0,
                 min_val=0.0, max_val=1.0, mask_radius=120, device=torch.device('cpu')):
        """
        带圆形掩码的灰度图亮度分布匹配损失函数

        参数:
        bins: 直方图bin数量
        dark_weight_factor: 暗区权重因子
        weight: 损失权重
        min_val: 像素值范围最小值
        max_val: 像素值范围最大值
        mask_radius: 圆形掩码半径比例 (0.0-1.0)
        """
        super().__init__()
        self.bins = bins
        self.dark_weight_factor = dark_weight_factor
        self.weight = weight
        self.min_val = min_val
        self.max_val = max_val
        self.mask_radius = mask_radius

        # 创建bin边界
        self.register_buffer('bin_edges', torch.linspace(min_val, max_val, bins + 1))

        # 创建暗区权重向量
        dark_weights = torch.linspace(dark_weight_factor, 1.0, bins).to(device)
        self.register_buffer('dark_weights', dark_weights / dark_weights.sum())

    def create_circular_mask(self, H, W, radius=None):
        """创建圆形掩码（中心为图像中心）"""
        center_x, center_y = W // 2, H // 2
        if radius is None:
            radius = min(H, W) * self.mask_radius / 2

        # 创建坐标网格
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
        dist = torch.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

        # 创建掩码 (1表示保留，0表示丢弃)
        mask = (dist <= radius).float()
        return mask

    def masked_histogram(self, x, mask):
        """计算带掩码的软直方图"""
        # 确保输入形状匹配
        B, C, H, W = x.shape
        mask = mask.view(1, 1, H, W).to(x.device)

        # 应用掩码 - 将掩码外像素设为无效值
        masked_x = x * mask + torch.ones_like(x) * (self.min_val - 1) * (1 - mask)

        # 将像素值映射到bin索引
        bin_pos = (masked_x - self.min_val) * (self.bins - 1) / (self.max_val - self.min_val)

        # 只处理掩码内有效像素
        valid_mask = (masked_x >= self.min_val) & (masked_x <= self.max_val)
        lower_idx = torch.floor(bin_pos).long().clamp(0, self.bins - 1) * valid_mask
        upper_idx = (lower_idx + 1).clamp(0, self.bins - 1) * valid_mask

        # 计算权重
        upper_weight = (bin_pos - lower_idx) * valid_mask
        lower_weight = (1 - upper_weight) * valid_mask

        # 构建直方图 (考虑掩码)
        hist = torch.zeros(B, self.bins, device=x.device)

        for b in range(B):
            # 展平当前batch的所有张量
            flat_lower_idx = lower_idx[b].view(-1)
            flat_upper_idx = upper_idx[b].view(-1)
            flat_lower_weight = lower_weight[b].view(-1)
            flat_upper_weight = upper_weight[b].view(-1)
            flat_valid = valid_mask[b].view(-1)

            # 只处理有效像素
            valid_idx = flat_valid.nonzero(as_tuple=True)[0]

            if len(valid_idx) > 0:
                hist[b].index_add_(
                    0,
                    flat_lower_idx[valid_idx],
                    flat_lower_weight[valid_idx]
                )
                hist[b].index_add_(
                    0,
                    flat_upper_idx[valid_idx],
                    flat_upper_weight[valid_idx]
                )

        # 归一化
        return hist / (hist.sum(dim=1, keepdim=True) + 1e-8)

    def earth_mover_distance(self, p, q):
        """计算两个分布之间的一维Earth Mover's Distance"""
        cdf_p = torch.cumsum(p, dim=1)
        cdf_q = torch.cumsum(q, dim=1)
        return torch.mean(torch.abs(cdf_p - cdf_q), dim=1)

    def forward(self, generated, style):
        """
        计算带圆形掩码的亮度分布匹配损失

        参数:
        generated: 生成图像 (B, 1, H, W) 或 (B, 3, H, W)
        style: 风格图像 (B, 1, H, W) 或 (B, 3, H, W)
        """
        # 确保输入为灰度
        if generated.dim() == 4 and generated.size(1) == 3:
            generated = generated[:, 0:1, :, :]
        if style.dim() == 4 and style.size(1) == 3:
            style = style[:, 0:1, :, :]

        # 归一化到0-1
        generated = (generated - generated.min()) / (generated.max() - generated.min() + 10e-8)
        style = (style - style.min()) / (style.max() - style.min() + 10e-8)

        plt.figure()
        plt.imshow(generated.cpu().squeeze(), cmap="gray")
        plt.show()

        plt.figure()
        plt.imshow(style.cpu().squeeze(), cmap="gray")
        plt.show()

        # 获取图像尺寸
        B, C, H, W = generated.shape

        # 创建圆形掩码
        mask = self.create_circular_mask(H, W, self.mask_radius).to(generated.device)

        # 计算掩码区域直方图
        gen_hist = self.masked_histogram(generated, mask)
        style_hist = self.masked_histogram(style, mask)

        # 应用暗区权重
        weighted_gen_hist = gen_hist * self.dark_weights
        weighted_style_hist = style_hist * self.dark_weights
        # weighted_gen_hist = gen_hist
        # weighted_style_hist = style_hist

        # 归一化加权直方图
        weighted_gen_hist = weighted_gen_hist / (weighted_gen_hist.sum(dim=1, keepdim=True) + 1e-8)
        weighted_style_hist = weighted_style_hist / (weighted_style_hist.sum(dim=1, keepdim=True) + 1e-8)

        # 绘制灰度直方图 (仅显示第一个batch)
        plt.figure(figsize=(12, 6))

        # 原始直方图
        plt.subplot(2, 2, 1)
        plt.bar(np.arange(self.bins), gen_hist[0].detach().cpu().numpy(), width=1.0)
        plt.title("Generated Image Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.ylim(0, gen_hist[0].max().item() * 1.1)

        plt.subplot(2, 2, 2)
        plt.bar(np.arange(self.bins), style_hist[0].detach().cpu().numpy(), width=1.0)
        plt.title("Style Image Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.ylim(0, style_hist[0].max().item() * 1.1)

        # 加权直方图
        plt.subplot(2, 2, 3)
        plt.bar(np.arange(self.bins), weighted_gen_hist[0].detach().cpu().numpy(), width=1.0)
        plt.title("Weighted Generated Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Weighted Frequency")
        plt.plot(self.dark_weights[0].detach().cpu().numpy(), 'r-', label="Dark Weights")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.bar(np.arange(self.bins), weighted_style_hist[0].detach().cpu().numpy(), width=1.0)
        plt.title("Weighted Style Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Weighted Frequency")
        plt.plot(self.dark_weights[0].detach().cpu().numpy(), 'r-', label="Dark Weights")
        plt.legend()

        plt.tight_layout()
        plt.show()

        # 计算EMD距离
        emd_loss = self.earth_mover_distance(weighted_gen_hist, weighted_style_hist)

        return self.weight * emd_loss.mean()

# 使用示例
if __name__ == "__main__":
    # 假设输入为两张256x256的灰度图
    H = 256
    W = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1 = torch.randn(1, 1, 256, 256)  # 固定图像 [B, C, H, W]
    img2 = torch.randn(1, 1, 256, 256) # 移动图像

    # patch_nce = PatchNCE(patch_size=16)
    # # patch_nce = PatchNCELoss(1)
    # nce = patch_nce(img1, img2)
    # print(nce)
    # s3 = 1 - kornia.losses.ssim_loss(img1, img2, window_size=7, reduction='mean') * 2
    # img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    # img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    #
    # s1 = masked_ssim(img1, img2)
    # s2 = masked_ssim2(img1, img2, window_size=7)
    # s3 = 1 - kornia.losses.ssim_loss(img1, img2, window_size=7, reduction='mean') * 2
    #
    # # 初始化PatchNCE计算器
    # nce_criterion = PatchNCE(patch_size=13)
    #
    # # 计算NCE损失
    # loss = nce_criterion(img1, img2)
    # print(f"PatchNCE Loss: {loss.item():.4f}")

    # # 读取图像并转换为Tensor (假设输入为0-255范围的Tensor)
    # img1 = torch.rand((1, 1, 256, 256)) * 255  # [1,H,W]
    # img2 = torch.rand((1, 1, 256, 256)) * 255
    #
    # # 转换为PyTorch Tensor并移动到GPU
    # img1_tensor = img1.to(torch.float32).to('cuda')  # [H,W]
    # img2_tensor = img2.to(torch.float32).to('cuda')
    #
    # # 计算互信息
    # mi1 = normalized_mutual_information(img1_tensor.cpu(), img2_tensor.cpu(), bins=30)
    # mi, joint_hist = mutual_information_tensor(img1_tensor, img2_tensor, bins=30)
    # print(f"互信息 MI = {mi.item():.4f} bits")
    #
    # # 可视化 (需要转回CPU和Numpy)
    # plt.figure(figsize=(10, 6))
    # plt.imshow(joint_hist.numpy().T, origin='lower', cmap='viridis',
    #            extent=[0, 255, 0, 255])
    # plt.colorbar(label="频数")
    # plt.xlabel("图像1像素值")
    # plt.ylabel("图像2像素值")
    # plt.title("联合直方图 (GPU加速)")
    # plt.show()

    masked_lum_loss = MaskedLuminanceDistributionLoss(
        bins=64,
        dark_weight_factor=100.0,
        weight=1.0,
        mask_radius=120,
        device=device
    )

    style_change = StyleChanger(
        "/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec5/80_net_G.pth",
        device=device,
        resize=256)

    root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    specimen = IntubationDataset(root, x_root, y_offset=155, z_cut=600)
    img, pose = specimen[3]
    transforms = Transforms(256)
    img = transforms(img, reverse=False).to(device)
    # img_ori = torch.tensor(img).to(device).to(torch.float32)
    img_change = style_change(img)

    # plt.figure()
    # plt.imshow(img.cpu().squeeze(), cmap="gray")
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(img_change.cpu().squeeze(), cmap="gray")
    # plt.show()

    root = "/home/zsr/project/diffpose/ours/drrStyle/trainB"
    img_name1 = os.path.join(root, os.listdir(root)[2])
    img_drr = torch.tensor(np.array(Image.open(img_name1).convert('RGB'))).permute(2, 0, 1).unsqueeze(0).to(device)


    loss = masked_lum_loss(img, img_change)
    print(loss)



