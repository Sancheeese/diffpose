import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from einops import rearrange
from sympy.logic.inference import valid

from ours.utils.CT_dataset import create_circle_mask
from ours.utils.CT_dataset_augment2 import toZeroOne


class GradSimilarity(torch.nn.Module):
    def __init__(self, size=256, radius=119, weight=(0.5, 0.5)):
        super().__init__()
        self.mask = create_circle_mask(size, radius)
        self.weight = weight

    def forward(self, image1, image2, device=torch.device('cpu')):
        direction_consistency, mag_corr = calculate_gradient_consistency_with_mask(image1, image2, self.mask)
        return self.weight[0] * direction_consistency + self.weight[1] * mag_corr


def calculate_gradient_consistency_with_mask(
        img1_tensor,  # 输入图像1, 形状 [B, 1, H, W]
        img2_tensor,  # 输入图像2, 形状 [B, 1, H, W]
        mask_tensor=None  # 遮挡区域掩码, 形状 [B, 1, H, W], 1表示遮挡区域，0表示有效区域
):
    """
    参数说明:
    - img1_tensor, img2_tensor: 灰度图像张量，范围建议 [0,1]
    - mask_tensor: 手术器械遮挡区域的二值掩码（1=遮挡，0=有效区域）
    - device: 计算设备 ('cuda' 或 'cpu')
    """
    if mask_tensor is None:
        mask_tensor = torch.zeros_like(img1_tensor)
    # 确保输入在GPU上并转换为浮点型
    device = img1_tensor.device
    img1 = img1_tensor.float().to(device)
    img2 = img2_tensor.float().to(device)
    mask = mask_tensor.float().to(device)

    # Sobel算子定义
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                  dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                  dtype=torch.float32, device=device).view(1, 1, 3, 3)

    # 计算梯度
    gx1 = F.conv2d(img1, sobel_kernel_x, padding=1)
    gy1 = F.conv2d(img1, sobel_kernel_y, padding=1)
    gx2 = F.conv2d(img2, sobel_kernel_x, padding=1)
    gy2 = F.conv2d(img2, sobel_kernel_y, padding=1)

    # 计算梯度幅值
    mag1 = torch.sqrt(gx1 ** 2 + gy1 ** 2 + 1e-8)
    mag2 = torch.sqrt(gx2 ** 2 + gy2 ** 2 + 1e-8)

    # 方向余弦相似度
    dot_product = gx1 * gx2 + gy1 * gy2
    mag_product = mag1 * mag2
    cos_similarity = dot_product / mag_product

    valid_mask = (mask > 0.5).float()

    # plt.figure()
    # plt.imshow(valid_mask.cpu().squeeze().squeeze(), cmap='gray')
    # plt.show()
    # 计算有效区域的均值
    valid_cos = cos_similarity * valid_mask
    direction_consistency = valid_cos.sum() / (valid_mask.sum() + 1e-8)

    # 幅值相关性（仅计算有效区域）
    mag1_masked = mag1 * valid_mask
    mag2_masked = mag2 * valid_mask
    mag_corr = torch.corrcoef(torch.stack([mag1_masked.view(-1), mag2_masked.view(-1)]))[0, 1]

    return direction_consistency, mag_corr

def gradient_ncc(
        img1_tensor,  # 输入图像1, 形状 [B, 1, H, W]
        img2_tensor,  # 输入图像2, 形状 [B, 1, H, W]
        mask_tensor=None  # 遮挡区域掩码, 形状 [B, 1, H, W]
):
    """
    参数说明:
    - img1_tensor, img2_tensor: 灰度图像张量，范围建议 [0,1]
    - mask_tensor: 手术器械遮挡区域的二值掩码（1=遮挡，0=有效区域）
    - device: 计算设备 ('cuda' 或 'cpu')
    """
    if mask_tensor is None:
        mask_tensor = torch.ones_like(img1_tensor)
    # 确保输入在GPU上并转换为浮点型
    device = img1_tensor.device
    img1_tensor = toZeroOne(img1_tensor)
    img2_tensor = toZeroOne(img2_tensor)
    img1 = img1_tensor.float().to(device)
    img2 = img2_tensor.float().to(device)
    mask = mask_tensor.float().to(device)

    # Sobel算子定义
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                  dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                  dtype=torch.float32, device=device).view(1, 1, 3, 3)

    # 计算梯度
    gx1 = F.conv2d(img1, sobel_kernel_x, padding=1)
    gy1 = F.conv2d(img1, sobel_kernel_y, padding=1)
    gx2 = F.conv2d(img2, sobel_kernel_x, padding=1)
    gy2 = F.conv2d(img2, sobel_kernel_y, padding=1)

    # 计算梯度幅值
    mag1 = torch.sqrt(gx1 ** 2 + gy1 ** 2 + 1e-8)  # [B, 1, H, W]
    mag2 = torch.sqrt(gx2 ** 2 + gy2 ** 2 + 1e-8)  # [B, 1, H, W]

    # Mask处理
    if mask is not None:
        valid_mask = (mask > 0.5).float()
        mag1 = mag1 * valid_mask
        mag2 = mag2 * valid_mask

    mag1 = toZeroOne(torch.clamp(toZeroOne(mag1), max=0.2))
    mag2 = toZeroOne(torch.clamp(toZeroOne(mag2), max=0.2))
    #
    # plt.figure()
    # plt.imshow(mag1.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    # plt.show()
    # plt.figure()
    # plt.imshow(mag2.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    # plt.show()

    # 计算梯度幅值的NCC
    mean1 = (mag1 * valid_mask).sum(dim=[-2, -1], keepdim=True) / (valid_mask.sum(dim=[-2, -1], keepdim=True) + 1e-6)
    mean2 = (mag2 * valid_mask).sum(dim=[-2, -1], keepdim=True) / (valid_mask.sum(dim=[-2, -1], keepdim=True) + 1e-6)

    numerator = ((mag1 - mean1) * (mag2 - mean2)).sum()  # [B]
    denom1 = torch.sqrt(((mag1 - mean1) ** 2).sum())  # [B]
    denom2 = torch.sqrt(((mag2 - mean2) ** 2).sum())  # [B]

    ncc = numerator / (denom1 * denom2 + 1e-8)  # [B]
    return ncc.mean()


def multiscale_gradient_ncc(
        img1_tensor,  # 输入图像1, 形状 [B, 1, H, W]
        img2_tensor,  # 输入图像2, 形状 [B, 1, H, W]
        mask_tensor=None,  # 遮挡区域掩码, 形状 [B, 1, H, W]
        patch_sizes=[7, 5, 3],  # 不同尺度的patch大小
        scale_weights=None  # 各尺度权重
):
    """
    多尺度梯度NCC - 在每个patch内独立计算梯度
    """
    if mask_tensor is None:
        mask_tensor = torch.ones_like(img1_tensor)

    device = img1_tensor.device
    img1 = torch.clamp(img1_tensor.float(), 0, 1)
    img2 = torch.clamp(img2_tensor.float(), 0, 1)
    mask = mask_tensor.float()

    if scale_weights is None:
        scale_weights = [1.0 / len(patch_sizes)] * len(patch_sizes)

    # Sobel算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)

    # 一次性计算整张图像的梯度
    gx1 = F.conv2d(img1, sobel_x, padding=1)
    gy1 = F.conv2d(img1, sobel_y, padding=1)
    gx2 = F.conv2d(img2, sobel_x, padding=1)
    gy2 = F.conv2d(img2, sobel_y, padding=1)

    # 计算梯度幅值
    mag1 = torch.sqrt(gx1 ** 2 + gy1 ** 2 + 1e-8)
    mag2 = torch.sqrt(gx2 ** 2 + gy2 ** 2 + 1e-8)

    # 应用mask和归一化
    valid_mask_global = (mask > 0.5).float()
    mag1 = mag1 * valid_mask_global
    mag2 = mag2 * valid_mask_global

    total_ncc = 0

    def to_patches(x, patch_size):
        # x = x.unfold(2, patch_size, patch_size // 2).unfold(3, patch_size, patch_size // 2)
        x = x.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
        return rearrange(x, "b c p1 p2 h w -> b (c p1 p2) h w")

    total_ncc = 0

    for patch_size, weight in zip(patch_sizes, scale_weights):
        # 将梯度幅值图像分割为patches
        patches1 = to_patches(mag1, patch_size)  # [B, N_patches, 1, patch_size, patch_size]
        patches2 = to_patches(mag2, patch_size)
        mask_patches = to_patches(valid_mask_global, patch_size)
        _, c, h, w = patches1.shape

        mean1 = (patches1 * mask_patches).sum(dim=[-2, -1], keepdim=True) / (
                mask_patches.sum(dim=[-2, -1], keepdim=True) + 1e-8)
        mean2 = (patches2 * mask_patches).sum(dim=[-2, -1], keepdim=True) / (
                mask_patches.sum(dim=[-2, -1], keepdim=True) + 1e-8)

        numerator = ((patches1 - mean1) * (patches2 - mean2) * mask_patches).sum(dim=[-2, -1])
        denom1 = torch.sqrt(((patches1 - mean1) ** 2 * mask_patches).sum(dim=[-2, -1]))
        denom2 = torch.sqrt(((patches2 - mean2) ** 2 * mask_patches).sum(dim=[-2, -1]))

        ncc = numerator / (denom1 * denom2 + 1e-8)
        ncc = ncc.sum()
        ncc = (ncc / c).squeeze()
        total_ncc += ncc * weight

    return total_ncc


def get_edge(img, mask=None):
    if mask is None:
        mask = torch.ones_like(img)
    # 确保输入在GPU上并转换为浮点型
    device = img.device
    img1_tensor = toZeroOne(img)
    img1 = img1_tensor.float().to(device)
    mask = mask.float().to(device)

    # Sobel算子定义
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                  dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                  dtype=torch.float32, device=device).view(1, 1, 3, 3)

    # 计算梯度
    gx1 = F.conv2d(img1, sobel_kernel_x, padding=1)
    gy1 = F.conv2d(img1, sobel_kernel_y, padding=1)

    # 计算梯度幅值
    mag1 = torch.sqrt(gx1 ** 2 + gy1 ** 2 + 1e-8)  # [B, 1, H, W]

    # Mask处理
    if mask is not None:
        valid_mask = (mask > 0.5).float()
        mag1 = mag1 * valid_mask

    plt.figure()
    plt.imshow(mag1.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    plt.show()

    return mag1

def log_ncc(
        img1_tensor,  # 输入图像1, 形状 [B, 1, H, W]
        img2_tensor,  # 输入图像2, 形状 [B, 1, H, W]
        mask_tensor=None,  # 遮挡区域掩码, 形状 [B, 1, H, W]
        sigma=1.0,  # 高斯核的标准差
        kernel_size=5  # LoG核的大小
):
    """
    基于LoG算子的NCC改进版本
    参数说明:
    - img1_tensor, img2_tensor: 灰度图像张量，范围建议 [0,1]
    - mask_tensor: 手术器械遮挡区域的二值掩码（1=遮挡，0=有效区域）
    - sigma: 高斯核的标准差，控制平滑程度
    - kernel_size: LoG核的大小
    """
    if mask_tensor is None:
        mask_tensor = torch.ones_like(img1_tensor)

    device = img1_tensor.device
    img1_tensor = toZeroOne(img1_tensor)
    img2_tensor = toZeroOne(img2_tensor)
    img1 = img1_tensor.float().to(device)
    img2 = img2_tensor.float().to(device)
    mask = mask_tensor.float().to(device)

    # 生成LoG算子
    def create_log_kernel(sigma, kernel_size, normalize=True):
        """创建Laplacian of Gaussian卷积核"""
        x = torch.arange(kernel_size, device=device) - kernel_size // 2
        y = torch.arange(kernel_size, device=device) - kernel_size // 2
        x, y = torch.meshgrid(x, y, indexing='ij')

        r2 = x ** 2 + y ** 2

        # 标准LoG公式
        log_kernel = -(1 - r2 / (2 * sigma ** 2)) * torch.exp(-r2 / (2 * sigma ** 2))

        # 可选：理论归一化（但通常零均值化更重要）
        if normalize:
            log_kernel = log_kernel / (torch.pi * sigma ** 4)

        # 必须零均值化
        log_kernel = log_kernel - log_kernel.mean()

        return log_kernel.view(1, 1, kernel_size, kernel_size)

    # 创建LoG卷积核
    log_kernel = create_log_kernel(sigma, kernel_size)

    # 应用LoG算子
    log1 = F.conv2d(img1, log_kernel, padding=kernel_size // 2)
    log2 = F.conv2d(img2, log_kernel, padding=kernel_size // 2)

    # Mask处理
    if mask is not None:
        valid_mask = (mask > 0.5).float()
        log1 = log1 * valid_mask
        log2 = log2 * valid_mask

    # plt.figure()
    # plt.imshow(log1.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    # plt.show()
    #
    # l1 = torch.clamp(log1, min=0)
    # l2 = torch.clamp(log1, max=0)
    # plt.figure()
    # plt.imshow(l1.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    # plt.show()
    # plt.figure()
    # plt.imshow(l2.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    # plt.show()
    #
    # img1 = img1 - log1
    # img1 = toZeroOne(img1)
    # plt.figure()
    # plt.imshow(img1.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    # plt.show()

    # 计算LoG响应图的NCC
    mean1 = (log1 * valid_mask).sum(dim=[-2, -1], keepdim=True) / (valid_mask.sum(dim=[-2, -1], keepdim=True) + 1e-6)
    mean2 = (log2 * valid_mask).sum(dim=[-2, -1], keepdim=True) / (valid_mask.sum(dim=[-2, -1], keepdim=True) + 1e-6)

    numerator = ((log1 - mean1) * (log2 - mean2)).sum()  # [B]
    denom1 = torch.sqrt(((log1 - mean1) ** 2).sum())  # [B]
    denom2 = torch.sqrt(((log2 - mean2) ** 2).sum())  # [B]

    ncc = numerator / (denom1 * denom2 + 1e-8)  # [B]
    return ncc.mean()

def masked_mean_std(x, mask):
    mu = (x * mask).sum(dim=[-2, -1], keepdim=True) / (mask.sum(dim=[-2, -1], keepdim=True) + 1e-8)
    x_centered = (x - mu) * mask
    var = (x_centered ** 2).sum(dim=[-2, -1], keepdim=True)
    std = var.sqrt()
    return mu, std, x_centered

def enhance_edge(img, transforms, mask):
    if mask is None:
        mask = torch.ones_like(img)
    # 确保输入在GPU上并转换为浮点型
    device = img.device
    img = toZeroOne(img)
    img = img.float().to(device)
    mask = mask.float().to(device)

    # Sobel算子定义
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                  dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                  dtype=torch.float32, device=device).view(1, 1, 3, 3)

    # 计算梯度
    gx = F.conv2d(img, sobel_kernel_x, padding=1)
    gy = F.conv2d(img, sobel_kernel_y, padding=1)

    # 计算梯度幅值
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)  # [B, 1, H, W]

    # Mask处理
    if mask is not None:
        valid_mask = (mask > 0.5).float()
        mag = mag * valid_mask

    # plt.figure()
    # plt.imshow(mag.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    # plt.show()

    img = img - 0.3 * toZeroOne(mag)
    img = toZeroOne(img)
    img = transforms(img, reverse=False)
    # plt.figure()
    # plt.imshow(img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    # plt.show()

    return img

def enhance_by_log(image_tensor, mask):

    # 定义锐化核
    # kernel = torch.tensor(
    #     [[0, -1, 0],
    #      [-1, 5, -1],
    #      [0, -1, 0]], dtype=torch.float32)
    kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
                                  dtype=torch.float32, device=image_tensor.device).view(1, 1, 3, 3)

    # 应用卷积
    sharpened = F.conv2d(image_tensor, kernel, padding=1) * mask
    sharpened = toZeroOne(sharpened) * mask
    # plt.figure()
    # plt.imshow(sharpened.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    # plt.show()

    return sharpened

def enhance_by_log_func(img):
    img = toZeroOne(img) * 9
    img = toZeroOne(torch.log(img + 1))
    # plt.figure()
    # plt.imshow(img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    # plt.show()
    return img

import torch

def dice_coefficient_with_mask(pred, truth, mask, epsilon=1e-6):
    """
    计算 Dice 相似度系数 (Dice Coefficient) 用于两个二值灰度图像，并考虑 mask。
    只有在 mask 中为 1 的位置才会被计入计算。
    :param pred: 预测图像 (Tensor, shape: [batch_size, 1, H, W])
    :param truth: 真实图像 (Tensor, shape: [batch_size, 1, H, W])
    :param mask: 有效区域的 mask (Tensor, shape: [batch_size, 1, H, W])
    :param epsilon: 避免除零的微小常数
    :return: Dice 相似度系数
    """
    # 扁平化图像和 mask
    if mask is None:
        mask = torch.ones_like(pred).to(pred.device)
    # pred_flat = pred.view(-1)
    # truth_flat = truth.view(-1)
    # mask_flat = mask.view(-1)

    # 只考虑 mask 中为 1 的部分
    # pred_flat = pred_flat * mask_flat
    # truth_flat = truth_flat * mask_flat
    pred_flat = pred * mask
    truth_flat = truth * mask

    # 计算交集
    intersection = torch.sum(pred_flat * truth_flat)

    # 计算 Dice 系数
    dice = (2. * intersection + epsilon) / (torch.sum(pred_flat) + torch.sum(truth_flat) + epsilon)
    return dice

def dice_batch(pred, truth, mask, epsilon=1e-6):
    pred = pred * mask
    truth = truth * mask

    # 计算交集
    intersection = torch.sum(pred * truth, dim=[-2, -1])

    # 计算 Dice 系数
    dice = (2. * intersection + epsilon) / (torch.sum(pred, dim=[-2, -1]) + torch.sum(truth, dim=[-2, -1]) + epsilon)
    return dice

##############################################################################
# 示例用法
if __name__ == "__main__":
    # 模拟输入（假设遮挡区域在中心）
    B, C, H, W = 1, 1, 256, 256
    img1 = torch.rand(B, C, H, W)# 随机生成图像1
    img2 = torch.rand(B, C, H, W)  # 随机生成图像2

    # 创建遮挡掩码（中心 100x100 区域为遮挡）
    mask = torch.zeros(B, C, H, W).cuda()
    mask[:, :, H // 2 - 50:H // 2 + 50, W // 2 - 50:W // 2 + 50] = 1

    # 计算梯度一致性（排除遮挡区域）
    dir_consistency, mag_consistency = calculate_gradient_consistency_with_mask(img1, img2, mask)

    print()
