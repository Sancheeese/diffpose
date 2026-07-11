import os
import random
from typing import Tuple, List

import cv2
import math
import nrrd
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from matplotlib import pyplot as plt

from ours.utils.CT_dataset import toZeroOne, create_circle_mask
import torch.nn.functional as FF
import matplotlib.colors as mcolors


def center_crop_and_resize_v2(image_tensor, crop_size):
    """
    使用torchvision.transforms实现中心裁剪并缩放

    Args:
        image_tensor: 输入图像张量，形状为 (B, 1, H, W)
        crop_size: 裁剪的正方形边长

    Returns:
        processed_image: 处理后的图像张量，形状与输入相同 (B, 1, H, W)
    """
    # 输入验证
    assert len(image_tensor.shape) == 4, "Input must be 4D tensor (B,1,H,W)"
    assert image_tensor.shape[1] == 1, "Input must have 1 channel"
    assert crop_size > 0, "Crop size must be positive"

    B, C, H, W = image_tensor.shape

    # 创建transform组合
    transform = T.Compose([
        T.CenterCrop(crop_size),  # 中心裁剪
        T.Resize((H, W))  # 缩放回原始尺寸
    ])

    # 对batch中的每张图像应用变换
    processed = torch.stack([
        transform(img.unsqueeze(0)).squeeze(0)  # 处理单张图像
        for img in image_tensor
    ])

    return processed


def batch_crop_largest_square_from_circle(batch_tensor, resize_to_original=True):
    """
    从圆形内容的图像中批量裁剪出最大的内接正方形（批处理版本）

    Args:
        batch_tensor: 输入批处理张量 (B, C, H, W)
        resize_to_original: 是否缩放回原始尺寸

    Returns:
        处理后的批处理张量 (B, C, H, W) 或 (B, C, square_size, square_size)
    """
    # 获取批处理尺寸
    B, C, H, W = batch_tensor.shape

    # 对批处理中的每个图像进行中心裁剪
    cropped_batch = []
    for i in range(B):
        single_image = batch_tensor[i]  # 获取单张图像 (C, H, W)
        cropped = F.center_crop(single_image, [int(H / math.sqrt(2)), int(W / math.sqrt(2))])
        cropped_batch.append(cropped)

    # 堆叠回批处理形式
    cropped_batch = torch.stack(cropped_batch)

    # 如果需要缩放回原始尺寸
    if resize_to_original:
        cropped_batch = F.resize(cropped_batch, [H, W])

    return cropped_batch

def save_tensor_as_image(img, filepath):
    tensor = torch.tensor(img)

    # 移除批次和通道维度，得到(H, W)形状
    image_np = tensor.squeeze(0).squeeze(0).numpy()

    # 归一化到0-255范围
    image_np = toZeroOne(image_np) * 255
    image_np = image_np.astype(np.uint8)

    cv2.imwrite(filepath, image_np)

def more_black(img, threshold=0.5):
    img = torch.clamp(toZeroOne(img), min=0, max=threshold)
    return img
    # return toZeroOne(img)

def print_tre(img, points, color='red'):
    plt.figure()
    plt.imshow(img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    delx = 256 / 305
    for x, y in points:
        x *= delx
        y *= delx
        plt.scatter(x, y, color=color)
    plt.show()

def print_tre_yx(img, points, color='red'):
    plt.figure()
    plt.imshow(img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    delx = 256 / 305
    for x, y in points:
        x *= delx
        y *= delx
        plt.scatter(y, x, color=color)
    plt.show()


# def testWeiNCC(x1, x2, mask, wei, test):
#     if wei is None:
#         wei = torch.ones_like(x1, device=x1.device, dtype=x1.dtype)
#     x1 = norm(x1, mask)
#     x2 = norm(x2, mask)
#     score = (x1[mask == 1] * x2[mask == 1] * wei[mask == 1]).sum()
#     score_test = (x1[test == 1] * x2[test == 1] * wei[test == 1]).sum()
#     score_test2 = (x1[test != 1] * x2[test != 1] * wei[test != 1]).sum()
#
#     a = test.sum() / 256 / 256
#
#     return score

def see_mid(img, mask, low=True):
    tmp = toZeroOne(img) * mask
    mu = tmp.sum() / mask.sum()
    if low:
        tmp[tmp < mu] = 0
    else:
        tmp[tmp > mu] = 1
    plt.figure()
    plt.imshow(tmp.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    plt.show()

    a = mask[tmp < mu].sum()
    print(a / mask.sum())

def norm(x, mask, eps=1e-4):
    mu = (x * mask).sum(dim=[-2, -1], keepdim=True) / (mask.sum(dim=[-2, -1], keepdim=True) + eps)
    x_centered = (x - mu) * mask
    var = (x_centered ** 2).sum(dim=[-2, -1], keepdim=True) / (mask.sum(dim=[-2, -1], keepdim=True) + eps)
    std = var.sqrt()
    return x_centered / (std + eps)


def histogram_matching(source_img, source_mask, reference_img, reference_mask, n_bins=256):
    """
    向量化版本的带掩码直方图匹配 - 更高效
    """
    # 确保输入是2D的 [H, W]
    source_img = torch.round(toZeroOne(source_img) * 255)
    reference_img = torch.round(toZeroOne(reference_img) * 255)
    if source_img.dim() == 3:
        source_img = source_img.squeeze(0)
    if source_mask.dim() == 3:
        source_mask = source_mask.squeeze(0)
    if reference_img.dim() == 3:
        reference_img = reference_img.squeeze(0)
    if reference_mask.dim() == 3:
        reference_mask = reference_mask.squeeze(0)

    device = source_img.device

    # 确保数据类型
    source_img = source_img.float()
    reference_img = reference_img.float()
    source_mask = source_mask.float()
    reference_mask = reference_mask.float()

    # 创建输出图像
    matched_image = source_img.clone()

    # 1. 提取掩码区域的像素
    source_pixels = source_img[source_mask > 0.5]
    reference_pixels = reference_img[reference_mask > 0.5]

    if len(source_pixels) == 0 or len(reference_pixels) == 0:
        return matched_image.unsqueeze(0) if source_img.dim() == 2 else matched_image

    # 2. 计算累积分布函数
    max_val = 1.0 if source_img.max() <= 1 else 255.0

    hist_source = torch.histc(source_pixels, bins=n_bins, min=0, max=max_val)
    cdf_source = hist_source.cumsum(0)
    cdf_source_normalized = cdf_source / cdf_source[-1].clamp(min=1e-8)

    hist_ref = torch.histc(reference_pixels, bins=n_bins, min=0, max=max_val)
    cdf_ref = hist_ref.cumsum(0)
    cdf_ref_normalized = cdf_ref / cdf_ref[-1].clamp(min=1e-8)

    # 3. 构建查找表
    bin_edges = torch.linspace(0, max_val, n_bins + 1, device=device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 为每个bin中心找到映射值
    lut = torch.zeros(n_bins, device=device)
    for i in range(n_bins):
        if cdf_source_normalized[i] <= 0:
            lut[i] = 0.0
        elif cdf_source_normalized[i] >= 1:
            lut[i] = max_val
        else:
            idx = torch.argmax((cdf_ref_normalized >= cdf_source_normalized[i] - 1e-6).float())
            lut[i] = bin_centers[idx]

    # 4. 应用查找表
    mask_indices = source_mask > 0.5
    source_values_masked = source_img[mask_indices]

    # 将像素值映射到bin索引
    bin_indices = torch.clamp((source_values_masked / max_val * (n_bins - 1)).long(), 0, n_bins - 1)

    # 应用映射
    matched_values = lut[bin_indices]
    matched_image[mask_indices] = matched_values

    plt.figure()
    plt.imshow(matched_image.detach().cpu(), cmap='gray')
    plt.show()

    return matched_image.unsqueeze(0) if source_img.dim() == 2 else matched_image


def histogram_equalization(image, mask, n_bins=256):
    """
    带掩码的直方图均衡化
    """
    image = torch.round(toZeroOne(image) * 255)
    # 确保输入是2D的 [H, W]
    dim = image.dim()
    image = image.squeeze()
    mask = mask.squeeze()

    equalized_image = image.clone().float()

    # 提取掩码区域的像素
    masked_pixels = image[mask > 0.5].float()

    if len(masked_pixels) == 0:
        return equalized_image

    # 计算像素值范围
    min_val = masked_pixels.min().item()
    max_val = masked_pixels.max().item()

    # 计算掩码区域的直方图
    hist = torch.histc(masked_pixels, bins=n_bins, min=min_val, max=max_val)

    # 计算累积分布函数 (CDF)
    cdf = hist.cumsum(0)
    cdf_normalized = cdf / cdf[-1]  # 归一化到 [0, 1]

    # 构建查找表：将CDF映射回像素值范围
    lut = cdf_normalized * (max_val - min_val) + min_val

    # 将原始像素值映射到bin索引
    bin_indices = torch.clamp(((masked_pixels - min_val) / (max_val - min_val) * (n_bins - 1)).long(), 0, n_bins - 1)

    # 应用均衡化
    equalized_values = lut[bin_indices]
    equalized_image[mask > 0.5] = equalized_values

    plt.figure()
    plt.imshow(equalized_image.detach().cpu(), cmap='gray')
    plt.show()

    return equalized_image.unsqueeze(0) if dim == 2 else equalized_image

def flip_img_w(img):
    return img.flip(dims=[-1])

def overlay_grayscale_with_red_tensor(background_tensor, overlay_tensor, alpha=0.6):
    """
    将两张灰度Tensor图像叠加，背景保持灰度，叠加图的白色部分变为半透明红色

    参数:
    - background_tensor: 背景灰度Tensor, 形状 [1, 1, 256, 256]
    - overlay_tensor: 叠加灰度Tensor, 形状 [1, 1, 256, 256]
    - alpha: 透明度, 默认0.6

    返回:
    - matplotlib figure对象
    """
    # 将Tensor转换为numpy数组并去除batch和channel维度
    bg_np = background_tensor.squeeze().detach().cpu().numpy()  # 形状 [256, 256]
    overlay_np = overlay_tensor.squeeze().cpu().detach().numpy()  # 形状 [256, 256]

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8))

    bg_np = toZeroOne(bg_np)
    # 显示背景灰度图
    ax.imshow(bg_np, cmap='gray')

    normalized_overlay = toZeroOne(overlay_np)

    # 创建红色RGBA遮罩
    red_mask = np.zeros((256, 256, 4))  # RGBA数组

    # 设置红色通道 (R=1.0, G=0.0, B=0.0)
    red_mask[..., 1] = 1.0  # 红色
    # 设置透明度基于原图亮度
    red_mask[..., 3] = normalized_overlay * alpha

    # 叠加显示红色遮罩
    ax.imshow(red_mask)

    # 美化图形
    ax.axis('off')
    # plt.tight_layout()
    plt.show()
    return fig

def overlay_grayscale_with_red_tensor_save(background_tensor, overlay_tensor, path, alpha=0.6):
    """
    将两张灰度Tensor图像叠加，背景保持灰度，叠加图的白色部分变为半透明红色

    参数:
    - background_tensor: 背景灰度Tensor, 形状 [1, 1, 256, 256]
    - overlay_tensor: 叠加灰度Tensor, 形状 [1, 1, 256, 256]
    - alpha: 透明度, 默认0.6

    返回:
    - matplotlib figure对象
    """
    # 将Tensor转换为numpy数组并去除batch和channel维度
    bg_np = background_tensor.squeeze().detach().cpu().numpy()  # 形状 [256, 256]
    overlay_np = overlay_tensor.squeeze().cpu().detach().numpy()  # 形状 [256, 256]

    # 创建图形
    fig, ax = plt.subplots(dpi=300)

    bg_np = toZeroOne(bg_np)
    # 显示背景灰度图
    ax.imshow(bg_np, cmap='gray')

    normalized_overlay = toZeroOne(overlay_np)

    # 创建红色RGBA遮罩
    red_mask = np.zeros((256, 256, 4))  # RGBA数组

    # 设置红色通道 (R=1.0, G=0.0, B=0.0)
    red_mask[..., 1] = 1.0  # 红色
    # 设置透明度基于原图亮度
    red_mask[..., 3] = normalized_overlay * alpha

    # 叠加显示红色遮罩
    ax.imshow(red_mask)

    # 美化图形
    ax.axis('off')
    # plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()
    return fig

def overlay_grayscale_with_red_tensor_save2(
        background_tensor,
        overlay_tensor,
        path,
        alpha=0.6):

    # 转 numpy
    bg_np = background_tensor.squeeze().detach().cpu().numpy()
    overlay_np = overlay_tensor.squeeze().detach().cpu().numpy()

    # 归一化
    bg_np = toZeroOne(bg_np)
    overlay_np = toZeroOne(overlay_np)

    if bg_np.ndim == 2:
        # -------- 1️⃣ 背景转 RGB --------
        bg_rgb = np.stack([bg_np, bg_np, bg_np], axis=-1)
    else:
        bg_rgb = bg_np

    # 构造绿色图
    green_rgb = np.zeros_like(bg_rgb)
    green_rgb[..., 1] = 1.0  # 绿色

    # alpha mask
    alpha_mask = overlay_np * alpha
    alpha_mask = alpha_mask[..., None]

    # 融合
    out = (1 - alpha_mask) * bg_rgb + alpha_mask * green_rgb

    # -------- 5️⃣ 显示 --------
    plt.figure(dpi=300)
    plt.imshow(out)
    plt.axis("off")
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

    return out

def canny_edge_torch(img, visualize=True):
    img_np = img.squeeze().detach().cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    edges = cv2.Canny(img_np, threshold1=50, threshold2=150)

    # 可视化
    if visualize:
        plt.figure(dpi=200)
        plt.imshow(edges, cmap='gray')
        plt.title("Canny Edge")
        plt.axis('off')
        plt.show()

    edges = torch.from_numpy(edges).float().to(img.device) / 255.0
    return edges.unsqueeze(0).unsqueeze(0)


def inpaint_with_opencv(img_tensor, mask_tensor, method=cv2.INPAINT_TELEA):
    """
    使用 OpenCV 进行图像修复。

    参数:
        img_tensor: (C, H, W) 范围 [0, 1] 的 PyTorch 张量。
        mask_tensor: (H, W) 的 PyTorch 张量，需要修复的区域为 1。
        method: cv2.INPAINT_TELEA 或 cv2.INPAINT_NS。

    返回:
        修复后的 PyTorch 张量。
    """
    # 1. 将 PyTorch Tensor 转换为 OpenCV 需要的 NumPy 格式
    # PyTorch: (C, H, W) -> NumPy: (H, W, C), 范围 [0, 255]
    img_tensor = toZeroOne(img_tensor)
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
    img_np = img_np.astype(np.uint8)

    # 2. 处理掩码
    mask_np = mask_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255 # 范围 0 或 255

    # 3. 调用 OpenCV 的修复函数
    result_np = cv2.inpaint(img_np, mask_np, inpaintRadius=3, flags=method)

    # 4. 将结果转换回 PyTorch Tensor
    result_tensor = torch.from_numpy(result_np.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    result_tensor = result_tensor.to(img_tensor.device)

    return result_tensor


from einops import rearrange
def rank_transform_tensor(input_tensor, mask=None, window_size=5, padding_mode='reflect'):
    """
    针对PyTorch Tensor的秩变换，支持mask

    参数:
        input_tensor: 输入tensor, 形状为 (1, 1, H, W) 或 (1, H, W)
        mask: 掩码tensor, 形状与input_tensor相同, 1表示有效区域，0表示忽略
        window_size: 邻域窗口大小，必须是奇数
        padding_mode: 填充模式 ('reflect', 'zeros', 'replicate')

    返回:
        rank_tensor: 秩变换后的tensor, 形状与输入相同
    """
    assert window_size % 2 == 1, "window_size必须是奇数"

    # 确保输入tensor形状为 (1, 1, H, W)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # (1, H, W) -> (1, 1, H, W)
    elif input_tensor.dim() == 4 and input_tensor.shape[1] != 1:
        raise ValueError("输入tensor的通道数必须为1")

    # 如果没有提供mask，创建全1的mask
    if mask is None:
        mask = torch.ones_like(input_tensor)
    else:
        # 确保mask形状与input_tensor一致
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        mask = mask.to(input_tensor.device)

    device = input_tensor.device
    batch_size, channels, height, width = input_tensor.shape
    offset = window_size // 2

    # 对输入进行填充
    padded_input = F.pad(input_tensor, [offset, offset, offset, offset], padding_mode=padding_mode)
    padded_mask = F.pad(mask, [offset, offset, offset, offset], padding_mode=padding_mode)

    # 初始化输出
    rank_tensor = torch.zeros_like(input_tensor)
    total_pixels = window_size * window_size

    # 提取所有局部窗口
    unfolded_input = padded_input.unfold(2, window_size, step=1).unfold(3, window_size, step=1).contiguous()
    unfolded_mask = padded_mask.unfold(2, window_size, step=1).unfold(3, window_size, step=1).contiguous()

    patches_input = unfolded_input.permute(0, 1, 4, 5, 2, 3).contiguous()
    patches_mask = unfolded_mask.permute(0, 1, 4, 5, 2, 3).contiguous()
    # 重塑为 (batch, channels, w*w, H, W)
    patches_input = patches_input.contiguous().view(
        batch_size, channels, window_size * window_size, height, width
    )  # (1, 1, window_size*window_size, H, W)

    patches_mask = patches_mask.contiguous().view(
        batch_size, channels, window_size * window_size, height, width
    )

    # 获取中心像素值 (在patches的中间位置)
    center_idx = (window_size * window_size) // 2
    center_values = patches_input[:, :, center_idx, :, :]  # (1, 1, H, W)

    # 计算秩：统计每个位置小于等于中心值的像素数量
    center_values_expanded = center_values.unsqueeze(2)  # (1, 1, 1, H, W)

    # 比较并考虑mask
    comparisons = (patches_input <= center_values_expanded).float()  # (1, 1, w*w, H, W)
    valid_comparisons = comparisons * (patches_mask > 0.5).float()

    # 计算每个位置的有效秩
    ranks = torch.sum(valid_comparisons, dim=2)  # (1, 1, H, W)

    # 计算每个位置的有效像素数
    valid_pixels = torch.sum((patches_mask > 0.5).float(), dim=2)
    valid_pixels = torch.clamp(valid_pixels, min=1.0)

    # 归一化到 [0, 1]
    rank_normalized = ranks / valid_pixels

    # 应用mask
    rank_tensor = rank_normalized * mask

    return rank_tensor

def read_seg(filename):
    # data, header = nrrd.read("/home/zsr/project/diffpose/ours/bone_seg/zyl_4.nrrd")
    data, header = nrrd.read(filename)
    data = torch.from_numpy(data).float()
    data = data.squeeze()
    data = data.permute(1, 0)

    plt.figure()
    plt.imshow(data)
    plt.show()
    data =  data.unsqueeze(0).unsqueeze(0)
    data = FF.interpolate(data, [256, 256], mode='bilinear')

    return data

def get_streak_artifacts(
    height: int,
    width: int,
    num_streaks_range: Tuple[int, int],
    width_range: Tuple[float, float],
    length_range: Tuple[float, float],
    shape_types: List[str],
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)

    num_streaks = random.randint(*num_streaks_range)

    for _ in range(num_streaks):
        streak_width = int(width * random.uniform(*width_range))
        streak_length = int(height * random.uniform(*length_range))
        shape_type = random.choice(shape_types)

        centerline = generate_centerline(
            height, width, streak_length, shape_type
        )

        if len(centerline) > 1:
            for i in range(len(centerline) - 1):
                pt1 = tuple(map(int, centerline[i]))
                pt2 = tuple(map(int, centerline[i + 1]))
                cv2.line(mask, pt1, pt2, 1.0, streak_width)

    return mask

def generate_centerline(
    height: int,
    width: int,
    length: int,
    shape_type: str,
) -> np.ndarray:
    start_y = random.randint(0, height)
    start_x = random.randint(0, width)

    # 随机方向角度
    angle = random.uniform(0, 2 * math.pi)
    # shape_type = 'curved'

    if shape_type == 'straight':
        # 直线
        end_x = start_x + length * math.cos(angle)
        end_y = start_y + length * math.sin(angle)
        return np.array([[start_x, start_y], [end_x, end_y]])

    elif shape_type == 'curved':
        # 曲线
        num_points = random.randint(3, 5)
        points = [[start_x, start_y]]

        for i in range(1, num_points):
            seg_length = length / (num_points - 1)
            curve_angle = angle + random.uniform(0, 1.2)  # 轻微弯曲

            x = points[-1][0] + seg_length * math.cos(curve_angle)
            y = points[-1][1] + seg_length * math.sin(curve_angle)
            points.append([x, y])
        return np.array(points)

    elif shape_type == 'curved2':
        # 简化的自然曲线生成
        num_points = 15  # 更多点使曲线更平滑
        points = []

        # 使用正弦函数生成自然弯曲
        curve_frequency = random.uniform(0.5, 2.0)  # 弯曲频率
        curve_amplitude = random.uniform(0.1, 0.3) * length  # 弯曲幅度

        for i in range(num_points):
            t = i / (num_points - 1)

            # 主要方向
            x_main = start_x + t * length * math.cos(angle)
            y_main = start_y + t * length * math.sin(angle)

            # 垂直方向的弯曲
            perpendicular_angle = angle + math.pi / 2  # 垂直于主方向
            bend = curve_amplitude * math.sin(t * math.pi * curve_frequency)

            x = x_main + bend * math.cos(perpendicular_angle)
            y = y_main + bend * math.sin(perpendicular_angle)

            # 确保点在图像范围内
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))

            points.append([x, y])

        return np.array(points)

def compute_iou(pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor = None,
                eps: float = 1e-6) -> torch.Tensor:
    """
    Compute IoU for tensors of shape (1, 1, H, W) with optional mask.

    Args:
        pred: predicted tensor, shape (1, 1, H, W), binary or probabilistic
        gt:   ground truth tensor, shape (1, 1, H, W), binary
        mask: optional mask tensor, same shape, 1 means valid region
        eps:  numerical stability

    Returns:
        iou: scalar tensor
    """

    assert pred.shape == gt.shape, "pred and gt must have the same shape"
    assert pred.dim() == 4, "input tensor must be 4D (B, C, H, W)"

    if mask is None:
        mask = torch.ones_like(pred)

    # 保证是 float
    pred = pred.float()
    gt = gt.float()
    mask = mask.float()

    # 只在 mask 区域计算
    pred = pred * mask
    gt = gt * mask

    intersection = torch.sum(pred * gt)
    union = torch.sum(pred) + torch.sum(gt) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou

def save_err(img1, img2, path):
    cmap = plt.get_cmap("bwr").copy()
    cmap.set_bad(color="black")
    err = toZeroOne(img1) - toZeroOne(img2)
    # err[err < 0] = toZeroOne(err[err < 0]) - 1
    # err[err > 0] = toZeroOne(err[err > 0])
    norm = mcolors.TwoSlopeNorm(
        vmin=-0.5,  # 最小值
        vcenter=0,  # 中心点
        vmax=0.5  # 最大值
    )
    plt.figure(dpi=300)
    plt.imshow(err.detach().cpu().squeeze(), cmap=cmap, norm=norm)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

def save_err_with_black(img1, img2, path, radius):
    circle_mask = create_circle_mask(256, radius).to(img1.device).unsqueeze(0).unsqueeze(0)
    cmap = plt.get_cmap("bwr").copy()
    cmap.set_bad(color="black")
    err = toZeroOne(img1) - toZeroOne(img2)
    # err[err < 0] = toZeroOne(err[err < 0]) - 1
    # err[err > 0] = toZeroOne(err[err > 0])
    err[circle_mask == 0] = torch.nan
    norm = mcolors.TwoSlopeNorm(
        vmin=-0.5,  # 最小值
        vcenter=0,  # 中心点
        vmax=0.5  # 最大值
    )
    plt.figure(dpi=300)
    plt.imshow(err.detach().cpu().squeeze(), cmap=cmap, norm=norm)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

def err_half_lr(img1, img2, img_half, path=None, err_right=True):
    circle_mask = create_circle_mask(256, 119).to(img1.device).unsqueeze(0).unsqueeze(0)
    err = toZeroOne(img1) - toZeroOne(img2)
    err[circle_mask == 0] = torch.nan
    err = err.detach().cpu().squeeze().numpy()
    norm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    cmap = plt.get_cmap("bwr").copy()
    cmap.set_bad(color="black")
    err_rgb = cmap(norm(err))[:, :, :3]

    gray_img = img_half.detach().cpu().squeeze().numpy()
    gray_img = toZeroOne(gray_img)
    if gray_img.ndim == 2:
        # 单通道 → 转RGB
        gray_rgb = plt.get_cmap("gray")(gray_img)[..., :3]
    else:
        # 已经是RGB
        gray_rgb = gray_img

    H, W, _ = gray_rgb.shape
    mid = W // 2
    if err_right:
        left = gray_rgb[:, :mid, :]
        right = err_rgb[:, mid:, :]
    else:
        right = gray_rgb[:, mid:, :]
        left = err_rgb[:, :mid, :]
    out = np.concatenate([left, right], axis=1)
    plt.figure(dpi=300)
    plt.imshow(out)
    plt.axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()
    return out

def err_half_ud(img1, img2, img_half, path=None, err_down=True):
    circle_mask = create_circle_mask(256, 119).to(img1.device).unsqueeze(0).unsqueeze(0)
    err = toZeroOne(img1) - toZeroOne(img2)
    err[circle_mask == 0] = torch.nan
    err = err.detach().cpu().squeeze().numpy()
    norm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    cmap = plt.get_cmap("bwr").copy()
    cmap.set_bad(color="black")
    err_rgb = cmap(norm(err))[:, :, :3]

    gray_img = img_half.detach().cpu().squeeze().numpy()
    gray_img = toZeroOne(gray_img)
    if gray_img.ndim == 2:
        # 单通道 → 转RGB
        gray_rgb = plt.get_cmap("gray")(gray_img)[..., :3]
    else:
        # 已经是RGB
        gray_rgb = gray_img

    H, W, _ = gray_rgb.shape
    mid = W // 2
    if err_down:
        up = gray_rgb[:mid, :, :]
        down = err_rgb[mid:, :, :]
    else:
        down = gray_rgb[mid:, :, :]
        up = err_rgb[:mid, :, :]
    out = np.concatenate([up, down], axis=0)
    plt.figure(dpi=300)
    plt.imshow(out)
    plt.axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()
    return out

def err_half_4(img1, img2, img_half, path=None, err_up=True):
    circle_mask = create_circle_mask(256, 119).to(img1.device).unsqueeze(0).unsqueeze(0)
    err = toZeroOne(img1) - toZeroOne(img2)
    err[circle_mask == 0] = torch.nan
    err = err.detach().cpu().squeeze().numpy()
    norm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    cmap = plt.get_cmap("bwr").copy()
    cmap.set_bad(color="black")
    err_rgb = cmap(norm(err))[:, :, :3]

    gray_img = img_half.detach().cpu().squeeze().numpy()
    gray_img = toZeroOne(gray_img)
    if gray_img.ndim == 2:
        # 单通道 → 转RGB
        gray_rgb = plt.get_cmap("gray")(gray_img)[..., :3]
    else:
        # 已经是RGB
        gray_rgb = gray_img

    out = quad_mix(gray_rgb, err_rgb)
    plt.figure(dpi=300)
    plt.imshow(out)
    plt.axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()
    return out

def quad_mix(gray_rgb, err_rgb):
    H, W, _ = gray_rgb.shape

    h_mid = H // 2
    w_mid = W // 2

    # 四个象限
    tl = gray_rgb[:h_mid, :w_mid]  # 第一象限
    tr = err_rgb[:h_mid, w_mid:]  # 第二象限
    bl = err_rgb[h_mid:, :w_mid]  # 第三象限
    br = gray_rgb[h_mid:, w_mid:]  # 第四象限

    # 上半部分
    top = np.concatenate([tl, tr], axis=1)

    # 下半部分
    bottom = np.concatenate([bl, br], axis=1)

    # 拼成完整图
    out = np.concatenate([top, bottom], axis=0)

    return out

def save_err_half(img1, img2, img_ori, overlay, edge, prefix, idx):
    err_half_lr(img1, img2, overlay, prefix + f"_err_lr_{idx}.png")
    err_half_ud(img1, img2, overlay, prefix + f"_err_ud_{idx}.png")
    err_half_4(img1, img2, overlay, prefix + f"_err_4_{idx}.png")

    out1 = torch.tensor(err_half_lr(img1, img2, img_ori))
    out2 = torch.tensor(err_half_ud(img1, img2, img_ori))
    out3 = torch.tensor(err_half_4(img1, img2, img_ori))

    overlay_grayscale_with_red_tensor_save2(out1, edge, prefix + f"_erroverlay_lr_{idx}.png", alpha=1)
    overlay_grayscale_with_red_tensor_save2(out2, edge, prefix + f"_erroverlay_ud_{idx}.png", alpha=1)
    overlay_grayscale_with_red_tensor_save2(out3, edge, prefix + f"_erroverlay_4_{idx}.png", alpha=1)



if __name__ == "__main__":
    for i in range(10):
        mask = get_streak_artifacts(
            height=256,
            width=256,
            num_streaks_range=(1, 4),
            width_range=(0.05, 0.15),
            length_range=(0.6, 1.0),
            shape_types=["straight", "curved", "curved2"],
        )

        plt.figure(figsize=(5, 5))
        plt.imshow(mask, cmap="gray")
        plt.title("Instrument Occlusion Mask")
        plt.axis("off")
        plt.show()




