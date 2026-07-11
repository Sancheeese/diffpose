import torch
import matplotlib.pyplot as plt


def histogram_matching_torch(src_tensor: torch.Tensor,
                             target_tensor: torch.Tensor,
                             bins: int = 1000,
                             epsilon: float = 1e-6) -> torch.Tensor:
    """
    PyTorch 实现的直方图匹配（支持 GPU）
    :param src_tensor: 源张量（范围 ~[-4, 4]）
    :param target_tensor: 目标张量（范围 ~[-4, 4]）
    :param bins: 直方图分箱数（控制精度）
    :param epsilon: 防止除以零的小量
    :return: 匹配后的张量
    """
    device = src_tensor.device

    # 展平张量并移动到 CPU（PyTorch 的直方图函数需 CPU）
    src_flat = src_tensor.flatten().cpu().float()
    target_flat = target_tensor.flatten().cpu().float()

    # 计算直方图分箱边界（固定范围 [-4, 4]）
    bin_edges = torch.linspace(-4, 4, bins + 1, device='cpu')

    # 计算源和目标的直方图
    src_hist = torch.histc(src_flat, bins=bins, min=-4, max=4)
    target_hist = torch.histc(target_flat, bins=bins, min=-4, max=4)

    # 计算归一化 CDF
    src_cdf = (src_hist.cumsum(dim=0) + epsilon) / (src_hist.sum() + epsilon)
    target_cdf = (target_hist.cumsum(dim=0) + epsilon) / (target_hist.sum() + epsilon)

    # 计算像素值到目标 CDF 的映射（使用插值）
    src_values = (bin_edges[:-1] + bin_edges[1:]) / 2  # 取分箱中点作为代表值
    target_values = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 将源 CDF 映射到目标 CDF
    matched_indices = torch.searchsorted(target_cdf, src_cdf, right=True)
    matched_indices = torch.clamp(matched_indices, 0, len(target_values) - 1)

    # 构建映射表（源分箱值 → 目标分箱值）
    mapping = target_values[matched_indices]

    # 将源张量的每个像素值映射到目标值（线性插值）
    bin_width = 8 / bins  # 分箱宽度（-4到4总跨度为8）
    bin_indices = torch.floor((src_flat + 4) / bin_width).long()
    bin_indices = torch.clamp(bin_indices, 0, bins - 1)

    matched_flat = mapping[bin_indices]

    # 还原形状并移回原设备
    matched_tensor = matched_flat.reshape(src_tensor.shape).to(device)
    return matched_tensor


# ------------------- 示例使用 -------------------
if __name__ == "__main__":
    # 1. 生成模拟数据（范围 [-4, 4]）
    torch.manual_seed(42)
    src_tensor = torch.randn(512, 512) * 1.0  # 低对比度
    src_tensor = torch.clamp(src_tensor, -4, 4)

    target_tensor = torch.randn(512, 512) * 2.0  # 高对比度
    target_tensor = torch.clamp(target_tensor, -4, 4)

    # 2. 执行直方图匹配
    matched_tensor = histogram_matching_torch(src_tensor, target_tensor)

    # 3. 可视化对比
    plt.figure(figsize=(15, 5))

    # 源图像直方图
    plt.subplot(1, 3, 1)
    plt.hist(src_tensor.numpy().flatten(), bins=100, range=(-4, 4), color='r', alpha=0.5)
    plt.title("Source Histogram")

    # 目标图像直方图
    plt.subplot(1, 3, 2)
    plt.hist(target_tensor.numpy().flatten(), bins=100, range=(-4, 4), color='g', alpha=0.5)
    plt.title("Target Histogram")

    # 匹配后图像直方图
    plt.subplot(1, 3, 3)
    plt.hist(matched_tensor.numpy().flatten(), bins=100, range=(-4, 4), color='b', alpha=0.5)
    plt.title("Matched Histogram")

    plt.tight_layout()
    plt.show()