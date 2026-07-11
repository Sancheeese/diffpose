import torch
import torch.nn.functional as F
import os
import glob
from PIL import Image
import numpy as np
import random


class MaskDataAugmentation:
    def __init__(self, mask_dir, resize_to=(256, 256)):
        """
        初始化数据增强类

        Args:
            mask_dir: 包含mask图像的目录路径
            resize_to: 将mask调整到的目标尺寸，默认(256, 256)
        """
        self.mask_dir = mask_dir
        self.resize_to = resize_to
        self.mask_paths = self._load_mask_paths()

    def _load_mask_paths(self):
        """加载目录中所有mask图像的路径"""
        # 支持常见的图像格式
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        mask_paths = []

        for ext in extensions:
            pattern = os.path.join(self.mask_dir, ext)
            mask_paths.extend(glob.glob(pattern))

        if not mask_paths:
            raise ValueError(f"在目录 {self.mask_dir} 中没有找到任何mask图像文件")

        print(f"加载了 {len(mask_paths)} 个mask图像")
        return mask_paths

    def _normalize_tensor(self, tensor):
        """将tensor归一化到0-1范围"""
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

    def _load_and_process_mask(self, mask_path):
        """加载并处理单个mask图像"""
        # 读取图像
        mask = Image.open(mask_path).convert('L')  # 转换为灰度图

        # 调整尺寸
        if mask.size != self.resize_to:
            mask = mask.resize(self.resize_to, Image.Resampling.LANCZOS)

        # 转换为numpy数组并归一化到0-1
        mask_array = np.array(mask, dtype=np.float32)
        mask_array = mask_array / 255.0

        return mask_array

    def augment(self, input_tensor):
        """
        对输入tensor进行数据增强

        Args:
            input_tensor: 形状为 (b, n, 256, 256) 的输入tensor

        Returns:
            augmented_tensor: 增强后的tensor，形状与输入相同
            used_mask_path: 使用的mask路径（用于调试）
        """
        b, n, h, w = input_tensor.shape

        # 1. 对输入tensor进行0-1归一化
        normalized_tensor = self._normalize_tensor(input_tensor)

        # 2. 随机选择一个mask
        mask_path = random.choice(self.mask_paths)
        mask_array = self._load_and_process_mask(mask_path)

        # 3. 将mask转换为tensor并扩展到与输入相同的形状
        mask_tensor = torch.from_numpy(mask_array).to(input_tensor.device)

        # # 扩展mask维度以匹配输入tensor: (h, w) -> (b, n, h, w)
        # mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
        # mask_tensor = mask_tensor.expand(b, n, h, w)

        # 4. 与归一化后的图像相乘
        augmented_tensor = normalized_tensor * mask_tensor

        return augmented_tensor, mask_tensor

    def augment2(self, input_tensor):
        b, n, h, w = input_tensor.shape

        # 1. 对输入tensor进行0-1归一化
        normalized_tensor = self._normalize_tensor(input_tensor)

        # 2. 随机选择一个mask
        mask_path = random.choice(self.mask_paths)
        mask_array = self._load_and_process_mask(mask_path)

        # 3. 将mask转换为tensor并扩展到与输入相同的形状
        mask_tensor = torch.from_numpy(mask_array).to(input_tensor.device)

        # # 扩展mask维度以匹配输入tensor: (h, w) -> (b, n, h, w)
        # mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
        # mask_tensor = mask_tensor.expand(b, n, h, w)

        # 4. 与归一化后的图像相乘
        augmented_tensor = normalized_tensor * mask_tensor

        return augmented_tensor, mask_tensor


    def get_mask_count(self):
        """获取可用的mask数量"""
        return len(self.mask_paths)


# 使用示例
def example_usage():
    # 创建模拟数据
    batch_size = 4
    channels = 1
    height, width = 256, 256

    # 生成随机输入tensor
    input_tensor = torch.randn(batch_size, channels, height, width)
    print(f"输入tensor形状: {input_tensor.shape}")
    print(f"输入范围: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

    # 初始化数据增强器（请将路径替换为实际的mask目录）
    # mask_directory = "/home/zsr/project/diffpose/ours/tube"  # 替换为您的mask目录路径
    mask_directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/tube"  # 替换为您的mask目录路径
    augmenter = MaskDataAugmentation(mask_directory)

    # 进行数据增强
    augmented_tensor, used_mask = augmenter.augment(input_tensor)

    print(f"增强后tensor形状: {augmented_tensor.shape}")
    print(f"增强后范围: [{augmented_tensor.min():.3f}, {augmented_tensor.max():.3f}]")

    return augmented_tensor, used_mask


# 批量处理示例
def batch_augmentation_example():
    """展示如何对多个批次进行处理"""
    # 模拟多个批次的数据
    num_batches = 3
    batch_size = 2
    channels = 1

    # 创建数据增强器
    augmenter = MaskDataAugmentation("/home/zsr/project/diffpose/ours/tube")

    for i in range(num_batches):
        # 生成随机批次数据
        batch_tensor = torch.randn(batch_size, channels, 256, 256)

        # 数据增强
        augmented_batch, mask_used = augmenter.augment(batch_tensor)

        print(f"批次 {i + 1}:")
        print(f"  输入范围: [{batch_tensor.min():.3f}, {batch_tensor.max():.3f}]")
        print(f"  输出范围: [{augmented_batch.min():.3f}, {augmented_batch.max():.3f}]")
        print()


if __name__ == "__main__":
    # 运行示例
    print("数据增强示例:")
    example_usage()

    print("\n批量处理示例:")
    batch_augmentation_example()