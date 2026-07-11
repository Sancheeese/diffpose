import random

import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import splprep, splev

def generate_custom_s_shape_centerline(
    image_shape,
    num_points=9,
    length_ratio=0.6,  # 第一段长度占总高度比例
    curvature1=0.02,   # 第一段弯曲强度（小）
    curvature2=0.1,    # 第二段弯曲强度（大）
    reverse=False,
    noise_strength=0.0001
):
    """
    生成一个可调节段长度与弯曲程度的 S 形管道中心线（从上到下）

    参数说明：
    - length_ratio: 第一段所占的高度比例（0~1之间）
    - curvature1: 第一段的弯曲幅度（建议 <= 0.1）
    - curvature2: 第二段的弯曲幅度（建议 >= 0.1）
    - reverse: False=S形，True=反S形
    """
    h, w = image_shape
    if random.random() < 0.7:
        start = (np.random.randint(0.6 * w, 0.8 * w), np.random.randint(0.02 * h, 0.08 * h))
        end = (np.random.randint(0.4 * w, 0.6 * w), np.random.randint(0.75 * h, 0.9 * h))
    elif random.random() < 0.85:
        start = (np.random.randint(0.1 * w, 0.6 * w), np.random.randint(0.02 * h, 0.08 * h))
        end = (np.random.randint(0.4 * w, 0.6 * w), np.random.randint(0.75 * h, 0.9 * h))
    else:
        start = (np.random.randint(0.1 * w, 0.6 * w), np.random.randint(0.02 * h, 0.08 * h))
        end = (np.random.randint(0.4 * w, 0.6 * w), np.random.randint(0.5 * h, 0.75 * h))
    start_x, start_y = start
    end_x, end_y = end

    control_points = []

    for i in range(num_points):
        frac = i / (num_points - 1)
        y = start_y + frac * (end_y - start_y)

        if frac < length_ratio:
            # 第一段
            direction = -1 if not reverse else 1
            curvature = curvature1
            local_frac = frac / length_ratio  # 归一化到0~1
        else:
            # 第二段
            direction = 1 if not reverse else -1
            curvature = curvature2
            local_frac = (frac - length_ratio) / (1 - length_ratio)  # 归一化

        x_offset = direction * curvature * w * np.sin(np.pi * local_frac)

        # 倾斜趋势
        x_trend = start_x + frac * (end_x - start_x)
        # x_trend = start_x + frac * (end_x - start_x)

        # 加入扰动
        noise_x = np.random.uniform(-noise_strength * w, noise_strength * w)
        noise_y = np.random.uniform(-noise_strength * h, noise_strength * h)

        x = x_trend + x_offset + noise_x
        y = y + noise_y

        control_points.append((x, y))

    # 样条拟合平滑
    tck, _ = splprep(np.array(control_points).T, s=5)
    u = np.linspace(0, 1, 100)
    centerline = np.column_stack(splev(u, tck)).astype(int)
    return centerline


def draw_tube_on_image(image, centerline, tube_radius=4, attenuation=0.3, show=True):
    """
    在图像上绘制模拟的导管（圆管），并显示结果。

    参数：
        image: 原始图像（H x W 或 H x W x 3）
        centerline: 中心线坐标 [N, 2]
        tube_radius: 管子半径（像素）
        attenuation: 衰减系数（越小越黑）
        show: 是否直接显示图像
    返回：
        image_with_tube: 画完管子的图像
        mask: 管子的灰度掩码
    """
    gray = image.copy()
    h, w = gray.shape[-1], gray.shape[-2]

    mask = np.zeros_like(gray, dtype=np.uint8)

    # 用白色粗线画出管子区域
    for i in range(len(centerline) - 1):
        pt1 = tuple(map(int, centerline[i]))
        pt2 = tuple(map(int, centerline[i + 1]))
        cv2.line(mask, pt1, pt2, 255, tube_radius * 2)

    # 模糊掩码边缘
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=tube_radius // 2)
    mask = mask.astype(np.float32) / 255.0

    # 图像融合，生成导管效果
    tube_img = gray * (1 - mask) + gray * attenuation * mask

    if show:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(gray, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Tube Mask")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Image with Tube")
        plt.imshow(tube_img, cmap='gray')
        plt.axis('off')
        plt.show()

    return tube_img, mask

def get_tube_on_image(image, tube_radius=4, attenuation=0.2, black=True):
    size = (image.shape[-1], image.shape[-2])
    num_points = np.random.randint(5, 12)
    length_ratio = np.random.uniform(0.5, 0.7) # 第一段长度占总高度比例
    curvature1 = np.random.uniform(0.01, 0.03)  # 第一段弯曲强度（小）
    curvature2 = np.random.uniform(0.03, 0.2) # 第二段弯曲强度（大）
    reverse = False if random.random() > 0.5 else True
    noise_strength = 0.0
    tube_radius = np.random.randint(5, 10)

    centerline = generate_custom_s_shape_centerline(size, num_points, length_ratio, curvature1, curvature2, reverse, noise_strength)
    # 创建一个与图像大小相同的掩码
    mask = np.zeros(size, dtype=np.uint8)

    # 用白色粗线画出管子区域
    for i in range(len(centerline) - 1):
        pt1 = tuple(map(int, centerline[i]))
        pt2 = tuple(map(int, centerline[i + 1]))
        cv2.line(mask, pt1, pt2, 255, tube_radius * 2)

    # 模糊掩码边缘
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1)
    mask = mask.astype(np.float32) / 255.0

    # 将 mask 转为 GPU 张量
    mask_tensor = torch.tensor(mask, device=image.device).float()
    random_deep = np.random.uniform(0.5, 0.9)
    image = (image - image.min()) / (image.max() - image.min())
    if not black:
        # image = image * mask_pos + mask_tensor
        image = image * (1 - mask_tensor) + random_deep * mask_tensor
    else:
        image = image * (1 - mask_tensor)

    # 对每一张图像进行处理
    # for i in range(1):
    #     plt.figure()
    #     plt.imshow(image[i].squeeze(0).cpu(), cmap='gray')
    #     plt.show()

    return image, mask_tensor

if __name__ == '__main__':
    img = torch.ones((1, 1, 256, 256), device=torch.device('cuda:0'))
    get_tube_on_image(img)

    print()

