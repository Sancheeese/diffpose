import os

import cv2
import torch
from PIL import Image
import numpy as np

from ours.case.cyx.CT_dataset import toZeroOne
from ours.utils.CT_dataset import Transforms

transformer = Transforms(256, radius=119)
def invert_png_images_in_folder(folder_path):
    """
    反转指定文件夹下所有PNG图像的颜色并保存回原文件

    Args:
        folder_path (str): 包含PNG文件的文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return

    # 获取文件夹中所有的PNG文件
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]

    if not png_files:
        print(f"在文件夹 '{folder_path}' 中没有找到PNG文件")
        return

    print(f"找到 {len(png_files)} 个PNG文件")

    # 处理每个PNG文件
    for filename in png_files:
        file_path = os.path.join(folder_path, filename)

        try:
            # 打开图像
            with Image.open(file_path) as img:
                # 转换为numpy数组进行处理
                img_array = np.array(img)

                # 反相处理 (255 - 像素值)
                img_tensor = torch.tensor(img_array).unsqueeze(0)
                img_tensor = transformer(img_tensor).squeeze()
                img_tensor = toZeroOne(img_tensor) * 255
                img = np.array(img_tensor).astype(np.uint8)

                cv2.imwrite(file_path, img)

            print(f"已处理: {filename}")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    print("所有PNG文件反相处理完成！")


# 使用示例
if __name__ == "__main__":
    folder_path = "/home/zsr/project/diffpose/ours/drrStyle_iso3/trainB"
    invert_png_images_in_folder(folder_path)