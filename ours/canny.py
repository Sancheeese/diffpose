import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

from ours.case.ysy.CT_dataset import IntubationDataset, toZeroOne


def canny_edge(img_tensor, low=50, high=150):
    """
    标准 OpenCV Canny

    img_tensor: (1,1,H,W), float [0,1]
    return: (1,1,H,W), float [0,1]
    """

    device = img_tensor.device

    # 转 numpy
    img = img_tensor.squeeze().detach().cpu().numpy()
    img = toZeroOne(img)
    img = (img * 255).astype(np.uint8)
    img_blur = cv2.GaussianBlur(img, (7, 7), 3)
    edges = cv2.Canny(img_blur, low, high)

    # 转回 tensor
    edges = torch.from_numpy(edges.astype(np.float32) / 255.0)
    edges = edges.unsqueeze(0).unsqueeze(0).to(device)

    plt.figure()
    plt.imshow(edges.cpu().squeeze(), cmap='gray')
    plt.show()

    return edges

if __name__ == "__main__":
    root = "/home/zsr/project/diffpose/ours/data/liwei/杨式瑜/CT/YangShiYu/20240315081250.536000/7"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/杨式瑜/ERCP/SHIYU^YANG^/20240515170510/1"
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/杨式瑜/CT/YangShiYu/20240315081250.536000/7"
    # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/杨式瑜/ERCP/SHIYU^YANG^/20240515170510/1"

    specimen = IntubationDataset(root, x_root, y_offset=50, z_offset=-50, z_cut=400, factors=[0.7, 1.5, 0.7])
    img, pose = specimen[13]

    plt.figure()
    plt.imshow(img.cpu().squeeze(), cmap='gray')
    plt.show()

    low = 10
    step = 5
    for i in range(10):
        high = canny_edge(img, low=low/3, high=low)
        low += step