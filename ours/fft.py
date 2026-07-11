import matplotlib.pyplot as plt
import torch
import torch.fft as fft

from ours.case.sxh.CT_dataset import IntubationDataset, toZeroOne
from ours.utils.CT_dataset import Transforms
from ours.utils.drr import DRR

transforms = Transforms(256, radius=119)
def high_pass_fft(img, cutoff_ratio=0.1):
    """
    对 (1,1,H,W) 图像做 FFT 高通滤波

    参数:
        img: torch.Tensor (1,1,H,W), float
        cutoff_ratio: 低频半径比例 (0~0.5)，越小保留的高频越多

    返回:
        high_freq_img: torch.Tensor (1,1,H,W)
    """

    B, C, H, W = img.shape
    device = img.device

    # 1️⃣ FFT
    freq = fft.fft2(img)
    freq = fft.fftshift(freq)

    # 2️⃣ 构造高通 mask
    y = torch.arange(H, device=device).reshape(H, 1)
    x = torch.arange(W, device=device).reshape(1, W)

    center_y = H // 2
    center_x = W // 2

    dist = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

    cutoff = cutoff_ratio * min(H, W)

    mask = (dist > cutoff).float()
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # 3️⃣ 应用 mask
    freq_filtered = freq * mask

    # 4️⃣ 逆变换
    freq_filtered = fft.ifftshift(freq_filtered)
    img_high = fft.ifft2(freq_filtered)

    # 5️⃣ 取实部
    img_high = img_high.real

    # plt.figure()
    # plt.imshow(img_high.cpu().squeeze(), cmap='gray')
    # plt.title(f"{cutoff_ratio}")
    # plt.show()

    # 能量归一化
    # img_high = img_high / (img_high.abs().mean() + 1e-6)

    return img_high


def high_pass_gaussian(img, sigma_ratio=0.1):
    B, C, H, W = img.shape
    device = img.device

    freq = fft.fft2(img)
    freq = fft.fftshift(freq)

    y = torch.arange(H, device=device).reshape(H, 1)
    x = torch.arange(W, device=device).reshape(1, W)

    center_y = H // 2
    center_x = W // 2

    dist_sq = (y - center_y) ** 2 + (x - center_x) ** 2

    sigma = sigma_ratio * min(H, W)

    gaussian_low = torch.exp(-dist_sq / (2 * sigma ** 2))
    high_mask = 1 - gaussian_low

    high_mask = high_mask.unsqueeze(0).unsqueeze(0)

    freq_filtered = freq * high_mask

    freq_filtered = fft.ifftshift(freq_filtered)
    img_high = fft.ifft2(freq_filtered)
    img_high = img_high.real

    # img_high = transforms(img_high, reverse=False)

    # plt.figure()
    # plt.imshow(img_high.cpu().squeeze(), cmap='gray')
    # plt.title(f"{sigma_ratio}")
    # plt.show()

    return img_high


if __name__ == "__main__":
    device = 'cuda:0'
    height = 256
    # root = "/home/zsr/project/diffpose/ours/data/liwei/杨式瑜/CT/YangShiYu/20240315081250.536000/7"
    # x_root = "/home/zsr/project/diffpose/ours/data/liwei/杨式瑜/ERCP/SHIYU^YANG^/20240515170510/1"
    # # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/杨式瑜/CT/YangShiYu/20240315081250.536000/7"
    # # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/杨式瑜/ERCP/SHIYU^YANG^/20240515170510/1"
    # specimen = IntubationDataset(root, x_root, y_offset=50, z_offset=-50, z_cut=400, factors=[0.7, 1.5, 0.7])

    root = "/home/zsr/project/diffpose/ours/data/liwei/孙新华/CT/SunXinHua/20240711020550.905000/3"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/孙新华/ERCP/SUNXINHUA^^/20240712155050/1"
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/孙新华/CT/SunXinHua/20240711020550.905000/3"
    # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/孙新华/ERCP/SUNXINHUA^^/20240712155050/1"
    specimen = IntubationDataset(root, x_root, x_offset=20, y_offset=200, z_offset=100, z_cut=250,factors=[0.6, 0.6, 1])


    img, pose = specimen[13]
    pose = pose.to(device)
    img = transforms(img, reverse=False)
    img = toZeroOne(img)

    plt.figure()
    plt.imshow(img.cpu().squeeze(), cmap='gray')
    plt.show()

    init = 0
    step = 0.01
    for i in range(50):
        # high = high_pass_fft(img, cutoff_ratio=0.1)
        high = high_pass_gaussian(img, init)
        init += step

    subsample = 512 / height
    delx = specimen.delx * subsample
    drr = DRR(
        specimen.volume,
        specimen.spacing,
        specimen.sdr,
        height,
        delx,
        reverse_x_axis=True,
        patch_size=height // 2,
        bone_attenuation_multiplier=3,
        bone_threshold=300
    ).to(device)
    pred_img = drr(None, None, None, pose=pose)
    pred_img = transforms(pred_img)
    pred_img = toZeroOne(pred_img)
    init = 0
    step = 0.01
    for i in range(50):
        # high = high_pass_fft(pred_img, init)
        high = high_pass_gaussian(pred_img, init)
        init += step


