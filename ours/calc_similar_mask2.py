import os
import time

import cv2
import math
import numpy as np
import pandas as pd
import torch
import glob

from diffdrr.detector import make_xrays
from matplotlib import pyplot as plt

from diffpose.calibration import RigidTransform, convert
from ours.cut.style_to_drr import StyleChanger

from utils.drr import DRR
from utils.drr_bone import DRR as DRR_Bone
from utils.metrics_mask_tube2_wei3 import MultiscaleNormalizedCrossCorrelation2d

from utils.CT_dataset import Transforms, toZeroOne
# from ours.utils.CT_dataset_PA import IntubationDataset, create_circle_mask
# from ours.case.xyl.CT_dataset import IntubationDataset, create_circle_mask
# from ours.case.wfl.CT_dataset import IntubationDataset, create_circle_mask
# from ours.case.ysy.CT_dataset import IntubationDataset, create_circle_mask
from ours.case.wch.CT_dataset import IntubationDataset, create_circle_mask
# from ours.case.sxh.CT_dataset import IntubationDataset, create_circle_mask
import kornia


def toZeroOne(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def load_model_and_data(parameterization, device="cuda:2"):
    # root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    # x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    # # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    # # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    # specimen = IntubationDataset(root, x_root, y_offset=300, z_cut=650, factors=[2, 0.4, 0.4])

    # root = "/home/zsr/project/diffpose/ours/data/liwei/许玉露/CT/XuYuLu/20240326172117.697000/3"
    # x_root = "/home/zsr/project/diffpose/ours/data/liwei/许玉露/ERCP/YULU^XU^/20240403154139/1"
    # # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/许玉露/CT/XuYuLu/20240326172117.697000/3"
    # # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/许玉露/ERCP/YULU^XU^/20240403154139/1"
    # specimen = IntubationDataset(root, x_root, z_offset=100, factors=[0.5, 4, 0.5])

    # root = "/home/zsr/project/diffpose/ours/data/liwei/王凤兰/CT/WangFengLan/20240311144245/306"
    # x_root = "/home/zsr/project/diffpose/ours/data/liwei/王凤兰/ERCP/FENGLAN^WANG^/20240313160330/1"
    # # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/王凤兰/CT/WangFengLan/20240311144245/306"
    # # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/王凤兰/ERCP/FENGLAN^WANG^/20240313160330/1"
    # specimen = IntubationDataset(root, x_root, x_offset=20, z_offset=50, z_cut=30, z_cut_end=250, factors=[0.5, 0.5, 1])

    # root = "/home/zsr/project/diffpose/ours/data/liwei/杨式瑜/CT/YangShiYu/20240315081250.536000/7"
    # x_root = "/home/zsr/project/diffpose/ours/data/liwei/杨式瑜/ERCP/SHIYU^YANG^/20240515170510/1"
    # # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/杨式瑜/CT/YangShiYu/20240315081250.536000/7"
    # # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/杨式瑜/ERCP/SHIYU^YANG^/20240515170510/1"
    # specimen = IntubationDataset(root, x_root, y_offset=50, z_offset=-50, z_cut=400, factors=[1, 1.5, 1])

    root = "/home/zsr/project/diffpose/ours/data/liwei/邬春花/CT/WuChunHua/20240708003323.343000/3"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/邬春花/ERCP/WU^CHUNHUA^/20240710150034/1"
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/邬春花/CT/WuChunHua/20240708003323.343000/3"
    # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/邬春花/ERCP/WU^CHUNHUA^/20240710150034/1"
    specimen = IntubationDataset(root, x_root, y_offset=100, z_cut=180, factors=[0.7, 0.7, 1])

    # root = "/home/zsr/project/diffpose/ours/data/liwei/孙新华/CT/SunXinHua/20240711020550.905000/3"
    # x_root = "/home/zsr/project/diffpose/ours/data/liwei/孙新华/ERCP/SUNXINHUA^^/20240712155050/1"
    # # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/孙新华/CT/SunXinHua/20240711020550.905000/3"
    # # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/孙新华/ERCP/SUNXINHUA^^/20240712155050/1"
    # specimen = IntubationDataset(root, x_root, x_offset=20, y_offset=200, z_offset=100, z_cut=250, factors=[0.6, 0.6, 1.5])

    height = 256
    subsample = 512 / height
    delx = specimen.delx * subsample

    drr = DRR(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.focal_len / 2,
        height=height,
        delx=delx,
        reverse_x_axis=True,
        bone_attenuation_multiplier=3,
    ).to(device)

    drr_bone = DRR_Bone(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.focal_len / 2,
        height=height,
        delx=delx,
        reverse_x_axis=True,
        bone_attenuation_multiplier=3,
    ).to(device)

    return specimen, drr, drr_bone, height


def process_csv_file(csv_file, specimen, style_change, drr, drr_bone, height, device="cuda:0"):
    """处理单个CSV文件"""
    # 从文件名中提取样本索引
    base_name = os.path.basename(csv_file)
    parts = base_name.split('_')
    id_number = int(parts[1][-3:])  # 提取003这样的数字
    # id_number = int(parts[2][-3:])  # 提取003这样的数字
    transforms = Transforms(drr.detector.height, radius=119)
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 获取最后一行中指定列的值
    last_row = df.iloc[-2]
    # alpha2 = last_row['alpha2']
    # beta2 = last_row['beta2']
    # gamma2 = last_row['gamma2']
    # bx2 = last_row['bx2']
    # by2 = last_row['by2']
    # bz2 = last_row['bz2']
    alpha2 = last_row['alpha']
    beta2 = last_row['beta']
    gamma2 = last_row['gamma']
    bx2 = last_row['bx']
    by2 = last_row['by']
    bz2 = last_row['bz']
    geo_r = last_row['geo_r'] / 500 * (180 / math.pi)
    geo_t = last_row['geo_t']

    # 构建优化结果
    x = [alpha2, beta2, gamma2, bx2, by2, bz2]

    # 转换为tensor并计算pose
    x = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=False)
    r = x[:3].unsqueeze(0)
    t = x[3:].unsqueeze(0)
    # pose = RigidTransform(r, t, parameterization="so3_log_map", device=device)
    pose = RigidTransform(r, t, "euler_angles", "ZYX")

    # img, true_pose = specimen[id_number]
    # 计算DRR图像
    p = drr(None, None, None, pose=pose)
    p = transforms(p).to(device)
    # plt.figure()
    # plt.imshow(p.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    # plt.show()
    true_fiducials, pred_fiducials = specimen.get_2d_fiducials(id_number, pose)
    mpd = torch.norm(true_fiducials - pred_fiducials, dim=2)
    mpd = torch.mean(mpd)
    tre = specimen.calc_tre(id_number, pose)

    img, true_pose = specimen[id_number]
    img = transforms(img, reverse=False).to(device)
    img_rev = transforms(img, reverse=True).to(device).to(torch.float32)
    tube = torch.ones_like(img_rev).to(device)
    tube[toZeroOne(img_rev) > 0.75] = 0
    circle_mask = create_circle_mask(256, 116).to(device).unsqueeze(0).unsqueeze(0)
    total_mask = (circle_mask.bool() & tube.bool()).float()
    # plt.figure()
    # plt.imshow(total_mask.cpu().squeeze(), cmap="gray")
    # plt.show()
    criterion = MultiscaleNormalizedCrossCorrelation2d([None, 21], [0.25, 0.75], device=device, step=[None, 1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
    criterion.set_mask(total_mask)

    img_change = style_change(img_rev)
    img_change = transforms(img_change, reverse=True).to(device).to(torch.float32)
    # plt.figure()
    # plt.imshow(img_change.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    # plt.show()
    # plt.figure()
    # plt.imshow(img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    # plt.show()

    # 计算相似度指标
    ncc = criterion(img, p)

    ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(p) * total_mask, toZeroOne(img_change) * total_mask, window_size=11, reduction='mean')
    ssim_ori = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(p) * total_mask, toZeroOne(img) * total_mask, window_size=11, reduction='mean')

    return {
        'file': csv_file,
        'id_number': id_number,
        'ncc': ncc.item(),
        'ssim': ssim.item(),
        'ssim_ori': ssim_ori.item(),
        'mtre': tre.item(),
        'mpd': mpd.item(),
        'geo_r': geo_r,
        'geo_t': geo_t,
        'alpha2': alpha2,
        'beta2': beta2,
        'gamma2': gamma2,
        'bx2': bx2,
        'by2': by2,
        'bz2': bz2
    }


def main(directory, style_change, prefix="zyl_xray", device="cuda:0"):
    """主函数"""
    # 查找所有匹配的CSV文件
    pattern = os.path.join(directory, f"{prefix}*se3_log_map.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"在目录 {directory} 中没有找到以 '{prefix}' 开头的CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件")

    results = []
    specimen, drr, drr_bone, height = load_model_and_data(
        parameterization="so3_log_map", device=device
    )

    count_tre = 0
    count_pd = 0
    for csv_file in csv_files:
        # print(f"处理文件: {csv_file}")

        # 从文件名提取ID
        base_name = os.path.basename(csv_file)
        parts = base_name.split('_')
        id_number = int(parts[1][-3:])
        # id_number = int(parts[2][-3:])

        # if id_number != 17:
        #     continue

        # 处理CSV文件
        result = process_csv_file(csv_file, specimen, style_change, drr, drr_bone, height, device)

        if result is not None:
            print(f"{id_number}：mtre={result['mtre']}    mpd={result['mpd']}")
            if result['mtre'] < 10:
                count_tre += 1
            # else:
            #     print(f"{id_number}：{result['mtre']}")
            if result['mpd'] < 10:
                count_pd += 1
            results.append(result)
            # print(f"ID {id_number}: NCC = {result['ncc']:.4f}, SSIM = {result['ssim']:.4f}")
        else:
            print(f"处理文件 {csv_file} 失败")


    # 输出结果统计
    if results:
        print("\n=== 结果统计 ===")

        # 定义要统计的键
        keys_to_calculate = ['ncc', 'ssim', 'ssim_ori', 'mtre', 'mpd', 'geo_r', 'geo_t']  # 添加你需要的键
        for key in keys_to_calculate:
            if key in results[0]:  # 检查键是否存在
                # 提取该键的所有值
                values = [r[key] for r in results if key in r]

                if len(values) > 0:
                    # 计算均值
                    avg = sum(values) / len(values)

                    # 计算标准差（n-1）
                    if len(values) > 1:
                        std = (sum((v - avg) ** 2 for v in values) / (len(values) - 1)) ** 0.5
                    else:
                        std = 0.0

                    print(f"平均 {key.upper()}: {avg:.4f} ± {std:.4f}")
        print(f"mtre_SR : {count_tre / len(csv_files)}")
        print(f"mpd_SR : {count_pd / len(csv_files)}")

        # 保存结果到CSV
        results_df = pd.DataFrame(results)
        output_file = os.path.join(directory, "similarity_results.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\n详细结果已保存到: {output_file}")


    results_df = pd.DataFrame(results)
    columns = [
        "id_number", "file",
        "ncc", "ssim", "ssim_ori",
        "mtre", "mpd",
        "geo_r", "geo_t",
        "alpha2", "beta2", "gamma2",
        "bx2", "by2", "bz2"
    ]
    results_df = results_df[columns]
    output_file = "./reg/ours/wchm_similarity_results.csv"
    # output_file = "./reg/diff/wfl_similarity_results.csv"
    # output_file = "/home/zsr/project/diffpose/ours/Bipose/runs/xyl_similarity_results.csv"
    # output_file = "/home/zsr/project/diffpose/ours/wsreg/runs/zyl_similarity_results.csv"
    # output_file = "/home/zsr/project/diffpose/ours/wsreg/runs/zyl_similarity_results.csv"
    results_df.to_csv(output_file, index=False)

    return results


if __name__ == "__main__":
    # 使用示例
    # directory_path = f"/home/zsr/project/diffpose/ours/runs/cma"  # 替换为你的CSV文件目录
    # directory_path = f"/home/zsr/project/diffpose/ours/runs/cma/norm"  # 替换为你的CSV文件目录
    # directory_path = f"/home/zsr/project/diffpose/ours/cnnnet/runs/mask/zyl"  # 替换为你的CSV文件目录
    # directory_path = f"/home/zsr/project/diffpose/ours/abret/gn"  # 替换为你的CSV文件目录
    # directory_path = f"/home/zsr/project/diffpose/ours/case/ysy/runs/mask/cma"  # 替换为你的CSV文件目录
    directory_path = f"/home/zsr/project/diffpose/ours/case/wch/runs/mask" # 替换为你的CSV文件目录
    # directory_path = f"/home/zsr/project/diffpose/ours/case/wfl/runs/wfl_cma" # 替换为你的CSV文件目录
    # directory_path = f"/home/zsr/project/diffpose/ours/Bipose/runs/xyl" # 替换为你的CSV文件目录
    # directory_path = f"/home/zsr/project/diffpose/ours/wsreg/runs/zyl" # 替换为你的CSV文件目录
    # directory_path = f"/home/zsr/project/diffpose/ours/case/wfl/runs/diff" # 替换为你的CSV文件目录
    # directory_path = f"/home/zsr/project/diffpose/ours/case/xyl/runs2/cma" # 替换为你的CSV文件目录
    # directory_path = f"/home/zsr/project/diffpose/ours/case/ysy/runs/mask/cma" # 替换为你的CSV文件目录
    # directory_path = f"/home/zsr/project/diffpose/ours/case/wch/runs/mask/cma" # 替换为你的CSV文件目录
    device = "cuda:0"
    # prefix = ""
    # prefix = "zyl_xray"
    # prefix = "orinodicezyl"
    # prefix = "zyl_xray"
    prefix = "wch"
    style_change = StyleChanger(
        "/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_iso/100_net_G.pth",
        device=device,
        resize=256)
    main(directory_path, style_change, prefix=prefix, device=device)