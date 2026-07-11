import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import kornia
from matplotlib import pyplot as plt

from ours.utils.drr import DRR
from utils.drr_bone import DRR as DRR_Bone
from diffpose.deepfluoro import DeepFluoroDataset, Evaluator, Transforms
from utils.CT_dataset_PA import IntubationDataset
from utils.registration import PoseRegressor
from diffpose.calibration import RigidTransform, convert
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d

def toZeroOne(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def load_model_and_data(id_number, parameterization, device="cuda:2"):
    specimen = DeepFluoroDataset(
        id_number,
        # filename="/media/sda1/PersonalFiles/yx/project/diffpose/diffpose/data/ipcai_2020_full_res_data.h5"
        filename="/home/zsr/project/diffpose/data/ipcai_2020_full_res_data.h5"
    )

    height = 256
    subsample = (1536 - 100) / height
    delx = 0.194 * subsample

    drr = DRR(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.focal_len / 2,
        height=height,
        delx=delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        bone_attenuation_multiplier=2.5,
    ).to(device)

    drr_bone = DRR_Bone(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.focal_len / 2,
        height=height,
        delx=delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        bone_attenuation_multiplier=2.5,
    ).to(device)

    return specimen, drr, drr_bone, height


def process_csv_file(csv_file, specimen, drr, drr_bone, height, device="cuda:0"):
    """处理单个CSV文件"""
    # 从文件名中提取样本索引
    base_name = os.path.basename(csv_file)
    parts = base_name.split('_')
    id_number = int(parts[1][-3:])  # 提取003这样的数字
    # id_number = int(parts[2][-3:])  # 提取003这样的数字
    transforms = Transforms(drr.detector.height)
    criterion = MultiscaleNormalizedCrossCorrelation2d([None, 13], [0.5, 0.5])

    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 获取最后一行中指定列的值
        last_row = df.iloc[-1]
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

        # 构建优化结果
        x = [alpha2, beta2, gamma2, bx2, by2, bz2]

        # 转换为tensor并计算pose
        x = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=False)
        r = x[:3].unsqueeze(0)
        t = x[3:].unsqueeze(0)
        # pose = RigidTransform(r, t, parameterization="so3_log_map", device=device)
        pose = RigidTransform(r, t, "euler_angles", "ZYX")

        # 计算DRR图像
        p = drr(None, None, None, pose=pose)
        p = transforms(p).to(device)
        # plt.figure()
        # plt.imshow(p.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()

        # 获取真实X光图像
        img, true_pose = specimen[id_number]
        img = transforms(img).to(device)  # 添加batch维度
        # plt.figure()
        # plt.imshow(img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()

        # 计算相似度指标
        ncc = criterion(img, p)

        ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(p), toZeroOne(img), window_size=11, reduction='mean')

        return {
            'file': csv_file,
            'id_number': id_number,
            'ncc': ncc.item(),
            'ssim': ssim.item(),
            'alpha2': alpha2,
            'beta2': beta2,
            'gamma2': gamma2,
            'bx2': bx2,
            'by2': by2,
            'bz2': bz2
        }

    except Exception as e:
        print(f"处理文件 {csv_file} 时出错: {e}")
        return None


def main(idx, directory, prefix="zyl_xray", device="cuda:0"):
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
        idx, "so3_log_map", device
    )

    for csv_file in csv_files:
        # print(f"处理文件: {csv_file}")

        # 从文件名提取ID
        base_name = os.path.basename(csv_file)
        parts = base_name.split('_')
        id_number = int(parts[1][-3:])
        # id_number = int(parts[2][-3:])

        try:
            # 处理CSV文件
            result = process_csv_file(csv_file, specimen, drr, drr_bone, height, device)

            if result is not None:
                results.append(result)
                # print(f"ID {id_number}: NCC = {result['ncc']:.4f}, SSIM = {result['ssim']:.4f}")
            else:
                print(f"处理文件 {csv_file} 失败")

        except Exception as e:
            print(f"处理文件 {csv_file} 时发生错误: {e}")
            continue

    # 输出结果统计
    if results:
        print("\n=== 结果统计 ===")
        avg_ncc = sum(r['ncc'] for r in results) / len(results)
        avg_ssim = sum(r['ssim'] for r in results) / len(results)

        # 计算样本标准差 (n-1)
        n = len(results)
        std_ncc = (sum((r['ncc'] - avg_ncc) ** 2 for r in results) / (n - 1)) ** 0.5
        std_ssim = (sum((r['ssim'] - avg_ssim) ** 2 for r in results) / (n - 1)) ** 0.5

        print(f"平均 NCC: {avg_ncc:.4f} ± {std_ncc:.4f}")
        print(f"平均 SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")

        # 保存结果到CSV
        results_df = pd.DataFrame(results)
        output_file = os.path.join(directory, "similarity_results.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\n详细结果已保存到: {output_file}")

    return results


if __name__ == "__main__":
    # 使用示例
    idx = 2
    # directory_path = f"/home/zsr/project/diffpose/ours/deepfluoro/runs/{idx}/cma" # 替换为你的CSV文件目录
    directory_path = f"/home/zsr/project/diffpose/ours/deepfluoro/runs/{idx}/norm" # 替换为你的CSV文件目录
    # directory_path = f"/home/zsr/project/diffpose/ours/deepfluoro/runs/norm" # 替换为你的CSV文件目录
    device = "cuda:0"
    # prefix = "nodice_specimen"
    prefix = "specimen"
    # prefix = "1specimen"
    main(idx, directory_path, prefix=prefix, device=device)