import os
import time
import math
import numpy as np
import pandas as pd
import torch
import glob
from diffpose.calibration import RigidTransform
from ours.utils.CT_dataset_PA import IntubationDataset


def load_model_and_data(device="cuda:2"):
    """只加载必要的数据"""
    root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    specimen = IntubationDataset(root, x_root, y_offset=300, z_cut=650, factors=[2, 0.4, 0.4])

    return specimen, device


def calculate_final_tre(csv_file, specimen, device="cuda:0"):
    """计算CSV文件最后一行的TRE和MPD"""
    try:
        # 从文件名中提取样本索引
        base_name = os.path.basename(csv_file)
        parts = base_name.split('_')
        id_number = int(parts[1][-3:])  # 提取003这样的数字

        # 读取CSV文件
        df = pd.read_csv(csv_file)

        if len(df) == 0:
            print(f"文件 {base_name}: 空文件，跳过")
            return None

        # 获取最后一行
        last_row = df.iloc[-1]

        # 从最后一行中获取姿态参数
        # 根据您的CSV文件列名选择正确的列
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

        # 计算TRE
        tre = specimen.calc_tre(id_number, pose)

        # 计算MPD（如果2D特征点可用）
        try:
            true_fiducials, pred_fiducials = specimen.get_2d_fiducials(id_number, pose)
            mpd = torch.norm(true_fiducials - pred_fiducials, dim=2)
            mpd = torch.mean(mpd)
            mpd_value = mpd.item()
        except:
            mpd_value = float('nan')

        tre_value = tre.item()

        # 更新最后一行
        df.at[df.index[-1], 'tre'] = tre_value
        # df.at[df.index[-1], 'mpd'] = mpd_value

        # 保存回原始文件
        df.to_csv(csv_file, index=False)

        print(f"文件 {base_name} (ID {id_number}): TRE={tre_value:.2f}mm, MPD={mpd_value:.2f}px")

        return {
            'file': csv_file,
            'id_number': id_number,
            'tre': tre_value,
            'mpd': mpd_value,
            'alpha': alpha2,
            'beta': beta2,
            'gamma': gamma2,
            'bx': bx2,
            'by': by2,
            'bz': bz2
        }

    except Exception as e:
        print(f"处理文件 {csv_file} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(directory, prefix="zyl_xray", device="cuda:0"):
    """主函数：重新计算CSV文件最后一行的TRE并保存回源文件"""
    # 查找所有匹配的CSV文件
    pattern = os.path.join(directory, f"{prefix}*se3_log_map.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"在目录 {directory} 中没有找到以 '{prefix}' 开头的CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 加载模型和数据
    specimen, device = load_model_and_data(device=device)

    results = []
    count_tre_success = 0
    count_mpd_success = 0
    tre_values = []
    mpd_values = []

    for csv_file in csv_files:
        print(f"\n处理文件: {os.path.basename(csv_file)}")

        # 处理CSV文件
        result = calculate_final_tre(csv_file, specimen, device)

        if result is not None:
            results.append(result)

            # 统计TRE < 10mm的成功率
            if not np.isnan(result['tre']) and result['tre'] < 10:
                count_tre_success += 1
            if not np.isnan(result['mpd']) and result['mpd'] < 10:
                count_mpd_success += 1

            # 收集值用于统计
            if not np.isnan(result['tre']):
                tre_values.append(result['tre'])
            if not np.isnan(result['mpd']):
                mpd_values.append(result['mpd'])

    # 输出统计结果
    if results:
        print("\n" + "=" * 60)
        print("处理完成！统计结果：")
        print("=" * 60)

        # 计算平均值和标准差
        if tre_values:
            avg_tre = np.mean(tre_values)
            std_tre = np.std(tre_values)
            tre_success_rate = count_tre_success / len(results) * 100

            print(f"TRE统计:")
            print(f"  平均值: {avg_tre:.2f} mm")
            print(f"  标准差: {std_tre:.2f} mm")
            print(f"  最小值: {np.min(tre_values):.2f} mm")
            print(f"  最大值: {np.max(tre_values):.2f} mm")
            print(f"  成功率 (<10mm): {tre_success_rate:.1f}% ({count_tre_success}/{len(results)})")

        if mpd_values:
            avg_mpd = np.mean(mpd_values)
            std_mpd = np.std(mpd_values)
            mpd_success_rate = count_mpd_success / len(results) * 100

            print(f"\nMPD统计:")
            print(f"  平均值: {avg_mpd:.2f} px")
            print(f"  标准差: {std_mpd:.2f} px")
            print(f"  最小值: {np.min(mpd_values):.2f} px")
            print(f"  最大值: {np.max(mpd_values):.2f} px")
            print(f"  成功率 (<10px): {mpd_success_rate:.1f}% ({count_mpd_success}/{len(results)})")

        # 保存汇总结果到新文件
        summary_df = pd.DataFrame(results)

        # 按TRE排序
        summary_df_sorted = summary_df.sort_values('tre')

        summary_file = os.path.join(directory, "tre_results_summary.csv")
        summary_df_sorted.to_csv(summary_file, index=False)

        # 创建详细结果文件
        detailed_results = []
        for result in results:
            detailed_results.append({
                'ID': result['id_number'],
                'TRE(mm)': f"{result['tre']:.2f}",
                'MPD(px)': f"{result['mpd']:.2f}",
                'TRE_Status': '成功' if result['tre'] < 10 else '失败',
                'MPD_Status': '成功' if result['mpd'] < 10 else '失败',
                'alpha': f"{result['alpha']:.4f}",
                'beta': f"{result['beta']:.4f}",
                'gamma': f"{result['gamma']:.4f}",
                'bx': f"{result['bx']:.2f}",
                'by': f"{result['by']:.2f}",
                'bz': f"{result['bz']:.2f}",
                'file': os.path.basename(result['file'])
            })

        detailed_df = pd.DataFrame(detailed_results)
        detailed_file = os.path.join(directory, "tre_detailed_results.csv")
        detailed_df.to_csv(detailed_file, index=False)

        print(f"\n详细结果已保存到:")
        print(f"  {summary_file}")
        print(f"  {detailed_file}")

        # 显示成功率排名
        print("\n按TRE排序的结果:")
        print("-" * 80)
        print(f"{'ID':<5} {'TRE(mm)':<10} {'MPD(px)':<10} {'状态':<8}")
        print("-" * 80)

        for _, row in summary_df_sorted.iterrows():
            status = "✓" if row['tre'] < 10 else "✗"
            print(f"{row['id_number']:<5} {row['tre']:<10.2f} {row['mpd']:<10.2f} {status:<8}")


if __name__ == "__main__":
    # 使用示例
    directory_path = f"/home/zsr/project/diffpose/ours/abret/wlndice"
    device = "cuda:1"
    prefix = "zyl_xray"

    main(directory_path, prefix=prefix, device=device)