import os
import glob
import pandas as pd
import numpy as np


def summarize_metrics(
    directory,
    pattern="*.csv",
    # metrics=("ncc", "ssim", "ssim_ori", "mtre", "mpd", "geo_r", "geo_t"),
    # metrics=("ncc", "ssim", "ssim_ori", "tre", "mpd", "geo_r", "geo_t"),
    metrics=("ncc", "ssim", "ssim_ori", "tre", "mtre", "mpd", "geo_r", "geo_t"),
    mtre_threshold=10.0,
):
    """
    统计目录下所有 CSV 文件中的指标
    """

    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        print(f"❌ 目录中未找到 CSV 文件: {directory}")
        return

    print(f"✅ 找到 {len(files)} 个 CSV 文件")

    all_rows = []

    for file in files:
        try:
            df = pd.read_csv(file)
            all_rows.append(df)
        except Exception as e:
            print(f"⚠️ 读取失败: {file}, {e}")

    if not all_rows:
        print("❌ 没有成功读取任何文件")
        return

    # 合并所有样本
    data = pd.concat(all_rows, ignore_index=True)

    print("\n===== Overall Statistics =====")

    for metric in metrics:
        if metric not in data.columns:
            print(f"⚠️ 跳过 {metric}（列不存在）")
            continue

        values = data[metric].dropna().values
        if len(values) == 0:
            continue

        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        # std = np.std(values, ddof=1)
        # std = np.sqrt(np.sum((values - mean) ** 2) / (len(values) - 1))

        print(f"{metric.upper():8s}: {mean:.2f} ± {std:.2f}")

    # ===== SR@10mm =====
    if "mtre" in data.columns:
        mtre_values = data["mtre"].dropna().values
        sr = np.mean(mtre_values < mtre_threshold)
        print(f"\nSR@{mtre_threshold:.0f}mm: {sr:.2%}")
    if "tre" in data.columns:
        mtre_values = data["tre"].dropna().values
        sr = np.mean(mtre_values < mtre_threshold)
        print(f"\nSR@{mtre_threshold:.0f}mm: {sr:.2%}")

    print(f"\n总样本数: {len(data)}")


if __name__ == "__main__":
    # directory = "./reg"  # 改成你的目录
    directory = "./reg/ours"  # 改成你的目录
    # directory = "./reg/diff"  # 改成你的目录
    # directory = "/home/zsr/project/diffpose/ours/wsreg/runs"  # 改成你的目录
    # directory = "/home/zsr/project/diffpose/ours/deepfluoro/runs/all"  # 改成你的目录
    # directory = "/home/zsr/project/diffpose/ours/deepfluoro/runs/all_bi"  # 改成你的目录
    # directory = "/home/zsr/project/diffpose/ours/deepfluoro/runs/new/all_wsr"  # 改成你的目录
    # directory = "/home/zsr/project/diffpose/ours/abmodel/base"  # 改成你的目录
    # directory = "/home/zsr/project/diffpose/ours/abmodel/noattn"  # 改成你的目录
    # directory = "/home/zsr/project/diffpose/ours/abmodel/coespdeco"  # 改成你的目录
    # directory = "/home/zsr/project/diffpose/ours/abmodel/coesp"  # 改成你的目录
    summarize_metrics(directory, mtre_threshold=10)
