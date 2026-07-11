from datetime import datetime

import math
import pandas as pd
import os
import numpy as np

def process_csv_files(directory, prefix, columns, output_filename):
    """
    处理目录下指定前缀的CSV文件，提取每列的最后一行数据，计算平均值和标准差

    参数:
    directory (str): 要搜索的目录路径
    prefix (str): 文件前缀
    columns (list): 要处理的列名列表
    output_filename (str): 输出结果的文件名

    返回:
    pd.DataFrame: 包含平均值和标准差的结果DataFrame
    """
    # 获取目录下所有以指定前缀开头的CSV文件
    csv_files = [f for f in os.listdir(directory)
                 if f.startswith(prefix) and f.endswith('.csv')]

    if not csv_files:
        print(f"在目录 {directory} 中没有找到以 '{prefix}' 开头的CSV文件")
        return None

    print(f"找到 {len(csv_files)} 个匹配的文件")

    # 存储每个文件的最后一行数据
    last_rows_data = {}

    # 初始化结果字典
    results = {
        'column': columns,
        'mean': [],
        'std': [],
        'mean_std_str': []
    }

    count = 0
    # 读取每个文件并提取指定列的最后一行
    for file in csv_files:
        file_path = os.path.join(directory, file)
        try:
            # mod_time = os.path.getmtime(file_path)
            # mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            # print(f"文件: {file}, 修改时间: {mod_time_str}")

            df = pd.read_csv(file_path)
            # 提取每列的最后一行
            last_row = df[columns].iloc[-1].to_dict()
            last_rows_data[file] = last_row
            # print(file_path)
            # print(last_row["fiducial"])
            # if last_row["fiducial"] > 10:
            if last_row["tre"] < 10:
                print(file_path)
                print(last_row["tre"])
                count += 1
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")

    if not last_rows_data:
        print("没有成功读取任何文件的数据")
        return None

    # 将数据转换为DataFrame以便计算
    data_df = pd.DataFrame.from_dict(last_rows_data, orient='index')

    # 计算每列的平均值和标准差
    for col in columns:
        if col in data_df.columns:
            mean_val = data_df[col].mean()
            std_val = data_df[col].std()
            if col == "geo_r":
                mean_val = mean_val / 500 * (180 / math.pi)
                std_val = std_val / 500 * (180 / math.pi)
            results['mean'].append(data_df[col].mean())
            results['std'].append(data_df[col].std())
            mean_std_str = f"{mean_val:.2f} ± {std_val:.2f}"
            results['mean_std_str'].append(mean_std_str)
        else:
            print(f"警告: 列 '{col}' 不存在于所有文件中")
            results['mean'].append(np.nan)
            results['std'].append(np.nan)
            results['mean_std_str'].append("N/A")

    # 创建结果DataFrame
    result_df = pd.DataFrame(results)

    # 保存结果到CSV文件
    result_df.to_csv(output_filename, index=False)
    print(f"结果已保存到: {output_filename}")
    print(f"{count}个mtre小于1mm")
    print(f"SR : {count / len(csv_files)}")

    return result_df


def print_statistics_pretty(result_df):
    """
    以美观的格式打印统计结果

    参数:
    result_df (pd.DataFrame): 包含统计结果的DataFrame
    """
    if result_df is None:
        print("没有可用的统计结果")
        return

    print("\n统计结果 (均值 ± 标准差):")
    print("=" * 50)
    for _, row in result_df.iterrows():
        column = row['column']
        mean_std_str = row['mean_std_str']
        print(f"{column:15}: {mean_std_str}")
    print("=" * 50)

# 使用示例
if __name__ == "__main__":
    # 示例用法
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/runs/cma/norm"  # 替换为你的目录路径
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/deepfluoro/runs/2/norm"  # 替换为你的目录路径
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/deepfluoro/runs/1"  # 替换为你的目录路径
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/deepfluoro/runs/mask/2"  # 替换为你的目录路径
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/deepfluoro/runs/wsr/6"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/deepfluoro/runs/norm"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/deepfluoro/runs"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/experiments/deepfluoro/runs"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/deepfluoro/runs/1/cma"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/deepfluoro/runs/5/norm"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/deepfluoro/runs/3"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/deepfluoro/runs/mask/6"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/deepfluoro/runs/wsr/"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/deepfluoro/runs/bipose/6"  # 替换为你的目录路径
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/deepfluoro/runs/mask/6"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/runs/grad/diff"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/runs/grad"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/runs/cma"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/runs/cma/norm"  # 替换为你的目录路径
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/runs/norm"  # 替换为你的目录路径
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/runs/bone"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/runs/bone"  # 替换为你的目录路径

    # directory = "/home/zsr/project/diffpose/ours/case/wfl/runs/cma"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/case/wfl/runs/norm"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/case/wfl/runs/diff"  # 替换为你的目录路径
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/case/wfl/runs/cma"  # 替换为你的目录路径
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/case/xyl/runs/bone"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/case/xyl/runs2/cma"  # 替换为你的目录路径
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/case/xyl/runs/cma"  # 替换为你的目录路径
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/case/xyl/runs/bone"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/case/ysy/runs/mask/cma"  # 替换为你的目录路径


    # directory = "/home/zsr/project/diffpose/ours/cnnnet/runs/zyl"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/cnnnet/runs/mask/zyl"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/case/xyl/runs/mask"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/case/wfl/runs/wfl_cma"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/case/wch/runs/mask"  # 替换为你的目录路径
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/case/wfl/runs/mask"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/wsreg/runs/zyl"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/wsreg/runs/all"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/Bipose/runs/all"  # 替换为你的目录路径
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/case/wfl/runs/mask"  # 替换为你的目录路径
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/cnnnet/runs/mask/zyl"  # 替换为你的目录路径
    # directory = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/deepfluoro/runs/mask/2"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/cnnnet/runs/mask/zyl"  # 替换为你的目录路径
    # directory = "/home/zsr/project/diffpose/ours/case/wch/runs/mask"  # 替换为你的目录路径
    directory = "/home/zsr/project/diffpose/ours/abret/ln"  # 替换为你的目录路径

    # prefix = "zyl_xray"   # 文件前缀
    # prefix = "orizyl_xray"   # 文件前缀
    # prefix = "orinodicezyl_xray"   # 文件前缀
    # prefix = "nodicezyl_xray"   # 文件前缀
    # prefix = "cma_zyl_xray"   # 文件前缀
    # prefix = "specimen"   # 文件前缀
    # prefix = "truebonespecimen"   # 文件前缀
    # prefix = "1specimen"   # 文件前缀
    # prefix = "nodice_specimen"   # 文件前缀
    # prefix = "xyl"
    # prefix = "nodicexyl"
    # prefix = "wfl"
    # prefix = "nodicewfl"

    # prefix = "localzyl_xray"
    # prefix = "weizyl_xray"
    # prefix = "zyl_xray"
    # prefix = "noweizyl_xray"
    # prefix = "graddicewfl_xray"
    # prefix = "graddicezyl_xray"
    # prefix = "gncczyl_xray"
    # prefix = "gnccdicezyl_xray"
    # prefix = "localncczyl_xray"
    # prefix = "localnccdicezyl_xray"
    # prefix = "weilncczyl_xray"
    # prefix = "ssimdicezyl_xray"
    # prefix = "ssimzyl_xray"
    # prefix = "nodicezyl_xray"

    prefix = ""
    # prefix = "zyl_xray"
    # prefix = "xyl_xray"
    # prefix = "wfl_xray"

    # columns = ["losses", "ssim", "fiducial", "geo_r", "geo_t", "geo_d", "geo_se3"]  # 要处理的列
    # columns = ["ssim", "fiducial", "geo_r", "geo_t", "geo_d", "geo_se3"]  # 要处理的列
    # columns = ["ncc", "fiducial", "geo_r", "geo_t", "geo_d", "geo_se3"]  # 要处理的列
    # columns = ["fiducial", "geo_r", "geo_t", "geo_d", "geo_se3"]  # 要处理的列
    columns = ["fiducial", "geo_r", "geo_t", "geo_d", "geo_se3", "tre"]  # 要处理的列
    # columns = ["ssim", "ssim_ori", "fiducial", "geo_r", "geo_t", "geo_d", "geo_se3"]  # 要处理的列
    output_file = "summary_statistics.csv"  # 输出文件名

    result = process_csv_files(directory, prefix, columns, output_file)
    print_statistics_pretty(result)
    # if result is not None:
    #     print("\n统计结果:")
    #     print(result)