from datetime import datetime
import math
import pandas as pd
import os
import numpy as np


def process_csv_files(directory_prefixes, columns, output_filename):
    """
    处理多个目录下指定前缀的CSV文件，提取每列的最后一行数据，计算平均值和标准差。
    每个目录和前缀组合分别计算。

    参数:
    directory_prefixes (dict): 字典，键为目录路径，值为该目录下的前缀列表
    columns (list): 要处理的列名列表
    output_filename (str): 输出结果的文件名

    返回:
    pd.DataFrame: 包含平均值和标准差的结果DataFrame
    """
    # 存储所有目录和前缀组合的结果
    all_results = {}

    # 遍历每个目录和前缀的组合
    for directory, prefixes in directory_prefixes.items():
        print(f"正在处理目录: {directory}")
        # 存储该目录下所有匹配的文件
        all_files = []
        for prefix in prefixes:
            # 获取目录下所有以指定前缀开头的CSV文件
            csv_files = [f for f in os.listdir(directory)
                         if f.startswith(prefix) and f.endswith('.csv')]

            if not csv_files:
                print(f"在目录 {directory} 中没有找到以 '{prefix}' 开头的CSV文件")
            else:
                print(f"找到 {len(csv_files)} 个匹配的文件，前缀: '{prefix}'")
                all_files.extend(csv_files)

        if not all_files:
            print(f"在目录 {directory} 中没有找到任何匹配的CSV文件")
            continue

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
        for file in all_files:
            file_path = os.path.join(directory, file)
            try:
                df = pd.read_csv(file_path)
                # 提取每列的最后一行
                last_row = df[columns].iloc[-1].to_dict()
                last_rows_data[file] = last_row

                # 处理 fiducial 列的判断逻辑
                if last_row["fiducial"] > 10:
                    print(f"文件路径: {file_path}")
                    print(f"fiducial: {last_row['fiducial']}")
                    count += 1
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")

        if not last_rows_data:
            print(f"目录 {directory} 中没有成功读取任何文件的数据")
            continue

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
                results['mean'].append(mean_val)
                results['std'].append(std_val)
                mean_std_str = f"{mean_val:.2f} ± {std_val:.2f}"
                results['mean_std_str'].append(mean_std_str)
            else:
                print(f"警告: 列 '{col}' 不存在于所有文件中")
                results['mean'].append(np.nan)
                results['std'].append(np.nan)
                results['mean_std_str'].append("N/A")

        # 创建结果DataFrame
        result_df = pd.DataFrame(results)

        # 保存该目录的结果到文件
        directory_output_filename = f"{output_filename}_{os.path.basename(directory)}.csv"
        result_df.to_csv(directory_output_filename, index=False)
        print(f"结果已保存到: {directory_output_filename}")
        print(f"{count}个fiducial大于10")
        print(f"SR : {count / len(all_files)}")

        # 将该目录的结果添加到总结果中
        all_results[directory] = result_df

    return all_results


def print_statistics_pretty(all_results):
    """
    以美观的格式打印所有目录的统计结果

    参数:
    all_results (dict): 包含多个目录结果的字典，键为目录路径，值为统计结果DataFrame
    """
    if not all_results:
        print("没有可用的统计结果")
        return

    for directory, result_df in all_results.items():
        print(f"\n统计结果 (均值 ± 标准差) - 目录: {directory}")
        print("=" * 50)
        for _, row in result_df.iterrows():
            column = row['column']
            mean_std_str = row['mean_std_str']
            print(f"{column:15}: {mean_std_str}")
        print("=" * 50)


# 使用示例
if __name__ == "__main__":
    # directory_prefixes 字典，键为目录路径，值为该目录下的前缀列表
    directory_prefixes = {
        "/media/sda1/PersonalFiles/yx/project/diffpose/ours/case/wfl/runs/cma": ["wfl", "specimen"],
        "/home/zsr/project/diffpose/ours/deepfluoro/runs/1": ["nodicewfl", "xyl"],
    }

    columns = ["fiducial", "geo_r", "geo_t", "geo_d", "geo_se3"]  # 要处理的列
    output_file = "summary_statistics"  # 输出文件名，不同目录结果会保存为不同的文件

    all_results = process_csv_files(directory_prefixes, columns, output_file)
    print_statistics_pretty(all_results)
