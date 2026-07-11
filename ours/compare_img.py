import numpy as np
import argparse
import sys


def compare_numpy_arrays(file1, file2, tolerance=1e-10, verbose=False):
    """
    比较两个 .npy 文件中的 NumPy 数组

    参数:
        file1: 第一个 .npy 文件路径
        file2: 第二个 .npy 文件路径
        tolerance: 浮点数比较的容差
        verbose: 是否输出详细差异信息
    """

    print(f"=== 比较 NumPy 数组 ===")
    print(f"文件1: {file1}")
    print(f"文件2: {file2}")
    print(f"容差: {tolerance}")
    print("-" * 50)

    try:
        # 加载数组
        arr1 = np.load(file1)
        arr2 = np.load(file2)

        print(f"数组1形状: {arr1.shape}, 数据类型: {arr1.dtype}")
        print(f"数组2形状: {arr2.shape}, 数据类型: {arr2.dtype}")
        print()

        # 检查形状是否相同
        if arr1.shape != arr2.shape:
            print(f"❌ 形状不匹配: {arr1.shape} vs {arr2.shape}")
            return False

        # 检查数据类型是否相同
        if arr1.dtype != arr2.dtype:
            print(f"⚠️  数据类型不同: {arr1.dtype} vs {arr2.dtype}")
            # 继续比较，但可能会影响结果

        # 逐个元素比较
        if np.issubdtype(arr1.dtype, np.floating) or np.issubdtype(arr1.dtype, np.complexfloating):
            # 浮点数或复数：使用容差比较
            differences = np.abs(arr1 - arr2)
            max_diff = np.max(differences)
            mean_diff = np.mean(differences)
            num_different = np.sum(differences > tolerance)

            print(f"最大差异: {max_diff:.6e}")
            print(f"平均差异: {mean_diff:.6e}")
            print(f"差异元素数量: {num_different} / {arr1.size}")
            print(f"差异比例: {num_different / arr1.size * 100:.4f}%")

            if num_different == 0:
                print("✅ 所有元素在容差范围内相同")
                return True
            else:
                print("❌ 存在差异元素")

                if verbose and num_different > 0:
                    print("\n=== 差异详情 ===")
                    # 找到差异最大的位置
                    max_diff_idx = np.unravel_index(np.argmax(differences), differences.shape)
                    print(f"最大差异位置: {max_diff_idx}")
                    print(f"数组1在该位置的值: {arr1[max_diff_idx]}")
                    print(f"数组2在该位置的值: {arr2[max_diff_idx]}")
                    print(f"差异值: {differences[max_diff_idx]:.6e}")

                    # 显示前几个差异元素
                    diff_indices = np.where(differences > tolerance)
                    if len(diff_indices[0]) > 0:
                        print(f"\n前5个差异元素:")
                        for i in range(min(5, len(diff_indices[0]))):
                            idx = tuple(dim[i] for dim in diff_indices)
                            print(f"位置 {idx}: {arr1[idx]:.6e} vs {arr2[idx]:.6e} (差异: {differences[idx]:.6e})")

                return False

        else:
            # 整数、布尔等：精确比较
            exact_match = np.array_equal(arr1, arr2)

            if exact_match:
                print("✅ 所有元素完全相同")
                return True
            else:
                differences = (arr1 != arr2)
                num_different = np.sum(differences)
                diff_positions = np.where(differences)

                print(f"❌ 差异元素数量: {num_different} / {arr1.size}")
                print(f"差异比例: {num_different / arr1.size * 100:.4f}%")

                if verbose and num_different > 0:
                    print("\n=== 差异详情 ===")
                    # 显示前几个差异元素
                    print(f"前5个差异元素:")
                    for i in range(min(5, len(diff_positions[0]))):
                        idx = tuple(dim[i] for dim in diff_positions)
                        print(f"位置 {idx}: {arr1[idx]} vs {arr2[idx]}")

                return False

    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        return False
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return False

if __name__ == "__main__":
    compare_numpy_arrays("gt_test.npy", "my_array.npy")
