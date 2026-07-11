import os

def add_prefix_to_files(folder_path, prefix):
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        # 只处理文件，跳过子目录
        if os.path.isfile(old_path):
            new_filename = prefix + filename
            new_path = os.path.join(folder_path, new_filename)

            # 避免重名覆盖
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
            else:
                print(f"跳过（已存在）: {new_filename}")

def add_suffix_to_filenames(folder_path, suffix, dry_run=True):
    """
    给指定目录下所有文件名添加后缀（不影响扩展名）

    folder_path: 目录路径
    suffix: 要添加的后缀字符串（如 "_aug"）
    dry_run: True 只打印，不真正重命名
    """

    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        # 只处理文件
        if not os.path.isfile(old_path):
            continue

        # 正确拆分
        name, ext = split_filename(filename)

        # 避免重复添加
        if name.endswith(suffix):
            continue

        new_filename = f"{name}{suffix}{ext}"
        new_path = os.path.join(folder_path, new_filename)

        if dry_run:
            print(f"{filename}  ->  {new_filename}")
        else:
            os.rename(old_path, new_path)


def get_prefix(filename):
    """
    获取文件名前缀：
    例：
    case001.png      -> case001
    case001.nii.gz   -> case001
    """
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    else:
        return os.path.splitext(filename)[0]


def delete_files_without_matching_prefix(dir_a, dir_b, dry_run=True):
    """
    dir_a: 待删除文件所在目录
    dir_b: 参考目录
    dry_run=True 表示只打印，不真正删除（强烈推荐先用）
    """

    # 1. 收集目录 B 中所有前缀
    prefix_set_b = set()
    for f in os.listdir(dir_b):
        if os.path.isfile(os.path.join(dir_b, f)):
            prefix_set_b.add(get_prefix(f))

    # 2. 遍历目录 A
    for f in os.listdir(dir_a):
        path_a = os.path.join(dir_a, f)
        if not os.path.isfile(path_a):
            continue

        prefix_a = get_prefix(f)

        if prefix_a not in prefix_set_b:
            if dry_run:
                print(f"[将删除] {f}")
            else:
                os.remove(path_a)
                print(f"[已删除] {f}")

import os

def rename_files(
    folder_path,
    file_start_prefix,
    insert_str,
    dry_run=True
):
    """
    folder_path: 目录路径
    file_start_prefix: 文件名必须以该前缀开头才处理
    insert_str: 要插入到最后一个 '_' 分割字段前的字符串
    dry_run: True 只打印，不真正重命名
    """

    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        if not os.path.isfile(old_path):
            continue

        # 1. 只处理指定前缀开头的文件
        if not filename.startswith(file_start_prefix):
            continue

        # 2. 用 '.' 分割，保留文件格式
        if "." not in filename:
            continue  # 无扩展名，跳过

        name_part, ext = filename.rsplit(".", 1)

        # 3. 用 '_' 分割文件名前缀
        parts = name_part.split("_")
        if len(parts) < 1:
            continue

        # 4. 给最后一个部分前面加指定字符串
        parts[-2] = insert_str + parts[-2]

        # 5. 拼接回新文件名
        new_name_part = "_".join(parts)
        new_filename = f"{new_name_part}.{ext}"
        new_path = os.path.join(folder_path, new_filename)

        if dry_run:
            print(f"{filename}  ->  {new_filename}")
        else:
            os.rename(old_path, new_path)

def split_filename(filename):
    """正确处理 .nii.gz"""
    if filename.endswith(".nii.gz"):
        return filename[:-7], ".nii.gz"
    else:
        return os.path.splitext(filename)

def add_index_suffix(folder_path, start_idx=1, dry_run=True):
    """
    给目录下文件按排序添加 _0001, _0002 ... 后缀
    """
    files = [f for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f))]

    files.sort()  # 按文件名排序

    for i, filename in enumerate(files, start=start_idx):
        name, ext = split_filename(filename)
        suffix = f"_{i:04d}"
        new_filename = f"{name}{suffix}{ext}"

        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        if dry_run:
            print(f"{filename}  ->  {new_filename}")
        else:
            os.rename(old_path, new_path)

#
# if __name__ == "__main__":
#     folder = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/bone_seg/nnUNet_raw/labelsTr"
#     file_start_prefix = "陈聪滨_"   # 只处理这个前缀开头的文件
#     insert_str = "1"           # 插入的字符串
#
#     # 强烈建议先 dry_run=True
#     rename_files(folder, file_start_prefix, insert_str, dry_run=False)


if __name__ == "__main__":
    # dir_a = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/bone_seg/nnUNet_raw/imagesTr"   # 待检查目录
    # dir_b = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/bone_seg/nnUNet_raw/labelsTr"   # 参考目录
    #
    # # 先 dry_run=True 看看效果
    # delete_files_without_matching_prefix(dir_a, dir_b, dry_run=True)

    # folder = "/home/zsr/project/diffpose/ours/bone_seg/nnUNet_raw_tmp/imagesTr"
    # folder = "/home/zsr/project/diffpose/ours/bone_seg/xyl_nii"
    # folder = "/home/zsr/project/diffpose/ours/bone_seg/nnUNet_raw_tmp/labelsTr"
    folder = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/bone_seg/xyb_nii"
    # folder = "/home/zsr/project/diffpose/ours/bone_seg/deep_bone/labelsTr"
    suffix = "_0000"  # 例如 "_aug"

    # 强烈建议先 dry_run=True
    add_suffix_to_filenames(folder, suffix, dry_run=False)

    # 强烈建议先 dry_run=True
    # add_index_suffix(folder, start_idx=1, dry_run=False)

# if __name__ == "__main__":
#     folder = "/home/zsr/project/diffpose/ours/bone_seg/images/zt"   # 替换为你的目录
#     prefix = "朱婷_"                # 要添加的前缀
#     add_prefix_to_files(folder, prefix)
