import os
from pathlib import Path

import nrrd
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import nibabel as nib
import pydicom
from scipy.ndimage import zoom

def read_dicom_matrix(dcm_file, visualize=False):
    """
    读取单张 DICOM 文件矩阵，并可视化
    dcm_file: 单个 .dcm 文件路径
    visualize: 是否显示图像
    """
    if not os.path.isfile(dcm_file):
        raise FileNotFoundError(f"{dcm_file} 不存在")

    # 读取单文件 DICOM
    image = sitk.ReadImage(dcm_file)
    array = sitk.GetArrayFromImage(image)  # numpy array (1, H, W)
    array = array[0]  # 单张图像，去掉第0维

    if visualize:
        plt.imshow(array, cmap='gray')
        plt.title(f"DICOM: {os.path.basename(dcm_file)}")
        plt.axis('off')
        plt.show()

    return array

def read_nrrd_matrix(nrrd_path, visualize=False, slice_idx=None):
    """读取 NRRD 文件矩阵，并可视化指定切片"""
    import nrrd
    data, _ = nrrd.read(nrrd_path)
    data = np.squeeze(data)
    data = np.swapaxes(data, 0, 1)

    if visualize:
        plt.imshow(data, cmap='gray')
        plt.axis('off')
        plt.show()

    return data

def save_nii(matrix, output_path):
    """保存 numpy 矩阵为 NIfTI，使用单位仿射"""
    import nibabel as nib
    affine = np.eye(4)  # identity affine
    nii = nib.Nifti1Image(matrix, affine)
    nib.save(nii, output_path)
    print(f"Saved {output_path}")

def convert_folder(input_path, output_folder, file_type):
    """
    统一转换
    file_type: "dcm" 或 "nrrd"
    """
    os.makedirs(output_folder, exist_ok=True)

    if file_type == "dcm":
        for foldername in os.listdir(input_path):
            folder_path = os.path.join(input_path, foldername)
            # matrix = read_dicom_matrix(folder_path, visualize=True)
            matrix = read_dicom_matrix(folder_path)
            out_path = os.path.join(output_folder, foldername.split('.')[0] + ".nii.gz")
            save_nii(matrix, out_path)

    elif file_type == "nrrd":
        for filename in os.listdir(input_path):
            if filename.endswith(".nrrd"):
                file_path = os.path.join(input_path, filename)
                # matrix = read_nrrd_matrix(file_path, visualize=True)
                matrix = read_nrrd_matrix(file_path)
                out_path = os.path.join(output_folder, filename.replace(".nrrd", ".nii.gz"))
                save_nii(matrix, out_path)

def visualize_nii_folder(folder_path, slice_idx=None):
    """
    遍历目录下所有 .nii.gz 文件并可视化
    folder_path: 目录路径
    slice_idx: 指定显示哪一层（None 表示中间层）
    """

    nii_files = [f for f in os.listdir(folder_path) if f.endswith(".nii.gz")]
    nii_files.sort()

    if not nii_files:
        print("目录下没有 nii.gz 文件")
        return

    for fname in nii_files:
        nii_path = os.path.join(folder_path, fname)

        nii = nib.load(nii_path)
        data = nii.get_fdata()
        data = np.squeeze(data)  # 去掉多余维度

        print(f"\n文件名: {fname}")
        print(f"shape: {data.shape}, dtype: {data.dtype}")

        # 可视化
        if data.ndim == 3:
            z = slice_idx if slice_idx is not None else data.shape[0] // 2
            img = data[z]
            title = f"{fname} | slice {z}"
        elif data.ndim == 2:
            img = data
            title = f"{fname} | 2D"
        else:
            print("不支持的维度，跳过")
            continue

        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()


def process_and_save_nii_folder(folder_path, save_folder=None, slice_idx=None):
    """
    遍历目录下所有 .nii.gz 文件，读取后转换宽高维度（转置 height 和 width），
    保存新的 .nii.gz 文件到指定目录，并可视化转置后的中间层（可选）。

    参数:
    folder_path: 输入目录路径（包含 .nii.gz 文件）
    save_folder: 保存目录（None 表示与输入目录相同）
    slice_idx: 指定显示哪一层（None 表示中间层）
    """
    if save_folder is None:
        save_folder = folder_path

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    nii_files = [f for f in os.listdir(folder_path) if f.endswith(".nii.gz")]
    nii_files.sort()

    if not nii_files:
        print("目录下没有 nii.gz 文件")
        return

    for fname in nii_files:
        nii_path = os.path.join(folder_path, fname)
        nii_img = nib.load(nii_path)
        data = nii_img.get_fdata()
        original_shape = data.shape

        print(f"\n文件名: {fname}")
        print(f"原始 shape: {original_shape}, dtype: {data.dtype}")

        # 调整 affine 矩阵以匹配新方向
        affine = np.eye(4)

        data = np.swapaxes(data, 0, 1)

        # 创建新 NIfTI 图像
        transposed_img = nib.Nifti1Image(data, affine)

        # 保存
        save_path = os.path.join(save_folder, fname)
        nib.save(transposed_img, save_path)
        print(f"已保存转置后的文件: {save_path} (新 shape: {data.shape})")

        # 可视化转置后的图像（squeeze 后）
        data_squeezed = np.squeeze(data)
        if data_squeezed.ndim == 3:
            z = slice_idx if slice_idx is not None else data_squeezed.shape[0] // 2
            img = data_squeezed[z]
            title = f"transposed_{fname} | slice {z}"
        elif data_squeezed.ndim == 2:
            img = data_squeezed
            title = f"transposed_{fname} | 2D"
        else:
            continue

        # plt.figure(figsize=(5, 5))
        # plt.imshow(img, cmap="gray")
        # plt.title(title)
        # plt.axis("off")
        # plt.show()

def process_and_save_nii_file(folder_path):
    nii_path = folder_path
    nii_img = nib.load(nii_path)
    data = nii_img.get_fdata()
    original_shape = data.shape

    print(f"\n文件名: {folder_path}")
    print(f"原始 shape: {original_shape}, dtype: {data.dtype}")

    # 调整 affine 矩阵以匹配新方向
    affine = np.eye(4)

    data = np.swapaxes(data, 0, 1)

    # 创建新 NIfTI 图像
    transposed_img = nib.Nifti1Image(data, affine)

    # 保存
    save_path = folder_path
    nib.save(transposed_img, folder_path)
    print(f"已保存转置后的文件: {save_path} (新 shape: {data.shape})")

    # plt.figure(figsize=(5, 5))
    # plt.imshow(img, cmap="gray")
    # plt.title(title)
    # plt.axis("off")
    # plt.show()

def dcm2nii_volume_sag(root, factors=[1,1,1]):
    file_name = os.listdir(root)
    file_name.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    file_name = file_name
    volume = None

    for f_name in file_name:
        file_path = os.path.join(root, f_name)
        volume_img = pydicom.dcmread(file_path).pixel_array
        volume_img = np.expand_dims(volume_img.astype(np.float32), axis=0)
        if volume is None:
            volume = volume_img
        else:
            volume = np.concatenate((volume, volume_img), axis=0)
            dcm_file = pydicom.dcmread(os.path.join(root, file_name[0]))
    dcm_file = pydicom.dcmread(os.path.join(root, file_name[0]))
    rescale_slope = dcm_file.get("RescaleSlope")
    rescale_intercept = dcm_file.get("RescaleIntercept")
    volume = volume * rescale_slope + rescale_intercept

    dcm_file = pydicom.dcmread(os.path.join(root, file_name[0]))
    pixel_spacing = dcm_file.get("PixelSpacing")
    slice_thickness = dcm_file.get("SliceThickness")
    pixel_spacing = np.array(pixel_spacing)
    pixel_spacing = np.append(pixel_spacing, slice_thickness)
    pixel_spacing[0], pixel_spacing[2] = pixel_spacing[2], pixel_spacing[0]
    volume = np.swapaxes(volume, 1, 2).copy()

    volume = volume[:, :, :771]

    volume = zoom(volume, factors, order=1)
    spacing = pixel_spacing / factors
    volume = np.flip(volume, axis=0).copy()
    volume = volume[1:, :, :]
    volume = volume[::-1, ::-1, :]
    print(spacing)

    affine = np.eye(4)  # 单位矩阵，如果有 voxel spacing 可以修改对角元素
    nii_img = nib.Nifti1Image(volume, affine)
    nib.save(nii_img, 'zyl.nii')

    # for i in range(150, volume.shape[1]):
    #     p = volume[i, :, :]
    #     plt.figure()
    #     plt.imshow(p)
    #     plt.show()

    return volume, spacing

def rename_bone_to_drr(directory):
    """
    将目录下所有 bone_ 开头的 .nii.gz 文件改名为 drr_ 开头
    例如：bone_101_0005.nii.gz  ->  drr_101_0005.nii.gz
    """
    # 转换为 Path 对象，更好处理路径
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"目录不存在: {dir_path}")
        return

    if not dir_path.is_dir():
        print(f"这不是一个目录: {dir_path}")
        return

    count = 0

    # 遍历目录下所有 .nii.gz 文件
    for file_path in dir_path.glob("*.nii.gz"):
        filename = file_path.name

        # 只处理以 bone_ 开头的文件
        if filename.startswith("bone_"):
            # 构造新文件名：把 bone_ 替换成 drr_
            new_filename = "drr_" + filename[5:]  # 5 是 "bone_" 的长度
            new_path = file_path.with_name(new_filename)

            # 检查目标文件是否已存在，避免覆盖
            if new_path.exists():
                print(f"跳过（目标文件已存在）: {new_filename}")
                continue

            # 真正执行重命名
            try:
                file_path.rename(new_path)
                print(f"已重命名: {filename}  →  {new_filename}")
                count += 1
            except Exception as e:
                print(f"重命名失败: {filename}  → 错误: {e}")

    print(f"\n完成！共处理并重命名了 {count} 个文件")

def rename_files_specific_prefix(directory, old_prefix, new_prefix):
    for filename in os.listdir(directory):
        if "_" in filename:
            parts = filename.split("_", 1)  # 分成前缀和剩余部分
            prefix, rest = parts
            if prefix == old_prefix:  # 只修改匹配的前缀
                new_filename = f"{new_prefix}_{rest}"
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                os.rename(old_path, new_path)
                print(f"已重命名: {filename} -> {new_filename}")

if __name__ == "__main__":
    # DICOM 转 NIfTI
    # convert_folder("/home/zsr/project/diffpose/ours/bone_seg/imagesTr", "/home/zsr/project/diffpose/ours/bone_seg/nnUNet_raw_tmp/imagesTr", file_type="dcm")
    # convert_folder("/home/zsr/project/diffpose/ours/bone_seg/qbt", "/home/zsr/project/diffpose/ours/bone_seg/qbt_nii", file_type="dcm")
    # convert_folder("/media/sda1/PersonalFiles/yx/project/diffpose/ours/bone_seg/nnUNet_raw_tmp/imagesTr", "/media/sda1/PersonalFiles/yx/project/diffpose/ours/bone_seg/nnUNet_raw/imagesTr", file_type="dcm")
    # convert_folder("/media/sda1/Data/ERCP/CT+X+MRCP/liwei/许玉坝/ERCP/XU^YUBEI^/20240731150611/1", "/media/sda1/PersonalFiles/yx/project/diffpose/ours/bone_seg/xyb_nii", file_type="dcm")

    # NRRD 转 NIfTI
    # convert_folder("/media/sda1/PersonalFiles/yx/project/diffpose/ours/bone_seg/nnUNet_raw_tmp/labelsTr", "/media/sda1/PersonalFiles/yx/project/diffpose/ours/bone_seg/nnUNet_raw/labelsTr", file_type="nrrd")

    # nii_file = "/home/zsr/project/diffpose/ours/bone_seg/nnUNet_raw_tmp/imagesTr"
    # nii_file = "/home/zsr/project/diffpose/ours/bone_seg/zyl_result150"
    # nii_file = "/home/zsr/project/diffpose/ours/bone_seg/deep_bone/imagesTr"
    # nii_file = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/bone_seg/xyb_result2_100"
    # nii_file = "/home/zsr/project/diffpose/ours/bone_seg/sxh_result2_100"
    # visualize_nii_folder(nii_file)

    # folder = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/cnnnet/runs/mask/zyl"
    # rename_files_specific_prefix(folder, "ssimdicezyl", "ssimzyl")

    # rename_bone_to_drr(nii_file)
    # root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    # f = 0.5
    # dcm2nii_volume_sag(root, factors=[3 * f, 0.450777 * f, 0.450777 * f])

    nii_file = "/home/zsr/project/diffpose/ours/bone_seg/wch_result2_100_reverse"
    save_folder = "/home/zsr/project/diffpose/ours/bone_seg/wch_result2_100_reverse"
    process_and_save_nii_folder(nii_file, save_folder)
    # process_and_save_nii_file("/media/sda1/PersonalFiles/yx/project/diffpose/ours/bone_seg/zyl_result2_100_reverse/95557672_20240311_1_230.nii.gz")

