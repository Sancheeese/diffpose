import os

import numpy as np
import pydicom
import SimpleITK as sitk

root = "/home/zsr/project/diffpose/ours/data/liwei/陈羽馨/CT/ChenYuXin/20240530105749/304"
x_root = "/home/zsr/project/diffpose/ours/data/liwei/陈羽馨/ERCP/YUXING^CHEN^/20240531155445/1"
# root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/陈羽馨/CT/ChenYuXin/20240530105749/304"
# x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/陈羽馨/ERCP/YUXING^CHEN^/20240531155445/1"
file_name = os.listdir(root)
file_name.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
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
rescale_slope = dcm_file.get("RescaleSlope")
rescale_intercept = dcm_file.get("RescaleIntercept")
volume = volume * rescale_slope + rescale_intercept

min_value = volume.min()
max_value = volume.max()
print(min_value)
print(max_value)
volume[volume < 450] = min_value

volume = volume.astype(np.float32)
# numpy (Z,Y,X) → ITK image
img = sitk.GetImageFromArray(volume)
# 1. spacing（从 DICOM 里读）
dcm = pydicom.dcmread(os.path.join(root, file_name[0]))

pixel_spacing = dcm.PixelSpacing      # [row, col] = [dy, dx]
slice_thickness = float(dcm.SliceThickness)

# 注意顺序是 (x, y, z)
img.SetSpacing((
    float(pixel_spacing[1]),
    float(pixel_spacing[0]),
    slice_thickness
))

# 2. origin（一般可设 0，或用 ImagePositionPatient）
img.SetOrigin((0.0, 0.0, 0.0))

# 3. direction（最安全做法：单位矩阵）
img.SetDirection((1,0,0,
                  0,1,0,
                  0,0,1))

# ====== 保存 ======
sitk.WriteImage(img, "ct_processed.nii.gz")
print()
