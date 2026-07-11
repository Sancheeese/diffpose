import os
import time

import cv2
import torch
import numpy as np
import pydicom

from utils.drr import DRR
from matplotlib import pyplot as plt

# from diffpose.deepfluoro import get_random_offset
from my_util2 import get_random_offset

from diffpose.calibration import RigidTransform
from ours.dataset.CT_dataset import Transforms

device = torch.device("cuda:0")
# root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/何美珠/CT/HeMeiZhu/20240710193503.877000/2"
root = "/home/zsr/project/diffpose/ours/data/liwei/史建基/CT/ShiJianJi/20231019091100.063/603"
# root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/史建基/CT/ShiJianJi/20231019091100.063/603"
file_name = os.listdir(root)
file_name.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
# temp.sort()
# file_name = []
# file_name.extend(temp[91 : ])
# file_name.extend(temp[0 : 91])

# dcm_file = pydicom.dcmread("/media/sda1/Data/ERCP/CT+X+MRCP/liwei/何美珠/CT/HeMeiZhu/20240710193503.877000/2/93356399_20240710_1_1288.dcm")
dcm_file = pydicom.dcmread(os.path.join(root, file_name[0]))

distance_to_detector = dcm_file.get("DistanceSourceToDetector")
distance_to_patient = dcm_file.get("DistanceSourceToPatient")
focal = distance_to_patient * distance_to_detector / (distance_to_detector + distance_to_patient)

# x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/何美珠/ERCP/HEMEIZHU^^/20240712152119/1/"
x_root = "/home/zsr/project/diffpose/ours/data/liwei/史建基/ERCP/SHI^JIANJI^/20231023150900/1"
# x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/史建基/ERCP/SHI^JIANJI^/20231023150900/1"
x_file = os.listdir(x_root)
x_filename = os.path.join(x_root, x_file[0])
# x_ray = pydicom.dcmread("/home/zsr/project/diffpose/ours/data/liwei/史建基/ERCP/SHI^JIANJI^/20231023150900/1/94108134_20231023_1_29.dcm")
x_ray = pydicom.dcmread(x_filename)
# x_ray = pydicom.dcmread(os.path.join(x_root, x_filename))
x_distance_to_detector = x_ray.get("DistanceSourceToDetector")
x_img = x_ray.pixel_array

transformer = Transforms(256)
transformer_x = Transforms(512)
# for x_name in x_file:
#     img = pydicom.dcmread(os.path.join(x_root, x_name)).pixel_array
#
#     plt.figure()
#     plt.imshow(img, cmap='gray')
#     plt.show()

# plt.figure()
# plt.imshow(x_img, cmap="gray")
# plt.show()
print("X_max:", x_img.max())
print("X_min:", x_img.min())

x_img_tensor = torch.tensor(x_img, dtype=torch.float32).unsqueeze(0)
x_img_tensor = transformer(x_img_tensor, reverse=False)
# x_img_tensor = (x_img_tensor - x_img_tensor.min()) / (x_img_tensor.max() - x_img_tensor.min())
# target_image = torch.tensor(x_img_tensor * 255, dtype=torch.uint8)
# target_image = np.array(target_image, dtype=np.uint8)
#
# target_image[target_image < 50] = 0
#
# plt.figure()
# plt.imshow(target_image[0][0], cmap="gray")
# plt.show()


plt.figure()
plt.imshow(x_img_tensor.squeeze(), cmap="gray")
plt.show()
print("X_max_after:", x_img_tensor.max())
print("X_min_after:", x_img_tensor.min())

pixel_spacing = dcm_file.get("PixelSpacing")
slice_thickness = dcm_file.get("SliceThickness")
pixel_spacing = np.array(pixel_spacing)
pixel_spacing = np.append(pixel_spacing, slice_thickness)
distance_to_detector = dcm_file.get("DistanceSourceToDetector")
intensifier_size = x_ray.get("IntensifierSize")
delx = intensifier_size / 512
height = 256
delx = 512 / height * delx

volume = None

for f_name in file_name:
    file_path = os.path.join(root, f_name)
    volume_img = pydicom.dcmread(file_path).pixel_array
    # plt.figure()
    # plt.imshow(volume_img, cmap='gray')
    # plt.show()
    volume_img = np.expand_dims(volume_img.astype(np.float32), axis=0)
    if volume is None:
        volume = volume_img
    else:
        volume = np.concatenate((volume, volume_img), axis=0)

rescale_slope = dcm_file.get("RescaleSlope")
rescale_intercept = dcm_file.get("RescaleIntercept")
volume = volume * rescale_slope + rescale_intercept
# plt.figure()
# plt.imshow(volume[0], cmap='gray')
# plt.show()
# volume = volume[30 : 100, :, :]

# plt.figure()
# plt.imshow(volume[0], cmap="gray")
# plt.show()
start = 0
end = 450
# volume = volume[start : end, : , :]
# factors = [2, 2, 4]
factors = [1, 1, 1]
pixel_spacing = pixel_spacing / factors
pixel_spacing[0], pixel_spacing[2] = pixel_spacing[2], pixel_spacing[0]
volume = np.swapaxes(volume, 1, 2).copy()
z_cut = 500
volume = volume[:, :, :z_cut]
# volume = volume[:, :, start : end]
# volume = zoom(volume, factors, order=1)
sdr = 500

drr = DRR(
    volume,
    pixel_spacing,
    sdr,
    height,
    delx=delx,
    reverse_x_axis=True
).to(device)

# isocenter_xyz = [512, 512, len(file_name)] * pixel_spacing / 2 * factors
isocenter_xyz = [135, 900, z_cut] * pixel_spacing / 2 * factors
# isocenter_xyz = [135, 900, 980] * pixel_spacing / 2 * factors
isocenter_xyz = torch.tensor(isocenter_xyz).unsqueeze(0)
# isocenter_rot = torch.tensor([[0.0, 0.0, torch.pi / 2]]).unsqueeze(0)
isocenter_rot = torch.tensor([[0.0, 0.0, torch.pi / 2 + torch.pi / 8]]).unsqueeze(0)

isocenter_pose = RigidTransform(
    isocenter_rot, isocenter_xyz, "euler_angles", "ZYX"
)
isocenter_pose = isocenter_pose.to(device)

contrast_distribution = torch.distributions.Uniform(1.0, 5.0)
contrast = contrast_distribution.sample().item()


init_z = 0

center_xyz = [135, 900, z_cut] * pixel_spacing / 2
center_xyz = torch.tensor(center_xyz).unsqueeze(0)
center_rot = torch.tensor([[0.0, 0.0, 0.0]]).unsqueeze(0)
center_pose = RigidTransform(
    center_rot, center_xyz, "euler_angles", "ZYX"
)
center_pose = center_pose.to(device)

back_xyz = [-135, -900, -z_cut] * pixel_spacing / 2
back_xyz = torch.tensor(back_xyz).unsqueeze(0)
back_rot = torch.tensor([[0.0, 0.0, 0.0]]).unsqueeze(0)
back_pose = RigidTransform(
    back_rot, back_xyz, "euler_angles", "ZYX"
)
back_pose = back_pose.to(device)

# for i in range(20):
#     pose_xyz = [0.0, 0.0, 0.0]
#     pose_xyz = torch.tensor(pose_xyz).unsqueeze(0)
#     pose_rot = torch.tensor([[init_z + torch.pi / 40, 0.0, 0.0]]).unsqueeze(0)
#     pose = RigidTransform(
#         pose_rot, pose_xyz, "euler_angles", "ZYX"
#     )
#     pose = pose.to(device)
#     pose = isocenter_pose.compose(back_pose).compose(pose).compose(center_pose)
#     # pose = pose.compose(isocenter_pose)
#     img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=3)
#     init_z += torch.pi / 40
#     # img = transformer(img)
#     plt.figure()
#     plt.imshow(img.cpu().squeeze(), cmap="gray")
#     plt.title(str(init_z))
#     plt.show()

img = drr(None, None, None, pose=isocenter_pose, bone_attenuation_multiplier=5)

print("img " + str(img.min()))
print("img " + str(img.max()))

# img = (img - img.min()) / (img.max() - img.min())
# img = 1 - img
# img = -img
img = transformer(img)
plt.figure()
plt.imshow(img.cpu().squeeze(), cmap="gray")
plt.show()


print("drr_max:" + str(img.max()))
print("drr_min:" + str(img.min()))
# img_tensor = torch.tensor(img, dtype=torch.float32)
# img_tensor = transformer(img_tensor)
#
# plt.figure()
# plt.imshow(img_tensor.cpu().squeeze(), cmap="gray")
# plt.show()
# print("drr_max_after:" + str(img_tensor.max()))
# print("drr_min_after:" + str(img_tensor.min()))


# def compute_cdf(image, mask=None):
#     if mask is not None:
#         image = image[mask]
#     else:
#         image = image.flatten()
#
#     # 将图像的像素值展开为一个一维数组，并计算直方图
#     hist, bins = np.histogram(image, bins=256, range=(0, 256), density=True)
#
#     # 计算累积分布函数（CDF）
#     cdf = np.cumsum(hist)
#     cdf_normalized = cdf / cdf[-1]  # 归一化 CDF
#     return cdf_normalized, bins[:-1]
#
#
# # 直方图匹配
# def histogram_matching(source, target):
#     size = 256
#     radius = 121
#     # 创建坐标数组
#     y_coord = np.arange(size) - size // 2
#     x_coord = np.arange(size) - size // 2
#     # 创建网格
#     Y, X = np.meshgrid(y_coord, x_coord)
#     # 计算每个点到中心的平方距离
#     distance_sq = X ** 2 + Y ** 2
#     # 创建圆形mask，满足条件的区域为1，其余为0
#     circle_mask = (distance_sq <= radius ** 2).astype(bool)
#     # mask = (distance_sq <= radius ** 2).astype(float)
#     # mask = mask
#     # plt.figure()
#     # plt.imshow(target * mask, cmap="gray")
#     # plt.show()
#
#     tube_mask = (target > 50).astype(bool)
#     # 计算源图像和目标图像的 CDF
#     cdf_source, bins_source = compute_cdf(source)
#     cdf_target, bins_target = compute_cdf(target, circle_mask & tube_mask)
#     # cdf_source, bins_source = compute_cdf(source)
#     # cdf_target, bins_target = compute_cdf(target)
#
#     # 创建映射表，将源图像的灰度值映射到目标图像的灰度值
#     mapping = np.interp(cdf_source, cdf_target, bins_target)
#
#     # 使用映射表调整源图像的灰度值
#     matched_image = np.interp(source.flatten(), bins_source, mapping).reshape(source.shape)
#
#     return matched_image

# img = transformer(img)
# img = (img - img.min()) / (img.max() - img.min())
# img = img.cpu()
# x_img_tensor = (x_img_tensor - x_img_tensor.min()) / (x_img_tensor.max() - x_img_tensor.min())
#
# source_image = np.array(img * 255, dtype=np.uint8)[0][0]
# target_image = np.array(x_img_tensor * 255, dtype=np.uint8)[0]
#
# plt.figure()
# plt.imshow(source_image, cmap="gray", vmin=0, vmax=255)
# plt.title("img")
# plt.show()
# plt.figure()
# plt.imshow(target_image, cmap="gray")
# plt.title("x_img")
# plt.show()
# match_image = histogram_matching(source_image, target_image)
# plt.figure()
# plt.imshow(match_image, cmap="gray", vmin=0, vmax=255)
# plt.title("change")
# plt.show()
# match_image[0][0] = 0
# match_image = torch.tensor(match_image, dtype=torch.float32).unsqueeze(0)
# match_image = transformer(match_image)
# plt.figure()
# plt.imshow(match_image[0], cmap="gray")
# plt.title("change_transform")
# plt.show()


for _ in range(10):
    offset = get_random_offset(2, device)
    pose = isocenter_pose.compose(offset)
    contrast = contrast_distribution.sample().item()
    start_time = time.time()
    img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=3)
    print("batch4执行了：" + str(time.time() - start_time))
    img = transformer(img)

    for im in img:
        plt.figure()
        plt.imshow(im.cpu().squeeze(), cmap="gray")
        plt.show()

# for _ in range(10):
#     contrast = contrast_distribution.sample().item()
#     offset = get_random_offset(1, 0, 10, device)
#     pose = isocenter_pose.compose(offset)
#     img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=4)
#
#     plt.figure()
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.show()

