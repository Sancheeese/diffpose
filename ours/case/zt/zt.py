import os
import time
import sys

sys.path.append('/home/zsr/project/diffpose/ours/cut')
sys.path.append('/home/zsr/project/diffpose/ours')
import cv2
import torch
import numpy as np
import pydicom

from ours.cut.style_to_drr import *
from ours.utils.drr_bone import DRR as DRR_Bone
from ours.utils.drr import DRR
from matplotlib import pyplot as plt
from ours.case.my_util2 import get_random_offset

from ours.utils.CT_dataset import Transforms
from scipy.ndimage import zoom
from diffpose.calibration import RigidTransform

device = torch.device("cuda:2")
root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/朱婷/CT/ZhuTing/20231016155152.774/1005"
file_name = os.listdir(root)
file_name.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

# dcm_file = pydicom.dcmread("/media/sda1/Data/ERCP/CT+X+MRCP/liwei/何美珠/CT/HeMeiZhu/20240710193503.877000/2/93356399_20240710_1_1288.dcm")
dcm_file = pydicom.dcmread(os.path.join(root, file_name[0]))
world2volume_trans = dcm_file.get("ImagePositionPatient")
world2volume_trans[2] = -world2volume_trans[2]
distance_to_detector = dcm_file.get("DistanceSourceToDetector")
distance_to_patient = dcm_file.get("DistanceSourceToPatient")
focal = distance_to_patient * distance_to_detector / (distance_to_detector + distance_to_patient)

x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/朱婷/ERCP/TING^ZHU^/20240515162906/1"
x_file = os.listdir(x_root)
x_filename = os.path.join(x_root, x_file[0])
x_ray = pydicom.dcmread(x_filename)
x_distance_to_detector = x_ray.get("DistanceSourceToDetector")
x_img = x_ray.pixel_array

transformer = Transforms(256)
transformer_x = Transforms(512)
# changer = StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_nec5/40_net_G.pth",
#                        device=device,
#                        resize=256)
# for x_name in x_file:
#     img = pydicom.dcmread(os.path.join(x_root, x_name)).pixel_array
#     img = img[None, None, :]
#
#     img = changer(img)
#
#     plt.figure()
#     plt.imshow(img.squeeze().squeeze().detach().cpu(), cmap='gray')
#     plt.show()

# plt.figure()
# plt.imshow(x_img, cmap="gray")
# plt.show()
print("X_max:", x_img.max())
print("X_min:", x_img.min())

x_img_tensor = torch.tensor(x_img, dtype=torch.float32).unsqueeze(0)
# x_img_tensor = transformer(x_img_tensor)
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
plt.scatter(100, 200, color='red')
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
    # volume_img = pydicom.dcmread(file_path).pixel_array
    f = pydicom.dcmread(file_path)
    volume_img = f.pixel_array
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
# factors = [0.5, 4, 0.5]
factors = [1, 1, 1]
# pixel_spacing[0], pixel_spacing[2] = pixel_spacing[2], pixel_spacing[0]
pixel_spacing = pixel_spacing / factors
volume = np.swapaxes(volume, 0, 2).copy()
volume = volume[:, :, :250]
volume = zoom(volume, factors, order=1)
# volume = np.flip(volume, axis=2).copy()

sdr = 500

drr = DRR(
    volume,
    pixel_spacing,
    sdr,
    height,
    delx=delx,
    reverse_x_axis=True,
    patch_size=height // 2
).to(device)

drr_bone = DRR_Bone(
    volume,
    pixel_spacing,
    sdr,
    height,
    delx=delx,
    reverse_x_axis=True,
    patch_size=height // 2
).to(device)

isocenter_xyz = [512, 512 + 250, 250 - 50] * pixel_spacing / 2 * factors
isocenter_xyz = torch.tensor(isocenter_xyz).unsqueeze(0)
# isocenter_rot = torch.tensor([[0.0, 0.0, torch.pi / 2 + torch.pi / 10]]).unsqueeze(0)
isocenter_rot = torch.tensor([[torch.pi / 2, 0.0, torch.pi / 2]]).unsqueeze(0)
isocenter_pose = RigidTransform(
    isocenter_rot, isocenter_xyz, "euler_angles", "ZYX"
)
isocenter_pose = isocenter_pose.to(device)

contrast_distribution = torch.distributions.Uniform(2.0, 4.0)
contrast = contrast_distribution.sample().item()

init_z = 0

center_xyz = [512, 512 + 250, 250 - 50] * pixel_spacing / 2 * factors
center_xyz = torch.tensor(center_xyz).unsqueeze(0)
center_rot = torch.tensor([[0.0, 0.0, 0.0]]).unsqueeze(0)
center_pose = RigidTransform(
    center_rot, center_xyz, "euler_angles", "ZYX"
)
center_pose = center_pose.to(device)

back_xyz = [-512, -512 - 250, -250 + 50] * pixel_spacing / 2 * factors
back_xyz = torch.tensor(back_xyz).unsqueeze(0)
back_rot = torch.tensor([[0.0, 0.0, 0.0]]).unsqueeze(0)
back_pose = RigidTransform(
    back_rot, back_xyz, "euler_angles", "ZYX"
)
back_pose = back_pose.to(device)

offset_xyz = [0.0 , 0.0, 0.0] * pixel_spacing / 2 * factors
offset_xyz = torch.tensor(offset_xyz).unsqueeze(0)
offset_rot = torch.tensor([[0.0, 0.0,torch.pi / 2]]).unsqueeze(0)
offset = RigidTransform(
    offset_xyz, offset_rot, "euler_angles", "ZYX"
)
offset = offset.to(device)

# for i in range(50):
#     pose_xyz = [0.0, 0.0, 0.0]
#     pose_xyz = torch.tensor(pose_xyz).unsqueeze(0)
#     # pose_rot = torch.tensor([[torch.pi / 4, -2 * torch.pi / 50, torch.pi / 9]]).unsqueeze(0)
#     pose_rot = torch.tensor([[torch.pi / 4 + init_z, 0.0, torch.pi / 2]]).unsqueeze(0)
#     pose = RigidTransform(
#         pose_rot, pose_xyz, "euler_angles", "ZYX"
#     )
#     pose = pose.to(device)
#     up_xyz = [230, -25, 0.0]
#     up_xyz = torch.tensor(up_xyz).unsqueeze(0)
#     up_rot = torch.tensor([[0.0, 0.0, 0.0]]).unsqueeze(0)
#     up = RigidTransform(
#         up_rot, up_xyz, "euler_angles", "ZYX"
#     )
#     up = up.to(device)
#
#     pose = isocenter_pose.compose(back_pose).compose(up).compose(pose).compose(center_pose)
#
#     rotation = pose.get_rotation()
#     translation = pose.get_translation()
#     calc_pose = RigidTransform(rotation, translation)
#
#     img = drr(None, None, None, pose=calc_pose, bone_attenuation_multiplier=3)
#     init_z += torch.pi / 40
#
#     true_fiducials, pred_fiducials = specimen.get_2d_fiducials(0, calc_pose)
#     tre = torch.norm(true_fiducials - pred_fiducials, dim=2)
#     tre = torch.mean(tre)
#
#     print_tre(img, pred_fiducials[0].detach().numpy())
#
#     # img = changer(img, reverse=True)
#     # plt.figure()
#     # plt.imshow(img.cpu().squeeze(), cmap="gray")
#     # plt.title(str(init_z))
#     # plt.show()
#     # img = transformer(img)
#     # plt.figure()
#     # plt.imshow(img.cpu().squeeze(), cmap="gray")
#     # plt.title(str(init_z))
#     # plt.show()


img = drr_bone(None, None, None, pose=isocenter_pose, bone_attenuation_multiplier=3)
img_bone = torch.tensor(img).to(torch.float32)
img_bone = transformer(img_bone, reverse=False)
img_bone = (img_bone - img_bone.min()) / (img_bone.max() - img_bone.min())
img_bone = torch.tanh(50 * img_bone)
print("img " + str(img.min()))
print("img " + str(img.max()))

# img = (img - img.min()) / (img.max() - img.min())
# img = 1 - img
# img = -img
img = transformer(img)
plt.figure()
plt.imshow(img.cpu().squeeze(), cmap="gray")
plt.show()
plt.figure()
plt.imshow(img_bone.cpu().squeeze(), cmap="gray")
plt.show()

for _ in range(40):
    offset = get_random_offset(4, device)
    pose = isocenter_pose.compose(back_pose).compose(offset).compose(center_pose)
    img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=3)
    img = transformer(img).to(torch.float32)

    img_bone = drr_bone(None, None, None, pose=pose, bone_attenuation_multiplier=3)
    img_bone = torch.tensor(img_bone).to(torch.float32)
    img_bone = transformer(img_bone, reverse=False)
    img_bone = (img_bone - img_bone.min()) / (img_bone.max() - img_bone.min())
    img_bone = torch.tanh(70 * img_bone)
    for im in img_bone:
        plt.figure()
        plt.imshow(im.cpu().squeeze(), cmap='gray')
        plt.show()

