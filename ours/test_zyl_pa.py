import os
import time
import sys

from ours.register_zyl_bone_mask_stage import print_tre
from ours.utils.CT_dataset_PA import IntubationDataset
from utils.generate_tube import get_tube_on_image

sys.path.append('/home/zsr/project/diffpose/ours/cut')
sys.path.append('/home/zsr/project/diffpose/ours')
import cv2
import torch
import numpy as np
import pydicom
import nibabel as nib

from cut.style_to_drr import *
from utils.drr_bone import DRR as DRR_Bone
from utils.drr import DRR
from matplotlib import pyplot as plt
# from diffpose.deepfluoro import get_random_offset
from my_util2 import get_random_offset

from ours.utils.CT_dataset import Transforms
from scipy.ndimage import zoom
from diffpose.calibration import RigidTransform, convert

device = torch.device("cuda:0")
root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
# root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
file_name = os.listdir(root)
file_name.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

# dcm_file = pydicom.dcmread("/media/sda1/Data/ERCP/CT+X+MRCP/liwei/何美珠/CT/HeMeiZhu/20240710193503.877000/2/93356399_20240710_1_1288.dcm")
dcm_file = pydicom.dcmread(os.path.join(root, file_name[0]))
world2volume_trans = dcm_file.get("ImagePositionPatient")
world2volume_trans[2] = -world2volume_trans[2]
distance_to_detector = dcm_file.get("DistanceSourceToDetector")
distance_to_patient = dcm_file.get("DistanceSourceToPatient")
focal = distance_to_patient * distance_to_detector / (distance_to_detector + distance_to_patient)

x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
# x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
x_file = os.listdir(x_root)
x_filename = os.path.join(x_root, x_file[0])
x_ray = pydicom.dcmread(x_filename)
x_distance_to_detector = x_ray.get("DistanceSourceToDetector")
x_img = x_ray.pixel_array

transformer = Transforms(256, radius=119)
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


# plt.figure()
# plt.imshow(x_img_tensor.squeeze(), cmap="gray")
# plt.show()
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
start = 0
end = 450
# volume = volume[start : end, : , :]
# factors = [2, 2, 4]
factors = [3, 0.5, 0.5]
pixel_spacing[0], pixel_spacing[2] = pixel_spacing[2], pixel_spacing[0]
pixel_spacing = pixel_spacing / factors
volume = np.swapaxes(volume, 1, 2).copy()
# volume = volume[:, :, :600]
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

isocenter_xyz = [116 + 5, 772 + 1000, 1258 - 600 - 150] * pixel_spacing / 2 * factors
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

center_xyz = [116 + 5, 772 + 1000, 600 - 150] * pixel_spacing / 2 * factors
center_xyz = torch.tensor(center_xyz).unsqueeze(0)
center_rot = torch.tensor([[0.0, 0.0, 0.0]]).unsqueeze(0)
center_pose = RigidTransform(
    center_rot, center_xyz, "euler_angles", "ZYX"
)
center_pose = center_pose.to(device)

back_xyz = [-116 - 5, -772 - 1000, -600 + 150] * pixel_spacing / 2 * factors
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

isocenter_pose = isocenter_pose.compose(offset)
root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
# root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
# x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
specimen = IntubationDataset(root, x_root, y_offset=300, z_cut=650, factors=[2, 0.3, 0.3])
gt_pose = specimen.get_manual_gt(0).to(device)
gt_img = drr(None, None, None, pose=gt_pose, bone_attenuation_multiplier=3)
gt_img = transformer(gt_img).to(device).to(torch.float32)
true_fiducials, pred_fiducials = specimen.get_2d_fiducials(0, isocenter_pose)
print_tre(gt_img, true_fiducials[0].detach().numpy())

t = 0
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
#     # pose = pose.compose(gt_pose)
#     # pose = gt_pose.compose()
#
#     rot = pose.get_rotation(parameterization="so3_log_map")
#     xyz = pose.get_translation()
#     rot[0][2] += t
#     t += 0.1
#     calc_pose = RigidTransform(rot, xyz, parameterization="so3_log_map", device=device)
#
#     img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=3)
#     init_z += torch.pi / 40
#
#     true_fiducials, pred_fiducials = specimen.get_2d_fiducials(0, pose)
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


img = drr(None, None, None, pose=isocenter_pose, bone_attenuation_multiplier=3)

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

# for _ in range(40):
#     offset = get_random_offset(4, device)
#     pose = isocenter_pose.compose(back_pose).compose(offset).compose(center_pose)
#     img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=3)
#     img = transformer(img).to(torch.float32)
#     for im in img:
#         plt.figure()
#         plt.imshow(im.cpu().squeeze(), cmap='gray')
#         plt.show()

drr = DRR(
    specimen.volume,
    specimen.spacing,
    specimen.sdr,
    height,
    delx,
    reverse_x_axis=True,
    patch_size=height // 2
).to(device)

i = 900
j = 0
gt_pose = specimen.get_manual_gt().to(device)
x = np.array([0.63, -0.34, 0.08, 46, -66, -26])
x = torch.tensor(x, dtype=torch.float32, device=device)
rot = x[:3].unsqueeze(0)
xyz = x[3:].unsqueeze(0)
gt_pose = RigidTransform(rot, xyz, parameterization="so3_log_map", device=device)
for _ in range(225):
    offset = get_random_offset(4, device)
    # pose = isocenter_pose.compose(offset)
    pose = gt_pose.compose(back_pose).compose(offset).compose(center_pose)
    start_time = time.time()
    img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=3)
    # img_bone = drr_bone(None, None, None, pose=pose, bone_attenuation_multiplier=3)
    print("batch4执行了：" + str(time.time() - start_time))
    # img = transformer(img)
    # img = get_tube_on_image(img, black=False)
    img = transformer(img, reverse=False)
    # img_bone = transformer(img_bone, reverse=False)
    # img_bone = (img_bone - img_bone.min()) / (img_bone.max() - img_bone.min())
    # img_bone[img_bone >= 0.01] = 1
    # img_bone[img_bone < 0.01] = 0
    # img = (img - img.min()) / (img.max() - img.min())
    # img = 1 - img
    # img = changer(img)

    for im in img:
        im_numpy = im.cpu().squeeze().numpy()
        im_numpy = (im_numpy - im_numpy.min()) / (im_numpy.max() - im_numpy.min())
        im_numpy = (im_numpy * 255).astype(np.uint8)
        # cv2.imwrite(f"drrStyle_white/trainB/drr_{i}.png", im_numpy)
        # nib.save(nib.Nifti1Image(im_numpy, np.eye(4)), f"/media/sda1/PersonalFiles/yx/dataset/nnUNet_raw/Dataset001_BrainTumour/imagesTr/drr_{str(i).zfill(4)}_0000.nii.gz")
        i += 1

        plt.figure()
        plt.imshow(im.cpu().squeeze(), cmap="gray")
        plt.show()

    # for im in img_bone:
    #     im_numpy = im.cpu().squeeze().numpy().astype(np.int8)
    #     nib.save(nib.Nifti1Image(im_numpy, np.eye(4)), f"/media/sda1/PersonalFiles/yx/dataset/nnUNet_raw/Dataset001_BrainTumour/labelsTr/drr_{str(j).zfill(4)}.nii.gz")
    #     j += 1

        # plt.figure()
        # plt.imshow(im.cpu().squeeze(), cmap="gray")
        # plt.show()
#
# print("done")
