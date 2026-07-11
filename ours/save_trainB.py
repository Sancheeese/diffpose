import os
import time
import sys

from ours.register_zyl_bone_mask_stage import print_tre
from ours.utils.CT_dataset_PA import IntubationDataset, toZeroOne
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
device = torch.device("cuda:1")
transformer = Transforms(256, radius=119)
root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
# root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
# x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"


# specimen = IntubationDataset(root, x_root, y_offset=300, z_cut=650, factors=[3, 1, 1])
specimen = IntubationDataset(root, x_root, y_offset=300, factors=[3, 1, 1])
isocenter_pose = specimen.isocenter_pose.to(device)
center_pose = specimen.center_pose.to(device)
back_pose = specimen.back_pose.to(device)
height = 256
subsample = 512 / height
delx = specimen.delx * subsample
drr = DRR(
    specimen.volume,
    specimen.spacing,
    specimen.sdr,
    height,
    delx,
    reverse_x_axis=True,
    patch_size=height // 2
).to(device)
gt_pose = specimen.get_manual_gt(0).to(device)
gt_img = drr(None, None, None, pose=gt_pose, bone_attenuation_multiplier=3)
gt_img = transformer(gt_img).to(device).to(torch.float32)
true_fiducials, pred_fiducials = specimen.get_2d_fiducials(0, isocenter_pose)
print_tre(gt_img, true_fiducials[0].detach().numpy())

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

i = 500
j = 0
gt_pose = specimen.get_manual_gt().to(device)
# x = np.array([0.63, -0.34, 0.08, 46, -66, -26])
# x = np.array([0.57, -0.2, 0.03, 64, -34, -18])
x = np.array([  1.6688,  -0.9184,  -0.8960, 193.6940,  93.6377, 113.5820])
x = torch.tensor(x, dtype=torch.float32, device=device)
rot = x[:3].unsqueeze(0)
xyz = x[3:].unsqueeze(0)
# pose = RigidTransform(rot, xyz, "euler_angles", "ZYX")
# gt_pose = isocenter_pose.compose(back_pose).compose(pose).compose(center_pose)
gt_pose = RigidTransform(rot, xyz, parameterization="so3_log_map")
for _ in range(250):
    offset = get_random_offset(2, device)
    # pose = isocenter_pose.compose(offset)
    pose = gt_pose.compose(back_pose).compose(offset).compose(center_pose)
    start_time = time.time()
    img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=3)
    # img_bone = drr_bone(None, None, None, pose=pose, bone_attenuation_multiplier=3)
    print("batch4执行了：" + str(time.time() - start_time))
    # img = transformer(img)
    # img = get_tube_on_image(img, black=False)
    img = transformer(img, reverse=True)
    # img_bone = transformer(img_bone, reverse=False)
    # img_bone = (img_bone - img_bone.min()) / (img_bone.max() - img_bone.min())
    # img_bone[img_bone >= 0.01] = 1
    # img_bone[img_bone < 0.01] = 0
    # img = (img - img.min()) / (img.max() - img.min())
    # img = 1 - img
    # img = changer(img)

    for im in img:
        im = toZeroOne(im)
        im = torch.pow(im, 1.1)
        im_numpy = im.cpu().squeeze().numpy()
        im_numpy = (im_numpy - im_numpy.min()) / (im_numpy.max() - im_numpy.min())
        im_numpy = (im_numpy * 255).astype(np.uint8)
        cv2.imwrite(f"drrStyle_iso3/trainB/drr_{i}.png", im_numpy)
        # nib.save(nib.Nifti1Image(im_numpy, np.eye(4)), f"/media/sda1/PersonalFiles/yx/dataset/nnUNet_raw/Dataset001_BrainTumour/imagesTr/drr_{str(i).zfill(4)}_0000.nii.gz")
        i += 1

        # plt.figure()
        # plt.imshow(im.cpu().squeeze(), cmap="gray")
        # plt.show()

    # for im in img_bone:
    #     im_numpy = im.cpu().squeeze().numpy().astype(np.int8)
    #     nib.save(nib.Nifti1Image(im_numpy, np.eye(4)), f"/media/sda1/PersonalFiles/yx/dataset/nnUNet_raw/Dataset001_BrainTumour/labelsTr/drr_{str(j).zfill(4)}.nii.gz")
    #     j += 1

        # plt.figure()
        # plt.imshow(im.cpu().squeeze(), cmap="gray")
        # plt.show()
#
# print("done")