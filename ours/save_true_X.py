import os
import time

import cv2
import torch
import numpy as np
import pydicom

from ours.cut.style_to_drr import StyleChanger
from ours.utils.CT_dataset import IntubationDataset, toZeroOne
from utils.drr import DRR
from matplotlib import pyplot as plt
from my_util2 import get_random_offset
from diffpose.calibration import RigidTransform
from ours.utils.CT_dataset import Transforms

transformer = Transforms(256, radius=119)
transforms = Transforms(256, radius=119)
root = "/home/zsr/project/diffpose/ours/data/liwei"
# root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei"
device = torch.torch.device("cuda:1")
# style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_white/80_net_G.pth",
# style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_new3/70_net_G.pth",
# style_change = StyleChanger("/media/sda1/PersonalFiles/yx/project/diffpose/ours/cut/drr_style_solid_5_new3/70_net_G.pth",
# style_change = StyleChanger("/media/sda1/PersonalFiles/yx/project/diffpose/ours/cut/drr_style_solid_5_whi te/70_net_G.pth",
# style_change = StyleChanger("/media/sda1/PersonalFiles/yx/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_white3/70_net_G.pth",
# style_change = StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_iso/80_net_G.pth",
style_change = StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_iso/80_net_G.pth",
# style_change =  StyleChanger("/media/sda1/PersonalFiles/yx/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_iso/80_net_G.pth",
# style_change = StyleChanger("/media/sda1/PersonalFiles/yx/project/diffpose/ours/cut/drr_style_solid_5_new4/150_net_G.pth",
                            device=device, resize=256)

persons = os.listdir(root)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False
s = torch.distributions.Uniform(0.65, 0.8)
for person in persons:
    if os.path.isfile(os.path.join(root, person)):
        continue

    x_root = os.path.join(root, person, "ERCP")
    if not os.path.exists(x_root):
        continue
    while len(os.listdir(x_root)) == 1:
        x_root = os.path.join(x_root, os.listdir(x_root)[0])

    x_files = os.listdir(x_root)
    name = person

    # if person != "姚明钻":
    if person != "邱碧潭":
        continue

    count = 0
    for i, x_name in enumerate(x_files):
        if i >= 20: break
        if os.path.isdir(os.path.join(x_root, x_name)):
            break
        # if count < 20:
        #     count += 1
        # else:
        #     break
        img = pydicom.dcmread(os.path.join(x_root, x_name)).pixel_array
        x_img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        # t = s.sample().item()
        # img_rev = transformer(x_img_tensor, reverse=True)
        # tube = torch.ones_like(img_rev).to(device)
        # tube[toZeroOne(img_rev) > 0.7] = 0
        # tube = toZeroOne(tube) * 255
        # tube = tube.cpu().squeeze()
        # tube = np.array(tube).astype(np.uint8)
        # cv2.imwrite(f"tube/{name}_{i}_1.png", tube)
        #
        # tube = torch.ones_like(img_rev).to(device)
        # tube[toZeroOne(img_rev) > 0.8] = 0
        # tube = toZeroOne(tube) * 255
        # tube = tube.cpu().squeeze()
        # tube = np.array(tube).astype(np.uint8)
        # cv2.imwrite(f"tube/{name}_{i}_2.png", tube)

        # plt.figure()
        # plt.imshow(tube, cmap="gray")
        # plt.show()

        # x_img_tensor = transformer(x_img_tensor, reverse=True)
        x_img_tensor = transformer(x_img_tensor, reverse=False)
        # x_img_tensor = (x_img_tensor - x_img_tensor.min()) / (x_img_tensor.max() - x_img_tensor.min()) * 255
        # img = np.array(x_img_tensor).astype(np.uint8)
        # cv2.imwrite(f"drrStyle_white/trainA/{name}_{i}.png", img)

        plt.figure()
        plt.imshow(x_img_tensor.detach().cpu().squeeze(), cmap='gray')
        plt.title(f"{person}")
        plt.show()

        x_img_tensor = transformer(x_img_tensor, reverse=True)
        img_change = style_change(x_img_tensor)
        img_change = transforms(img_change, reverse=True).to(device).to(torch.float32)
        plt.figure()
        plt.imshow(img_change.detach().cpu().squeeze(), cmap='gray')
        plt.title(f"{person}")
        plt.show()

    print(person)


# import nibabel as nib
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# # for file_num in range (50):
# #
# #     # 1. 读取 .nii.gz 文件
# #     file_path = "/media/sda1/PersonalFiles/yx/dataset/zyl_result/zyl_" + str(file_num).zfill(4) + ".nii.gz"  # 替换为你的文件路径
# #     # file_path = "/media/sda1/PersonalFiles/yx/dataset/nnUNet_test/drr_" + str(file_num).zfill(4) + ".nii.gz"  # 替换为你的文件路径
# #     nii_img = nib.load(file_path)
# #     img_data = nii_img.get_fdata()
# #     img_data = img_data[::-1, :]
# #
# #     # 2. 检查数据维度（3D or 4D）
# #     print("Image shape:", img_data.shape)  # 例如 (256, 256, 60) 表示 3D 数据
# #
# #     # 4. 绘制灰度图
# #     plt.figure(figsize=(8, 6))
# #     plt.imshow(img_data, cmap="gray", origin="lower")  # 转置并设置坐标系
# #     plt.colorbar(label="Intensity")
# #     plt.axis("off")  # 隐藏坐标轴
# #     plt.show()
# #
# #
# #     file_path = "/media/sda1/PersonalFiles/yx/dataset/zyl_change/zyl_" + str(file_num).zfill(4) + "_0000.nii.gz"  # 替换为你的文件路径
# #     # file_path = "/media/sda1/PersonalFiles/yx/dataset/nnUNet_raw/Dataset001_DRR/imagesTr/drr_" + str(file_num).zfill(4) + "_0000.nii.gz"  # 替换为你的文件路径
# #     nii_img = nib.load(file_path)
# #     img_data = nii_img.get_fdata()  # 获取图像数据（numpy 数组）
# #
# #     # 2. 检查数据维度（3D or 4D）
# #     print("Image shape:", img_data.shape)  # 例如 (256, 256, 60) 表示 3D 数据
# #
# #     # 4. 绘制灰度图
# #     plt.figure(figsize=(8, 6))
# #     plt.imshow(img_data, cmap="gray", origin="lower")  # 转置并设置坐标系
# #     plt.colorbar(label="Intensity")
# #     plt.axis("off")  # 隐藏坐标轴
# #     plt.show()
#
#
# device = torch.device("cuda:3")
#
# root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
# x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
# specimen = IntubationDataset(root, x_root, y_offset=155, z_cut=600)
#
# # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_nec5/40_net_G.pth",
# # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec5/70_net_G.pth",
# style_change =  StyleChanger("cut/ckpt/70_net_G.pth",
# # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_4_choose/100_net_G.pth",
#        device=device,
#        resize=256)
#
# for i in range(len(specimen)):
#        img, pose = specimen[i]
#        filename = specimen.get_x_filename(i).split(".")[0] + "_nochange"
#        img = transforms(img, reverse=False).to(device).to(torch.float32)
#        # img = self.transforms(img, reverse=False)
#        img_ori = torch.tensor(img).to(device).to(torch.float32)
#        # img_change = style_change(img)
#        # img_change = transforms(img_change, reverse=False).to(device).to(torch.float32)
#        im_numpy = img.cpu().squeeze().squeeze().numpy()
#        nib.save(nib.Nifti1Image(im_numpy, np.eye(4)),
#                 f"/media/sda1/PersonalFiles/yx/dataset/zyl_change/{filename}_0000.nii.gz")






