import torch
from matplotlib import pyplot as plt

from CT_dataset import IntubationDataset
from ours.utils.CT_dataset import Transforms
from ours.utils.drr import DRR

device = "cuda:0"

# root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
# x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
# # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
# # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"

root = "/home/zsr/project/diffpose/ours/data/liwei/王定国/MRCP/WangDingGuo/20240718091937/1101"
x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
# root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
# x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
# specimen = IntubationDataset(root, x_root, y_offset=300, z_cut=650, factors=[2, 0.4, 0.4])
specimen = IntubationDataset(root, x_root)
height = 256
subsample = 512 / height
delx = specimen.delx * subsample

drr = DRR(
    specimen.volume,
    specimen.spacing,
    sdr=specimen.sdr,
    height=height,
    delx=delx,
    reverse_x_axis=True,
    bone_attenuation_multiplier=3,
).to(device)
transforms = Transforms(drr.detector.height, radius=119)

isocenter_pose = specimen.isocenter_pose.to(device)
pose = isocenter_pose
img = drr(None, None, None, pose=isocenter_pose, bone_attenuation_multiplier=3)
img = transforms(img).to(device).to(torch.float32)
# img = flip_img_w(img)
plt.figure()
plt.imshow(img.cpu().squeeze(), cmap="gray")
plt.show()
