import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpmath.calculus.optimization import Ridder
from skimage.morphology import max_tree

from diffpose.calibration import RigidTransform
from ours.cut.style_to_drr import StyleChanger
from ours.utils.CT_dataset import Transforms, create_circle_mask
from ours.utils.CT_dataset import IntubationDataset
from ours.utils.drr import DRR

from utils.metrics_mask_tube2 import MultiscaleNormalizedCrossCorrelation2d

device = torch.device("cuda:0")
root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
specimen = IntubationDataset(root, x_root, y_offset=155, z_cut=600)
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

gt_pose = specimen.get_manual_gt().to(device)
r = gt_pose.get_rotation(parameterization="so3_log_map").to(device)
t = gt_pose.get_translation().to(device)

# tran_best
t = torch.tensor([[364.7583, 294.4677, 172.4318]]).to(device)

# 采样范围（示例：±50mm）
pixel_spacing = 1.0  # 根据实际调整
factors = 1.0  # 根据实际调整
range_mm = 20
step = 5
range_r = 0.5
x_values = np.linspace(-range_mm, range_mm, 20)
y_values = np.linspace(-range_mm, range_mm, 20)
z_values = np.linspace(-range_mm, range_mm, 20)
x_r = np.linspace(-range_r, range_r, 20)
y_r = np.linspace(-range_r, range_r, 20)
z_r = np.linspace(-range_r, range_r, 20)

# 固定X轴，变化Y和Z
fixed_x = 0.0  # 固定X轴为原始值
loss_grid = np.zeros((len(y_values), len(z_values)))
img, _ = specimen[3]

transforms = Transforms(drr.detector.height)
mncc_loss = MultiscaleNormalizedCrossCorrelation2d([None, 30], [0.5, 0.5], device=device)
style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec5/80_net_G.pth",
                       device=device,
                       resize=256)
img = transforms(img, reverse=False).to(device)
img_change = style_change(img)
img_change = transforms(img_change, reverse=False).to(device).to(torch.float32)
diff = img - img_change
diff = (diff - diff.min()) / (diff.max() - diff.min()).to(device)
threshold = 0.5
diff[diff <= threshold] = 0
diff[diff > threshold] = 1
circle_mask = create_circle_mask(256, 119).to(device)
total_mask = (circle_mask.bool() & diff.bool()).float()
mncc_loss.set_mask(total_mask)
lo = mncc_loss(img, img_change)

# gt_img = drr(None, None, None, pose=gt_pose, bone_attenuation_multiplier=3).to(device).to(torch.float32)
# gt_img = transforms(gt_img).to(device).to(torch.float32)
# plt.figure()
# plt.imshow(gt_img.cpu().squeeze(), cmap="gray")
# plt.show()
# lo = mncc_loss(img, img_change)
#
# pose = RigidTransform(r, t, parameterization="so3_log_map")
# gt_img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=3).to(device).to(torch.float32)
# gt_img = transforms(gt_img).to(device).to(torch.float32)
# plt.figure()
# plt.imshow(gt_img.cpu().squeeze(), cmap="gray")
# plt.show()

plt.figure()
plt.imshow(img_change.cpu().squeeze(), cmap="gray")
plt.show()

max_loss = 0
max_t = t
max_r = r
gt_img = drr(None, None, None, pose=gt_pose, bone_attenuation_multiplier=3)
gt_img = transforms(gt_img).to(device).to(torch.float32)
loss = mncc_loss(gt_img, img_change)
print("gt_loss" + str(loss))
for i, y in enumerate(y_values):
    for j, z in enumerate(z_values):
        xyz_perturbation = torch.tensor([fixed_x, y, z]) * pixel_spacing * factors
        t[0][0] += y
        t[0][2] += z
        pose = RigidTransform(r, t, parameterization="so3_log_map", device=device)
        img_sample = drr(None, None, None, pose=pose).to(device).to(torch.float32)
        img_sample = transforms(img_sample).to(device).to(torch.float32)
        # plt.figure()
        # plt.imshow(img_sample.cpu().squeeze(), cmap="gray")
        # plt.show()
        # plt.figure()
        # plt.imshow(img_change.cpu().squeeze(), cmap="gray")
        # plt.show()
        loss = mncc_loss(img_sample, img_change)
        loss_grid[i, j] = loss
        if loss > max_loss:
            max_t = torch.tensor(t).to(device)
            max_loss = loss
        t[0][0] -= y
        t[0][2] -= z

# for i, y in enumerate(y_r):
#     for j, z in enumerate(z_r):
#         xyz_perturbation = torch.tensor([fixed_x, y, z]) * pixel_spacing * factors
#         r[0][0] += y
#         r[0][2] += z
#         pose = RigidTransform(r, t, parameterization="so3_log_map", device=device)
#         img_sample = drr(None, None, None, pose=pose).to(device).to(torch.float32)
#         img_sample = transforms(img_sample).to(device).to(torch.float32)
#         # plt.figure()
#         # plt.imshow(img_sample.cpu().squeeze(), cmap="gray")
#         # plt.show()
#         # plt.figure()
#         # plt.imshow(img_change.cpu().squeeze(), cmap="gray")
#         # plt.show()
#         loss = mncc_loss(img_sample, img)
#         loss_grid[i, j] = loss
#         if loss > max_loss:
#             max_r = torch.tensor(r).to(device)
#             max_loss = loss
#         r[0][0] -= y
#         r[0][2] -= z

# for k, x in enumerate(x_r):
#     for i, y in enumerate(y_r):
#         for j, z in enumerate(z_r):
#             r[0][0] += x
#             r[0][1] += y
#             r[0][2] += z
#
#             pose = RigidTransform(r, t, parameterization="so3_log_map", device=device)
#             img_sample = drr(None, None, None, pose=pose).to(device).to(torch.float32)
#             img_sample = transforms(img_sample).to(device).to(torch.float32)
#             # plt.figure()
#             # plt.imshow(img_sample.cpu().squeeze(), cmap="gray")
#             # plt.show()
#             # plt.figure()
#             # plt.imshow(img_change.cpu().squeeze(), cmap="gray")
#             # plt.show()
#             loss = mncc_loss(img_sample, img_change)
#             loss_grid[i, j] = loss
#             if loss > max_loss:
#                 max_r = torch.tensor(r).to(device)
#                 max_loss = loss
#             r[0][0] -= x
#             r[0][1] -= y
#             r[0][2] -= z

pose = RigidTransform(max_r, max_t, parameterization="so3_log_map", device=device)
img = drr(None, None, None, pose=pose).to(device).to(torch.float32)
img = transforms(img).to(device).to(torch.float32)
plt.figure()
plt.imshow(img.cpu().squeeze(), cmap="gray")
plt.show()
print(max_loss)

# 绘制三维曲面
Y, Z = np.meshgrid(y_values, z_values)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Y, Z, loss_grid.T, cmap='bwr')  # 转置保证维度对齐
ax.set_xlabel('Y (mm)')
ax.set_ylabel('Z (mm)')
ax.set_zlabel('Loss')
ax.set_title(f'Loss vs. Y/Z Perturbation (Fixed X={fixed_x}mm)')
plt.show()

