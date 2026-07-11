import torch
from matplotlib import pyplot as plt

from ours.utils.CT_dataset import IntubationDataset, Transforms
from ours.utils.drr import DRR
from ours.utils.registration_unet_cat import PoseRegressor

device = torch.device("cuda:1")
height = 256
root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
# root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
# x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"

specimen = IntubationDataset(root, x_root, y_offset=155, z_cut=600, factors=[3, 0.5, 0.5])
isocenter_pose = specimen.isocenter_pose.to(device)

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
transforms = Transforms(height)

model_params = {
    "model_name": "resnet34",
    "parameterization": "se3_log_map",
    "convention": None,
    "norm_layer": "groupnorm",
}
model = PoseRegressor(**model_params)
ckpt = torch.load("checkpoints/zyl_800_cat_best.ckpt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)

gt_pose = specimen.get_manual_gt().to(device)
gt_img = drr(None, None, None, pose=gt_pose, bone_attenuation_multiplier=3)
gt_img = transforms(gt_img).to(device).to(torch.float32)
for im in gt_img:
    plt.figure()
    plt.imshow(im.cpu().permute(1, 2, 0), cmap='gray')
    plt.show()

with torch.no_grad():
    pred_offset = model(gt_img)
pred_pose = isocenter_pose.compose(pred_offset)
pred_img = drr(None, None, None, pose=pred_pose).to(torch.float32)
pred_img = transforms(pred_img)
for im in pred_img:
    plt.figure()
    plt.imshow(im.detach().cpu().permute(1, 2, 0), cmap='gray')
    plt.show()

print()


