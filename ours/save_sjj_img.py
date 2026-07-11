import time
import h5py
import numpy as np
import torch
from diffdrr.drr import DRR
from matplotlib import pyplot as plt

from diffpose.calibration import RigidTransform, convert
from dataset.CT_dataset import IntubationDataset

def get_random_offset(batch_size: int, device):
    r1 = torch.distributions.Normal(0, 0.25).sample((batch_size,))
    r2 = torch.distributions.Normal(0, 0.1).sample((batch_size,))
    r3 = torch.distributions.Normal(0, 0.2).sample((batch_size,))
    t1 = torch.distributions.Normal(0, 40).sample((batch_size,))
    t2 = torch.distributions.Normal(0, 20).sample((batch_size,))
    t3 = torch.distributions.Normal(0, 5).sample((batch_size,))
    # t3 = -torch.abs(torch.distributions.Normal(0, 10).sample((batch_size,)))
    log_R_vee = torch.stack([r1, r2, r3], dim=1).to(device)
    log_t_vee = torch.stack([t1, t2, t3], dim=1).to(device)
    return log_R_vee, log_t_vee, convert(
        [log_R_vee, log_t_vee],
        "se3_log_map",
        "se3_exp_map",
    )

epoch = 200
n_batch = 400
batch_size = 2
device = torch.device("cuda:0")
contrast_distribution = torch.distributions.Uniform(2.0, 9.0)
root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/史建基/CT/ShiJianJi/20231019091100.063/603"
x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/史建基/ERCP/SHI^JIANJI^/20231023150900/1"
specimen = IntubationDataset(root, x_root)
isocenter_pose = specimen.isocenter_pose.to(device)
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
).to(device)


with h5py.File('sjj_data.h5', 'w') as f:
    images_group = f.create_group('images')
    R_group = f.create_group('R')
    t_group = f.create_group('t')
    for i in range(epoch):
        for j in range(n_batch):
            print(f"saving===>epoch:{i} and n_batch:{j}")
            contrast = contrast_distribution.sample().item()
            R, t, offset = get_random_offset(batch_size, device)
            pose = isocenter_pose.compose(offset)
            img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=contrast)
            # for im in img:
            #     plt.figure()
            #     plt.imshow(im.cpu().squeeze(), cmap="gray")
            #     plt.show()

            images_group.create_dataset(f"img_{i}_{j}", data=img.cpu().numpy())
            R_group.create_dataset(f"R_{i}_{j}", data=R.cpu().numpy())
            t_group.create_dataset(f"t_{i}_{j}", data=t.cpu().numpy())



