import h5py
import numpy as np
import torch
from diffdrr.drr import DRR
from matplotlib import pyplot as plt

from diffpose.calibration import RigidTransform
from diffpose.deepfluoro import get_random_offset

device = torch.device("cpu")

filename = "/media/sda1/PersonalFiles/yx/project/diffpose/data/ipcai_2020_full_res_data.h5"
f = h5py.File(filename, "r")
specimen = f["18-1109"]
projections = specimen["projections"]
volume = specimen["vol/pixels"][:].astype(np.float32)

# nii_ct = nib.Nifti1Image(volume, affine=np.eye(4))
# nib.save(nii_ct, "test2.nii")

volume = np.swapaxes(volume, 0, 2)[::-1].copy()
volume = volume
print(volume.max())
print(volume.min())

spacing = specimen["vol/spacing"][:].flatten()
height = 256
subsample = (1536 - 100) / height
delx = 0.194 * subsample

drr = DRR(
    volume,
    spacing,
    500,
    height,
    delx=delx,
    reverse_x_axis=True
).to(device)

isocenter_xyz = torch.tensor(volume.shape) * spacing / 2
isocenter_xyz[1] = isocenter_xyz[1]
isocenter_xyz = torch.tensor(isocenter_xyz).unsqueeze(0)

isocenter_rot = torch.tensor([[torch.pi / 2, 0.0, -torch.pi / 2]]).unsqueeze(0)
# isocenter_rot = torch.tensor([[0.0, 0.0, 0.0]]).unsqueeze(0)

isocenter_pose = RigidTransform(
    isocenter_rot, isocenter_xyz, "euler_angles", "ZYX"
).to(device)

contrast_distribution = torch.distributions.Uniform(1.0, 10.0)
contrast = contrast_distribution.sample().item()

offset = get_random_offset(4, device)
pose = isocenter_pose.compose(offset)
img = drr(None, None, None, pose=isocenter_pose, bone_attenuation_multiplier=4)

for im in img:
    plt.figure()
    plt.imshow(im.cpu().squeeze(), cmap="gray")
    plt.show()




