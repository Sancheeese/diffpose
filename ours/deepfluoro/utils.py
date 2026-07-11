import nibabel as nib
import os

import numpy as np
import torch
from matplotlib import pyplot as plt


def get_bone(id_num, idx):
    root = "/home/zsr/project/diffpose/ours/bone_seg/deep_result"
    # root = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/bone_seg/deep_result"
    fname = os.path.join(root, f"{id_num}_{idx}.nii.gz")
    nii = nib.load(fname)
    data = nii.get_fdata()
    data = np.squeeze(data)
    plt.figure(figsize=(5, 5))
    plt.imshow(data, cmap="gray")
    plt.axis("off")
    plt.show()
    return torch.tensor(data).unsqueeze(0).unsqueeze(0)

