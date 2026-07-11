import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
# import submitit
import torch
from diffdrr.detector import make_xrays
from diffdrr.drr import DRR
from ours.utils.drr_bone import DRR as DRR_Bone
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from matplotlib import pyplot as plt
from pyexpat import features
from torchvision.transforms.functional import resize
from tqdm import tqdm

from diffpose.calibration import RigidTransform, convert
from diffpose.deepfluoro import DeepFluoroDataset, Evaluator, Transforms
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from diffpose.registration import PoseRegressor
from diffpose.registration import SparseRegistration
from diffpose.deepfluoro import DeepFluoroDataset, Transforms, get_random_offset
import nibabel as nib

def save_nii(matrix, output_path):
    """保存 numpy 矩阵为 NIfTI，使用单位仿射"""
    affine = np.eye(4)  # identity affine
    nii = nib.Nifti1Image(matrix, affine)
    nib.save(nii, output_path)
    print(f"Saved {output_path}")

device = torch.device("cuda:1")
id_num = 6
specimen = DeepFluoroDataset(id_num, filename = "/home/zsr/project/diffpose/data/ipcai_2020_full_res_data.h5")
height = 256
subsample = (1536 - 100) / height
delx = 0.194 * subsample

# drr = DRR(
#     specimen.volume,
#     specimen.spacing,
#     sdr=specimen.focal_len / 2,
#     height=height,
#     delx=delx,
#     x0=specimen.x0,
#     y0=specimen.y0,
#     reverse_x_axis=True,
#     bone_attenuation_multiplier=2.5,
# ).to(device)
#
# drr_bone = DRR_Bone(
#     specimen.volume,
#     specimen.spacing,
#     specimen.focal_len / 2,
#     height,
#     delx,
#     x0=specimen.x0,
#     y0=specimen.y0,
#     reverse_x_axis=True,
#     patch_size=height // 2,
#     bone_attenuation_multiplier=2.5
# ).to(device)
transforms = Transforms(height)
f_path = "/home/zsr/project/diffpose/ours/bone_seg/deep_nii"
for i in range(len(specimen)):
    img, pose = specimen[i]
    img = transforms(img)
    img = img.squeeze().cpu().numpy()
    fname = os.path.join(f_path, f"{id_num}_{i}.nii.gz")
    save_nii(img, fname)
    # plt.figure()
    # plt.imshow(img, cmap='gray')
    # plt.show()

