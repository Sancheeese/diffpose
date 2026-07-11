import os
import numpy as np
import pydicom
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from diffpose.calibration import RigidTransform
from torchvision.transforms.functional import center_crop, gaussian_blur
from diffpose.deepfluoro import DeepFluoroDataset
from torchvision.transforms import Compose, Lambda, Normalize, Resize, GaussianBlur, RandomErasing
from torchvision.transforms.v2 import ElasticTransform
from ours.utils.drr import DRR


class IntubationDataset(Dataset):
    def __init__(self, root, x_root, preprocess=True, x_offset=0, y_offset=0, z_offset=0, z_cut=0):
        self.preprocess = preprocess
        self.root = root
        file_name = os.listdir(self.root)
        file_name.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        self.file_name = file_name

        self.x_root = x_root
        x_file = os.listdir(x_root)
        x_file.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        self.x_file = x_file
        self.x_filename = os.path.join(x_root, x_file[0])

        (self.volume, self.spacing, self.sdr, self.delx) = self.getInfo()

        if z_cut > 0:
            self.volume = self.volume[:, :, :z_cut]

        isocenter_xyz = [self.volume.shape[0] - x_offset, self.volume.shape[1] - y_offset, self.volume.shape[2] - z_offset] \
                         * self.spacing / 2
        isocenter_xyz = torch.tensor(isocenter_xyz).unsqueeze(0)
        isocenter_rot = torch.tensor([[0.0, 0.0, torch.pi / 2 + torch.pi / 8]]).unsqueeze(0)
        self.isocenter_pose = RigidTransform(
            isocenter_rot, isocenter_xyz, "euler_angles", "ZYX"
        )

        center_xyz = [self.volume.shape[0] - x_offset, self.volume.shape[1] - y_offset, self.volume.shape[2] - z_offset] \
                         * self.spacing / 2
        center_xyz = torch.tensor(center_xyz).unsqueeze(0)
        center_rot = torch.tensor([[0.0, 0.0, 0.0]]).unsqueeze(0)
        center_pose = RigidTransform(
            center_rot, center_xyz, "euler_angles", "ZYX"
        )
        self.center_pose = center_pose

        back_xyz = [-self.volume.shape[0] + x_offset, -self.volume.shape[1] + y_offset, -self.volume.shape[2] + z_offset] \
                         * self.spacing / 2
        back_xyz = torch.tensor(back_xyz).unsqueeze(0)
        back_rot = torch.tensor([[0.0, 0.0, 0.0]]).unsqueeze(0)
        back_pose = RigidTransform(
            back_rot, back_xyz, "euler_angles", "ZYX"
        )
        self.back_pose = back_pose

    def __len__(self):
        return len(self.x_file)

    def __getitem__(self, idx):
        if idx == 0:
            device = torch.device("cuda:0")
            delx = self.delx * 2
            drr = DRR(
                self.volume,
                self.spacing,
                self.sdr,
                256,
                delx=delx,
                reverse_x_axis=True
            ).to(device)
            self.isocenter_pose = self.isocenter_pose.to(device)
            img = drr(None, None, None, pose=self.isocenter_pose, bone_attenuation_multiplier=3)
            return img, self.isocenter_pose

        img = pydicom.dcmread(os.path.join(self.x_root, self.x_file[idx])).pixel_array
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        pose = RigidTransform(
            torch.eye(3),
            torch.tensor([0.0, 0.0, 0.0]),)

        if self.preprocess:
            preprocess(img)

        return img, pose

    def getInfo(self):
        volume = None
        for f_name in self.file_name:
            file_path = os.path.join(self.root, f_name)
            volume_img = pydicom.dcmread(file_path).pixel_array
            volume_img = np.expand_dims(volume_img.astype(np.float32), axis=0)
            if volume is None:
                volume = volume_img
            else:
                volume = np.concatenate((volume, volume_img), axis=0)
        dcm_file = pydicom.dcmread(os.path.join(self.root, self.file_name[0]))
        rescale_slope = dcm_file.get("RescaleSlope")
        rescale_intercept = dcm_file.get("RescaleIntercept")
        volume = volume * rescale_slope + rescale_intercept

        dcm_file = pydicom.dcmread(os.path.join(self.root, self.file_name[0]))
        pixel_spacing = dcm_file.get("PixelSpacing")
        slice_thickness = dcm_file.get("SliceThickness")
        pixel_spacing = np.array(pixel_spacing)
        pixel_spacing = np.append(pixel_spacing, slice_thickness)
        pixel_spacing[0], pixel_spacing[2] = pixel_spacing[2], pixel_spacing[0]
        volume = np.swapaxes(volume, 1, 2).copy()

        x_ray = pydicom.dcmread(self.x_filename)
        sdr = x_ray.get("DistanceSourceToDetector") / 2
        intensifier_size = x_ray.get("IntensifierSize")
        delx = intensifier_size / 512

        return volume, pixel_spacing, sdr, delx

def preprocess(img, size=None, initial_energy=torch.tensor(65487.0)):
    """
    Recover the line integral: $L[i,j] = \log I_0 - \log I_f[i,j]$

    (1) Remove edge due to collimator
    (2) Smooth the image to make less noisy
    (3) Subtract the log initial energy for each ray
    (4) Recover the line integral image
    (5) Rescale image to [0, 1]
    """
    img = center_crop(img, (500, 500))
    img = gaussian_blur(img, (5, 5), sigma=1.0)
    img = initial_energy.log() - img.log()
    img = (img - img.min()) / (img.max() - img.min())
    return img

class Transforms:
    def __init__(
        self,
        size: int,  # Dimension to resize image
        eps: float = 1e-6,
    ):
        """Transform X-rays and DRRs before inputting to CNN."""
        self.eps = eps
        self.transforms = Compose(
            [
                # Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + eps)),
                Resize((size, size), antialias=True),
                Normalize(mean=0.3080, std=0.1494),
            ]
        )

        y_coord = torch.arange(size) - size // 2
        x_coord = torch.arange(size) - size // 2
        Y, X = torch.meshgrid(y_coord, x_coord, indexing='ij')
        distance_sq = X ** 2 + Y ** 2  # 使用平方避免开根号
        mask = (distance_sq <= 121 ** 2).float()
        self.mask = mask.unsqueeze(0).unsqueeze(0)

    def __call__(self, x, reverse=True):
        x = (x - x.min()) / (x.max() - x.min() + self.eps)
        if reverse:
            x = 1 - x
        x = self.transforms(x)

        # 计算每个样本的最小值 (B,1,1,1)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        min_values = x.view(x.size(0), -1).min(dim=1)[0][:, None, None, None]

        # 应用蒙版：圆形区域保持原值，外围设为该样本最小值
        mask = self.mask.to(x.device)
        x = x * mask + min_values * (1 - mask)
        # x[x < -1] = min_values

        return x


if __name__ == "__main__":
    root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/史建基/CT/ShiJianJi/20231019091100.063/603"
    x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/史建基/ERCP/SHI^JIANJI^/20231023150900/1"
    dataset = IntubationDataset(root=root, x_root=x_root)

    # specimen = DeepFluoroDataset(1, filename="/media/sda1/PersonalFiles/yx/project/diffpose/data/ipcai_2020_full_res_data.h5")
    # img1, pose1 = specimen[1]
    # img2, pose2 = specimen[2]
    # print(pose1)


