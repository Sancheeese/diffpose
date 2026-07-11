import json
import os
import sys

from diffpose.calibration import perspective_projection, convert

script_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_path)
import nibabel as nib
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from diffpose.calibration import RigidTransform
from scipy.ndimage import zoom
from torch.nn.functional import pad
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize
from torchvision.transforms.functional import center_crop, gaussian_blur

from ours.utils.drr import DRR


class IntubationDataset(Dataset):
    def __init__(
        self,
        nii_path,
        x_root,
        preprocess=True,
        x_offset=0,
        y_offset=0,
        z_offset=0,
        z_cut=0,
        z_cut_end=-1,
        factors=[1, 1, 1],
    ):
        self.preprocess = preprocess
        self.nii_path = str(nii_path)
        self.root = self.nii_path
        self.gt_pose_dir = "/home/zsr/project/diffpose/ours/gt_pose/wfl"

        self.x_root = str(x_root)
        x_file = os.listdir(self.x_root)
        x_file.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
        self.x_file = x_file
        self.x_filename = os.path.join(self.x_root, x_file[0])

        self.z_cut = z_cut
        self.z_cut_end = z_cut_end
        (self.volume, self.spacing, self.sdr, self.delx, self.focal_len, self.lps2volume, self.intrinsic) = self.getInfo()
        self.fiducials = self.get_fiducials()

        if z_cut > 0 or z_cut_end != -1:
            self.volume = self.volume[:, :, z_cut:z_cut_end]

        isocenter_xyz = [self.volume.shape[0] - x_offset, self.volume.shape[1] - y_offset, self.volume.shape[2] - z_offset] \
                         * self.spacing / 2
        isocenter_xyz = torch.tensor(isocenter_xyz).unsqueeze(0)
        isocenter_rot = torch.tensor([[-torch.pi / 2, 0.0, torch.pi / 2]]).unsqueeze(0)
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

        self.volume = zoom(self.volume, factors, order=1)
        self.spacing = self.spacing / factors
        self.volume = np.flip(self.volume, axis=0).copy()

        self.flip_xz = RigidTransform(
            torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.zeros(3),
        )
        self.translate = RigidTransform(
            torch.eye(3),
            torch.tensor([self.focal_len / 2, 0.0, 0.0]),
        )
        self.flip_180 = RigidTransform(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
            torch.zeros(3),
        )

    def __len__(self):
        return len(self.x_file)

    def __getitem__(self, idx):
        if idx == -1:
            device = torch.device("cuda:1")
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
            self.center_pose = self.center_pose.to(device)
            self.back_pose = self.back_pose.to(device)

            offset = get_random_offset(1, device=device)
            pose = self.isocenter_pose.compose(self.back_pose).compose(offset).compose(self.center_pose)

            img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=3)
            return img, pose

        img = pydicom.dcmread(os.path.join(self.x_root, self.x_file[idx])).pixel_array
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        if self.preprocess:
            preprocess(img)

        return img, self.get_manual_gt(idx)

    def getInfo(self):
        nii = nib.load(self.nii_path)
        volume = np.asanyarray(nii.dataobj).astype(np.float32)
        # Match the existing WFL DICOM pipeline:
        # DICOM stacks slices then swapaxes(0, 2), yielding (x, y, z).
        # The converted NIfTI already has that axis order, but its slice axis
        # is reversed relative to the DICOM filename ordering.
        volume = np.flip(volume, axis=2).copy()

        zooms = nii.header.get_zooms()[:3]
        pixel_spacing = np.array([zooms[0], zooms[1], zooms[2]], dtype=np.float64)

        x_ray = pydicom.dcmread(self.x_filename)
        sdr = x_ray.get("DistanceSourceToDetector") / 2
        focal_len = x_ray.get("DistanceSourceToDetector")
        intensifier_size = x_ray.get("IntensifierSize")
        delx = intensifier_size / 512

        ras = nii.affine @ np.array([0.0, 0.0, nii.shape[2] - 1.0, 1.0])
        lps = np.array([-ras[0], -ras[1], ras[2]], dtype=np.float64)
        origin = torch.tensor(lps, dtype=torch.float32)
        origin[2] = -origin[2]
        origin = -origin
        origin[2] -= self.z_cut * pixel_spacing[2]
        lps2volume = RigidTransform(torch.eye(3), origin)

        intrinsic = torch.tensor([[focal_len, 0, 0], [0, focal_len, 0], [0, 0, 1]])

        return volume, pixel_spacing, sdr, delx, focal_len, lps2volume, intrinsic

    def get_x_filename(self, idx):
        return self.x_file[idx]

    def get_manual_gt(self, idx=None):
        idx_str = f"{idx:04d}"
        pose_file = os.path.join(self.gt_pose_dir, f"pose_{idx_str}.json")

        if not os.path.exists(pose_file):
            raise FileNotFoundError(f"Pose file not found: {pose_file}")

        with open(pose_file, "r") as f:
            pose_data = json.load(f)

        pose_params = pose_data["pose_params"]
        rot = torch.tensor([pose_params[:3]], dtype=torch.float32)
        xyz = torch.tensor([pose_params[3:]], dtype=torch.float32)

        return RigidTransform(rot, xyz, parameterization="so3_log_map")

    def get_fiducials(self):
        fiducials = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "fid_wfl.json")
        with open(file_path) as f:
            data = json.load(f)
            for point in data["markups"][0]["controlPoints"]:
                p = torch.tensor(point["position"]).unsqueeze(0)
                p[..., 2] = -p[..., 2]
                if fiducials is None:
                    fiducials = p
                else:
                    fiducials = torch.concat((fiducials, p), dim=0)

        fiducials = fiducials.unsqueeze(0)
        fiducials = self.lps2volume.transform_points(fiducials)
        return fiducials

    def get_2d_fiducials(self, idx, pose):
        true_pose = self.get_manual_gt(idx)

        total_pose = self.flip_xz.compose(self.translate).compose(true_pose)
        extrinsic = total_pose.inverse()
        true_fiducials = perspective_projection(
            extrinsic, self.intrinsic, self.fiducials
        )
        true_fiducials[..., 0] = -true_fiducials[..., 0]
        true_fiducials += 152.5

        total_pose = self.flip_xz.compose(self.translate).compose(pose.cpu())
        extrinsic = total_pose.inverse()
        pred_fiducials = perspective_projection(
            extrinsic, self.intrinsic, self.fiducials
        )
        pred_fiducials[..., 0] = -pred_fiducials[..., 0]
        pred_fiducials += 152.5

        return true_fiducials, pred_fiducials

    def get_3d_fiducials(self, idx, pose):
        true_pose = self.get_manual_gt(idx)

        total_pose = self.flip_xz.compose(self.translate).compose(true_pose)
        extrinsic = total_pose.inverse()
        true_fiducials = perspective_projection(
            extrinsic, self.intrinsic, self.fiducials
        )
        true_fiducials = self.focal_len * torch.einsum(
            "ij, bnj -> bni",
            self.intrinsic.inverse(),
            pad(true_fiducials, (0, 1), value=1),
        )
        true_fiducials = total_pose.transform_points(true_fiducials)

        pred_pose = self.flip_xz.compose(self.translate).compose(pose.cpu())
        extrinsic_pred = pred_pose.inverse()
        pred_fiducials = perspective_projection(
            extrinsic_pred, self.intrinsic, self.fiducials
        )
        pred_fiducials = self.focal_len * torch.einsum(
            "ij, bnj -> bni",
            self.intrinsic.inverse(),
            pad(pred_fiducials, (0, 1), value=1),
        )
        pred_fiducials = pred_pose.transform_points(pred_fiducials)

        return true_fiducials, pred_fiducials

    def get_bone(self, idx):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(current_dir, "..", "..", "bone_seg", "wfl_result2_100_reverse")
        nii_files = [f for f in os.listdir(folder_path) if f.endswith(".nii.gz")]
        nii_files.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))

        fname = nii_files[idx]
        nii_path = os.path.join(folder_path, fname)
        nii = nib.load(nii_path)
        data = nii.get_fdata()
        data = np.squeeze(data)

        img = torch.tensor(data).unsqueeze(0).unsqueeze(0)
        img = F.interpolate(img, [256, 256], mode="bilinear")

        return img

    def calc_tre(self, idx, pose):
        true_pose = self.get_manual_gt(idx)

        total_pose = self.flip_xz.compose(self.translate).compose(true_pose)
        extrinsic = total_pose.inverse()
        true_fiducials = extrinsic.transform_points(self.fiducials)

        pred_pose = self.flip_xz.compose(self.translate).compose(pose.cpu())
        extrinsic_pred = pred_pose.inverse()
        pred_fiducials = extrinsic_pred.cpu().transform_points(self.fiducials)

        tre = torch.norm(true_fiducials - pred_fiducials, dim=2)
        tre = torch.mean(tre)

        return tre


def preprocess(img, size=None, initial_energy=torch.tensor(65487.0)):
    img = center_crop(img, (500, 500))
    img = gaussian_blur(img, (5, 5), sigma=1.0)
    img = initial_energy.log() - img.log()
    img = (img - img.min()) / (img.max() - img.min())
    return img


def get_random_offset(batch_size: int, device) -> RigidTransform:
    r1 = torch.distributions.Normal(0, torch.pi / 7).sample((batch_size,))
    r2 = torch.distributions.Normal(0, torch.pi / 14).sample((batch_size,))
    r3 = torch.distributions.Normal(0, torch.pi / 12).sample((batch_size,))
    t1 = torch.distributions.Normal(0, 70).sample((batch_size,))
    t2 = torch.distributions.Normal(0, 30).sample((batch_size,))
    t3 = torch.distributions.Normal(0, 30).sample((batch_size,))
    log_R_vee = torch.stack([r1, r2, r3], dim=1).to(device)
    log_t_vee = torch.stack([t1, t2, t3], dim=1).to(device)

    isocenter_pose = RigidTransform(
        log_R_vee, log_t_vee, "euler_angles", "ZYX"
    )
    isocenter_pose = isocenter_pose.to(device)

    return isocenter_pose


def toZeroOne(x):
    return (x - x.min()) / (x.max() - x.min())


class Transforms:
    def __init__(
        self,
        size: int,
        eps: float = 1e-6,
    ):
        self.eps = eps
        self.transforms = Compose(
            [
                Normalize(mean=0.3080, std=0.1494),
            ]
        )
        self.resize = Resize((size, size), antialias=True)

        y_coord = torch.arange(size) - size // 2
        x_coord = torch.arange(size) - size // 2
        Y, X = torch.meshgrid(y_coord, x_coord, indexing="ij")
        distance_sq = X ** 2 + Y ** 2
        mask = (distance_sq <= 121 ** 2).float()
        self.mask = mask.unsqueeze(0).unsqueeze(0)

    def __call__(self, x, reverse=True):
        x = self.resize(x)
        x = (x - x.min()) / (x.max() - x.min() + self.eps)
        if reverse:
            x = 1 - x

        mask = self.mask.to(x.device)
        x = x * mask
        x = self.transforms(x)

        return x


def create_circle_mask(size, radius):
    y_coord = torch.arange(size) - size // 2
    x_coord = torch.arange(size) - size // 2
    Y, X = torch.meshgrid(y_coord, x_coord, indexing="ij")
    distance_sq = X ** 2 + Y ** 2
    mask = (distance_sq <= radius ** 2).float()

    return mask


def create_circle_mask_reverse(size, radius):
    return 1 - create_circle_mask(size, radius)


if __name__ == "__main__":
    nii_path = "/home/zsr/project/mrct/data/王凤兰/CT/306.nii"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/王凤兰/ERCP/FENGLAN^WANG^/20240313160330/1"

    specimen = IntubationDataset(
        nii_path,
        x_root,
        x_offset=20,
        z_offset=50,
        z_cut=30,
        z_cut_end=250,
        factors=[0.5, 0.5, 1],
    )
    pose = specimen.isocenter_pose
    matrix = pose.get_matrix()
    fiducials = specimen.fiducials
