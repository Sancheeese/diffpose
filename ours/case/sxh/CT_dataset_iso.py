import json
import os
import sys

from diffpose.calibration import perspective_projection, convert

script_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_path)
import numpy as np
import pydicom
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from diffpose.calibration import RigidTransform
from torchvision.transforms.functional import center_crop, gaussian_blur
from diffpose.deepfluoro import DeepFluoroDataset
from torchvision.transforms import Compose, Lambda, Normalize, Resize
from ours.utils.drr import DRR
# from drr_bone import DRR
# from drr_bone import DRR as DRR_Bone
from scipy.ndimage import zoom
import nibabel as nib
import torch.nn.functional as F
from torch.nn.functional import pad

class IntubationDataset(Dataset):
    def __init__(self, root, x_root, preprocess=True, x_offset=0, y_offset=0, z_offset=0, z_cut=0, factors=[1,1,1]):
        self.preprocess = preprocess
        self.root = root
        if root.startswith("/home/zsr/project/diffpose/ours/data/liwei/"):
            self.gt_pose_dir = "/home/zsr/project/diffpose/ours/gt_pose/sxh"
        elif root.startswith("/media/sda1/Data/ERCP/CT+X+MRCP/liwei/"):
            self.gt_pose_dir = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/gt_pose/sxh"
        file_name = os.listdir(self.root)
        file_name.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        self.file_name = file_name

        self.x_root = x_root
        x_file = os.listdir(x_root)
        x_file.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        self.x_file = x_file
        self.x_filename = os.path.join(x_root, x_file[0])

        (self.volume, self.spacing, self.sdr, self.delx, self.focal_len, self.lps2volume, self.intrinsic) = self.getInfo()
        self.fiducials = self.get_fiducials()

        if z_cut > 0:
            self.volume = self.volume[:, :, :z_cut]

        self.isocenter_pose = self.get_iso_pose()

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

        # self.spacing[1] *= 0.9

        # Miscellaneous transformation matrices for wrangling SE(3) poses
        self.flip_xz = RigidTransform(
            torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.zeros(3),
        )
        # self.flip_xz = RigidTransform(
        #     torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        #     torch.zeros(3),
        # )
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

            pose_xyz = [0.0, -50, 0.0]
            pose_xyz = torch.tensor(pose_xyz).unsqueeze(0)
            pose_rot = torch.tensor([[torch.pi / 4, 3 * torch.pi / 40, 0.0]]).unsqueeze(0)
            pose = RigidTransform(
                pose_rot, pose_xyz, "euler_angles", "ZYX"
            )
            pose = pose.to(device)

            offset = get_random_offset(1, device=device)
            pose = self.isocenter_pose.compose(self.back_pose).compose(offset).compose(self.center_pose)

            img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=3)
            return img, pose

        img = pydicom.dcmread(os.path.join(self.x_root, self.x_file[idx])).pixel_array
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        pose = RigidTransform(
            torch.eye(3),
            torch.tensor([0.0, 0.0, 0.0]),)

        if self.preprocess:
            preprocess(img)

        return img, self.get_manual_gt(idx)

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
        volume = np.swapaxes(volume, 0, 2).copy()
        # volume = np.swapaxes(volume, 0, 1).copy()

        x_ray = pydicom.dcmread(self.x_filename)
        sdr = x_ray.get("DistanceSourceToDetector") / 2
        focal_len = x_ray.get("DistanceSourceToDetector")
        intensifier_size = x_ray.get("IntensifierSize")
        delx = intensifier_size / 512

        origin = torch.tensor(dcm_file.get("ImagePositionPatient"))
        origin[2] = -origin[2]
        origin = -origin
        lps2volume = RigidTransform(torch.eye(3), origin)

        len = focal_len
        intrinsic = torch.tensor([[len, 0, 0], [0, len, 0], [0, 0, 1]])
        # intrinsic = torch.tensor([[len, 0, 152.5], [0, len, 152.5], [0, 0, 1]])

        return volume, pixel_spacing, sdr, delx, focal_len, lps2volume, intrinsic

    def get_x_filename(self, idx):
        return self.x_file[idx]

    def get_manual_gt(self, idx=None):
        if idx is None:
            idx = 30
        idx_str = f"{idx:04d}"

        # 构建位姿文件路径
        pose_file = os.path.join(self.gt_pose_dir, f"pose_{idx_str}.json")

        # 检查文件是否存在
        if not os.path.exists(pose_file):
            raise FileNotFoundError(f"Pose file not found: {pose_file}")

        # 读取JSON文件
        with open(pose_file, 'r') as f:
            pose_data = json.load(f)

        # 提取pose_params
        pose_params = pose_data["pose_params"]

        # 计算位姿
        rot = torch.tensor([pose_params[:3]], dtype=torch.float32)
        xyz = torch.tensor([pose_params[3:]], dtype=torch.float32)

        # 创建RigidTransform对象
        pose = RigidTransform(rot, xyz, parameterization="so3_log_map")

        return pose

    def get_iso_pose(self):
        # 构建位姿文件路径
        pose_file = os.path.join(self.gt_pose_dir, f"sxh.json")

        # 检查文件是否存在
        if not os.path.exists(pose_file):
            raise FileNotFoundError(f"Pose file not found: {pose_file}")

        # 读取JSON文件
        with open(pose_file, 'r') as f:
            pose_data = json.load(f)

        # 提取pose_params
        pose_params = pose_data["pose_params"]

        # 计算位姿
        rot = torch.tensor([pose_params[:3]], dtype=torch.float32)
        xyz = torch.tensor([pose_params[3:]], dtype=torch.float32)

        # 创建RigidTransform对象
        pose = RigidTransform(rot, xyz, parameterization="so3_log_map")

        return pose

    def get_fiducials(self):
        fiducials = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 计算相对路径
        file_path = os.path.join(current_dir, "fid_sxh.json")
        # with open("/home/zsr/project/diffpose/ours/case/sxh/fid_sxh.json") as f:
        # with open("/media/sda1/PersonalFiles/yx/project/diffpose/ours/case/sxh/fid_sxh.json") as f:
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
        # Get the fiducials from the true camera pose
        true_pose = self.get_manual_gt(idx)
        # extrinsic = (
        #     self.lps2volume.inverse()
        #     .compose(true_pose.inverse())
        #     .compose(self.translate)
        #     .compose(self.flip_xz)
        # )

        # total_pose = self.flip_xz.compose(self.translate).compose(true_pose)
        # test_pose = total_pose.inverse()
        # zero_point = torch.tensor([[.0, .0, .0]])
        # p1 = torch.tensor([[1.0, 1.0, 1.0]])
        # p2 = torch.tensor([[1.0, .0, 1.0]])
        # t = total_pose.transform_points(zero_point)
        # t1 = test_pose.transform_points(torch.tensor([[593.1561, -382.4844,   95.1508]]))
        # after = test_pose.transform_points(zero_point)
        # a1 = test_pose.transform_points(p1)
        # a2 = test_pose.transform_points(p2)

        total_pose = self.flip_xz.compose(self.translate).compose(true_pose)
        extrinsic = total_pose.inverse()
        true_fiducials = perspective_projection(
            extrinsic, self.intrinsic, self.fiducials
        )
        temp = torch.tensor(true_fiducials[..., 0])
        # true_fiducials[..., 0] = true_fiducials[..., 1]
        true_fiducials[..., 0] = -temp
        true_fiducials += 152.5


        total_pose = self.flip_xz.compose(self.translate).compose(pose.cpu())
        extrinsic = total_pose.inverse()
        pred_fiducials = perspective_projection(
            extrinsic, self.intrinsic, self.fiducials
        )
        temp = torch.tensor(pred_fiducials[..., 0])
        # pred_fiducials[..., 0] = pred_fiducials[..., 1]
        pred_fiducials[..., 0] = -temp
        pred_fiducials += 152.5

        return true_fiducials, pred_fiducials

    def get_3d_fiducials(self, idx, pose):
        # Get the fiducials from the true camera pose
        true_pose = self.get_manual_gt(idx)

        # total_pose = self.flip_xz.compose(self.translate)
        total_pose = self.flip_xz.compose(self.translate).compose(true_pose)
        extrinsic = total_pose.inverse()
        true_fiducials = perspective_projection(
            extrinsic, self.intrinsic, self.fiducials
        )
        true_fiducials = self.focal_len * torch.einsum(
            "ij, bnj -> bni",
            self.intrinsic.inverse(),
            pad(true_fiducials, (0, 1), value=1),  # Convert to homogenous coordinates
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
            pad(pred_fiducials, (0, 1), value=1),  # Convert to homogenous coordinates
        )
        pred_fiducials = pred_pose.transform_points(pred_fiducials)

        # a = true_pose.transform_points(self.fiducials)
        # pose = pose.to('cpu')
        # b = pose.transform_points(self.fiducials)

        return true_fiducials, pred_fiducials
        # return a, b

    def get_bone(self, idx):
        # folder_path = "/home/zsr/project/diffpose/ours/bone_seg/wfl_result"
        # folder_path = "/media/sda1/PersonalFiles/yx/project/diffpose/ours/bone_seg/xyl_result2_100_reverse"
        # 获取当前文件的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 计算相对路径
        folder_path = os.path.join(current_dir, "..", "..", "bone_seg", "sxh_result2_100_reverse")
        nii_files = [f for f in os.listdir(folder_path) if f.endswith(".nii.gz")]
        nii_files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

        fname = nii_files[idx]
        nii_path = os.path.join(folder_path, fname)
        nii = nib.load(nii_path)
        data = nii.get_fdata()
        data = np.squeeze(data)

        plt.figure(figsize=(5, 5))
        plt.imshow(data, cmap="gray")
        plt.axis("off")
        plt.show()

        img = torch.tensor(data).unsqueeze(0).unsqueeze(0)
        img = F.interpolate(img, [256, 256], mode='bilinear')

        return img

    def calc_tre(self, idx, pose):
        # Get the fiducials from the true camera pose
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


def simple_projection(points_3d, source_position, detector_center):
    """
    简化的透视投影计算
    points_3d: [N, 3] 3D点坐标
    source_position: [3] X射线源位置
    detector_center: [3] 探测器中心位置

    假设探测器平面垂直于光源-探测器中心连线
    """
    # 计算探测器平面法向量（从探测器中心指向光源）
    detector_normal = source_position - detector_center
    detector_normal = detector_normal / torch.norm(detector_normal)  # 单位化

    # 计算探测器平面方程: n·(p - p0) = 0
    p0 = detector_center

    # 对每个3D点进行投影
    projected_points = []

    for point in points_3d:
        # 射线方向: 从光源到3D点
        ray_dir = point - source_position

        # 计算射线与探测器平面的交点参数t
        # 平面方程: n·(source + t*ray_dir - p0) = 0
        # 解得: t = n·(p0 - source) / n·ray_dir
        numerator = torch.dot(detector_normal, p0 - source_position)
        denominator = torch.dot(detector_normal, ray_dir)

        if abs(denominator) > 1e-8:  # 避免除零
            t = numerator / denominator
            intersection = source_position + t * ray_dir
            projected_points.append(intersection)
        else:
            # 射线与平面平行，无交点
            projected_points.append(torch.full((3,), float('nan')))

    return torch.stack(projected_points)


def project_points(landmarks_3d, transforms, K):
    """
    将3D点投影到2D图像平面（含透视除法）

    参数:
    landmarks_3d (torch.Tensor): 3D点坐标, 形状 (n, 3)
    transforms (torch.Tensor): 外参矩阵, 形状 (b, 4, 4)
    K (torch.Tensor): 内参矩阵, 形状 (3, 3)

    返回:
    torch.Tensor: 投影后的2D坐标, 形状 (b, n, 2)
    """
    n = landmarks_3d.shape[0]
    device = landmarks_3d.device

    # 1. 转换为齐次坐标 (n, 3) -> (n, 4)
    ones = torch.ones(n, 1, device=device)
    points_homo = torch.cat([landmarks_3d, ones], dim=1)  # (n, 4)

    # 2. 变换到相机坐标系 (b, 4, 4) @ (n, 4, 1) -> (b, n, 4, 1)
    points_cam = torch.matmul(transforms, points_homo.T.unsqueeze(0))  # (b, 4, n)
    points_cam = points_cam.permute(0, 2, 1)  # (b, n, 4)

    # 3. 透视除法 (除以Z_c)
    points_cam_normalized = points_cam[:, :, :3] / points_cam[:, :, 3:4]  # (b, n, 3)

    # 4. 投影到2D (b, n, 3) = (b, n, 3) @ (3, 3)
    points_2d = torch.matmul(points_cam_normalized, K.T)  # (b, n, 3)

    # 5. 提取u, v坐标 (忽略最后一维的1)
    points_2d = points_2d[:, :, :2]  # (b, n, 2)

    return points_2d


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

def get_random_offset(batch_size: int, device) -> RigidTransform:
    r1 = torch.distributions.Normal(0, torch.pi / 7).sample((batch_size,))
    r2 = torch.distributions.Normal(0, torch.pi / 14).sample((batch_size,))
    r3 = torch.distributions.Normal(0, torch.pi / 12).sample((batch_size,))
    # 前后
    t1 = torch.distributions.Normal(0, 70).sample((batch_size,))
    # 左右
    t2 = torch.distributions.Normal(0, 30).sample((batch_size,))
    # 上下
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
        size: int,  # Dimension to resize image
        eps: float = 1e-6,
    ):
        """Transform X-rays and DRRs before inputting to CNN."""
        self.eps = eps
        self.transforms = Compose(
            [
                # Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + eps)),
                # Resize((size, size), antialias=True),
                Normalize(mean=0.3080, std=0.1494),
            ]
        )
        self.resize = Resize((size, size), antialias=True)

        y_coord = torch.arange(size) - size // 2
        x_coord = torch.arange(size) - size // 2
        Y, X = torch.meshgrid(y_coord, x_coord, indexing='ij')
        distance_sq = X ** 2 + Y ** 2  # 使用平方避免开根号
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

        # # 计算每个样本的最小值 (B,1,1,1)
        # if len(x.shape) == 3:
        #     x = x.unsqueeze(0)
        # min_values = x.view(x.size(0), -1).min(dim=1)[0][:, None, None, None]
        #
        # # 应用蒙版：圆形区域保持原值，外围设为该样本最小值
        # mask = self.mask.to(x.device)
        # x = x * mask + min_values * (1 - mask)
        # # x[x < -1] = min_values

        return x

def create_circle_mask(size, radius):
    y_coord = torch.arange(size) - size // 2
    x_coord = torch.arange(size) - size // 2
    Y, X = torch.meshgrid(y_coord, x_coord, indexing='ij')
    distance_sq = X ** 2 + Y ** 2  # 使用平方避免开根号
    mask = (distance_sq <= radius ** 2).float()

    return mask

def create_circle_mask_reverse(size, radius):
    return 1 - create_circle_mask(size, radius)

if __name__ == "__main__":
    root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"

    specimen = IntubationDataset(root, x_root, y_offset=155, z_cut=600, factors=[3, 0.5, 0.5])
    pose = specimen.isocenter_pose
    matrix = pose.get_matrix()
    fiducials = specimen.fiducials

