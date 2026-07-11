import json
import os
import sys
import time

import math

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
from drr import DRR
# from drr_bone import DRR
# from drr_bone import DRR as DRR_Bone
from scipy.ndimage import zoom


class IntubationDataset(Dataset):
    def __init__(self, root, x_root, preprocess=True, x_offset=0, y_offset=0, z_offset=0, z_cut=0, factors=[1,1,1]):
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

        (self.volume, self.spacing, self.sdr, self.delx, self.focal_len, self.lps2volume, self.intrinsic) = self.getInfo()
        self.fiducials = self.get_fiducials()

        if z_cut > 0:
            self.volume = self.volume[:, :, :z_cut]

        self.volume = zoom(self.volume, factors, order=1)
        self.spacing = self.spacing / factors

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

        # Miscellaneous transformation matrices for wrangling SE(3) poses
        self.flip_xz = RigidTransform(
            torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.zeros(3),
        )
        self.translate = RigidTransform(
            torch.eye(3),
            torch.tensor([-self.focal_len / 2, 0.0, 0.0]),
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
        focal_len = x_ray.get("DistanceSourceToDetector")
        intensifier_size = x_ray.get("IntensifierSize")
        delx = intensifier_size / 512

        origin = torch.tensor(dcm_file.get("ImagePositionPatient"))
        origin[2] = -origin[2]
        lps2volume = RigidTransform(torch.eye(3), origin)

        len = focal_len / delx
        intrinsic = torch.tensor([[len, 0, 256], [0, len, 256,], [0, 0, 1]])

        return volume, pixel_spacing, sdr, delx, focal_len, lps2volume, intrinsic

    def get_x_filename(self, idx):
        return self.x_file[idx]

    def get_manual_gt(self):
        rotation = torch.tensor([[[ 0.6849,  0.0513,  0.7268],
                                 [ 0.6849, -0.3858, -0.6182],
                                 [ 0.2487,  0.9212, -0.2993]]])
        translation = torch.tensor([[349.2028, 278.9122, 192.4318]])

        # rot = torch.tensor([[1.8550, -0.7532, -0.5649]])
        # trans = torch.tensor([[267.7361, 155.9216, -65.8517]])

        rot = torch.tensor([[1.8571, -0.7570, -0.5641]])
        trans = torch.tensor([[267.8960, 155.9548, -65.7282]])

        # return RigidTransform(rot, trans, parameterization="so3_log_map")
        return convert(
            [rot, trans],
            input_parameterization="se3_log_map",
            output_parameterization="se3_exp_map")

    def get_fiducials(self):
        fiducials = None
        f_name = "_".join(self.get_x_filename(0).split(".")[0].split("_")[:-1]) + ".json"
        with open(os.path.join("/media/sda1/PersonalFiles/yx/project/diffpose/ours/gt", f_name)) as f:
            data = json.load(f)
            for point in data["markups"][0]["controlPoints"]:
                p = torch.tensor(point["position"]).unsqueeze(0)
                if fiducials is None:
                    fiducials = p
                else:
                    fiducials = torch.concat((fiducials, p), dim=0)

        fiducials = fiducials.unsqueeze(0)
        return fiducials

    def get_2d_fiducials(self, idx, pose):
        # Get the fiducials from the true camera pose
        true_pose = self.get_manual_gt()
        extrinsic = (
            self.lps2volume.inverse()
            .compose(true_pose.inverse())
            .compose(self.translate)
            .compose(self.flip_xz)
        )
        true_fiducials = perspective_projection(
            extrinsic, self.intrinsic, self.fiducials
        )

        # Get the fiducials from the predicted camera pose
        extrinsic = (
            self.lps2volume.inverse()
            .compose(pose.cpu().inverse())
            .compose(self.translate)
            .compose(self.flip_xz)
        )
        pred_fiducials = perspective_projection(
            extrinsic, self.intrinsic, self.fiducials
        )

        return true_fiducials, pred_fiducials


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


import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, Resize, Lambda, Normalize
import numpy as np
import random
from typing import Optional, Tuple
import cv2


class RandomBrightnessContrast:
    """随机调整亮度和对比度"""

    def __init__(self, brightness_range: Tuple[float, float] = (0.7, 1.3),
                 contrast_range: Tuple[float, float] = (0.5, 2),
                 p: float = 0.8):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p  # 应用概率

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x

        # 生成随机因子
        contrast_factor = random.uniform(*self.contrast_range)

        # 应用对比度调整

        x = torch.pow(x, contrast_factor)
        # x = torch.clamp(x, 0, 1)

        # for im in x:
        #     plt.figure()
        #     plt.imshow(im.detach().cpu().permute(1, 2, 0), cmap='gray')
        #     plt.show()

        return x


class RandomCLAHE:
    """使用CLAHE进行自适应的直方图均衡化增强"""

    def __init__(self,
                 p: float = 0.5,
                 clip_limit_range: Tuple[float, float] = (1.0, 3.0),
                 tile_grid_sizes: list[int] = [4, 8, 16]):
        """
        参数:
            p: 应用增强的概率
            clip_limit_range: 对比度限制范围，值越大增强越强
            tile_grid_sizes: 网格大小列表，值越小局部性越强
        """
        self.p = p
        self.clip_limit_range = clip_limit_range
        self.tile_grid_sizes = tile_grid_sizes

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x

        # # 转换为numpy格式
        # x_np = x.squeeze().cpu().numpy()
        #
        # # 确保值在0-1范围内，然后转换到0-255
        # x_np = (x_np * 255).astype(np.uint8)
        #
        # # 随机选择CLAHE参数
        # clip_limit = random.uniform(*self.clip_limit_range)
        # grid_size = random.choice(self.tile_grid_sizes)
        #
        # # 创建CLAHE对象并应用
        # clahe = cv2.createCLAHE(
        #     clipLimit=clip_limit,
        #     tileGridSize=(grid_size, grid_size)
        # )
        # x_eq = clahe.apply(x_np)
        #
        # # 转换回PyTorch张量
        # x_eq = torch.from_numpy(x_eq.astype(np.float32) / 255.0)
        # x_eq = x_eq.unsqueeze(0) if x.dim() == 3 else x_eq
        #
        # return x_eq.to(x.device)

        device = x.device
        results = []
        for i in range(x.shape[0]):
            # 获取单张图像 [1, H, W]
            img = x[i]

            # 转换为numpy格式并移除通道维度 [H, W]
            img_np = img.squeeze().cpu().numpy()

            # 确保值在0-1范围内，然后转换到0-255
            if img_np.max() <= 1.0:
                img_np_uint8 = (img_np * 255).astype(np.uint8)
            else:
                img_np_uint8 = img_np.astype(np.uint8)

            # 随机选择CLAHE参数
            clip_limit = random.uniform(*self.clip_limit_range)
            grid_size = random.choice(self.tile_grid_sizes)

            # 创建CLAHE对象并应用
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=(grid_size, grid_size)
            )
            img_eq = clahe.apply(img_np_uint8)

            # 转换回PyTorch张量并恢复形状 [1, H, W]
            img_eq_tensor = torch.from_numpy(img_eq.astype(np.float32) / 255.0)
            img_eq_tensor = img_eq_tensor.unsqueeze(0)  # 添加通道维度

            results.append(img_eq_tensor)

            # 堆叠回批量 [B, 1, H, W]
        # x = torch.stack(results).to(device)
        # for im in x:
        #     plt.figure()
        #     plt.imshow(im.detach().cpu().permute(1, 2, 0), cmap='gray')
        #     plt.show()
        return torch.stack(results).to(device)

class RandomHistogramEqualization:
    """随机直方图均衡化"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x

        # 转换为numpy进行直方图均衡化
        x_np = x.squeeze().cpu().numpy() * 255
        x_np = x_np.astype(np.uint8)

        # 应用直方图均衡化
        x_eq = cv2.equalizeHist(x_np)

        # 转换回tensor
        x_eq = torch.from_numpy(x_eq.astype(np.float32) / 255.0)
        x_eq = x_eq.unsqueeze(0) if x.dim() == 3 else x_eq

        return x_eq.to(x.device)


class RandomArtifacts:
    """随机添加伪影"""

    def __init__(self,
                 noise_p: float = 0.4,
                 max_noise_level: float = 0.03,
                 blur_range: Tuple[float, float] = (0.5, 2.0)):
        self.noise_p = noise_p
        self.max_noise_level = max_noise_level
        self.blur_range = blur_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # 随机添加高斯噪声
        if random.random() < self.noise_p:
            noise_level = random.uniform(0, self.max_noise_level)
            noise = torch.randn_like(x) * noise_level
            x = x + noise
            x = torch.clamp(x, 0, 1)

        # # 随机添加模糊
        # if random.random() < self.blur_p:
        #     # 使用高斯模糊核模拟
        #     blur_sigma = random.uniform(*self.blur_range)
        #     # 创建高斯核
        #     kernel_size = 5
        #     ax = torch.arange(kernel_size) - kernel_size // 2
        #     xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        #     kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * blur_sigma ** 2))
        #     kernel = kernel / kernel.sum()
        #     kernel = kernel.view(1, 1, kernel_size, kernel_size).to(x.device)
        #
        #     # 应用卷积实现模糊
        #     if x.dim() == 3:
        #         x = x.unsqueeze(0)
        #     x = torch.nn.functional.conv2d(x, kernel, padding=kernel_size // 2)
        #     if x.shape[0] == 1:
        #         x = x.squeeze(0)

        # for im in x:
        #     plt.figure()
        #     plt.imshow(im.detach().cpu().permute(1, 2, 0), cmap='gray')
        #     plt.show()

        # for im in x:
        #     plt.figure()
        #     plt.imshow(im.detach().cpu().permute(1, 2, 0), cmap='gray')
        #     plt.show()

        return x


class RandomShadowArtifact:
    def __init__(self,
                 p: float = 0.5,
                 intensity_range: Tuple[float, float] = (0.8, 0.9),
                 num_shadows_range: Tuple[int, int] = (1, 3),
                 size_range: Tuple[float, float] = (0.1, 0.3)):
        """
        初始化随机阴影伪影增强

        Args:
            p: 应用增强的概率
            intensity_range: 阴影强度范围 (min, max)
            num_shadows_range: 阴影数量范围 (min, max)
            size_range: 阴影大小范围 (相对于图像尺寸的比例)
        """
        self.p = p
        self.intensity_range = intensity_range
        self.num_shadows_range = num_shadows_range
        self.size_range = size_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x

        # 确保输入是4D张量 (b, 1, h, w)
        original_shape = x.shape
        if x.dim() == 3:
            x = x.unsqueeze(0)  # 如果是单张图像，添加batch维度

        batch_size, channels, height, width = x.shape

        for im in x:
            plt.figure()
            plt.imshow(im.detach().cpu().permute(1, 2, 0), cmap='gray')
            plt.show()

        # 处理batch中的每个样本
        augmented_batch = []
        for i in range(batch_size):
            # 转换为numpy进行处理
            img_np = x[i].squeeze().cpu().numpy()

            # 添加阴影
            img_with_shadow = self._add_shadow_artifacts(img_np, height, width)

            # 转换回tensor
            img_tensor = torch.from_numpy(img_with_shadow).float()
            if channels == 1:
                img_tensor = img_tensor.unsqueeze(0)  # 保持通道维度

            augmented_batch.append(img_tensor)

        # 重新组合batch
        result = torch.stack(augmented_batch).to(x.device)

        # 恢复原始形状
        # if len(original_shape) == 3:
        #     result = result.squeeze(0)

        # for im in result:
        #     plt.figure()
        #     plt.imshow(im.detach().cpu().permute(1, 2, 0), cmap='gray')
        #     plt.show()

        return result

    def _add_shadow_artifacts(self,
                              image: np.ndarray,
                              height: int,
                              width: int) -> np.ndarray:
        # 创建图像副本
        result = image.copy()

        # 随机确定阴影数量
        num_shadows = random.randint(self.num_shadows_range[0], self.num_shadows_range[1])

        for _ in range(num_shadows):
            # 随机阴影强度
            intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])

            # 随机阴影大小
            shadow_size = int(min(height, width) * random.uniform(self.size_range[0], self.size_range[1]))

            # 随机阴影位置
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)

            # 创建阴影遮罩
            shadow_mask = np.ones_like(image)

            # 生成圆形或椭圆形阴影
            if random.random() > 0.5:
                # 圆形阴影
                y, x = np.ogrid[:height, :width]
                mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= shadow_size ** 2
            else:
                # 椭圆形阴影
                y, x = np.ogrid[:height, :width]
                ellipse_size_x = shadow_size
                ellipse_size_y = int(shadow_size * random.uniform(0.5, 1.5))
                mask = ((x - center_x) ** 2 / ellipse_size_x ** 2 +
                        (y - center_y) ** 2 / ellipse_size_y ** 2) <= 1

            # 应用阴影
            # result[mask] = result[mask] * (1 - intensity)
            result[mask] = intensity

        return np.clip(result, 0, 1)


class RandomStreakShadow:
    def __init__(self,
                 p: float = 0.5,
                 intensity_range: Tuple[float, float] = (0.75, 0.95),
                 num_shadows_range: Tuple[int, int] = (1, 3),
                 length_range: Tuple[float, float] = (0.2, 0.5),
                 width_range: Tuple[float, float] = (0.01, 0.05),
                 shape_types: list[str] = None):
        """
        初始化随机条状阴影增强

        Args:
            p: 应用增强的概率
            intensity_range: 阴影强度范围
            num_shadows_range: 阴影数量范围
            length_range: 阴影长度范围 (相对于图像尺寸的比例)
            width_range: 阴影宽度范围 (相对于图像尺寸的比例)
            shape_types: 阴影形状类型 ['straight', 'curved', 's_shape', 'zigzag', 'random']
        """
        self.p = p
        self.intensity_range = intensity_range
        self.num_shadows_range = num_shadows_range
        self.length_range = length_range
        self.width_range = width_range
        self.shape_types = shape_types or ['straight', 'curved', 'curved2']

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x

        # for im in x:
        #     plt.figure()
        #     plt.imshow(im.detach().cpu().permute(1, 2, 0), cmap='gray')
        #     plt.show()
        original_shape = x.shape
        if x.dim() == 3:
            x = x.unsqueeze(0)

        batch_size, channels, height, width = x.shape

        augmented_batch = []
        for i in range(batch_size):
            img_np = x[i].detach().squeeze().cpu().numpy()
            img_with_shadow = self._add_streak_shadows(img_np, height, width)
            img_tensor = torch.from_numpy(img_with_shadow).float()
            if channels == 1:
                img_tensor = img_tensor.unsqueeze(0)
            augmented_batch.append(img_tensor)

        result = torch.stack(augmented_batch).to(x.device)
        if len(original_shape) == 3:
            result = result.squeeze(0)
        # for im in result:
        #     plt.figure()
        #     plt.imshow(im.detach().cpu().permute(1, 2, 0), cmap='gray')
        #     plt.show()
        return result

    def _add_streak_shadows(self, image: np.ndarray, height: int, width: int) -> np.ndarray:
        """添加随机条状阴影"""
        result = image.copy()
        num_shadows = random.randint(self.num_shadows_range[0], self.num_shadows_range[1])

        for _ in range(num_shadows):
            intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
            shadow_width = int(width * random.uniform(self.width_range[0], self.width_range[1]))
            shadow_length = int(height * random.uniform(self.length_range[0], self.length_range[1]))

            # 随机选择形状类型
            shape_type = random.choice(self.shape_types)

            # 生成中心线
            centerline = self._generate_centerline(height, width, shadow_length, shape_type)

            if len(centerline) > 1:
                # 创建阴影掩码
                shadow_mask = np.zeros_like(image, dtype=np.uint8)

                # 绘制条状阴影
                for i in range(len(centerline) - 1):
                    pt1 = tuple(map(int, centerline[i]))
                    pt2 = tuple(map(int, centerline[i + 1]))
                    cv2.line(shadow_mask, pt1, pt2, 255, shadow_width)

                # 模糊边缘
                shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=3)
                shadow_mask = shadow_mask.astype(np.float32) / 255.0

                # 应用阴影
                result = result * (1 - shadow_mask * intensity)

        return np.clip(result, 0, 1)

    def _generate_centerline(self, height: int, width: int, length: int, shape_type: str) -> np.ndarray:
        """生成随机形状的中心线"""
        # 随机起始点
        start_y = random.randint(0, height)
        start_x = random.randint(0, width)

        # 随机方向角度
        angle = random.uniform(0, 2 * math.pi)
        shape_type = 'curved'

        if shape_type == 'straight':
            # 直线
            end_x = start_x + length * math.cos(angle)
            end_y = start_y + length * math.sin(angle)
            return np.array([[start_x, start_y], [end_x, end_y]])

        elif shape_type == 'curved':
            # 曲线
            num_points = random.randint(3, 5)
            points = [[start_x, start_y]]

            for i in range(1, num_points):
                seg_length = length / (num_points - 1)
                curve_angle = angle + random.uniform(0, 1.2)  # 轻微弯曲

                x = points[-1][0] + seg_length * math.cos(curve_angle)
                y = points[-1][1] + seg_length * math.sin(curve_angle)
                points.append([x, y])
            return np.array(points)

        elif shape_type == 'curved2':
            # 简化的自然曲线生成
            num_points = 15  # 更多点使曲线更平滑
            points = []

            # 使用正弦函数生成自然弯曲
            curve_frequency = random.uniform(0.5, 2.0)  # 弯曲频率
            curve_amplitude = random.uniform(0.1, 0.3) * length  # 弯曲幅度

            for i in range(num_points):
                t = i / (num_points - 1)

                # 主要方向
                x_main = start_x + t * length * math.cos(angle)
                y_main = start_y + t * length * math.sin(angle)

                # 垂直方向的弯曲
                perpendicular_angle = angle + math.pi / 2  # 垂直于主方向
                bend = curve_amplitude * math.sin(t * math.pi * curve_frequency)

                x = x_main + bend * math.cos(perpendicular_angle)
                y = y_main + bend * math.sin(perpendicular_angle)

                # 确保点在图像范围内
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))

                points.append([x, y])

            return np.array(points)

class Transforms:
    def __init__(
            self,
            size: int,
            eps: float = 1e-6,
            apply_augmentations: bool = True  # 新增：控制是否应用数据增强
    ):
        self.eps = eps
        self.apply_augmentations = apply_augmentations

        # 基础变换
        self.base_transforms = Compose([
            Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + eps)),
            Normalize(mean=0.3080, std=0.1494),
        ])

        # 数据增强变换（只在训练时应用）
        self.augmentation_transforms1 = Compose([
            RandomBrightnessContrast(p=0.6, brightness_range=(0.6, 1.4), contrast_range=(0.3, 2)),
        ])
        self.augmentation_transforms2 = Compose([
            RandomStreakShadow(p=0.8, intensity_range=(0.6, 1.0), width_range=(0.02, 0.08), length_range=(0.2, 0.9))
        ])

        self.resize = Resize((size, size), antialias=True)

        # 创建圆形mask
        y_coord = torch.arange(size) - size // 2
        x_coord = torch.arange(size) - size // 2
        Y, X = torch.meshgrid(y_coord, x_coord, indexing='ij')
        distance_sq = X ** 2 + Y ** 2
        mask = (distance_sq <= 119 ** 2).float()
        self.mask = mask.unsqueeze(0).unsqueeze(0)

    def __call__(self, x: torch.Tensor, reverse: bool = True, is_training: bool = True):
        # 调整大小
        x = self.resize(x)

        # 归一化到[0,1]
        x = (x - x.min()) / (x.max() - x.min() + self.eps)

        # 反转（如果需要）
        if reverse:
            x = 1 - x

        # for im in x:
        #     plt.figure()
        #     plt.imshow(im.detach().cpu().permute(1, 2, 0), cmap='gray')
        #     plt.show()

        # 只在训练时应用数据增强
        # x = self.augmentation_transforms1(x)
        temp = x
        x = self.augmentation_transforms2(x)
        diff = torch.abs(temp - x)

        # 应用圆形mask
        mask = self.mask.to(x.device)
        x = x * mask

        # 应用基础变换
        x = self.base_transforms(x)

        return x, diff


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

