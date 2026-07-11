import os
import torch
import numpy as np
import pydicom
from diffdrr.drr import DRR
from matplotlib import pyplot as plt
from nibabel.nicom.tests.test_utils import pydicom

from diffpose.deepfluoro import get_random_offset

from diffpose.calibration import RigidTransform
from scipy.ndimage import zoom

device = torch.device("cpu")
# root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/何美珠/CT/HeMeiZhu/20240710193503.877000/2"
root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/史建基/CT/ShiJianJi/20231019091100.063/602"
file_name = os.listdir(root)
file_name.sort(reverse=True)

# dcm_file = pydicom.dcmread("/media/sda1/Data/ERCP/CT+X+MRCP/liwei/何美珠/CT/HeMeiZhu/20240710193503.877000/2/93356399_20240710_1_1288.dcm")
dcm_file = pydicom.dcmread(os.path.join(root, file_name[0]))

distance_to_detector = dcm_file.get("DistanceSourceToDetector")
distance_to_patient = dcm_file.get("DistanceSourceToPatient")
focal = distance_to_patient * distance_to_detector / (distance_to_detector + distance_to_patient)

# x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/何美珠/ERCP/HEMEIZHU^^/20240712152119/1/"
x_root= "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/史建基/ERCP/SHI^JIANJI^/20231023150900/1"
x_file = os.listdir(x_root)
x_filename = os.path.join(x_root, x_file[0])
x_ray = pydicom.dcmread("/media/sda1/Data/ERCP/CT+X+MRCP/liwei/史建基/ERCP/SHI^JIANJI^/20231023150900/1/94108134_20231023_1_29.dcm")
x_distance_to_detector = x_ray.get("DistanceSourceToDetector")
x_img = x_ray.pixel_array

# for x_name in x_file:
#     img = pydicom.dcmread(os.path.join(x_root, x_name)).pixel_array
#     plt.figure()
#     plt.imshow(img, cmap='gray')
#     plt.show()
# plt.figure()
# plt.imshow(x_img, cmap="gray")
# plt.show()

pixel_spacing = dcm_file.get("PixelSpacing")
slice_thickness = dcm_file.get("SliceThickness")
pixel_spacing = np.array(pixel_spacing)
pixel_spacing = np.append(pixel_spacing, slice_thickness)
distance_to_detector = dcm_file.get("DistanceSourceToDetector")
intensifier_size = x_ray.get("IntensifierSize")
delx = intensifier_size / 512
height = 256
delx = 512 / height * delx

volume = None

for f_name in file_name:
    file_path = os.path.join(root, f_name)
    volume_img = pydicom.dcmread(file_path).pixel_array
    # plt.figure()
    # plt.imshow(volume_img, cmap='gray')
    # plt.show()
    volume_img = np.expand_dims(volume_img.astype(np.float32), axis=0)
    if volume is None:
        volume = volume_img
    else:
        volume = np.concatenate((volume, volume_img), axis=0)

rescale_slope = dcm_file.get("RescaleSlope")
rescale_intercept = dcm_file.get("RescaleIntercept")
volume = volume * rescale_slope + rescale_intercept
# plt.figure()
# plt.imshow(volume[0], cmap='gray')
# plt.show()
# volume = volume[30 : 100, :, :]

# plt.figure()
# plt.imshow(volume[0], cmap="gray")
# plt.show()
start = 0
end = 135
# volume = volume[start : end, : , :]
factors = [2, 2, 4]
factors = [1, 1, 1]
pixel_spacing = pixel_spacing / factors
pixel_spacing[0], pixel_spacing[2] = pixel_spacing[2], pixel_spacing[0]
volume = np.swapaxes(volume, 1, 2).copy()
# volume = zoom(volume, factors, order=1)

sdr = torch.tensor(500, device=device)

drr = DRR(
    volume,
    pixel_spacing,
    sdr,
    height,
    delx=3,
    reverse_x_axis=True
)

# isocenter_xyz = [512, 512, len(file_name)] * pixel_spacing / 2 * factors
# isocenter_xyz = [512, 512, end - start] * pixel_spacing / 2 * factors
isocenter_xyz = [115, 900, 980] * pixel_spacing / 2 * factors
isocenter_xyz = torch.tensor(isocenter_xyz).unsqueeze(0)
# isocenter_rot = torch.tensor([[torch.pi / 2, 0.0, -torch.pi / 2]]).unsqueeze(0)
isocenter_rot = torch.tensor([[-torch.pi / 2, 0.0, torch.pi / 2]]).unsqueeze(0)

isocenter_pose = RigidTransform(
    isocenter_rot, isocenter_xyz, "euler_angles", "ZYX"
)
isocenter_pose = isocenter_pose.to(device)

contrast_distribution = torch.distributions.Uniform(1.0, 10.0)
contrast = contrast_distribution.sample().item()

offset = get_random_offset(1, device)
pose = isocenter_pose.compose(offset)
img = drr(None, None, None, pose=isocenter_pose, bone_attenuation_multiplier=4)
plt.figure()
plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# for _ in range(10):
#     offset = get_random_offset(1, device)
#     pose = isocenter_pose.compose(offset)
#     img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=3)
#
#     plt.figure()
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.show()

