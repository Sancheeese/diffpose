
import time

import cv2
import numpy as np
import pandas as pd
import torch

from diffpose.calibration import RigidTransform, convert
from ours.cut.style_to_drr import StyleChanger
from ours.register_zyl_bone_mask_stage import inpaint_with_opencv
from ours.utils.grad_similar import calculate_gradient_consistency_with_mask, gradient_ncc, dice_coefficient_with_mask, \
    get_edge, multiscale_gradient_ncc
from ours.utils.img_utils import see_mid, rank_transform_tensor

from utils.drr import DRR
from utils.drr_bone import DRR as DRR_Bone
# from utils.drr_bone import DRR
# from diffpose.metrics import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics2 import MultiscaleNormalizedCrossCorrelation2d
from utils.metrics_mask_tube2_wei3 import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics_mask_tube2 import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics2 import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics_mask_tube_weight import MultiscaleNormalizedCrossCorrelation2d
from matplotlib import pyplot as plt
from torchvision.transforms.functional import resize
from tqdm import tqdm

from utils.CT_dataset import Transforms, toZeroOne
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from utils.registration_bone_mask3 import PoseRegressor
from utils.registration import SparseRegistration, VectorizedNormalizedCrossCorrelation2d
from ours.utils.CT_dataset_PA import IntubationDataset, create_circle_mask
import kornia
import cma
import torch.nn.functional as F
import nibabel as nib
import torchvision.transforms.v2 as transforms


class Registration:
    def __init__(
        self,
        drr,
        drr_bone,
        specimen,
        model,
        parameterization,
        convention=None,
        n_iters=500,
        verbose=False,
        device="cuda:0",
    ):
        self.device = torch.device(device)
        self.drr = drr.to(self.device)
        self.drr_bone = drr_bone.to(self.device)
        self.model = model.to(self.device)
        model.eval()

        self.specimen = specimen
        self.isocenter_pose = specimen.isocenter_pose.to(self.device)
        self.center_pose = specimen.center_pose.to(device)
        self.back_pose = specimen.back_pose.to(device)

        self.geodesics = GeodesicSE3()
        self.doublegeo = DoubleGeodesic(sdr=self.specimen.sdr)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 50], [0.7, 0.3], device=self.device)
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 21], [0.5, 0.5], device=self.device, step=[None, 1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 32], [0.5, 0.5], device=self.device, step=[None, 16])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([128, 15], [0.5, 0.5], device=self.device, step=[64, 1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 32, 8], [0.3, 0.4, 0.3], device=self.device, step=[None, 16, 4])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([15], [1], device=self.device, step=[1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        # self.transforms = Transforms(self.drr.detector.height)
        self.transforms = Transforms(self.drr.detector.height, radius=119)
        self.parameterization = parameterization
        self.convention = convention

        self.n_iters = n_iters
        self.verbose = verbose

        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_nec5/40_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec5/70_net_G.pth",
        # self.style_change = StyleChanger("cut/ckpt/70_net_G.pth",
        # self.style_change = StyleChanger("/media/sda1/PersonalFiles/yx/project/diffpose/ours/cut/drr_style_solid_5_white/70_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_new/75_net_G.pth",
        self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_new3/70_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_white/80_net_G.pth",
        # self.style_change =  StyleChanger("/media/sda1/PersonalFiles/yx/project/diffpose/ours/cut/drr_style_solid_5_new3/70_net_G.pth",
        # self.style_change =  StyleChanger("/media/sda1/PersonalFiles/yx/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_white3/70_net_G.pth",
        # self.style_change =  StyleChanger("/media/sda1/PersonalFiles/yx/project/diffpose/ours/cut/drr_style_solid_5_new4/90_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_4_choose/100_net_G.pth",
                       device=self.device,
                       resize=256)
        self.times = []
        self.losses = []
        self.i = 0
        self.tres = []
        self.cir = create_circle_mask(256, 115)
        self.change = False

    def initialize_registration(self, img):
        with torch.no_grad():
            offset = self.model(img)
            features = self.model.backbone.forward_features(img)
            features = resize(
                features,
                (self.drr.detector.height, self.drr.detector.width),
                interpolation=3,
                antialias=True,
            )
            features = features.sum(dim=[0, 1], keepdim=True)
            features -= features.min()
            features /= features.max() - features.min()
            features /= features.sum()
        pred_pose = self.isocenter_pose.compose(offset)

        return SparseRegistration(
            self.drr,
            pose=pred_pose,
            parameterization=self.parameterization,
            convention=self.convention,
            features=features,
        )

    def initialize_optimizer(self, registration):
        # optimizer = torch.optim.Adam(
        #     [
        #         # {"params": [registration.rotation], "lr": 7.5e-3},
        #         {"params": [registration.rotation], "lr": 1.5e-2},
        #         {"params": [registration.translation], "lr": 7.5e0},
        #     ],
        #     maximize=True,
        # )
        optimizer = torch.optim.SGD(
            [
                {"params": [registration.rotation], "lr": 1.5e-2},
                {"params": [registration.translation], "lr": 15e0},
            ],
            maximize=True,
            momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=25,
            gamma=0.9,
        )
        return optimizer, scheduler

    def evaluate(self, registration):
        est_pose = registration.get_current_pose()
        rot = est_pose.get_rotation("euler_angles", "ZYX")
        xyz = est_pose.get_translation()
        alpha, beta, gamma = rot.squeeze().tolist()
        bx, by, bz = xyz.squeeze().tolist()
        param = [alpha, beta, gamma, bx, by, bz]
        geo = (
            torch.concat(
                [
                    *self.doublegeo(est_pose, self.pose),
                    self.geodesics(est_pose, self.pose),
                ]
            )
            .squeeze()
            .tolist()
        )
        # tre = self.target_registration_error(est_pose.cpu()).item()
        # return param, geo, tre
        return param, geo

    def normalize(self, x):
        """将参数从 [lb, ub] 映射到 [0, 1]"""
        return (x + 1000) / 2000

    def denormalize(self, x_normalized):
        """将参数从 [0, 1] 映射回 [lb, ub]"""
        return x_normalized * 2000 - 1000

    def nlopt_objective(self, x):
        with torch.no_grad():
            # 将numpy数组转换为torch tensor
            start_time = time.time()
            # x = self.denormalize(x)
            x = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=True)

            # 更新registration的位姿参数
            rot = x[:3].unsqueeze(0)
            xyz = x[3:].unsqueeze(0)

            pose = RigidTransform(rot, xyz, parameterization="so3_log_map", device=self.device)
            # pose = RigidTransform(rot, xyz, "euler_angles", "ZYX")
            # pose = pose.compose(self.isocenter_pose)
            # pose = self.isocenter_pose.compose(pose)
            # pose = self.isocenter_pose.compose(self.back_pose).compose(pose).compose(self.center_pose)

            pred_img = self.drr(None, None, None, pose=pose)
            pred_img = self.transforms(pred_img).to(self.device).to(torch.float32)
            # pred_img = torch.pow(toZeroOne(pred_img), 0.7)
            # pred_img = self.transforms(pred_img, reverse=False).to(self.device).to(torch.float32)

            pred_img_bone = self.drr_bone(None, None, None, pose=pose)
            pred_img_bone = self.transforms(pred_img_bone, reverse=False)
            pred_img_bone = (pred_img_bone - pred_img_bone.min()) / (pred_img_bone.max() - pred_img_bone.min())
            pred_img_bone1 = torch.tanh(50 * pred_img_bone)
            pred_img_bone2 = torch.tanh(1 * pred_img_bone)
            pred_bone_wei = torch.ones_like(pred_img_bone1).to(self.device)
            pred_bone_wei[pred_img_bone1 > 0.9] = 1.8
            pred_bone_wei[pred_img_bone1 <= 0.9] = 0.2

            # get_edge(self.img, self.total_mask)
            # wei2 = torch.ones_like(pred_img_bone, requires_grad=False)
            # # wei2[pred_img_bone > 0.1] = 1
            # wei2[pred_img_bone <= 0.1] = 0
            # plt.figure()
            # plt.imshow(pred_img_bone.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
            # plt.show()

            # plt.figure()
            # plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            # plt.show()
            # plt.figure()
            # plt.imshow(self.img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            # plt.show()
            # ncc = self.criterion(pred_img, self.img)
            weight = 0.5
            # loss = (1 - weight) * gradient_ncc(pred_img, self.img, self.total_mask) + weight * self.criterion(pred_img, self.img)
            # l1 = (1 - weight) * gradient_ncc(pred_img, self.img, self.total_mask)
            # l2 = weight * self.criterion(pred_img, self.img, None, self.wei_img)
            dice = dice_coefficient_with_mask(pred_img_bone1, self.mask_bone, self.total_mask)
            loss = dice + self.criterion(pred_img, self.img_ori)
            # loss = dice + self.criterion(pred_img, self.img_ori, pred_bone_wei, self.mask_bone_wei)
            # loss = weight * dice + weight * self.criterion(pred_img, self.img)
            # loss = weight * dice + weight * multiscale_gradient_ncc(pred_img_bone2, self.img, self.total_mask, [256, 13], [0.5, 0.5])
            # loss = (1 - weight) * gradient_ncc(pred_img, self.img, self.total_mask) + weight * self.criterion(pred_img, self.img)
            # if not self.change:
            #     loss = dice
            #     # loss = dice + gradient_ncc(pred_img_bone, self.img, self.total_mask)
            # else:
            #     loss = dice + self.criterion(pred_img, self.img)
            # loss = dice + gradient_ncc(pred_img, self.img, self.total_mask) + self.criterion(pred_img, self.img)
            # ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * self.total_mask, toZeroOne(self.img) * self.total_mask, window_size=11, reduction='mean')
            ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * self.total_mask, toZeroOne(self.img_ori) * self.total_mask, window_size=11, reduction='mean')
            # ssim = (1 - weight) * gradient_ncc(pred_img, self.img, self.total_mask) + weight * self.criterion(pred_img, self.img)
            # grad_ncc = gradient_ncc(pred_img, self.img, self.total_mask)
            # loss = ncc
            # loss = ssim
            # loss = dice + ssim

            # plt.figure()
            # plt.imshow(self.img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            # plt.show()

            if self.i % 100 == 0:
                print(self.criterion(pred_img, self.img_ori))
                plt.figure()
                plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
                plt.show()
                # plt.figure()
                # plt.imshow(pred_img_bone1.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
                # plt.show()
                # see_mid(pred_img, self.total_mask)

            if self.i % 50 == 0:
                rot = pose.get_rotation("euler_angles", "ZYX")
                xyz = pose.get_translation()
                alpha, beta, gamma = rot.squeeze().tolist()
                bx, by, bz = xyz.squeeze().tolist()
                param = [alpha, beta, gamma, bx, by, bz]

                rot2 = pose.get_rotation(parameterization="so3_log_map")
                xyz2 = pose.get_translation()
                alpha2, beta2, gamma2 = rot2.squeeze().tolist()
                bx2, by2, bz2 = xyz2.squeeze().tolist()
                param2 = [alpha2, beta2, gamma2, bx2, by2, bz2]

                true_fiducials, pred_fiducials = self.specimen.get_2d_fiducials(self.idx, pose)
                tre = torch.norm(true_fiducials - pred_fiducials, dim=2)
                tre = torch.mean(tre)

                # print_tre(self.gt_img, true_fiducials[0].detach().numpy())
                # print_tre(pred_img, pred_fiducials[0].detach().numpy())

                geo = (
                    torch.concat(
                        [
                            *self.doublegeo(pose, self.gt_pose),
                            self.geodesics(pose, self.gt_pose),
                        ]
                    )
                    .squeeze()
                    .tolist()
                )
                self.geodesic.append(geo)
                self.params.append(param)
                self.losses.append(loss.item())
                self.fiducial.append(tre.item())
                self.times.append(time.time() - start_time)
                self.ssims.append(ssim.item())
                self.params2.append(param2)
            self.i += 1

        # return 1 - loss.item()
        return -loss.item()


    def run(self, idx):
        img, pose = self.specimen[idx]
        self.idx = idx
        self.gt_pose = pose.to(self.device)

        img_rev = self.transforms(img, reverse=True).to(self.device).to(torch.float32)
        tube = torch.ones_like(img_rev).to(self.device)
        tube[toZeroOne(img_rev) > 0.7] = 0
        plt.figure()
        plt.imshow(tube.cpu().squeeze(), cmap="gray")
        plt.show()

        gt_pose = self.specimen.get_manual_gt(idx).to(self.device)
        gt_img = self.drr(None, None, None, pose=gt_pose, bone_attenuation_multiplier=3)
        gt_img = self.transforms(gt_img, reverse=True).to(self.device).to(torch.float32)
        # gt_img = toZeroOne(gt_img)
        # # for i in range(50):
        # #     im = torch.tanh((i + 1) * gt_img)
        # #     plt.figure()
        # #     plt.imshow(im.cpu().squeeze(), cmap="gray")
        # #     plt.show()
        # gt_img = torch.tanh(50 * gt_img)
        # gt_img = 2 * gt_img - 1
        # # gt_img[tube == 0] = -1
        # self.gt_img = gt_img
        plt.figure()
        plt.imshow(gt_img.cpu().squeeze(), cmap="gray")
        plt.show()
        # filename = self.specimen.get_x_filename(idx).split(".")[0] + "_nochange"
        # file_path = f"/media/sda1/PersonalFiles/yx/dataset/zyl_result/{filename}.nii.gz"
        # # file_path = f"nnuet/zyl_result/{filename}.nii.gz"
        # nii_img = nib.load(file_path)
        # img_data = nii_img.get_fdata()
        # # img_data = img_data[::-1, :].copy()
        # bone_mask_gt = torch.tensor(img_data, device=self.device)
        # bone_mask_gt = torch.tensor(bone_mask_gt, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        # bone_mask_gt = transforms.Resize(256)(bone_mask_gt)
        # bone_mask_gt[bone_mask_gt >= 0.5] = 1
        # bone_mask_gt[bone_mask_gt < 0.5] = 0
        # bone_mask_gt = 2 * bone_mask_gt - 1
        # bone_mask_gt[tube == 0] = -1
        # plt.figure()
        # plt.imshow(bone_mask_gt.cpu().squeeze(), cmap="gray")
        # plt.show()
        # self.model.mask1 = F.interpolate(bone_mask_gt, [64, 64], mode='bilinear')
        # self.model.mask2 = F.interpolate(bone_mask_gt, [32, 32], mode='bilinear')

        # plt.figure()
        # plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        # img = get_tube_on_image(img, black=False)
        img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        # img = self.transforms(img, reverse=False)
        img_ori = torch.tensor(img).to(self.device).to(torch.float32)
        self.img_ori = img_ori

        img_change = self.style_change(img)
        img_change = self.transforms(img_change, reverse=False).to(self.device).to(torch.float32)
        # diff = torch.abs(img - img_change)
        # print(diff.min())
        # print(diff.max())
        # diff = (diff - diff.min()) / (diff.max() - diff.min()).to(self.device)
        # threshold = 0.25
        # diff[diff <= threshold] = 0
        # diff[diff > threshold] = 1
        # diff = 1 - diff
        # img_rev = self.transforms(img, reverse=True).to(self.device).to(torch.float32)
        # diff = img_rev - img_change
        diff = img - img_change
        diff = (diff - diff.min()) / (diff.max() - diff.min()).to(self.device)
        threshold = 0.23
        diff[diff <= threshold] = 0
        diff[diff > threshold] = 1
        circle_mask = create_circle_mask(256, 116).to(self.device).unsqueeze(0).unsqueeze(0)
        # total_mask = (circle_mask.bool() & diff.bool()).float()
        total_mask = (circle_mask.bool() & tube.bool()).float()

        # white_mask = torch.ones_like(img)
        # white_mask[toZeroOne(img) > 0.7] = 0
        # plt.figure()
        # plt.imshow(circle_mask.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        # white_mask = white_mask * circle_mask
        # # white_mask = circle_mask
        # plt.figure()
        # plt.imshow(white_mask.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        # self.total_mask = white_mask
        # self.criterion.set_mask(total_mask)
        # self.criterion.set_mask(circle_mask)
        # self.criterion.set_weight_mask(spine_mask)

        plt.figure()
        plt.imshow(total_mask.cpu().squeeze(0).permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
        plt.show()
        img = img_change
        img = (img - img.min()) / (img.max() - img.min())
        black = 1 - diff
        black[black > 0] = 1
        black[black <= 0] = 0
        print(black.min())
        print(black.max())
        plt.figure()
        plt.imshow(diff.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()
        # img[black == 1] = 0.15
        # img[black == 1] = img[black == 1].pow(0.8)
        # gt_img[black == 1] = 0.15
        # gt_img[black == 1]= img_change[black == 1]
        img_input = torch.pow(img, 1)
        img_input = inpaint_with_opencv(img_input, black)
        # img_input = inpaint_with_opencv(img_input, 1 - tube)
        # img_input[tube == 0] = 0
        img_input = self.transforms(img_input, reverse=False).to(self.device).to(torch.float32)
        img = inpaint_with_opencv(img, black)
        # img = toZeroOne(gt_img)
        # img = torch.pow(img, 1.5)
        # img = transforms_aug(img)
        img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        self.pose = pose.to(self.device)
        img = img_input
        self.img = img
        # img = rank_transform_tensor(img)
        plt.figure()
        plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"NO.{idx}")
        plt.show()

        white_mask = torch.ones_like(img)
        white_mask[toZeroOne(img) > 0.7] = 0
        plt.figure()
        plt.imshow(img_ori.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()
        white_mask = white_mask * circle_mask
        # white_mask = circle_mask
        # plt.figure()
        # plt.imshow(white_mask.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        total_mask = total_mask * white_mask
        plt.figure()
        plt.imshow(total_mask.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()
        self.total_mask = total_mask
        # self.criterion.set_mask(white_mask)
        self.criterion.set_mask(total_mask)

        a = 3
        tmp = img
        # img = img_ori
        # self.img = img_ori
        wei_img = torch.zeros_like(img)
        wei_img[circle_mask > 0] = torch.exp(-a * toZeroOne(img[circle_mask > 0]))
        wei_img[circle_mask <= 0] = wei_img[circle_mask > 0].min()
        print(wei_img.min())
        print(wei_img.max())
        plt.figure()
        plt.imshow(wei_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()

        self.wei_img = wei_img
        self.params = []
        self.losses = []
        self.geodesic = []
        self.fiducial = []
        self.times = []
        self.ssims = []
        self.params2 = []
        self.ncc = []

        # img_ori = toZeroOne(img_ori)
        # img_ori[tube == 0] = 0
        # img_ori = self.transforms(img_ori, reverse=False).to(self.device).to(torch.float32)
        # plt.figure()
        # plt.imshow(img_ori.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        initial_pose, _ = self.model(img)
        initial_pose = self.isocenter_pose.compose(initial_pose)
        pred_pose = initial_pose
        # initial_pose = self.back_pose.inverse().compose(self.isocenter_pose.inverse()).compose(initial_pose).compose(self.center_pose.inverse())
        # initial_pose = initial_pose.compose(self.isocenter_pose.inverse())

        mask_bone = F.interpolate(self.model.mask, img.shape[-2:], mode='bilinear')
        mask_bone[mask_bone > 0] = 1
        mask_bone[mask_bone <= 0] = 0
        mask_bone = toZeroOne(mask_bone)
        plt.figure()
        plt.imshow(mask_bone.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()
        self.mask_bone = mask_bone
        mask_bone_wei = torch.ones_like(mask_bone).to(self.device)
        mask_bone_wei[mask_bone == 1] = 1.8
        mask_bone_wei[mask_bone == 0] = 0.2
        self.mask_bone_wei = mask_bone_wei

        # initial_pose = get_random_offset(1, self.device)
        # isocenter_pose = self.specimen.isocenter_pose.to(self.device)
        # back_pose = self.specimen.back_pose.to(self.device)
        # center_pose = self.specimen.center_pose.to(self.device)
        # initial_pose = self.isocenter_pose.compose(back_pose).compose(initial_pose).compose(center_pose)
        p = self.drr(None, None, None, pose=pred_pose)
        p = self.transforms(p).to(self.device)
        plt.figure()
        plt.imshow(p.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()

        # see_mid(p, total_mask)
        # see_mid(gt_img, total_mask)
        # see_mid(img, total_mask)

        # self.target_registration_error = Evaluator(self.specimen, idx)
        rot = initial_pose.get_rotation(parameterization="so3_log_map").detach().cpu().numpy()[0]
        # rot = initial_pose.get_rotation("euler_angles", "ZYX").detach().cpu().numpy()[0]
        xyz = initial_pose.get_translation().detach().cpu().numpy()[0]
        # rot = gt_pose.get_rotation(parameterization="so3_log_map").detach().cpu().numpy()[0]
        # xyz = gt_pose.get_translation().detach().cpu().numpy()[0]

        # 对于se3对数映射参数化
        dim = 6  # 3旋转 + 3平移
        initial_params = np.hstack([rot, xyz])
        # initial_params = np.array([1.74, -0.8, -0.7, 248, -7.3, 123])

        r_l = 0.05
        t_l = 10
        s = [r_l, r_l, r_l, t_l, t_l, t_l]
        # , 左右,
        # s = [1e-7, 1e-7, 1e-7, 5, 5, 5]
        seed = 333
        option = {
            "seed": seed,
            # "CMA_rankmu": 0.2,
            "popsize": 100,
            "CMA_stds": s,
            "maxiter": 50,
            "verb_disp": 1,
            "tolfun": 1e-6,
            "tolx": 1e-6,
            # "CMA_active": True
        }
        r_l = 0.1
        t_l = 10
        s2 = [r_l, r_l, r_l, t_l, t_l, t_l]
        option2 = {
            "seed": seed,
            # "CMA_rankmu": 0.2,
            "popsize": 100,
            "CMA_stds": s2,
            "maxiter": 15,
            "verb_disp": 1,
            "tolfun": 1e-6,
            "tolx": 1e-6,
            # "CMA_active": True
        }

        l1 = gradient_ncc(gt_img, self.img, self.total_mask)
        l2 = self.criterion(gt_img, self.img_ori)

        self.change = False
        optimizer = cma.CMAEvolutionStrategy(x0=initial_params, sigma0=1, options=option)
        optimizer.optimize(self.nlopt_objective)
        self.change = True
        for i in range(1):
            best_solution = optimizer.result.xbest
            best_fvalue = optimizer.result.fbest
            x = torch.tensor(best_solution, dtype=torch.float32, device=self.device, requires_grad=False)
            r = x[:3].unsqueeze(0)
            t = x[3:].unsqueeze(0)
            pose = RigidTransform(r, t, parameterization="so3_log_map", device=self.device)
            # pose = RigidTransform(r, t, "euler_angles", "ZYX")
            # pose = self.isocenter_pose.compose(self.back_pose).compose(pose).compose(self.center_pose)
            p = self.drr(None, None, None, pose=pose)
            p = self.transforms(p).to(self.device)
            plt.figure()
            plt.imshow(p.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            plt.show()

            print(f"重启前第{i + 1}轮最佳值: {best_fvalue}")
            optimizer = cma.CMAEvolutionStrategy(x0=best_solution, sigma0=1, options=option2)
            # 再次运行优化
            optimizer.optimize(self.nlopt_objective)

        x = optimizer.result.xbest
        x = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=False)
        r = x[:3].unsqueeze(0)
        t = x[3:].unsqueeze(0)
        pose = RigidTransform(r, t, parameterization="so3_log_map", device=self.device)
        # pose = RigidTransform(r, t, "euler_angles", "ZYX")
        # pose = self.isocenter_pose.compose(self.back_pose).compose(pose).compose(self.center_pose)
        p = self.drr(None, None, None, pose=pose)
        p = self.transforms(p).to(self.device)
        plt.figure()
        plt.imshow(p.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()

        rot = pose.get_rotation("euler_angles", "ZYX")
        xyz = pose.get_translation()
        alpha, beta, gamma = rot.squeeze().tolist()
        bx, by, bz = xyz.squeeze().tolist()
        param = [alpha, beta, gamma, bx, by, bz]

        rot2 = pose.get_rotation(parameterization="so3_log_map")
        xyz2 = pose.get_translation()
        alpha2, beta2, gamma2 = rot2.squeeze().tolist()
        bx2, by2, bz2 = xyz2.squeeze().tolist()
        param2 = [alpha2, beta2, gamma2, bx2, by2, bz2]

        true_fiducials, pred_fiducials = self.specimen.get_2d_fiducials(self.idx, pose)
        tre = torch.norm(true_fiducials - pred_fiducials, dim=2)
        tre = torch.mean(tre)

        # print_tre(self.gt_img, true_fiducials[0].detach().numpy())
        # print_tre(pred_img, pred_fiducials[0].detach().numpy())

        geo = (
            torch.concat(
                [
                    *self.doublegeo(pose, self.gt_pose),
                    self.geodesics(pose, self.gt_pose),
                ]
            )
            .squeeze()
            .tolist()
        )
        self.geodesic.append(geo)
        self.params.append(param)
        self.losses.append(0)
        self.fiducial.append(tre.item())
        self.times.append(0)
        self.ssims.append(0)
        self.params2.append(param2)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(self.img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"no.{idx}")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(p.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"no.{idx}")
        plt.axis('off')
        plt.savefig(f"reg_result/cma/zyl_stage_xray{idx:03d}.jpg")
        plt.show()

        df = pd.DataFrame(self.params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
        df["losses"] = self.losses
        df[["geo_r", "geo_t", "geo_d", "geo_se3"]] = self.geodesic
        df["fiducial"] = self.fiducial
        df["time"] = self.times
        df["idx"] = idx
        df["parameterization"] = self.parameterization
        df["ssim"] = self.ssims
        df2 = pd.DataFrame(self.params2, columns=["alpha2", "beta2", "gamma2", "bx2", "by2", "bz2"])
        df = pd.concat([df, df2], axis=1)

        return df


def main(id_number, parameterization):
    ckpt = torch.load(f"checkpoints/zyl_800_norm_bone_mask3_pa_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_800_norm_bone_mask3_pa_best2.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_800_norm_bone_mask3_pa_best3.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_800_norm_bone_mask_pa_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_800_norm_bone_mask3_noaug_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_mncc_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/sjj_500_2_best.ckpt", map_location="cuda:1")
    model = PoseRegressor(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = "cuda:0"

    root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    # specimen = IntubationDataset(root, x_root, y_offset=-250, z_cut=600, factors=[3, 0.5, 0.5])
    specimen = IntubationDataset(root, x_root, y_offset=300, z_cut=650, factors=[2, 0.4, 0.4])

    height = ckpt["height"]
    subsample = 512 / height
    delx = specimen.delx * subsample

    drr = DRR(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.sdr,
        height=height,
        delx=delx,
        reverse_x_axis=True,
        bone_attenuation_multiplier=3,
    )

    drr_bone = DRR_Bone(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.sdr,
        height=height,
        delx=delx,
        reverse_x_axis=True,
        bone_attenuation_multiplier=3,
    )

    registration = Registration(
        drr,
        drr_bone,
        specimen,
        model,
        parameterization,
        device=device,
        n_iters=250
    )
    for idx in tqdm(range(9, len(specimen)), ncols=100):
        if idx % 3 != 0:
            continue
        df = registration.run(idx)
        df.to_csv(
            f"runs/cma/zyl_xray{idx:03d}_{parameterization}.csv",
            index=False,
        )


if __name__ == "__main__":
    # seed = 123
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    main(1, "se3_log_map")