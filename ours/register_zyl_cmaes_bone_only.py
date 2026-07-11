
import time

import cv2
import numpy as np
import pandas as pd
import torch
from diffdrr.detector import make_xrays
from diffdrr.siddon import siddon_raycast
from more_itertools.more import run_length

from diffpose.calibration import RigidTransform, convert
from ours.cut.style_to_drr import StyleChanger
from ours.register_zyl_bone_mask_stage import inpaint_with_opencv
from ours.utils.grad_similar import calculate_gradient_consistency_with_mask, gradient_ncc, dice_coefficient_with_mask
from ours.utils.img_utils import print_tre, read_seg
from ours.utils.loss_func import PatchNCE, masked_ssim, masked_ssim2
from ours.utils.test_mask import get_spine_mask
from utils.generate_tube import get_tube_on_image
from utils.drr import DRR
from utils.drr_bone import DRR as DRR_Bone
# from utils.drr_bone import DRR
# from diffpose.metrics import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics2 import MultiscaleNormalizedCrossCorrelation2d
from utils.metrics_mask_tube2_wei2 import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics_mask_tube_weight import MultiscaleNormalizedCrossCorrelation2d
from matplotlib import pyplot as plt
from torchvision.transforms.functional import resize
from tqdm import tqdm

from utils.CT_dataset import Transforms, toZeroOne
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from utils.registration import PoseRegressor
from utils.registration import SparseRegistration, VectorizedNormalizedCrossCorrelation2d
from ours.utils.CT_dataset_PA import IntubationDataset, create_circle_mask
from PIL import Image, ImageSequence
from utils.test_mask import get_tube_mask
import torchvision.transforms.v2 as transforms
import kornia
from skimage.metrics import normalized_mutual_information
import nlopt
import cma
import nibabel as nib
import torch.nn.functional as F

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
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 8], [0.5, 0.5], device=self.device, step=[None, 4])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 32, 8], [0.3, 0.4, 0.3], device=self.device, step=[None, 16, 4])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 13], [0.5, 0.5], device=self.device, step=[None, 1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 21], [0.25, 0.75], device=self.device, step=[None, 1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 40, 9], [0.1, 0.45, 0.45], device=self.device, step=[None, 20, 4])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        self.transforms = Transforms(self.drr.detector.height, radius=119)
        self.parameterization = parameterization
        self.convention = convention

        self.n_iters = n_iters
        self.verbose = verbose

        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_nec5/40_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec5/70_net_G.pth",
        # self.style_change = StyleChanger("cut/ckpt/70_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_new/75_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_new3/70_net_G.pth",
        # self.style_change =  StyleChanger("/media/sda1/PersonalFiles/yx/project/diffpose/ours/cut/drr_style_solid_5_new3/70_net_G.pth",
        self.style_change =  StyleChanger("/media/sda1/PersonalFiles/yx/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_iso/100_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_iso/100_net_G.pth",
        # self.style_change =  StyleChanger("/media/sda1/PersonalFiles/yx/project/diffpose/ours/cut/drr_style_solid_5_new4/90_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_4_choose/100_net_G.pth",
                       device=self.device,
                       resize=256)
        self.times = []
        self.losses = []
        self.i = 0
        self.tres = []

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

            # pose = RigidTransform(rot, xyz, parameterization="so3_log_map", device=self.device)
            pose = RigidTransform(rot, xyz, "euler_angles", "ZYX")
            # pose = pose.compose(self.isocenter_pose)
            pose = self.isocenter_pose.compose(self.back_pose).compose(pose).compose(self.center_pose)

            pred_img = self.drr(None, None, None, pose=pose)
            pred_img = self.transforms(pred_img).to(self.device).to(torch.float32)

            pred_img_bone = self.drr_bone(None, None, None, pose=pose)
            pred_img_bone = self.transforms(pred_img_bone, reverse=False)
            pred_img_bone = (pred_img_bone - pred_img_bone.min()) / (pred_img_bone.max() - pred_img_bone.min())
            pred_img_bone = torch.tanh(50 * pred_img_bone)
            # pred_img_bone[pred_img_bone > 0.01] = 1
            # pred_img_bone[pred_img_bone <= 0.01] = 0
            # wei2 = torch.ones_like(pred_img_bone, requires_grad=False)
            # # wei2[pred_img_bone > 0.1] = 1
            # wei2[pred_img_bone <= 0.1] = 0.2
            # # plt.figure()
            # # plt.imshow(wei2.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
            # # plt.show()

            # plt.figure()
            # plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            # plt.show()
            # plt.figure()
            # plt.imshow(self.img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            # plt.show()
            # ncc = self.criterion(pred_img, self.img)
            weight = 0.5
            dice = dice_coefficient_with_mask(pred_img_bone, self.img_bone, self.total_mask)
            loss = self.criterion(pred_img, self.img_ori) + dice
            # loss = self.criterion(pred_img, self.img_ori)
            # loss = (1 - weight) * gradient_ncc(pred_img, self.img, self.total_mask) + weight * self.criterion(pred_img, self.img)
            # l1 = (1 - weight) * gradient_ncc(pred_img, self.img, self.total_mask)
            # l2 = weight * self.criterion(pred_img, self.img, None, self.wei_img)
            # dice = dice_coefficient_with_mask(pred_img_bone, self.mask_bone, self.total_mask)
            # loss = (1 - weight) * gradient_ncc(pred_img, self.img, self.total_mask) + weight * self.criterion(pred_img, self.img) + weight * dice
            # loss = (1 - weight) * gradient_ncc(pred_img, self.img, self.total_mask) + weight * self.criterion(pred_img, self.img)
            # ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * self.total_mask, toZeroOne(self.img) * self.total_mask, window_size=11, reduction='mean')
            ssim_ori = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * self.total_mask, toZeroOne(self.img_ori) * self.total_mask, window_size=11, reduction='mean')
            # grad_ncc = gradient_ncc(pred_img, self.img, self.total_mask)
            # loss = ncc
            # loss = ssim

            # plt.figure()
            # plt.imshow(self.img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            # plt.show()

            if self.i % 100 == 0:
                pred_img = toZeroOne(pred_img) * pred_img_bone
                plt.figure()
                plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
                plt.show()
                # plt.figure()
                # plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
                # plt.show()

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
                true_fiducials, pred_fiducials = self.specimen.get_3d_fiducials(self.idx, pose)
                tre_3d = torch.norm(true_fiducials - pred_fiducials, dim=2)
                tre_3d = torch.mean(tre_3d)

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
                self.tre.append(tre_3d.item())
                self.times.append(time.time() - start_time)
                # self.ssims.append(ssim.item())
                self.params2.append(param2)
                self.ssimsori.append(ssim_ori.item())
            self.i += 1

        return 1 - loss.item()


    def run(self, idx):
        img, pose = self.specimen[idx]
        self.idx = idx
        self.gt_pose = pose.to(self.device)
        img_rev = self.transforms(img, reverse=True).to(self.device).to(torch.float32)
        tube = torch.ones_like(img_rev).to(self.device)
        tube[toZeroOne(img_rev) > 0.75] = 0
        plt.figure()
        plt.imshow(tube.cpu().squeeze(), cmap="gray")
        plt.show()

        gt_pose = self.specimen.get_manual_gt(idx).to(self.device)
        gt_img = self.drr(None, None, None, pose=gt_pose, bone_attenuation_multiplier=3)
        gt_img = self.transforms(gt_img).to(self.device).to(torch.float32)
        self.gt_img = gt_img
        plt.figure()
        plt.imshow(gt_img.cpu().squeeze(), cmap="gray")
        plt.show()

        bone = self.specimen.get_bone(idx).to(self.device)
        # bone = read_seg("/home/zsr/project/diffpose/ours/bone_seg/zyl_4.nrrd").to(self.device)
        self.img_bone = bone
        img_with_bone = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        img_with_bone = toZeroOne(img_with_bone) * bone
        img_with_bone = self.transforms(img_with_bone, reverse=False).to(self.device).to(torch.float32)


        # plt.figure()
        # plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        # img = get_tube_on_image(img, black=False)
        img = self.transforms(img, reverse=True).to(self.device).to(torch.float32)
        # img = self.transforms(img, reverse=False)
        img_ori = self.transforms(img, reverse=True).to(self.device).to(torch.float32)
        img_ori = torch.tensor(img_ori).to(self.device).to(torch.float32)
        self.img_ori = img_ori
        img_change = self.style_change(img)
        img_change = self.transforms(img_change, reverse=True).to(self.device).to(torch.float32)
        # diff = torch.abs(img - img_change)
        # print(diff.min())
        # print(diff.max())
        # diff = (diff - diff.min()) / (diff.max() - diff.min()).to(self.device)
        # threshold = 0.25
        # diff[diff <= threshold] = 0
        # diff[diff > threshold] = 1
        # diff = 1 - diff
        diff = img - img_change
        diff = (diff - diff.min()) / (diff.max() - diff.min()).to(self.device)
        threshold = 0.23
        diff[diff <= threshold] = 0
        diff[diff > threshold] = 1
        circle_mask = create_circle_mask(256, 116).to(self.device).unsqueeze(0).unsqueeze(0)
        # total_mask = (circle_mask.bool() & diff.bool()).float()
        total_mask = (circle_mask.bool() & tube.bool()).float()
        self.criterion.set_mask(total_mask)
        self.total_mask = total_mask

        plt.figure()
        plt.imshow(total_mask.cpu().squeeze(0).permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
        plt.show()

        plt.figure()
        plt.imshow(img_with_bone.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"NO.{idx}")
        plt.show()
        plt.figure()
        plt.imshow(img_ori.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()

        # self.wei_img = wei_img
        # self.mask_bone = wei_img
        self.params = []
        self.losses = []
        self.geodesic = []
        self.fiducial = []
        self.tre = []
        self.times = []
        # self.ssims = []
        self.params2 = []
        self.ssimsori = []

        initial_pose = self.model(img_with_bone)
        # initial_pose = self.model(bone)
        initial_pose = self.isocenter_pose.compose(initial_pose)
        pred_pose = initial_pose
        initial_pose = self.back_pose.inverse().compose(self.isocenter_pose.inverse()).compose(initial_pose).compose(
            self.center_pose.inverse())
        # initial_pose = initial_pose.compose(self.isocenter_pose.inverse())
        # initial_pose = initial_pose.compose(self.isocenter_pose)

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
        # rot = initial_pose.get_rotation(parameterization="so3_log_map").detach().cpu().numpy()[0]
        rot = initial_pose.get_rotation("euler_angles", "ZYX").detach().cpu().numpy()[0]
        xyz = initial_pose.get_translation().detach().cpu().numpy()[0]
        # rot = gt_pose.get_rotation(parameterization="so3_log_map").detach().cpu().numpy()[0]
        # xyz = gt_pose.get_translation().detach().cpu().numpy()[0]

        # 对于se3对数映射参数化
        dim = 6  # 3旋转 + 3平移
        initial_params = np.hstack([rot, xyz])
        # initial_params = np.array([1.74, -0.8, -0.7, 248, -7.3, 123])

        r_l = 0.02
        t_l = 10
        s = [r_l, r_l, r_l, t_l, t_l, t_l]
        # , 左右,
        # s = [1e-7, 1e-7, 1e-7, 5, 5, 5]
        option = {
            "seed": 333,
            # "CMA_rankmu": 0.2,
            "popsize": 10,
            "CMA_stds": s,
            "maxiter": 80,
            "verb_disp": 1,
            "tolfun": 1e-6,
            "tolx": 1e-6,
            # "CMA_active": True
        }
        r_l = 0.05
        t_l = 10
        s2 = [r_l, r_l, r_l, t_l, t_l, t_l]
        option2 = {
            "seed": 333,
            # "CMA_rankmu": 0.2,
            "popsize": 15,
            "CMA_stds": s2,
            "maxiter": 50,
            "verb_disp": 1,
            "tolfun": 1e-6,
            "tolx": 1e-6,
            # "CMA_active": True
        }

        self.change = False
        optimizer = cma.CMAEvolutionStrategy(x0=initial_params, sigma0=1, options=option)
        optimizer.optimize(self.nlopt_objective)
        self.change = True
        for i in range(0):
            best_solution = optimizer.result.xbest
            best_fvalue = optimizer.result.fbest
            x = torch.tensor(best_solution, dtype=torch.float32, device=self.device, requires_grad=False)
            r = x[:3].unsqueeze(0)
            t = x[3:].unsqueeze(0)
            # pose = RigidTransform(r, t, parameterization="so3_log_map", device=self.device)
            pose = RigidTransform(r, t, "euler_angles", "ZYX")
            pose = self.isocenter_pose.compose(self.back_pose).compose(pose).compose(self.center_pose)
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
        # pose = RigidTransform(r, t, parameterization="so3_log_map", device=self.device)
        pose = RigidTransform(r, t, "euler_angles", "ZYX")
        pose = self.isocenter_pose.compose(self.back_pose).compose(pose).compose(self.center_pose)
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
        true_fiducials, pred_fiducials = self.specimen.get_3d_fiducials(self.idx, pose)
        tre_3d = torch.norm(true_fiducials - pred_fiducials, dim=2)
        tre_3d = torch.mean(tre_3d)

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
        # ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(p) * self.total_mask,
        #                                        toZeroOne(self.img) * self.total_mask, window_size=11, reduction='mean')
        ssim_ori = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(p) * self.total_mask,
                                                   toZeroOne(self.img_ori) * self.total_mask, window_size=11,
                                                   reduction='mean')
        self.geodesic.append(geo)
        self.params.append(param)
        self.losses.append(0)
        self.fiducial.append(tre.item())
        self.tre.append(tre_3d.item())
        self.times.append(0)
        # self.ssims.append(ssim.item())
        self.params2.append(param2)
        self.ssimsori.append(ssim_ori.item())

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(self.img_ori.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"no.{idx}")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(p.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"no.{idx}")
        plt.axis('off')
        # plt.savefig(f"reg_result/cma/norm/zyl_stage_xray{idx:03d}.jpg")
        plt.show()

        df = pd.DataFrame(self.params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
        df["losses"] = self.losses
        df[["geo_r", "geo_t", "geo_d", "geo_se3"]] = self.geodesic
        df["fiducial"] = self.fiducial
        df["tre"] = self.tre
        df["time"] = self.times
        df["idx"] = idx
        df["parameterization"] = self.parameterization
        # df["ssim"] = self.ssims
        df["ssim_ori"] = self.ssimsori
        df2 = pd.DataFrame(self.params2, columns=["alpha2", "beta2", "gamma2", "bx2", "by2", "bz2"])
        df = pd.concat([df, df2], axis=1)

        return df


def main(id_number, parameterization):
    ckpt = torch.load(f"checkpoints/zyl_bone_multi_pa_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_bone_only_pa_best.ckpt", map_location="cuda:1")


    model = PoseRegressor(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = "cuda:0"

    # root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    # x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    specimen = IntubationDataset(root, x_root, y_offset=300, z_cut=650, factors=[2, 0.4, 0.4])
    # specimen = IntubationDataset(root, x_root, y_offset=155, z_cut=600, factors=[3, 0.5, 0.5])
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
    # for idx in tqdm(range(63, 67), ncols=100):
    for idx in tqdm(range(29, len(specimen)), ncols=100):
        # if idx % 3 != 0:
        #     continue
        df = registration.run(idx)
        df.to_csv(
            f"runs/bone/zyl_xray{idx:03d}_{parameterization}.csv",
            # f"runs/norm/zyl_xray{idx:03d}_{parameterization}.csv",
            index=False,
        )


if __name__ == "__main__":
    # seed = 123
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    main(1, "se3_log_map")