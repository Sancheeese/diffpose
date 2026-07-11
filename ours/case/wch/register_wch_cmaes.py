
import time

import cv2
import numpy as np
import pandas as pd
import torch

from diffpose.calibration import RigidTransform, convert
from ours.cut.style_to_drr import StyleChanger
from ours.utils.grad_similar import calculate_gradient_consistency_with_mask, gradient_ncc, dice_coefficient_with_mask, \
    get_edge, multiscale_gradient_ncc
from ours.utils.img_utils import see_mid, rank_transform_tensor, overlay_grayscale_with_red_tensor, print_tre, \
    overlay_grayscale_with_red_tensor_save

from ours.utils.drr import DRR
from ours.utils.drr_bone import DRR as DRR_Bone
# from utils.drr_bone import DRR
# from diffpose.metrics import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics2 import MultiscaleNormalizedCrossCorrelation2d
from ours.utils.metrics_mask_tube2_wei2 import MultiscaleNormalizedCrossCorrelation2d as MultiscaleNormalizedCrossCorrelation2dNowei
from ours.utils.metrics_mask_tube2_weiwei import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics_mask_tube2 import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics2 import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics_mask_tube_weight import MultiscaleNormalizedCrossCorrelation2d
from matplotlib import pyplot as plt
from torchvision.transforms.functional import resize
from tqdm import tqdm

from ours.utils.CT_dataset import Transforms, toZeroOne
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from ours.utils.model2 import PoseRegressorCoeSpDeco, PoseRegressorAddSpDeco
from ours.utils.registration_bone_mask3 import PoseRegressor
from ours.utils.registration import SparseRegistration, VectorizedNormalizedCrossCorrelation2d
from ours.utils.CT_dataset_PA import create_circle_mask
from CT_dataset import IntubationDataset
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
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 21], [0.5, 0.5], device=self.device, step=[None, 1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 21], [0.5, 0.5], device=self.device, step=[None, 1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([21], [1], device=self.device, step=[1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        self.criterion_no_wei = MultiscaleNormalizedCrossCorrelation2dNowei([21], [1], device=self.device, step=[1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        # self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([32], [1], device=self.device, step=[1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
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

        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_nec5/50_net_G.pth",
        self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec5/40_net_G.pth",
        # self.style_change = StyleChanger("cut/ckpt/70_net_G.pth",
        # self.style_change = StyleChanger("/media/sda1/PersonalFiles/yx/project/diffpose/ours/cut/drr_style_solid_5_white/70_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_all/55_net_G.pth",
        # self.style_change =  StyleChanger("/media/sda1/PersonalFiles/yx/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_iso/100_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_iso/100_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_new3/70_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_white/30_net_G.pth",
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

    def initialize_registration(self, img, mask):
        with torch.no_grad():
            offset, pred_mask = self.model(img, mask)
            features = None
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
            # pose = self.isocenter_pose.compose(pose)
            pose = self.isocenter_pose.compose(self.back_pose).compose(pose).compose(self.center_pose)

            pred_img = self.drr(None, None, None, pose=pose)
            pred_img = self.transforms(pred_img).to(self.device).to(torch.float32)
            # pred_img = torch.pow(toZeroOne(pred_img), 0.7)
            # pred_img = self.transforms(pred_img, reverse=False).to(self.device).to(torch.float32)

            pred_img_bone = self.drr_bone(None, None, None, pose=pose)
            pred_img_bone = self.transforms(pred_img_bone, reverse=False)
            pred_img_bone = (pred_img_bone - pred_img_bone.min()) / (pred_img_bone.max() - pred_img_bone.min())
            pred_img_bone1 = torch.tanh(50 * pred_img_bone)
            pred_img_bone2 = torch.tanh(1 * pred_img_bone)

            dice = dice_coefficient_with_mask(pred_img_bone1, self.img_bone, self.total_mask)
            loss = self.criterion(pred_img, self.img_ori) + dice
            # loss = self.criterion_no_wei(pred_img, self.img_ori) + dice
            # grad_ncc = gradient_ncc(pred_img, self.img_ori, self.total_mask)
            # loss = dice + grad_ncc

            # ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * self.total_mask, toZeroOne(self.img) * self.total_mask, window_size=11, reduction='mean')
            ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * self.total_mask, toZeroOne(self.img) * self.total_mask, window_size=11, reduction='mean')
            ssim_ori = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * self.total_mask, toZeroOne(self.img_ori) * self.total_mask, window_size=11, reduction='mean')
            # ssim = (1 - weight) * gradient_ncc(pred_img, self.img, self.total_mask) + weight * self.criterion(pred_img, self.img)
            # grad_ncc = gradient_ncc(pred_img, self.img, self.total_mask)
            # loss = ncc
            # loss = ssim
            # loss = dice + ssim

            # plt.figure()
            # plt.imshow(self.img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            # plt.show()

            # if self.i % 100 == 0:
            #     plt.figure()
            #     plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            #     plt.show()
                # plt.figure()
                # plt.imshow(pred_img_bone1.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
                # plt.show()
                # see_mid(pred_img, self.total_mask)

            if self.i % 15 == 0:
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
                mpd = torch.norm(true_fiducials - pred_fiducials, dim=2)
                mpd = torch.mean(mpd)

                # print_tre(self.gt_img, true_fiducials[0].detach().numpy())
                # print_tre(pred_img, pred_fiducials[0].detach().numpy())

                # true_fiducials, pred_fiducials = self.specimen.get_3d_fiducials(self.idx, pose)
                # mtre = torch.norm(true_fiducials - pred_fiducials, dim=2)
                mtre = self.specimen.calc_tre(self.idx, pose)

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
                self.fiducial.append(mpd.item())
                self.tre.append(mtre.item())
                self.times.append(time.time() - start_time)
                self.ssims.append(ssim.item())
                self.params2.append(param2)
                self.ssimsori.append(ssim_ori.item())
            self.i += 1

        # return 1 - loss.item()
        return -loss.item()


    def run(self, idx):
        img, pose = self.specimen[idx]
        self.idx = idx
        pose = pose.to(self.device)
        self.gt_pose = pose.to(self.device)

        img_rev = self.transforms(img, reverse=True).to(self.device).to(torch.float32)
        tube = torch.ones_like(img_rev).to(self.device)
        tube[toZeroOne(img_rev) > 0.75] = 0
        plt.figure()
        plt.imshow(tube.cpu().squeeze(), cmap="gray")
        plt.show()

        bone = self.specimen.get_bone(idx).to(self.device).to(torch.float32)
        # bone = read_seg("/home/zsr/project/diffpose/ours/bone_seg/wch_4.nrrd").to(self.device)
        pred_img_bone = self.drr_bone(None, None, None, pose=pose)
        pred_img_bone = self.transforms(pred_img_bone, reverse=False).to(torch.float32)
        pred_img_bone = (pred_img_bone - pred_img_bone.min()) / (pred_img_bone.max() - pred_img_bone.min())
        pred_img_bone = torch.tanh(50 * pred_img_bone)
        # bone = pred_img_bone

        self.img_bone = bone

        gt_pose = self.specimen.get_manual_gt(idx).to(self.device)
        gt_img = self.drr(None, None, None, pose=gt_pose)
        gt_img = self.transforms(gt_img, reverse=True).to(self.device).to(torch.float32)
        self.gt_img = gt_img

        plt.figure()
        plt.imshow(gt_img.cpu().squeeze(), cmap="gray")
        plt.show()
        img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        # img = self.transforms(img, reverse=False)
        img_ori = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        img_ori = torch.tensor(img_ori).to(self.device).to(torch.float32)
        self.img_ori = img_ori

        img_change = self.style_change(img)
        img_change = self.transforms(img_change, reverse=False).to(self.device).to(torch.float32)
        circle_mask = create_circle_mask(256, 116).to(self.device).unsqueeze(0).unsqueeze(0)
        # total_mask = (circle_mask.bool() & diff.bool()).float()
        total_mask = (circle_mask.bool() & tube.bool()).float()

        plt.figure()
        plt.imshow(total_mask.cpu().squeeze(0).permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
        plt.show()
        img = img_change
        self.img = img

        # img_ori = inpaint_with_opencv(toZeroOne(img_ori), 1 - tube)
        # img_ori = self.transforms(img_ori, reverse=False)

        # img = rank_transform_tensor(img)
        plt.figure(dpi=300)
        plt.imshow(img_ori.cpu().squeeze(), cmap="gray")
        plt.axis('off')
        plt.savefig("img_ori.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.figure(dpi=300)
        plt.imshow(img.cpu().squeeze(), cmap="gray")
        plt.axis('off')
        plt.show()

        plt.figure()
        plt.imshow(total_mask.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()
        self.total_mask = total_mask
        # self.criterion.set_mask(white_mask)
        self.criterion.set_wei(bone)
        self.criterion.set_mask(total_mask)
        self.criterion_no_wei.set_mask(total_mask)
        # self.criterion2.set_mask(total_mask)

        self.params = []
        self.losses = []
        self.geodesic = []
        self.fiducial = []
        self.tre = []
        self.times = []
        self.ssims = []
        self.params2 = []
        self.ncc = []
        self.ssimsori = []

        # a = torch.zeros_like(img).to(self.device).to(torch.float32)
        initial_pose, _ = self.model(img, bone)
        for layer_name, feat in self.model.features.items():
            feat = feat.mean(dim=1)
            feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
            plt.figure()
            plt.imshow(feat.cpu().permute(1, 2, 0))
            plt.title(layer_name)
            plt.show()

        initial_pose = self.isocenter_pose.compose(initial_pose)
        pred_pose = initial_pose
        initial_pose = self.back_pose.inverse().compose(self.isocenter_pose.inverse()).compose(initial_pose).compose(self.center_pose.inverse())
        # initial_pose = initial_pose.compose(self.isocenter_pose.inverse())

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

        # true_fiducials, pred_fiducials = self.specimen.get_2d_fiducials(self.idx, pred_pose)
        # tre = torch.norm(true_fiducials - pred_fiducials, dim=2)
        # tre = torch.mean(tre)
        #
        # print_tre(self.gt_img, true_fiducials[0].detach().numpy())
        # print_tre(p, pred_fiducials[0].detach().numpy())

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

        r_l = 0.08
        t_l = 10
        s = [r_l, r_l, r_l, t_l, t_l, t_l]
        # , 左右,
        # s = [1e-7, 1e-7, 1e-7, 5, 5, 5]
        seed = 3
        option = {
            "seed": seed,
            # "CMA_rankmu": 0.2,
            "popsize": 20,
            "CMA_stds": s,
            "maxiter": 60,
            "verb_disp": 1,
            "tolfun": 1e-6,
            "tolx": 1e-6,
            # "CMA_active": True
        }
        r_l = 0.02
        t_l = 10
        s2 = [r_l, r_l, r_l, t_l, t_l, t_l]
        option2 = {
            "seed": seed,
            # "CMA_rankmu": 0.2,
            "popsize": 10,
            "CMA_stds": s2,
            "maxiter": 50,
            "verb_disp": 1,
            "tolfun": 1e-6,
            "tolx": 1e-6,
            # "CMA_active": True
        }

        l1 = gradient_ncc(gt_img, self.img, self.total_mask)
        l2 = self.criterion(gt_img, self.img)

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
        plt.figure(dpi=300)
        plt.imshow(img_ori.cpu().squeeze(), cmap="gray")
        plt.axis('off')
        plt.savefig(f"./ret/img_ori_{idx}.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        p = self.drr(None, None, None, pose=pose, bone_attenuation_multiplier=3)
        p = self.transforms(p).to(self.device)
        plt.figure(dpi=300)
        plt.imshow(p.detach().cpu().squeeze(), cmap="gray")
        plt.axis('off')
        plt.savefig(f"./ret/pred_img_{idx}.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        pred_img_bone = self.drr_bone(None, None, None, pose=pose)
        pred_img_bone = self.transforms(pred_img_bone, reverse=False)
        pred_img_bone = (pred_img_bone - pred_img_bone.min()) / (pred_img_bone.max() - pred_img_bone.min())
        pred_img_bone = torch.tanh(1 * pred_img_bone)
        circle_mask = create_circle_mask(256, 118).to(self.device).unsqueeze(0).unsqueeze(0)
        edge = get_edge(pred_img_bone, circle_mask)
        overlay_grayscale_with_red_tensor_save(img_ori, edge, f"./ret/overlay_{idx}.png", alpha=1)

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
        mpd = torch.norm(true_fiducials - pred_fiducials, dim=2)
        mpd = torch.mean(mpd)

        # print_tre(self.gt_img, true_fiducials[0].detach().numpy())
        # print_tre(pred_img, pred_fiducials[0].detach().numpy())

        mtre = self.specimen.calc_tre(idx, pose)

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
        ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(p) * self.total_mask,
                                               toZeroOne(self.img) * self.total_mask, window_size=11, reduction='mean')
        ssim_ori = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(p) * self.total_mask,
                                                   toZeroOne(self.img_ori) * self.total_mask, window_size=11,
                                                   reduction='mean')
        self.geodesic.append(geo)
        self.params.append(param)
        self.losses.append(0)
        self.fiducial.append(mpd.item())
        self.tre.append(mtre.item())
        self.times.append(0)
        self.ssims.append(ssim.item())
        self.params2.append(param2)
        self.ssimsori.append(ssim_ori.item())

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(self.img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"no.{idx}")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(p.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"no.{idx}")
        plt.axis('off')
        plt.show()

        df = pd.DataFrame(self.params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
        df["losses"] = self.losses
        df[["geo_r", "geo_t", "geo_d", "geo_se3"]] = self.geodesic
        df["fiducial"] = self.fiducial
        df["tre"] = self.tre
        df["time"] = self.times
        df["idx"] = idx
        df["parameterization"] = self.parameterization
        df["ssim"] = self.ssims
        df["ssim_ori"] = self.ssimsori
        df2 = pd.DataFrame(self.params2, columns=["alpha2", "beta2", "gamma2", "bx2", "by2", "bz2"])
        df = pd.concat([df, df2], axis=1)

        return df


def main(id_number, parameterization):
    # ckpt = torch.load(f"checkpoints/wch_coespdeco_en_best.ckpt", map_location="cuda:0")
    ckpt = torch.load(f"checkpoints/wch_addspdeco_best.ckpt", map_location="cuda:0")
    # ckpt = torch.load(f"checkpoints/wch_addspdeco_epoch300.ckpt", map_location="cuda:0")
    model = PoseRegressorAddSpDeco(
    # model = PoseRegressorCoeSpDeco(
        # model = PoseRegressorAttnWei(
        # model = PoseRegressorAttn(
        # model = PoseRegressorCatCBAM(
        # model = PoseRegressorCat(
        # model = PoseRegressorAttnNoWei(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = "cuda:0"

    root = "/home/zsr/project/diffpose/ours/data/liwei/邬春花/CT/WuChunHua/20240708003323.343000/3"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/邬春花/ERCP/WU^CHUNHUA^/20240710150034/1"
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/邬春花/CT/WuChunHua/20240708003323.343000/3"
    # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/邬春花/ERCP/WU^CHUNHUA^/20240710150034/1"

    specimen = IntubationDataset(root, x_root, y_offset=100, z_cut=180, factors=[0.7, 0.7, 1])

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
    for idx in tqdm(range(6, len(specimen)), ncols=100):
    # for idx in tqdm(range(49, len(specimen)), ncols=100):
    # for idx in tqdm(range(78, len(specimen)), ncols=100):
    # for idx in tqdm(range(84, len(specimen)), ncols=100):
        # if idx % 3 != 0:
        #     continue
        df = registration.run(idx)
        df.to_csv(
            f"runs/mask/cma/wch_xray{idx:03d}_{parameterization}.csv",
            # f"runs/cma/orinodicewch_xray{idx:03d}_{parameterization}.csv",
            index=False,
        )

def inpaint_with_opencv(img_tensor, mask_tensor, method=cv2.INPAINT_NS):
    """
    使用 OpenCV 进行图像修复。

    参数:
        img_tensor: (C, H, W) 范围 [0, 1] 的 PyTorch 张量。
        mask_tensor: (H, W) 的 PyTorch 张量，需要修复的区域为 1。
        method: cv2.INPAINT_TELEA 或 cv2.INPAINT_NS。

    返回:
        修复后的 PyTorch 张量。
    """
    # 1. 将 PyTorch Tensor 转换为 OpenCV 需要的 NumPy 格式
    # PyTorch: (C, H, W) -> NumPy: (H, W, C), 范围 [0, 255]
    img_tensor = toZeroOne(img_tensor)
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
    img_np = img_np.astype(np.uint8)

    # 2. 处理掩码
    mask_np = mask_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255 # 范围 0 或 255

    # 3. 调用 OpenCV 的修复函数
    result_np = cv2.inpaint(img_np, mask_np, inpaintRadius=20, flags=method)

    # 4. 将结果转换回 PyTorch Tensor
    result_tensor = torch.from_numpy(result_np.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    result_tensor = result_tensor.to(img_tensor.device)

    return result_tensor


if __name__ == "__main__":
    # seed = 123
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    main(1, "se3_log_map")