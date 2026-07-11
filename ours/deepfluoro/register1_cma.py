import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
# import submitit
import torch

from ours.utils.CT_dataset import toZeroOne
from ours.utils.drr import DRR
from ours.utils.drr_bone import DRR as DRR_Bone
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from matplotlib import pyplot as plt
from pyexpat import features
from torchvision.transforms.functional import resize
from tqdm import tqdm

from diffpose.calibration import RigidTransform, convert
from diffpose.deepfluoro import DeepFluoroDataset, Evaluator, Transforms
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from ours.utils.grad_similar import dice_coefficient_with_mask, gradient_ncc
from ours.utils.registration_bone_mask3 import PoseRegressor
from diffpose.registration import SparseRegistration
# from ours.utils.siddon_registration import SparseRegistration

import kornia
import cma
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

        self.geodesics = GeodesicSE3()
        self.doublegeo = DoubleGeodesic(sdr=self.specimen.focal_len / 2)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 50], [0.7, 0.3], device=self.device)
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 13], [0.5, 0.5])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)

        self.transforms = Transforms(self.drr.detector.height)
        self.parameterization = parameterization
        self.convention = convention

        self.n_iters = n_iters
        self.verbose = verbose
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

            pose = RigidTransform(rot, xyz, parameterization="so3_log_map", device=self.device)
            # pose = pose.compose(self.isocenter_pose)

            pred_img = self.drr(None, None, None, pose=pose)
            pred_img = self.transforms(pred_img).to(self.device).to(torch.float32)

            pred_img_bone = self.drr_bone(None, None, None, pose=pose)
            pred_img_bone = (pred_img_bone - pred_img_bone.min()) / (pred_img_bone.max() - pred_img_bone.min())
            pred_img_bone1 = torch.tanh(30 * pred_img_bone)

            # get_edge(self.img, self.total_mask)
            # wei2 = torch.ones_like(pred_img_bone, requires_grad=False)
            # # wei2[pred_img_bone > 0.1] = 1
            # wei2[pred_img_bone <= 0.1] = 0
            # plt.figure()
            # plt.imshow(pred_img_bone1.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
            # plt.show()

            dice = dice_coefficient_with_mask(pred_img_bone1, self.mask_bone, self.total_mask)
            loss = dice + self.criterion(pred_img, self.img)
            # loss = self.criterion(pred_img, self.img)
            # loss = (1 - weight) * gradient_ncc(pred_img, self.img, self.total_mask) + weight * self.criterion(pred_img, self.img)
            # loss = (1 - weight) * gradient_ncc(pred_img, self.img, self.total_mask) + weight * self.criterion(pred_img, self.img, None, self.wei_img)
            ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * self.total_mask, toZeroOne(self.img) * self.total_mask, window_size=11, reduction='mean')
            # ssim = (1 - weight) * gradient_ncc(pred_img, self.img, self.total_mask) + weight * self.criterion(pred_img, self.img)
            # grad_ncc = gradient_ncc(pred_img, self.img, self.total_mask)
            # loss = ncc
            # loss = ssim

            # plt.figure()
            # plt.imshow(self.img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            # plt.show()

            if self.i % 100 == 0:
                plt.figure()
                plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
                plt.show()

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

                tre = self.target_registration_error(pose.cpu()).item()

                # print_tre(self.gt_img, true_fiducials[0].detach().numpy())
                # print_tre(pred_img, pred_fiducials[0].detach().numpy())

                geo = (
                    torch.concat(
                        [
                            *self.doublegeo(pose, self.pose),
                            self.geodesics(pose, self.pose),
                        ]
                    )
                    .squeeze()
                    .tolist()
                )
                self.geodesic.append(geo)
                self.params.append(param)
                self.losses.append(loss.item())
                self.fiducial.append(tre)
                self.times.append(time.time() - start_time)
                self.ssims.append(ssim.item())
                self.params2.append(param2)
            self.i += 1

        # return 1 - loss.item()
        return -loss.item()


    def run(self, idx):
        img, gt_pose = self.specimen[idx]
        gt_pose = gt_pose.to(self.device)
        plt.figure()
        plt.imshow(img.squeeze(0).permute(1, 2, 0))
        plt.show()
        img = self.transforms(img).to(self.device)
        self.img = img
        self.total_mask = torch.ones_like(img).to(self.device)
        self.idx = idx
        self.pose = gt_pose.to(self.device)
        self.target_registration_error = Evaluator(self.specimen, idx)

        self.params = []
        self.losses = []
        self.geodesic = []
        self.fiducial = []
        self.times = []
        self.ssims = []
        self.params2 = []
        self.ncc = []

        initial_pose, _ = self.model(img)
        initial_pose = self.isocenter_pose.compose(initial_pose)
        # initial_pose = initial_pose.compose(self.isocenter_pose.inverse())
        # initial_pose = initial_pose.compose(self.isocenter_pose)

        mask_bone = F.interpolate(self.model.mask, img.shape[-2:], mode='bilinear')
        mask_bone[mask_bone > 0] = 1
        mask_bone[mask_bone <= 0] = 0
        mask_bone = toZeroOne(mask_bone)
        # plt.figure()
        # plt.imshow(mask_bone.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        self.mask_bone = mask_bone

        # initial_pose = get_random_offset(1, self.device)
        # isocenter_pose = self.specimen.isocenter_pose.to(self.device)
        # back_pose = self.specimen.back_pose.to(self.device)
        # center_pose = self.specimen.center_pose.to(self.device)
        # initial_pose = isocenter_pose.compose(back_pose).compose(initial_pose).compose(center_pose)
        p = self.drr(None, None, None, pose=gt_pose)
        p = self.transforms(p).to(self.device)
        plt.figure()
        plt.imshow(p.detach().cpu().squeeze(0).permute(1, 2, 0))
        plt.show()

        rot = initial_pose.get_rotation(parameterization="so3_log_map").detach().cpu().numpy()[0]
        xyz = initial_pose.get_translation().detach().cpu().numpy()[0]
        # rot = gt_pose.get_rotation(parameterization="so3_log_map").detach().cpu().numpy()[0]
        # xyz = gt_pose.get_translation().detach().cpu().numpy()[0]

        # 对于se3对数映射参数化
        dim = 6  # 3旋转 + 3平移
        initial_params = np.hstack([rot, xyz])
        # initial_params = np.array([1.74, -0.8, -0.7, 248, -7.3, 123])

        x = torch.tensor(initial_params, dtype=torch.float32, device=self.device, requires_grad=False)
        r = x[:3].unsqueeze(0)
        t = x[3:].unsqueeze(0)
        pose = RigidTransform(r, t, parameterization="so3_log_map", device=self.device)
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

        tre = self.target_registration_error(pose.cpu()).item()

        geo = (
            torch.concat(
                [
                    *self.doublegeo(pose, self.pose),
                    self.geodesics(pose, self.pose),
                ]
            )
            .squeeze()
            .tolist()
        )
        self.geodesic.append(geo)
        self.params.append(param)
        self.losses.append(0)
        self.fiducial.append(tre)
        self.times.append(0)
        self.ssims.append(0)
        self.params2.append(param2)

        r_l = 1e-2
        t_l = 1
        s = [r_l, r_l, r_l, t_l, t_l, t_l]
        # , 左右,
        # s = [1e-7, 1e-7, 1e-7, 5, 5, 5]
        option = {
            # "seed": 333,
            # "CMA_rankmu": 0.2,
            "popsize": 10,
            "CMA_stds": s,
            "maxiter": 80,
            "verb_disp": 1,
            "tolfun": 1e-6,
            "tolx": 1e-6,
            # "CMA_active": True
        }

        optimizer = cma.CMAEvolutionStrategy(x0=initial_params, sigma0=1, options=option)
        optimizer.optimize(self.nlopt_objective)

        for i in range(0):
            best_solution = optimizer.result.xbest
            best_fvalue = optimizer.result.fbest

            print(f"重启前第{i + 1}轮最佳值: {best_fvalue}")
            optimizer = cma.CMAEvolutionStrategy(x0=best_solution, sigma0=1, options=option)
            # 再次运行优化
            optimizer.optimize(self.nlopt_objective)

        # 5. 输出结果
        print("一阶段最优解:", optimizer.result.xbest)
        print("一阶段目标值:", optimizer.result.fbest)

        x = optimizer.result.xbest
        x = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=False)
        r = x[:3].unsqueeze(0)
        t = x[3:].unsqueeze(0)
        pose = RigidTransform(r, t, parameterization="so3_log_map", device=self.device)
        p = self.drr(None, None, None, pose=pose)
        p = self.transforms(p).to(self.device)
        # plt.figure()
        # plt.imshow(p.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()

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

        tre = self.target_registration_error(pose.cpu()).item()

        geo = (
            torch.concat(
                [
                    *self.doublegeo(pose, self.pose),
                    self.geodesics(pose, self.pose),
                ]
            )
            .squeeze()
            .tolist()
        )
        self.geodesic.append(geo)
        self.params.append(param)
        self.losses.append(0)
        self.fiducial.append(tre)
        self.times.append(0)
        self.ssims.append(0)
        self.params2.append(param2)

        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(self.img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.title(f"no.{idx}")
        # plt.axis('off')
        # plt.subplot(1, 2, 2)
        # plt.imshow(p.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.title(f"no.{idx}")
        # plt.axis('off')
        # # plt.savefig(f"reg_result/cma/zyl_stage_xray{idx:03d}.jpg")
        # plt.show()

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
    ckpt = torch.load(f"checkpoints/specimen_{id_number:02d}_bone_best.ckpt", map_location="cuda:1")
    model = PoseRegressor(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
    )
    model.load_state_dict(ckpt["model_state_dict"])

    # specimen = DeepFluoroDataset(id_number, filename="/media/sda1/PersonalFiles/yx/project/diffpose/diffpose/data/ipcai_2020_full_res_data.h5")
    specimen = DeepFluoroDataset(id_number, filename = "/home/zsr/project/diffpose/data/ipcai_2020_full_res_data.h5")
    height = ckpt["height"]
    subsample = (1536 - 100) / height
    delx = 0.194 * subsample

    drr = DRR(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.focal_len / 2,
        height=height,
        delx=delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        bone_attenuation_multiplier=2.5,
    )
    drr_bone = DRR_Bone(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.focal_len / 2,
        height=height,
        delx=delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        bone_attenuation_multiplier=2.5,
    )

    registration = Registration(
        drr,
        drr_bone,
        specimen,
        model,
        parameterization,
        device="cuda:1",
        n_iters=250
    )
    for idx in tqdm(range(69, len(specimen)), ncols=100):
        # if idx % 5 != 3:
        #     continue
        df = registration.run(idx)
        df.to_csv(
            f"runs/1/cma/specimen_{idx:03d}_{parameterization}.csv",
            # f"runs/1/cma/nodice_specimen_{idx:03d}_{parameterization}.csv",
            index=False,
        )


if __name__ == "__main__":
    # seed = 123
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    main(1, "se3_log_map")