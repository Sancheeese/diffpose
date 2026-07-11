
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
from ours.utils.grad_similar import calculate_gradient_consistency_with_mask, gradient_ncc
from ours.utils.loss_func import PatchNCE, masked_ssim, masked_ssim2
from ours.utils.test_mask import get_spine_mask
from utils.generate_tube import get_tube_on_image
from utils.drr import DRR
# from utils.drr_bone import DRR
# from diffpose.metrics import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics2 import MultiscaleNormalizedCrossCorrelation2d
from utils.metrics_mask_tube2 import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics_mask_tube_weight import MultiscaleNormalizedCrossCorrelation2d
from matplotlib import pyplot as plt
from torchvision.transforms.functional import resize
from tqdm import tqdm

from utils.CT_dataset import Transforms, toZeroOne
from diffpose.deepfluoro import DeepFluoroDataset, Evaluator
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from utils.registration_bone_mask3 import PoseRegressor
from utils.registration import SparseRegistration, VectorizedNormalizedCrossCorrelation2d
from ours.utils.CT_dataset import IntubationDataset, create_circle_mask
from PIL import Image, ImageSequence
from utils.test_mask import get_tube_mask
import torchvision.transforms.v2 as transforms
import kornia
from skimage.metrics import normalized_mutual_information
import nlopt
import cma
import nibabel as nib

class Registration:
    def __init__(
        self,
        drr,
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
        self.model = model.to(self.device)
        model.eval()

        self.specimen = specimen
        self.isocenter_pose = specimen.isocenter_pose.to(self.device)

        self.geodesics = GeodesicSE3()
        self.doublegeo = DoubleGeodesic(sdr=self.specimen.sdr)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 50], [0.7, 0.3], device=self.device)
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 40, 9], [0.1, 0.45, 0.45], device=self.device, step=[None, 20, 4])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        self.transforms = Transforms(self.drr.detector.height)
        self.parameterization = parameterization
        self.convention = convention

        self.n_iters = n_iters
        self.verbose = verbose

        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_nec5/40_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec5/70_net_G.pth",
        self.style_change = StyleChanger("cut/ckpt/70_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_4_choose/100_net_G.pth",
                       device=self.device,
                       resize=256)
        self.times = []
        self.losses = []
        self.i = 0

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
        """NLopt优化目标函数"""
        # 将numpy数组转换为torch tensor
        start_time = time.time()
        # x = self.denormalize(x)
        x = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=True)

        # 更新registration的位姿参数
        rot = x[:3].unsqueeze(0)
        xyz = x[3:].unsqueeze(0)

        pose = RigidTransform(rot, xyz, parameterization="so3_log_map", device=self.device)

        pred_img = self.drr(None, None, None, pose=pose)
        pred_img = self.transforms(pred_img).to(self.device).to(torch.float32)
        # plt.figure()
        # plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        # plt.figure()
        # plt.imshow(self.img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        # ncc = self.criterion(pred_img, self.img)
        weight = 0.5
        loss = (1 - weight) * gradient_ncc(pred_img, self.img, self.total_mask) + weight * self.criterion(pred_img, self.img)
        # ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * self.total_mask, toZeroOne(self.img) * self.total_mask, window_size=11, reduction='mean')
        # grad_ncc = gradient_ncc(pred_img, self.img, self.total_mask)
        # loss = ncc
        # loss = ssim
        self.losses.append(loss)

        x.grad = None
        loss.backward()

        if self.i % 50 == 0:
            plt.figure()
            plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            plt.show()
        self.i += 1

        del pred_img
        torch.cuda.empty_cache()

        end_time = time.time()
        self.times.append(end_time - start_time)

        return 1 - loss.item()


    def run(self, idx):
        # # idx = 3
        # img, pose = self.specimen[idx]
        # # tube_mask = get_tube_mask("/home/zsr/project/diffpose/ours/seg",
        # #                           self.specimen.get_x_filename(idx))
        # # tube_mask = torch.tensor(tube_mask, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        # # tube_mask = transforms.Resize(256)(tube_mask)
        # # tube_mask[tube_mask < 0.5] = 0
        # # tube_mask[tube_mask >= 0.5] = 1
        # # self.criterion.set_mask(tube_mask)
        #
        # # spine_mask = get_spine_mask("/home/zsr/project/diffpose/ours/seg",
        # #                           self.specimen.get_x_filename(idx))
        # # spine_mask = torch.tensor(spine_mask, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        # # spine_mask = transforms.Resize(256)(spine_mask)
        # # spine_mask[spine_mask >= 0.5] = 1
        # # spine_mask[spine_mask < 0.5] = 5
        # # plt.figure()
        # # plt.imshow(spine_mask.cpu().squeeze(), cmap="gray")
        # # plt.show()
        #
        # gt_pose = self.specimen.get_manual_gt().to(self.device)
        # gt_img = self.drr(None, None, None, pose=gt_pose, bone_attenuation_multiplier=3)
        # gt_img = self.transforms(gt_img).to(self.device).to(torch.float32)
        # plt.figure()
        # plt.imshow(gt_img.cpu().squeeze(), cmap="gray")
        # plt.show()
        #
        # # plt.figure()
        # # plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # # plt.show()
        # # img = get_tube_on_image(img, black=False)
        # img = self.transforms(img, reverse=False).to(self.device)
        # # img = self.transforms(img, reverse=False)
        # img_ori = torch.tensor(img).to(self.device).to(torch.float32)
        # img_change = self.style_change(img)
        # img_change = self.transforms(img_change, reverse=False).to(self.device).to(torch.float32)
        # diff = img - img_change
        # diff = (diff - diff.min()) / (diff.max() - diff.min()).to(self.device)
        # threshold = 0.55
        # diff[diff <= threshold] = 0
        # diff[diff > threshold] = 1
        # circle_mask = create_circle_mask(256, 120).to(self.device).unsqueeze(0).unsqueeze(0)
        # total_mask = (circle_mask.bool() & diff.bool()).float()
        # # total_mask = (circle_mask.bool() & diff.bool()).float()
        # self.criterion.set_mask(total_mask)
        # self.total_mask = total_mask
        # # self.criterion.set_mask(circle_mask)
        # # self.criterion.set_weight_mask(spine_mask)
        #
        # plt.figure()
        # plt.imshow(total_mask.cpu().squeeze(0).permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
        # plt.show()
        # img = img_change
        # img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        # self.pose = pose.to(self.device)
        # self.img = img
        # self.img_ori = img_ori
        #
        # plt.figure()
        # plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        # filename = str(time.time())
        # # img_save = img.detach().clone()
        # # img_save = img_save.cpu().squeeze(0).squeeze(0)
        # # img_save = np.array(((img_save - img_save.min()) / (img_save.max() - img_save.min())) * 255).astype(np.uint8)
        # # cv2.imwrite(f"test_img/{filename}.png", img_save)




        img, pose = self.specimen[idx]

        filename = self.specimen.get_x_filename(idx).split(".")[0] + "_nochange"
        file_path = f"/media/sda1/PersonalFiles/yx/dataset/zyl_result/{filename}.nii.gz"
        # file_path = f"nnuet/zyl_result/{filename}.nii.gz"
        nii_img = nib.load(file_path)
        img_data = nii_img.get_fdata()
        # img_data = img_data[::-1, :].copy()
        bone_mask_gt = torch.tensor(img_data, device=self.device)
        bone_mask_gt = torch.tensor(bone_mask_gt, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        bone_mask_gt = transforms.Resize(256)(bone_mask_gt)
        bone_mask_gt[bone_mask_gt >= 0.5] = 1
        bone_mask_gt[bone_mask_gt < 0.5] = 0
        # plt.figure()
        # plt.imshow(bone_mask_gt.cpu().squeeze(), cmap="gray")
        # plt.show()

        gt_pose = self.specimen.get_manual_gt().to(self.device)
        gt_img = self.drr(None, None, None, pose=gt_pose, bone_attenuation_multiplier=3)
        gt_img = self.transforms(gt_img).to(self.device).to(torch.float32)
        plt.figure()
        plt.imshow(gt_img.cpu().squeeze(), cmap="gray")
        plt.show()

        # plt.figure()
        # plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        # img = get_tube_on_image(img, black=False)
        img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        # img = self.transforms(img, reverse=False)
        img_ori = torch.tensor(img).to(self.device).to(torch.float32)
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
        diff = img - img_change
        diff = (diff - diff.min()) / (diff.max() - diff.min()).to(self.device)
        threshold = 0.55
        diff[diff <= threshold] = 0
        diff[diff > threshold] = 1
        circle_mask = create_circle_mask(256, 120).to(self.device).unsqueeze(0).unsqueeze(0)
        total_mask = (circle_mask.bool() & diff.bool()).float()
        self.total_mask = total_mask
        self.criterion.set_mask(total_mask)
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
        img_input = torch.pow(img, 1.5)
        img_input = inpaint_with_opencv(img_input, black)
        img_input = self.transforms(img_input, reverse=False).to(self.device).to(torch.float32)
        img = inpaint_with_opencv(img, black)
        # img = toZeroOne(gt_img)
        # img = torch.pow(img, 1.5)
        # img = transforms_aug(img)
        img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        self.pose = pose.to(self.device)
        img = img_input
        self.img = img
        plt.figure()
        plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"NO.{idx}")
        plt.show()

        initial_pose, _ = self.model(img)
        initial_pose = self.isocenter_pose.compose(initial_pose)
        img = self.drr(None, None, None, pose=initial_pose, bone_attenuation_multiplier=3)
        img = self.transforms(img).to(self.device)
        plt.figure()
        plt.imshow(img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()

        # self.target_registration_error = Evaluator(self.specimen, idx)
        rot = initial_pose.get_rotation(parameterization="so3_log_map").detach().cpu().numpy()[0]
        xyz = initial_pose.get_translation().detach().cpu().numpy()[0]
        # rot = gt_pose.get_rotation(parameterization="so3_log_map").detach().cpu().numpy()[0]
        # xyz = gt_pose.get_translation().detach().cpu().numpy()[0]

        # 对于se3对数映射参数化
        dim = 6  # 3旋转 + 3平移
        initial_params = np.hstack([rot, xyz])
        # initial_params = self.normalize(initial_params)
        # lb = np.array([-np.inf] * 6)  # 下界
        # ub = np.array([np.inf] * 6)  # 上界
        lb = np.array([-np.pi, -np.pi, -np.pi, -2000.0, -2000.0, -2000.0])  # SE(3)下界
        ub = np.array([np.pi, np.pi, np.pi, 2000.0, 2000.0, 2000.0])

        r_l = 1e-2
        t_l = 30
        s = [r_l, r_l, r_l, t_l, t_l, t_l]
        # , 左右,
        # s = [1e-7, 1e-7, 1e-7, 5, 5, 5]
        option = {
            "popsize": 10,
            "CMA_stds": s,
            "maxiter": 300,
            "verb_disp": 1,
            # "CSA_dampfac": 2
        }
        optimizer = cma.CMAEvolutionStrategy(x0=initial_params, sigma0=1, options=option)
        optimizer.optimize(self.nlopt_objective)

        # 5. 输出结果
        print("最优解:", optimizer.result.xbest)
        print("目标值:", optimizer.result.fbest)

        # 6. 可视化收敛过程
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(optimizer.result.fbest_history, 'b-', label="最佳目标值")
        plt.xlabel("迭代次数")
        plt.ylabel("目标函数值")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(optimizer.result.xbest_history[:, 0], label="x1")
        plt.plot(optimizer.result.xbest_history[:, 1], label="x2")
        plt.xlabel("迭代次数")
        plt.ylabel("变量值")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return None


def main(id_number, parameterization):
    # ckpt = torch.load(f"checkpoints/zyl_800_norm_best_afe.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_epoch050.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_unknow.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_tube_no_change_best2.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_tube_no_change_best_shallow.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_800_norm_best.ckpt", map_location="cuda:1")
    ckpt = torch.load(f"checkpoints/zyl_800_norm_bone_mask3_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_mncc_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/sjj_500_2_best.ckpt", map_location="cuda:1")
    model = PoseRegressor(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = "cuda:2"

    # root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    # x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    specimen = IntubationDataset(root, x_root, y_offset=155, z_cut=600, factors=[3, 0.5, 0.5])
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

    registration = Registration(
        drr,
        specimen,
        model,
        parameterization,
        device=device,
        n_iters=250
    )
    for idx in tqdm(range(77, len(specimen)), ncols=100):
        df = registration.run(3)
        df.to_csv(
            f"runs/zyl_xray{idx:03d}_{parameterization}.csv",
            index=False,
        )


if __name__ == "__main__":
    # seed = 123
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    main(1, "se3_log_map")