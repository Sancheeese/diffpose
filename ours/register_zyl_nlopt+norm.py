import time

import cv2
import nlopt
import numpy as np
import pandas as pd
import torch
from skimage.metrics.tests.test_simple_metrics import sigma
from torch.nn.functional import binary_cross_entropy

from diffpose.calibration import RigidTransform, convert
from ours.cut.style_to_drr import StyleChanger
from ours.my_util2 import get_random_offset
from ours.utils.grad_similar import calculate_gradient_consistency_with_mask, gradient_ncc
from ours.utils.loss_func import PatchNCE, masked_ssim, masked_ssim2
from ours.utils.test_mask import get_spine_mask
from utils.generate_tube import get_tube_on_image
from utils.drr import DRR
from utils.drr_bone import DRR as DRR_Bone
# from diffpose.metrics import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics2 import MultiscaleNormalizedCrossCorrelation2d
from utils.metrics_mask_tube2 import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics_mask_tube_weight import MultiscaleNormalizedCrossCorrelation2d
from matplotlib import pyplot as plt
from torchvision.transforms.functional import resize
from tqdm import tqdm

from dataset.CT_dataset import Transforms
from diffpose.deepfluoro import DeepFluoroDataset, Evaluator
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from utils.registration import PoseRegressor, SparseRegistration, VectorizedNormalizedCrossCorrelation2d
from ours.utils.CT_dataset import IntubationDataset, create_circle_mask
from PIL import Image, ImageSequence
from utils.test_mask import get_tube_mask
import torchvision.transforms.v2 as transforms
import kornia
from skimage.metrics import normalized_mutual_information


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
        self.doublegeo = DoubleGeodesic(sdr=self.specimen.sdr)
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 30, 9], [0.45, 0.45, 0.1], device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 100, 13], [0.3, 0.4, 0.3], device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 30, 7], [0.3, 0.3, 0.4], device=self.device)

        self.transforms = Transforms(self.drr.detector.height)
        self.parameterization = parameterization
        self.convention = convention

        self.n_iters = n_iters
        self.verbose = verbose

        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_nec5/40_net_G.pth",
        self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec5/70_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_4_choose/100_net_G.pth",
                       device=self.device,
                       resize=256)
        self.times = []
        self.losses = []

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
        optimizer = torch.optim.Adam(
            [
                # {"params": [registration.rotation], "lr": 7.5e-3},
                {"params": [registration.rotation], "lr": 1.5e-2},
                {"params": [registration.translation], "lr": 7.5e0},
            ],
            maximize=True,
        )
        # optimizer = torch.optim.SGD(
        #     [
        #         {"params": [registration.rotation], "lr": 1.5e-2},
        #         {"params": [registration.translation], "lr": 15e0},
        #     ],
        #     maximize=True,
        #     momentum=0.9
        # )
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

    def nlopt_objective(self, x, grad):
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
        ncc = self.criterion(pred_img, self.img)
        # ssim = 1 - 2 * kornia.losses.ssim_loss(pred_img, self.img_ori, window_size=5, reduction='mean')
        loss = ncc
        self.losses.append(loss)

        x.grad = None
        loss.backward()

        # 如果需要计算梯度
        if grad.size > 0:
            grad[:] = x.grad.detach().cpu().numpy()

        del pred_img
        torch.cuda.empty_cache()

        end_time = time.time()
        self.times.append(end_time - start_time)

        return loss.item()

    def run(self, idx):
        idx = 8
        img, pose = self.specimen[idx]

        # tube_mask = get_tube_mask("/home/zsr/project/diffpose/ours/seg",
        #                           self.specimen.get_x_filename(idx))
        # tube_mask = torch.tensor(tube_mask, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        # tube_mask = transforms.Resize(256)(tube_mask)
        # tube_mask[tube_mask < 0.5] = 0
        # tube_mask[tube_mask >= 0.5] = 1
        # self.criterion.set_mask(tube_mask)

        # spine_mask = get_spine_mask("/home/zsr/project/diffpose/ours/seg",
        #                           self.specimen.get_x_filename(idx))
        # spine_mask = torch.tensor(spine_mask, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        # spine_mask = transforms.Resize(256)(spine_mask)
        # spine_mask[spine_mask >= 0.5] = 1
        # spine_mask[spine_mask < 0.5] = 5
        # plt.figure()
        # plt.imshow(spine_mask.cpu().squeeze(), cmap="gray")
        # plt.show()

        gt_pose = self.specimen.get_manual_gt().to(self.device)
        gt_img = self.drr(None, None, None, pose=gt_pose, bone_attenuation_multiplier=5)
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
        diff = img - img_change
        print(diff.min())
        print(diff.max())
        diff = (diff - diff.min()) / (diff.max() - diff.min()).to(self.device)
        threshold = 0.55
        diff[diff <= threshold] = 0
        diff[diff > threshold] = 1
        circle_mask = create_circle_mask(256, 120).to(self.device).unsqueeze(0).unsqueeze(0)
        total_mask = (circle_mask.bool() & diff.bool()).float()
        self.criterion.set_mask(total_mask)
        # self.criterion.set_mask(circle_mask)
        # self.criterion.set_weight_mask(spine_mask)

        plt.figure()
        plt.imshow(diff.cpu().squeeze(0).permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
        plt.show()
        img = img_change
        img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        self.pose = pose.to(self.device)
        self.img = img
        self.img_ori = img_ori

        plt.figure()
        plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()
        filename = str(time.time())
        # img_save = img.detach().clone()
        # img_save = img_save.cpu().squeeze(0).squeeze(0)
        # img_save = np.array(((img_save - img_save.min()) / (img_save.max() - img_save.min())) * 255).astype(np.uint8)
        # cv2.imwrite(f"test_img/{filename}.png", img_save)

        registration = self.initialize_registration(img)
        optimizer, scheduler = self.initialize_optimizer(registration)
        # self.target_registration_error = Evaluator(self.specimen, idx)

        # Initial loss
        # param, geo, tre = self.evaluate(registration)
        param, geo = self.evaluate(registration)
        params = [param]
        losses = []
        geodesic = [geo]
        # fiducial = [tre]
        times = []

        itr = (
            tqdm(range(self.n_iters), ncols=75) if self.verbose else range(self.n_iters)
        )
        # 创建视频写入对象
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # 使用 mp4v 编码
        # video_writer = cv2.VideoWriter(f'video/zyl_{idx}.mp4', fourcc, 30, (256, 256), isColor=False)
        video_writer = cv2.VideoWriter(f'video/{filename}.mp4', fourcc, 30, (256, 256), isColor=False)
        patch_nce = PatchNCE(patch_size=13)
        for _ in itr:
            t0 = time.perf_counter()
            optimizer.zero_grad()
            pred_img, mask = registration()
            loss = self.criterion(pred_img, img)
            # dir_consistency, mag_consistency = calculate_gradient_consistency_with_mask(pred_img, img, total_mask)
            # loss = dir_consistency
            # ncc = self.criterion(pred_img, img)
            # mi = normalized_mutual_information(pred_img, img)
            # ssim_ours = masked_ssim(pred_img, img)
            # pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min())
            # img = (img - img.min()) / (img.max() - img.min())
            # ssim = 1 - 2 * kornia.losses.ssim_loss(pred_img, img, window_size=5, reduction='mean')
            # ssim = masked_ssim2(pred_img, img, mask=circle_mask)
            # loss = 0.3 * ncc + 0.7 * ssim

            # grad_ncc = gradient_ncc(pred_img, img, total_mask)
            # loss = grad_ncc
            loss.backward()
            optimizer.step()
            scheduler.step()
            t1 = time.perf_counter()

            # param, geo, tre = self.evaluate(registration)
            param, geo = self.evaluate(registration)
            params.append(param)
            losses.append(loss.item())
            geodesic.append(geo)
            # fiducial.append(tre)
            times.append(t1 - t0)

            img_save = pred_img.detach().clone()
            img_save = img_save.cpu().squeeze(0).squeeze(0)
            img_save = np.array(((img_save - img_save.min()) / (img_save.max() - img_save.min())) * 255).astype(np.uint8)

            # plt.figure()
            # plt.imshow(img_save, cmap='gray')
            # plt.show()

            # video_writer.write(img_save)

        # video_writer.release()

        # Loss at final iteration
        pred_img, mask = registration()
        loss = self.criterion(pred_img, img)
        losses.append(loss.item())
        times.append(0)

        plt.figure()
        plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()

        pose = convert(
            [registration.rotation, registration.translation],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )
        # rot = pose.get_rotation(parameterization="so3_log_map")
        # xyz = pose.get_translation()
        rot = pose.get_rotation(parameterization="so3_log_map").detach().cpu().numpy()[0]
        xyz = pose.get_translation().detach().cpu().numpy()[0]

        # pose = RigidTransform(rot, xyz, parameterization="so3_log_map", device=self.device)
        # pred_img = self.drr(None, None, None, pose=pose)
        # pred_img = self.transforms(pred_img).to(self.device).to(torch.float32)
        # plt.figure()
        # plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy(), cmap='gray')
        # plt.show()

        # 对于se3对数映射参数化
        dim = 6  # 3旋转 + 3平移
        initial_params = np.hstack([rot, xyz])
        # initial_params = self.normalize(initial_params)
        # lb = np.array([-np.inf] * 6)  # 下界
        # ub = np.array([np.inf] * 6)  # 上界
        lb = np.array([-np.pi, -np.pi, -np.pi, -1000.0, -1000.0, -1000.0])  # SE(3)下界
        ub = np.array([np.pi, np.pi, np.pi, 1000.0, 1000.0, 1000.0])

        # 创建NLopt优化器
        opt = nlopt.opt(nlopt.LD_LBFGS, dim)
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_max_objective(self.nlopt_objective)
        # opt.set_max_objective(lambda params, grad: self.nlopt_objective(params, grad))
        opt.set_maxeval(self.n_iters)  # 最大迭代次数
        opt.set_ftol_rel(1e-5)
        dx = [0.1, 0.1, 0.1, 30, 30, 30]
        # opt.set_initial_step(dx)
        step = opt.get_initial_step(initial_params)

        # 运行优化
        final_params = None
        try:
            final_params = opt.optimize(initial_params)
            max_loss = opt.last_optimum_value()
        except Exception as e:
            print(e)
            max_loss = np.inf

        rot = torch.tensor(final_params[:3], dtype=torch.float32, device=self.device).unsqueeze(0)
        xyz = torch.tensor(final_params[3:], dtype=torch.float32, device=self.device).unsqueeze(0)
        final_pose = RigidTransform(rot, xyz, parameterization="so3_log_map", device=self.device)
        img = self.drr(None, None, None, pose=final_pose, bone_attenuation_multiplier=5)
        img = self.transforms(img).to(self.device)
        plt.figure()
        plt.imshow(img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()

        # # 评估最终结果
        # param, geo = self.evaluate()
        #
        # # 保存结果到DataFrame (与原来相同)
        # df = pd.DataFrame([param], columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
        # df["ncc"] = [-min_loss]  # 因为我们之前取了负
        # df[["geo_r", "geo_t", "geo_d", "geo_se3"]] = [geo]
        # df["time"] = [0]  # 可以添加实际耗时
        # df["idx"] = idx
        # df["parameterization"] = self.parameterization

        print(max_loss)

        return None


def main(id_number, parameterization):
    # ckpt = torch.load(f"checkpoints/zyl_800_norm_best_afe.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_epoch050.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_unknow.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_tube_no_change_best2.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_tube_no_change_best_shallow.ckpt", map_location="cuda:1")
    ckpt = torch.load(f"checkpoints/zyl_800_norm_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_mncc_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/sjj_500_2_best.ckpt", map_location="cuda:1")
    model = PoseRegressor(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = "cuda:1"

    root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    specimen = IntubationDataset(root, x_root, y_offset=155, z_cut=600)
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
    for idx in tqdm(range(len(specimen)), ncols=100):
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