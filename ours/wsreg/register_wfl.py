import time

import numpy as np
import pandas as pd
import torch
import torchvision.transforms

from diffpose.calibration import convert, RigidTransform
from ours.utils.registration import PoseRegressor
from ours.Bipose.model import CSPConvNeXt
from ours.cut.style_to_drr import StyleChanger
from ours.utils.CT_dataset_augment import RandomCLAHE
from ours.utils.grad_similar import gradient_ncc, get_edge, dice_coefficient_with_mask
from ours.utils.img_utils import *
from ours.utils.loss_func import PatchNCE
from ours.utils.drr import DRR
from ours.utils.drr_bone import DRR as DRR_Bone
# from diffpose.metrics import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics2 import MultiscaleNormalizedCrossCorrelation2d
from ours.utils.metrics_mask_tube2_wei2 import MultiscaleNormalizedCrossCorrelation2d
from ours.utils.metrics_mask_tube_add import MultiscaleIoU2d
# from utils.metrics_mask_tube_weight import MultiscaleNormalizedCrossCorrelation2d
from matplotlib import pyplot as plt
from tqdm import tqdm

from ours.utils.CT_dataset import Transforms
# from dataset.CT_dataset import Transforms
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from ours.utils.registration_bone import SparseRegistration
from ours.case.wfl.CT_dataset import IntubationDataset
from ours.utils.CT_dataset_PA import create_circle_mask, create_circle_mask_reverse, toZeroOne
import kornia
import cv2
from ours.utils.registration_unet_gn import UNet
import torch.nn.functional as F
import matplotlib.colors as mcolors

# print(torch.__version__)          # 查看 PyTorch 版本
# print(torch.version.cuda)         # 查看 CUDA 版本（如果有 GPU）
# print(torch.backends.cudnn.version())  # 查看 cuDNN 版本

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
        self.drr_bone.set_bone_attenuation_multiplier(3)
        self.model = model.to(self.device)
        model.eval()

        self.specimen = specimen
        self.isocenter_pose = specimen.isocenter_pose.to(self.device)

        self.geodesics = GeodesicSE3()
        self.doublegeo = DoubleGeodesic(sdr=self.specimen.sdr)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 40, 11], [0.3, 0.4, 0.3], device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 40, 9], [0.3, 0.4, 0.3], device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([30, 13], [0.5, 0.5], device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 13], [0.5, 0.5], device=self.device, step=[None, 1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([21], [1], device=self.device, step=[1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 21], [0.25, 0.75], device=self.device, step=[None, 1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None], [1], device=self.device, step=[None])
        # self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([None, 128], [0.5, 0.5], device=self.device, step=[None, 128 // 2])
        # self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([None, 32, 8], [0.3, 0.4, 0.3], device=self.device, step=[None, 16, 4])
        self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([None, 64], [0.5, 0.5], device=self.device, step=[None, 16])
        # self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([None, 40, 9], [0.1, 0.45, 0.45], device=self.device, step=[None, 20, 4])
        # self.mIOU = MultiscaleIoU2d([None, 30], [0.5, 0.5], device=self.device)
        self.mIOU = MultiscaleIoU2d([None], [1], device=self.device)
        self.transforms = Transforms(self.drr.detector.height, radius=119)
        self.parameterization = parameterization
        self.convention = convention

        self.n_iters = n_iters
        self.verbose = verbose

        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec5/70_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_new/75_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_white/60_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_new3/70_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_iso/100_net_G.pth",
        self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec5/70_net_G.pth",
        # self.style_change =  StyleChanger("/media/sda1/PersonalFiles/yx/project/diffpose/ours/cut/drr_style_solid_5_new3/70_net_G.pth",
        # self.style_change =  StyleChanger("/media/sda1/PersonalFiles/yx/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_white3/50_net_G.pth",
        # self.style_change =  StyleChanger("/media/sda1/PersonalFiles/yx/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_iso/100_net_G.pth",
        # self.style_change =  StyleChanger("cut/ckpt/70_net_G.pth",
        # self.style_change =  StyleChanger("cut/ckpt/latest_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_4_choose/100_net_G.pth",
                       device=self.device,
                       resize=256)

    def initialize_registration(self, img, guide_mask=None):
        with torch.no_grad():
            offset = self.model(img)
            features = None
        pred_pose = self.isocenter_pose.compose(offset)

        return SparseRegistration(
            self.drr,
            self.drr_bone,
            pose=pred_pose,
            parameterization=self.parameterization,
            convention=self.convention,
            features=features,
        )

    def initialize_optimizer(self, registration, r_lr=15e-3, t_lr=7.5e0):
        optimizer = torch.optim.Adam(
            [
                # {"params": [registration.rotation], "lr": 7.5e-3},
                {"params": [registration.rotation], "lr": r_lr},
                {"params": [registration.translation], "lr": t_lr},
            ],
            maximize=True,
        )
        # optimizer = torch.optim.SGD(
        #     [
        #         {"params": [registration.rotation], "lr": 15e-3},
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
            # loss = self.criterion(pred_img, self.img)
            loss = (1 - weight) * gradient_ncc(pred_img, self.img, self.total_mask) + weight * self.criterion(pred_img, self.img, None, self.wei_img)
            # ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * self.total_mask, toZeroOne(self.img) * self.total_mask, window_size=11, reduction='mean')
            # grad_ncc = gradient_ncc(pred_img, self.img, self.total_mask)
            # loss = ncc
            # loss = ssim
            self.losses.append(loss)

            # x.grad = None
            # loss.backward()

            if self.i % 50 == 0:
                plt.figure()
                plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
                plt.show()

                # true_fiducials, pred_fiducials = self.specimen.get_2d_fiducials(0, pose)
                # tre = torch.norm(true_fiducials - pred_fiducials, dim=2)
                # tre = torch.mean(tre)
                # self.tres.append(tre.item())
            self.i += 1

            # del pred_img
            # torch.cuda.empty_cache()

            end_time = time.time()
            self.times.append(end_time - start_time)

        return 1 - loss.item()

    def run(self, idx):
        # idx = 15
        img, pose = self.specimen[idx]

        img_rev = self.transforms(img, reverse=True).to(self.device).to(torch.float32)
        tube = torch.ones_like(img_rev).to(self.device)
        tube[toZeroOne(img_rev) > 0.75] = 0
        plt.figure()
        plt.imshow(tube.cpu().squeeze(), cmap="gray")
        plt.show()

        bone = self.specimen.get_bone(idx).to(self.device).to(torch.float32)
        # bone = read_seg("/home/zsr/project/diffpose/ours/bone_seg/wfl_4.nrrd").to(self.device)
        self.img_bone = bone

        gt_pose = self.specimen.get_manual_gt(idx).to(self.device)
        gt_img = self.drr(None, None, None, pose=gt_pose, bone_attenuation_multiplier=3)
        gt_img = self.transforms(gt_img).to(self.device).to(torch.float32)
        # gt_img = flip_img_w(gt_img)
        plt.figure()
        plt.imshow(gt_img.cpu().squeeze(), cmap="gray")
        plt.show()

        # plt.figure()
        # plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        # img = get_tube_on_image(img, black=False)
        img = self.transforms(img, reverse=True).to(self.device).to(torch.float32)
        # img = self.transforms(img, reverse=False)
        img_ori = self.transforms(img, reverse=True).to(self.device).to(torch.float32)
        img_ori = torch.tensor(img_ori).to(self.device).to(torch.float32)
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
        # self.criterion.set_mask(total_mask)
        # self.criterion2.set_mask(total_mask)
        # self.criterion.set_mask(circle_mask)
        # self.criterion2.set_mask(circle_mask)
        # self.criterion.set_mask(circle_mask)
        # self.criterion.set_weight_mask(spine_mask)
        self.mIOU.set_mask(total_mask)

        plt.figure()
        plt.imshow(total_mask.cpu().squeeze(0).permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
        plt.show()
        img = img_change
        img = (img - img.min()) / (img.max() - img.min())
        black = 1 - diff
        black[black > 0] = 1
        black[black <=0 ] = 0
        print(black.min())
        print(black.max())
        plt.figure()
        plt.imshow(black.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()
        # img[black == 1] = 0.15
        # img[black == 1] = img[black == 1].pow(0.8)
        # gt_img[black == 1] = 0.15
        # gt_img[black == 1]= img_change[black == 1]
        img_input = torch.pow(img, 1)
        # img_input = inpaint_with_opencv(img_input, black)
        # img_input = inpaint_with_opencv(img_input, 1 - tube)
        # img_input[black == 1] = 0.8
        img_input = self.transforms(img_input, reverse=False).to(self.device).to(torch.float32)
        img = inpaint_with_opencv(img, black)
        transforms_aug = torchvision.transforms.Compose([RandomCLAHE(p=1.0, tile_grid_sizes=[16], clip_limit_range=(1.0, 1.1))])
        img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        self.pose = pose.to(self.device)
        img = img_input
        # img = self.style_change(img)
        # img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)

        plt.figure()
        plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"NO.{idx}")
        plt.show()

        plt.figure(dpi=300)
        plt.imshow(img_ori.cpu().squeeze(), cmap="gray")
        plt.axis('off')
        plt.savefig(f"./ret/wfl/img_ori_{idx}.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        filename = str(time.time())
        # img_save = img.detach().clone()
        # img_save = img_save.cpu().squeeze(0).squeeze(0)
        # img_save = np.array(((img_save - img_save.min()) / (img_save.max() - img_save.min())) * 255).astype(np.uint8)
        # cv2.imwrite(f"test_img/{filename}.png", img_save)


        # img = flip_img_w(img)
        registration = self.initialize_registration(img_ori)
        # img = img_ori
        optimizer, scheduler = self.initialize_optimizer(registration, r_lr=7.5e-3, t_lr=7.5)
        # self.target_registration_error = Evaluator(self.specimen, idx)

        # Initial loss
        # param, geo, tre = self.evaluate(registration)
        param, geo = self.evaluate(registration)
        params = [param]
        losses = []
        geodesic = [geo]
        fiducial = []
        tre = []
        times = []
        ssims = []
        rot1 = []
        rot2 = []
        rot3 = []
        trans1 = []
        trans2 = []
        trans3 = []

        feats = []
        # for layer_name, feat in self.model.features.items():
        #     feat = feat.mean(dim=1)
        #     feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        #     feats.append(feat)
        #     plt.figure()
        #     plt.imshow(feat.cpu().permute(1, 2, 0))
        #     plt.title(layer_name)
        #     plt.show()

        # registration = self.initialize_registration(gt_img)
        # i = 0
        # for layer_name, feat in self.model.features.items():
        #     feat = feat.mean(dim=1)
        #     feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        #     mse = torch.mean((feat - feats[i]) ** 2)
        #     i += 1
        #     print(mse)
        #     plt.figure()
        #     plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        #     plt.title(layer_name)
        #     plt.show()

        itr = (
            tqdm(range(self.n_iters), ncols=75) if self.verbose else range(self.n_iters)
        )
        # 创建视频写入对象
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # 使用 mp4v 编码
        # video_writer = cv2.VideoWriter(f'video/wfl_{idx}.mp4', fourcc, 30, (256, 256), isColor=False)
        video_writer = cv2.VideoWriter(f'video/{filename}.mp4', fourcc, 30, (256, 256), isColor=False)
        patch_nce = PatchNCE(patch_size=13)
        circle_mask_reverse = create_circle_mask_reverse(256, 120).to(self.device)

        white_mask = torch.ones_like(img)
        white_mask[toZeroOne(img) > 0.7] = 0
        plt.figure()
        plt.imshow(circle_mask.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()
        white_mask = white_mask * circle_mask
        # white_mask = circle_mask
        plt.figure()
        plt.imshow(total_mask.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()
        # total_mask = white_mask

        # img = histogram_matching(img.squeeze(0), total_mask.squeeze(0), gt_img.squeeze(0), total_mask.squeeze(0), 256).unsqueeze(0)
        plt.figure()
        plt.imshow(img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()

        # edge = get_edge(img, total_mask)
        # edge = torch.clamp(toZeroOne(edge), max=0.5)
        # # edge += 0.2
        # edge = toZeroOne(edge)
        # edge += 0.1
        # plt.figure()
        # plt.imshow(edge.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        img = self.transforms(img, reverse=False)

        a = 3
        wei_img = torch.zeros_like(img)
        wei_img[circle_mask > 0] = torch.exp(-a * toZeroOne(img[circle_mask > 0]))
        wei_img[circle_mask <= 0] = wei_img[circle_mask > 0].min()
        print(wei_img.min())
        print(wei_img.max())
        plt.figure()
        plt.imshow(wei_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()

        # self.criterion.set_mask(white_mask)
        # self.criterion2.set_mask(white_mask)
        # self.criterion.set_mask(total_mask)
        # self.criterion2.set_mask(total_mask)
        self.criterion.set_mask(circle_mask)
        self.criterion2.set_mask(circle_mask)

        st = time.time()
        wei = 0.5
        for i in itr:
            t0 = time.perf_counter()
            optimizer.zero_grad()
            pred_img, pred_img_bone, mask, pred_pose = registration()

            pred_img_bone = self.transforms(pred_img_bone, reverse=True)
            pred_img_bone = (pred_img_bone - pred_img_bone.min()) / (pred_img_bone.max() - pred_img_bone.min())
            pred_img_bone = torch.tanh(50 * pred_img_bone)

            dice = dice_coefficient_with_mask(bone, pred_img_bone, total_mask)
            loss = self.criterion(pred_img, img_ori) + 0.1 * dice
            # loss = self.criterion(pred_img, img) + dice
            # loss = gradient_ncc(pred_img, img, total_mask) + self.criterion(pred_img, img)
            # loss = (1 - weight) * gradient_ncc(pred_img, img, total_mask) + weight * self.criterion2(pred_img, img, None, wei_img)
            # loss = (1 - weight) * gradient_ncc(pred_img, img, total_mask) + weight * self.criterion2(pred_img, img, edge, pred_edge)


            loss.backward()
            optimizer.step()
            scheduler.step()
            t1 = time.perf_counter()

            ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * total_mask, toZeroOne(img) * total_mask, window_size=11, reduction='mean')
            ssims.append(ssim.item())
            true_fiducials, pred_fiducials = self.specimen.get_2d_fiducials(idx, pred_pose)
            mpd = torch.norm(true_fiducials - pred_fiducials, dim=2)
            mpd = torch.mean(mpd)
            true_fiducials, pred_fiducials = self.specimen.get_3d_fiducials(idx, pred_pose)
            mtre = torch.norm(true_fiducials - pred_fiducials, dim=2)
            mtre = torch.mean(mtre)

            # print_tre(gt_img, true_fiducials[0].detach().numpy())
            # print_tre(pred_img, pred_fiducials[0].detach().numpy())

            rotation, translation = convert(
                pred_pose,
                input_parameterization="se3_exp_map",
                output_parameterization=registration.parameterization,
                output_convention=None,
            )
            rot1.append(rotation[0][0].item())
            rot2.append(rotation[0][1].item())
            rot3.append(rotation[0][2].item())
            trans1.append(translation[0][0].item())
            trans2.append(translation[0][1].item())
            trans3.append(translation[0][2].item())

            # param, geo, tre = self.evaluate(registration)
            param, geo = self.evaluate(registration)
            params.append(param)
            losses.append(loss.item())
            geodesic.append(geo)
            fiducial.append(mpd.item())
            tre.append(mtre.item())
            times.append(t1 - t0)

            img_save = pred_img.detach().clone()
            img_save = img_save.cpu().squeeze(0).squeeze(0)
            img_save = np.array(((img_save - img_save.min()) / (img_save.max() - img_save.min())) * 255).astype(np.uint8)

            # if i == 0:
            #     first_img = pred_img
            #     plt.figure()
            #     plt.imshow(img_save, cmap='gray')
            #     plt.show()
            # if i == self.n_iters / 2:
            #     mid = pred_img
            #     plt.figure()
            #     plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            #     plt.show()
            #
            # if i % 50 == 0:
            #     # plt.figure()
            #     # plt.imshow(pred_img_bone.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            #     # plt.show()
            #     plt.figure()
            #     plt.imshow(img_save, cmap='gray')
            #     plt.show()

            # 及时释放不需要的变量
            del pred_img, pred_img_bone, mask, pred_pose
            torch.cuda.empty_cache()

        #     video_writer.write(img_save)
        #
        # video_writer.release()
        pred_img, pred_img_bone, mask, pred_pose = registration()

        self.img = img
        self.total_mask = total_mask
        self.wei_img = wei_img
        self.losses = []
        self.times = []
        self.i = 0

        initial_pose = pred_pose
        p = self.drr(None, None, None, pose=initial_pose, bone_attenuation_multiplier=3)
        p = self.transforms(p).to(self.device)
        plt.figure(dpi=300)
        plt.imshow(p.detach().cpu().squeeze(), cmap="gray")
        plt.axis('off')
        plt.savefig(f"./ret/wfl/pred_img_{idx}.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        pred_img_bone = self.drr_bone(None, None, None, pose=pred_pose)
        pred_img_bone = self.transforms(pred_img_bone, reverse=False)
        pred_img_bone = (pred_img_bone - pred_img_bone.min()) / (pred_img_bone.max() - pred_img_bone.min())
        pred_img_bone = torch.tanh(1 * pred_img_bone)
        circle_mask = create_circle_mask(256, 118).to(self.device).unsqueeze(0).unsqueeze(0)
        edge = get_edge(pred_img_bone, circle_mask)
        overlay = overlay_grayscale_with_red_tensor_save2(img_ori, edge, f"./ret/wfl/overlay_{idx}.png", alpha=1)
        overlay = torch.tensor(overlay)
        err = toZeroOne(p) - toZeroOne(gt_img)
        norm = mcolors.TwoSlopeNorm(
            vmin=-0.5,  # 最小值
            vcenter=0,  # 中心点
            vmax=0.5  # 最大值
        )
        plt.figure(dpi=300)
        plt.imshow(err.detach().cpu().squeeze(), cmap="bwr", norm=norm)
        plt.axis('off')
        plt.savefig(f"./ret/wfl/wfl_wsr_err_{idx}.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        save_err_with_black(pred_img, gt_img, f"./ret/wfl/wfl_wsr_err_black_{idx}.png", 119)
        save_err_half(pred_img, gt_img, img_ori, overlay, edge, "./ret/wfl/wfl_wsr", idx)

        print(f"配准完成需要{time.time() - st}")
        # Loss at final iteration
        # pred_img, pred_img_bone, mask, pred_pose = registration()
        loss = self.criterion2(pred_img, img)
        test = self.criterion2(gt_img, img)
        losses.append(test.item())
        times.append(0)
        ssims.append((1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * total_mask, toZeroOne(img) * total_mask, window_size=11, reduction='mean')).item())
        rot1.append(0)
        rot2.append(0)
        rot3.append(0)
        trans1.append(0)
        trans2.append(0)
        trans3.append(0)
        true_fiducials, pred_fiducials = self.specimen.get_2d_fiducials(idx, pred_pose)
        mpd = torch.norm(true_fiducials - pred_fiducials, dim=2)
        mpd = torch.mean(mpd)
        true_fiducials, pred_fiducials = self.specimen.get_3d_fiducials(idx, pred_pose)
        mtre = torch.norm(true_fiducials - pred_fiducials, dim=2)
        mtre = torch.mean(mtre)
        fiducial.append(mpd.item())
        tre.append(mtre.item())

        # plt.figure()
        # plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        # plt.figure()
        # plt.subplot(1, 4, 1)
        # plt.imshow(first_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.title(f"no.{idx}")
        # plt.axis('off')
        # plt.subplot(1, 4, 2)
        # plt.imshow(mid.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.title(f"no.{idx}")
        # plt.axis('off')
        # plt.subplot(1, 4, 3)
        # plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.title(f"no.{idx}")
        # plt.axis('off')
        # plt.subplot(1, 4, 4)
        # plt.imshow(img_ori.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.title(f"no.{idx}")
        # plt.axis('off')
        # # plt.savefig(f"reg_result/no_stage/wfl_stage_xray{idx:03d}.jpg")
        # plt.show()


        # Write results to dataframe
        df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
        df["ncc"] = losses
        df[["geo_r", "geo_t", "geo_d", "geo_se3"]] = geodesic
        df["fiducial"] = fiducial
        df["tre"] = tre
        df["time"] = times
        df["idx"] = idx
        df["parameterization"] = self.parameterization
        df["ssim"] = ssims
        df["alpha2"] = rot1
        df["beta2"] = rot2
        df["gamma2"] = rot3
        df["bx2"] = trans1
        df["by2"] = trans2
        df["bz2"] = trans3

        return df, losses


def main(id_number, parameterization):
    ckpt = torch.load(f"checkpoints/wfl_wsr_best.ckpt", map_location="cuda:1")

    model = PoseRegressor(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = "cuda:1"

    root = "/home/zsr/project/diffpose/ours/data/liwei/王凤兰/CT/WangFengLan/20240311144245/306"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/王凤兰/ERCP/FENGLAN^WANG^/20240313160330/1"
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/王凤兰/CT/WangFengLan/20240311144245/306"
    # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/王凤兰/ERCP/FENGLAN^WANG^/20240313160330/1"

    # specimen = DeepFluoroDataset(1)
    specimen = IntubationDataset(root, x_root, x_offset=20, z_offset=50, z_cut=30, z_cut_end=250, factors=[0.5, 0.5, 1])
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
    nccs = []
    for idx in tqdm(range(0, len(specimen)), ncols=100):
    # for idx in tqdm(range(62, 66), ncols=100):
        df, ncc = registration.run(idx)
        df.to_csv(
            # f"runs/grad/orinodicewfl_xray{idx:03d}_{parameterization}.csv",
            f"runs/wfl/wfl_xray{idx:03d}_{parameterization}.csv",
            index=False,
        )

        nccs.append(ncc)
    print(nccs)

def inpaint_with_opencv(img_tensor, mask_tensor, method=cv2.INPAINT_TELEA):
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
    result_np = cv2.inpaint(img_np, mask_np, inpaintRadius=3, flags=method)

    # 4. 将结果转换回 PyTorch Tensor
    result_tensor = torch.from_numpy(result_np.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    result_tensor = result_tensor.to(img_tensor.device)

    return result_tensor

def print_tre(img, points, color='red'):
    plt.figure()
    plt.imshow(img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    delx = 256 / 305
    for x, y in points:
        x *= delx
        y *= delx
        plt.scatter(x, y, color=color)
    plt.show()

if __name__ == "__main__":
    # seed = 123
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    main(1, "se3_log_map")