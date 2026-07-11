
import time

import numpy as np
import pandas as pd
import torch
import torchvision.transforms

from diffpose.calibration import convert, RigidTransform
from ours.cut.style_to_drr import StyleChanger
from ours.utils.CT_dataset_augment import RandomCLAHE
from ours.utils.grad_similar import gradient_ncc, get_edge, dice_coefficient_with_mask
from ours.utils.img_utils import flip_img_w, overlay_grayscale_with_red_tensor
from ours.utils.loss_func import PatchNCE
from ours.utils.drr import DRR
from ours.utils.drr_bone_alter import DRR as DRR_Bone
# from diffpose.metrics import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics2 import MultiscaleNormalizedCrossCorrelation2d
from ours.utils.metrics_mask_tube2_wei2 import MultiscaleNormalizedCrossCorrelation2d as MultiscaleNormalizedCrossCorrelation2dNowei
from ours.utils.metrics_mask_tube2_weiwei import MultiscaleNormalizedCrossCorrelation2d
# from ours.utils.metrics_ori_wei import MultiscaleNormalizedCrossCorrelation2d
from ours.utils.metrics_mask_tube_add import MultiscaleIoU2d
# from utils.metrics_mask_tube_weight import MultiscaleNormalizedCrossCorrelation2d
from matplotlib import pyplot as plt
from tqdm import tqdm

from ours.utils.CT_dataset import Transforms
# from dataset.CT_dataset import Transforms
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from ours.utils.model2 import PoseRegressorCoe, PoseRegressorCoeSpDeco
from ours.utils.registration_bone_mask3 import PoseRegressor
from ours.utils.registration_bone import SparseRegistration
from ours.utils.CT_dataset_PA import create_circle_mask, create_circle_mask_reverse, toZeroOne
from CT_dataset import IntubationDataset
import kornia
import cv2
import torch.nn.functional as F

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

        self.criterion_ori = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 21], [0.5, 0.5], device=self.device, step=[None, 1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([21], [1], device=self.device, step=[1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        self.criterion_no_wei = MultiscaleNormalizedCrossCorrelation2dNowei([21], [1], device=self.device, step=[1])        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
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
        # self.style_change =  StyleChanger("cut/ckpt/70_net_G.pth",
        # self.style_change =  StyleChanger("cut/ckpt/latest_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_4_choose/100_net_G.pth",
                       device=self.device,
                       resize=256)

    def initialize_registration(self, img, mask):
        with torch.no_grad():
            offset, pred_mask = self.model(img, mask)
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
            step_size=50,
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
        # bone = read_seg("/home/zsr/project/diffpose/ours/bone_seg/xyb_4.nrrd").to(self.device)
        self.img_bone = bone
        pose = pose.to(self.device)
        img_bone = self.drr_bone(None, None, None, pose=pose).to(self.device)
        img_bone = self.transforms(img_bone, reverse=False)
        img_bone = torch.tensor(img_bone).to(torch.float32)
        img_bone = (img_bone - img_bone.min()) / (img_bone.max() - img_bone.min())
        img_bone = torch.tanh(50 * img_bone)
        img_bone[img_bone > 0.1] = 1
        img_bone[img_bone <= 0.1] = 0
        plt.figure()
        plt.imshow(img_bone.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"NO.{idx}")
        plt.show()

        gt_pose = self.specimen.get_manual_gt(idx).to(self.device)
        self.pose = gt_pose
        gt_img = self.drr(None, None, None, pose=pose, bone_attenuation_multiplier=3)
        gt_img = self.transforms(gt_img).to(self.device).to(torch.float32)
        # gt_img = flip_img_w(gt_img)
        plt.figure()
        plt.imshow(gt_img.cpu().squeeze(), cmap="gray")
        plt.show()
        img_bone = self.drr_bone(None, None, None, pose=self.specimen.isocenter_pose.to(self.device))
        img_bone = torch.tensor(img_bone).to(torch.float32)
        img_bone = self.transforms(img_bone, reverse=False)
        img_bone = (img_bone - img_bone.min()) / (img_bone.max() - img_bone.min())
        img_bone = torch.tanh(50 * img_bone)

        # gt_img = F.interpolate(gt_img, size=(1024, 1024), mode='bilinear')
        # plt.figure(dpi=300)
        # plt.imshow(gt_img.cpu().squeeze(), cmap="gray")
        # plt.axis('off')
        # plt.savefig("bone.png", bbox_inches='tight', pad_inches=0)
        # plt.show()


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

        # img = self.style_change(img)
        # img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)

        plt.figure()
        plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"NO.{idx}")
        plt.show()

        plt.figure()
        plt.imshow(img_ori.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"NO.{idx}")
        plt.show()
        filename = str(time.time())
        # img_save = img.detach().clone()
        # img_save = img_save.cpu().squeeze(0).squeeze(0)
        # img_save = np.array(((img_save - img_save.min()) / (img_save.max() - img_save.min())) * 255).astype(np.uint8)
        # cv2.imwrite(f"test_img/{filename}.png", img_save)


        # img = flip_img_w(img)
        registration = self.initialize_registration(img_ori, bone)
        # img = img_ori
        optimizer, scheduler = self.initialize_optimizer(registration, r_lr=7.5e-3, t_lr=5)
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
        # video_writer = cv2.VideoWriter(f'video/xyb_{idx}.mp4', fourcc, 30, (256, 256), isColor=False)
        video_writer = cv2.VideoWriter(f'video/{filename}.mp4', fourcc, 30, (256, 256), isColor=False)
        patch_nce = PatchNCE(patch_size=13)
        circle_mask_reverse = create_circle_mask_reverse(256, 120).to(self.device)

        img = self.transforms(img, reverse=False)

        self.criterion.set_wei(bone)
        self.criterion.set_mask(total_mask)
        self.criterion2.set_mask(total_mask)
        self.criterion_no_wei.set_mask(total_mask)

        st = time.time()
        wei = 0.5
        for i in itr:
            t0 = time.perf_counter()
            optimizer.zero_grad()
            pred_img, pred_img_bone, mask, pred_pose = registration()

            pred_img_bone = self.transforms(pred_img_bone, reverse=True)
            pred_img_bone = (pred_img_bone - pred_img_bone.min()) / (pred_img_bone.max() - pred_img_bone.min())
            pred_img_bone = torch.tanh(50 * pred_img_bone)

            # plt.figure()
            # plt.imshow(pred_img_bone.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            # plt.show()

            # sim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * total_mask, toZeroOne(img_ori) * total_mask, window_size=11, reduction='mean')
            dice = dice_coefficient_with_mask(bone, pred_img_bone, total_mask)
            loss = wei * self.criterion(pred_img, img_ori) + (1 - wei) * dice
            # loss = wei * self.criterion_no_wei(pred_img, img_ori) + (1 - wei) * dice
            # loss = self.criterion_no_wei(pred_img, img_ori)
            # loss = self.criterion(pred_img, img_ori)
            # loss = sim
            # loss = dice

            loss.backward()
            optimizer.step()
            scheduler.step()
            t1 = time.perf_counter()

            ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * total_mask, toZeroOne(img) * total_mask, window_size=11, reduction='mean')
            ssims.append(ssim.item())
            # true_fiducials, pred_fiducials = self.specimen.get_2d_fiducials(idx, pred_pose)
            # mpd = torch.norm(true_fiducials - pred_fiducials, dim=2)
            # mpd = torch.mean(mpd)
            # true_fiducials, pred_fiducials = self.specimen.get_3d_fiducials(idx, pred_pose)
            # mtre = torch.norm(true_fiducials - pred_fiducials, dim=2)
            # mtre = torch.mean(mtre)

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
            # fiducial.append(mpd.item())
            # tre.append(mtre.item())
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
            if i % 50 == 0:
                # plt.figure()
                # plt.imshow(pred_img_bone.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
                # plt.show()
                plt.figure()
                plt.imshow(img_save, cmap='gray')
                plt.show()

            # 及时释放不需要的变量
            del pred_img, pred_img_bone, mask, pred_pose
            torch.cuda.empty_cache()

        #     video_writer.write(img_save)
        #
        # video_writer.release()
        pred_img, pred_img_bone, mask, pred_pose = registration()

        self.img = img
        self.total_mask = total_mask
        self.losses = []
        self.times = []
        self.i = 0

        initial_pose = pred_pose
        p = self.drr(None, None, None, pose=initial_pose, bone_attenuation_multiplier=3)
        p = self.transforms(p).to(self.device)
        plt.figure()
        plt.imshow(p.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()

        print(f"配准完成需要{time.time() - st}")
        # Loss at final iteration
        # pred_img, pred_img_bone, mask, pred_pose = registration()
        # true_fiducials, pred_fiducials = self.specimen.get_2d_fiducials(idx, pred_pose)
        # mpd = torch.norm(true_fiducials - pred_fiducials, dim=2)
        # mpd = torch.mean(mpd)
        # true_fiducials, pred_fiducials = self.specimen.get_3d_fiducials(idx, pred_pose)
        # mtre = torch.norm(true_fiducials - pred_fiducials, dim=2)
        # mtre = torch.mean(mtre)

        dice = dice_coefficient_with_mask(bone, pred_img_bone, total_mask)
        loss = wei * self.criterion(pred_img, img_ori) + (1 - wei) * dice
        losses.append(loss.item())
        times.append(0)
        ssims.append((1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * total_mask, toZeroOne(img) * total_mask, window_size=11, reduction='mean')).item())
        rot1.append(0)
        rot2.append(0)
        rot3.append(0)
        trans1.append(0)
        trans2.append(0)
        trans3.append(0)
        # fiducial.append(mpd.item())
        # tre.append(mtre.item())

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
        # plt.show()
        pred_img_bone = self.transforms(pred_img_bone, reverse=True)
        pred_img_bone = (pred_img_bone - pred_img_bone.min()) / (pred_img_bone.max() - pred_img_bone.min())
        pred_img_bone = torch.tanh(50 * pred_img_bone)
        # plt.figure()
        # plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
        # plt.show()


        # Write results to dataframe
        df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
        df["ncc"] = losses
        df[["geo_r", "geo_t", "geo_d", "geo_se3"]] = geodesic
        # df["fiducial"] = fiducial
        # df["tre"] = tre
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
    # ckpt = torch.load(f"checkpoints/xyb_coespdeco_best.ckpt", map_location="cuda:0")
    ckpt = torch.load(f"checkpoints/xyb_coespdeco_en_best.ckpt", map_location="cuda:0")
    # ckpt = torch.load(f"checkpoints/xyb_coespdeco_epoch300.ckpt", map_location="cuda:0")

    model = PoseRegressorCoeSpDeco(
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
    device = "cuda:1"

    root = "/home/zsr/project/diffpose/ours/data/liwei/许玉坝/CT/XuYuBa/20240730122203.162/201"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/许玉坝/ERCP/XU^YUBEI^/20240731150611/1"
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/许玉坝/CT/XuYuBa/20240730122203.162/201"
    # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/许玉坝/ERCP/XU^YUBEI^/20240731150611/1"

    specimen = IntubationDataset(root, x_root, z_cut=90, factors=[0.8, 0.8, 2])
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
        bone_threshold=340
    )

    registration = Registration(
        drr,
        drr_bone,
        specimen,
        model,
        parameterization,
        device=device,
        n_iters=300
    )
    nccs = []
    for idx in tqdm(range(30, len(specimen)), ncols=100):
    # for idx in tqdm(range(0, 50), ncols=100):
    # for idx in tqdm(range(2, 30), ncols=100):
        df, ncc = registration.run(idx)
        df.to_csv(
            # f"runs/mask/xyb/localnccxyb_xray{idx:03d}_{parameterization}.csv",
            f"runs/mask/xyb_xray{idx:03d}_{parameterization}.csv",
            # f"runs/mask/xyb/xyb_xray{idx:03d}_{parameterization}.csv",
            # f"runs/grad/xyb_xray{idx:03d}_{parameterization}.csv",
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