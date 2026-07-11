import time

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision.transforms

from diffpose.calibration import convert
from ours.cut.style_to_drr import StyleChanger
from ours.utils.CT_dataset_augment import RandomHistogramEqualization, RandomCLAHE
from ours.utils.grad_similar import calculate_gradient_consistency_with_mask, gradient_ncc, log_ncc, enhance_edge, \
    enhance_by_log, enhance_by_log_func, get_edge
from ours.utils.loss_func import PatchNCE, masked_ssim, masked_ssim2
from ours.utils.drr import DRR
from ours.utils.drr_bone import DRR as DRR_Bone
from ours.utils.metrics_mask_tube2_wei2 import MultiscaleNormalizedCrossCorrelation2d
from ours.utils.metrics_mask_tube_add import MultiscaleIoU2d
from matplotlib import pyplot as plt
from tqdm import tqdm

from CT_dataset import Transforms
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from ours.utils.registration_bone_mask3 import PoseRegressor
from ours.utils.registration_bone import SparseRegistration, VectorizedNormalizedCrossCorrelation2d
from CT_dataset import IntubationDataset, create_circle_mask, create_circle_mask_reverse, toZeroOne, \
    simple_projection
import kornia
import cv2
from ours.utils.registration_unet_gn import UNet

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
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 128], [0.5, 0.5], device=self.device, step=[None, 128 // 2])
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None], [1], device=self.device, step=[None])
        self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([None, 8], [0.5, 0.5], device=self.device, step=[None, 4])
        # self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([None, 8], [0.5, 0.5], device=self.device, step=[None, 4])
        # self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([None, 8], [0.5, 0.5], device=self.device, step=[None, 4])
        # self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([None, 40, 9], [0.1, 0.45, 0.45], device=self.device, step=[None, 20, 4])
        # self.mIOU = MultiscaleIoU2d([None, 30], [0.5, 0.5], device=self.device)
        self.mIOU = MultiscaleIoU2d([None], [1], device=self.device)
        self.transforms = Transforms(self.drr.detector.height)
        self.parameterization = parameterization
        self.convention = convention

        self.n_iters = n_iters
        self.verbose = verbose

        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_nec5/40_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_new/75_net_G.pth",
        self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_new3/70_net_G.pth",
        # self.style_change =  StyleChanger("/media/sda1/PersonalFiles/yx/project/diffpose/ours/cut/drr_style_solid_5_new3/70_net_G.pth",
        # self.style_change =  StyleChanger("cut/ckpt/latest_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_4_choose/100_net_G.pth",
                       device=self.device,
                       resize=256)
        # self.unet = UNet(1, 1).to(self.device)
        # unet_ckpt = torch.load("./checkpoints/deal_noise3.ckpt", map_location=device)
        # self.unet.load_state_dict(unet_ckpt["model_state_dict"])

    def initialize_registration(self, img, gt=None):
        with torch.no_grad():
            offset, mask_loss = self.model(img, None, gt)
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

    def run(self, idx):
        # idx = 15
        img, pose = self.specimen[idx]
        # img_bone_gt = self.drr_bone(None, None, None, pose=pose, bone_attenuation_multiplier=3)
        # img_bone_gt = self.transforms(img_bone_gt, reverse=False)
        # img_bone_gt = (img_bone_gt - img_bone_gt.min()) / (img_bone_gt.max() - img_bone_gt.min())
        # img_bone_gt[img_bone_gt >= 0.1] = 1
        # img_bone_gt[img_bone_gt < 0.1] = 0
        # plt.figure()
        # plt.imshow(img_bone_gt.cpu().squeeze(), cmap="gray")
        # plt.show()

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

        # bone_mask_gt = get_bone_mask("/media/sda1/PersonalFiles/yx/project/diffpose/ours/seg",
        # filename =  self.specimen.get_x_filename(idx).split(".")[0] + "_nochange"
        # # file_path = f"/media/sda1/PersonalFiles/yx/dataset/yjj_result/{filename}.nii.gz"
        # file_path = f"nnuet/yjj_result/{filename}.nii.gz"
        # nii_img = nib.load(file_path)
        # img_data = nii_img.get_fdata()
        # # img_data = img_data[::-1, :].copy()
        # bone_mask_gt = torch.tensor(img_data, device=self.device)
        # bone_mask_gt = torch.tensor(bone_mask_gt, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        # bone_mask_gt = transforms.Resize(256)(bone_mask_gt)
        # bone_mask_gt[bone_mask_gt >= 0.5] = 1
        # bone_mask_gt[bone_mask_gt < 0.5] = 0
        # plt.figure()
        # plt.imshow(bone_mask_gt.cpu().squeeze(), cmap="gray")
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
        threshold = 0.5
        diff[diff <= threshold] = 0
        diff[diff > threshold] = 1
        circle_mask = create_circle_mask(256, 116).to(self.device).unsqueeze(0).unsqueeze(0)
        total_mask = (circle_mask.bool() & diff.bool()).float()
        self.criterion.set_mask(total_mask)
        self.criterion2.set_mask(total_mask)
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
        # img_input[black == 1] = 0.8
        img_input = self.transforms(img_input, reverse=False).to(self.device).to(torch.float32)
        img = inpaint_with_opencv(img, black)
        transforms_aug = torchvision.transforms.Compose([RandomCLAHE(p=1.0, tile_grid_sizes=[16], clip_limit_range=(1.0, 1.1))])
        img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        self.pose = pose.to(self.device)
        img = img_input
        # img = self.style_change(img)
        # img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)

        img = img.flip(dims=[-1])
        plt.figure()
        plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"NO.{idx}")
        plt.show()

        img_ori = img_ori.flip(dims=[-1])
        plt.figure()
        plt.imshow(img_ori.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"NO.{idx}")
        plt.show()
        filename = str(time.time())
        # img_save = img.detach().clone()
        # img_save = img_save.cpu().squeeze(0).squeeze(0)
        # img_save = np.array(((img_save - img_save.min()) / (img_save.max() - img_save.min())) * 255).astype(np.uint8)
        # cv2.imwrite(f"test_img/{filename}.png", img_save)


        registration = self.initialize_registration(img)
        # img = img_ori
        optimizer, scheduler = self.initialize_optimizer(registration, r_lr=10e-3, t_lr=5)
        # self.target_registration_error = Evaluator(self.specimen, idx)

        # Initial loss
        # param, geo, tre = self.evaluate(registration)
        param, geo = self.evaluate(registration)
        params = [param]
        losses = []
        geodesic = [geo]
        fiducial = []
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
        #     plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
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
        # video_writer = cv2.VideoWriter(f'video/yjj_{idx}.mp4', fourcc, 30, (256, 256), isColor=False)
        video_writer = cv2.VideoWriter(f'video/{filename}.mp4', fourcc, 30, (256, 256), isColor=False)
        patch_nce = PatchNCE(patch_size=13)
        circle_mask_reverse = create_circle_mask_reverse(256, 120).to(self.device)
        # bone_mask_gt += circle_mask_reverse
        # bone_mask_gt = bone_mask_gt.bool().float()
        # black = (1 - bone_mask_gt) * img.max()
        # img = img * bone_mask_gt + black
        # img_ori = img_ori * bone_mask_gt + black
        # plt.figure()
        # plt.imshow(img_ori.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        # double = DoubleGeodesic(self.drr.detector.sdr)
        # geodesic = GeodesicSE3()
        # img = enhance_edge(img, self.transforms, total_mask)
        # plt.figure()
        # plt.imshow(img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        # see_mid(gt_img, total_mask)
        # see_mid(img, total_mask)

        # img = histogram_matching(img.squeeze(0), total_mask.squeeze(0), gt_img.squeeze(0), total_mask.squeeze(0), 256)
        # img = histogram_equalization(img.squeeze(0), total_mask, 256)
        # img = histogram_matching(img_ori.squeeze(0), total_mask.squeeze(0), gt_img.squeeze(0), total_mask.squeeze(0), 256)
        # white_wei = torch.ones_like(img)
        # white_wei[img > 250] = 0
        # white_wei = white_wei.unsqueeze(0)
        # plt.figure()
        # plt.imshow(white_wei.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()

        # with torch.no_grad():
        #     img_input = self.unet(img_ori, img_input)["output"]
        # plt.figure()
        # plt.imshow(img_input.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()

        edge = get_edge(img, total_mask)
        edge = torch.clamp(toZeroOne(edge), max=0.5)
        edge += 0.5
        plt.figure()
        plt.imshow(edge.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()
        img = self.transforms(img, reverse=False)

        st = time.time()
        for i in itr:
            t0 = time.perf_counter()
            optimizer.zero_grad()
            # optimizer = optimizer2
            # scheduler = scheduler2
            # optimizer.zero_grad()
            pred_img, pred_img_bone, mask, pred_pose = registration()
            # pred_img = toZeroOne(pred_img)
            # pred_img = torch.pow(pred_img, 1.5)
            # tmp = transforms_aug(tmp)

            # ncc = self.criterion(pred_img, gt_img)
            # # test = metric(pred_img, pred_img)
            # # test2 = metric(img, img)
            # log_geodesic = geodesic(pred_pose, gt_pose)
            # geodesic_rot, geodesic_xyz, double_geodesic = double(pred_pose, gt_pose)
            # loss = 1 - ncc + 1e-2 * (log_geodesic + double_geodesic)

            # ncc = self.criterion(pred_img, img_ori)
            # return 1, ncc
            bone_mask = torch.zeros_like(pred_img_bone)
            pred_img_bone = self.transforms(pred_img_bone, reverse=True)
            pred_img_bone = (pred_img_bone - pred_img_bone.min()) / (pred_img_bone.max() - pred_img_bone.min())
            pred_img_bone = torch.tanh(30 * pred_img_bone)
            # pred_img_bone[pred_img_bone > 0] = 1
            # pred_img_bone[pred_img_bone <= 0] = 0
            # pred_img_bone += circle_mask_reverse
            # pred_img_bone = pred_img_bone.bool().float()
            # black = (1 - pred_img_bone) * img.max()
            # pred_img = pred_img * pred_img_bone + black
            # plt.figure()
            # plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
            # plt.show()
            # print(pred_img_bone.max())
            # print(pred_img_bone.min())

            # pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min())
            # img = (img - img.min()) / (img.max() - img.min())
            # loss = - (torch.abs(pred_img - img) * mask).mean()
            # loss = self.criterion(pred_img, img, mask)

            # intersection = (pred_img_bone * bone_mask_gt * total_mask).sum()
            # union = (pred_img_bone * total_mask).sum() + (bone_mask_gt * total_mask).sum() - intersection
            # iou = intersection / (union + 1e-5)

            # ncc = self.criterion2(pred_img, img)
            # # iou = self.mIOU(pred_img_bone, bone_mask_gt)
            # # loss = 0.5 * iou + 0.5 * ncc
            # loss = ncc

            wei2 = torch.ones_like(img, requires_grad=False)
            wei2[pred_img_bone > 0.1] = 2
            # pred_img = toZeroOne(pred_img)
            # pred_img[pred_img_bone > 0.1] = pred_img[pred_img_bone > 0.1] ** 1.5
            # pred_img = self.transforms(pred_img, reverse=False)
            weight = 0.5

            pred_edge = get_edge(pred_img, total_mask)
            pred_edge = torch.clamp(toZeroOne(pred_edge), max=0.5)
            pred_edge += 0.5
            # loss = self.criterion(pred_img * pred_img_bone, img * bone_mask_gt)
            # pred_img = enhance_edge(pred_img, self.transforms, total_mask)
            loss = self.criterion2(pred_img, img)
            # loss = (1 - weight) * gradient_ncc(pred_img, img, total_mask) + weight * self.criterion2(pred_img, img)
            # loss = self.criterion(self.transforms(toZeroOne(pred_img)), self.transforms(toZeroOne(img)))
            # loss = self.criterion(torch.log(toZeroOne(pred_img) + 1e-6), torch.log(toZeroOne(img) + 1e-6))
            # loss = 10 * log_ncc(pred_img, img, total_mask, kernel_size=11, sigma=1)
            # loss = gradient_ncc(pred_img, img, total_mask)                # loss = 0.5 * log_ncc(pred_img, img, total_mask) + 0.5 * self.criterion(pred_img, img)
            # loss = self.criterion(toZeroOne(pred_img) * pred_img_bone, toZeroOne(img) * bone_mask_gt)
            # loss = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * total_mask, toZeroOne(img) * total_mask, window_size=11, reduction='mean')
            # loss = self.mIOU(pred_img_bone, bone_mask_gt)

            ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * total_mask, toZeroOne(img) * total_mask, window_size=11, reduction='mean')
            ssims.append(ssim.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            t1 = time.perf_counter()

            # true_fiducials, pred_fiducials = self.specimen.get_2d_fiducials(0, pred_pose)
            # tre = torch.norm(true_fiducials - pred_fiducials, dim=2)
            # tre = torch.mean(tre)

            # print_tre(gt_img, true_fiducials[0].detach().numpy())

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
            # fiducial.append(tre.item())
            times.append(t1 - t0)

            img_save = pred_img.detach().clone()
            img_save = img_save.cpu().squeeze(0).squeeze(0)
            img_save = np.array(((img_save - img_save.min()) / (img_save.max() - img_save.min())) * 255).astype(np.uint8)

            if i == 0:
                first_img = pred_img
                plt.figure()
                plt.imshow(img_save, cmap='gray')
                plt.show()
            if i == self.n_iters / 2:
                mid = pred_img
                plt.figure()
                plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
                plt.show()
            if i % 50 == 0:
                plt.figure()
                plt.imshow(img_save, cmap='gray')
                plt.show()

            # 及时释放不需要的变量
            del pred_img, pred_img_bone, mask, pred_pose
            torch.cuda.empty_cache()

            video_writer.write(img_save)

        video_writer.release()

        print(f"配准完成需要{time.time() - st}")
        # Loss at final iteration
        pred_img, pred_img_bone, mask, pred_pose = registration()
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
        fiducial.append(0)

        plt.figure()
        plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()
        plt.figure()
        plt.subplot(1, 4, 1)
        plt.imshow(first_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"no.{idx}")
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.imshow(mid.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"no.{idx}")
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"no.{idx}")
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.imshow(img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"no.{idx}")
        plt.axis('off')
        plt.savefig(f"reg_result/yjj_stage_xray{idx:03d}.jpg")
        plt.show()
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
        df["fiducial"] = fiducial
        df["time"] = times
        df["idx"] = idx
        df["parameterization"] = self.parameterization
        df["ssim"] = ssims
        df["r1"] = rot1
        df["r2"] = rot2
        df["r3"] = rot3
        df["t1"] = trans1
        df["t2"] = trans2
        df["t3"] = trans3

        return df, losses


def main(id_number, parameterization):
    ckpt = torch.load(f"checkpoints/yjj_800_norm_bone_mask3_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/yjj_800_norm_bone_mask3_best_500.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/yjj_800_norm_bone_mask5_best.ckpt", map_location="cuda:1")

    model = PoseRegressor(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = "cuda:0"

    root = "/home/zsr/project/diffpose/ours/data/liwei/杨嘉洁/CT/YangJiaJie/20240301222402.121000/3"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/杨嘉洁/ERCP/JIAJIE^YANG^/20240304161225/1"
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/杨嘉洁/CT/YangJiaJie/20240301222402.121000/3"
    # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/杨嘉洁/ERCP/JIAJIE^YANG^/20240304161225/1"
    specimen = IntubationDataset(root, x_root, y_offset=-100,  z_offset=100, factors=[0.5, 4, 0.5])
    # specimen = IntubationDataset(root, x_root, y_offset=155, z_cut=600)
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
    for idx in tqdm(range(30, len(specimen)), ncols=100):
        df, ncc = registration.run(idx)
        df.to_csv(
            f"runs/yjj_stage_xray{idx:03d}_{parameterization}.csv",
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