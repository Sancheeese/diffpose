import time

import numpy as np
import torch
import torchvision.transforms
from torch import nn

from diffpose.calibration import RigidTransform
from ours.cut.style_to_drr import StyleChanger
from ours.utils.CT_dataset_augment import RandomCLAHE
from ours.utils.loss_func import PatchNCE
from utils.drr import DRR
from utils.drr_bone import DRR as DRR_Bone
# from diffpose.metrics import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics2 import MultiscaleNormalizedCrossCorrelation2d
from utils.metrics_mask_tube2 import MultiscaleNormalizedCrossCorrelation2d
from utils.metrics_mask_tube_add import MultiscaleIoU2d
from utils.metrics_mask_add import MultiscaleNormalizedCrossCorrelation2d as MultiscaleNormalizedCrossCorrelation2dAdd
from utils.metrics_ori import MultiscaleNormalizedCrossCorrelation2d as MultiscaleNormalizedCrossCorrelation2dOri
# from utils.metrics_mask_tube_weight import MultiscaleNormalizedCrossCorrelation2d
from matplotlib import pyplot as plt
from torchvision.transforms.functional import resize
from tqdm import tqdm

from dataset.CT_dataset import Transforms
from ours.deepfluoro import DeepFluoroDataset, Evaluator
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from utils.registration_bone import PoseRegressor, SparseRegistration
from ours.utils.CT_dataset import IntubationDataset, create_circle_mask, create_circle_mask_reverse, toZeroOne
import torchvision.transforms.v2 as transforms
import nibabel as nib
import cv2

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
        self.criterion_ori = MultiscaleNormalizedCrossCorrelation2dOri([None, 128], [0.5, 0.5], step=[None, 128 // 2])
        self.criterion2_ori = MultiscaleNormalizedCrossCorrelation2dOri([None, 40, 8], [0., 0.45, 0.45], step=[None, 20, 4])
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 40, 9], [0.3, 0.4, 0.3], device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([30, 13], [0.5, 0.5], device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 128], [0.5, 0.5], device=self.device, step=[None, 128 // 2])
        # self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([None, 8], [0.2, 0.8], device=self.device, step=[None, 4])
        self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([None, 32, 8], [0.2, 0.4, 0.4], device=self.device, step=[None, 16, 4])
        # self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([128, 32], [0.5, 0.5], device=self.device, step=[64, 8])
        # self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([128, 32], [0.5, 0.5], device=self.device, step=[64, 16])
        # self.mIOU = MultiscaleIoU2d([None, 30], [0.5, 0.5], device=self.device)
        self.criterion_add = MultiscaleNormalizedCrossCorrelation2dAdd([None], [1], device=self.device, step=[None])
        self.criterion2_add = MultiscaleNormalizedCrossCorrelation2dAdd([None, 40, 9], [0.2, 0.4, 0.4],device=self.device, step=[None, 1, 1])
        self.mIOU = MultiscaleIoU2d([None], [1], device=self.device)
        self.transforms = Transforms(self.drr.detector.height)
        self.parameterization = parameterization
        self.convention = convention

        self.n_iters = n_iters
        self.verbose = verbose

        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_nec5/40_net_G.pth",
        self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec5/70_net_G.pth",
        # self.style_change =  StyleChanger("cut/ckpt/70_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_4_choose/100_net_G.pth",
                       device=self.device,
                       resize=256)

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
        #         {"params": [registration.rotation], "lr": r_lr},
        #         {"params": [registration.translation], "lr": t_lr},
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

    def run(self, idx):
        # idx = 15
        img, pose = self.specimen[idx]

        # bone_mask_gt = get_bone_mask("/media/sda1/PersonalFiles/yx/project/diffpose/ours/seg",
        filename =  self.specimen.get_x_filename(idx).split(".")[0] + "_nochange"
        # file_path = f"/media/sda1/PersonalFiles/yx/dataset/zyl_result/{filename}.nii.gz"
        file_path = f"nnuet/zyl_result/{filename}.nii.gz"
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
        diff = img - img_change
        diff = (diff - diff.min()) / (diff.max() - diff.min()).to(self.device)
        threshold = 0.55
        diff[diff <= threshold] = 0
        diff[diff > threshold] = 1
        circle_mask = create_circle_mask(256, 120).to(self.device).unsqueeze(0).unsqueeze(0)
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
        black[black <= 0] = 0
        self.black = black.to(self.device)
        print(black.min())
        print(black.max())
        plt.figure()
        plt.imshow(black.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()
        img_input = torch.pow(img, 1.5)
        img_input = inpaint_with_opencv(img_input, black)
        img_input = self.transforms(img_input, reverse=False).to(self.device).to(torch.float32)
        img = inpaint_with_opencv(img, black)
        # img = toZeroOne(gt_img)
        # img = torch.pow(img, 1.5)
        transforms_aug = torchvision.transforms.Compose([RandomCLAHE(p=1.0, tile_grid_sizes=[16], clip_limit_range=(1.0, 1.1))])
        # img = transforms_aug(img)
        img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        self.pose = pose.to(self.device)
        img = img_input

        plt.figure()
        plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"NO.{idx}")
        plt.show()
        filename = str(time.time())
        # img_save = img.detach().clone()
        # img_save = img_save.cpu().squeeze(0).squeeze(0)
        # img_save = np.array(((img_save - img_save.min()) / (img_save.max() - img_save.min())) * 255).astype(np.uint8)
        # cv2.imwrite(f"test_img/{filename}.png", img_save)

        # self.target_registration_error = Evaluator(self.specimen, idx)

        # Initial loss
        # param, geo, tre = self.evaluate(registration)
        params = []
        losses = []
        geodesic = []
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
        # video_writer = cv2.VideoWriter(f'video/zyl_{idx}.mp4', fourcc, 30, (256, 256), isColor=False)
        video_writer = cv2.VideoWriter(f'video/{filename}.mp4', fourcc, 30, (256, 256), isColor=False)
        patch_nce = PatchNCE(patch_size=13)
        circle_mask_reverse = create_circle_mask_reverse(256, 120).to(self.device)

        init_pose = self.model(img)
        init_pose = self.isocenter_pose.compose(init_pose)
        losses1, pred_pose = self.opt(img, init_pose, self.n_iters // 2, 15e-3, 10, self.criterion_ori, video_writer, reverse=True, use_wei=False)

        losses2, pred_pose = self.opt(img, pred_pose, self.n_iters // 2, 15e-3, 7.5, self.criterion2_ori, video_writer, reverse=True, use_wei=False)

        video_writer.release()



    def opt(self, img, init_pose, n_iters, r_lr, t_lr, criterion, video_writer, reverse=False, use_wei=False):
        rot_data = init_pose.get_rotation(parameterization="so3_log_map")
        xyz_data = init_pose.get_translation()

        rot = nn.Parameter(rot_data.detach().clone(), requires_grad=True)
        xyz = nn.Parameter(xyz_data.detach().clone(), requires_grad=True)
        optimizer = torch.optim.Adam(
            [
                {"params": rot, "lr": r_lr},
                {"params": xyz, "lr": t_lr},
            ],
            maximize=True,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,
            gamma=0.9,
        )

        phase_losses = []
        for i in range(n_iters):
            optimizer.zero_grad()
            pose = RigidTransform(rot, xyz, parameterization="so3_log_map", device=self.device)
            pred_img = self.drr(None, None, None, pose=pose)
            pred_img_bone = self.drr_bone(None, None, None, pose=pose)
            pred_img = self.transforms(pred_img).to(torch.float32)
            pred_img_bone = self.transforms(pred_img_bone)

            pred_img_bone = self.transforms(pred_img_bone, reverse=True)
            pred_img_bone = (pred_img_bone - pred_img_bone.min()) / (pred_img_bone.max() - pred_img_bone.min())
            pred_img_bone = torch.tanh(30 * pred_img_bone)

            # plt.figure()
            # plt.imshow((toZeroOne(pred_img) * (1 - self.black)).detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            # plt.show()
            # plt.figure()
            # plt.imshow((toZeroOne(img) * (1 - self.black)).detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            # plt.show()

            # 使用对应阶段的损失函数
            if use_wei:
                wei = torch.tensor(pred_img_bone)
                wei[wei > 0.1] = 1.5
                wei[wei <= 0.1] = 0.5
                if reverse:
                    loss = criterion(1 - toZeroOne(pred_img) * (1 - self.black), 1 - toZeroOne(img) * (1 - self.black), wei)
                else:
                    loss = criterion(pred_img, img, wei)
            else:
                if reverse:
                    loss = criterion(1 - toZeroOne(pred_img), 1 - toZeroOne(img))
                else:
                    loss = criterion(pred_img, img)
            loss.backward()
            optimizer.step()
            scheduler.step()

            phase_losses.append(loss.item())

            img_save = pred_img.detach().clone()
            img_save = img_save.cpu().squeeze(0).squeeze(0)
            img_save = np.array(((img_save - img_save.min()) / (img_save.max() - img_save.min())) * 255).astype(np.uint8)
            if i == 0:
                first_img = pred_img
                plt.figure()
                plt.imshow(img_save, cmap='gray')
                plt.show()
                # print_tre(gt_img, true_fiducials[0].detach().numpy())
                # print_tre(pred_img, pred_fiducials[0].detach().numpy())
            if i == n_iters - 1:
                mid = pred_img
                plt.figure()
                plt.imshow(img_save, cmap='gray')
                plt.show()

            video_writer.write(img_save)

            # 及时释放内存
            # del pred_img, pred_img_bone, mask, pred_pose
            # torch.cuda.empty_cache()


        return phase_losses, pose



def main(id_number, parameterization):
    # ckpt = torch.load(f"checkpoints/zyl_800_norm_best_afe.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_epoch050.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_unknow.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_tube_no_change_best2.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_tube_no_change_best_shallow.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_800_norm_best.ckpt", map_location="cuda:1")
    ckpt = torch.load(f"checkpoints/zyl_1000_aug_best.ckpt", map_location="cuda:0")
    # ckpt = torch.load(f"checkpoints/zyl_800_norm_bone_net34_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_800_unet_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_800_norm_bone_net_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/zyl_800_norm_mncc_best.ckpt", map_location="cuda:2")
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
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    specimen = IntubationDataset(root, x_root, y_offset=155, z_cut=600, factors=[3, 0.5, 0.5])
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
        n_iters=300
    )
    nccs = []
    for idx in tqdm(range(77, len(specimen)), ncols=100):
        df, ncc = registration.run(idx)
        df.to_csv(
            f"runs/stage/zyl_stage_xray{idx:03d}_{parameterization}.csv",
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
        plt.scatter(y, x, color=color)
    plt.show()

if __name__ == "__main__":
    # seed = 123
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    main(1, "se3_log_map")