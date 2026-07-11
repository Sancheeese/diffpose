import time

import math
import numpy as np
import pandas as pd
import torch
import torchvision.transforms

from ours.cut.style_to_drr import StyleChanger
from ours.utils.CT_dataset_augment import RandomCLAHE
from ours.utils.img_utils import print_tre
from ours.utils.drr import DRR
from ours.utils.drr_bone import DRR as DRR_Bone
# from diffpose.metrics import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics2 import MultiscaleNormalizedCrossCorrelation2d
from ours.utils.metrics_mask_tube2 import MultiscaleNormalizedCrossCorrelation2d
from ours.utils.metrics_mask_tube_add import MultiscaleIoU2d
# from utils.metrics_mask_tube_weight import MultiscaleNormalizedCrossCorrelation2d
from matplotlib import pyplot as plt
from torchvision.transforms.functional import resize
from tqdm import tqdm

from ours.utils.CT_dataset import Transforms
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from ours.utils.registration_bone import SparseRegistration
from ours.utils.model2 import PoseRegressor, PoseRegressorCat, PoseRegressorAttn, PoseRegressorAttnWei, \
    PoseRegressorCatCBAM, PoseRegressorAttnCBAM, PoseRegressorAttnNoWei, PoseRegressorCoe, PoseRegressorCoeDeco, \
    PoseRegressorFieldDeco, PoseRegressorCoeSpDeco, PoseRegressorMultiSpDeco, PoseRegressorMultiFrFTDeco, \
    PoseRegressorFrFTMaxDeco, PoseRegressorFrFTAllDeco, PoseRegressorSpAddDeco, PoseRegressorSpCatDeco, \
    PoseRegressorMultiSpGlobalDeco2, PoseRegressorCoeMscaleSpDeco, PoseRegressorGSpDeco
from CT_dataset import IntubationDataset
from ours.utils.CT_dataset import create_circle_mask, toZeroOne
import kornia
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
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 40, 11], [0.3, 0.4, 0.3], device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 40, 9], [0.3, 0.4, 0.3], device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([30, 13], [0.5, 0.5], device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(device=self.device)
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 128], [0.5, 0.5], device=self.device, step=[None, 128 // 2])
        self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([None, 13], [0.5, 0.5], device=self.device, step=[None, 1])
        # self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([None, 64], [0.5, 0.5], device=self.device, step=[None, 16])
        # self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([None, 40, 9], [0.1, 0.45, 0.45], device=self.device, step=[None, 40 // 2, 9 // 2])
        # self.criterion2 = MultiscaleNormalizedCrossCorrelation2d([None, 30], [0.3, 0.7], device=self.device, step=[None, 30 // 3])
        # self.mIOU = MultiscaleIoU2d([None, 30], [0.5, 0.5], device=self.device)
        self.mIOU = MultiscaleIoU2d([None], [1], device=self.device)
        self.transforms = Transforms(self.drr.detector.height, radius=119)
        self.parameterization = parameterization
        self.convention = convention

        self.n_iters = n_iters
        self.verbose = verbose

        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_nec5/40_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec5/70_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_new/70_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_new3/70_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_5_iso/100_net_G.pth",
        self.style_change =  StyleChanger("/media/sda1/PersonalFiles/yx/project/diffpose/ours/cut/drr_style_solid_5_new3/90_net_G.pth",
        # self.style_change =  StyleChanger("/media/sda1/PersonalFiles/yx/project/diffpose/ours/cut/drr_style_solid_5_new4/100_net_G.pth",
        # self.style_change =  StyleChanger("cut/ckpt/70_net_G.pth",
        # self.style_change =  StyleChanger("/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_4_choose/100_net_G.pth",
                       device=self.device,
                       resize=256)

    def initialize_registration(self, img, mask):
        with torch.no_grad():
            # st = time.time()
            offset, pred_mask = self.model(img, mask)
            # plt.figure()
            # plt.imshow(pred_mask.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            # plt.show()

            # print(f"model{time.time() - st}")
            # features = self.model.backbone.forward_features(img)
            # features = resize(
            #     features,
            #     (self.drr.detector.height, self.drr.detector.width),
            #     interpolation=3,
            #     antialias=True,
            # )
            # features = features.sum(dim=[0, 1], keepdim=True)
            # features -= features.min()
            # features /= features.max() - features.min()
            # features /= features.sum()
        pred_pose = self.isocenter_pose.compose(offset)

        return SparseRegistration(
            self.drr,
            self.drr_bone,
            pose=pred_pose,
            parameterization=self.parameterization,
            convention=self.convention,
            # features=features,
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
        geo[0] = geo[0] / 500 * (180 / math.pi)
        return param, geo

    def run(self, idx):
        # idx = 15
        img, pose = self.specimen[idx]

        gt_pose = self.specimen.get_manual_gt(idx).to(self.device)
        gt_img = self.drr(None, None, None, pose=gt_pose, bone_attenuation_multiplier=3)
        gt_img = self.transforms(gt_img).to(self.device).to(torch.float32)
        # plt.figure()
        # plt.imshow(gt_img.cpu().squeeze(), cmap="gray")
        # plt.show()

        pred_img_bone = self.drr_bone(None, None, None, pose=gt_pose)
        pred_img_bone = self.transforms(pred_img_bone, reverse=False).to(torch.float32)
        pred_img_bone = (pred_img_bone - pred_img_bone.min()) / (pred_img_bone.max() - pred_img_bone.min())
        pred_img_bone = torch.tanh(50 * pred_img_bone)
        # plt.figure()
        # plt.imshow(pred_img_bone.cpu().squeeze(), cmap="gray")
        # plt.show()

        bone = self.specimen.get_bone(idx).to(self.device).to(torch.float32)
        # bone = read_seg("/home/zsr/project/diffpose/ours/bone_seg/wfl_4.nrrd").to(self.device)
        self.img_bone = bone

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
        total_mask = (circle_mask.bool() & diff.bool()).float()
        self.criterion.set_mask(total_mask)
        self.criterion2.set_mask(total_mask)
        # self.criterion.set_mask(circle_mask)
        # self.criterion.set_weight_mask(spine_mask)
        self.mIOU.set_mask(total_mask)

        # plt.figure()
        # plt.imshow(total_mask.cpu().squeeze(0).permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
        # plt.show()
        img = img_change
        img = (img - img.min()) / (img.max() - img.min())
        black = 1 - diff
        black[black > 0] = 1
        black[black <=0 ] = 0
        print(black.min())
        print(black.max())
        # plt.figure()
        # plt.imshow(black.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        # img[black == 1] = 0.15
        # img[black == 1] = img[black == 1].pow(0.8)
        # gt_img[black == 1] = 0.15
        # gt_img[black == 1]= img_change[black == 1]
        img_input = torch.pow(img, 1)
        # img_input = inpaint_with_opencv(img_input, black)
        img_input = self.transforms(img_input, reverse=False).to(self.device).to(torch.float32)
        img = inpaint_with_opencv(img, black)
        # img = toZeroOne(gt_img)
        # img = torch.pow(img, 1.5)
        transforms_aug = torchvision.transforms.Compose([RandomCLAHE(p=1.0, tile_grid_sizes=[16], clip_limit_range=(1.0, 1.1))])
        # img = transforms_aug(img)
        img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        self.pose = pose.to(self.device)
        img = img_input
        # img = self.style_change(img)
        # img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        # img = img_ori

        # plt.figure()
        # plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.title(f"NO.{idx}")
        # plt.show()
        # plt.figure()
        # plt.imshow(img_ori.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.title(f"NO.{idx}")
        # plt.show()

        registration = self.initialize_registration(img_ori, bone)
        # registration = self.initialize_registration(gt_img, pred_img_bone)

        # Initial loss
        # param, geo, tre = self.evaluate(registration)
        param, geo = self.evaluate(registration)

        # for layer_name, feat in self.model.features.items():
        #     feat = feat.mean(dim=1)
        #     feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        #     plt.figure()
        #     plt.imshow(feat.cpu().permute(1, 2, 0), cmap='gray')
        #     plt.title(layer_name)
        #     plt.show()

        pred_img, pred_img_bone, mask, pred_pose = registration()
        # plt.figure()
        # plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.title(f"NO.{idx}")
        # plt.show()
        rot2 = pose.get_rotation(parameterization="so3_log_map")
        xyz2 = pred_pose.get_translation()
        alpha2, beta2, gamma2 = rot2.squeeze().tolist()
        bx2, by2, bz2 = xyz2.squeeze().tolist()
        param2 = [alpha2, beta2, gamma2, bx2, by2, bz2]

        ret = {}
        ret["param"] = param
        ret["param2"] = param2
        ret["geo"] = geo

        ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * circle_mask, toZeroOne(img) * circle_mask, window_size=11, reduction='mean')
        # ssim = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * total_mask, toZeroOne(img) * total_mask, window_size=11, reduction='mean')
        ssim_ori = 1 - 2 * kornia.losses.ssim_loss(toZeroOne(pred_img) * circle_mask, toZeroOne(img_ori) * circle_mask, window_size=11, reduction='mean')
        ret["ssim"] = ssim.item()
        ret["ssim_ori"] = ssim_ori.item()

        true_fiducials, pred_fiducials = self.specimen.get_2d_fiducials(idx, pred_pose)
        mpd = torch.norm(true_fiducials - pred_fiducials, dim=-1)
        mpd = torch.mean(mpd)
        tre = self.specimen.calc_tre(idx, pred_pose)
        ret["tre"] = tre.item()
        ret["mpd"] = mpd.item()
        # print_tre(gt_img, true_fiducials[0].detach().numpy())
        # print_tre(pred_img, pred_fiducials[0].detach().numpy())

        return ret


def main(id_number, parameterization):
    device_str = 'cuda:0'
    # ckpt = torch.load(f"checkpoints/wfl_cat_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/wfl_aw_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/wfl_awc_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/wfl_coe_best.ckpt", map_location="cuda:0")
    # ckpt = torch.load(f"checkpoints/wfl_coedeco_best.ckpt", map_location="cuda:0")
    # ckpt = torch.load(f"checkpoints/wfl_catdeco_best.ckpt", map_location="cuda:0")
    # ckpt = torch.load(f"checkpoints/wfl_fielddeco_best.ckpt", map_location="cuda:0")
    ckpt = torch.load(f"checkpoints/wfl_coespdeco_best.ckpt", map_location=device_str)
    # ckpt = torch.load(f"checkpoints/wfl_coespdeco_epoch300.ckpt", map_location="cuda:0")
    # ckpt = torch.load(f"checkpoints/wfl_multideco_best.ckpt", map_location="cuda:0")
    # ckpt = torch.load(f"checkpoints/wfl_multideco_epoch500.ckpt", map_location="cuda:0")
    # ckpt = torch.load(f"checkpoints/wfl_mscaledeco_best.ckpt", map_location="cuda:0")
    # ckpt = torch.load(f"checkpoints/wfl_mscaledeco_epoch300.ckpt", map_location="cuda:0")
    # ckpt = torch.load(f"checkpoints/wfl_multifrftdeco_best.ckpt", map_location="cuda:0")
    # ckpt = torch.load(f"checkpoints/wfl_frftalldeco_best.ckpt", map_location="cuda:0")
    # ckpt = torch.load(f"checkpoints/wfl_frftmaxdeco_best.ckpt", map_location="cuda:0")
    # ckpt = torch.load(f"checkpoints/wfl_nowei_best.ckpt", map_location="cuda:1")
    # ckpt = torch.load(f"checkpoints/wfl_cbam_best.ckpt", map_location="cuda:1")
    # model = PoseRegressorCoeDeco(
    # model = PoseRegressorSpCatDeco(
    # model = PoseRegressorFieldDeco(
    # model = PoseRegressorGSpDeco(
    model = PoseRegressorCoeSpDeco(
    # model = PoseRegressorCoeMscaleSpDeco(
    # model = PoseRegressorMultiSpDeco(
    # model = PoseRegressorMultiFrFTDeco(
    # model = PoseRegressorFrFTAllDeco(
    # model = PoseRegressorFrFTMaxDeco(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
        # device=device_str
    )

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = device_str

    # root = "/home/zsr/project/diffpose/ours/data/liwei/王凤兰/CT/WangFengLan/20240311144245/306"
    # x_root = "/home/zsr/project/diffpose/ours/data/liwei/王凤兰/ERCP/FENGLAN^WANG^/20240313160330/1"
    root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/王凤兰/CT/WangFengLan/20240311144245/306"
    x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/王凤兰/ERCP/FENGLAN^WANG^/20240313160330/1"

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
    all_results = []
    # for idx in tqdm(range(2, 85), ncols=100):
    for idx in tqdm(range(0, len(specimen)), ncols=100):
    #     if idx >= 62 and idx <= 65:
    #         continue
        try:
            result = registration.run(idx)
            result['idx'] = idx  # 添加样本索引
            all_results.append(result)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

        # 计算所有指标的均值和标准差
    if all_results:
        calculate_and_save_statistics(all_results, parameterization)

    return all_results

def calculate_and_save_statistics(all_results, parameterization):
    """计算并保存所有指标的统计信息"""

    # 提取各项指标
    geo_errors = np.array([result['geo'] for result in all_results])
    ssim_values = np.array([result['ssim'] for result in all_results])
    ssim_ori_values = np.array([result['ssim_ori'] for result in all_results])
    tre_values = np.array([result['tre'] for result in all_results])
    tre_lt_10_ratio = np.mean(tre_values < 10)
    print(tre_lt_10_ratio * 100)

    # 提取参数
    param_values = np.array([result['param'] for result in all_results])  # [alpha, beta, gamma, bx, by, bz]
    param2_values = np.array([result['param2'] for result in all_results])  # [alpha2, beta2, gamma2, bx2, by2, bz2]

    # 计算统计指标
    stats = {
        # 几何误差统计
        'geo_r_mean': np.mean(geo_errors[:, 0]),
        'geo_r_std': np.std(geo_errors[:, 0]),
        'geo_t_mean': np.mean(geo_errors[:, 1]),
        'geo_t_std': np.std(geo_errors[:, 1]),
        'geo_d_mean': np.mean(geo_errors[:, 2]),
        'geo_d_std': np.std(geo_errors[:, 2]),
        'geo_se3_mean': np.mean(geo_errors[:, 3]),
        'geo_se3_std': np.std(geo_errors[:, 3]),

        # 图像质量指标
        'ssim_mean': np.mean(ssim_values),
        'ssim_std': np.std(ssim_values),
        'ssim_ori_mean': np.mean(ssim_ori_values),
        'ssim_ori_std': np.std(ssim_ori_values),

        # 配准误差
        'tre_mean': np.mean(tre_values),
        'tre_std': np.std(tre_values),

        # 初始参数统计
        'alpha_mean': np.mean(param_values[:, 0]),
        'alpha_std': np.std(param_values[:, 0]),
        'beta_mean': np.mean(param_values[:, 1]),
        'beta_std': np.std(param_values[:, 1]),
        'gamma_mean': np.mean(param_values[:, 2]),
        'gamma_std': np.std(param_values[:, 2]),
        'bx_mean': np.mean(param_values[:, 3]),
        'bx_std': np.std(param_values[:, 3]),
        'by_mean': np.mean(param_values[:, 4]),
        'by_std': np.std(param_values[:, 4]),
        'bz_mean': np.mean(param_values[:, 5]),
        'bz_std': np.std(param_values[:, 5]),

        # 预测参数统计
        'alpha2_mean': np.mean(param2_values[:, 0]),
        'alpha2_std': np.std(param2_values[:, 0]),
        'beta2_mean': np.mean(param2_values[:, 1]),
        'beta2_std': np.std(param2_values[:, 1]),
        'gamma2_mean': np.mean(param2_values[:, 2]),
        'gamma2_std': np.std(param2_values[:, 2]),
        'bx2_mean': np.mean(param2_values[:, 3]),
        'bx2_std': np.std(param2_values[:, 3]),
        'by2_mean': np.mean(param2_values[:, 4]),
        'by2_std': np.std(param2_values[:, 4]),
        'bz2_mean': np.mean(param2_values[:, 5]),
        'bz2_std': np.std(param2_values[:, 5]),

        'num_samples': len(all_results),
        'parameterization': parameterization
    }

    # 保存详细结果
    detailed_data = []
    for result in all_results:
        row = {
            'idx': result['idx'],
            'ssim': result['ssim'],
            'ssim_ori': result['ssim_ori'],
            'tre': result['tre'],
            'mpd': result['mpd'],
            'geo_r': result['geo'][0],
            'geo_t': result['geo'][1],
            'geo_d': result['geo'][2],
            'geo_se3': result['geo'][3],
            'alpha': result['param'][0],
            'beta': result['param'][1],
            'gamma': result['param'][2],
            'bx': result['param'][3],
            'by': result['param'][4],
            'bz': result['param'][5],
            'alpha2': result['param2'][0],
            'beta2': result['param2'][1],
            'gamma2': result['param2'][2],
            'bx2': result['param2'][3],
            'by2': result['param2'][4],
            'bz2': result['param2'][5],
        }
        detailed_data.append(row)

    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(f"initial_pose_detailed_results.csv", index=False)

    # 保存统计结果
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(f"initial_pose_statistics.csv", index=False)

    # 打印结果
    print("\n" + "=" * 80)
    print("初始位姿预测结果统计")
    print("=" * 80)
    print(f"样本数量: {stats['num_samples']}")
    print(f"参数化方法: {parameterization}")

    print(f"\n--- 几何误差统计 ---")
    print(f"旋转误差 (geo_r): {stats['geo_r_mean']:.2f} ± {stats['geo_r_std']:.2f} mm")
    print(f"平移误差 (geo_t): {stats['geo_t_mean']:.2f} ± {stats['geo_t_std']:.2f} mm")
    print(f"双几何误差 (geo_d): {stats['geo_d_mean']:.2f} ± {stats['geo_d_std']:.2f}")
    print(f"SE3几何误差: {stats['geo_se3_mean']:.2f} ± {stats['geo_se3_std']:.2f}")

    print(f"\n--- 图像质量指标 ---")
    print(f"SSIM (生成图像): {stats['ssim_mean']:.2f} ± {stats['ssim_std']:.2f}")
    print(f"SSIM (原始图像): {stats['ssim_ori_mean']:.2f} ± {stats['ssim_ori_std']:.2f}")

    print(f"\n--- 目标配准误差 ---")
    print(f"TRE: {stats['tre_mean']:.2f} ± {stats['tre_std']:.2f} mm")

    print(f"\n--- 初始参数统计 ---")
    print(f"Alpha: {stats['alpha_mean']:.2f} ± {stats['alpha_std']:.2f}")
    print(f"Beta: {stats['beta_mean']:.2f} ± {stats['beta_std']:.2f}")
    print(f"Gamma: {stats['gamma_mean']:.2f} ± {stats['gamma_std']:.2f}")
    print(f"Bx: {stats['bx_mean']:.2f} ± {stats['bx_std']:.2f}")
    print(f"By: {stats['by_mean']:.2f} ± {stats['by_std']:.2f}")
    print(f"Bz: {stats['bz_mean']:.2f} ± {stats['bz_std']:.2f}")

    print(f"\n--- 预测参数统计 ---")
    print(f"Alpha2: {stats['alpha2_mean']:.2f} ± {stats['alpha2_std']:.2f}")
    print(f"Beta2: {stats['beta2_mean']:.2f} ± {stats['beta2_std']:.2f}")
    print(f"Gamma2: {stats['gamma2_mean']:.2f} ± {stats['gamma2_std']:.2f}")
    print(f"Bx2: {stats['bx2_mean']:.2f} ± {stats['bx2_std']:.2f}")
    print(f"By2: {stats['by2_mean']:.2f} ± {stats['by2_std']:.2f}")
    print(f"Bz2: {stats['bz2_mean']:.2f} ± {stats['bz2_std']:.2f}")
    print("=" * 80)

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

if __name__ == "__main__":
    # seed = 123
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    main(1, "se3_log_map")