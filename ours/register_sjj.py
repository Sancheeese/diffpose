import time

import cv2
import numpy as np
import pandas as pd
import torch
from utils.drr import DRR
# from diffpose.metrics import MultiscaleNormalizedCrossCorrelation2d
from utils.metrics2 import MultiscaleNormalizedCrossCorrelation2d
from matplotlib import pyplot as plt
from torchvision.transforms.functional import resize
from tqdm import tqdm
from ours.cut.style_to_drr import StyleChanger

from dataset.CT_dataset import Transforms
from diffpose.deepfluoro import DeepFluoroDataset, Evaluator
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from utils.registration import PoseRegressor, SparseRegistration
from ours.utils.CT_dataset import IntubationDataset
from PIL import Image, ImageSequence
import kornia


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
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 13], [0.5, 0.5], device=self.device)
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 13], [0.5, 0.5])
        self.transforms = Transforms(self.drr.detector.height)
        self.parameterization = parameterization
        self.convention = convention

        self.n_iters = n_iters
        self.verbose = verbose

        self.style_change = StyleChanger(
            "/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_nec5/40_net_G.pth",
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
            pose=pred_pose,
            parameterization=self.parameterization,
            convention=self.convention,
            features=features,
        )

    def initialize_optimizer(self, registration):
        optimizer = torch.optim.Adam(
            [
                {"params": [registration.rotation], "lr": 7.5e-2},
                {"params": [registration.translation], "lr": 7.5e0},
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
        return param, geo

    def run(self, idx):
        img, pose = self.specimen[idx]

        plt.figure()
        plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()

        img = self.transforms(img, reverse=False)
        img_ori = torch.tensor(img).to(self.device).to(torch.float32)
        img = self.style_change(img)
        img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        # img = self.transforms(img).to(self.device).to(torch.float32)
        self.pose = pose.to(self.device)

        plt.figure()
        plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()

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
        # images = []
        # 创建视频写入对象
        # fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # 使用 mp4v 编码
        # video_writer = cv2.VideoWriter(f'video/sjj_{idx}.mp4', fourcc, 30, (256, 256), isColor=False)
        for _ in itr:
            t0 = time.perf_counter()
            optimizer.zero_grad()
            pred_img, mask = registration()
            # loss = self.criterion(pred_img, img)
            loss = 0.5 * self.criterion(pred_img, img) + 0.5 * kornia.losses.ssim_loss(pred_img, img, window_size=11, reduction='mean')
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

            plt.figure()
            plt.imshow(img_save, cmap='gray')
            plt.show()

            # images.append(Image.fromarray(img_save))
            # video_writer.write(img_save)

        # save gif
        # images[0].save(f"gif/sjj_{idx}.gif", save_all=True, append_images=images[1:], duration=30, loop=0)
        # video_writer.release()

        # Loss at final iteration
        pred_img, mask = registration()
        loss = self.criterion(pred_img, img)
        losses.append(loss.item())
        times.append(0)

        # Write results to dataframe
        df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
        df["ncc"] = losses
        df[["geo_r", "geo_t", "geo_d", "geo_se3"]] = geodesic
        # df["fiducial"] = fiducial
        df["time"] = times
        df["idx"] = idx
        df["parameterization"] = self.parameterization
        return df


def main(id_number, parameterization):
    ckpt = torch.load(f"checkpoints/sjj_best.ckpt", map_location="cuda:0")
    # ckpt = torch.load(f"checkpoints/zyl_best.ckpt", map_location="cuda:0")
    model = PoseRegressor(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    device = "cuda:0"

    root = "/home/zsr/project/diffpose/ours/data/liwei/史建基/CT/ShiJianJi/20231019091100.063/603"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/史建基/ERCP/SHI^JIANJI^/20231023150900/1"
    specimen = IntubationDataset(root, x_root, y_offset= 100, z_offset=500)
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
        device=device
    )
    for idx in tqdm(range(len(specimen)), ncols=100):
        df = registration.run(3)
        df.to_csv(
            f"runs/sjj_xray{idx:03d}_{parameterization}.csv",
            index=False,
        )


if __name__ == "__main__":
    # seed = 123
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    main(1, "se3_log_map")