import time
from itertools import product
from pathlib import Path

import pandas as pd
# import submitit
import torch

from ours.deepfluoro.utils import get_bone
from ours.utils.drr import DRR
from ours.utils.grad_similar import dice_coefficient_with_mask
# from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from weimetrics import MultiscaleNormalizedCrossCorrelation2d
from matplotlib import pyplot as plt
from pyexpat import features
from torchvision.transforms.functional import resize
from tqdm import tqdm

from diffpose.calibration import RigidTransform, convert
from diffpose.deepfluoro import DeepFluoroDataset, Evaluator, Transforms
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from ours.utils.model2 import PoseRegressorCoeSpDeco
from ours.utils.registration_bone_mask3 import PoseRegressor
from diffpose.registration import SparseRegistration
# from ours.utils.siddon_registration import SparseRegistration
from ours.utils.drr_bone import DRR as DRR_Bone


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
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 13], [0.5, 0.5])
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([21], [1])
        self.transforms = Transforms(self.drr.detector.height)
        self.parameterization = parameterization
        self.convention = convention

        self.n_iters = n_iters
        self.verbose = verbose

    def initialize_registration(self, img, mask):
        with torch.no_grad():
            offset, pred_mask = self.model(img, mask)
            # plt.figure()
            # plt.imshow(pred_mask.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            # plt.show()
            features = None
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
            pose=pred_pose,
            parameterization=self.parameterization,
            convention=self.convention,
            features=features,
        )

    def initialize_optimizer(self, registration):
        optimizer = torch.optim.Adam(
            [
                {"params": [registration.rotation], "lr": 7.5e-3},
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
        tre = self.target_registration_error(est_pose.cpu()).item()
        return param, geo, tre

    def run(self, idx):
        img, pose = self.specimen[idx]

        plt.figure()
        plt.imshow(img.squeeze(0).permute(1, 2, 0))
        plt.title(f"NO.{idx}")
        plt.show()

        img = self.transforms(img).to(self.device)
        pose = pose.to(self.device)
        self.pose = pose

        bone = get_bone(1, idx).to(self.device).to(torch.float32)
        img_bone = self.drr_bone(None, None, None, pose=pose).to(self.device)
        img_bone = torch.tensor(img_bone).to(torch.float32)
        img_bone = (img_bone - img_bone.min()) / (img_bone.max() - img_bone.min())
        img_bone = torch.tanh(50 * img_bone)
        img_bone[img_bone > 0.1] = 1
        img_bone[img_bone <= 0.1] = 0
        self.criterion.set_wei(img_bone)
        plt.figure()
        plt.imshow(img_bone.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.title(f"NO.{idx}")
        plt.show()

        registration = self.initialize_registration(img, img_bone)
        optimizer, scheduler = self.initialize_optimizer(registration)
        self.target_registration_error = Evaluator(self.specimen, idx)

        # Initial loss
        param, geo, tre = self.evaluate(registration)
        params = [param]
        losses = []
        geodesic = [geo]
        fiducial = [tre]
        times = []

        # for layer_name, feat in self.model.features.items():
        #     feat = feat.mean(dim=1)
        #     feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        #     plt.figure()
        #     plt.imshow(feat.cpu().permute(1, 2, 0))
        #     plt.title(layer_name)
        #     plt.show()

        itr = (
            tqdm(range(self.n_iters), ncols=75) if self.verbose else range(self.n_iters)
        )
        wei = 0.5
        for i in itr:
            t0 = time.perf_counter()
            optimizer.zero_grad()
            pred_img, mask = registration()
            pred_pose = registration.get_current_pose()
            pred_bone = self.drr_bone(None, None, None, pose=pred_pose).to(self.device)
            pred_bone = pred_bone.to(torch.float32)
            pred_bone = (pred_bone - pred_bone.min()) / (pred_bone.max() - pred_bone.min())
            pred_bone = torch.tanh(50 * pred_bone)
            dice = dice_coefficient_with_mask(img_bone, pred_bone, None)
            loss = wei * self.criterion(pred_img, img) + (1 - wei) * dice
            # loss = self.criterion(pred_img, img)
            # loss = dice
            loss.backward()
            optimizer.step()
            scheduler.step()
            t1 = time.perf_counter()

            if i % 10 == 0 or i == self.n_iters - 1:
                param, geo, tre = self.evaluate(registration)
                params.append(param)
                losses.append(loss.item())
                geodesic.append(geo)
                fiducial.append(tre)
                times.append(t1 - t0)

            if i % 50 == 0:
                plt.figure()
                plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0))
                plt.show()

        # Loss at final iteration
        pred_img, mask = registration()
        loss = self.criterion(pred_img, img)
        losses.append(loss.item())
        times.append(0)

        # Write results to dataframe
        df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
        df["ncc"] = losses
        df[["geo_r", "geo_t", "geo_d", "geo_se3"]] = geodesic
        df["fiducial"] = fiducial
        df["time"] = times
        df["idx"] = idx
        df["parameterization"] = self.parameterization
        return df


def main(id_number, parameterization):
    ckpt = torch.load(f"checkpoints/specimen_01_coespdeco_best.ckpt", map_location="cuda:1")
    model = PoseRegressorCoeSpDeco(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
    )
    model.load_state_dict(ckpt["model_state_dict"])

    specimen = DeepFluoroDataset(id_number, filename = "/media/sda1/PersonalFiles/yx/project/diffpose/diffpose/data/ipcai_2020_full_res_data.h5")
    # specimen = DeepFluoroDataset(id_number, filename = "/home/zsr/project/diffpose/data/ipcai_2020_full_res_data.h5")
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
        specimen.focal_len / 2,
        height,
        delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        patch_size=height // 2,
        bone_attenuation_multiplier=3
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
    for idx in tqdm(range(0, len(specimen)), ncols=100):
        df = registration.run(idx)
        df.to_csv(
            f"runs/mask/1/truebonespecimen{id_number:02d}_xray{idx:03d}_{parameterization}.csv",
            index=False,
        )


if __name__ == "__main__":
    # seed = 333
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    main(1, "se3_log_map")