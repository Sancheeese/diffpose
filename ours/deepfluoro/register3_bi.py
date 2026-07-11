import time
from itertools import product
from pathlib import Path

import pandas as pd
# import submitit
import torch
from diffdrr.detector import make_xrays
from diffdrr.drr import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from matplotlib import pyplot as plt
from pyexpat import features
from torchvision.transforms.functional import resize
from tqdm import tqdm

from diffpose.calibration import RigidTransform, convert
from diffpose.deepfluoro import DeepFluoroDataset, Evaluator, Transforms
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from diffpose.registration import PoseRegressor
from diffpose.registration import SparseRegistration
from ours.Bipose.model import CSPConvNeXt


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
        self.doublegeo = DoubleGeodesic(sdr=self.specimen.focal_len / 2)
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 13], [0.5, 0.5])
        self.transforms = Transforms(self.drr.detector.height)
        self.parameterization = parameterization
        self.convention = convention

        self.n_iters = n_iters
        self.verbose = verbose

    def initialize_registration(self, img):
        with torch.no_grad():
            offset = self.model(img)
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
        plt.show()

        img = self.transforms(img).to(self.device)
        self.pose = pose.to(self.device)

        registration = self.initialize_registration(img)
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
        for i in itr:
            t0 = time.perf_counter()
            optimizer.zero_grad()
            pred_img, mask = registration()
            loss = self.criterion(pred_img, img)
            loss.backward()
            optimizer.step()
            scheduler.step()
            t1 = time.perf_counter()

            param, geo, tre = self.evaluate(registration)
            p = convert(
                [registration.rotation, registration.translation],
                input_parameterization=self.parameterization,
                output_parameterization="se3_exp_map",
                input_convention=self.convention,
            )
            true_fiducials, pred_fiducials = self.specimen.get_2d_fiducials(idx, p)
            mpd = torch.norm(true_fiducials - pred_fiducials, dim=-1).mean()
            mpd *= 0.194
            # print(f"{idx}：mtre={tre}    mpd={mpd.item()}")
            # source1, target1 = make_xrays(p.to(self.device), self.drr.detector.source, self.drr.detector.target)
            # source2, target2 = make_xrays(pose.to(self.device), self.drr.detector.source, self.drr.detector.target)
            # target_err = torch.norm(target1 - target2, dim=2)
            # target_err = torch.mean(target_err)
            params.append(param)
            losses.append(loss.item())
            geodesic.append(geo)
            fiducial.append(tre)
            times.append(t1 - t0)

            # if i % 50 == 0:
            #     plt.figure()
            #     plt.imshow(pred_img.detach().cpu().squeeze(0).permute(1, 2, 0))
            #     plt.show()

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
    ckpt = torch.load(f"checkpoints/specimen_{id_number:02d}_bi_best.ckpt", map_location="cuda:1")
    model = CSPConvNeXt()
    model.load_state_dict(ckpt["model_state_dict"])

    # specimen = DeepFluoroDataset(id_number, filename = "/media/sda1/PersonalFiles/yx/project/diffpose/diffpose/data/ipcai_2020_full_res_data.h5")
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

    registration = Registration(
        drr,
        specimen,
        model,
        parameterization,
        device="cuda:1",
        n_iters=250
    )
    for idx in tqdm(range(0, len(specimen)), ncols=100):
        df = registration.run(idx)
        df.to_csv(
            f"runs/bipose/{id_number}/specimen{id_number:02d}_xray{idx:03d}_{parameterization}.csv",
            index=False,
        )

def print_tre(img, points, color='red'):
    plt.figure()
    plt.imshow(img.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    # delx = 256 / 305
    delx = 1
    for x, y in points:
        x *= delx
        y *= delx
        plt.scatter(x, y, color=color)
    plt.show()

if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main(3, "se3_log_map")