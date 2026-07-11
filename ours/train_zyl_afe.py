import os
import time

import matplotlib.pyplot as plt
import torch
from more_itertools.more import strip
from scipy.stats.tests.test_continuous_fit_censored import optimizer

from utils.afe_dateset import AfeDataSet
# from cut.style_to_drr import StyleChanger
from utils.generate_tube import get_tube_on_image
from utils.drr import DRR
from utils.metrics2 import MultiscaleNormalizedCrossCorrelation2d
# from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from utils.registration_afe import PoseRegressor
from utils.CT_dataset import IntubationDataset
from utils.CT_dataset import Transforms
from my_util2 import get_random_offset
from utils.enhance_transforms import Transforms as ETransforms
import random


def load(height, device):
    # root = "/home/zsr/project/diffpose/ours/data/liwei/史建基/CT/ShiJianJi/20231019091100.063/603"
    # x_root = "/home/zsr/project/diffpose/ours/data/liwei/史建基/ERCP/SHI^JIANJI^/20231023150900/1"
    root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    specimen = IntubationDataset(root, x_root, y_offset=155, z_cut=600, factors=[3, 0.5, 0.5])
    isocenter_pose = specimen.isocenter_pose.to(device)

    subsample = 512 / height
    delx = specimen.delx * subsample
    drr = DRR(
        specimen.volume,
        specimen.spacing,
        specimen.sdr,
        height,
        delx,
        reverse_x_axis=True,
        patch_size=height // 2
    ).to(device)
    transforms = Transforms(height)

    return specimen, isocenter_pose, transforms, drr


def train(
    model,
    optimizer,
    optimizer_dis,
    scheduler,
    scheduler_dis,
    drr,
    transforms,
    specimen,
    isocenter_pose,
    device,
    batch_size,
    n_epochs,
    n_batches_per_epoch,
    model_params,
    start_epoch=0,
    best_loss=torch.inf,
    accumulate_step = 4
):
    # metric = MultiscaleNormalizedCrossCorrelation2d([None, 50], [0.5, 0.5], eps=1e-4, device=device)
    # metric = MultiscaleNormalizedCrossCorrelation2d([None, 50], [0.5, 0.5], eps=1e-4)
    metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4, device=device)
    geodesic = GeodesicSE3()
    double = DoubleGeodesic(drr.detector.sdr)
    contrast_distribution = torch.distributions.Uniform(1.0, 7.0)

    # best_loss = torch.inf

    model.train()
    center_pose = specimen.center_pose.to(device)
    back_pose = specimen.back_pose.to(device)
    etransforms = ETransforms(256)
    # style_change = StyleChanger(
    #     "/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec1/50_net_G.pth",
    #     device=device,
    #     resize=256)
    fake_date = AfeDataSet("/home/zsr/project/diffpose/ours/drrStyle/trainA",
                          "/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec5/90_net_G.pth",
                          batch_size,
                          device)
    for epoch in range(n_epochs + 1):
        record_epoch = epoch + start_epoch
        losses = []
        losses_D = []
        for batch_idx in (itr := tqdm(range(n_batches_per_epoch), leave=False)):
            # torch.cuda.empty_cache()  # 释放未使用的缓存
            contrast = contrast_distribution.sample().item()
            offset = get_random_offset(batch_size, device)
            # pose = isocenter_pose.compose(offset)
            pose = isocenter_pose.compose(back_pose).compose(offset).compose(center_pose)

            img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=contrast)
            img = transforms(img).to(torch.float32)

            # for im in img:
            #     plt.figure()
            #     plt.imshow(im.cpu().squeeze(), cmap="gray")
            #     plt.show()

            fake = fake_date.get()
            pred_offset, real_feature, fake_feature = model(img, fake)
            pred_pose = isocenter_pose.compose(pred_offset)
            pred_img = drr(None, None, None, pose=pred_pose).to(torch.float32)
            pred_img = transforms(pred_img)

            ncc = metric(pred_img, img)
            log_geodesic = geodesic(pred_pose, pose)
            geodesic_rot, geodesic_xyz, double_geodesic = double(pred_pose, pose)
            real_logit = model.discriminator(real_feature)
            fake_logit = model.discriminator(fake_feature)
            loss_afe = torch.log(real_logit) + torch.log(1 - fake_logit)
            loss = 1 - ncc + 1e-2 * (log_geodesic + double_geodesic) + 0.2 * loss_afe

            if loss.isnan().any():
                print("Aaaaaaand we've crashed...")
                print(ncc)
                print(log_geodesic)
                print(geodesic_rot)
                print(geodesic_xyz)
                print(double_geodesic)
                print(pose.get_matrix())
                print(pred_pose.get_matrix())
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "height": drr.detector.height,
                        "epoch": record_epoch,
                        "batch_size": batch_size,
                        "n_epochs": n_epochs,
                        "n_batches_per_epoch": n_batches_per_epoch,
                        "pose": pose.get_matrix().cpu(),
                        "pred_pose": pred_pose.get_matrix().cpu(),
                        "img": img.cpu(),
                        "pred_img": pred_img.cpu(),
                        **model_params,
                    },
                    f"checkpoints/zyl_crashed.ckpt",
                )
                print("Nan loss")
                continue
                # raise Exception("sad...")

            model.discriminator.requires_grad_(False)
            loss.mean().backward()
            if (batch_idx + 1) % accumulate_step == 0:
                adaptive_clip_grad_(list(model.backbone.parameters()) +
                                    list(model.xyz_regression.parameters()) +
                                    list(model.rot_regression.parameters()))
                optimizer.step()
                optimizer.zero_grad()
            model.discriminator.requires_grad_(True)

            real_logit = model.discriminator(model.backbone(img).detach())
            fake_logit = model.discriminator(model.backbone(fake).detach())
            loss_D = -torch.log(real_logit) - torch.log(1 - fake_logit)
            model.backbone.requires_grad_(False)
            model.xyz_regression.requires_grad_(False)
            model.rot_regression.requires_grad_(False)
            loss_D.mean().backward()
            if (batch_idx + 1) % accumulate_step == 0:
                adaptive_clip_grad_(model.discriminator.parameters())
                optimizer_dis.step()
                optimizer_dis.zero_grad()
            model.backbone.requires_grad_(True)
            model.xyz_regression.requires_grad_(True)
            model.rot_regression.requires_grad_(True)

            scheduler.step()
            scheduler_dis.step()

            losses.append(loss.mean().item())
            losses_D.append(loss_D.mean().item())

            # Update progress bar
            itr.set_description(f"Epoch [{epoch}/{n_epochs}]")
            itr.set_postfix(
                geodesic_rot=geodesic_rot.mean().item(),
                geodesic_xyz=geodesic_xyz.mean().item(),
                geodesic_dou=double_geodesic.mean().item(),
                geodesic_se3=log_geodesic.mean().item(),
                loss=loss.mean().item(),
                loss_D=loss_D.mean().item(),
                ncc=ncc.mean().item(),
            )

            prev_pose = pose
            prev_pred_pose = pred_pose

        losses = torch.tensor(losses)
        losses_D = torch.tensor(losses_D)
        tqdm.write(f"Epoch {epoch + 1:04d} | Loss {losses.mean().item():.4f} | Loss_D {losses_D.mean().item()}")
        if losses.mean() < best_loss and not losses.isnan().any():
            best_loss = losses.mean().item()
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "height": drr.detector.height,
                    "epoch": record_epoch,
                    "loss": losses.mean().item(),
                    "loss_D":losses_D.mean().item(),
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "n_batches_per_epoch": n_batches_per_epoch,
                    **model_params,
                },
                f"checkpoints/zyl_800_norm_best_afe.ckpt",
            )

        # 保存上一个epoch
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "height": drr.detector.height,
                "epoch": record_epoch,
                "loss": losses.mean().item(),
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "n_batches_per_epoch": n_batches_per_epoch,
                **model_params,
            },
            f"checkpoints/zyl_last.ckpt",
        )

        if record_epoch % 50 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "height": drr.detector.height,
                    "epoch": record_epoch,
                    "loss": losses.mean().item(),
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "n_batches_per_epoch": n_batches_per_epoch,
                    **model_params,
                },
                f"checkpoints/zyl_epoch{record_epoch:03d}.ckpt",
            )

def main(
    height=256,
    restart=None,
    model_name="resnet18",
    parameterization="se3_log_map",
    convention=None,
    lr=1e-3,
    batch_size=8,
    n_epochs=1000,
    n_batches_per_epoch=100,
    accumulate_step=4
):
    print("------")
    print("start training")
    print("------")

    device = torch.device("cuda:1")
    specimen, isocenter_pose, transforms, drr = load(height, device)

    model_params = {
        "model_name": model_name,
        "parameterization": parameterization,
        "convention": convention,
        "norm_layer": "groupnorm",
    }
    model = PoseRegressor(**model_params)
    encoder_params = list(model.backbone.parameters()) + list(model.xyz_regression.parameters()) + list(model.rot_regression.parameters())
    optimizer = torch.optim.Adam(encoder_params, lr=lr)
    optimizer_dis = torch.optim.Adam(model.discriminator.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0
    best_loss = torch.inf
    if restart is not None:
        ckpt = torch.load(restart, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        # optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        # best_loss = torch.load("/home/zsr/project/diffpose/ours/checkpoints/zyl_tube_no_change_best2.ckpt", map_location=device)["loss"]

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    model = model.to(device)

    scheduler = WarmupCosineSchedule(
        optimizer,
        5 * n_batches_per_epoch,
        n_epochs * n_batches_per_epoch - 5 * n_batches_per_epoch,
    )
    scheduler_dis = WarmupCosineSchedule(
        optimizer_dis,
        5 * n_batches_per_epoch,
        n_epochs * n_batches_per_epoch - 5 * n_batches_per_epoch,
    )

    train(
        model,
        optimizer,
        optimizer_dis,
        scheduler,
        scheduler_dis,
        drr,
        transforms,
        specimen,
        isocenter_pose,
        device,
        batch_size,
        n_epochs,
        n_batches_per_epoch,
        model_params,
        start_epoch,
        torch.inf,
        accumulate_step
    )

if __name__ == "__main__":
    main(batch_size=4, n_batches_per_epoch=100, n_epochs=200, accumulate_step=1, lr=0.0001, restart="checkpoints/zyl_800_norm_best.ckpt")

