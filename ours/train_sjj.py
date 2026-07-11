import os
import time
import sys
sys.path.append('/home/zsr/project/diffpose/ours/cut')
sys.path.append('/home/zsr/project/diffpose/ours')

import matplotlib.pyplot as plt
import torch

from cut.style_to_drr import StyleChanger
from utils.drr import DRR
from utils.metrics import MultiscaleNormalizedCrossCorrelation2d
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from diffpose.registration import PoseRegressor
from utils.CT_dataset import IntubationDataset, Transforms
from my_util2 import get_random_offset


def load(height, device):
    # root = "/home/zsr/project/diffpose/ours/data/liwei/史建基/CT/ShiJianJi/20231019091100.063/603"
    # x_root = "/home/zsr/project/diffpose/ours/data/liwei/史建基/ERCP/SHI^JIANJI^/20231023150900/1"
    root = "/home/zsr/project/diffpose/ours/data/liwei/史建基/CT/ShiJianJi/20231019091100.063/603"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/史建基/ERCP/SHI^JIANJI^/20231023150900/1"
    specimen = IntubationDataset(root, x_root, y_offset= 100, z_offset=500)
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
    scheduler,
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
    metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4, device=device)
    # metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4)
    geodesic = GeodesicSE3()
    double = DoubleGeodesic(drr.detector.sdr)
    contrast_distribution = torch.distributions.Uniform(1.0, 5.0)

    # best_loss = torch.inf

    model.train()
    center_pose = specimen.center_pose.to(device)
    back_pose = specimen.back_pose.to(device)
    style_change = StyleChanger(
        "/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_nec5/90_net_G.pth",
        device=device,
        resize=256)
    for epoch in range(n_epochs + 1):
        record_epoch = epoch + start_epoch
        losses = []
        for batch_idx in (itr := tqdm(range(n_batches_per_epoch), leave=False)):
            contrast = contrast_distribution.sample().item()
            offset = get_random_offset(batch_size, device)
            # pose = isocenter_pose.compose(offset)
            pose = isocenter_pose.compose(back_pose).compose(offset).compose(center_pose)
            img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=contrast)

            img = transforms(img)
            img = style_change(img)
            img = transforms(img, reverse=False).to(torch.float32)

            # for im in img:
            #     plt.figure()
            #     plt.imshow(im.cpu().squeeze(), cmap="gray")
            #     plt.show()

            pred_offset = model(img)
            pred_pose = isocenter_pose.compose(pred_offset)
            pred_img = drr(None, None, None, pose=pred_pose).to(torch.float32)
            pred_img = transforms(pred_img)

            ncc = metric(pred_img, img)
            log_geodesic = geodesic(pred_pose, pose)
            geodesic_rot, geodesic_xyz, double_geodesic = double(pred_pose, pose)
            loss = 1 - ncc + 1e-2 * (log_geodesic + double_geodesic)
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
                    f"checkpoints/sjj_crashed.ckpt",
                )
                print("Nan loss")
                continue

            # optimizer.zero_grad()
            # loss.mean().backward()
            # adaptive_clip_grad_(model.parameters())
            # optimizer.step()
            # scheduler.step()

            loss.mean().backward()
            if (batch_idx + 1) % accumulate_step == 0:
                adaptive_clip_grad_(model.parameters())
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            losses.append(loss.mean().item())

            # Update progress bar
            itr.set_description(f"Epoch [{epoch}/{n_epochs}]")
            itr.set_postfix(
                geodesic_rot=geodesic_rot.mean().item(),
                geodesic_xyz=geodesic_xyz.mean().item(),
                geodesic_dou=double_geodesic.mean().item(),
                geodesic_se3=log_geodesic.mean().item(),
                loss=loss.mean().item(),
                ncc=ncc.mean().item(),
            )

            prev_pose = pose
            prev_pred_pose = pred_pose

        losses = torch.tensor(losses)
        tqdm.write(f"Epoch {epoch + 1:04d} | Loss {losses.mean().item():.4f}")
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
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "n_batches_per_epoch": n_batches_per_epoch,
                    **model_params,
                },
                f"checkpoints/sjj_best.ckpt",
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
            f"checkpoints/sjj_last.ckpt",
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
                f"checkpoints/sjj_epoch{record_epoch:03d}.ckpt",
            )

        if record_epoch > n_epochs: break


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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0
    best_loss = torch.inf
    if restart is not None:
        ckpt = torch.load(restart, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_loss = torch.load("/home/zsr/project/diffpose/ours/checkpoints/sjj_best.ckpt", map_location=device)["loss"]

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
    if restart is not None: scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    train(
        model,
        optimizer,
        scheduler,
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
        best_loss,
        accumulate_step
    )

if __name__ == "__main__":
    main(batch_size=4, n_batches_per_epoch=200, n_epochs=1000, accumulate_step=1, lr=0.005, restart="/home/zsr/project/diffpose/ours/checkpoints/sjj_best.ckpt")

