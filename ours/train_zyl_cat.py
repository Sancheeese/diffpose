import os
import time

import matplotlib.pyplot as plt
import torch
from exceptiongroup import catch

from ours.cut.style_to_drr import StyleChanger
from ours.register_zyl_bone import inpaint_with_opencv
from ours.utils.img_utils import center_crop_and_resize_v2, batch_crop_largest_square_from_circle
# from cut.style_to_drr import StyleChanger
from utils.generate_tube import get_tube_on_image
from utils.drr import DRR
from utils.metrics2 import MultiscaleNormalizedCrossCorrelation2d
# from utils.metrics_mask_tube_add import MultiscaleNormalizedCrossCorrelation2d
# from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
# from diffpose.registration import PoseRegressor
from utils.registration_unet_cat import PoseRegressor
from utils.CT_dataset import IntubationDataset, create_circle_mask
from utils.CT_dataset import Transforms
from utils.CT_dataset_augment import Transforms as TransForms_augment
from my_util2 import get_random_offset
from utils.enhance_transforms import Transforms as ETransforms
import random


def load(height, device):
    # root = "/home/zsr/project/diffpose/ours/data/liwei/史建基/CT/ShiJianJi/20231019091100.063/603"
    # x_root = "/home/zsr/project/diffpose/ours/data/liwei/史建基/ERCP/SHI^JIANJI^/20231023150900/1"
    root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"

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
    # metric = MultiscaleNormalizedCrossCorrelation2d([None, 50], [0.5, 0.5], eps=1e-4, device=device)
    # metric = MultiscaleNormalizedCrossCorrelation2d([None, 30, 11], [0.2, 0.4, 0.4], eps=1e-4, device=device)
    metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4, device=device)
    geodesic = GeodesicSE3()
    double = DoubleGeodesic(drr.detector.sdr)
    contrast_distribution = torch.distributions.Uniform(1.0, 6.0)

    # best_loss = torch.inf

    model.train()
    center_pose = specimen.center_pose.to(device)
    back_pose = specimen.back_pose.to(device)
    etransforms = ETransforms(256)
    # style_change = StyleChanger(
    #     "/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec1/50_net_G.pth",
    #     device=device,
    #     resize=256)
    transforms_aug = TransForms_augment(256)
    for epoch in range(n_epochs + 1):
        record_epoch = epoch + start_epoch
        losses = []
        for batch_idx in (itr := tqdm(range(n_batches_per_epoch), leave=False)):
            # torch.cuda.empty_cache()  # 释放未使用的缓存
            contrast = contrast_distribution.sample().item()
            offset = get_random_offset(batch_size, device)
            # pose = isocenter_pose.compose(offset)
            pose = isocenter_pose.compose(back_pose).compose(offset).compose(center_pose)

            img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=contrast)
            img = transforms(img).to(torch.float32)
            img_input = transforms_aug(img, reverse=False).to(torch.float32)

            # for im in img:
            #     plt.figure()
            #     plt.imshow(im.cpu().permute(1, 2, 0), cmap='gray')
            #     plt.show()

            pred_offset = model(img_input)
            # pred_offset = model(img)
            pred_pose = isocenter_pose.compose(pred_offset)
            pred_img = drr(None, None, None, pose=pred_pose).to(torch.float32)
            pred_img = transforms(pred_img)
            # for im in pred_img:
            #     plt.figure()
            #     plt.imshow(im.detach().cpu().permute(1, 2, 0), cmap='gray')
            #     plt.show()

            # ncc = metric(pred_img, img, mask)
            ncc = metric(pred_img, img)
            # test = metric(pred_img, pred_img)
            # test2 = metric(img, img)
            log_geodesic = geodesic(pred_pose, pose)
            geodesic_rot, geodesic_xyz, double_geodesic = double(pred_pose, pose)
            loss = 1 - ncc + 1e-2 * (log_geodesic + double_geodesic)
            if loss.isnan().any():
                # print("Aaaaaaand we've crashed...")
                # print(ncc)
                # print(log_geodesic)
                # print(geodesic_rot)
                # print(geodesic_xyz)
                # print(double_geodesic)
                # print(pose.get_matrix())
                # print(pred_pose.get_matrix())
                # torch.save(
                #     {
                #         "model_state_dict": model.state_dict(),
                #         "optimizer_state_dict": optimizer.state_dict(),
                #         "scheduler_state_dict": scheduler.state_dict(),
                #         "height": drr.detector.height,
                #         "epoch": record_epoch,
                #         "batch_size": batch_size,
                #         "n_epochs": n_epochs,
                #         "n_batches_per_epoch": n_batches_per_epoch,
                #         "pose": pose.get_matrix().cpu(),
                #         "pred_pose": pred_pose.get_matrix().cpu(),
                #         "img": img.cpu(),
                #         "pred_img": pred_img.cpu(),
                #         **model_params,
                #     },
                #     f"checkpoints/zyl_crashed.ckpt",
                # )
                print("Nan loss")
                continue
                # raise Exception("sad...")

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
                f"checkpoints/zyl_800_cat_best.ckpt",
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

        if record_epoch > n_epochs: break


def main(
    height=256,
    restart=None,
    model_name="resnet34",
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
        best_loss = torch.load("checkpoints/zyl_800_cat_best.ckpt", map_location=device)["loss"]

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
    main(batch_size=4, n_batches_per_epoch=200, n_epochs=1000, accumulate_step=1, lr=0.005)
    while True:
        try:
            main(batch_size=4, n_batches_per_epoch=200, n_epochs=1000, accumulate_step=1, lr=0.005, restart="checkpoints/zyl_800_cat_best.ckpt")
            break
        except torch.OutOfMemoryError:
            print("\nCUDA Out of Memory! Retrying in 10 seconds...")
            time.sleep(10)  # 等待 10 秒再重启
            torch.cuda.empty_cache()  # 清空 GPU 缓存
            continue
        except torch.cuda.OutOfMemoryError:
            print("\nCUDA Out of Memory! Retrying in 10 seconds...")
            time.sleep(10)  # 等待 10 秒再重启
            torch.cuda.empty_cache()  # 清空 GPU 缓存
            continue

