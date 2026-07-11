import os
import time

import matplotlib.pyplot as plt
import torch
from exceptiongroup import catch

from ours.utils.CT_dataset import toZeroOne
from ours.utils.registration_unet_seg import UNet

from utils.drr import DRR
from utils.drr_bone import DRR as DRR_Bone

from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm
from utils.CT_dataset import IntubationDataset
from utils.CT_dataset import Transforms
from my_util2 import get_random_offset
import random
import torch.nn.functional as F
from utils.CT_dataset_augment2 import Transforms as TransForms_augment
from ours.register_zyl_bone_mask_stage import inpaint_with_opencv


def add_circle_mask(x, size=256, radius=120):
    y_coord = torch.arange(size) - size // 2
    x_coord = torch.arange(size) - size // 2
    Y, X = torch.meshgrid(y_coord, x_coord, indexing='ij')
    distance_sq = X ** 2 + Y ** 2  # 使用平方避免开根号
    mask = (distance_sq <= radius ** 2).float().to(x.device)

    return mask * x

def load(height, device):
    # root = "/home/zsr/project/diffpose/ours/data/liwei/史建基/CT/ShiJianJi/20231019091100.063/603"
    # x_root = "/home/zsr/project/diffpose/ours/data/liwei/史建基/ERCP/SHI^JIANJI^/20231023150900/1"
    # root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    # x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"

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

    drr_bone = DRR_Bone(
        specimen.volume,
        specimen.spacing,
        specimen.sdr,
        height,
        delx,
        reverse_x_axis=True,
        patch_size=height // 2
    ).to(device)
    transforms = Transforms(height)

    return specimen, isocenter_pose, transforms, drr, drr_bone


def train(
    model,
    optimizer,
    scheduler,
    drr,
    drr_bone,
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
    contrast_distribution = torch.distributions.Uniform(0.5, 6.0)

    model.train()
    center_pose = specimen.center_pose.to(device)
    back_pose = specimen.back_pose.to(device)
    transforms_aug = TransForms_augment(256)
    for epoch in range(n_epochs + 1):
        record_epoch = epoch + start_epoch
        losses = []
        for batch_idx in (itr := tqdm(range(n_batches_per_epoch), leave=False)):
            # torch.cuda.empty_cache()  # 释放未使用的缓存
            contrast = contrast_distribution.sample().item()
            offset = get_random_offset(batch_size, device)
            pose = isocenter_pose.compose(back_pose).compose(offset).compose(center_pose)

            img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=contrast)
            img = transforms(img).to(torch.float32)
            img_input, diff = transforms_aug(img, reverse=False)
            img_input.to(torch.float32)

            diff = toZeroOne(diff)
            diff[diff > 0.05] = 1
            diff[diff <= 0.05] = 0
            if torch.isnan(diff[0][0][0][0]):
                diff = torch.zeros_like(img).to(device).to(torch.float32)

            img_input = transforms(img_input, reverse=False)
            # for im in mask:
            #     plt.figure()
            #     plt.imshow(im.cpu().permute(1, 2, 0), cmap='gray')
            #     plt.show()
            # for im in img_input:
            #     plt.figure()
            #     plt.imshow(im.cpu().permute(1, 2, 0), cmap='gray')
            #     plt.show()

            pred_img = model(img_input, img)

            # for im in diff:
            #     plt.figure()
            #     plt.imshow(im.detach().cpu().permute(1, 2, 0), cmap='gray')
            #     plt.show()

            l1 = F.l1_loss(pred_img[diff == 1], diff[diff == 1])
            l2 = F.l1_loss(pred_img[diff == 0], diff[diff == 0])

            loss = l2
            if not l1.isnan():
                loss += l1
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
                loss=loss.mean().item(),
            )

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
                f"checkpoints/seg_unet.ckpt",
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
            f"checkpoints/deal_noise_last.ckpt",
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
                f"checkpoints/deal_noise_epoch{record_epoch:03d}.ckpt",
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

    device = torch.device("cuda:2")
    specimen, isocenter_pose, transforms, drr, drr_bone = load(height, device)

    model_params = {
        "model_name": model_name,
        "parameterization": parameterization,
        "convention": convention,
        "norm_layer": "groupnorm",
    }
    model = UNet(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0
    best_loss = torch.inf
    if restart is not None:
        ckpt = torch.load(restart, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_loss = torch.load("checkpoints/seg_unet.ckpt", map_location=device)["loss"]

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
        drr_bone,
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
    main(batch_size=4, n_batches_per_epoch=200, n_epochs=800, accumulate_step=1, lr=0.005)
    while True:
        try:
            main(batch_size=4, n_batches_per_epoch=200, n_epochs=800, accumulate_step=1, lr=0.005, restart="checkpoints/seg_unet.ckpt")
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

