import os
import time
from pathlib import Path

import torch
from ours.utils.drr import DRR
from ours.utils.drr_bone import DRR as DRR_Bone
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from matplotlib import pyplot as plt
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm

from diffpose.deepfluoro import DeepFluoroDataset, Transforms, get_random_offset
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from ours.utils.registration_bone_mask import PoseRegressor


def add_circle_mask(x, size=256, radius=120):
    y_coord = torch.arange(size) - size // 2
    x_coord = torch.arange(size) - size // 2
    Y, X = torch.meshgrid(y_coord, x_coord, indexing='ij')
    distance_sq = X ** 2 + Y ** 2  # 使用平方避免开根号
    mask = (distance_sq <= radius ** 2).float().to(x.device)

    return mask * x

def load(id_number, height, device):
    specimen = DeepFluoroDataset(id_number)
    isocenter_pose = specimen.isocenter_pose.to(device)

    subsample = (1536 - 100) / height
    delx = 0.194 * subsample
    drr = DRR(
        specimen.volume,
        specimen.spacing,
        specimen.focal_len / 2,
        height,
        delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        patch_size=height // 2
    ).to(device)

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
    ).to(device)
    transforms = Transforms(height)
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
    # metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4, device=device)
    metric = MultiscaleNormalizedCrossCorrelation2d(patch_sizes=[None, 13], patch_weights=[0.5, 0.5])
    geodesic = GeodesicSE3()
    double = DoubleGeodesic(drr.detector.sdr)
    contrast_distribution = torch.distributions.Uniform(1.0, 6.0)

    # best_loss = torch.inf

    model.train()
    # center_pose = specimen.center_pose.to(device)
    # back_pose = specimen.back_pose.to(device)
    # style_change = StyleChanger(
    #     "/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec1/50_net_G.pth",
    #     device=device,
    #     resize=256)
    for epoch in range(n_epochs + 1):
        record_epoch = epoch + start_epoch
        losses = []
        for batch_idx in (itr := tqdm(range(n_batches_per_epoch), leave=False)):
            # torch.cuda.empty_cache()  # 释放未使用的缓存
            contrast = contrast_distribution.sample().item()
            offset = get_random_offset(batch_size, device)
            # pose = isocenter_pose.compose(offset)
            pose = isocenter_pose.compose(offset)

            img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=contrast)
            img_bone = drr_bone(None, None, None, pose=pose)
            img_bone = torch.tensor(img_bone).to(torch.float32)
            img_bone = transforms(img_bone)
            img_bone = (img_bone - img_bone.min()) / (img_bone.max() - img_bone.min())
            img_bone = torch.tanh(50 * img_bone)
            img = transforms(img).to(torch.float32)

            # for im in img_bone:
            #     plt.figure()
            #     plt.imshow(im.cpu().permute(1, 2, 0), cmap='gray')
            #     plt.show()
            # for im in img:
            #     plt.figure()
            #     plt.imshow(im.cpu().permute(1, 2, 0), cmap='gray')
            #     plt.show()

            pred_offset, mask_loss = model(img, img_bone)
            pred_pose = isocenter_pose.compose(pred_offset)
            pred_img = drr(None, None, None, pose=pred_pose).to(torch.float32)
            pred_img = transforms(pred_img)
            # for im in pred_img:
            #     plt.figure()
            #     plt.imshow(im.detach().cpu().permute(1, 2, 0), cmap='gray')
            #     plt.show()

            ncc = metric(img, pred_img)

            log_geodesic = geodesic(pred_pose, pose)
            geodesic_rot, geodesic_xyz, double_geodesic = double(pred_pose, pose)
            loss = 1 - ncc + 1e-2 * (log_geodesic + double_geodesic) + 30 * mask_loss
            if loss.isnan().any():
                print("Aaaaaaand we've crashed...")
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
                f"checkpoints/specimen_02_0bone_best.ckpt",
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
            f"checkpoints/specimen_0bone_last.ckpt",
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
                f"checkpoints/specimen_02_0bone_epoch{record_epoch:03d}.ckpt",
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

    device = torch.device("cuda:2")
    specimen, isocenter_pose, transforms, drr, drr_bone = load(2, height, device)

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
        best_loss = torch.load("checkpoints/specimen_02_0bone_best.ckpt", map_location=device)["loss"]

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
    main(batch_size=8, n_batches_per_epoch=100, n_epochs=1000, accumulate_step=1, lr=0.001, restart="checkpoints/specimen_02_0bone_epoch900.ckpt")
    while True:
        try:
            main(batch_size=8, n_batches_per_epoch=100, n_epochs=1000, accumulate_step=1, lr=0.001, restart="checkpoints/specimen_0bone_last.ckpt")
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

