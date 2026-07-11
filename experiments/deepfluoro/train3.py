import os
from pathlib import Path
from sched import scheduler

import submitit
import torch
from diffdrr.drr import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm

from diffpose.deepfluoro import DeepFluoroDataset, Transforms, get_random_offset
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from diffpose.registration import PoseRegressor

import warnings
import concurrent.futures

def load(id_number, height, device):
    specimen = DeepFluoroDataset(id_number, filename="/media/sda1/PersonalFiles/yx/project/diffpose/data/ipcai_2020_full_res_data.h5")
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
    ).to(device)
    transforms = Transforms(height)

    return specimen, isocenter_pose, transforms, drr


def train(
    id_number,
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
    best_loss=torch.inf
):
    metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4)
    geodesic = GeodesicSE3()
    double = DoubleGeodesic(drr.detector.sdr)
    contrast_distribution = torch.distributions.Uniform(1.0, 10.0)

    # best_loss = torch.inf

    model.train()
    for epoch in range(n_epochs + 1):
        record_epoch = epoch + start_epoch
        losses = []
        for _ in (itr := tqdm(range(n_batches_per_epoch), leave=False)):
            contrast = contrast_distribution.sample().item()
            offset = get_random_offset(batch_size, device)
            pose = isocenter_pose.compose(offset)
            img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=contrast)
            img = transforms(img)

            pred_offset = model(img)
            pred_pose = isocenter_pose.compose(pred_offset)
            pred_img = drr(None, None, None, pose=pred_pose)
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
                    f"checkpoints/specimen_{id_number:02d}_crashed.ckpt",
                )
                raise RuntimeError("NaN loss")

            optimizer.zero_grad()
            loss.mean().backward()
            adaptive_clip_grad_(model.parameters())
            optimizer.step()
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
                f"checkpoints/specimen_{id_number:02d}_best.ckpt",
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
            f"checkpoints/specimen_{id_number:02d}_last.ckpt",
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
                f"checkpoints/specimen_{id_number:02d}_epoch{record_epoch:03d}.ckpt",
            )

        if record_epoch > 1000: break


def main(
    id_number,
    height=256,
    restart=None,
    model_name="resnet18",
    parameterization="se3_log_map",
    convention=None,
    lr=1e-3,
    batch_size=8,
    n_epochs=1000,
    n_batches_per_epoch=100,
):
    print("------")
    print("start training")
    print("------")
    id_number = int(id_number)

    device = torch.device("cuda:3")
    specimen, isocenter_pose, transforms, drr = load(id_number, height, device)

    model_params = {
        "model_name": model_name,
        "parameterization": parameterization,
        "convention": convention,
        "norm_layer": "groupnorm",
    }
    model = PoseRegressor(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0
    if restart is not None:
        ckpt = torch.load(restart, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_loss = torch.load("/media/sda1/PersonalFiles/yx/project/diffpose/experiments/deepfluoro/checkpoints/specimen_02_best.ckpt", map_location=device)["loss"]

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
        id_number,
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
        best_loss
    )


if __name__ == "__main__":
    print("------------")
    print("start")
    print("------------")
    id_numbers = [1, 2, 3, 4, 5, 6]
    Path("checkpoints").mkdir(exist_ok=True)

    gpu_ids = [1, 2, 3]
    gpu_idx = [gpu_ids[i % len(gpu_ids)] for i in id_numbers]

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="deepfluoro",
        gpus_per_node=1,
        mem_gb=20,
        timeout_min=10_000,
    )
    # jobs = executor.map_array(main, id_numbers)

    while (True):
        try:
            main(2, batch_size=4, n_batches_per_epoch=200, restart="/media/sda1/PersonalFiles/yx/project/diffpose/experiments/deepfluoro/checkpoints/specimen_02_last.ckpt")
            # main(2, batch_size=4, n_batches_per_epoch=200)
            break
        except Exception as e:
            print("retry......")



    # # 提交每个任务
    # futures = []
    # for i, id_number in enumerate(id_numbers):
    #     print(gpu_idx[i])
    #     future = executor.submit(main, id_number, gpu_idx[i])
    #     futures.append(future)
    #
    # # 等待所有任务完成
    # for future in futures:
    #     print("hello")
    #     future.result()

