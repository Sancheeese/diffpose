import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from click.core import batch
from diffdrr.drr import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from pycparser.plyparser import parameterized
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from torch import nn
from tqdm import tqdm

from diffpose.calibration import RigidTransform
from diffpose.deepfluoro import DeepFluoroDataset, Transforms
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from diffpose.registration import PoseRegressor
from dataset.CT_dataset import IntubationDataset
from my_util import get_random_offset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from ours.dataParallel import RigidTransformDataParallel


def load(height, device):
    root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/史建基/CT/ShiJianJi/20231019091100.063/603"
    x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/史建基/ERCP/SHI^JIANJI^/20231023150900/1"
    specimen = IntubationDataset(root, x_root)
    isocenter_pose = specimen.isocenter_pose.to(device)

    subsample = (1536 - 100) / height
    delx = 0.194 * subsample
    drr = DRR(
        specimen.volume,
        specimen.spacing,
        specimen.sdr,
        height,
        delx,
        reverse_x_axis=True,
    ).to(device)
    transforms = Transforms(height)

    return specimen, isocenter_pose, transforms, drr


def train(
    rank,
    world_size,
    conf
):
    batch_size = conf["batch_size"]
    n_epochs = conf["n_epochs"]
    n_batches_per_epoch = conf["n_batches_per_epoch"]
    model_params = conf["model_params"]
    restart = conf["restart"]
    height = conf["height"]
    lr = conf["lr"]

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    specimen, isocenter_pose, transforms, drr = load(height, device)

    model = PoseRegressor(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0
    best_loss = torch.inf
    if restart is not None:
        ckpt = torch.load(restart, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_loss = torch.load(
            "/media/sda1/PersonalFiles/yx/project/diffpose/experiments/deepfluoro/checkpoints/specimen_01_best.ckpt",
            map_location=device)["loss"]

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # model = model.to(device)

    scheduler = WarmupCosineSchedule(
        optimizer,
        5 * n_batches_per_epoch,
        n_epochs * n_batches_per_epoch - 5 * n_batches_per_epoch,
    )
    if restart is not None: scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4)
    geodesic = GeodesicSE3()
    double = DoubleGeodesic(drr.detector.sdr)
    contrast_distribution = torch.distributions.Uniform(1.0, 10.0)

    model.train()
    for epoch in range(n_epochs + 1):
        record_epoch = epoch + start_epoch
        losses = []
        for _ in (itr := tqdm(range(n_batches_per_epoch), leave=False)):
            torch.cuda.empty_cache()
            contrast = contrast_distribution.sample().item()
            offset = get_random_offset(batch_size, device)
            pose = isocenter_pose.compose(offset)
            img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=5)

            # for im in img:
            #     plt.figure()
            #     plt.imshow(im.cpu().squeeze(), cmap="gray")
            #     plt.show()

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
                    f"checkpoints/sjj_crashed.ckpt",
                )
                # raise RuntimeError("NaN loss")
                continue

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
            torch.cuda.empty_cache()

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

        # if record_epoch > 1000: break


def main(
    height=256,
    restart=None,
    model_name="resnet18",
    parameterization="se3_log_map",
    convention=None,
    lr=1e-3,
    batch_size=8,
    n_epochs=1000,
    n_batches_per_epoch=100
):
    print("------")
    print("start training")
    print("------")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    model_params = {
        "model_name": model_name,
        "parameterization": parameterization,
        "convention": convention,
        "norm_layer": "groupnorm",
    }

    conf = {
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "n_batches_per_epoch": n_batches_per_epoch,
        "model_params": model_params,
        "height": height,
        "lr": lr,
        "restart": None
    }

    world_size = 4
    mp.spawn(
        train,
        args=(world_size, conf),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":

    main(batch_size=4, n_batches_per_epoch=200, n_epochs=200)
