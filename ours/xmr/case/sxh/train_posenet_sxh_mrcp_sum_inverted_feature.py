"""Train SXH PoseNet from frozen sum-inverted MRCP common features."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm

try:
    from pytorch_transformers.optimization import WarmupCosineSchedule
except ModuleNotFoundError:
    class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
        """Small local fallback matching the old pytorch-transformers scheduler."""

        def __init__(self, optimizer, warmup_steps: int, t_total: int, cycles: float = 0.5, last_epoch: int = -1):
            if warmup_steps < 0:
                raise ValueError("warmup_steps must be non-negative")
            if t_total <= 0:
                raise ValueError("t_total must be positive")
            self.warmup_steps = warmup_steps
            self.t_total = t_total
            self.cycles = cycles
            super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

        def lr_lambda(self, step: int) -> float:
            if self.warmup_steps > 0 and step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * 2.0 * self.cycles * progress)).item()))

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DIFFPOSE_ROOT = PROJECT_ROOT / "diffpose"
OURS_ROOT = DIFFPOSE_ROOT / "ours"
for path in (DIFFPOSE_ROOT, OURS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import timm  # noqa: E402
from diffpose.calibration import RigidTransform, convert  # noqa: E402
from diffdrr.utils import so3_log_map  # noqa: E402
from ours.my_util2 import get_random_offset  # noqa: E402
from ours.utils.metrics2 import MultiscaleNormalizedCrossCorrelation2d  # noqa: E402
from ours.xmr.case.sxh.feature_training_renderer import (  # noqa: E402
    SXHFeatureTrainingRenderer,
    compose_centered_perturbations,
)
from ours.xmr.case.sxh.optimize_feature_pose_sxh import load_model as load_feature_model  # noqa: E402
from ours.xmr.feature_network_comir_v2 import (  # noqa: E402
    CanonicalSquareCrop,
    invert_legacy_standardized_intensity,
)


SXH_ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURE_CHECKPOINT = (
    SXH_ROOT / "runs" / "feature_common_comir_antishortcut_sum_mrcp_inverted" / "checkpoints" / "last.pt"
)
DEFAULT_OUTPUT_DIR = SXH_ROOT / "runs" / "posenet_mrcp_sum_inverted_feature"
DEFAULT_GUIDEWIRE_RESULT = SXH_ROOT / "outputs" / "guidewire_registration" / "xray031" / "result.json"


@dataclass(frozen=True)
class TrainConfig:
    feature_checkpoint: str
    feature_checkpoint_step: int
    guidewire_result: str
    output_dir: str
    mrcp_registration: str
    device: str
    height: int
    model_name: str
    parameterization: str
    convention: str | None
    batch_size: int
    n_epochs: int
    n_batches_per_epoch: int
    learning_rate: float
    accumulate_step: int
    render_chunk_size: int
    checkpoint_interval: int
    seed: int
    feature_contrast_range: list[float]
    feature_noise_std_range: list[float]
    loss: str


class FeaturePoseRegressor(torch.nn.Module):
    """Pose regressor for dense frozen feature maps instead of one-channel images."""

    def __init__(
        self,
        model_name: str,
        parameterization: str,
        convention: str | None = None,
        in_chans: int = 32,
        pretrained: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = {
            "axis_angle": 3,
            "euler_angles": 3,
            "se3_log_map": 3,
            "quaternion": 4,
            "rotation_6d": 6,
            "rotation_10d": 10,
            "quaternion_adjugate": 10,
        }[parameterization]
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            **kwargs,
        )
        with torch.no_grad():
            output = self.backbone(torch.zeros(1, in_chans, 256, 256)).shape[-1]
        self.xyz_regression = torch.nn.Linear(output, 3)
        self.rot_regression = torch.nn.Linear(output, n_angular_components)

    def forward(self, x: torch.Tensor) -> RigidTransform:
        features = self.backbone(x)
        rot = self.rot_regression(features)
        xyz = self.xyz_regression(features)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )


class GeodesicSE3(torch.nn.Module):
    """Distance between transforms in the log-space of SE(3), matching diffpose.metrics."""

    def forward(self, pose_1: RigidTransform, pose_2: RigidTransform) -> torch.Tensor:
        return pose_2.compose(pose_1.inverse()).get_se3_log().norm(dim=1)


class DoubleGeodesic(torch.nn.Module):
    """Angular and translational geodesics, matching diffpose.metrics.DoubleGeodesic."""

    def __init__(self, sdr: float, eps: float = 1e-4) -> None:
        super().__init__()
        self.sdr = sdr
        self.eps = eps

    def forward(self, pose_1: RigidTransform, pose_2: RigidTransform):
        r1 = pose_1.get_rotation()
        r2 = pose_2.get_rotation()
        rotation_delta = r1 @ r2.transpose(-1, -2)
        angular_geodesic = self.sdr * so3_log_map(rotation_delta).norm(dim=-1)
        translation_geodesic = (pose_1.get_translation() - pose_2.get_translation()).norm(dim=1)
        double_geodesic = (angular_geodesic.square() + translation_geodesic.square() + self.eps).sqrt()
        return angular_geodesic, translation_geodesic, double_geodesic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PoseNet from frozen sum-inverted MRCP features")
    parser.add_argument("--feature-checkpoint", type=Path, default=DEFAULT_FEATURE_CHECKPOINT)
    parser.add_argument("--guidewire-result", type=Path, default=DEFAULT_GUIDEWIRE_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mrcp-registration", choices=["original", "refined"], default="original")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--parameterization", type=str, default="se3_log_map")
    parser.add_argument("--convention", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=800)
    parser.add_argument("--n-batches-per-epoch", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--accumulate-step", type=int, default=1)
    parser.add_argument("--render-chunk-size", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--feature-contrast-min", type=float, default=0.8)
    parser.add_argument("--feature-contrast-max", type=float, default=1.2)
    parser.add_argument("--feature-noise-std-min", type=float, default=0.0)
    parser.add_argument("--feature-noise-std-max", type=float, default=0.03)
    parser.add_argument("--restart", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_guidewire_initial_pose(path: Path, device: torch.device) -> RigidTransform:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pose = payload["initial_pose"]
    rotation = torch.tensor([pose["euler_zyx"]], dtype=torch.float32, device=device)
    translation = torch.tensor([pose["translation"]], dtype=torch.float32, device=device)
    return RigidTransform(rotation, translation, "euler_angles", "ZYX")


def render_mrcp_sum_inverted(
    renderer: SXHFeatureTrainingRenderer,
    crop: CanonicalSquareCrop,
    poses: RigidTransform,
) -> torch.Tensor:
    raw = renderer._render_in_pose_chunks(renderer.adjuster.drr_mrcp, poses, len(poses))
    normalized = renderer.normalizer(raw).to(torch.float32)
    return invert_legacy_standardized_intensity(crop(normalized))


def augment_feature_input(
    images: torch.Tensor,
    contrast_min: float,
    contrast_max: float,
    noise_std_min: float,
    noise_std_max: float,
) -> torch.Tensor:
    if contrast_min <= 0 or contrast_max <= 0 or contrast_min > contrast_max:
        raise ValueError("feature contrast range must be positive and ordered")
    if noise_std_min < 0 or noise_std_max < 0 or noise_std_min > noise_std_max:
        raise ValueError("feature noise std range must be non-negative and ordered")
    batch_size = images.shape[0]
    contrast = torch.empty((batch_size, 1, 1, 1), device=images.device, dtype=images.dtype).uniform_(
        contrast_min,
        contrast_max,
    )
    augmented = images * contrast
    if noise_std_max > 0:
        noise_std = torch.empty((batch_size, 1, 1, 1), device=images.device, dtype=images.dtype).uniform_(
            noise_std_min,
            noise_std_max,
        )
        augmented = augmented + torch.randn_like(augmented) * noise_std
    return augmented


def save_debug_images(
    output_dir: Path,
    target_img: torch.Tensor,
    pred_img: torch.Tensor,
    input_features: torch.Tensor,
) -> None:
    import matplotlib.pyplot as plt

    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    target = target_img[0, 0].detach().float().cpu()
    pred = pred_img[0, 0].detach().float().cpu()
    feature = input_features[0, 0].detach().float().cpu()
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    axes[0].imshow(target, cmap="gray")
    axes[0].set_title("target mrcp")
    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title("pred mrcp")
    axes[2].imshow(feature, cmap="viridis")
    axes[2].set_title("feature ch0")
    for axis in axes:
        axis.axis("off")
    fig.savefig(debug_dir / "step_000001_inputs.png", dpi=160)
    plt.close(fig)


def checkpoint_payload(
    model: FeaturePoseRegressor,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineSchedule,
    config: TrainConfig,
    epoch: int,
    loss: float,
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "loss": loss,
    }


def main() -> None:
    args = parse_args()
    if args.height != 256:
        raise ValueError("This feature PoseNet is fixed to 256x256 inputs")
    if args.accumulate_step <= 0:
        raise ValueError("accumulate-step must be positive")
    if args.feature_contrast_min <= 0 or args.feature_contrast_max <= 0:
        raise ValueError("feature contrast bounds must be positive")
    if args.feature_contrast_min > args.feature_contrast_max:
        raise ValueError("feature contrast min cannot exceed max")
    if args.feature_noise_std_min < 0 or args.feature_noise_std_max < 0:
        raise ValueError("feature noise std bounds must be non-negative")
    if args.feature_noise_std_min > args.feature_noise_std_max:
        raise ValueError("feature noise std min cannot exceed max")
    if args.smoke:
        args.n_epochs = 1
        args.n_batches_per_epoch = 1
        args.batch_size = min(args.batch_size, 2)

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    renderer = SXHFeatureTrainingRenderer(
        projection_mode="sum",
        render_chunk_size=args.render_chunk_size,
        device=args.device,
        mrcp_registration=args.mrcp_registration,
    )
    device = renderer.device
    crop = CanonicalSquareCrop().to(device)
    feature_model, feature_step = load_feature_model(args.feature_checkpoint, device)
    guidewire_initial_pose = load_guidewire_initial_pose(args.guidewire_result, device)

    model_params = {
        "model_name": args.model_name,
        "parameterization": args.parameterization,
        "convention": args.convention,
        "norm_layer": "groupnorm",
    }
    model = FeaturePoseRegressor(in_chans=32, **model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    total_steps = args.n_epochs * args.n_batches_per_epoch
    warmup_steps = min(5 * args.n_batches_per_epoch, max(0, total_steps - 1))
    cosine_steps = max(1, total_steps - warmup_steps)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps, cosine_steps)
    start_epoch = 0
    best_loss = torch.inf
    if args.restart is not None:
        checkpoint = torch.load(args.restart, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("loss", torch.inf))
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    output_dir = args.output_dir
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        feature_checkpoint=str(args.feature_checkpoint),
        feature_checkpoint_step=feature_step,
        guidewire_result=str(args.guidewire_result),
        output_dir=str(output_dir),
        mrcp_registration=args.mrcp_registration,
        device=str(device),
        height=args.height,
        model_name=args.model_name,
        parameterization=args.parameterization,
        convention=args.convention,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        n_batches_per_epoch=args.n_batches_per_epoch,
        learning_rate=args.learning_rate,
        accumulate_step=args.accumulate_step,
        render_chunk_size=args.render_chunk_size,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        feature_contrast_range=[args.feature_contrast_min, args.feature_contrast_max],
        feature_noise_std_range=[args.feature_noise_std_min, args.feature_noise_std_max],
        loss="1 - MS-NCC(pred_mrcp_sum_inverted, target_mrcp_sum_inverted) + 1e-2 * (GeodesicSE3 + DoubleGeodesic)",
    )
    (output_dir / "config.json").write_text(
        json.dumps({"training": asdict(config), "renderer": renderer.metadata()}, indent=2),
        encoding="utf-8",
    )

    metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4, device=device)
    geodesic = GeodesicSE3()
    double = DoubleGeodesic(renderer.adjuster.drr.detector.sdr)
    log_path = output_dir / "train.jsonl"

    print(json.dumps({"event": "start", "config": asdict(config)}, ensure_ascii=False))
    model.train()
    feature_model.eval()

    for epoch in range(start_epoch, args.n_epochs):
        losses = []
        ncc_values = []
        geodesic_values = []
        double_values = []
        rot_values = []
        xyz_values = []
        mrcp_render_seconds = []
        pred_render_seconds = []
        feature_seconds = []
        network_seconds = []

        optimizer.zero_grad(set_to_none=True)
        iterator = tqdm(range(args.n_batches_per_epoch), leave=False)
        for batch_idx in iterator:
            offsets = get_random_offset(args.batch_size, device)
            target_pose = compose_centered_perturbations(
                guidewire_initial_pose,
                renderer.specimen_center_pose,
                offsets,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.no_grad():
                target_img = render_mrcp_sum_inverted(renderer, crop, target_pose)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            mrcp_render_seconds.append(time.perf_counter() - start)

            start = time.perf_counter()
            with torch.no_grad():
                feature_input = augment_feature_input(
                    target_img,
                    args.feature_contrast_min,
                    args.feature_contrast_max,
                    args.feature_noise_std_min,
                    args.feature_noise_std_max,
                )
                input_features = feature_model.mrcp_net(feature_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            feature_seconds.append(time.perf_counter() - start)

            start = time.perf_counter()
            pred_offset = model(input_features)
            pred_pose = compose_centered_perturbations(
                guidewire_initial_pose,
                renderer.specimen_center_pose,
                pred_offset,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            network_seconds.append(time.perf_counter() - start)

            start = time.perf_counter()
            pred_img = render_mrcp_sum_inverted(renderer, crop, pred_pose)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            pred_render_seconds.append(time.perf_counter() - start)

            ncc = metric(pred_img, target_img)
            log_geodesic = geodesic(pred_pose, target_pose)
            geodesic_rot, geodesic_xyz, double_geodesic = double(pred_pose, target_pose)
            loss = 1 - ncc + 1e-2 * (log_geodesic + double_geodesic)
            loss_mean = loss.mean()
            if not torch.isfinite(loss_mean):
                print(json.dumps({"event": "skip_nonfinite_loss", "epoch": epoch, "batch": batch_idx}, ensure_ascii=False))
                optimizer.zero_grad(set_to_none=True)
                continue

            (loss_mean / args.accumulate_step).backward()
            if (batch_idx + 1) % args.accumulate_step == 0:
                adaptive_clip_grad_(model.parameters())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            losses.append(float(loss_mean.item()))
            ncc_values.append(float(ncc.mean().item()))
            geodesic_values.append(float(log_geodesic.mean().item()))
            double_values.append(float(double_geodesic.mean().item()))
            rot_values.append(float(geodesic_rot.mean().item()))
            xyz_values.append(float(geodesic_xyz.mean().item()))
            iterator.set_description(f"Epoch [{epoch + 1}/{args.n_epochs}]")
            iterator.set_postfix(
                loss=loss_mean.item(),
                ncc=ncc.mean().item(),
                geodesic_xyz=geodesic_xyz.mean().item(),
            )

            if args.smoke and epoch == 0 and batch_idx == 0:
                save_debug_images(output_dir, feature_input, pred_img, input_features)

        if not losses:
            raise RuntimeError(f"No finite losses in epoch {epoch}")

        epoch_loss = float(torch.tensor(losses).mean().item())
        record = {
            "epoch": epoch,
            "loss": epoch_loss,
            "ncc": float(torch.tensor(ncc_values).mean().item()),
            "geodesic_se3": float(torch.tensor(geodesic_values).mean().item()),
            "geodesic_rot": float(torch.tensor(rot_values).mean().item()),
            "geodesic_xyz": float(torch.tensor(xyz_values).mean().item()),
            "double_geodesic": float(torch.tensor(double_values).mean().item()),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "mrcp_render_seconds": float(torch.tensor(mrcp_render_seconds).mean().item()),
            "pred_render_seconds": float(torch.tensor(pred_render_seconds).mean().item()),
            "feature_seconds": float(torch.tensor(feature_seconds).mean().item()),
            "network_seconds": float(torch.tensor(network_seconds).mean().item()),
            "gpu_memory_mb": torch.cuda.memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))

        payload = checkpoint_payload(model, optimizer, scheduler, config, epoch, epoch_loss)
        torch.save(payload, checkpoints_dir / "last.ckpt")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(payload, checkpoints_dir / "best.ckpt")
        if epoch % args.checkpoint_interval == 0:
            torch.save(payload, checkpoints_dir / f"epoch_{epoch:03d}.ckpt")

    print(f"completed PoseNet training; output: {output_dir}")


if __name__ == "__main__":
    main()
