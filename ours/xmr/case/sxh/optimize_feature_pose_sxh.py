"""Optimize SXH MRCP projection pose with a frozen cross-modal feature network."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DIFFPOSE_ROOT = PROJECT_ROOT / "diffpose"
OURS_ROOT = DIFFPOSE_ROOT / "ours"
for path in (DIFFPOSE_ROOT, OURS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diffpose.calibration import RigidTransform  # noqa: E402
from ours.xmr.case.sxh.feature_training_renderer import (  # noqa: E402
    SXHFeatureTrainingRenderer,
    compose_centered_perturbations,
)
from ours.xmr.case.sxh.image_io import write_gray_png, write_overlay_png  # noqa: E402
from ours.xmr.case.sxh.web_drr_server_nii_sxh_centerline import (  # noqa: E402
    project_points_to_detector,
    render_chain_points_overlay,
)
from ours.xmr.feature_network_comir_v2 import (  # noqa: E402
    CanonicalSquareCrop,
    CoMIRTwoBranchFeatureNetwork,
    invert_legacy_standardized_intensity,
)


SXH_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SXH_ROOT / "runs" / "feature_common_comir_antishortcut" / "checkpoints" / "last.pt"
DEFAULT_OUTPUT_ROOT = SXH_ROOT / "runs" / "feature_pose_optimization"


@dataclass(frozen=True)
class OptimizationConfig:
    mode: str
    index: int
    projection: str
    checkpoint: str
    output_dir: str
    device: str
    steps: int
    patch_size: int
    patch_centers: int
    rotation_lr: float
    translation_lr: float
    gradient_clip: float
    contrast_multiplier: float
    init_rotation_std: tuple[float, float, float]
    init_translation_std: tuple[float, float, float]
    seed: int
    mrcp_invert: bool
    real_xray_invert: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize MRCP projection pose with frozen CoMIR features")
    parser.add_argument("--mode", choices=["virtual", "real"], default="virtual")
    parser.add_argument("--projection", choices=["sum", "max"], default="sum")
    parser.add_argument("--index", type=int, default=31)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--patch-centers", type=int, default=32)
    parser.add_argument("--rotation-lr", type=float, default=0.03)
    parser.add_argument("--translation-lr", type=float, default=2.0)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--contrast-multiplier", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--mrcp-invert", action="store_true")
    parser.add_argument("--real-xray-invert", action="store_true")
    parser.add_argument("--init-rotation-std", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--init-translation-std", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def pose_to_numpy(pose: RigidTransform) -> np.ndarray:
    return pose.get_matrix().detach().cpu().numpy()


def make_offset(rotation: Tensor, translation: Tensor) -> RigidTransform:
    return RigidTransform(rotation, translation, "euler_angles", "ZYX")


def normalize_and_crop(renderer: SXHFeatureTrainingRenderer, crop: CanonicalSquareCrop, image: Tensor) -> Tensor:
    return crop(renderer.normalizer(image).to(torch.float32))


def render_ct_drr(
    renderer: SXHFeatureTrainingRenderer,
    crop: CanonicalSquareCrop,
    pose: RigidTransform,
    contrast_multiplier: float,
) -> Tensor:
    renderer.adjuster.drr.set_bone_attenuation_multiplier(contrast_multiplier)
    raw = renderer.adjuster.drr(None, None, None, pose=pose)
    return normalize_and_crop(renderer, crop, raw)


def render_mrcp_projection(
    renderer: SXHFeatureTrainingRenderer,
    crop: CanonicalSquareCrop,
    pose: RigidTransform,
    *,
    invert: bool,
) -> Tensor:
    raw = renderer.adjuster.drr_mrcp(None, None, None, pose=pose)
    image = normalize_and_crop(renderer, crop, raw)
    if invert:
        image = invert_legacy_standardized_intensity(image)
    return image


def load_real_xray(
    renderer: SXHFeatureTrainingRenderer,
    crop: CanonicalSquareCrop,
    index: int,
    *,
    invert: bool = False,
) -> Tensor:
    xray, _ = renderer.adjuster.specimen[index]
    if not isinstance(xray, Tensor):
        xray = torch.as_tensor(xray)
    xray = xray.to(device=renderer.device, dtype=torch.float32)
    if xray.ndim == 2:
        xray = xray.unsqueeze(0).unsqueeze(0)
    elif xray.ndim == 3:
        xray = xray.unsqueeze(0)
    image = normalize_and_crop(renderer, crop, xray)
    if invert:
        image = 1.0 - image
    return image


def display_xray_background(renderer: SXHFeatureTrainingRenderer, index: int) -> np.ndarray:
    xray, _ = renderer.adjuster.specimen[index]
    return renderer.adjuster.transforms(xray, reverse=False).squeeze().detach().cpu().numpy()


def write_centerline_guidewire_overlay(
    path: Path,
    renderer: SXHFeatureTrainingRenderer,
    index: int,
    pose: RigidTransform,
) -> None:
    background = display_xray_background(renderer, index)
    with torch.no_grad():
        projected_centerline = project_points_to_detector(
            renderer.adjuster.centerline_vertices.to(renderer.device),
            pose,
            renderer.adjuster.drr.detector,
        )
    overlay = render_chain_points_overlay(
        background,
        projected_centerline.detach().cpu().numpy(),
        renderer.adjuster.centerline_chains,
        renderer.adjuster.guidewire_points_for_index(index),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), overlay):
        raise IOError(f"Failed to write image: {path}")


def regular_patch_centers(count: int, patch_size: int, image_size: int = 256, device: torch.device | str = "cpu") -> Tensor:
    if count <= 0:
        raise ValueError("count must be positive")
    margin = patch_size // 2
    side = int(math.ceil(math.sqrt(count)))
    coordinates = torch.linspace(margin, image_size - margin - 1, side, device=device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    centers = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
    return centers[:count].round().to(torch.long)


def gather_patches(features: Tensor, centers_yx: Tensor, patch_size: int) -> Tensor:
    patches = []
    half = patch_size // 2
    for center_y, center_x in centers_yx.tolist():
        y0 = int(center_y) - half
        x0 = int(center_x) - half
        patches.append(features[..., y0 : y0 + patch_size, x0 : x0 + patch_size])
    return torch.cat(patches, dim=0)


def feature_patch_loss(xray_features: Tensor, mrcp_features: Tensor, centers_yx: Tensor, patch_size: int) -> Tensor:
    xray_patches = gather_patches(xray_features, centers_yx, patch_size)
    mrcp_patches = gather_patches(mrcp_features, centers_yx, patch_size)
    xray_flat = F.normalize(xray_patches.flatten(1), p=2, dim=1, eps=1e-6)
    mrcp_flat = F.normalize(mrcp_patches.flatten(1), p=2, dim=1, eps=1e-6)
    return (1.0 - (xray_flat * mrcp_flat).sum(dim=1)).mean()


def joint_pca_rgb(a_features: Tensor, b_features: Tensor) -> tuple[np.ndarray, np.ndarray]:
    a = a_features[0].detach().float().cpu().permute(1, 2, 0).reshape(-1, a_features.shape[1])
    b = b_features[0].detach().float().cpu().permute(1, 2, 0).reshape(-1, b_features.shape[1])
    stacked = torch.cat([a, b], dim=0)
    stacked = stacked - stacked.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(stacked, full_matrices=False)
    rgb = (stacked @ vh[:3].T).numpy()
    rgb = (rgb - np.percentile(rgb, 1, axis=0)) / (np.percentile(rgb, 99, axis=0) - np.percentile(rgb, 1, axis=0) + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    image_size = a_features.shape[-1]
    return rgb[: image_size * image_size].reshape(image_size, image_size, 3), rgb[image_size * image_size :].reshape(image_size, image_size, 3)


def write_rgb_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr_u8):
        raise IOError(f"Failed to write image: {path}")


def write_feature_pca(path: Path, fixed_features: Tensor, mrcp_features: Tensor) -> None:
    fixed_rgb, mrcp_rgb = joint_pca_rgb(fixed_features, mrcp_features)
    combined = np.concatenate([fixed_rgb, mrcp_rgb], axis=1)
    write_rgb_png(path, combined)


def write_loss_curve(path: Path, losses: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4), dpi=140)
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("step")
    plt.ylabel("feature patch loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_visuals(
    output_dir: Path,
    renderer: SXHFeatureTrainingRenderer,
    index: int,
    initial_pose: RigidTransform,
    final_pose: RigidTransform,
    fixed_image: Tensor,
    initial_mrcp: Tensor,
    final_mrcp: Tensor,
    fixed_features_initial: Tensor,
    mrcp_features_initial: Tensor,
    fixed_features_final: Tensor,
    mrcp_features_final: Tensor,
    losses: list[float],
) -> None:
    fixed_np = fixed_image[0, 0].detach().cpu().numpy()
    initial_np = initial_mrcp[0, 0].detach().cpu().numpy()
    final_np = final_mrcp[0, 0].detach().cpu().numpy()
    write_gray_png(output_dir / "initial_fixed.png", fixed_np)
    write_gray_png(output_dir / "initial_mrcp.png", initial_np)
    write_gray_png(output_dir / "final_mrcp.png", final_np)
    write_overlay_png(output_dir / "initial_overlay.png", fixed_np, initial_np)
    write_overlay_png(output_dir / "final_overlay.png", fixed_np, final_np)
    write_feature_pca(output_dir / "feature_pca_initial.png", fixed_features_initial, mrcp_features_initial)
    write_feature_pca(output_dir / "feature_pca_final.png", fixed_features_final, mrcp_features_final)
    write_loss_curve(output_dir / "loss_curve.png", losses)
    write_centerline_guidewire_overlay(
        output_dir / "initial_centerline_guidewire_overlay.png",
        renderer,
        index,
        initial_pose,
    )
    write_centerline_guidewire_overlay(
        output_dir / "final_centerline_guidewire_overlay.png",
        renderer,
        index,
        final_pose,
    )


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[CoMIRTwoBranchFeatureNetwork, int]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feature_channels = int(checkpoint.get("config", {}).get("feature_channels", 32))
    model = CoMIRTwoBranchFeatureNetwork(feature_channels=feature_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, int(checkpoint.get("step", -1))


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.patch_size != 32:
        raise ValueError("This first optimizer is fixed to 32x32 feature patches")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    renderer = SXHFeatureTrainingRenderer(projection_mode=args.projection, render_chunk_size=1, device=args.device)
    device = renderer.device
    crop = CanonicalSquareCrop().to(device)
    model, checkpoint_step = load_model(args.checkpoint, device)
    centers_yx = regular_patch_centers(args.patch_centers, args.patch_size, device=device)

    output_dir = args.output_root / f"{args.mode}_xray{args.index:03d}_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    init_rotation = torch.tensor([args.init_rotation_std], dtype=torch.float32, device=device) * torch.randn((1, 3), device=device)
    init_translation = torch.tensor([args.init_translation_std], dtype=torch.float32, device=device) * torch.randn((1, 3), device=device)
    rotation_delta = torch.nn.Parameter(init_rotation)
    translation_delta = torch.nn.Parameter(init_translation)

    optimizer = torch.optim.Adam(
        [
            {"params": [rotation_delta], "lr": args.rotation_lr},
            {"params": [translation_delta], "lr": args.translation_lr},
        ]
    )

    trusted_pose = renderer.trusted_pose
    initial_pose = compose_centered_perturbations(
        trusted_pose,
        renderer.specimen_center_pose,
        make_offset(rotation_delta.detach(), translation_delta.detach()),
    )

    if args.mode == "virtual":
        with torch.no_grad():
            fixed_image = render_ct_drr(renderer, crop, trusted_pose, args.contrast_multiplier)
    else:
        with torch.no_grad():
            fixed_image = load_real_xray(renderer, crop, args.index, invert=args.real_xray_invert)

    with torch.no_grad():
        initial_mrcp = render_mrcp_projection(renderer, crop, initial_pose, invert=args.mrcp_invert)
        fixed_features_initial, mrcp_features_initial = model(fixed_image, initial_mrcp)

    config = OptimizationConfig(
        mode=args.mode,
        index=args.index,
        projection=args.projection,
        checkpoint=str(args.checkpoint),
        output_dir=str(output_dir),
        device=str(device),
        steps=args.steps,
        patch_size=args.patch_size,
        patch_centers=args.patch_centers,
        rotation_lr=args.rotation_lr,
        translation_lr=args.translation_lr,
        gradient_clip=args.gradient_clip,
        contrast_multiplier=args.contrast_multiplier,
        init_rotation_std=tuple(float(v) for v in args.init_rotation_std),
        init_translation_std=tuple(float(v) for v in args.init_translation_std),
        seed=args.seed,
        mrcp_invert=args.mrcp_invert,
        real_xray_invert=args.real_xray_invert,
    )
    (output_dir / "config.json").write_text(
        json.dumps({"optimizer": asdict(config), "checkpoint_step": checkpoint_step, "renderer": renderer.metadata()}, indent=2),
        encoding="utf-8",
    )
    np.save(output_dir / "initial_pose.npy", pose_to_numpy(initial_pose))

    history: list[dict[str, object]] = []
    losses: list[float] = []
    pose_matrices: list[np.ndarray] = []
    start_time = time.perf_counter()
    log_path = output_dir / "history.jsonl"

    fixed_features = model.xray_net(fixed_image)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        offset = make_offset(rotation_delta, translation_delta)
        pose = compose_centered_perturbations(trusted_pose, renderer.specimen_center_pose, offset)
        moving_mrcp = render_mrcp_projection(renderer, crop, pose, invert=args.mrcp_invert)
        mrcp_features = model.mrcp_net(moving_mrcp)
        loss = feature_patch_loss(fixed_features, mrcp_features, centers_yx, args.patch_size)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite feature pose loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([rotation_delta, translation_delta], args.gradient_clip)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        record = {
            "step": step,
            "loss": float(loss.item()),
            "gradient_norm": float(grad_norm.item()),
            "rotation_delta_zyx_radians": rotation_delta.detach().cpu().tolist()[0],
            "translation_delta_mm": translation_delta.detach().cpu().tolist()[0],
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        losses.append(record["loss"])
        pose_matrices.append(pose_to_numpy(pose)[0])
        history.append(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(json.dumps(record, ensure_ascii=False))

    final_offset = make_offset(rotation_delta.detach(), translation_delta.detach())
    final_pose = compose_centered_perturbations(trusted_pose, renderer.specimen_center_pose, final_offset)
    np.save(output_dir / "final_pose.npy", pose_to_numpy(final_pose))
    np.savez_compressed(output_dir / "pose_history.npz", pose_matrices=np.asarray(pose_matrices))

    with torch.no_grad():
        final_mrcp = render_mrcp_projection(renderer, crop, final_pose, invert=args.mrcp_invert)
        fixed_features_final, mrcp_features_final = model(fixed_image, final_mrcp)
    save_visuals(
        output_dir,
        renderer,
        args.index,
        initial_pose,
        final_pose,
        fixed_image,
        initial_mrcp,
        final_mrcp,
        fixed_features_initial,
        mrcp_features_initial,
        fixed_features_final,
        mrcp_features_final,
        losses,
    )

    summary = {
        "output_dir": str(output_dir),
        "mode": args.mode,
        "checkpoint_step": checkpoint_step,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "best_step": int(np.argmin(losses) + 1),
        "final_rotation_delta_zyx_radians": rotation_delta.detach().cpu().tolist()[0],
        "final_translation_delta_mm": translation_delta.detach().cpu().tolist()[0],
        "history_tail": history[-5:],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
