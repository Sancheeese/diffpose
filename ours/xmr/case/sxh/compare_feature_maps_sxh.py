"""Compare virtual X-ray, real X-ray, and MRCP projection feature maps for SXH."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

from ours.xmr.case.sxh.feature_training_renderer import SXHFeatureTrainingRenderer, compose_centered_perturbations  # noqa: E402
from ours.xmr.case.sxh.image_io import to_zero_one, write_gray_png, write_overlay_png  # noqa: E402
from ours.xmr.case.sxh.optimize_feature_pose_sxh import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    load_model,
    load_real_xray,
    make_offset,
    render_ct_drr,
    render_mrcp_projection,
    timestamp,
)
from ours.xmr.feature_network_comir_v2 import CanonicalSquareCrop  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / "feature_map_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SXH cross-modal feature maps")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--index", type=int, default=31)
    parser.add_argument("--projection", choices=["max", "sum"], default="max")
    parser.add_argument("--mrcp-registration", choices=["refined", "original"], default="refined")
    parser.add_argument("--contrast-multiplier", type=float, default=3.0)
    parser.add_argument("--mrcp-invert", action="store_true")
    parser.add_argument("--real-xray-invert", action="store_true")
    parser.add_argument("--rotation-delta", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--translation-delta", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    return parser.parse_args()


def write_rgb_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr_u8):
        raise IOError(f"Failed to write image: {path}")


def joint_pca_rgb(feature_maps: list[Tensor]) -> list[np.ndarray]:
    if not feature_maps:
        raise ValueError("feature_maps cannot be empty")
    channels = feature_maps[0].shape[1]
    flattened = [
        features[0].detach().float().cpu().permute(1, 2, 0).reshape(-1, channels)
        for features in feature_maps
    ]
    stacked = torch.cat(flattened, dim=0)
    stacked = stacked - stacked.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(stacked, full_matrices=False)
    projected = (stacked @ vh[:3].T).numpy()
    lo = np.percentile(projected, 1, axis=0)
    hi = np.percentile(projected, 99, axis=0)
    projected = np.clip((projected - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    image_size = feature_maps[0].shape[-1]
    pixels_per_image = image_size * image_size
    return [
        projected[i * pixels_per_image : (i + 1) * pixels_per_image].reshape(image_size, image_size, 3)
        for i in range(len(feature_maps))
    ]


def feature_energy(features: Tensor) -> np.ndarray:
    return features[0].detach().float().pow(2).mean(dim=0).sqrt().cpu().numpy()


def pixel_cosine(a_features: Tensor, b_features: Tensor) -> np.ndarray:
    a = F.normalize(a_features.detach().float(), p=2, dim=1, eps=1e-6)
    b = F.normalize(b_features.detach().float(), p=2, dim=1, eps=1e-6)
    return (a * b).sum(dim=1)[0].cpu().numpy()


def full_cosine(a_features: Tensor, b_features: Tensor) -> float:
    a = F.normalize(a_features.detach().float().flatten(1), p=2, dim=1, eps=1e-6)
    b = F.normalize(b_features.detach().float().flatten(1), p=2, dim=1, eps=1e-6)
    return float((a * b).sum(dim=1).mean().item())


def plot_panel(
    output_path: Path,
    virtual_xray: Tensor,
    real_xray: Tensor,
    mrcp: Tensor,
    virtual_pca: np.ndarray,
    real_pca: np.ndarray,
    mrcp_pca: np.ndarray,
    virtual_energy: np.ndarray,
    real_energy: np.ndarray,
    mrcp_energy: np.ndarray,
    virtual_mrcp_cosine: np.ndarray,
    real_mrcp_cosine: np.ndarray,
    metrics: dict[str, float],
) -> None:
    images = [
        (virtual_xray[0, 0].detach().cpu().numpy(), "virtual xray input", "gray"),
        (real_xray[0, 0].detach().cpu().numpy(), "real xray input", "gray"),
        (mrcp[0, 0].detach().cpu().numpy(), "mrcp max input", "gray"),
        (virtual_pca, "virtual xray feature PCA", None),
        (real_pca, "real xray feature PCA", None),
        (mrcp_pca, "mrcp feature PCA", None),
        (virtual_energy, "virtual feature energy", "magma"),
        (real_energy, "real feature energy", "magma"),
        (mrcp_energy, "mrcp feature energy", "magma"),
        (virtual_mrcp_cosine, f"pixel cosine virtual-mrcp\nmean={metrics['pixel_cosine_virtual_mrcp_mean']:.3f}", "viridis"),
        (real_mrcp_cosine, f"pixel cosine real-mrcp\nmean={metrics['pixel_cosine_real_mrcp_mean']:.3f}", "viridis"),
        (np.abs(to_zero_one(real_pca) - to_zero_one(virtual_pca)).mean(axis=2), "PCA abs diff real-virtual", "inferno"),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(11, 14), dpi=140)
    for ax, (image, title, cmap) in zip(axes.flat, images):
        if cmap == "viridis":
            im = ax.imshow(image, cmap=cmap, vmin=-1.0, vmax=1.0)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        elif cmap is None:
            ax.imshow(image)
        else:
            ax.imshow(to_zero_one(image), cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        "SXH feature map comparison | "
        f"full cosine virtual-mrcp={metrics['full_cosine_virtual_mrcp']:.4f}, "
        f"real-mrcp={metrics['full_cosine_real_mrcp']:.4f}, "
        f"real-virtual={metrics['full_cosine_real_virtual']:.4f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    renderer = SXHFeatureTrainingRenderer(
        projection_mode=args.projection,
        render_chunk_size=1,
        device=args.device,
        mrcp_registration=args.mrcp_registration,
    )
    device = renderer.device
    crop = CanonicalSquareCrop().to(device)
    model, checkpoint_step = load_model(args.checkpoint, device)

    rotation = torch.tensor([args.rotation_delta], dtype=torch.float32, device=device)
    translation = torch.tensor([args.translation_delta], dtype=torch.float32, device=device)
    pose = compose_centered_perturbations(renderer.trusted_pose, renderer.specimen_center_pose, make_offset(rotation, translation))

    real_suffix = "_realinv" if args.real_xray_invert else ""
    output_dir = args.output_root / f"xray{args.index:03d}_{args.projection}_{args.mrcp_registration}{real_suffix}_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        virtual_xray = render_ct_drr(renderer, crop, renderer.trusted_pose, args.contrast_multiplier)
        real_xray = load_real_xray(renderer, crop, args.index, invert=args.real_xray_invert)
        mrcp = render_mrcp_projection(renderer, crop, pose, invert=args.mrcp_invert)
        virtual_features = model.xray_net(virtual_xray)
        real_features = model.xray_net(real_xray)
        mrcp_features = model.mrcp_net(mrcp)

    virtual_pca, real_pca, mrcp_pca = joint_pca_rgb([virtual_features, real_features, mrcp_features])
    virtual_energy = feature_energy(virtual_features)
    real_energy = feature_energy(real_features)
    mrcp_energy = feature_energy(mrcp_features)
    virtual_mrcp_cosine = pixel_cosine(virtual_features, mrcp_features)
    real_mrcp_cosine = pixel_cosine(real_features, mrcp_features)
    real_virtual_cosine = pixel_cosine(real_features, virtual_features)

    metrics = {
        "checkpoint_step": float(checkpoint_step),
        "mrcp_registration": args.mrcp_registration,
        "real_xray_invert": bool(args.real_xray_invert),
        "full_cosine_virtual_mrcp": full_cosine(virtual_features, mrcp_features),
        "full_cosine_real_mrcp": full_cosine(real_features, mrcp_features),
        "full_cosine_real_virtual": full_cosine(real_features, virtual_features),
        "pixel_cosine_virtual_mrcp_mean": float(virtual_mrcp_cosine.mean()),
        "pixel_cosine_real_mrcp_mean": float(real_mrcp_cosine.mean()),
        "pixel_cosine_real_virtual_mean": float(real_virtual_cosine.mean()),
        "pixel_cosine_virtual_mrcp_p95": float(np.percentile(virtual_mrcp_cosine, 95)),
        "pixel_cosine_real_mrcp_p95": float(np.percentile(real_mrcp_cosine, 95)),
        "pixel_cosine_real_virtual_p95": float(np.percentile(real_virtual_cosine, 95)),
    }

    write_gray_png(output_dir / "virtual_xray.png", virtual_xray[0, 0].detach().cpu().numpy())
    write_gray_png(output_dir / "real_xray.png", real_xray[0, 0].detach().cpu().numpy())
    write_gray_png(output_dir / "mrcp_projection.png", mrcp[0, 0].detach().cpu().numpy())
    write_overlay_png(
        output_dir / "virtual_xray_mrcp_overlay.png",
        virtual_xray[0, 0].detach().cpu().numpy(),
        mrcp[0, 0].detach().cpu().numpy(),
    )
    write_overlay_png(
        output_dir / "real_xray_mrcp_overlay.png",
        real_xray[0, 0].detach().cpu().numpy(),
        mrcp[0, 0].detach().cpu().numpy(),
    )
    for name, image in [
        ("virtual_xray_feature_pca.png", virtual_pca),
        ("real_xray_feature_pca.png", real_pca),
        ("mrcp_feature_pca.png", mrcp_pca),
    ]:
        write_rgb_png(output_dir / name, image)
    write_gray_png(output_dir / "virtual_mrcp_pixel_cosine.png", virtual_mrcp_cosine)
    write_gray_png(output_dir / "real_mrcp_pixel_cosine.png", real_mrcp_cosine)
    write_gray_png(output_dir / "real_virtual_pixel_cosine.png", real_virtual_cosine)
    plot_panel(
        output_dir / "feature_map_comparison_summary.png",
        virtual_xray,
        real_xray,
        mrcp,
        virtual_pca,
        real_pca,
        mrcp_pca,
        virtual_energy,
        real_energy,
        mrcp_energy,
        virtual_mrcp_cosine,
        real_mrcp_cosine,
        metrics,
    )
    (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
