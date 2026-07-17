"""Visualize anti-shortcut CoMIR features and true augmented patch correspondences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DIFFPOSE_ROOT = PROJECT_ROOT / "diffpose"
OURS_ROOT = DIFFPOSE_ROOT / "ours"
for path in (DIFFPOSE_ROOT, OURS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ours.xmr.case.sxh.feature_training_renderer import SXHFeatureTrainingRenderer  # noqa: E402
from ours.xmr.feature_network_comir_v2 import (  # noqa: E402
    CanonicalSquareCrop,
    CoMIRTwoBranchFeatureNetwork,
    CrossModalPatchInfoNCE,
    IndependentCropC4Augment,
)


DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "runs" / "feature_common_comir_antishortcut"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize anti-shortcut CoMIR checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN_DIR / "checkpoints" / "last.pt")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--contrast-multiplier", type=float, default=3.0)
    return parser.parse_args()


def display(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().float().cpu().numpy()
    low, high = np.percentile(image_np, (1, 99))
    return np.clip((image_np - low) / max(high - low, 1e-6), 0, 1)


def joint_pca_rgb(xray_features: torch.Tensor, mrcp_features: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    vectors = torch.cat((xray_features[0].flatten(1).T, mrcp_features[0].flatten(1).T), dim=0).float().cpu().numpy()
    mean = vectors.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(vectors - mean, full_matrices=False)
    rendered = []
    for features in (xray_features, mrcp_features):
        values = features[0].detach().float().cpu().numpy().transpose(1, 2, 0)
        rgb = (values - mean) @ components[:3].T
        low, high = np.percentile(rgb.reshape(-1, 3), (1, 99), axis=0)
        rendered.append(np.clip((rgb - low) / np.maximum(high - low, 1e-6), 0, 1))
    return rendered[0], rendered[1]


def draw_overview(path: Path, augmented, xray_features: torch.Tensor, mrcp_features: torch.Tensor, centers: torch.Tensor, augment) -> dict[str, list[list[float]]]:
    xray_centers = augment.canonical_to_output(centers, augmented.parameters, "xray")[0]
    mrcp_centers = augment.canonical_to_output(centers, augmented.parameters, "mrcp")[0]
    xray_pca, mrcp_pca = joint_pca_rgb(xray_features, mrcp_features)
    figure, axes = plt.subplots(2, 2, figsize=(14, 14))
    entries = [
        (axes[0, 0], display(augmented.xray[0, 0]), "Augmented CT-DRR", xray_centers, "gray"),
        (axes[0, 1], display(augmented.mrcp[0, 0]), "Augmented MRCP max", mrcp_centers, "gray"),
        (axes[1, 0], xray_pca, "X-ray CoMIR PCA", xray_centers, None),
        (axes[1, 1], mrcp_pca, "MRCP CoMIR PCA", mrcp_centers, None),
    ]
    colors = plt.cm.tab20(np.linspace(0, 1, len(centers[0])))
    for axis, image, title, mapped_centers, cmap in entries:
        axis.imshow(image, cmap=cmap)
        for index, (y, x) in enumerate(mapped_centers.detach().cpu().tolist()):
            axis.add_patch(Rectangle((x - 16, y - 16), 32, 32, fill=False, color=colors[index], linewidth=1.1))
            axis.text(x, y, str(index + 1), color=colors[index], fontsize=7, ha="center", va="center")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {
        "canonical_centers_yx": centers[0].detach().cpu().tolist(),
        "xray_centers_yx": xray_centers.detach().cpu().tolist(),
        "mrcp_centers_yx": mrcp_centers.detach().cpu().tolist(),
    }


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    torch.manual_seed(args.seed)
    renderer = SXHFeatureTrainingRenderer(projection_mode="max", render_chunk_size=1, device=args.device)
    device = renderer.device
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CoMIRTwoBranchFeatureNetwork().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    crop = CanonicalSquareCrop().to(device)
    augment = IndependentCropC4Augment().to(device)
    criterion = CrossModalPatchInfoNCE().to(device)

    batch = renderer.render_poses(renderer.trusted_pose, contrast_multiplier=args.contrast_multiplier, include_bile_projection=False)
    augmented = augment(crop(batch.ct_drr), crop(batch.mrcp_projection))
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        xray_features, mrcp_features = model(augmented.xray, augmented.mrcp)
        loss = criterion(xray_features, mrcp_features, augmented.parameters, augment)

    step = int(checkpoint["step"])
    output_dir = args.output_dir or args.checkpoint.parent.parent / "visualizations" / f"antishortcut_step_{step:06d}"
    coordinates = draw_overview(output_dir / "overview.png", augmented, xray_features, mrcp_features, loss.canonical_centers_yx, augment)
    summary = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "loss": float(loss.total.item()),
        "xray_to_mrcp": float(loss.xray_to_mrcp.item()),
        "mrcp_to_xray": float(loss.mrcp_to_xray.item()),
        "descriptor_shape": list(loss.descriptor_shape),
        "positive_patches_per_query": 1,
        "cross_modal_negative_patches_per_query": 23,
        "crop_size": augmented.parameters.crop_size.detach().cpu().tolist(),
        "xray_origin_yx": augmented.parameters.xray_origin_yx.detach().cpu().tolist(),
        "mrcp_origin_yx": augmented.parameters.mrcp_origin_yx.detach().cpu().tolist(),
        "xray_rotation_k": augmented.parameters.xray_rotation_k.detach().cpu().tolist(),
        "mrcp_rotation_k": augmented.parameters.mrcp_rotation_k.detach().cpu().tolist(),
        **coordinates,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
