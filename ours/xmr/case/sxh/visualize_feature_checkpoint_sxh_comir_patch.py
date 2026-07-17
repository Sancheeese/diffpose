"""Visualize SXH two-branch CoMIR patch features from a checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DIFFPOSE_ROOT = PROJECT_ROOT / "diffpose"
OURS_ROOT = DIFFPOSE_ROOT / "ours"
for path in (DIFFPOSE_ROOT, OURS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ours.case.my_util2 import get_random_offset  # noqa: E402
from ours.xmr.case.sxh.feature_training_renderer import (  # noqa: E402
    SXHFeatureTrainingRenderer,
    compose_centered_perturbations,
)
from ours.xmr.feature_network_comir import CoMIRTwoBranchFeatureNetwork, SymmetricPatchInfoNCE  # noqa: E402


DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "runs" / "feature_common_comir_patch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SXH CoMIR patch feature checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN_DIR / "checkpoints" / "last.pt")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--anchors", type=int, default=4)
    parser.add_argument("--metric-samples", type=int, default=128)
    parser.add_argument("--contrast-multiplier", type=float, default=3.0)
    parser.add_argument("--patch-size", type=int, default=32)
    return parser.parse_args()


def load_checkpoint_with_retry(path: Path, device: torch.device, attempts: int = 8) -> dict:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except (EOFError, RuntimeError, OSError) as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Could not load checkpoint after {attempts} attempts: {path}") from last_error


def to_display(image: torch.Tensor) -> np.ndarray:
    array = image.detach().float().cpu().numpy()
    low, high = np.percentile(array, (1, 99))
    return np.clip((array - low) / max(high - low, 1e-6), 0, 1)


def joint_pca_rgb(feature_maps: list[torch.Tensor], valid_mask: torch.Tensor) -> list[np.ndarray]:
    mask = valid_mask[0, 0].detach().cpu().numpy().astype(bool)
    vectors = np.concatenate([features[0, :, mask].detach().float().cpu().numpy().T for features in feature_maps], axis=0)
    mean = vectors.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(vectors - mean, full_matrices=False)
    projected = []
    all_values = []
    for features in feature_maps:
        values = features[0].detach().float().cpu().numpy().transpose(1, 2, 0)
        rgb = (values - mean) @ components[:3].T
        projected.append(rgb)
        all_values.append(rgb[mask])
    scale_values = np.concatenate(all_values, axis=0)
    low = np.percentile(scale_values, 1, axis=0)
    high = np.percentile(scale_values, 99, axis=0)
    result = []
    for rgb in projected:
        rgb = np.clip((rgb - low) / np.maximum(high - low, 1e-6), 0, 1)
        rgb[~mask] = 0
        result.append(rgb)
    return result


def valid_patch_center_mask(valid_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    mask = valid_mask[:, :1].float()
    kernel = torch.ones((1, 1, patch_size, patch_size), dtype=torch.float32, device=valid_mask.device)
    counts = F.conv2d(mask, kernel)
    valid_top_left = counts.eq(float(patch_size * patch_size))
    center_mask = torch.zeros_like(mask, dtype=torch.bool)
    half = patch_size // 2
    center_mask[:, :, half : half + valid_top_left.shape[-2], half : half + valid_top_left.shape[-1]] = valid_top_left
    return center_mask


def select_anchors(
    valid_centers: torch.Tensor,
    bile_projection: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    valid = torch.nonzero(valid_centers[0, 0], as_tuple=False)
    bile = torch.nonzero((bile_projection[0, 0] > 0.01) & valid_centers[0, 0], as_tuple=False)
    selected: list[torch.Tensor] = []
    bile_count = min(count // 2, len(bile))
    if bile_count:
        selected.extend(bile[torch.randperm(len(bile), generator=generator, device=bile.device)[:bile_count]])
    remaining = count - len(selected)
    selected.extend(valid[torch.randperm(len(valid), generator=generator, device=valid.device)[:remaining]])
    return torch.stack(selected)


def patch_similarity_map(query_map: torch.Tensor, key_map: torch.Tensor, center_yx: torch.Tensor, patch_size: int) -> torch.Tensor:
    half = patch_size // 2
    y, x = [int(value) for value in center_yx.tolist()]
    query_patch = query_map[:, y - half : y + patch_size - half, x - half : x + patch_size - half]
    query_patch = F.normalize(query_patch.reshape(1, -1).float(), p=2, dim=1, eps=1e-6)
    weight = query_patch.reshape(1, key_map.shape[0], patch_size, patch_size)
    raw = F.conv2d(key_map[None].float(), weight)[0, 0]
    patch_norm = F.conv2d(key_map[None].float().square().sum(dim=1, keepdim=True), torch.ones((1, 1, patch_size, patch_size), device=key_map.device))[0, 0]
    patch_norm = patch_norm.clamp_min(1e-12).sqrt()
    similarity_top_left = raw / patch_norm
    similarity = torch.full(key_map.shape[-2:], -torch.inf, dtype=torch.float32, device=key_map.device)
    similarity[half : half + similarity_top_left.shape[0], half : half + similarity_top_left.shape[1]] = similarity_top_left
    return similarity


def patch_correspondence_metrics(
    query_map: torch.Tensor,
    key_map: torch.Tensor,
    valid_centers: torch.Tensor,
    patch_size: int,
    sample_count: int,
    generator: torch.Generator,
) -> dict[str, float]:
    centers = torch.nonzero(valid_centers[0, 0], as_tuple=False)
    selected = centers[torch.randperm(len(centers), generator=generator, device=centers.device)[:sample_count]]
    errors = []
    for center in selected:
        similarity = patch_similarity_map(query_map[0], key_map[0], center, patch_size).masked_fill(~valid_centers[0, 0], -torch.inf)
        predicted = torch.nonzero(similarity == similarity.max(), as_tuple=False)[0]
        errors.append(torch.linalg.vector_norm((predicted - center).float()))
    errors_tensor = torch.stack(errors)
    return {
        "mean_error_px": float(errors_tensor.mean().item()),
        "median_error_px": float(errors_tensor.median().item()),
        "recall_at_1px": float((errors_tensor <= 1).float().mean().item()),
        "recall_at_2px": float((errors_tensor <= 2).float().mean().item()),
        "recall_at_5px": float((errors_tensor <= 5).float().mean().item()),
    }


def draw_feature_overview(output_path: Path, pairs: list[dict], pca_maps: list[np.ndarray]) -> None:
    figure, axes = plt.subplots(len(pairs), 4, figsize=(16, 8 * len(pairs)))
    if len(pairs) == 1:
        axes = axes[None, :]
    for row, pair in enumerate(pairs):
        entries = [
            (to_display(pair["batch"].ct_drr[0, 0]), "CT-DRR"),
            (to_display(pair["batch"].mrcp_projection[0, 0]), "MRCP max"),
            (pca_maps[row * 2], "xray CoMIR PCA"),
            (pca_maps[row * 2 + 1], "mrcp CoMIR PCA"),
        ]
        for axis, (image, title) in zip(axes[row], entries):
            axis.imshow(image, cmap=None if image.ndim == 3 else "gray")
            axis.set_title(f"{pair['name']}: {title}")
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def draw_anchor_maps(output_path: Path, pair: dict, anchors: torch.Tensor, valid_centers: torch.Tensor, patch_size: int) -> list[dict]:
    figure, axes = plt.subplots(len(anchors), 3, figsize=(12, 4 * len(anchors)))
    if len(anchors) == 1:
        axes = axes[None, :]
    ct_image = to_display(pair["batch"].ct_drr[0, 0])
    mrcp_image = to_display(pair["batch"].mrcp_projection[0, 0])
    half = patch_size // 2
    records = []
    for row, center in enumerate(anchors):
        y, x = [int(value) for value in center.tolist()]
        similarity = patch_similarity_map(pair["xray_features"][0], pair["mrcp_features"][0], center, patch_size)
        similarity = similarity.masked_fill(~valid_centers[0, 0], -torch.inf)
        predicted = torch.nonzero(similarity == similarity.max(), as_tuple=False)[0]
        pred_y, pred_x = [int(value) for value in predicted.tolist()]
        error = float(torch.linalg.vector_norm((predicted - center).float()).item())

        axes[row, 0].imshow(ct_image, cmap="gray")
        axes[row, 0].add_patch(Rectangle((x - half, y - half), patch_size, patch_size, fill=False, color="lime", linewidth=2))
        axes[row, 0].set_title(f"CT patch center ({x}, {y})")
        axes[row, 1].imshow(mrcp_image, cmap="gray")
        axes[row, 1].add_patch(Rectangle((x - half, y - half), patch_size, patch_size, fill=False, color="lime", linewidth=2))
        axes[row, 1].add_patch(Rectangle((pred_x - half, pred_y - half), patch_size, patch_size, fill=False, color="red", linewidth=2))
        axes[row, 1].set_title(f"MRCP true/pred error={error:.1f}px")
        heatmap = similarity.detach().float().cpu().numpy()
        finite = heatmap[np.isfinite(heatmap)]
        axes[row, 2].imshow(heatmap, cmap="magma", vmin=np.percentile(finite, 1), vmax=np.max(finite))
        axes[row, 2].add_patch(Rectangle((x - half, y - half), patch_size, patch_size, fill=False, color="lime", linewidth=2))
        axes[row, 2].add_patch(Rectangle((pred_x - half, pred_y - half), patch_size, patch_size, fill=False, color="cyan", linewidth=2))
        axes[row, 2].set_title("xray patch similarity over MRCP")
        for axis in axes[row]:
            axis.axis("off")
        records.append({"anchor_yx": [y, x], "predicted_yx": [pred_y, pred_x], "error_px": error})
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return records


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This visualization requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Use a CUDA device")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = load_checkpoint_with_retry(args.checkpoint, device)
    checkpoint_step = int(checkpoint["step"])
    output_dir = args.output_dir or args.checkpoint.parent.parent / "visualizations" / f"comir_patch_step_{checkpoint_step:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = SXHFeatureTrainingRenderer(projection_mode="max", render_chunk_size=2, device=args.device)
    model = CoMIRTwoBranchFeatureNetwork(feature_channels=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    center_batch = renderer.render_poses(
        renderer.trusted_pose,
        contrast_multiplier=args.contrast_multiplier,
        include_bile_projection=True,
    )
    offset = get_random_offset(1, device)
    perturbed_pose = compose_centered_perturbations(renderer.trusted_pose, renderer.specimen_center_pose, offset)
    perturbed_batch = renderer.render_poses(
        perturbed_pose,
        contrast_multiplier=args.contrast_multiplier,
        include_bile_projection=True,
        offsets=offset,
    )

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        center_xray_features, center_mrcp_features = model(center_batch.ct_drr, center_batch.mrcp_projection)
        perturbed_xray_features, perturbed_mrcp_features = model(perturbed_batch.ct_drr, perturbed_batch.mrcp_projection)

    pairs = [
        {
            "name": "center pose",
            "batch": center_batch,
            "xray_features": center_xray_features,
            "mrcp_features": center_mrcp_features,
        },
        {
            "name": "perturbed pose",
            "batch": perturbed_batch,
            "xray_features": perturbed_xray_features,
            "mrcp_features": perturbed_mrcp_features,
        },
    ]
    pca_maps = joint_pca_rgb(
        [center_xray_features, center_mrcp_features, perturbed_xray_features, perturbed_mrcp_features],
        center_batch.valid_mask,
    )
    draw_feature_overview(output_dir / "feature_overview.png", pairs, pca_maps)

    criterion = SymmetricPatchInfoNCE(patches_per_image=32, patch_size=args.patch_size)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "device": str(device),
        "patch_size": args.patch_size,
        "contrast_multiplier": args.contrast_multiplier,
        "pairs": {},
        "perturbed_offset_euler_zyx_radians": offset.get_rotation("euler_angles", "ZYX").detach().cpu().tolist(),
        "perturbed_offset_translation_mm": offset.get_translation().detach().cpu().tolist(),
    }
    for index, pair in enumerate(pairs):
        valid_centers = valid_patch_center_mask(pair["batch"].valid_mask, args.patch_size)
        anchors = select_anchors(valid_centers, pair["batch"].bile_projection, args.anchors, generator)
        loss_sample = criterion(pair["xray_features"], pair["mrcp_features"], pair["batch"].valid_mask)
        anchor_records = draw_anchor_maps(
            output_dir / f"patch_correspondence_{index:02d}.png",
            pair,
            anchors,
            valid_centers,
            args.patch_size,
        )
        summary["pairs"][pair["name"]] = {
            "sampled_patch_loss": float(loss_sample.total.item()),
            "sampled_xray_to_mrcp": float(loss_sample.xray_to_mrcp.item()),
            "sampled_mrcp_to_xray": float(loss_sample.mrcp_to_xray.item()),
            "xray_to_mrcp_patch_metrics": patch_correspondence_metrics(
                pair["xray_features"],
                pair["mrcp_features"],
                valid_centers,
                args.patch_size,
                args.metric_samples,
                generator,
            ),
            "mrcp_to_xray_patch_metrics": patch_correspondence_metrics(
                pair["mrcp_features"],
                pair["xray_features"],
                valid_centers,
                args.patch_size,
                args.metric_samples,
                generator,
            ),
            "anchors": anchor_records,
        }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"checkpoint step: {checkpoint_step}")
    print(f"visualizations: {output_dir}")
    print(json.dumps(summary["pairs"], indent=2))


if __name__ == "__main__":
    main()
