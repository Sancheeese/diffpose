"""Visualize CT-DRR/MRCP common descriptors from an SXH training checkpoint."""

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
from matplotlib.patches import Circle
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
from ours.xmr.feature_network import CommonFeatureNetwork  # noqa: E402


DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "runs" / "feature_common"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained SXH CT-DRR/MRCP feature checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN_DIR / "checkpoints" / "last.pt")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--anchors", type=int, default=4)
    parser.add_argument("--metric-samples", type=int, default=128)
    parser.add_argument("--contrast-multiplier", type=float, default=3.0)
    return parser.parse_args()


def load_checkpoint_with_retry(path: Path, device: torch.device, attempts: int = 8) -> dict:
    """Read a checkpoint safely while training may atomically replace it soon."""
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


def joint_pca_rgb(descriptor_maps: list[torch.Tensor], valid_mask: torch.Tensor) -> list[np.ndarray]:
    """Project all descriptor maps into a common three-channel PCA color space."""
    mask = valid_mask[0, 0].detach().cpu().numpy().astype(bool)
    vectors = np.concatenate(
        [descriptor[0, :, mask].detach().float().cpu().numpy().T for descriptor in descriptor_maps], axis=0
    )
    mean = vectors.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(vectors - mean, full_matrices=False)
    projected = []
    all_values = []
    for descriptor in descriptor_maps:
        values = descriptor[0].detach().float().cpu().numpy().transpose(1, 2, 0)
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


def select_anchors(valid_mask: torch.Tensor, bile_projection: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    valid = torch.nonzero(valid_mask[0, 0], as_tuple=False)
    bile = torch.nonzero((bile_projection[0, 0] > 0.01) & valid_mask[0, 0], as_tuple=False)
    selected: list[torch.Tensor] = []
    bile_count = min(count // 2, len(bile))
    if bile_count:
        selected.extend(bile[torch.randperm(len(bile), generator=generator, device=bile.device)[:bile_count]])
    remaining = count - len(selected)
    selected.extend(valid[torch.randperm(len(valid), generator=generator, device=valid.device)[:remaining]])
    return torch.stack(selected)


def similarity_map(query: torch.Tensor, key_map: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    similarities = torch.einsum("c,chw->hw", query, key_map)
    return similarities.masked_fill(~valid_mask[0, 0], -torch.inf)


def correspondence_metrics(
    queries: torch.Tensor,
    keys: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_count: int,
    generator: torch.Generator,
) -> dict[str, float]:
    coordinates = torch.nonzero(valid_mask[0, 0], as_tuple=False)
    selected = coordinates[torch.randperm(len(coordinates), generator=generator, device=coordinates.device)[:sample_count]]
    key_descriptors = keys[0, :, valid_mask[0, 0]].transpose(0, 1)
    query_descriptors = queries[0, :, selected[:, 0], selected[:, 1]].transpose(0, 1)
    similarities = query_descriptors.float() @ key_descriptors.float().transpose(0, 1)
    predictions = coordinates[similarities.argmax(dim=1)]
    errors = torch.linalg.vector_norm((predictions - selected).float(), dim=1)
    return {
        "mean_error_px": float(errors.mean().item()),
        "median_error_px": float(errors.median().item()),
        "recall_at_1px": float((errors <= 1).float().mean().item()),
        "recall_at_2px": float((errors <= 2).float().mean().item()),
        "recall_at_5px": float((errors <= 5).float().mean().item()),
    }


def draw_feature_overview(output_path: Path, pairs: list[dict], pca_maps: list[np.ndarray]) -> None:
    figure, axes = plt.subplots(len(pairs), 4, figsize=(16, 8 * len(pairs)))
    if len(pairs) == 1:
        axes = axes[None, :]
    for row, pair in enumerate(pairs):
        entries = [
            (to_display(pair["batch"].ct_drr[0, 0]), "CT-DRR"),
            (to_display(pair["batch"].mrcp_projection[0, 0]), "MRCP max"),
            (pca_maps[row * 2], "CT descriptor PCA"),
            (pca_maps[row * 2 + 1], "MRCP descriptor PCA"),
        ]
        for axis, (image, title) in zip(axes[row], entries):
            axis.imshow(image, cmap=None if image.ndim == 3 else "gray")
            axis.set_title(f"{pair['name']}: {title}")
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def draw_anchor_maps(
    output_path: Path,
    pair: dict,
    anchors: torch.Tensor,
    valid_mask: torch.Tensor,
) -> list[dict[str, object]]:
    figure, axes = plt.subplots(len(anchors), 3, figsize=(12, 4 * len(anchors)))
    if len(anchors) == 1:
        axes = axes[None, :]
    ct_image = to_display(pair["batch"].ct_drr[0, 0])
    mrcp_image = to_display(pair["batch"].mrcp_projection[0, 0])
    records = []
    for row, coordinate in enumerate(anchors):
        y, x = [int(value) for value in coordinate.tolist()]
        similarity = similarity_map(pair["ct_features"][0, :, y, x], pair["mrcp_features"][0], valid_mask)
        predicted = torch.nonzero(similarity == similarity.max(), as_tuple=False)[0]
        pred_y, pred_x = [int(value) for value in predicted.tolist()]
        error = float(torch.linalg.vector_norm((predicted - coordinate).float()).item())
        axes[row, 0].imshow(ct_image, cmap="gray")
        axes[row, 0].add_patch(Circle((x, y), radius=4, fill=False, color="lime", linewidth=2))
        axes[row, 0].set_title(f"CT anchor ({x}, {y})")
        axes[row, 1].imshow(mrcp_image, cmap="gray")
        axes[row, 1].add_patch(Circle((x, y), radius=4, fill=False, color="lime", linewidth=2, label="true"))
        axes[row, 1].add_patch(Circle((pred_x, pred_y), radius=4, fill=False, color="red", linewidth=2, label="pred"))
        axes[row, 1].set_title(f"MRCP true/pred error={error:.1f}px")
        heatmap = similarity.detach().float().cpu().numpy()
        axes[row, 2].imshow(heatmap, cmap="magma", vmin=np.nanpercentile(heatmap[np.isfinite(heatmap)], 1), vmax=np.nanmax(heatmap))
        axes[row, 2].add_patch(Circle((x, y), radius=4, fill=False, color="lime", linewidth=2))
        axes[row, 2].add_patch(Circle((pred_x, pred_y), radius=4, fill=False, color="cyan", linewidth=2))
        axes[row, 2].set_title("CT anchor similarity over MRCP")
        for axis in axes[row]:
            axis.axis("off")
        records.append(
            {"anchor_yx": [y, x], "predicted_yx": [pred_y, pred_x], "error_px": error}
        )
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
        raise ValueError("Use a CUDA device so evaluation does not block training")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = load_checkpoint_with_retry(args.checkpoint, device)
    checkpoint_step = int(checkpoint["step"])
    output_dir = args.output_dir or args.checkpoint.parent.parent / "visualizations" / f"step_{checkpoint_step:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = SXHFeatureTrainingRenderer(projection_mode="max", render_chunk_size=2, device=args.device)
    model = CommonFeatureNetwork().to(device)
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
        center_ct_features = model(center_batch.ct_drr)
        center_mrcp_features = model(center_batch.mrcp_projection)
        perturbed_ct_features = model(perturbed_batch.ct_drr)
        perturbed_mrcp_features = model(perturbed_batch.mrcp_projection)

    pairs = [
        {"name": "center pose", "batch": center_batch, "ct_features": center_ct_features, "mrcp_features": center_mrcp_features},
        {
            "name": "perturbed pose",
            "batch": perturbed_batch,
            "ct_features": perturbed_ct_features,
            "mrcp_features": perturbed_mrcp_features,
        },
    ]
    pca_maps = joint_pca_rgb(
        [center_ct_features, center_mrcp_features, perturbed_ct_features, perturbed_mrcp_features],
        center_batch.valid_mask,
    )
    draw_feature_overview(output_dir / "feature_overview.png", pairs, pca_maps)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "device": str(device),
        "contrast_multiplier": args.contrast_multiplier,
        "pairs": {},
        "perturbed_offset_euler_zyx_radians": offset.get_rotation("euler_angles", "ZYX").detach().cpu().tolist(),
        "perturbed_offset_translation_mm": offset.get_translation().detach().cpu().tolist(),
    }
    for index, pair in enumerate(pairs):
        anchors = select_anchors(pair["batch"].valid_mask, pair["batch"].bile_projection, args.anchors, generator)
        anchor_records = draw_anchor_maps(
            output_dir / f"correspondence_{index:02d}.png", pair, anchors, pair["batch"].valid_mask
        )
        summary["pairs"][pair["name"]] = {
            "ct_to_mrcp": correspondence_metrics(
                pair["ct_features"], pair["mrcp_features"], pair["batch"].valid_mask, args.metric_samples, generator
            ),
            "mrcp_to_ct": correspondence_metrics(
                pair["mrcp_features"], pair["ct_features"], pair["batch"].valid_mask, args.metric_samples, generator
            ),
            "anchors": anchor_records,
        }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"checkpoint step: {checkpoint_step}")
    print(f"visualizations: {output_dir}")
    print(json.dumps(summary["pairs"], indent=2))


if __name__ == "__main__":
    main()
