"""Test whether CoMIR patch matching follows MRCP image shifts or coordinates."""

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
from ours.xmr.case.sxh.visualize_feature_checkpoint_sxh_comir_patch import (  # noqa: E402
    load_checkpoint_with_retry,
    patch_similarity_map,
    to_display,
    valid_patch_center_mask,
)
from ours.xmr.feature_network_comir import CoMIRTwoBranchFeatureNetwork  # noqa: E402


DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "runs" / "feature_common_comir_patch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MRCP shift sensitivity for SXH CoMIR patch features")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN_DIR / "checkpoints" / "step_100000.pt")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--contrast-multiplier", type=float, default=3.0)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--shifts", type=int, nargs="+", default=[0, 8, 16, 32, 48])
    return parser.parse_args()


def shift_right(tensor: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels < 0:
        raise ValueError("This test expects non-negative right shifts")
    if pixels == 0:
        return tensor.clone()
    shifted = torch.zeros_like(tensor)
    shifted[..., :, pixels:] = tensor[..., :, :-pixels]
    return shifted


def choose_anchor(
    valid_centers: torch.Tensor,
    bile_projection: torch.Tensor,
    max_shift: int,
    patch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    half = patch_size // 2
    valid = valid_centers[0, 0].clone()
    valid[:, -max_shift:] = False
    valid[:half] = False
    valid[-half:] = False
    bile_candidates = torch.nonzero((bile_projection[0, 0] > 0.01) & valid, as_tuple=False)
    if len(bile_candidates):
        return bile_candidates[torch.randperm(len(bile_candidates), generator=generator, device=bile_candidates.device)[0]]
    candidates = torch.nonzero(valid, as_tuple=False)
    if len(candidates) == 0:
        raise ValueError("No valid anchor can support the requested shifts")
    return candidates[torch.randperm(len(candidates), generator=generator, device=candidates.device)[0]]


def draw_shift_test(
    output_path: Path,
    ct_image: torch.Tensor,
    shifted_cases: list[dict],
    anchor_yx: torch.Tensor,
    patch_size: int,
) -> list[dict[str, object]]:
    rows = len(shifted_cases)
    figure, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    if rows == 1:
        axes = axes[None, :]
    half = patch_size // 2
    y, x = [int(value) for value in anchor_yx.tolist()]
    records = []
    ct_display = to_display(ct_image[0, 0])
    for row, case in enumerate(shifted_cases):
        dx = int(case["dx"])
        true_y = y
        true_x = x + dx
        similarity = case["similarity"]
        predicted = torch.nonzero(similarity == similarity.max(), as_tuple=False)[0]
        pred_y, pred_x = [int(value) for value in predicted.tolist()]
        true_error = float(torch.linalg.vector_norm((predicted - torch.tensor([true_y, true_x], device=predicted.device)).float()).item())
        original_error = float(torch.linalg.vector_norm((predicted - anchor_yx).float()).item())

        axes[row, 0].imshow(ct_display, cmap="gray")
        axes[row, 0].add_patch(Rectangle((x - half, y - half), patch_size, patch_size, fill=False, color="lime", linewidth=2))
        axes[row, 0].set_title(f"CT fixed patch ({x}, {y})")

        axes[row, 1].imshow(to_display(case["mrcp_image"][0, 0]), cmap="gray")
        axes[row, 1].add_patch(Rectangle((true_x - half, true_y - half), patch_size, patch_size, fill=False, color="lime", linewidth=2))
        axes[row, 1].add_patch(Rectangle((x - half, y - half), patch_size, patch_size, fill=False, color="yellow", linewidth=2))
        axes[row, 1].add_patch(Rectangle((pred_x - half, pred_y - half), patch_size, patch_size, fill=False, color="red", linewidth=2))
        axes[row, 1].set_title(f"MRCP shift +{dx}px, true_err={true_error:.1f}, orig_err={original_error:.1f}")

        heatmap = similarity.detach().float().cpu().numpy()
        finite = heatmap[np.isfinite(heatmap)]
        axes[row, 2].imshow(heatmap, cmap="magma", vmin=np.percentile(finite, 1), vmax=np.max(finite))
        axes[row, 2].add_patch(Rectangle((true_x - half, true_y - half), patch_size, patch_size, fill=False, color="lime", linewidth=2))
        axes[row, 2].add_patch(Rectangle((x - half, y - half), patch_size, patch_size, fill=False, color="yellow", linewidth=2))
        axes[row, 2].add_patch(Rectangle((pred_x - half, pred_y - half), patch_size, patch_size, fill=False, color="cyan", linewidth=2))
        axes[row, 2].set_title("similarity heatmap")
        for axis in axes[row]:
            axis.axis("off")
        records.append(
            {
                "dx": dx,
                "anchor_yx": [y, x],
                "true_shifted_yx": [true_y, true_x],
                "predicted_yx": [pred_y, pred_x],
                "true_error_px": true_error,
                "original_coordinate_error_px": original_error,
            }
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
        raise ValueError("Use a CUDA device")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = load_checkpoint_with_retry(args.checkpoint, device)
    checkpoint_step = int(checkpoint["step"])
    output_dir = args.output_dir or args.checkpoint.parent.parent / "visualizations" / f"comir_patch_shift_test_step_{checkpoint_step:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    renderer = SXHFeatureTrainingRenderer(projection_mode="max", render_chunk_size=2, device=args.device)
    model = CoMIRTwoBranchFeatureNetwork(feature_channels=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batch = renderer.render_poses(
        renderer.trusted_pose,
        contrast_multiplier=args.contrast_multiplier,
        include_bile_projection=True,
    )
    valid_centers = valid_patch_center_mask(batch.valid_mask, args.patch_size)
    anchor = choose_anchor(valid_centers, batch.bile_projection, max(args.shifts), args.patch_size, generator)

    shifted_cases = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        xray_features = model.xray_net(batch.ct_drr)
        for dx in args.shifts:
            shifted_mrcp = shift_right(batch.mrcp_projection, dx)
            shifted_mask = shift_right(batch.valid_mask.float(), dx).bool()
            shifted_centers = valid_patch_center_mask(shifted_mask, args.patch_size)
            mrcp_features = model.mrcp_net(shifted_mrcp)
            similarity = patch_similarity_map(xray_features[0], mrcp_features[0], anchor, args.patch_size)
            similarity = similarity.masked_fill(~shifted_centers[0, 0], -torch.inf)
            shifted_cases.append(
                {
                    "dx": dx,
                    "mrcp_image": shifted_mrcp,
                    "similarity": similarity,
                }
            )

    records = draw_shift_test(output_dir / "mrcp_shift_test.png", batch.ct_drr, shifted_cases, anchor, args.patch_size)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "device": str(device),
        "patch_size": args.patch_size,
        "shifts_px_right": args.shifts,
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"checkpoint step: {checkpoint_step}")
    print(f"visualizations: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
