"""Visualize CoMIR patch features for two identical and two different poses."""

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
from ours.xmr.case.sxh.visualize_feature_checkpoint_sxh_comir_patch import (  # noqa: E402
    draw_feature_overview,
    joint_pca_rgb,
    load_checkpoint_with_retry,
)
from ours.xmr.feature_network_comir import CoMIRTwoBranchFeatureNetwork  # noqa: E402


DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "runs" / "feature_common_comir_patch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize four SXH CoMIR pose cases")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN_DIR / "checkpoints" / "step_047000.pt")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--contrast-multiplier", type=float, default=3.0)
    return parser.parse_args()


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
    output_dir = args.output_dir or args.checkpoint.parent.parent / "visualizations" / f"comir_patch_4poses_step_{checkpoint_step:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = SXHFeatureTrainingRenderer(projection_mode="max", render_chunk_size=2, device=args.device)
    model = CoMIRTwoBranchFeatureNetwork(feature_channels=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Cases 1 and 2 deliberately use the exact same pose. Cases 3 and 4 use
    # two independently sampled local perturbations.
    offsets = [
        None,
        None,
        get_random_offset(1, device),
        get_random_offset(1, device),
    ]
    pairs = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for index, offset in enumerate(offsets):
            if offset is None:
                pose = renderer.trusted_pose
                name = f"same center pose {index + 1}"
            else:
                pose = compose_centered_perturbations(renderer.trusted_pose, renderer.specimen_center_pose, offset)
                name = f"different pose {index - 1}"
            batch = renderer.render_poses(
                pose,
                contrast_multiplier=args.contrast_multiplier,
                include_bile_projection=False,
                offsets=offset,
            )
            xray_features, mrcp_features = model(batch.ct_drr, batch.mrcp_projection)
            pairs.append(
                {
                    "name": name,
                    "batch": batch,
                    "xray_features": xray_features,
                    "mrcp_features": mrcp_features,
                    "offset": offset,
                }
            )

    pca_maps = joint_pca_rgb(
        [feature for pair in pairs for feature in (pair["xray_features"], pair["mrcp_features"])],
        pairs[0]["batch"].valid_mask,
    )
    draw_feature_overview(output_dir / "feature_overview_4poses.png", pairs, pca_maps)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "device": str(device),
        "cases": [
            {
                "name": pair["name"],
                "offset_euler_zyx_radians": None
                if pair["offset"] is None
                else pair["offset"].get_rotation("euler_angles", "ZYX").detach().cpu().tolist(),
                "offset_translation_mm": None
                if pair["offset"] is None
                else pair["offset"].get_translation().detach().cpu().tolist(),
            }
            for pair in pairs
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"checkpoint step: {checkpoint_step}")
    print(f"visualizations: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
