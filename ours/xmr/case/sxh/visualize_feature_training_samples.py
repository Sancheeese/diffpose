"""Render normalized SXH CT/MRCP feature-training samples near the trusted xray031 pose.

The random offset distribution matches ``ours.case.my_util2.get_random_offset``
used by the earlier SXH CT-DRR PoseNet training. The zero-offset pose is the
automatic CT-X-ray registration pose used by the refined MRCP viewer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DIFFPOSE_ROOT = PROJECT_ROOT / "diffpose"
OURS_ROOT = DIFFPOSE_ROOT / "ours"
for path in (DIFFPOSE_ROOT, OURS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from diffpose.calibration import RigidTransform  # noqa: E402
from ours.xmr.feature_network.transformer import PerImageLegacyTransform  # noqa: E402
from ours.xmr.case.sxh.image_io import write_gray_png, write_overlay_png  # noqa: E402
from ours.xmr.case.sxh.web_drr_server_nii_sxh_refined_mrcp import (  # noqa: E402
    DEFAULT_START_INDEX,
    build_adjuster,
    parse_args as parse_refined_server_args,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs" / "feature_training_samples_xray031"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render randomized normalized CT-DRR/MRCP training samples near xray031")
    parser.add_argument("--count", type=int, default=4, help="Number of pose samples to render")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--projection", choices=["sum", "max"], default="max")
    return parser.parse_args()


def sample_old_training_offsets(count: int, device: torch.device) -> tuple[RigidTransform, torch.Tensor, torch.Tensor]:
    """Sample the exact active normal distributions from ``get_random_offset``."""
    rotation = torch.stack(
        [
            torch.distributions.Normal(0, torch.pi / 8).sample((count,)),
            torch.distributions.Normal(0, torch.pi / 10).sample((count,)),
            torch.distributions.Normal(0, torch.pi / 12).sample((count,)),
        ],
        dim=1,
    ).to(device)
    translation = torch.stack(
        [
            torch.distributions.Normal(0, 30).sample((count,)),
            torch.distributions.Normal(0, 50).sample((count,)),
            torch.distributions.Normal(0, 30).sample((count,)),
        ],
        dim=1,
    ).to(device)
    return RigidTransform(rotation, translation, "euler_angles", "ZYX"), rotation, translation


def compose_centered_perturbations(
    center_pose: RigidTransform,
    specimen_center_pose: RigidTransform,
    offsets: RigidTransform,
) -> RigidTransform:
    """Apply old-training local offsets while retaining ``center_pose`` at zero offset.

    The old training samples ``isocenter @ back @ delta @ center``. Replacing
    its zero-offset global pose with the trusted registered pose yields
    ``center_pose @ center^-1 @ delta @ center``.
    """
    return center_pose.compose(specimen_center_pose.inverse()).compose(offsets).compose(specimen_center_pose)


def make_triptych(ct_image: np.ndarray, mrcp_image: np.ndarray, overlay_path: Path, output_path: Path, title: str) -> None:
    ct_u8 = cv2.imread(str(overlay_path.with_name(overlay_path.name.replace("_bile_overlay", "_ct_drr"))), cv2.IMREAD_GRAYSCALE)
    mrcp_u8 = cv2.imread(str(overlay_path.with_name(overlay_path.name.replace("_bile_overlay", "_mrcp"))), cv2.IMREAD_GRAYSCALE)
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    if ct_u8 is None or mrcp_u8 is None or overlay is None:
        raise IOError(f"Could not read rendered panels for {output_path}")

    ct_color = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)
    mrcp_color = cv2.cvtColor(mrcp_u8, cv2.COLOR_GRAY2BGR)
    panels = [ct_color, mrcp_color, overlay]
    labels = ["CT DRR (normalized)", "MRCP max (normalized)", "CT DRR + MRCP bile mask"]
    header_height = 52
    labeled = []
    for panel, label in zip(panels, labels):
        header = np.full((header_height, panel.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(header, label, (8, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        labeled.append(np.vstack([header, panel]))
    montage = np.hstack(labeled)
    title_bar = np.full((38, montage.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(title_bar, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), np.vstack([title_bar, montage])):
        raise IOError(f"Failed to write montage: {output_path}")


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    server_args = parse_refined_server_args(["--projection", args.projection])
    adjuster = build_adjuster(server_args)
    device = torch.device(adjuster.device)
    adjuster.current_index = DEFAULT_START_INDEX
    adjuster.apply_initial_pose(DEFAULT_START_INDEX)
    center_pose = adjuster.get_current_pose()

    offsets, offset_rotation, offset_translation = sample_old_training_offsets(args.count, device)
    poses = compose_centered_perturbations(center_pose, adjuster.center_pose, offsets)

    # The old PoseNet renderer samples one attenuation multiplier per batch.
    contrast = torch.distributions.Uniform(0.5, 8.0).sample().item()
    with torch.no_grad():
        ct_raw = adjuster.drr(None, None, None, pose=poses, bone_attenuation_multiplier=contrast)
        mrcp_raw = adjuster.drr_mrcp(None, None, None, pose=poses)
        bile_mask = adjuster.drr_bone(None, None, None, pose=poses)

        # Preserve the legacy transform's remaining operations but normalize
        # every rendered image independently, rather than across the batch.
        normalizer = PerImageLegacyTransform(256, radius=119)
        ct_normalized = normalizer(ct_raw).to(torch.float32)
        mrcp_normalized = normalizer(mrcp_raw).to(torch.float32)
        bile_mask = normalizer.resize(bile_mask).to(torch.float32)

    if not torch.isfinite(ct_normalized).all() or not torch.isfinite(mrcp_normalized).all():
        raise RuntimeError("Non-finite normalized training image")

    output_dir = args.output_dir / f"seed_{args.seed}_n{args.count}_{args.projection}"
    output_dir.mkdir(parents=True, exist_ok=True)
    global_rotation = poses.get_rotation("euler_angles", "ZYX").detach().cpu().numpy()
    global_translation = poses.get_translation().detach().cpu().numpy()
    records = []

    for index in range(args.count):
        stem = f"sample_{index:02d}"
        ct_path = output_dir / f"{stem}_ct_drr.png"
        mrcp_path = output_dir / f"{stem}_mrcp.png"
        overlay_path = output_dir / f"{stem}_bile_overlay.png"
        montage_path = output_dir / f"{stem}_triptych.png"
        ct_image = ct_normalized[index, 0].detach().cpu().numpy()
        mrcp_image = mrcp_normalized[index, 0].detach().cpu().numpy()
        bile_image = bile_mask[index, 0].detach().cpu().numpy()
        write_gray_png(ct_path, ct_image)
        write_gray_png(mrcp_path, mrcp_image)
        write_overlay_png(overlay_path, ct_image, bile_image)
        title = (
            f"sample {index:02d} | delta rot(deg)="
            f"{np.rad2deg(offset_rotation[index].detach().cpu().numpy()).round(2).tolist()} | "
            f"delta trans(mm)={offset_translation[index].detach().cpu().numpy().round(2).tolist()}"
        )
        make_triptych(ct_image, mrcp_image, overlay_path, montage_path, title)
        records.append(
            {
                "sample": index,
                "contrast_multiplier": contrast,
                "offset_euler_zyx_radians": offset_rotation[index].detach().cpu().numpy().tolist(),
                "offset_euler_zyx_degrees": np.rad2deg(offset_rotation[index].detach().cpu().numpy()).tolist(),
                "offset_translation_mm": offset_translation[index].detach().cpu().numpy().tolist(),
                "global_euler_zyx_radians": global_rotation[index].tolist(),
                "global_translation_mm": global_translation[index].tolist(),
                "ct_drr": str(ct_path),
                "mrcp": str(mrcp_path),
                "bile_overlay": str(overlay_path),
                "triptych": str(montage_path),
            }
        )

    manifest = {
        "seed": args.seed,
        "sample_count": args.count,
        "mrcp_projection_mode": args.projection,
        "center_pose": {
            "euler_zyx_radians": center_pose.get_rotation("euler_angles", "ZYX").detach().cpu().numpy()[0].tolist(),
            "translation_mm": center_pose.get_translation().detach().cpu().numpy()[0].tolist(),
        },
        "old_training_offset_distribution": {
            "rotation_std_radians": [float(torch.pi / 8), float(torch.pi / 10), float(torch.pi / 12)],
            "translation_std_mm": [30.0, 50.0, 30.0],
            "ct_bone_attenuation_multiplier_uniform": [0.5, 8.0],
        },
        "normalization": "ours.xmr.feature_network.PerImageLegacyTransform(256, radius=119): resize, per-image min-max, inversion, circular FOV, Normalize(0.3080, 0.1494)",
        "samples": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved {args.count} samples to: {output_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
