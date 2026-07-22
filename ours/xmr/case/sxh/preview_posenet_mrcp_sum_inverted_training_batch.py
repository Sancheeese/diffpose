"""Preview one online training batch for SXH MRCP feature PoseNet."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DIFFPOSE_ROOT = PROJECT_ROOT / "diffpose"
OURS_ROOT = DIFFPOSE_ROOT / "ours"
for path in (DIFFPOSE_ROOT, OURS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diffpose.calibration import RigidTransform  # noqa: E402
from ours.my_util2 import get_random_offset  # noqa: E402
from ours.xmr.case.sxh.feature_training_renderer import (  # noqa: E402
    SXHFeatureTrainingRenderer,
    compose_centered_perturbations,
)
from ours.xmr.case.sxh.optimize_feature_pose_sxh import load_model as load_feature_model  # noqa: E402
from ours.xmr.case.sxh.train_posenet_sxh_mrcp_sum_inverted_feature import (  # noqa: E402
    DEFAULT_FEATURE_CHECKPOINT,
    DEFAULT_GUIDEWIRE_RESULT,
    augment_feature_input,
    render_mrcp_sum_inverted,
)
from ours.xmr.feature_network_comir_v2 import CanonicalSquareCrop  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs" / "posenet_mrcp_sum_inverted_feature_preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview one PosNet MRCP training batch")
    parser.add_argument("--feature-checkpoint", type=Path, default=DEFAULT_FEATURE_CHECKPOINT)
    parser.add_argument("--guidewire-result", type=Path, default=DEFAULT_GUIDEWIRE_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--render-chunk-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--feature-contrast-min", type=float, default=0.8)
    parser.add_argument("--feature-contrast-max", type=float, default=1.2)
    parser.add_argument("--feature-noise-std-min", type=float, default=0.0)
    parser.add_argument("--feature-noise-std-max", type=float, default=0.03)
    return parser.parse_args()


def load_guidewire_initial_pose(path: Path, device: torch.device) -> RigidTransform:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pose = payload["initial_pose"]
    rotation = torch.tensor([pose["euler_zyx"]], dtype=torch.float32, device=device)
    translation = torch.tensor([pose["translation"]], dtype=torch.float32, device=device)
    return RigidTransform(rotation, translation, "euler_angles", "ZYX")


def to_display(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().float().cpu()
    return (image - image.min()) / (image.max() - image.min() + 1e-6)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    renderer = SXHFeatureTrainingRenderer(
        projection_mode="sum",
        render_chunk_size=args.render_chunk_size,
        device=args.device,
        mrcp_registration="original",
    )
    device = renderer.device
    crop = CanonicalSquareCrop().to(device)
    feature_model, feature_step = load_feature_model(args.feature_checkpoint, device)
    center_pose = load_guidewire_initial_pose(args.guidewire_result, device)
    offsets = get_random_offset(args.batch_size, device)
    poses = compose_centered_perturbations(center_pose, renderer.specimen_center_pose, offsets)

    with torch.no_grad():
        raw = renderer._render_in_pose_chunks(renderer.adjuster.drr_mrcp, poses, args.batch_size)
        normalized = renderer.normalizer(raw).to(torch.float32)
        cropped = crop(normalized)
        training_images = render_mrcp_sum_inverted(renderer, crop, poses)
        augmented_training_images = augment_feature_input(
            training_images,
            args.feature_contrast_min,
            args.feature_contrast_max,
            args.feature_noise_std_min,
            args.feature_noise_std_max,
        )
        features = feature_model.mrcp_net(augmented_training_images)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = args.batch_size
    fig, axes = plt.subplots(4, count, figsize=(2.3 * count, 8.6), constrained_layout=True)
    if count == 1:
        axes = axes.reshape(4, 1)
    rotations = offsets.get_rotation("euler_angles", "ZYX").detach().cpu()
    translations = offsets.get_translation().detach().cpu()
    for index in range(count):
        axes[0, index].imshow(to_display(raw[index, 0]), cmap="gray")
        axes[0, index].set_title(f"raw {index}", fontsize=9)
        axes[1, index].imshow(to_display(cropped[index, 0]), cmap="gray")
        axes[1, index].set_title("crop", fontsize=9)
        axes[2, index].imshow(to_display(augmented_training_images[index, 0]), cmap="gray")
        axes[2, index].set_title("aug revert input", fontsize=9)
        axes[3, index].imshow(to_display(features[index, 0]), cmap="viridis")
        axes[3, index].set_title(
            f"feat ch0\nr={rotations[index].numpy().round(2)}\nt={translations[index].numpy().round(1)}",
            fontsize=7,
        )
        for row in range(4):
            axes[row, index].axis("off")

    preview_path = args.output_dir / "training_batch_preview.png"
    fig.savefig(preview_path, dpi=160)
    plt.close(fig)

    torch.save(
        {
            "raw_mrcp_sum": raw.detach().cpu(),
            "cropped_normalized": cropped.detach().cpu(),
            "training_images_sum_inverted": training_images.detach().cpu(),
            "augmented_training_images_sum_inverted": augmented_training_images.detach().cpu(),
            "features": features.detach().cpu(),
            "offset_rotation_zyx": rotations,
            "offset_translation": translations,
            "feature_checkpoint_step": feature_step,
        },
        args.output_dir / "training_batch_preview.pt",
    )
    print(json.dumps({"preview": str(preview_path), "feature_checkpoint_step": feature_step}, ensure_ascii=False))


if __name__ == "__main__":
    main()
