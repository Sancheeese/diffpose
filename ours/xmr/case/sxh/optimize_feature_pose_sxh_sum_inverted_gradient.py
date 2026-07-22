"""Gradient optimization of an SXH sum-MRCP pose using frozen full-image features."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
import torch
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
from ours.xmr.case.sxh.optimize_feature_pose_sxh import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    load_model,
    load_real_xray,
    make_offset,
    pose_to_numpy,
    render_ct_drr,
    render_mrcp_projection,
    save_visuals,
    timestamp,
)
from ours.xmr.case.sxh.web_drr_server_nii_sxh_centerline import (  # noqa: E402
    project_points_to_detector,
    render_chain_points_overlay,
)
from ours.xmr.feature_network_comir_v2 import CanonicalSquareCrop  # noqa: E402


SXH_ROOT = Path(__file__).resolve().parent
DEFAULT_SUM_INVERTED_CHECKPOINT = (
    SXH_ROOT / "runs" / "feature_common_comir_antishortcut_sum_mrcp_inverted" / "checkpoints" / "last.pt"
)
DEFAULT_OUTPUT_ROOT = SXH_ROOT / "runs" / "feature_pose_optimization_sum_inverted_gradient"
DEFAULT_GUIDEWIRE_RESULT = SXH_ROOT / "outputs" / "guidewire_registration" / "xray031" / "result.json"


@dataclass(frozen=True)
class OptimizationConfig:
    mode: str
    index: int
    checkpoint: str
    output_dir: str
    device: str
    steps: int
    rotation_lr: float
    translation_lr: float
    gradient_clip: float
    contrast_multiplier: float
    mrcp_registration: str
    real_xray_invert: bool
    loss: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-feature gradient optimization for sum + inverted MRCP")
    parser.add_argument("--mode", choices=["virtual", "real"], default="real")
    parser.add_argument("--index", type=int, default=31)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SUM_INVERTED_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--rotation-lr", type=float, default=0.01)
    parser.add_argument("--translation-lr", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--contrast-multiplier", type=float, default=3.0)
    parser.add_argument("--mrcp-registration", choices=["refined", "original"], default="refined")
    parser.add_argument("--guidewire-result", type=Path, default=DEFAULT_GUIDEWIRE_RESULT)
    parser.add_argument("--real-xray-invert", action="store_true")
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def full_feature_cosine_loss(fixed_features: torch.Tensor, moving_features: torch.Tensor) -> torch.Tensor:
    fixed = F.normalize(fixed_features.flatten(1), p=2, dim=1, eps=1e-6)
    moving = F.normalize(moving_features.flatten(1), p=2, dim=1, eps=1e-6)
    return 1.0 - (fixed * moving).sum(dim=1).mean()


def load_guidewire_initial_pose(path: Path, device: torch.device) -> tuple[RigidTransform, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pose = payload["initial_pose"]
    rotation = torch.tensor([pose["euler_zyx"]], dtype=torch.float32, device=device)
    translation = torch.tensor([pose["translation"]], dtype=torch.float32, device=device)
    return RigidTransform(rotation, translation, "euler_angles", "ZYX"), int(payload["selected_chain_index"])


def write_selected_chain_overlay(
    path: Path,
    renderer: SXHFeatureTrainingRenderer,
    index: int,
    pose: RigidTransform,
    selected_chain_index: int,
) -> None:
    background = renderer.adjuster.transforms(renderer.adjuster.specimen[index][0], reverse=False).squeeze().cpu().numpy()
    with torch.no_grad():
        projected = project_points_to_detector(renderer.adjuster.centerline_vertices.to(renderer.device), pose, renderer.adjuster.drr.detector)
    overlay = render_chain_points_overlay(
        background,
        projected.detach().cpu().numpy(),
        [renderer.adjuster.centerline_chains[selected_chain_index]],
        renderer.adjuster.guidewire_points_for_index(index),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), overlay):
        raise IOError(f"Failed to write overlay: {path}")


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    renderer = SXHFeatureTrainingRenderer(
        projection_mode="sum",
        render_chunk_size=1,
        device=args.device,
        mrcp_registration=args.mrcp_registration,
    )
    device = renderer.device
    crop = CanonicalSquareCrop().to(device)
    model, checkpoint_step = load_model(args.checkpoint, device)

    # Use the exact pose saved before guidewire/centerline optimization, rather than relying on an implicit reload.
    if args.mrcp_registration == "original":
        trusted_pose, selected_chain_index = load_guidewire_initial_pose(args.guidewire_result, device)
    else:
        trusted_pose = renderer.trusted_pose
        selected_chain_index = -1
    rotation_delta = torch.nn.Parameter(torch.zeros((1, 3), dtype=torch.float32, device=device))
    translation_delta = torch.nn.Parameter(torch.zeros((1, 3), dtype=torch.float32, device=device))
    initial_pose = compose_centered_perturbations(
        trusted_pose,
        renderer.specimen_center_pose,
        make_offset(rotation_delta.detach(), translation_delta.detach()),
    )

    output_dir = args.output_root / f"{args.mode}_sum_inverted_xray{args.index:03d}_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        if args.mode == "virtual":
            fixed_image = render_ct_drr(renderer, crop, trusted_pose, args.contrast_multiplier)
        else:
            fixed_image = load_real_xray(renderer, crop, args.index, invert=args.real_xray_invert)
        fixed_features = model.xray_net(fixed_image)
        initial_mrcp = render_mrcp_projection(renderer, crop, initial_pose, invert=True)
        initial_features = model.mrcp_net(initial_mrcp)
        initial_loss = float(full_feature_cosine_loss(fixed_features, initial_features).item())

    optimizer = torch.optim.Adam(
        [
            {"params": [rotation_delta], "lr": args.rotation_lr},
            {"params": [translation_delta], "lr": args.translation_lr},
        ]
    )
    config = OptimizationConfig(
        mode=args.mode,
        index=args.index,
        checkpoint=str(args.checkpoint),
        output_dir=str(output_dir),
        device=str(device),
        steps=args.steps,
        rotation_lr=args.rotation_lr,
        translation_lr=args.translation_lr,
        gradient_clip=args.gradient_clip,
        contrast_multiplier=args.contrast_multiplier,
        mrcp_registration=args.mrcp_registration,
        real_xray_invert=args.real_xray_invert,
        loss="1 - cosine(flatten(xray_feature[32,256,256]), flatten(mrcp_feature[32,256,256]))",
    )
    (output_dir / "config.json").write_text(
        json.dumps({"optimizer": asdict(config), "checkpoint_step": checkpoint_step, "renderer": renderer.metadata()}, indent=2),
        encoding="utf-8",
    )
    np.save(output_dir / "initial_pose.npy", pose_to_numpy(initial_pose))

    history: list[dict[str, object]] = []
    losses: list[float] = []
    best_loss = initial_loss
    best_step = 0
    best_rotation = rotation_delta.detach().clone()
    best_translation = translation_delta.detach().clone()
    start_time = time.perf_counter()
    history_path = output_dir / "history.jsonl"
    print(json.dumps({"event": "start", "output_dir": str(output_dir), "initial_loss": initial_loss}, ensure_ascii=False))

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        pose = compose_centered_perturbations(
            trusted_pose,
            renderer.specimen_center_pose,
            make_offset(rotation_delta, translation_delta),
        )
        moving_mrcp = render_mrcp_projection(renderer, crop, pose, invert=True)
        moving_features = model.mrcp_net(moving_mrcp)
        loss = full_feature_cosine_loss(fixed_features, moving_features)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite full-feature loss at step {step}: {loss.item()}")
        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            best_step = step
            best_rotation = rotation_delta.detach().clone()
            best_translation = translation_delta.detach().clone()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([rotation_delta, translation_delta], args.gradient_clip)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        record = {
            "step": step,
            "loss": float(loss.item()),
            "gradient_norm": float(gradient_norm.item()),
            "rotation_delta_zyx_radians": rotation_delta.detach().cpu().tolist()[0],
            "translation_delta_mm": translation_delta.detach().cpu().tolist()[0],
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        history.append(record)
        losses.append(record["loss"])
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(json.dumps(record, ensure_ascii=False))

    best_pose = compose_centered_perturbations(
        trusted_pose,
        renderer.specimen_center_pose,
        make_offset(best_rotation, best_translation),
    )
    np.save(output_dir / "best_pose.npy", pose_to_numpy(best_pose))
    with torch.no_grad():
        best_mrcp = render_mrcp_projection(renderer, crop, best_pose, invert=True)
        best_features = model.mrcp_net(best_mrcp)
        best_loss_verified = float(full_feature_cosine_loss(fixed_features, best_features).item())
    save_visuals(
        output_dir,
        renderer,
        args.index,
        initial_pose,
        best_pose,
        fixed_image,
        initial_mrcp,
        best_mrcp,
        fixed_features,
        initial_features,
        fixed_features,
        best_features,
        losses,
    )
    if selected_chain_index >= 0:
        write_selected_chain_overlay(
            output_dir / "guidewire_method_initial_selected_chain_overlay.png",
            renderer,
            args.index,
            initial_pose,
            selected_chain_index,
        )
        write_selected_chain_overlay(
            output_dir / "guidewire_method_best_selected_chain_overlay.png",
            renderer,
            args.index,
            best_pose,
            selected_chain_index,
        )
    summary = {
        "output_dir": str(output_dir),
        "checkpoint_step": checkpoint_step,
        "initial_loss": initial_loss,
        "best_loss": best_loss_verified,
        "best_step": best_step,
        "best_rotation_delta_zyx_radians": best_rotation.detach().cpu().tolist()[0],
        "best_translation_delta_mm": best_translation.detach().cpu().tolist()[0],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
