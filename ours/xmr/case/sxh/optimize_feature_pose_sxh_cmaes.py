"""CMA-ES optimization of SXH MRCP projection pose with full-image features."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cma
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
from ours.xmr.case.sxh.optimize_feature_pose_sxh import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    display_xray_background,
    load_model,
    load_real_xray,
    pose_to_numpy,
    render_ct_drr,
    render_mrcp_projection,
    save_visuals,
    timestamp,
    write_centerline_guidewire_overlay,
    write_loss_curve,
)
from ours.xmr.case.sxh.image_io import write_gray_png, write_overlay_png  # noqa: E402
from ours.xmr.feature_network_comir_v2 import CanonicalSquareCrop  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / "feature_pose_optimization_cmaes"


@dataclass(frozen=True)
class CMAESConfig:
    mode: str
    index: int
    projection: str
    mrcp_registration: str
    checkpoint: str
    output_dir: str
    device: str
    seed: int
    popsize: int
    maxiter: int
    sigma0: float
    cma_stds: tuple[float, float, float, float, float, float]
    tolfun: float
    tolx: float
    contrast_multiplier: float
    mrcp_invert: bool
    real_xray_invert: bool
    loss: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CMA-ES SXH full-feature MRCP pose optimization")
    parser.add_argument("--mode", choices=["virtual", "real"], default="virtual")
    parser.add_argument("--projection", choices=["max", "sum"], default="max")
    parser.add_argument("--mrcp-registration", choices=["refined", "original"], default="refined")
    parser.add_argument("--index", type=int, default=31)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--cma-stds", type=float, nargs=6, default=(0.05, 0.05, 0.05, 5.0, 5.0, 5.0))
    parser.add_argument("--tolfun", type=float, default=1e-6)
    parser.add_argument("--tolx", type=float, default=1e-6)
    parser.add_argument("--contrast-multiplier", type=float, default=3.0)
    parser.add_argument("--mrcp-invert", action="store_true")
    parser.add_argument("--real-xray-invert", action="store_true")
    return parser.parse_args()


def make_offset_from_vector(x: np.ndarray, device: torch.device) -> RigidTransform:
    values = torch.as_tensor(x, dtype=torch.float32, device=device)
    rotation = values[:3].unsqueeze(0)
    translation = values[3:].unsqueeze(0)
    return RigidTransform(rotation, translation, "euler_angles", "ZYX")


def full_feature_cosine_loss(fixed_features: Tensor, moving_features: Tensor) -> Tensor:
    fixed_flat = F.normalize(fixed_features.flatten(1), p=2, dim=1, eps=1e-6)
    moving_flat = F.normalize(moving_features.flatten(1), p=2, dim=1, eps=1e-6)
    return 1.0 - (fixed_flat * moving_flat).sum(dim=1).mean()


def pose_from_vector(
    x: np.ndarray,
    renderer: SXHFeatureTrainingRenderer,
    trusted_pose: RigidTransform,
) -> RigidTransform:
    offset = make_offset_from_vector(np.asarray(x, dtype=np.float32), renderer.device)
    return compose_centered_perturbations(trusted_pose, renderer.specimen_center_pose, offset)


def write_cmaes_visuals(
    output_dir: Path,
    renderer: SXHFeatureTrainingRenderer,
    index: int,
    fixed_image: Tensor,
    initial_pose: RigidTransform,
    best_pose: RigidTransform,
    initial_mrcp: Tensor,
    best_mrcp: Tensor,
    fixed_features: Tensor,
    initial_features: Tensor,
    best_features: Tensor,
    losses: list[float],
) -> None:
    save_visuals(
        output_dir,
        renderer,
        index,
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
    # Keep CMA-ES names explicit in addition to the reused final_* filenames.
    write_gray_png(output_dir / "best_mrcp.png", best_mrcp[0, 0].detach().cpu().numpy())
    write_overlay_png(
        output_dir / "best_overlay.png",
        fixed_image[0, 0].detach().cpu().numpy(),
        best_mrcp[0, 0].detach().cpu().numpy(),
    )
    write_centerline_guidewire_overlay(output_dir / "best_centerline_guidewire_overlay.png", renderer, index, best_pose)
    write_loss_curve(output_dir / "loss_curve.png", losses)


def main() -> None:
    args = parse_args()
    if args.popsize <= 0:
        raise ValueError("popsize must be positive")
    if args.maxiter <= 0:
        raise ValueError("maxiter must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    renderer = SXHFeatureTrainingRenderer(
        projection_mode=args.projection,
        render_chunk_size=1,
        device=args.device,
        mrcp_registration=args.mrcp_registration,
    )
    device = renderer.device
    crop = CanonicalSquareCrop().to(device)
    model, checkpoint_step = load_model(args.checkpoint, device)
    trusted_pose = renderer.trusted_pose
    initial_pose = pose_from_vector(np.zeros(6, dtype=np.float32), renderer, trusted_pose)

    output_dir = args.output_root / f"{args.mode}_{args.projection}_xray{args.index:03d}_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        if args.mode == "virtual":
            fixed_image = render_ct_drr(renderer, crop, trusted_pose, args.contrast_multiplier)
        else:
            fixed_image = load_real_xray(renderer, crop, args.index, invert=args.real_xray_invert)
        fixed_features = model.xray_net(fixed_image)
        initial_mrcp = render_mrcp_projection(renderer, crop, initial_pose, invert=args.mrcp_invert)
        initial_features = model.mrcp_net(initial_mrcp)
        initial_loss = float(full_feature_cosine_loss(fixed_features, initial_features).item())

    config = CMAESConfig(
        mode=args.mode,
        index=args.index,
        projection=args.projection,
        mrcp_registration=args.mrcp_registration,
        checkpoint=str(args.checkpoint),
        output_dir=str(output_dir),
        device=str(device),
        seed=args.seed,
        popsize=args.popsize,
        maxiter=args.maxiter,
        sigma0=args.sigma0,
        cma_stds=tuple(float(v) for v in args.cma_stds),
        tolfun=args.tolfun,
        tolx=args.tolx,
        contrast_multiplier=args.contrast_multiplier,
        mrcp_invert=args.mrcp_invert,
        real_xray_invert=args.real_xray_invert,
        loss="full-image feature cosine: 1 - cosine(flatten(xray_feature), flatten(mrcp_feature))",
    )
    (output_dir / "config.json").write_text(
        json.dumps({"optimizer": asdict(config), "checkpoint_step": checkpoint_step, "renderer": renderer.metadata()}, indent=2),
        encoding="utf-8",
    )
    np.save(output_dir / "initial_pose.npy", pose_to_numpy(initial_pose))

    log_path = output_dir / "history.jsonl"
    history: list[dict[str, object]] = []
    losses: list[float] = []
    best_loss = initial_loss
    best_x = np.zeros(6, dtype=np.float64)
    best_eval_index = 0
    eval_index = 0
    start_time = time.perf_counter()

    options = {
        "seed": args.seed,
        "popsize": args.popsize,
        "CMA_stds": list(args.cma_stds),
        "maxiter": args.maxiter,
        "verb_disp": 1,
        "tolfun": args.tolfun,
        "tolx": args.tolx,
        "verb_filenameprefix": str(output_dir / "cma_"),
    }
    optimizer = cma.CMAEvolutionStrategy(x0=np.zeros(6, dtype=np.float64), sigma0=args.sigma0, inopts=options)

    print(
        json.dumps(
            {
                "event": "start",
                "output_dir": str(output_dir),
                "checkpoint_step": checkpoint_step,
                "initial_loss": initial_loss,
                "projection": args.projection,
                "mrcp_registration": args.mrcp_registration,
                "loss": "full_feature_cosine",
            },
            ensure_ascii=False,
        )
    )

    while not optimizer.stop() and optimizer.countiter < args.maxiter:
        iteration = optimizer.countiter + 1
        candidates = optimizer.ask()
        candidate_losses: list[float] = []
        for candidate_index, candidate in enumerate(candidates):
            eval_index += 1
            with torch.no_grad():
                pose = pose_from_vector(np.asarray(candidate, dtype=np.float32), renderer, trusted_pose)
                moving_mrcp = render_mrcp_projection(renderer, crop, pose, invert=args.mrcp_invert)
                moving_features = model.mrcp_net(moving_mrcp)
                loss = float(full_feature_cosine_loss(fixed_features, moving_features).item())
            if not np.isfinite(loss):
                loss = float("inf")
            candidate_losses.append(loss)
            is_best = loss < best_loss
            if is_best:
                best_loss = loss
                best_x = np.asarray(candidate, dtype=np.float64).copy()
                best_eval_index = eval_index
            record = {
                "eval_index": eval_index,
                "iteration": iteration,
                "candidate_index": candidate_index,
                "loss": loss,
                "is_best": is_best,
                "rotation_delta_zyx_radians": [float(v) for v in candidate[:3]],
                "translation_delta_mm": [float(v) for v in candidate[3:]],
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            history.append(record)
            losses.append(loss)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        optimizer.tell(candidates, candidate_losses)
        optimizer.disp()
        best_record = {
            "iteration": iteration,
            "best_loss": best_loss,
            "iteration_best_loss": float(np.min(candidate_losses)),
            "iteration_median_loss": float(np.median(candidate_losses)),
            "eval_index": eval_index,
            "best_eval_index": best_eval_index,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        print(json.dumps(best_record, ensure_ascii=False))

    best_pose = pose_from_vector(best_x.astype(np.float32), renderer, trusted_pose)
    np.save(output_dir / "best_pose.npy", pose_to_numpy(best_pose))

    with torch.no_grad():
        best_mrcp = render_mrcp_projection(renderer, crop, best_pose, invert=args.mrcp_invert)
        best_features = model.mrcp_net(best_mrcp)
    write_cmaes_visuals(
        output_dir,
        renderer,
        args.index,
        fixed_image,
        initial_pose,
        best_pose,
        initial_mrcp,
        best_mrcp,
        fixed_features,
        initial_features,
        best_features,
        losses,
    )

    summary = {
        "output_dir": str(output_dir),
        "mode": args.mode,
        "projection": args.projection,
        "mrcp_registration": args.mrcp_registration,
        "checkpoint_step": checkpoint_step,
        "initial_loss": initial_loss,
        "best_loss": best_loss,
        "best_eval_index": best_eval_index,
        "best_rotation_delta_zyx_radians": [float(v) for v in best_x[:3]],
        "best_translation_delta_mm": [float(v) for v in best_x[3:]],
        "evaluations": eval_index,
        "iterations": optimizer.countiter,
        "stop": optimizer.stop(),
        "history_tail": history[-5:],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
