"""Train anti-shortcut CT-DRR/MRCP common feature maps on SXH synthetic pairs."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DIFFPOSE_ROOT = PROJECT_ROOT / "diffpose"
OURS_ROOT = DIFFPOSE_ROOT / "ours"
for path in (DIFFPOSE_ROOT, OURS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ours.xmr.case.sxh.feature_training_renderer import SXHFeatureTrainingRenderer  # noqa: E402
from ours.xmr.case.sxh.image_io import write_gray_png, write_overlay_png  # noqa: E402
from ours.xmr.feature_network_comir_v2 import (  # noqa: E402
    CanonicalSquareCrop,
    CoMIRTwoBranchFeatureNetwork,
    CrossModalPatchInfoNCE,
    IndependentCropC4Augment,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs" / "feature_common_comir_antishortcut"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SXH anti-shortcut CoMIR feature maps")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8, help="Rendered CT-DRR/MRCP pairs per optimizer step")
    parser.add_argument("--patch-pairs-per-image", type=int, default=24)
    parser.add_argument("--render-chunk-size", type=int, default=2)
    parser.add_argument("--feature-channels", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, minimum_lr: float):
    base_lr = optimizer.param_groups[0]["lr"]

    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_lr / base_lr + (1.0 - minimum_lr / base_lr) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def save_debug_images(output_dir: Path, step: int, raw, canonical_xray, canonical_mrcp, augmented, canonical_bile) -> None:
    debug_dir = output_dir / "debug"
    stem = f"step_{step:06d}"
    write_gray_png(debug_dir / f"{stem}_raw_ct_drr.png", raw.ct_drr[0, 0].detach().cpu().numpy())
    write_gray_png(debug_dir / f"{stem}_raw_mrcp_max.png", raw.mrcp_projection[0, 0].detach().cpu().numpy())
    write_gray_png(debug_dir / f"{stem}_canonical_ct_drr.png", canonical_xray[0, 0].detach().cpu().numpy())
    write_gray_png(debug_dir / f"{stem}_canonical_mrcp_max.png", canonical_mrcp[0, 0].detach().cpu().numpy())
    write_overlay_png(
        debug_dir / f"{stem}_canonical_ct_drr_bile_overlay.png",
        canonical_xray[0, 0].detach().cpu().numpy(),
        canonical_bile[0, 0].detach().cpu().numpy(),
    )
    write_gray_png(debug_dir / f"{stem}_augmented_ct_drr.png", augmented.xray[0, 0].detach().cpu().numpy())
    write_gray_png(debug_dir / f"{stem}_augmented_mrcp_max.png", augmented.mrcp[0, 0].detach().cpu().numpy())


def checkpoint_payload(step: int, model, optimizer, scheduler, scaler, config, renderer_metadata) -> dict[str, object]:
    return {
        "step": step,
        "model_state_dict": model.state_dict(),
        "xray_net_state_dict": model.xray_net.state_dict(),
        "mrcp_net_state_dict": model.mrcp_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
        "renderer_metadata": renderer_metadata,
    }


def main() -> None:
    args = parse_args()
    if args.patch_pairs_per_image != 24:
        raise ValueError("This CoMIR configuration is fixed to 24 corresponding feature-patch pairs per rendered image pair")
    if args.feature_channels != 32 or args.patch_size != 32:
        raise ValueError("This first anti-shortcut experiment is fixed to 32 feature channels and 32x32 feature patches")
    if args.smoke:
        args.steps = min(args.steps, 3)

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    renderer = SXHFeatureTrainingRenderer(projection_mode="max", render_chunk_size=args.render_chunk_size, device=args.device)
    device = renderer.device
    canonical_crop = CanonicalSquareCrop().to(device)
    augment = IndependentCropC4Augment().to(device)
    model = CoMIRTwoBranchFeatureNetwork(feature_channels=args.feature_channels).to(device)
    criterion = CrossModalPatchInfoNCE(
        patch_pairs_per_image=args.patch_pairs_per_image,
        patch_size=args.patch_size,
        temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.steps, args.warmup_steps, args.min_learning_rate)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.amp_dtype]
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16")

    output_dir = args.output_dir
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config.update({"positive_patches_per_query": 1, "cross_modal_negative_patches_per_query": 23})
    renderer_metadata = renderer.metadata()
    (output_dir / "config.json").write_text(json.dumps({"training": config, "renderer": renderer_metadata}, indent=2), encoding="utf-8")

    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_step = int(checkpoint["step"])

    log_path = output_dir / "train.jsonl"
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        batch = renderer.render_batch(args.batch_size, include_bile_projection=args.smoke and step == 1)
        canonical_xray = canonical_crop(batch.ct_drr)
        canonical_mrcp = canonical_crop(batch.mrcp_projection)
        augmented = augment(canonical_xray, canonical_mrcp)
        if args.smoke and step == 1:
            assert batch.bile_projection is not None
            canonical_bile = canonical_crop(batch.bile_projection)
            save_debug_images(output_dir, step, batch, canonical_xray, canonical_mrcp, augmented, canonical_bile)

        optimizer.zero_grad(set_to_none=True)
        network_start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            xray_features, mrcp_features = model(augmented.xray, augmented.mrcp)
            expected = (args.batch_size, 32, 256, 256)
            if xray_features.shape != expected or mrcp_features.shape != expected:
                raise RuntimeError(f"Unexpected feature shape: {tuple(xray_features.shape)}, {tuple(mrcp_features.shape)}")
            loss_output = criterion(xray_features, mrcp_features, augmented.parameters, augment)
        if not torch.isfinite(loss_output.total):
            raise RuntimeError("Non-finite cross-modal patch loss")
        if scaler.is_enabled():
            scaler.scale(loss_output.total).backward()
            scaler.unscale_(optimizer)
        else:
            loss_output.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=False)
        gradient_finite = bool(torch.isfinite(gradient_norm).item())
        if gradient_finite:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        torch.cuda.synchronize(device)

        params = augmented.parameters
        record = {
            "step": step,
            "loss": loss_output.total.item(),
            "xray_to_mrcp": loss_output.xray_to_mrcp.item(),
            "mrcp_to_xray": loss_output.mrcp_to_xray.item(),
            "gradient_norm": gradient_norm.item(),
            "gradient_finite": gradient_finite,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "ct_render_seconds": batch.ct_render_seconds,
            "mrcp_render_seconds": batch.mrcp_render_seconds,
            "network_seconds": time.perf_counter() - network_start,
            "gpu_memory_mb": torch.cuda.memory_allocated(device) / 1024**2,
            "contrast_multiplier": batch.contrast_multiplier,
            "patch_pairs_per_rendered_image": args.patch_pairs_per_image,
            "patch_pairs_per_step": args.batch_size * args.patch_pairs_per_image,
            "positive_patches_per_query": 1,
            "cross_modal_negative_patches_per_query": 23,
            "patch_descriptor_shape": loss_output.descriptor_shape,
            "temperature": args.temperature,
            "crop_size": params.crop_size.detach().cpu().tolist(),
            "xray_origin_yx": params.xray_origin_yx.detach().cpu().tolist(),
            "mrcp_origin_yx": params.mrcp_origin_yx.detach().cpu().tolist(),
            "xray_rotation_k": params.xray_rotation_k.detach().cpu().tolist(),
            "mrcp_rotation_k": params.mrcp_rotation_k.detach().cpu().tolist(),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if step == 1 or step % args.log_interval == 0:
            print(json.dumps(record, sort_keys=True))

        if step % args.checkpoint_interval == 0 or step == args.steps:
            payload = checkpoint_payload(step, model, optimizer, scheduler, scaler, config, renderer_metadata)
            torch.save(payload, output_dir / "checkpoints" / "last.pt")
            torch.save(payload, output_dir / "checkpoints" / f"step_{step:06d}.pt")

    print(f"completed {args.steps} anti-shortcut CoMIR steps; output: {output_dir}")


if __name__ == "__main__":
    main()
