"""Train the SXH two-branch CoMIR-style CT-DRR/MRCP patch feature network."""

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
from ours.xmr.feature_network_comir import CoMIRTwoBranchFeatureNetwork, SymmetricPatchInfoNCE  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs" / "feature_common_comir_patch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SXH CoMIR-style patch common features")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--render-chunk-size", type=int, default=2)
    parser.add_argument("--feature-channels", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--patches-per-image", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--local-negative-exclusion", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument(
        "--amp-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="CUDA autocast dtype; bf16 is the stable default on RTX 30-series GPUs",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", type=str, default=None, help="Renderer/training device, for example cuda:0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a saved checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Run 2-3 steps and save first-batch QA images")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer: torch.optim.Optimizer, steps: int, warmup_steps: int, min_learning_rate: float):
    base_learning_rate = optimizer.param_groups[0]["lr"]
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0 < min_learning_rate <= base_learning_rate:
        raise ValueError("min_learning_rate must be in (0, learning_rate]")

    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_learning_rate / base_learning_rate + (1.0 - min_learning_rate / base_learning_rate) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def save_debug_images(output_dir: Path, step: int, ct_drr: torch.Tensor, mrcp: torch.Tensor, bile: torch.Tensor) -> None:
    debug_dir = output_dir / "debug"
    stem = f"step_{step:06d}"
    write_gray_png(debug_dir / f"{stem}_ct_drr.png", ct_drr[0, 0].detach().cpu().numpy())
    write_gray_png(debug_dir / f"{stem}_mrcp_max.png", mrcp[0, 0].detach().cpu().numpy())
    write_overlay_png(
        debug_dir / f"{stem}_ct_drr_bile_overlay.png",
        ct_drr[0, 0].detach().cpu().numpy(),
        bile[0, 0].detach().cpu().numpy(),
    )


def save_checkpoint(
    output_dir: Path,
    name: str,
    step: int,
    model: CoMIRTwoBranchFeatureNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: dict[str, object],
    renderer_metadata: dict[str, object],
) -> None:
    torch.save(
        {
            "step": step,
            "xray_net_state_dict": model.xray_net.state_dict(),
            "mrcp_net_state_dict": model.mrcp_net.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "renderer_metadata": renderer_metadata,
        },
        output_dir / "checkpoints" / name,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.patches_per_image <= 0:
        raise ValueError("batch-size and patches-per-image must be positive")
    if args.feature_channels != 32:
        raise ValueError("The first CoMIR patch experiment is fixed to 32 output channels")
    if args.patch_size != 32:
        raise ValueError("The first CoMIR patch experiment is fixed to 32x32 patches")
    if args.smoke:
        args.steps = min(args.steps, 3)

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    renderer = SXHFeatureTrainingRenderer(
        projection_mode="max",
        render_chunk_size=args.render_chunk_size,
        device=args.device,
    )
    device = renderer.device
    model = CoMIRTwoBranchFeatureNetwork(feature_channels=args.feature_channels).to(device)
    criterion = SymmetricPatchInfoNCE(
        patches_per_image=args.patches_per_image,
        patch_size=args.patch_size,
        temperature=args.temperature,
        local_negative_exclusion=args.local_negative_exclusion,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(optimizer, args.steps, args.warmup_steps, args.min_learning_rate)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.amp_dtype]
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16")

    output_dir = args.output_dir
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    renderer_metadata = renderer.metadata()

    start_step = 0
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler.is_enabled() and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_step = int(checkpoint["step"])
        if start_step >= args.steps:
            raise ValueError(f"Checkpoint is already at step {start_step}, target is {args.steps}")
        print(f"resuming from checkpoint: {args.resume} at step {start_step}")

    (output_dir / "config.json").write_text(
        json.dumps({"training": config, "renderer": renderer_metadata, "start_step": start_step}, indent=2),
        encoding="utf-8",
    )

    log_path = output_dir / "train.jsonl"
    previous_parameter = None
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        batch = renderer.render_batch(
            args.batch_size,
            include_bile_projection=args.smoke or step == 1 or step % args.checkpoint_interval == 0,
        )
        if args.smoke and step == 1:
            assert batch.bile_projection is not None
            save_debug_images(output_dir, step, batch.ct_drr, batch.mrcp_projection, batch.bile_projection)

        network_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            xray_features, mrcp_features = model(batch.ct_drr, batch.mrcp_projection)
            expected_shape = (args.batch_size, args.feature_channels, 256, 256)
            if xray_features.shape != expected_shape or mrcp_features.shape != expected_shape:
                raise RuntimeError(
                    f"Unexpected feature shapes: {tuple(xray_features.shape)}, {tuple(mrcp_features.shape)}"
                )
            loss_output = criterion(xray_features, mrcp_features, batch.valid_mask)

        if not torch.isfinite(loss_output.total):
            raise RuntimeError("Non-finite patch contrastive loss")
        if scaler.is_enabled():
            scaler.scale(loss_output.total).backward()
            scaler.unscale_(optimizer)
        else:
            loss_output.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip, error_if_nonfinite=False
        )
        finite_gradients = bool(torch.isfinite(gradient_norm).item() and gradient_norm.item() > 0)
        skipped_step = not finite_gradients
        if args.smoke and step == 1:
            previous_parameter = next(model.parameters()).detach().clone()

        if finite_gradients:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.update(new_scale=max(scaler.get_scale() / 2.0, 1.0))
        scheduler.step()
        if args.smoke and step == 1:
            update_size = (next(model.parameters()).detach() - previous_parameter).abs().max().item()
            if update_size <= 0:
                raise RuntimeError("Smoke run found no optimizer update")

        torch.cuda.synchronize(device)
        network_seconds = time.perf_counter() - network_start
        record = {
            "step": step,
            "loss": loss_output.total.item(),
            "xray_to_mrcp": loss_output.xray_to_mrcp.item(),
            "mrcp_to_xray": loss_output.mrcp_to_xray.item(),
            "gradient_norm": gradient_norm.item(),
            "gradient_finite": finite_gradients,
            "skipped_step": skipped_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "ct_render_seconds": batch.ct_render_seconds,
            "mrcp_render_seconds": batch.mrcp_render_seconds,
            "network_seconds": network_seconds,
            "gpu_memory_mb": torch.cuda.memory_allocated(device) / 1024**2,
            "contrast_multiplier": batch.contrast_multiplier,
            "patch_size": args.patch_size,
            "patches_per_image": args.patches_per_image,
            "patch_descriptors_shape": loss_output.patch_descriptors_shape,
            "feature_channels": args.feature_channels,
            "temperature": args.temperature,
            "local_negative_exclusion": args.local_negative_exclusion,
            "amp_dtype": args.amp_dtype,
            "amp_scale": scaler.get_scale() if scaler.is_enabled() else 1.0,
        }
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record) + "\n")
        if step == 1 or step % args.log_interval == 0:
            print(json.dumps(record, sort_keys=True))

        if step % args.checkpoint_interval == 0 or step == args.steps:
            save_checkpoint(
                output_dir,
                "last.pt",
                step,
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                renderer_metadata,
            )
            save_checkpoint(
                output_dir,
                f"step_{step:06d}.pt",
                step,
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                renderer_metadata,
            )

        del batch, xray_features, mrcp_features, loss_output
        torch.cuda.empty_cache()

    print(f"completed {args.steps} training steps; output: {output_dir}")


if __name__ == "__main__":
    main()
