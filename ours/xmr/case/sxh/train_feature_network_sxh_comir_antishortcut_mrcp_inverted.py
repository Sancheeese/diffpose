"""Train the separate anti-shortcut CoMIR variant with inverted MRCP intensity."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DIFFPOSE_ROOT = PROJECT_ROOT / "diffpose"
OURS_ROOT = DIFFPOSE_ROOT / "ours"
for path in (DIFFPOSE_ROOT, OURS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ours.xmr.case.sxh.feature_training_renderer import SXHFeatureTrainingRenderer  # noqa: E402
from ours.xmr.case.sxh.train_feature_network_sxh_comir_antishortcut import (  # noqa: E402
    checkpoint_payload,
    make_scheduler,
    save_debug_images,
    seed_everything,
)
from ours.xmr.feature_network_comir_v2 import (  # noqa: E402
    CanonicalSquareCrop,
    CoMIRTwoBranchFeatureNetwork,
    CrossModalPatchInfoNCE,
    IndependentCropC4Augment,
    invert_legacy_standardized_intensity,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs" / "feature_common_comir_antishortcut_mrcp_inverted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anti-shortcut CoMIR with black-white inverted MRCP")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
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


def main() -> None:
    args = parse_args()
    if args.patch_pairs_per_image != 24:
        raise ValueError("This CoMIR configuration is fixed to 24 corresponding feature-patch pairs per rendered image pair")
    if args.feature_channels != 32 or args.patch_size != 32:
        raise ValueError("This experiment is fixed to 32 feature channels and 32x32 feature patches")
    if args.smoke:
        args.steps = min(args.steps, 3)

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    renderer = SXHFeatureTrainingRenderer(projection_mode="max", render_chunk_size=args.render_chunk_size, device=args.device)
    device = renderer.device
    canonical_crop = CanonicalSquareCrop().to(device)
    augment = IndependentCropC4Augment().to(device)
    model = CoMIRTwoBranchFeatureNetwork(feature_channels=32).to(device)
    criterion = CrossModalPatchInfoNCE(patch_pairs_per_image=24, patch_size=32, temperature=args.temperature).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.steps, args.warmup_steps, args.min_learning_rate)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.amp_dtype]
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16")

    output_dir = args.output_dir
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config.update(
        {
            "mrcp_intensity_variant": "black_white_inverted_in_legacy_unit_intensity_space",
            "positive_patches_per_query": 1,
            "cross_modal_negative_patches_per_query": 23,
        }
    )
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
        canonical_mrcp = invert_legacy_standardized_intensity(canonical_crop(batch.mrcp_projection))
        augmented = augment(canonical_xray, canonical_mrcp)
        if args.smoke and step == 1:
            assert batch.bile_projection is not None
            save_debug_images(
                output_dir,
                step,
                batch,
                canonical_xray,
                canonical_mrcp,
                augmented,
                canonical_crop(batch.bile_projection),
            )

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
            "mrcp_intensity_variant": "black_white_inverted",
            "patch_pairs_per_rendered_image": 24,
            "patch_pairs_per_step": args.batch_size * 24,
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

    print(f"completed {args.steps} inverted-MRCP anti-shortcut CoMIR steps; output: {output_dir}")


if __name__ == "__main__":
    main()
