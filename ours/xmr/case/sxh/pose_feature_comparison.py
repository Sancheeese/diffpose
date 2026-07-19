"""Compare SXH anti-shortcut CoMIR features at matched and mismatched poses."""

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
import torch.nn.functional as F

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
from ours.xmr.feature_network_comir_v2 import (  # noqa: E402
    CanonicalSquareCrop,
    CoMIRTwoBranchFeatureNetwork,
)


DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "runs" / "feature_common_comir_antishortcut"


def summarize_similarity(
    xray: torch.Tensor,
    mrcp: torch.Tensor,
    seed: int = 0,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Summarize cosine similarity at corresponding and seeded-permuted pixels.

    Feature tensors use the conventional ``(batch, channels, *spatial)`` layout.
    """
    if xray.shape != mrcp.shape:
        raise ValueError("xray and mrcp must have identical shapes")
    if xray.device != mrcp.device:
        raise ValueError("xray and mrcp must be on the same device")
    if xray.ndim < 3:
        raise ValueError("feature tensors must have batch, channel, and spatial dimensions")
    if xray.shape[1] == 0:
        raise ValueError("feature tensors must contain at least one channel")
    if xray.shape[0] == 0 or any(size == 0 for size in xray.shape[2:]):
        raise ValueError("feature tensors must contain at least one spatial sample")

    xray_samples = xray.movedim(1, -1).reshape(-1, xray.shape[1])
    mrcp_samples = mrcp.movedim(1, -1).reshape(-1, mrcp.shape[1])
    if valid_mask is not None:
        if valid_mask.device != xray.device:
            raise ValueError("valid_mask must be on the same device as feature tensors")
        expected_shape = (xray.shape[0], *xray.shape[2:])
        if valid_mask.shape == (xray.shape[0], 1, *xray.shape[2:]):
            valid_mask = valid_mask[:, 0]
        if valid_mask.shape != expected_shape:
            raise ValueError(f"valid_mask must have shape {expected_shape} or include one channel")
        sample_mask = valid_mask.to(dtype=torch.bool).reshape(-1)
        if not sample_mask.any():
            raise ValueError("valid_mask must contain at least one valid spatial sample")
        xray_samples = xray_samples[sample_mask]
        mrcp_samples = mrcp_samples[sample_mask]
    corresponding = F.cosine_similarity(xray_samples, mrcp_samples, dim=1)

    generator = torch.Generator(device=mrcp.device).manual_seed(seed)
    permutation = torch.randperm(
        mrcp_samples.shape[0], device=mrcp.device, generator=generator
    )
    permuted = F.cosine_similarity(xray_samples, mrcp_samples[permutation], dim=1)

    return {
        "corresponding_samples": corresponding,
        "permuted_samples": permuted,
        "corresponding_mean": corresponding.mean(),
        "corresponding_median": corresponding.median(),
        "permuted_mean": permuted.mean(),
        "permuted_median": permuted.median(),
    }


def pair_specs(a, b):
    """Return the report's positive controls followed by the pose-mismatch control."""
    return [("A/A", a, a), ("B/B", b, b), ("A/B", a, b)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN_DIR / "checkpoints" / "last.pt")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--contrast-multiplier", type=float, default=3.0)
    parser.add_argument("--render-chunk-size", type=int, default=1)
    return parser.parse_args()


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Non-finite {name}")


def _as_display(image: torch.Tensor, valid_mask: torch.Tensor | None = None) -> np.ndarray:
    array = image.detach().float().cpu().numpy()
    mask = None
    if valid_mask is not None:
        mask = valid_mask[0, 0].detach().cpu().numpy().astype(bool)
        if not mask.any():
            raise ValueError("valid_mask must contain at least one display pixel")
    values = array[mask] if mask is not None else array
    low, high = np.percentile(values, (1, 99))
    result = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
    if mask is not None:
        result[~mask] = 0
    return result


def _pose_metadata(pose, coordinate_frame: str) -> dict[str, object]:
    return {
        "coordinate_frame": coordinate_frame,
        "parameterization": "Euler ZYX rotation (radians) plus translation (mm)",
        "rotation_euler_zyx_radians": pose.get_rotation("euler_angles", "ZYX").detach().cpu().tolist(),
        "translation_mm": pose.get_translation().detach().cpu().tolist(),
        "matrix": pose.get_matrix().detach().cpu().tolist(),
    }


def _joint_pca_rgb(feature_maps: list[torch.Tensor], valid_masks: list[torch.Tensor]) -> list[np.ndarray]:
    if len(feature_maps) != len(valid_masks):
        raise ValueError("feature_maps and valid_masks must have the same length")
    masks = [mask[0, 0].detach().cpu().numpy().astype(bool) for mask in valid_masks]
    if not all(mask.any() for mask in masks):
        raise ValueError("Each feature map must contain at least one valid FOV pixel")
    vectors = np.concatenate(
        [
            feature[0].detach().float().cpu().numpy().transpose(1, 2, 0)[mask]
            for feature, mask in zip(feature_maps, masks)
        ],
        axis=0,
    )
    mean = vectors.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(vectors - mean, full_matrices=False)
    projected = []
    for feature in feature_maps:
        values = feature[0].detach().float().cpu().numpy().transpose(1, 2, 0)
        projected.append((values - mean) @ components[:3].T)
    all_values = np.concatenate([image[mask] for image, mask in zip(projected, masks)], axis=0)
    low = np.percentile(all_values, 1, axis=0)
    high = np.percentile(all_values, 99, axis=0)
    result = []
    for image, mask in zip(projected, masks):
        image = np.clip((image - low) / np.maximum(high - low, 1e-6), 0, 1)
        image[~mask] = 0
        result.append(image)
    return result


def load_checkpoint_with_retry(path: Path, device: torch.device, attempts: int = 8) -> dict:
    """Read a checkpoint safely while a training process may be replacing it."""
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except (EOFError, RuntimeError, OSError) as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Could not load checkpoint after {attempts} attempts: {path}") from last_error


def _summary_metrics(summary: dict[str, torch.Tensor]) -> dict[str, float]:
    corresponding = summary["corresponding_samples"]
    permuted = summary["permuted_samples"]
    return {
        "corresponding_mean_cosine": float(summary["corresponding_mean"].item()),
        "corresponding_median_cosine": float(summary["corresponding_median"].item()),
        "permuted_mean_cosine": float(summary["permuted_mean"].item()),
        "permuted_median_cosine": float(summary["permuted_median"].item()),
        "corresponding_gt_permuted_fraction": float((corresponding > permuted).float().mean().item()),
    }


def _draw_feature_comparison(output_path: Path, pairs: list[dict[str, object]], pca_maps: list[np.ndarray]) -> None:
    figure, axes = plt.subplots(len(pairs), 4, figsize=(16, 5 * len(pairs)))
    for row, pair in enumerate(pairs):
        entries = [
            (_as_display(pair["xray_image"][0, 0], pair["valid_mask"]), "CT-DRR"),
            (_as_display(pair["mrcp_image"][0, 0], pair["valid_mask"]), "MRCP max"),
            (pca_maps[row * 2], "CT-DRR feature PCA"),
            (pca_maps[row * 2 + 1], "MRCP feature PCA"),
        ]
        for axis, (image, title) in zip(axes[row], entries):
            axis.imshow(image, cmap=None if image.ndim == 3 else "gray")
            axis.set_title(f"{pair['name']}: {title}")
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _draw_similarity_maps(output_path: Path, pairs: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(len(pairs), 3, figsize=(15, 4.5 * len(pairs)))
    for row, pair in enumerate(pairs):
        height, width = pair["xray_features"].shape[-2:]
        summary = pair["similarity"]
        mask = pair["valid_mask"][0, 0].detach().cpu().numpy().astype(bool)
        corresponding = np.full((height, width), np.nan, dtype=np.float32)
        permuted = np.full((height, width), np.nan, dtype=np.float32)
        corresponding[mask] = summary["corresponding_samples"].detach().float().cpu().numpy()
        permuted[mask] = summary["permuted_samples"].detach().float().cpu().numpy()
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("black")
        for axis, image, title in (
            (axes[row, 0], corresponding, "same detector coordinates"),
            (axes[row, 1], permuted, "seeded permuted-pixel control"),
        ):
            axis.imshow(image, cmap=cmap, vmin=-1.0, vmax=1.0)
            axis.set_title(f"{pair['name']}: {title}")
            axis.axis("off")
        axes[row, 2].hist(summary["corresponding_samples"].detach().float().cpu().numpy(), bins=50, alpha=0.65, label="same coordinates")
        axes[row, 2].hist(summary["permuted_samples"].detach().float().cpu().numpy(), bins=50, alpha=0.65, label="permuted")
        axes[row, 2].set_title(f"{pair['name']}: cosine distribution")
        axes[row, 2].set_xlabel("cosine similarity")
        axes[row, 2].set_ylabel("pixels")
        axes[row, 2].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires CUDA")
    requested_device = torch.device(args.device)
    if requested_device.type != "cuda":
        raise ValueError("Use a CUDA device for SXH feature evaluation")
    if args.render_chunk_size <= 0:
        raise ValueError("render-chunk-size must be positive")
    if not np.isfinite(args.contrast_multiplier) or args.contrast_multiplier <= 0:
        raise ValueError("contrast-multiplier must be finite and positive")

    torch.manual_seed(args.seed)
    renderer = SXHFeatureTrainingRenderer(
        projection_mode="max", render_chunk_size=args.render_chunk_size, device=args.device
    )
    if renderer.device.type != "cuda":
        raise RuntimeError("SXH feature evaluation requires CUDA")
    device = renderer.device
    checkpoint = load_checkpoint_with_retry(checkpoint_path, device)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint has no model_state_dict")
    model = CoMIRTwoBranchFeatureNetwork().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    crop = CanonicalSquareCrop().to(device)

    pose_a = renderer.trusted_pose
    offset_b = get_random_offset(1, renderer.device)
    pose_b = compose_centered_perturbations(pose_a, renderer.specimen_center_pose, offset_b)
    batch_a = renderer.render_poses(pose_a, contrast_multiplier=args.contrast_multiplier, include_bile_projection=False)
    batch_b = renderer.render_poses(pose_b, contrast_multiplier=args.contrast_multiplier, include_bile_projection=False, offsets=offset_b)
    batches = {id(pose_a): batch_a, id(pose_b): batch_b}

    prepared = {}
    for name, batch in (("A", batch_a), ("B", batch_b)):
        valid_mask = crop(batch.valid_mask.to(dtype=torch.float32), mode="nearest") > 0.5
        if not valid_mask.any():
            raise RuntimeError(f"Canonical valid FOV mask is empty for pose {name}")
        prepared[name] = (crop(batch.ct_drr), crop(batch.mrcp_projection), valid_mask)
        _require_finite(f"canonical CT-DRR {name}", prepared[name][0])
        _require_finite(f"canonical MRCP {name}", prepared[name][1])

    pair_results: list[dict[str, object]] = []
    with torch.no_grad():
        for index, (name, xray_pose, mrcp_pose) in enumerate(pair_specs(pose_a, pose_b)):
            xray_batch = batches[id(xray_pose)]
            mrcp_batch = batches[id(mrcp_pose)]
            xray_prepared = prepared["A" if xray_pose is pose_a else "B"]
            mrcp_prepared = prepared["A" if mrcp_pose is pose_a else "B"]
            xray_image, mrcp_image = xray_prepared[0], mrcp_prepared[1]
            valid_mask = xray_prepared[2]
            if not torch.equal(valid_mask, mrcp_prepared[2]):
                raise RuntimeError("CT-DRR and MRCP canonical FOV masks differ")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                xray_features, mrcp_features = model(xray_image, mrcp_image)
            _require_finite(f"{name} CT-DRR features", xray_features)
            _require_finite(f"{name} MRCP features", mrcp_features)
            pair_results.append(
                {
                    "name": name,
                    "xray_pose": xray_pose,
                    "mrcp_pose": mrcp_pose,
                    "xray_batch": xray_batch,
                    "mrcp_batch": mrcp_batch,
                    "xray_image": xray_image,
                    "mrcp_image": mrcp_image,
                    "xray_features": xray_features,
                    "mrcp_features": mrcp_features,
                    "valid_mask": valid_mask,
                    "similarity": summarize_similarity(
                        xray_features, mrcp_features, seed=args.seed + index, valid_mask=valid_mask
                    ),
                    "is_registration_metric": name != "A/B",
                }
            )

    all_features = [feature for pair in pair_results for feature in (pair["xray_features"], pair["mrcp_features"])]
    all_masks = [pair["valid_mask"] for pair in pair_results for _ in range(2)]
    pca_maps = _joint_pca_rgb(all_features, all_masks)
    step = checkpoint.get("step")
    output_dir = args.output_dir or checkpoint_path.parent.parent / "pose_feature_comparison"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _draw_feature_comparison(output_dir / "pose_feature_comparison.png", pair_results, pca_maps)
    _draw_similarity_maps(output_dir / "similarity_maps.png", pair_results)

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(step) if step is not None else None,
        "device": str(device),
        "seed": args.seed,
        "contrast_multiplier": args.contrast_multiplier,
        "renderer": renderer.metadata(),
        "poses": {
            "A_trusted_global": _pose_metadata(pose_a, coordinate_frame="global CT-to-X-ray"),
            "B_centered_perturbation_global": _pose_metadata(pose_b, coordinate_frame="global CT-to-X-ray"),
            "B_local_centered_offset": _pose_metadata(
                offset_b, coordinate_frame="local-centred offset about the SXH specimen centre"
            ),
        },
        "pairs": [
            {
                "name": pair["name"],
                "xray_pose": "A" if pair["xray_pose"] is pose_a else "B",
                "mrcp_pose": "A" if pair["mrcp_pose"] is pose_a else "B",
                "same_detector_coordinates": True,
                "valid_fov_pixel_count": int(pair["valid_mask"].sum().item()),
                "is_registration_metric": pair["is_registration_metric"],
                "note": "A/B is a deliberate pose mismatch, not a registration metric." if pair["name"] == "A/B" else "Same-pose synthetic positive control.",
                **_summary_metrics(pair["similarity"]),
            }
            for pair in pair_results
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "pairs": summary["pairs"]}, indent=2))


if __name__ == "__main__":
    main()
