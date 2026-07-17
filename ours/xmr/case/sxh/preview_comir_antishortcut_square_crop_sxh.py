"""Preview canonical square crop inputs for SXH CoMIR anti-shortcut training."""

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
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ours.xmr.case.sxh.feature_training_renderer import SXHFeatureTrainingRenderer  # noqa: E402
from ours.xmr.case.sxh.image_io import to_zero_one, write_gray_png  # noqa: E402
from ours.xmr.feature_network_comir_v2 import CanonicalSquareCrop  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "runs" / "feature_common_comir_antishortcut" / "previews" / "square_crop"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview SXH canonical square crop preprocessing")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contrast-multiplier", type=float, default=3.0)
    return parser.parse_args()


def draw_rectangle(image: np.ndarray, start: int, end: int) -> np.ndarray:
    gray = (to_zero_one(image) * 255).astype(np.uint8)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(color, (start, start), (end - 1, end - 1), (0, 255, 0), 2)
    return color


def write_overview(path: Path, images: list[tuple[str, np.ndarray]], bounds_start: int, bounds_end: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for title, image in images:
        if title.startswith("raw"):
            tile = draw_rectangle(image, bounds_start, bounds_end)
        else:
            gray = (to_zero_one(image) * 255).astype(np.uint8)
            tile = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.putText(tile, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    top = np.concatenate(tiles[:2], axis=1)
    bottom = np.concatenate(tiles[2:], axis=1)
    overview = np.concatenate([top, bottom], axis=0)
    if not cv2.imwrite(str(path), overview):
        raise IOError(f"Failed to write image: {path}")


def main() -> None:
    args = parse_args()
    renderer = SXHFeatureTrainingRenderer(projection_mode="max", render_chunk_size=1, device=args.device)
    cropper = CanonicalSquareCrop(image_size=256, radius=119, output_size=256).to(renderer.device)
    batch = renderer.render_poses(
        renderer.trusted_pose,
        contrast_multiplier=args.contrast_multiplier,
        include_bile_projection=True,
    )

    with torch.no_grad():
        ct_square = cropper(batch.ct_drr)
        mrcp_square = cropper(batch.mrcp_projection)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_ct = batch.ct_drr[0, 0].detach().cpu().numpy()
    raw_mrcp = batch.mrcp_projection[0, 0].detach().cpu().numpy()
    square_ct = ct_square[0, 0].detach().cpu().numpy()
    square_mrcp = mrcp_square[0, 0].detach().cpu().numpy()

    write_gray_png(output_dir / "raw_ct_drr.png", raw_ct)
    write_gray_png(output_dir / "raw_mrcp_max.png", raw_mrcp)
    write_gray_png(output_dir / "canonical_square_ct_drr.png", square_ct)
    write_gray_png(output_dir / "canonical_square_mrcp_max.png", square_mrcp)
    write_overview(
        output_dir / "square_crop_overview.png",
        [
            ("raw CT-DRR", raw_ct),
            ("raw MRCP max", raw_mrcp),
            ("square CT-DRR", square_ct),
            ("square MRCP max", square_mrcp),
        ],
        cropper.bounds.start_y,
        cropper.bounds.end_y,
    )

    metadata = {
        "input_size": 256,
        "fov_radius": 119,
        "square_bounds": cropper.bounds.__dict__,
        "output_size": 256,
        "output_dir": str(output_dir),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
