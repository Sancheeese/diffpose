"""Render SXH MRCP bile duct overlays from CT-Xray registration CSVs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DIFFPOSE_ROOT = PROJECT_ROOT / "diffpose"
OURS_ROOT = DIFFPOSE_ROOT / "ours"
SXH_ROOT = OURS_ROOT / "case" / "sxh"

DEFAULT_RUNS_DIR = SXH_ROOT / "runs" / "mask"
DEFAULT_OUTPUT_DIR = SXH_ROOT / "mrcp_xray_overlays"
DEFAULT_CT_NII = PROJECT_ROOT / "mrct" / "data" / "孙新华" / "CT" / "3.nii"
DEFAULT_XRAY_ROOT = (
    OURS_ROOT
    / "data"
    / "liwei"
    / "孙新华"
    / "ERCP"
    / "SUNXINHUA^^"
    / "20240712155050"
    / "1"
)

REQUIRED_POSE_COLUMNS = ("alpha", "beta", "gamma", "bx", "by", "bz")
RESULT_RE = re.compile(r"sxh_xray(\d+)_se3_log_map\.csv$")


@dataclass(frozen=True)
class FinalPose:
    idx: int
    params: list[float]
    source_csv: Path
    metrics: dict[str, object]


def discover_result_csvs(runs_dir: str | Path) -> list[Path]:
    """Return primary SXH registration CSVs, excluding summaries and nested methods."""
    runs_path = Path(runs_dir)
    files = []
    for path in runs_path.glob("sxh_xray*_se3_log_map.csv"):
        if path.is_file() and RESULT_RE.match(path.name):
            files.append(path)

    return sorted(files, key=lambda p: int(RESULT_RE.match(p.name).group(1)))


def parse_final_pose_row(csv_path: str | Path) -> FinalPose:
    """Read the final registration pose from the last row of a result CSV."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty registration CSV: {path}")

    missing = [col for col in REQUIRED_POSE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing pose columns {missing} in {path}")

    row = df.iloc[-1]
    match = RESULT_RE.match(path.name)
    if match:
        idx = int(match.group(1))
    elif "idx" in row:
        idx = int(row["idx"])
    else:
        raise ValueError(f"Cannot determine X-ray index from {path}")

    params = [float(row[col]) for col in REQUIRED_POSE_COLUMNS]
    metrics = {}
    for col, value in row.items():
        if col in REQUIRED_POSE_COLUMNS:
            continue
        if pd.isna(value):
            metrics[col] = ""
        elif isinstance(value, np.generic):
            metrics[col] = value.item()
        else:
            metrics[col] = value

    return FinalPose(idx=idx, params=params, source_csv=path, metrics=metrics)


def to_zero_one(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + eps)


def write_gray_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = (to_zero_one(image) * 255).astype(np.uint8)
    if not cv2.imwrite(str(path), image_u8):
        raise IOError(f"Failed to write image: {path}")


def write_overlay_png(path: Path, background: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    background_u8 = (to_zero_one(background) * 255).astype(np.uint8)
    mask_u8 = (to_zero_one(mask) * 255).astype(np.uint8)

    bg_color = cv2.cvtColor(background_u8, cv2.COLOR_GRAY2BGR)
    red_mask = np.zeros_like(bg_color)
    red_mask[:, :, 2] = 255
    alpha_mask = (mask_u8 * alpha / 255.0).astype(np.float32)
    result = bg_color.astype(np.float32) * (1 - alpha_mask[:, :, None]) + red_mask.astype(np.float32) * alpha_mask[:, :, None]
    result = np.clip(result, 0, 255).astype(np.uint8)

    if not cv2.imwrite(str(path), result):
        raise IOError(f"Failed to write image: {path}")


def render_cases(
    csv_paths: list[Path],
    output_dir: Path,
    device_arg: str = "auto",
) -> list[dict[str, object]]:
    import torch

    for path in (DIFFPOSE_ROOT, OURS_ROOT, SXH_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    from diffpose.calibration import RigidTransform
    from ours.case.sxh.CT_dataset_MR import IntubationDatasetMR
    from ours.case.sxh.CT_dataset_nii import Transforms
    from ours.utils.drr import DRR
    from ours.utils.drr_seg import DRRSeg

    if device_arg == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    specimen = IntubationDatasetMR(
        DEFAULT_CT_NII,
        DEFAULT_XRAY_ROOT,
        x_offset=20,
        y_offset=200,
        z_offset=100,
        z_cut=250,
        factors=[0.6, 0.6, 1.5],
    )

    height = 256
    subsample = 512 / height
    delx = specimen.delx * subsample

    drr = DRR(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.sdr,
        height=height,
        delx=delx,
        reverse_x_axis=True,
        bone_attenuation_multiplier=3,
    ).to(device)
    drr_bile_duct = DRRSeg(
        specimen.mr_mask,
        specimen.spacing,
        sdr=specimen.sdr,
        height=height,
        delx=delx,
        reverse_x_axis=True,
    ).to(device)
    transforms = Transforms(height)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for csv_path in csv_paths:
        pose_info = parse_final_pose_row(csv_path)
        row: dict[str, object] = {
            "idx": pose_info.idx,
            "source_csv": str(csv_path),
            "status": "ok",
        }

        try:
            rot = torch.tensor([pose_info.params[:3]], dtype=torch.float32, device=device)
            xyz = torch.tensor([pose_info.params[3:]], dtype=torch.float32, device=device)
            pose = RigidTransform(rot, xyz, parameterization="euler_angles", convention="ZYX")

            with torch.no_grad():
                pred_drr = drr(None, None, None, pose=pose, bone_attenuation_multiplier=3)
                pred_drr = transforms(pred_drr).to(device).to(torch.float32)
                drr_np = pred_drr.squeeze().detach().cpu().numpy()

                pred_mask = drr_bile_duct(None, None, None, pose=pose)
                pred_mask = transforms.resize(pred_mask)
                mask_np = pred_mask.squeeze().detach().cpu().numpy()
                mask_np = (mask_np > 0.01).astype(np.float32)

            img, _ = specimen[pose_info.idx]
            background = transforms(img, reverse=False).to(device).to(torch.float32)
            background_np = background.squeeze().detach().cpu().numpy()

            overlay_path = output_dir / "overlay" / f"sxh_xray{pose_info.idx:03d}_overlay.png"
            drr_path = output_dir / "drr" / f"sxh_xray{pose_info.idx:03d}_drr.png"

            write_overlay_png(overlay_path, background_np, mask_np)
            write_gray_png(drr_path, drr_np)

            row.update(
                {
                    "overlay_path": str(overlay_path),
                    "drr_path": str(drr_path),
                    "alpha": pose_info.params[0],
                    "beta": pose_info.params[1],
                    "gamma": pose_info.params[2],
                    "bx": pose_info.params[3],
                    "by": pose_info.params[4],
                    "bz": pose_info.params[5],
                }
            )
            row.update(pose_info.metrics)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)

        rows.append(row)

    return rows


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "idx",
        "status",
        "source_csv",
        "overlay_path",
        "drr_path",
        "alpha",
        "beta",
        "gamma",
        "bx",
        "by",
        "bz",
        "ncc",
        "ssim",
        "fiducial",
        "tre",
        "error",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    csv_paths = discover_result_csvs(args.runs_dir)
    if args.indices:
        wanted = set(args.indices)
        csv_paths = [path for path in csv_paths if int(RESULT_RE.match(path.name).group(1)) in wanted]
    if args.limit is not None:
        csv_paths = csv_paths[: args.limit]

    if not csv_paths:
        print(f"No registration CSVs found in {args.runs_dir}")
        return 1

    rows = render_cases(csv_paths, args.output_dir, device_arg=args.device)
    write_summary(args.output_dir / "summary.csv", rows)

    ok = sum(1 for row in rows if row.get("status") == "ok")
    print(f"Rendered {ok}/{len(rows)} cases")
    print(f"Output: {args.output_dir}")
    return 0 if ok == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
