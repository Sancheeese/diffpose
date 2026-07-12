"""Load SXH CT-X-ray registration poses from register_sxh_mask CSV outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_RUNS_MASK_DIR = PROJECT_ROOT / "diffpose" / "ours" / "case" / "sxh" / "runs" / "mask"

REQUIRED_POSE_COLUMNS = ("alpha", "beta", "gamma", "bx", "by", "bz")
RESULT_RE = re.compile(r"sxh_xray(\d+)_se3_log_map\.csv$")


@dataclass(frozen=True)
class RegisteredPose:
    index: int
    params: list[float]
    source_csv: Path


def registered_pose_csv_path(index: int, runs_dir: str | Path = DEFAULT_RUNS_MASK_DIR) -> Path:
    return Path(runs_dir) / f"sxh_xray{index:03d}_se3_log_map.csv"


def parse_final_pose_row(csv_path: str | Path) -> RegisteredPose:
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
    return RegisteredPose(index=idx, params=params, source_csv=path)


def load_registered_pose(index: int, runs_dir: str | Path = DEFAULT_RUNS_MASK_DIR) -> list[float] | None:
    """Return euler ZYX pose params for one X-ray index, or None if CSV is missing."""
    csv_path = registered_pose_csv_path(index, runs_dir)
    if not csv_path.is_file():
        print(f"warning: registration CSV not found: {csv_path}")
        return None

    try:
        pose = parse_final_pose_row(csv_path)
    except Exception as exc:
        print(f"warning: failed to read registration pose from {csv_path}: {exc}")
        return None

    if pose.index != index:
        print(
            f"warning: CSV index {pose.index} does not match requested index {index}; "
            f"using CSV values from {csv_path}"
        )

    return pose.params


def discover_registered_indices(runs_dir: str | Path = DEFAULT_RUNS_MASK_DIR) -> list[int]:
    """Return sorted dataset indices that have sxh_xrayNNN registration CSV files."""
    runs_path = Path(runs_dir)
    indices: list[int] = []
    for path in runs_path.glob("sxh_xray*_se3_log_map.csv"):
        match = RESULT_RE.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def default_registered_index(runs_dir: str | Path = DEFAULT_RUNS_MASK_DIR) -> int | None:
    """Return the smallest index with a registration CSV, or None."""
    indices = discover_registered_indices(runs_dir)
    return indices[0] if indices else None
