"""Fixed first-pass parameters for the SXH MRCP-X-ray MIND experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Mind2DParameters:
    """Parameters shared by X-ray and MRCP-projection MIND extraction."""

    patch_radius: int
    offsets: tuple[tuple[int, int], ...]
    gaussian_sigma: float
    normalization_percentiles: tuple[float, float]
    roi_erode_pixels: int
    pyramid_scales: tuple[float, ...]
    epsilon: float = 1e-6


PROJECT_ROOT = Path(__file__).resolve().parents[5]
SXH_XRAY_ROOT = (
    PROJECT_ROOT
    / "diffpose"
    / "ours"
    / "data"
    / "liwei"
    / "孙新华"
    / "ERCP"
    / "SUNXINHUA^^"
    / "20240712155050"
    / "1"
)
SXH_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

_EIGHT_DIRECTIONS = (
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)

SXH_MIND_PARAMETERS = Mind2DParameters(
    patch_radius=1,
    offsets=tuple(
        (dy * distance, dx * distance)
        for distance in (1, 2, 3)
        for dy, dx in _EIGHT_DIRECTIONS
    ),
    gaussian_sigma=0.6,
    normalization_percentiles=(1.0, 99.0),
    roi_erode_pixels=2,
    pyramid_scales=(1.0, 0.5, 0.25),
)
SXH_TRANSLATION_MAX_SHIFT = 32
