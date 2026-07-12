"""Convert SXH ERCP DICOM images to PNG and 2D MIND tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from xmr.mind import mind_descriptor

from .config import Mind2DParameters, SXH_MIND_PARAMETERS, SXH_OUTPUT_DIR, SXH_XRAY_ROOT


def _save_feature_image(feature: torch.Tensor, path: Path) -> None:
    """Save a single normalized MIND feature image without per-image rescaling."""
    array = (feature.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def save_mind_visualizations(features: torch.Tensor, visualization_dir: str | Path) -> Path:
    """Write per-channel and channel-aggregate MIND maps to one directory."""
    if features.ndim != 3:
        raise ValueError("features must have shape (C, H, W)")

    visualization_dir = Path(visualization_dir)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    for channel, feature in enumerate(features):
        _save_feature_image(feature, visualization_dir / f"channel_{channel:02d}.png")

    _save_feature_image(features.mean(dim=0), visualization_dir / "mean.png")
    _save_feature_image(features.max(dim=0).values, visualization_dir / "max.png")
    _save_feature_image(features.min(dim=0).values, visualization_dir / "min.png")
    return visualization_dir


def preprocess_xray_for_mind(
    pixels: torch.Tensor,
    parameters: Mind2DParameters = SXH_MIND_PARAMETERS,
) -> torch.Tensor:
    """Apply the SXH display and MIND preprocessing to one 2D X-ray image."""
    if pixels.ndim != 2:
        raise ValueError("pixels must have shape (H, W)")

    image = pixels.detach().cpu().numpy().astype(np.float32, copy=False)
    # SXH DICOM uses MONOCHROME2 with a display pixel relationship. Preserve
    # its low-to-high brightness order instead of applying a physical -log map.
    lower, upper = np.percentile(image, parameters.normalization_percentiles)
    if upper <= lower:
        normalized = np.zeros_like(image, dtype=np.float32)
    else:
        normalized = np.clip((image - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)

    smoothed = gaussian_filter(normalized, sigma=parameters.gaussian_sigma).astype(np.float32)
    return torch.from_numpy(smoothed)


def extract_mind_from_array(
    pixels: torch.Tensor,
    source_path: str | Path,
    output_dir: str | Path,
    parameters: Mind2DParameters = SXH_MIND_PARAMETERS,
) -> tuple[Path, Path]:
    """Write one display PNG and one MIND tensor using the DICOM source stem."""
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    image_dir = output_dir / source_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)

    image = preprocess_xray_for_mind(pixels, parameters)
    features = mind_descriptor(
        image,
        patch_radius=parameters.patch_radius,
        offsets=parameters.offsets,
        epsilon=parameters.epsilon,
    ).squeeze(0).cpu()

    png_path = image_dir / f"{source_path.stem}.png"
    mind_path = image_dir / f"{source_path.stem}.mind.pt"
    Image.fromarray((image.numpy() * 255.0).round().astype(np.uint8), mode="L").save(png_path)
    torch.save(features, mind_path)
    save_mind_visualizations(features, image_dir / "visualizations")
    return png_path, mind_path


def _read_dicom_pixels(path: Path) -> torch.Tensor:
    dataset = pydicom.dcmread(path)
    pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    if pixels.ndim != 2:
        raise ValueError(f"Only single-frame 2D DICOM is supported: {path}")
    if getattr(dataset, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        pixels = pixels.max() + pixels.min() - pixels
    return torch.from_numpy(pixels.copy())


def extract_mind_from_dicom(
    source_path: str | Path,
    output_dir: str | Path = SXH_OUTPUT_DIR,
    parameters: Mind2DParameters = SXH_MIND_PARAMETERS,
) -> tuple[Path, Path]:
    """Convert one single-frame ERCP DICOM to its matching PNG and MIND file."""
    source_path = Path(source_path)
    return extract_mind_from_array(_read_dicom_pixels(source_path), source_path, output_dir, parameters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SXH ERCP X-ray PNG and MIND tensors.")
    parser.add_argument("input", type=Path, nargs="?", default=SXH_XRAY_ROOT)
    parser.add_argument("--output", type=Path, default=SXH_OUTPUT_DIR)
    args = parser.parse_args()

    sources = [args.input] if args.input.is_file() else sorted(args.input.glob("*.dcm"))
    if not sources:
        raise FileNotFoundError(f"No DICOM files found under {args.input}")
    for source in sources:
        png_path, mind_path = extract_mind_from_dicom(source, args.output)
        print(f"{source.name} -> {png_path.name}, {mind_path.name}")


if __name__ == "__main__":
    main()
