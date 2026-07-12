"""Tests for SXH X-ray conversion and MIND output naming."""

from pathlib import Path

import torch

from xmr.case.sxh.extract_xray_mind import extract_mind_from_array, preprocess_xray_for_mind


def test_extractor_writes_png_and_mind_tensor_with_source_stem(tmp_path: Path) -> None:
    pixels = torch.full((32, 32), 100.0)
    pixels[10:22, 15:17] = 20.0
    source = Path("93968938_20240712_1_158.dcm")

    png_path, mind_path = extract_mind_from_array(pixels, source, tmp_path)

    image_dir = tmp_path / "93968938_20240712_1_158"
    assert png_path == image_dir / "93968938_20240712_1_158.png"
    assert mind_path == image_dir / "93968938_20240712_1_158.mind.pt"
    assert png_path.is_file()
    assert mind_path.is_file()
    assert torch.load(mind_path, weights_only=True).shape == (24, 32, 32)

    visualization_dir = image_dir / "visualizations"
    expected_images = {
        "channel_00.png",
        "channel_01.png",
        "channel_02.png",
        "channel_03.png",
        "channel_04.png",
        "channel_05.png",
        "channel_06.png",
        "channel_07.png",
        "channel_08.png",
        "channel_09.png",
        "channel_10.png",
        "channel_11.png",
        "channel_12.png",
        "channel_13.png",
        "channel_14.png",
        "channel_15.png",
        "channel_16.png",
        "channel_17.png",
        "channel_18.png",
        "channel_19.png",
        "channel_20.png",
        "channel_21.png",
        "channel_22.png",
        "channel_23.png",
        "mean.png",
        "max.png",
        "min.png",
    }
    assert {path.name for path in visualization_dir.iterdir()} == expected_images


def test_display_preprocessing_preserves_monochrome2_brightness_order() -> None:
    pixels = torch.full((32, 32), 100.0)
    pixels[:, 16:] = 500.0

    image = preprocess_xray_for_mind(pixels)

    assert image[:, 20:].mean() > image[:, :12].mean()
