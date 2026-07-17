import torch

from ours.xmr.feature_network_comir_v2 import CanonicalSquareCrop, canonical_square_bounds


def test_canonical_square_bounds_for_sxh_fov() -> None:
    bounds = canonical_square_bounds(image_size=256, radius=119)
    assert bounds.side == 168
    assert bounds.start_y == 44
    assert bounds.end_y == 212
    assert bounds.start_x == 44
    assert bounds.end_x == 212


def test_canonical_square_crop_resizes_to_output_size() -> None:
    cropper = CanonicalSquareCrop(image_size=256, radius=119, output_size=256)
    image = torch.randn(2, 1, 256, 256)
    output = cropper(image)
    assert output.shape == (2, 1, 256, 256)
