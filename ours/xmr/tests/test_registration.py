"""Tests for the MRCP/X-ray MIND registration baseline."""

from __future__ import annotations

import torch

from xmr.mind import mind_descriptor, mind_ssd
from xmr.register import search_translation_mind, translate_2d


def _phantom() -> torch.Tensor:
    image = torch.zeros((64, 64), dtype=torch.float32)
    image[18:46, 30:34] = 1.0
    image[30:34, 18:46] = 1.0
    image[22:27, 22:27] = 0.5
    return image


def test_mind_ssd_is_lower_for_aligned_structures() -> None:
    fixed = _phantom()
    shifted = translate_2d(fixed, dy=3, dx=-2)

    assert mind_ssd(fixed, fixed) < mind_ssd(fixed, shifted)


def test_mind_descriptor_supports_three_pixel_offsets() -> None:
    image = _phantom()

    features = mind_descriptor(image, offsets=((0, 3), (3, 3)))

    assert features.shape == (1, 2, 64, 64)


def test_grid_search_recovers_known_translation_inside_roi() -> None:
    fixed = _phantom()
    moving = translate_2d(fixed, dy=3, dx=-2)
    roi = torch.zeros_like(fixed, dtype=torch.bool)
    roi[8:56, 8:56] = True

    result = search_translation_mind(fixed, moving, max_shift=4, mask=roi)

    assert (result.dy, result.dx) == (-3, 2)
    assert result.loss < mind_ssd(fixed, moving, mask=roi)
