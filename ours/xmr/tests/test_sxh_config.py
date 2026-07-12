"""Tests for the fixed initial MIND configuration of the SXH case."""

from xmr.case.sxh.config import SXH_MIND_PARAMETERS, SXH_TRANSLATION_MAX_SHIFT


def test_sxh_uses_eight_directions_at_three_offset_distances() -> None:
    assert SXH_MIND_PARAMETERS.patch_radius == 1
    unit_offsets = (
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )
    assert SXH_MIND_PARAMETERS.offsets == tuple(
        (dy * distance, dx * distance)
        for distance in (1, 2, 3)
        for dy, dx in unit_offsets
    )
    assert SXH_MIND_PARAMETERS.gaussian_sigma == 0.6
    assert SXH_MIND_PARAMETERS.normalization_percentiles == (1.0, 99.0)
    assert SXH_MIND_PARAMETERS.roi_erode_pixels == 2
    assert SXH_MIND_PARAMETERS.pyramid_scales == (1.0, 0.5, 0.25)
    assert SXH_TRANSLATION_MAX_SHIFT == 32
