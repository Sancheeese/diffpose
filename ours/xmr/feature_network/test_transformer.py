import torch

from ours.xmr.feature_network.transformer import PerImageLegacyTransform


def test_per_image_minmax_is_independent_across_batch() -> None:
    transform = PerImageLegacyTransform(size=4, radius=10, mean=0.0, std=1.0)
    first = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    second = torch.tensor([[[[100.0, 110.0], [120.0, 130.0]]]])

    together = transform(torch.cat((first, second), dim=0))
    separately = torch.cat((transform(first), transform(second)), dim=0)

    torch.testing.assert_close(together, separately)


def test_constant_image_is_finite() -> None:
    transform = PerImageLegacyTransform(size=8, radius=3)
    result = transform(torch.ones(2, 1, 4, 5))
    assert result.shape == (2, 1, 8, 8)
    assert torch.isfinite(result).all()
