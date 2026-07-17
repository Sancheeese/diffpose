import torch

from ours.xmr.feature_network.contrastive import SymmetricCrossModalInfoNCE
from ours.xmr.feature_network.model import CommonFeatureNetwork


def test_common_feature_network_is_full_resolution_and_differentiable() -> None:
    model = CommonFeatureNetwork()
    images = torch.randn(2, 1, 64, 64, requires_grad=True)

    descriptors = model(images)
    assert descriptors.shape == (2, 32, 64, 64)
    torch.testing.assert_close(descriptors.norm(dim=1), torch.ones(2, 64, 64), rtol=1e-4, atol=1e-4)

    descriptors.square().mean().backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()


def test_symmetric_infonce_is_finite_and_backpropagates() -> None:
    torch.manual_seed(7)
    raw_descriptors = torch.randn(2, 32, 32, 32, requires_grad=True)
    descriptors = torch.nn.functional.normalize(raw_descriptors, dim=1)
    valid_mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)
    loss = SymmetricCrossModalInfoNCE(samples_per_image=32)(descriptors, descriptors, valid_mask)

    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.xray_to_mrcp)
    assert torch.isfinite(loss.mrcp_to_xray)
    loss.total.backward()
    assert raw_descriptors.grad is not None
    assert torch.isfinite(raw_descriptors.grad).all()
