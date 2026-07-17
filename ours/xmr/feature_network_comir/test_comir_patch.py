import torch

from ours.xmr.feature_network_comir import CoMIRTwoBranchFeatureNetwork, SymmetricPatchInfoNCE


def test_two_branch_network_outputs_normalized_full_resolution_features() -> None:
    model = CoMIRTwoBranchFeatureNetwork(feature_channels=32)
    xray = torch.randn(2, 1, 64, 64, requires_grad=True)
    mrcp = torch.randn(2, 1, 64, 64, requires_grad=True)

    xray_features, mrcp_features = model(xray, mrcp)

    assert xray_features.shape == (2, 32, 64, 64)
    assert mrcp_features.shape == (2, 32, 64, 64)
    torch.testing.assert_close(xray_features.norm(dim=1), torch.ones(2, 64, 64), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(mrcp_features.norm(dim=1), torch.ones(2, 64, 64), rtol=1e-4, atol=1e-4)


def test_patch_infonce_is_finite_and_backpropagates() -> None:
    torch.manual_seed(11)
    raw_xray = torch.randn(2, 32, 64, 64, requires_grad=True)
    raw_mrcp = torch.randn(2, 32, 64, 64, requires_grad=True)
    xray_features = torch.nn.functional.normalize(raw_xray, dim=1)
    mrcp_features = torch.nn.functional.normalize(raw_mrcp, dim=1)
    valid_mask = torch.ones(2, 1, 64, 64, dtype=torch.bool)

    criterion = SymmetricPatchInfoNCE(patches_per_image=8, patch_size=16, local_negative_exclusion=16)
    output = criterion(xray_features, mrcp_features, valid_mask)

    assert torch.isfinite(output.total)
    assert torch.isfinite(output.xray_to_mrcp)
    assert torch.isfinite(output.mrcp_to_xray)
    assert output.patch_descriptors_shape == (16, 32 * 16 * 16)
    output.total.backward()
    assert raw_xray.grad is not None
    assert raw_mrcp.grad is not None
    assert torch.isfinite(raw_xray.grad).all()
    assert torch.isfinite(raw_mrcp.grad).all()
