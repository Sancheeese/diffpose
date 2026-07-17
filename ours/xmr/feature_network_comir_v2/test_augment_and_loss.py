import torch

from ours.xmr.feature_network_comir_v2 import (
    CropC4Parameters,
    CrossModalPatchInfoNCE,
    IndependentCropC4Augment,
    invert_legacy_standardized_intensity,
)


def test_c4_coordinate_mapping_matches_torch_rot90() -> None:
    augment = IndependentCropC4Augment(crop_sizes=(256,))
    parameters = CropC4Parameters(
        crop_size=torch.tensor([256]),
        xray_origin_yx=torch.tensor([[0, 0]]),
        mrcp_origin_yx=torch.tensor([[0, 0]]),
        xray_rotation_k=torch.tensor([1]),
        mrcp_rotation_k=torch.tensor([0]),
    )
    mapped = augment.canonical_to_output(torch.tensor([[[10.0, 20.0]]]), parameters, "xray")
    assert torch.allclose(mapped[0, 0], torch.tensor([235.0, 10.0]))


def test_cross_modal_patch_loss_uses_24_pairs_per_rendered_image() -> None:
    torch.manual_seed(3)
    batch_size = 2
    augment = IndependentCropC4Augment(crop_sizes=(224,), max_relative_shift=0)
    parameters = augment.sample_parameters(batch_size, torch.device("cpu"))
    features_x = torch.randn(batch_size, 32, 256, 256, requires_grad=True)
    features_m = torch.randn(batch_size, 32, 256, 256, requires_grad=True)
    criterion = CrossModalPatchInfoNCE(patch_pairs_per_image=24, patch_size=32)
    result = criterion(features_x, features_m, parameters, augment)
    assert result.descriptor_shape == (batch_size, 24, 32 * 32 * 32)
    assert torch.isfinite(result.total)
    result.total.backward()
    assert features_x.grad is not None and torch.isfinite(features_x.grad).all()
    assert features_m.grad is not None and torch.isfinite(features_m.grad).all()


def test_mrcp_black_white_inversion_happens_in_underlying_unit_intensity_space() -> None:
    unit_intensity = torch.tensor([0.0, 0.25, 0.75, 1.0])
    standardized = (unit_intensity - 0.3080) / 0.1494
    inverted = invert_legacy_standardized_intensity(standardized)
    reconstructed = inverted * 0.1494 + 0.3080
    assert torch.allclose(reconstructed, 1.0 - unit_intensity)
