import numpy as np
import pytest
import torch

from ours.xmr.case.sxh.pose_feature_comparison import (
    _as_display,
    _pose_metadata,
    pair_specs,
    summarize_similarity,
)


def test_pair_specs_returns_correct_and_mismatched_pose_pairs_in_report_order():
    pose_a = object()
    pose_b = object()

    assert pair_specs(pose_a, pose_b) == [
        ("A/A", pose_a, pose_a),
        ("B/B", pose_b, pose_b),
        ("A/B", pose_a, pose_b),
    ]


def test_similarity_summary_separates_matched_from_permuted_features():
    xray = torch.eye(4).reshape(1, 4, 2, 2)
    mrcp = xray.clone()
    result = summarize_similarity(xray, mrcp, seed=7)
    assert result["corresponding_mean"] == pytest.approx(1.0)
    assert result["corresponding_mean"] > result["permuted_mean"]


def test_similarity_summary_is_deterministic_for_the_same_seed():
    xray = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
    mrcp = xray.flip(-1)

    first = summarize_similarity(xray, mrcp, seed=11)
    second = summarize_similarity(xray, mrcp, seed=11)

    torch.testing.assert_close(first["permuted_samples"], second["permuted_samples"])


def test_similarity_summary_excludes_invalid_fov_pixels_from_every_metric():
    xray = torch.tensor([[[[1.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]])
    mrcp = xray.clone()
    valid_mask = torch.tensor([[[[True, True, False]]]])
    altered_mrcp = mrcp.clone()
    altered_mrcp[..., 2] = torch.tensor([[-1.0], [0.0]])

    baseline = summarize_similarity(xray, mrcp, seed=3, valid_mask=valid_mask)
    altered = summarize_similarity(xray, altered_mrcp, seed=3, valid_mask=valid_mask)

    for key in baseline:
        torch.testing.assert_close(baseline[key], altered[key])
    assert baseline["corresponding_samples"].numel() == 2


class _PoseForMetadata:
    def get_rotation(self, *_args):
        return torch.tensor([[0.1, 0.2, 0.3]])

    def get_translation(self):
        return torch.tensor([[1.0, 2.0, 3.0]])

    def get_matrix(self):
        return torch.eye(4).unsqueeze(0)


def test_pose_metadata_records_explicit_global_and_local_coordinate_frames():
    global_metadata = _pose_metadata(_PoseForMetadata(), coordinate_frame="global CT-to-X-ray")
    local_metadata = _pose_metadata(_PoseForMetadata(), coordinate_frame="local-centred offset")

    assert global_metadata["coordinate_frame"] == "global CT-to-X-ray"
    assert local_metadata["coordinate_frame"] == "local-centred offset"


def test_display_percentiles_ignore_invalid_fov_pixels():
    valid_mask = torch.tensor([[[[True, True, False, False]]]])
    image = torch.tensor([[[[1.0, 2.0, 1_000_000.0, -1_000_000.0]]]])
    altered = image.clone()
    altered[..., 2:] = torch.tensor([[[[-3_000_000.0, 3_000_000.0]]]])

    displayed = _as_display(image[0, 0], valid_mask)
    altered_displayed = _as_display(altered[0, 0], valid_mask)

    np.testing.assert_allclose(displayed[:, :2], altered_displayed[:, :2])
    np.testing.assert_array_equal(displayed[:, 2:], 0.0)


@pytest.mark.parametrize("shape", [(0, 3, 2, 2), (1, 3, 0, 2)])
def test_similarity_summary_rejects_feature_tensors_without_samples(shape):
    features = torch.empty(shape)

    with pytest.raises(ValueError, match="at least one spatial sample"):
        summarize_similarity(features, features)


def test_similarity_summary_rejects_feature_tensors_without_channels():
    features = torch.empty(1, 0, 2, 2)

    with pytest.raises(ValueError, match="at least one channel"):
        summarize_similarity(features, features)


def test_similarity_summary_rejects_inputs_on_different_devices():
    xray = torch.ones(1, 3, 2, 2)
    mrcp = torch.ones(1, 3, 2, 2, device="meta")

    with pytest.raises(ValueError, match="same device"):
        summarize_similarity(xray, mrcp)
