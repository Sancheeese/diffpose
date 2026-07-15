"""Tests for the SXH centerline-projection web server helpers."""

from __future__ import annotations

import json

import numpy as np
import torch
from diffdrr.detector import Detector
from diffpose.calibration import RigidTransform

from ours.xmr.case.sxh.guidewire_registration import (
    PoseOptimizationResult,
    optimize_fixed_chain_pose,
    select_fixed_chain,
    tangent_alignment_loss,
)
from ours.xmr.case.sxh.register_sxh_guidewire_centerline import save_registration_result

from ours.xmr.case.sxh.web_drr_server_nii_sxh_centerline import (
    build_centerline_http_response,
    ct_voxels_to_drr_mm,
    decompose_graph_into_chains,
    extract_guidewire_points,
    extract_ordered_guidewire_chain,
    guidewire_indices_for_specimen,
    matching_guidewire_mask_path,
    project_points_to_detector,
    reflect_points_across_y_equals_x,
    render_chain_points_overlay,
    render_soft_centerline,
    subsample_chain_points,
)


def _identity_pose() -> RigidTransform:
    return RigidTransform(
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        parameterization="euler_angles",
        convention="ZYX",
    )


def test_project_points_to_detector_maps_isocenter_to_detector_center() -> None:
    detector = Detector(sdr=100.0, height=5, width=5, delx=1.0, dely=1.0, x0=0.0, y0=0.0)

    pixels = project_points_to_detector(torch.tensor([[0.0, 0.0, 0.0]]), _identity_pose(), detector)

    torch.testing.assert_close(pixels, torch.tensor([[2.0, 2.0]]), atol=1e-5, rtol=0)


def test_soft_centerline_raster_is_continuous_and_differentiable() -> None:
    detector = Detector(sdr=100.0, height=9, width=9, delx=1.0, dely=1.0, x0=0.0, y0=0.0)
    vertices = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
    edges = torch.tensor([[0, 1]], dtype=torch.int64)

    image = render_soft_centerline(vertices, edges, _identity_pose(), detector, sigma_px=0.8)
    image.sum().backward()

    assert image.shape == (1, 1, 9, 9)
    assert float(image[0, 0, 4, 4]) > 0.5
    assert torch.isfinite(vertices.grad).all()


def test_ct_voxel_mapping_does_not_apply_an_extra_x_flip() -> None:
    ct_voxels = torch.tensor([[10.0, 20.0, 30.0]]).numpy()

    drr_mm = ct_voxels_to_drr_mm(
        ct_voxels,
        ct_shape=(100, 100, 80),
        drr_spacing=torch.tensor([2.0, 3.0, 4.0]).numpy(),
        factors=torch.tensor([0.5, 0.5, 2.0]).numpy(),
    )

    # DRR/DRRSeg flips axis 0 internally; only the input pipeline's z flip remains here.
    torch.testing.assert_close(torch.tensor(drr_mm), torch.tensor([[10.0, 30.0, 392.0]], dtype=torch.float64))


def test_matching_guidewire_mask_uses_the_xray_stem(tmp_path) -> None:
    expected = tmp_path / "93968938_20240712_1_132.nii.gz"
    expected.touch()
    (tmp_path / "93968938_20240712_1_165.nii.gz").touch()

    result = matching_guidewire_mask_path(tmp_path / "93968938_20240712_1_132.dcm", tmp_path)

    assert result == expected


def test_extract_guidewire_points_keeps_the_largest_skeleton_component() -> None:
    mask = torch.zeros((12, 12), dtype=torch.uint8).numpy()
    mask[2:9, 5] = 1
    mask[10, 10] = 1

    points = extract_guidewire_points(mask)

    assert len(points) == 7
    assert set(points[:, 0]) == {5.0}
    assert points[:, 1].min() == 2.0
    assert points[:, 1].max() == 8.0


def test_reflect_points_across_y_equals_x_swaps_detector_coordinates() -> None:
    points = torch.tensor([[5.0, 2.0], [5.0, 8.0]]).numpy()

    reflected = reflect_points_across_y_equals_x(points)

    torch.testing.assert_close(torch.tensor(reflected), torch.tensor([[2.0, 5.0], [8.0, 5.0]]))


def test_extract_ordered_guidewire_chain_runs_between_the_two_endpoints() -> None:
    mask = torch.zeros((12, 12), dtype=torch.uint8).numpy()
    mask[2:9, 5] = 1

    chain = extract_ordered_guidewire_chain(mask)

    assert len(chain) == 7
    assert {tuple(chain[0]), tuple(chain[-1])} == {(5.0, 2.0), (5.0, 8.0)}
    assert np.all(np.linalg.norm(np.diff(chain, axis=0), axis=1) <= np.sqrt(2.0))


def test_decompose_graph_into_chains_splits_a_y_tree_at_its_junction() -> None:
    edges = torch.tensor([[0, 1], [1, 2], [1, 3]], dtype=torch.long)

    chains = decompose_graph_into_chains(4, edges)

    assert {tuple(chain.tolist()) for chain in chains} == {(0, 1), (1, 2), (1, 3)}


def test_decompose_graph_into_chains_merges_adjacent_junction_voxels() -> None:
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [2, 4], [1, 5]], dtype=torch.long)

    chains = decompose_graph_into_chains(6, edges)

    assert {tuple(chain.tolist()) for chain in chains} == {(0, 1), (2, 3), (2, 4), (1, 5)}


def test_subsample_chain_points_keeps_endpoints_and_minimum_display_spacing() -> None:
    points = np.column_stack((np.arange(6, dtype=np.float32), np.zeros(6, dtype=np.float32)))

    displayed = subsample_chain_points(points, min_spacing_px=2.5)

    np.testing.assert_allclose(displayed, np.array([[0.0, 0.0], [3.0, 0.0], [5.0, 0.0]], dtype=np.float32))


def test_chain_overlay_uses_uniform_bile_duct_and_guidewire_colors() -> None:
    overlay = render_chain_points_overlay(
        np.zeros((12, 12), dtype=np.float32),
        np.array([[2.0, 2.0]], dtype=np.float32),
        [np.array([0], dtype=np.int64)],
        np.array([[8.0, 8.0]], dtype=np.float32),
    )

    np.testing.assert_array_equal(overlay[2, 2], np.array([128, 255, 255], dtype=np.uint8))
    np.testing.assert_array_equal(overlay[8, 8], np.array([0, 0, 255], dtype=np.uint8))


def test_select_fixed_chain_prefers_chain_covering_the_guidewire() -> None:
    guidewire = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    chains = [
        torch.tensor([[0.0, 4.0], [2.0, 4.0]]),
        torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
    ]

    assert select_fixed_chain(guidewire, chains) == 1


def test_tangent_loss_is_zero_for_reversed_collinear_chains() -> None:
    guidewire = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    duct = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

    assert torch.isclose(tangent_alignment_loss(guidewire, duct), torch.tensor(0.0), atol=1e-6)


def test_fixed_chain_pose_optimization_has_finite_result_and_preserves_a_match() -> None:
    detector = Detector(sdr=100.0, height=9, width=9, delx=1.0, dely=1.0, x0=0.0, y0=0.0)
    initial_pose = _identity_pose()
    vertices = torch.tensor([[0.0, -20.0, 0.0], [0.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    guidewire = project_points_to_detector(vertices, initial_pose, detector).detach()

    result = optimize_fixed_chain_pose(
        vertices,
        guidewire,
        detector,
        initial_pose,
        project_points_to_detector,
        n_iters=3,
    )

    assert torch.isfinite(result.final_loss)
    assert torch.isfinite(result.pose.get_matrix()).all()
    assert torch.linalg.vector_norm(result.pose.get_se3_log() - initial_pose.get_se3_log()) < 0.2


def test_guidewire_registration_result_contains_locked_chain_and_losses(tmp_path) -> None:
    result = PoseOptimizationResult(
        _identity_pose(),
        [
            {"total": 5.0, "position": 4.0, "tangent": 0.5, "pose": 0.0},
            {"total": 2.0, "position": 1.0, "tangent": 0.5, "pose": 0.0},
        ],
    )

    result_path = save_registration_result(tmp_path, result, selected_chain_index=3, selection_score=1.25)
    payload = json.loads(result_path.read_text())

    assert payload["selected_chain_index"] == 3
    assert payload["selection_score"] == 1.25
    assert payload["losses"]["initial"]["total"] == 5.0
    assert payload["losses"]["final"]["total"] == 2.0


def test_guidewire_indices_for_specimen_uses_matching_dicom_stems(tmp_path) -> None:
    (tmp_path / "frame_132.nii.gz").touch()

    class _Specimen:
        x_file = ["frame_101.dcm", "frame_132.dcm"]

        def get_x_filename(self, index: int) -> str:
            return self.x_file[index]

        def __len__(self) -> int:
            return len(self.x_file)

    assert guidewire_indices_for_specimen(_Specimen(), tmp_path) == [1]


def test_centerline_http_response_replaces_mrcp_panel_with_centerline_overlay() -> None:
    html = "<title>SXH DRR + MRCP</title><div>MRCP Projection</div>"

    updated = build_centerline_http_response(html)

    assert "SXH DRR + Centerline" in updated
    assert "MRCP Branch Chains / Guidewire Chain" in updated
    assert "MRCP Projection" not in updated
