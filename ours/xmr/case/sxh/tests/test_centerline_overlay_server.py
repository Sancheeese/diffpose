"""Tests for the SXH centerline-projection web server helpers."""

from __future__ import annotations

import torch
from diffdrr.detector import Detector
from diffpose.calibration import RigidTransform

from ours.xmr.case.sxh.web_drr_server_nii_sxh_centerline import (
    build_centerline_http_response,
    ct_voxels_to_drr_mm,
    project_points_to_detector,
    render_soft_centerline,
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


def test_centerline_http_response_replaces_mrcp_panel_with_centerline_overlay() -> None:
    html = "<title>SXH DRR + MRCP</title><div>MRCP Projection</div>"

    updated = build_centerline_http_response(html)

    assert "SXH DRR + Centerline" in updated
    assert "MRCP Centerline Overlay" in updated
    assert "MRCP Projection" not in updated
