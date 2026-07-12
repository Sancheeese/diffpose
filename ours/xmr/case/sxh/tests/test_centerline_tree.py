"""Tests for the raw-MRCP biliary skeleton-tree exporter."""

from __future__ import annotations

import numpy as np

from ours.xmr.case.sxh.centerline import CenterlineTree, build_viewer_html, extract_centerline_tree


def test_extract_centerline_tree_keeps_a_three_voxel_line_in_affine_world_space() -> None:
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1:4, 2, 2] = 1
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    tree = extract_centerline_tree(mask, affine)

    assert tree.skeleton_mask.dtype == np.uint8
    assert tree.skeleton_mask.sum() == 3
    np.testing.assert_allclose(
        tree.vertices_mm,
        np.array([[12.0, 26.0, 38.0], [14.0, 26.0, 38.0], [16.0, 26.0, 38.0]]),
    )
    assert tree.edges.tolist() == [[0, 1], [1, 2]]


def test_viewer_contains_independent_mask_and_centerline_controls() -> None:
    tree = CenterlineTree(
        skeleton_mask=np.zeros((1, 1, 1), dtype=np.uint8),
        vertices_mm=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        edges=np.array([[0, 1]], dtype=np.int32),
    )

    html = build_viewer_html(np.array([[0.0, 0.0, 0.0]]), tree)

    assert 'id="showMask"' in html
    assert 'id="showCenterline"' in html
    assert "gl.LINES" in html
    assert "drawMask" in html
    assert "drawCenterline" in html
