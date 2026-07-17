import nibabel as nib
import numpy as np

from ours.xmr.case.sxh.refine_ct_mrcp_from_guidewire import ct_voxels_to_drr_matrix, derive_refinement


def test_refinement_converts_the_trusted_detector_pose_into_the_ct_pose_frame():
    ct_to_xray = np.eye(4)
    mrcp_to_xray = np.eye(4)
    mrcp_to_xray[:3, 3] = [4.0, -2.0, 7.0]

    refinement = derive_refinement(
        ct_to_xray,
        mrcp_to_xray,
        ct_world_from_voxel=np.eye(4),
        drr_from_ct_voxel=np.eye(4),
        ct_to_mr_original=np.eye(4),
    )

    np.testing.assert_allclose(refinement.drr_delta, np.linalg.inv(mrcp_to_xray))
    np.testing.assert_allclose(refinement.mr_to_ct_refined, np.linalg.inv(mrcp_to_xray))
    np.testing.assert_allclose(refinement.ct_to_mr_refined, mrcp_to_xray)
    assert refinement.closure_error == 0.0


def test_refinement_conjugates_the_drr_delta_into_ct_world_coordinates():
    ct_to_xray = np.eye(4)
    mrcp_to_xray = np.eye(4)
    mrcp_to_xray[0, 3] = 10.0
    ct_world_from_voxel = np.diag([2.0, 3.0, 4.0, 1.0])
    drr_from_ct_voxel = np.diag([5.0, 7.0, 11.0, 1.0])

    refinement = derive_refinement(
        ct_to_xray,
        mrcp_to_xray,
        ct_world_from_voxel,
        drr_from_ct_voxel,
        ct_to_mr_original=np.eye(4),
    )

    # -10 DRR units correspond to -4 CT-world units along x in this setup.
    np.testing.assert_allclose(refinement.ct_world_delta[:3, 3], [-4.0, 0.0, 0.0])


def test_ct_voxel_to_drr_matrix_uses_the_renderer_spacing_when_provided():
    ct_nii = nib.Nifti1Image(np.zeros((5, 6, 7), dtype=np.float32), np.eye(4))

    matrix = ct_voxels_to_drr_matrix(
        ct_nii,
        factors=np.array([0.5, 2.0, 3.0]),
        drr_spacing=np.array([4.0, 5.0, 6.0]),
    )

    np.testing.assert_allclose(np.diag(matrix)[:3], [2.0, 10.0, -18.0])
    assert matrix[2, 3] == 108.0
