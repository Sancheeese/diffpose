from pathlib import Path
import sys

import torch
from diffpose.calibration import RigidTransform


PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "diffpose"))

from ours.case.sxh.CT_dataset_MR import IntubationDatasetMR  # noqa: E402
from ours.xmr.case.sxh.pose_io import load_registered_pose  # noqa: E402
from ours.xmr.case.sxh.web_drr_server_nii_sxh import SXHWebPoseAdjuster  # noqa: E402


CT_NII = PROJECT_ROOT / "mrct" / "data" / "孙新华" / "CT" / "3.nii"
XRAY_ROOT = (
    PROJECT_ROOT
    / "diffpose"
    / "ours"
    / "data"
    / "liwei"
    / "孙新华"
    / "ERCP"
    / "SUNXINHUA^^"
    / "20240712155050"
    / "1"
)
RUNS_MASK_DIR = PROJECT_ROOT / "diffpose" / "ours" / "case" / "sxh" / "runs" / "mask"
SXH_DRR_PARAMS = dict(
    x_offset=20,
    y_offset=200,
    z_offset=100,
    z_cut=250,
    factors=[0.6, 0.6, 1.5],
)


def _make_adjuster() -> SXHWebPoseAdjuster:
    specimen = IntubationDatasetMR(CT_NII, XRAY_ROOT, **SXH_DRR_PARAMS)
    adjuster = object.__new__(SXHWebPoseAdjuster)
    adjuster.device = "cpu"
    adjuster.init_pose_mode = "registered"
    adjuster.runs_mask_dir = RUNS_MASK_DIR
    adjuster.isocenter_pose = specimen.isocenter_pose
    adjuster.back_pose = specimen.back_pose
    adjuster.center_pose = specimen.center_pose
    return adjuster


def _global_pose_from_csv(index: int) -> RigidTransform:
    params = load_registered_pose(index, RUNS_MASK_DIR)
    assert params is not None
    return RigidTransform(
        torch.tensor([params[:3]], dtype=torch.float32),
        torch.tensor([params[3:]], dtype=torch.float32),
        "euler_angles",
        "ZYX",
    )


def test_registered_pose_reconstructs_the_csv_global_transform():
    adjuster = _make_adjuster()
    adjuster.apply_initial_pose(3)

    torch.testing.assert_close(
        adjuster.get_current_pose().get_matrix(),
        _global_pose_from_csv(3).get_matrix(),
    )


def test_zero_mode_uses_the_identity_local_pose():
    adjuster = _make_adjuster()
    adjuster.init_pose_mode = "zero"

    assert adjuster.apply_initial_pose(0) == [0.0] * 6


def test_registered_reset_restores_the_frame_local_base_pose():
    adjuster = _make_adjuster()
    adjuster.current_index = 3
    adjuster.apply_initial_pose(3)
    expected = adjuster.pose_reset.copy()
    adjuster.pose_params[0] += 0.1

    adjuster.reset_pose_params()

    assert adjuster.pose_params == expected
