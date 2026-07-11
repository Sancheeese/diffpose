from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "diffpose"))

from ours.case.wfl.CT_dataset import IntubationDataset as DicomIntubationDataset  # noqa: E402
from ours.case.wfl.CT_dataset_MR import IntubationDatasetMR  # noqa: E402
from ours.case.wfl.CT_dataset_nii import IntubationDataset  # noqa: E402

CT_NII = PROJECT_ROOT / "mrct" / "data" / "王凤兰" / "CT" / "306.nii"
CT_DICOM_ROOT = (
    PROJECT_ROOT
    / "diffpose"
    / "ours"
    / "data"
    / "liwei"
    / "王凤兰"
    / "CT"
    / "WangFengLan"
    / "20240311144245"
    / "306"
)
XRAY_ROOT = (
    PROJECT_ROOT
    / "diffpose"
    / "ours"
    / "data"
    / "liwei"
    / "王凤兰"
    / "ERCP"
    / "FENGLAN^WANG^"
    / "20240313160330"
    / "1"
)
WFL_DRR_PARAMS = dict(
    x_offset=20,
    z_offset=50,
    z_cut=30,
    z_cut_end=250,
    factors=[0.5, 0.5, 1.0],
)


def test_wfl_nii_dataset_matches_existing_dicom_dataset_pipeline():
    dicom = DicomIntubationDataset(
        str(CT_DICOM_ROOT),
        str(XRAY_ROOT),
        **WFL_DRR_PARAMS,
    )
    nii = IntubationDataset(
        CT_NII,
        XRAY_ROOT,
        **WFL_DRR_PARAMS,
    )

    assert nii.volume.shape == dicom.volume.shape == (256, 256, 220)
    np.testing.assert_allclose(nii.spacing, dicom.spacing, rtol=1e-6)
    np.testing.assert_allclose(nii.volume, dicom.volume, rtol=1e-6, atol=2e-4)
    np.testing.assert_allclose(
        nii.isocenter_pose.get_translation().numpy(),
        dicom.isocenter_pose.get_translation().numpy(),
        rtol=1e-6,
    )
    assert len(nii) == len(dicom) > 0
    assert nii.fiducials.shape[-1] == 3


def test_wfl_mrcp_bile_duct_mask_projects_on_ct_drr_grid():
    specimen = IntubationDatasetMR(
        CT_NII,
        XRAY_ROOT,
        **WFL_DRR_PARAMS,
    )

    assert specimen.volume.shape == specimen.mr_mask.shape
    assert specimen.mr_mask.dtype == np.float32
    assert int((specimen.mr_mask > 0).sum()) > 0
