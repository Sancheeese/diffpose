from pathlib import Path
import sys

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "diffpose"))

from ours.case.sxh.CT_dataset import IntubationDataset as DicomIntubationDataset  # noqa: E402
from ours.case.sxh.CT_dataset_MR import (  # noqa: E402
    DEFAULT_ICP_TRANSFORM,
    DEFAULT_REGISTERED_MR_BILE_DUCT,
    IntubationDatasetMR,
    apply_drr_axis_pipeline,
    load_ct_to_mr_transform,
    load_registered_ct_grid_mask,
)
from ours.case.sxh.CT_dataset_nii import IntubationDataset  # noqa: E402
from ours.case.sxh.MRCP_dataset_nii import (  # noqa: E402
    DEFAULT_MRCP_NII,
    resample_mrcp_to_ct_grid,
)


CT_NII = PROJECT_ROOT / "mrct" / "data" / "孙新华" / "CT" / "3.nii"
CT_DICOM_ROOT = (
    PROJECT_ROOT
    / "diffpose"
    / "ours"
    / "data"
    / "liwei"
    / "孙新华"
    / "CT"
    / "SunXinHua"
    / "20240711020550.905000"
    / "3"
)
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
SXH_DRR_PARAMS = dict(
    x_offset=20,
    y_offset=200,
    z_offset=100,
    z_cut=250,
    factors=[0.6, 0.6, 1.5],
)


def test_sxh_nii_dataset_matches_existing_dicom_dataset_pipeline():
    dicom = DicomIntubationDataset(
        str(CT_DICOM_ROOT),
        str(XRAY_ROOT),
        **SXH_DRR_PARAMS,
    )
    nii = IntubationDataset(
        CT_NII,
        XRAY_ROOT,
        **SXH_DRR_PARAMS,
    )

    assert nii.volume.shape == dicom.volume.shape == (307, 307, 321)
    np.testing.assert_allclose(nii.spacing, dicom.spacing, rtol=1e-6)
    np.testing.assert_allclose(nii.volume, dicom.volume, rtol=1e-6, atol=2e-4)
    np.testing.assert_allclose(
        nii.isocenter_pose.get_translation().numpy(),
        dicom.isocenter_pose.get_translation().numpy(),
        rtol=1e-6,
    )
    assert len(nii) == len(dicom) > 0
    assert nii.fiducials.shape[-1] == 3


def test_sxh_mrcp_bile_duct_mask_projects_on_ct_drr_grid():
    specimen = IntubationDatasetMR(
        CT_NII,
        XRAY_ROOT,
        **SXH_DRR_PARAMS,
    )

    assert specimen.volume.shape == specimen.mr_mask.shape
    assert specimen.mr_mask.dtype == np.float32
    assert int((specimen.mr_mask > 0).sum()) > 0


def test_sxh_mrcp_projection_uses_the_ct_drr_grid():
    axis_params = dict(z_cut=SXH_DRR_PARAMS["z_cut"], factors=SXH_DRR_PARAMS["factors"])
    ct_raw = np.asanyarray(nib.load(str(CT_NII)).dataobj).astype(np.float32)
    ct_pipeline = apply_drr_axis_pipeline(ct_raw, **axis_params, order=1)

    ct_to_mr = load_ct_to_mr_transform(DEFAULT_ICP_TRANSFORM)
    mrcp_on_ct = resample_mrcp_to_ct_grid(DEFAULT_MRCP_NII, CT_NII, ct_to_mr)
    mrcp_pipeline = apply_drr_axis_pipeline(mrcp_on_ct, **axis_params, order=1)

    mask_on_ct = load_registered_ct_grid_mask(DEFAULT_REGISTERED_MR_BILE_DUCT)
    mask_pipeline = apply_drr_axis_pipeline(mask_on_ct, **axis_params, order=0)

    assert mrcp_pipeline.shape == mask_pipeline.shape == ct_pipeline.shape == (307, 307, 321)
    assert int((mrcp_pipeline > 0).sum()) > 0
    assert int((mask_pipeline > 0).sum()) > 0
