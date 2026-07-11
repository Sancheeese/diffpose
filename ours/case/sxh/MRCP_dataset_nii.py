"""SXH MRCP NIfTI + X-ray dataset for DRR projection.

MRCP 501 intensity is resampled onto the CT 3 grid (ICP), then min-max
normalized and passed through the same DRR axis pipeline as sxh CT.
``mr_mask`` (mrcp_006 bile duct on the CT grid) is loaded for optional
``DRRSeg`` use; the current MRCP web server does not overlay it.
"""

from __future__ import annotations

import os
from pathlib import Path

import json

import nibabel as nib
import numpy as np
import torch
from diffpose.calibration import RigidTransform
from nibabel.processing import resample_from_to

from ours.case.sxh.CT_dataset_nii import IntubationDataset, Transforms  # noqa: F401
from ours.case.sxh.CT_dataset_MR import (  # noqa: F401
    DEFAULT_ICP_TRANSFORM,
    DEFAULT_REGISTERED_MR_BILE_DUCT,
    apply_drr_axis_pipeline,
    load_ct_to_mr_transform,
    load_registered_ct_grid_mask,
    resample_mr_to_ct_grid,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CT_NII = PROJECT_ROOT / "mrct" / "data" / "孙新华" / "CT" / "3.nii"
DEFAULT_MRCP_NII = PROJECT_ROOT / "mrct" / "data" / "孙新华" / "MRCP" / "501.nii"
DEFAULT_BILE_DUCT = PROJECT_ROOT / "mrct" / "data-duet" / "bile_duct" / "mrcp_006.nii.gz"
DEFAULT_REGISTERED_MRCP = (
    PROJECT_ROOT
    / "mrct"
    / "outputs"
    / "sxh_ct3_mrcp006_gallbladder_icp"
    / "mrcp_501_registered_to_ct.nii.gz"
)
DEFAULT_XRAY_ROOT = (
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


def normalize_mrcp_for_drr(volume: np.ndarray) -> np.ndarray:
    """Linear min-max normalization to [0, 1] for the existing DRR class."""
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - vmin) / (vmax - vmin)).astype(np.float32)


def resample_mrcp_to_ct_grid(
    mrcp_path: str | Path,
    ct_nii_path: str | Path,
    ct_to_mr_transform: np.ndarray,
) -> np.ndarray:
    """Resample MRCP intensity onto CT's native NIfTI grid using the ICP transform."""
    mrcp_nii = nib.load(str(mrcp_path))
    ct_nii = nib.load(str(ct_nii_path))
    mr_to_ct = np.linalg.inv(ct_to_mr_transform)
    moved = nib.Nifti1Image(
        np.asanyarray(mrcp_nii.dataobj).astype(np.float32),
        mr_to_ct @ mrcp_nii.affine,
        mrcp_nii.header,
    )
    return np.asanyarray(resample_from_to(moved, ct_nii, order=1).dataobj).astype(np.float32)


def load_or_resample_mrcp_to_ct_grid(
    mrcp_path: str | Path,
    ct_nii_path: str | Path,
    ct_to_mr_transform: np.ndarray,
    cached_path: str | Path,
) -> np.ndarray:
    """Load the cached CT-grid MRCP volume, or create it once using linear interpolation."""
    cached_path = Path(cached_path)
    if cached_path.exists():
        return np.asanyarray(nib.load(str(cached_path)).dataobj).astype(np.float32)

    print(f"Resampling MRCP 501 to CT 3 grid: {cached_path}", flush=True)
    mrcp_on_ct = resample_mrcp_to_ct_grid(mrcp_path, ct_nii_path, ct_to_mr_transform)
    ct_nii = nib.load(str(ct_nii_path))
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached = nib.Nifti1Image(mrcp_on_ct, ct_nii.affine, ct_nii.header)
    cached.set_data_dtype(np.float32)
    nib.save(cached, str(cached_path))
    return mrcp_on_ct


class IntubationDatasetMRCP(IntubationDataset):
    """MRCP 501 intensity and MRCP 006 bile duct on the SXH CT DRR grid."""

    def __init__(
        self,
        mrcp_path,
        x_root,
        bile_duct_path: str | Path | None = None,
        ct_nii_path: str | Path = DEFAULT_CT_NII,
        registered_mrcp_path: str | Path = DEFAULT_REGISTERED_MRCP,
        registered_mr_mask_path: str | Path = DEFAULT_REGISTERED_MR_BILE_DUCT,
        icp_transform_path: str | Path = DEFAULT_ICP_TRANSFORM,
        preprocess: bool = True,
        x_offset: int = 0,
        y_offset: int = 0,
        z_offset: int = 0,
        z_cut: int = 0,
        z_cut_end: int = -1,
        factors: list[float] | tuple[float, float, float] | None = None,
    ):
        if factors is None:
            factors = [1.0, 1.0, 1.0]

        self.mrcp_path = Path(mrcp_path)
        self.bile_duct_path = Path(bile_duct_path) if bile_duct_path else DEFAULT_BILE_DUCT
        self.ct_nii_path = Path(ct_nii_path)
        self.registered_mrcp_path = Path(registered_mrcp_path)
        self.registered_mr_mask_path = Path(registered_mr_mask_path)
        self.icp_transform_path = Path(icp_transform_path)

        super().__init__(
            self.ct_nii_path,
            x_root,
            preprocess=preprocess,
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
            z_cut=z_cut,
            z_cut_end=z_cut_end,
            factors=factors,
        )

        ct_to_mr = load_ct_to_mr_transform(self.icp_transform_path)
        mrcp_on_ct_native = load_or_resample_mrcp_to_ct_grid(
            self.mrcp_path,
            self.ct_nii_path,
            ct_to_mr,
            self.registered_mrcp_path,
        )
        self.volume = normalize_mrcp_for_drr(
            apply_drr_axis_pipeline(mrcp_on_ct_native, factors, z_cut=z_cut, z_cut_end=z_cut_end, order=1)
        )
        self.gt_pose_dir = str(PROJECT_ROOT / "diffpose" / "ours" / "gt_pose" / "sxh_mrcp")

        if self.registered_mr_mask_path.exists():
            mask_on_ct_native = load_registered_ct_grid_mask(self.registered_mr_mask_path)
        else:
            mask_on_ct_native = resample_mr_to_ct_grid(
                self.bile_duct_path,
                self.ct_nii_path,
                ct_to_mr,
            )
        self.mr_mask = apply_drr_axis_pipeline(
            mask_on_ct_native,
            factors,
            z_cut=z_cut,
            z_cut_end=z_cut_end,
            order=0,
        )

        if self.mr_mask.shape != self.volume.shape:
            raise ValueError(
                "MRCP bile duct mask shape does not match MRCP projection volume on the CT DRR grid: "
                f"mask={self.mr_mask.shape}, volume={self.volume.shape}"
            )

    def get_manual_gt(self, idx=None):
        """Return saved pose if present; otherwise a zero pose for web server startup."""
        idx_str = f"{idx:04d}"
        pose_file = os.path.join(self.gt_pose_dir, f"pose_{idx_str}.json")
        if not os.path.exists(pose_file):
            return RigidTransform(
                torch.zeros(1, 3),
                torch.zeros(1, 3),
                parameterization="so3_log_map",
            )

        with open(pose_file) as f:
            pose_data = json.load(f)

        pose_params = pose_data["pose_params"]
        rot = torch.tensor([pose_params[:3]], dtype=torch.float32)
        xyz = torch.tensor([pose_params[3:]], dtype=torch.float32)
        return RigidTransform(rot, xyz, parameterization="so3_log_map")

    def get_mr_mask(self) -> np.ndarray:
        return self.mr_mask

    def get_mr_mask_tensor(self, device: str = "cpu"):
        return torch.tensor(self.mr_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)


if __name__ == "__main__":
    specimen = IntubationDatasetMRCP(
        DEFAULT_MRCP_NII,
        DEFAULT_XRAY_ROOT,
        x_offset=20,
        y_offset=200,
        z_offset=100,
        z_cut=250,
        factors=[0.6, 0.6, 1.5],
    )

    print(f"MRCP path:        {specimen.mrcp_path}")
    print(f"Bile duct path:   {specimen.bile_duct_path}")
    print(f"MRCP volume shape:{specimen.volume.shape}")
    print(f"MRCP spacing:     {specimen.spacing}")
    print(f"MRCP intensity:   min={specimen.volume.min():.4f}, max={specimen.volume.max():.4f}")
    print(f"MR mask shape:    {specimen.mr_mask.shape}")
    print(f"MR foreground:    {(specimen.mr_mask > 0).sum()}")
    print(f"X-ray count:      {len(specimen)}")
