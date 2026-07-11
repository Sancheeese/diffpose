"""WFL CT + X-ray dataset with MRCP bile duct segmentation on the CT DRR grid.

The CT/MRCP ICP matrix is a patient-space transform between the two volumes.
For WFL, the matrix was estimated from gallbladder masks, but the projection
target used here is the MRCP 019 bile duct mask.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from scipy.ndimage import zoom

script_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_path)

from CT_dataset_nii import IntubationDataset  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ICP_TRANSFORM = (
    PROJECT_ROOT
    / "mrct"
    / "outputs"
    / "wfl_ct306_mrcp019_gallbladder_icp"
    / "ct_to_mr_icp_transform.txt"
)
DEFAULT_RAW_MR_BILE_DUCT = PROJECT_ROOT / "mrct" / "data-duet" / "bile_duct" / "mrcp_019.nii.gz"
DEFAULT_REGISTERED_MR_BILE_DUCT = (
    PROJECT_ROOT
    / "mrct"
    / "outputs"
    / "wfl_ct306_mrcp019_gallbladder_icp"
    / "mr_bile_duct_registered_to_ct.nii.gz"
)


def load_ct_to_mr_transform(path: str | Path) -> np.ndarray:
    transform = np.loadtxt(str(path), dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got shape {transform.shape} from {path}")
    return transform


def resample_mr_to_ct_grid(
    mr_path: str | Path,
    ct_nii_path: str | Path,
    ct_to_mr_transform: np.ndarray,
    label_value: int | float | None = None,
) -> np.ndarray:
    """Warp an MR-space label image onto the native CT NIfTI voxel grid."""
    mr_nii = nib.load(str(mr_path))
    ct_nii = nib.load(str(ct_nii_path))

    data = np.asanyarray(mr_nii.dataobj)
    if label_value is None:
        mask = (data > 0).astype(np.float32)
    else:
        mask = (data == label_value).astype(np.float32)

    mr_to_ct = np.linalg.inv(ct_to_mr_transform)
    moved_affine = mr_to_ct @ mr_nii.affine
    moved = nib.Nifti1Image(mask, moved_affine, mr_nii.header)
    resampled = resample_from_to(moved, ct_nii, order=0)
    return np.asanyarray(resampled.dataobj).astype(np.float32)


def load_registered_ct_grid_mask(mask_path: str | Path, label_value: int | float | None = None) -> np.ndarray:
    nii = nib.load(str(mask_path))
    data = np.asanyarray(nii.dataobj)
    if label_value is None:
        return (data > 0).astype(np.float32)
    return (data == label_value).astype(np.float32)


def apply_drr_axis_pipeline(
    volume: np.ndarray,
    factors: list[float] | tuple[float, float, float],
    z_cut: int = 0,
    z_cut_end: int = -1,
    order: int = 0,
) -> np.ndarray:
    """Match ``CT_dataset_nii.IntubationDataset`` post-load axis processing."""
    vol = np.flip(volume, axis=2).copy()
    if z_cut > 0 or z_cut_end != -1:
        vol = vol[:, :, z_cut:z_cut_end]
    if not np.allclose(factors, 1.0):
        vol = zoom(vol, factors, order=order)
    vol = np.flip(vol, axis=0).copy()
    return vol


class IntubationDatasetMR(IntubationDataset):
    """CT NIfTI + ERCP X-rays, with MRCP 019 bile duct aligned to CT 306."""

    def __init__(
        self,
        nii_path,
        x_root,
        registered_mr_mask_path: str | Path | None = None,
        raw_mr_mask_path: str | Path | None = None,
        icp_transform_path: str | Path | None = None,
        label_value: int | float | None = None,
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

        self.registered_mr_mask_path = (
            Path(registered_mr_mask_path) if registered_mr_mask_path else DEFAULT_REGISTERED_MR_BILE_DUCT
        )
        self.raw_mr_mask_path = Path(raw_mr_mask_path) if raw_mr_mask_path else DEFAULT_RAW_MR_BILE_DUCT
        self.icp_transform_path = Path(icp_transform_path) if icp_transform_path else DEFAULT_ICP_TRANSFORM
        self.label_value = label_value
        self._drr_factors = list(factors)
        self._drr_z_cut = z_cut
        self._drr_z_cut_end = z_cut_end

        super().__init__(
            nii_path,
            x_root,
            preprocess=preprocess,
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
            z_cut=z_cut,
            z_cut_end=z_cut_end,
            factors=factors,
        )

        if self.registered_mr_mask_path.exists():
            mr_on_ct_native = load_registered_ct_grid_mask(
                self.registered_mr_mask_path,
                label_value=self.label_value,
            )
        else:
            ct_to_mr = load_ct_to_mr_transform(self.icp_transform_path)
            mr_on_ct_native = resample_mr_to_ct_grid(
                self.raw_mr_mask_path,
                self.nii_path,
                ct_to_mr,
                label_value=self.label_value,
            )

        self.mr_mask = apply_drr_axis_pipeline(
            mr_on_ct_native,
            self._drr_factors,
            z_cut=self._drr_z_cut,
            z_cut_end=self._drr_z_cut_end,
            order=0,
        )

        if self.mr_mask.shape != self.volume.shape:
            raise ValueError(
                "MRCP bile duct mask shape does not match CT volume after DRR preprocessing: "
                f"mr={self.mr_mask.shape}, ct={self.volume.shape}"
            )

    def get_mr_mask(self) -> np.ndarray:
        return self.mr_mask

    def get_mr_mask_tensor(self, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.tensor(self.mr_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)


if __name__ == "__main__":
    nii_path = "/home/zsr/project/mrct/data/王凤兰/CT/306.nii"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/王凤兰/ERCP/FENGLAN^WANG^/20240313160330/1"

    specimen = IntubationDatasetMR(
        nii_path,
        x_root,
        x_offset=20,
        z_offset=50,
        z_cut=30,
        z_cut_end=250,
        factors=[0.5, 0.5, 1],
    )

    print(f"CT volume shape:  {specimen.volume.shape}")
    print(f"MR mask shape:    {specimen.mr_mask.shape}")
    print(f"MR foreground voxels: {(specimen.mr_mask > 0).sum()}")
