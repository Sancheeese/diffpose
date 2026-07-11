"""CT + X-ray dataset with MRCP segmentation aligned to CT for DRR projection.

For ysy (杨式瑜), the MRCP label is ``mrcp_003`` bile duct:
``mrct/data-duet/bile_duct/mrcp_003.nii.gz``.

Pipeline (order matters):
  1. Resample MR mask onto the CT NIfTI grid in patient physical space (ICP transform).
  2. Apply the same axis pipeline as ``CT_dataset_nii`` (transpose → z_cut → zoom → flip).
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

from CT_dataset_nii import IntubationDataset, Transforms  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ICP_TRANSFORM = PROJECT_ROOT / "mrct" / "outputs" / "gallbladder_icp" / "ct_to_mr_icp_transform.txt"
# ysy (杨式瑜) MRCP segmentation: mrcp_003 bile duct
DEFAULT_MR_MASK = PROJECT_ROOT / "mrct" / "data-duet" / "bile_duct" / "mrcp_003.nii.gz"


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
    """Warp MR (moving) onto the native CT NIfTI voxel grid using the ICP matrix."""
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


def apply_drr_axis_pipeline(
    volume: np.ndarray,
    factors: list[float] | tuple[float, float, float],
    z_cut: int = 0,
    order: int = 0,
) -> np.ndarray:
    """Match ``CT_dataset_nii.IntubationDataset`` post-load axis reordering."""
    vol = np.transpose(volume, (0, 2, 1)).copy()
    if z_cut > 0:
        vol = vol[:, :, :z_cut]
    if not np.allclose(factors, 1.0):
        vol = zoom(vol, factors, order=order)
    vol = np.flip(vol, axis=0).copy()
    return vol


class IntubationDatasetMR(IntubationDataset):
    """CT NIfTI + ERCP X-rays, with MRCP mask rigidly aligned to the CT volume."""

    def __init__(
        self,
        nii_path,
        x_root,
        mr_mask_path: str | Path | None = None,
        icp_transform_path: str | Path | None = None,
        label_value: int | float | None = None,
        preprocess: bool = True,
        x_offset: int = 0,
        y_offset: int = 0,
        z_offset: int = 0,
        z_cut: int = 0,
        factors: list[float] | tuple[float, float, float] | None = None,
    ):
        if factors is None:
            factors = [1.0, 1.0, 1.0]

        self.mr_mask_path = Path(mr_mask_path) if mr_mask_path else DEFAULT_MR_MASK
        self.icp_transform_path = (
            Path(icp_transform_path) if icp_transform_path else DEFAULT_ICP_TRANSFORM
        )
        self.label_value = label_value
        self._drr_factors = list(factors)
        self._drr_z_cut = z_cut

        super().__init__(
            nii_path,
            x_root,
            preprocess=preprocess,
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
            z_cut=z_cut,
            factors=factors,
        )

        ct_to_mr = load_ct_to_mr_transform(self.icp_transform_path)
        mr_on_ct_native = resample_mr_to_ct_grid(
            self.mr_mask_path,
            self.nii_path,
            ct_to_mr,
            label_value=self.label_value,
        )
        self.mr_mask = apply_drr_axis_pipeline(
            mr_on_ct_native,
            self._drr_factors,
            z_cut=self._drr_z_cut,
            order=0,
        )

        if self.mr_mask.shape != self.volume.shape:
            raise ValueError(
                "MR mask shape does not match CT volume after DRR preprocessing: "
                f"mr={self.mr_mask.shape}, ct={self.volume.shape}"
            )

    def get_mr_mask(self) -> np.ndarray:
        return self.mr_mask

    def get_mr_mask_tensor(self, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.tensor(self.mr_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from ours.utils.drr_seg import DRRSeg

    nii_path = "/home/zsr/project/mrct/data/杨式瑜/CT/7.nii"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/杨式瑜/ERCP/SHIYU^YANG^/20240515170510/1"

    specimen = IntubationDatasetMR(
        nii_path,
        x_root,
        y_offset=50,
        z_offset=-50,
        z_cut=400,
        factors=[0.7, 1.5, 0.7],
    )

    print(f"CT volume shape:  {specimen.volume.shape}")
    print(f"MR mask shape:    {specimen.mr_mask.shape}")
    print(f"MR foreground voxels: {(specimen.mr_mask > 0).sum()}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    height = 256
    subsample = 512 / height
    delx = specimen.delx * subsample

    drr_seg = DRRSeg(
        specimen.mr_mask,
        specimen.spacing,
        sdr=specimen.sdr,
        height=height,
        delx=delx,
        reverse_x_axis=True,
    ).to(device)

    pose = specimen.isocenter_pose.to(device)
    proj = drr_seg(None, None, None, pose=pose)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(specimen.mr_mask.max(axis=0), cmap="hot")
    axes[0].set_title("Bile duct mask (ysy mrcp_003, axial MIP)")
    axes[1].imshow(proj.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Bile duct seg DRR (binary)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
