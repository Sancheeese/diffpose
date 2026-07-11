"""SXH CT NIfTI + X-ray dataset.

The converted CT NIfTI for Sun Xinhua has the same axis order and spacing as
the existing DICOM stack after ``swapaxes(0, 2)``. Its slice axis is reversed
relative to DICOM filename order, so this module reuses the WFL NIfTI loader,
which applies ``np.flip(volume, axis=2)`` before the standard DRR pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
import torch
from diffpose.calibration import RigidTransform

from ours.case.wfl.CT_dataset_nii import (  # noqa: F401
    IntubationDataset as WFLNiiIntubationDataset,
    Transforms,
    create_circle_mask,
    create_circle_mask_reverse,
    get_random_offset,
    preprocess,
    toZeroOne,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class IntubationDataset(WFLNiiIntubationDataset):
    """NIfTI equivalent of ``sxh.CT_dataset.IntubationDataset``."""

    def __init__(
        self,
        nii_path,
        x_root,
        preprocess=True,
        x_offset=0,
        y_offset=0,
        z_offset=0,
        z_cut=0,
        z_cut_end=-1,
        factors=[1, 1, 1],
    ):
        super().__init__(
            nii_path,
            x_root,
            preprocess=preprocess,
            x_offset=0,
            y_offset=0,
            z_offset=0,
            z_cut=0,
            z_cut_end=-1,
            factors=[1, 1, 1],
        )
        self.gt_pose_dir = str(PROJECT_ROOT / "diffpose" / "ours" / "gt_pose" / "sxh")

        # The WFL base constructor applies the shared final x-axis flip. Undo
        # it so SXH can crop first, matching sxh.CT_dataset exactly.
        self.volume = np.flip(self.volume, axis=0).copy()

        if z_cut > 0:
            self.volume = self.volume[:, :, :z_cut]

        isocenter_xyz = [self.volume.shape[0] - x_offset, self.volume.shape[1] - y_offset, self.volume.shape[2] - z_offset]
        isocenter_xyz = torch.tensor(isocenter_xyz) * torch.tensor(self.spacing) / 2
        self.isocenter_pose = RigidTransform(
            torch.tensor([[-torch.pi / 2, 0.0, torch.pi / 2]]),
            isocenter_xyz.unsqueeze(0),
            "euler_angles",
            "ZYX",
        )

        center_xyz = [self.volume.shape[0] - x_offset, self.volume.shape[1] - y_offset, self.volume.shape[2] - z_offset]
        center_xyz = torch.tensor(center_xyz) * torch.tensor(self.spacing) / 2
        self.center_pose = RigidTransform(
            torch.tensor([[0.0, 0.0, 0.0]]),
            center_xyz.unsqueeze(0),
            "euler_angles",
            "ZYX",
        )

        back_xyz = [-self.volume.shape[0] + x_offset, -self.volume.shape[1] + y_offset, -self.volume.shape[2] + z_offset]
        back_xyz = torch.tensor(back_xyz) * torch.tensor(self.spacing) / 2
        self.back_pose = RigidTransform(
            torch.tensor([[0.0, 0.0, 0.0]]),
            back_xyz.unsqueeze(0),
            "euler_angles",
            "ZYX",
        )

        self.volume = zoom(self.volume, factors, order=1)
        self.spacing = self.spacing / np.asarray(factors)
        self.volume = np.flip(self.volume, axis=0).copy()

    def get_fiducials(self):
        fiducials = None
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fid_sxh.json")
        with open(file_path) as f:
            data = json.load(f)
            for point in data["markups"][0]["controlPoints"]:
                p = torch.tensor(point["position"]).unsqueeze(0)
                p[..., 2] = -p[..., 2]
                if fiducials is None:
                    fiducials = p
                else:
                    fiducials = torch.concat((fiducials, p), dim=0)

        fiducials = fiducials.unsqueeze(0)
        return self.lps2volume.transform_points(fiducials)


if __name__ == "__main__":
    nii_path = PROJECT_ROOT / "mrct" / "data" / "孙新华" / "CT" / "3.nii"
    x_root = (
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

    specimen = IntubationDataset(
        nii_path,
        x_root,
        x_offset=20,
        y_offset=200,
        z_offset=100,
        z_cut=250,
        factors=[0.6, 0.6, 1.5],
    )

    print(f"CT volume shape: {specimen.volume.shape}")
    print(f"CT spacing:      {specimen.spacing}")
