"""MRCP intensity DRR: normalized volume projection without CT HU remapping."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from diffdrr.detector import Detector, make_xrays
from diffdrr.utils import Transform3d
from fastcore.basics import patch

from ours.utils import siddon as siddon_mod
from ours.utils.drr import reshape_subsampled_drr
from ours.utils.siddon import siddon_raycast

__all__ = ["DRRMRCP", "siddon_raycast_max"]


def siddon_raycast_max(
    source: torch.Tensor,
    target: torch.Tensor,
    volume: torch.Tensor,
    spacing: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Siddon ray cast returning the maximum sampled voxel intensity along each ray."""
    dims = torch.tensor(volume.shape, device=volume.device) + 1
    alphas, maxidx = siddon_mod._get_alphas(source, target, spacing, dims, eps)
    alphamid = (alphas[..., 0:-1] + alphas[..., 1:]) / 2

    voxels = siddon_mod._get_voxel(alphamid, source, target, volume, spacing, dims, maxidx, eps)

    step_length = torch.diff(alphas, dim=-1)
    voxels = voxels.masked_fill(torch.isnan(step_length), float("-inf"))

    drr = voxels.max(dim=-1).values
    return torch.where(torch.isfinite(drr), drr, torch.zeros_like(drr))


class DRRMRCP(nn.Module):
    """Project normalized MRCP intensity with Siddon ray casting (no CT HU remapping)."""

    def __init__(
        self,
        volume: np.ndarray,
        spacing: np.ndarray,
        sdr: float,
        height: int,
        delx: float,
        width: int | None = None,
        dely: float | None = None,
        x0: float = 0.0,
        y0: float = 0.0,
        p_subsample: float | None = None,
        reshape: bool = True,
        reverse_x_axis: bool = False,
        patch_size: int | None = None,
        projection_mode: str = "sum",
    ):
        super().__init__()

        if projection_mode not in {"sum", "max"}:
            raise ValueError(f"projection_mode must be 'sum' or 'max', got {projection_mode!r}")

        width = height if width is None else width
        dely = delx if dely is None else dely
        n_subsample = int(height * width * p_subsample) if p_subsample is not None else None
        self.detector = Detector(
            sdr,
            height,
            width,
            delx,
            dely,
            x0,
            y0,
            n_subsample=n_subsample,
            reverse_x_axis=reverse_x_axis,
        )

        intensity = np.asarray(volume, dtype=np.float32)
        self.register_buffer("spacing", torch.tensor(spacing, dtype=torch.float32))
        self.register_buffer("volume", torch.tensor(intensity).flip([0]))
        self.reshape = reshape
        self.patch_size = patch_size
        self.projection_mode = projection_mode
        if self.patch_size is not None:
            self.n_patches = (height * width) // (self.patch_size**2)

    def reshape_transform(self, img: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.reshape:
            if self.detector.n_subsample is None:
                img = img.view(-1, 1, self.detector.height, self.detector.width)
            else:
                img = reshape_subsampled_drr(img, self.detector, batch_size)
        return img

    def _raycast(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.projection_mode == "max":
            return siddon_raycast_max(source, target, self.volume, self.spacing)
        return siddon_raycast(source, target, self.volume, self.spacing)


@patch
def forward(
    self: DRRMRCP,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    parameterization: str,
    convention: str = None,
    pose: Transform3d = None,
):
    """Render an MRCP intensity DRR for the given pose."""
    if pose is None:
        assert len(rotation) == len(translation)
        batch_size = len(rotation)
        source, target = self.detector(
            rotation=rotation,
            translation=translation,
            parameterization=parameterization,
            convention=convention,
        )
    else:
        batch_size = len(pose)
        source, target = make_xrays(pose, self.detector.source, self.detector.target)

    if self.patch_size is not None:
        n_points = target.shape[1] // self.n_patches
        img_parts = []
        for idx in range(self.n_patches):
            t = target[:, idx * n_points : (idx + 1) * n_points]
            img_parts.append(self._raycast(source, t))
        img = torch.cat(img_parts, dim=1)
    else:
        img = self._raycast(source, target)

    return self.reshape_transform(img, batch_size=batch_size)
