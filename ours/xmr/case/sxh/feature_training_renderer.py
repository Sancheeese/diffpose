"""SXH-specific online CT-DRR/MRCP pair renderer for common-feature training."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from diffpose.calibration import RigidTransform
from ours.case.my_util2 import get_random_offset
from ours.xmr.case.sxh.web_drr_server_nii_sxh_refined_mrcp import (
    DEFAULT_CT_NII,
    DEFAULT_REFINED_BILE_DUCT,
    DEFAULT_REFINED_CT_TO_MR,
    DEFAULT_REFINED_MRCP,
    DEFAULT_START_INDEX,
    DEFAULT_GUIDEWIRE_OVERLAY,
    build_adjuster,
    load_guidewire_chain_from_overlay,
    parse_args as parse_refined_server_args,
)
from ours.case.sxh.MRCP_dataset_nii import DEFAULT_MRCP_NII, IntubationDatasetMRCP
from ours.utils.drr_mrcp import DRRMRCP
from ours.xmr.case.sxh.web_drr_server_nii_sxh_centerline import (
    DEFAULT_CENTERLINE_NII,
    DEFAULT_GUIDEWIRE_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_CENTERLINE_OUTPUT_DIR,
    DEFAULT_RUNS_MASK_DIR,
    DRR,
    DRRSeg,
    DEFAULT_XRAY_ROOT,
    IntubationDatasetMR,
    SXHCenterlineWebPoseAdjuster,
    SXH_DRR_PARAMS,
    Transforms,
    load_raw_centerline_graph_in_drr_coordinates,
)
from ours.xmr.feature_network.transformer import PerImageLegacyTransform


PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_ORIGINAL_CENTERLINE_MASK = PROJECT_ROOT / "mrct" / "data-duet" / "bile_duct" / "mrcp_006.nii.gz"


@dataclass(frozen=True)
class SXHFeatureTrainingBatch:
    """One same-pose CT-DRR/MRCP batch, plus optional bile projection for QA."""

    ct_drr: Tensor
    mrcp_projection: Tensor
    valid_mask: Tensor
    poses: RigidTransform
    offsets: RigidTransform
    contrast_multiplier: float
    ct_render_seconds: float
    mrcp_render_seconds: float
    bile_projection: Tensor | None = None


def compose_centered_perturbations(
    trusted_pose: RigidTransform,
    specimen_center_pose: RigidTransform,
    offsets: RigidTransform,
) -> RigidTransform:
    """Map old local SXH offsets around the trusted automatic xray031 pose."""
    return trusted_pose.compose(specimen_center_pose.inverse()).compose(offsets).compose(specimen_center_pose)


class SXHFeatureTrainingRenderer:
    """Build once and render online samples using the refined xray031 geometry."""

    def __init__(
        self,
        projection_mode: str = "max",
        render_chunk_size: int = 2,
        device: str | None = None,
        mrcp_registration: str = "refined",
    ) -> None:
        if projection_mode not in {"max", "sum"}:
            raise ValueError(f"projection_mode must be 'max' or 'sum', got {projection_mode!r}")
        if mrcp_registration not in {"refined", "original"}:
            raise ValueError(f"mrcp_registration must be 'refined' or 'original', got {mrcp_registration!r}")
        if render_chunk_size <= 0:
            raise ValueError("render_chunk_size must be positive")

        self.mrcp_registration = mrcp_registration
        if mrcp_registration == "refined":
            server_argv = ["--projection", projection_mode]
            if device is not None:
                server_argv.extend(["--device", device])
            server_args = parse_refined_server_args(server_argv)
            self.adjuster = build_adjuster(server_args)
        else:
            self.adjuster = self._build_original_adjuster(projection_mode, device)
        self.device = torch.device(self.adjuster.device)
        if self.device.type != "cuda":
            raise RuntimeError("SXH feature training requires CUDA; run in the real GPU environment")

        self.adjuster.current_index = DEFAULT_START_INDEX
        self.adjuster.apply_initial_pose(DEFAULT_START_INDEX)
        self.trusted_pose = self.adjuster.get_current_pose().to(self.device)
        self.specimen_center_pose = self.adjuster.center_pose.to(self.device)
        self.normalizer = PerImageLegacyTransform(size=256, radius=119).to(self.device)
        self.contrast_distribution = torch.distributions.Uniform(0.5, 8.0)
        self.render_chunk_size = render_chunk_size
        self.projection_mode = projection_mode

    @staticmethod
    def _build_original_adjuster(projection_mode: str, device: str | None):
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        specimen = IntubationDatasetMR(DEFAULT_CT_NII, DEFAULT_XRAY_ROOT, **SXH_DRR_PARAMS)
        centerline_path = DEFAULT_CENTERLINE_NII if DEFAULT_CENTERLINE_NII.is_file() else DEFAULT_ORIGINAL_CENTERLINE_MASK
        vertices, edges = load_raw_centerline_graph_in_drr_coordinates(
            centerline_path, tuple(specimen.volume.shape), specimen.spacing
        )
        height = 256
        delx = specimen.delx * (512 / height)
        drr = DRR(specimen.volume, specimen.spacing, sdr=specimen.sdr, height=height, delx=delx, reverse_x_axis=True).to(
            resolved_device
        )
        drr_bone = DRRSeg(
            specimen.mr_mask,
            specimen.spacing,
            sdr=specimen.sdr,
            height=height,
            delx=delx,
            reverse_x_axis=True,
        ).to(resolved_device)
        mrcp_specimen = IntubationDatasetMRCP(DEFAULT_MRCP_NII, DEFAULT_XRAY_ROOT, **SXH_DRR_PARAMS)
        drr_mrcp = DRRMRCP(
            mrcp_specimen.volume,
            mrcp_specimen.spacing,
            sdr=mrcp_specimen.sdr,
            height=height,
            delx=delx,
            reverse_x_axis=True,
            projection_mode=projection_mode,
        ).to(resolved_device)
        adjuster = SXHCenterlineWebPoseAdjuster(
            drr,
            drr_bone,
            specimen,
            Transforms(height),
            resolved_device,
            centerline_vertices=vertices,
            centerline_edges=edges,
            guidewire_dir=DEFAULT_GUIDEWIRE_DIR,
            init_pose_mode="registered",
            runs_mask_dir=DEFAULT_RUNS_MASK_DIR,
            output_dir=DEFAULT_CENTERLINE_OUTPUT_DIR,
            projection_mode="centerline",
        )
        adjuster.drr_mrcp = drr_mrcp
        trusted_guidewire = load_guidewire_chain_from_overlay(DEFAULT_GUIDEWIRE_OVERLAY).astype(np.float32)

        def guidewire_points_for_index(index: int) -> np.ndarray:
            if index != DEFAULT_START_INDEX:
                return np.empty((0, 2), dtype=np.float32)
            return trusted_guidewire

        adjuster.guidewire_points_for_index = guidewire_points_for_index
        return adjuster

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def metadata(self) -> dict[str, object]:
        return {
            "registration": "automatic_ct_xray031_with_guidewire_refined_ct_mrcp",
            "mrcp_registration": self.mrcp_registration,
            "center_xray_index": DEFAULT_START_INDEX,
            "mrcp_projection_mode": self.projection_mode,
            "ct_nii": str(DEFAULT_CT_NII),
            "refined_mrcp_nii": str(DEFAULT_REFINED_MRCP),
            "refined_bile_duct_nii": str(DEFAULT_REFINED_BILE_DUCT),
            "refined_ct_to_mr": str(DEFAULT_REFINED_CT_TO_MR),
            "trusted_pose_euler_zyx_radians": self.trusted_pose.get_rotation("euler_angles", "ZYX")
            .detach()
            .cpu()
            .tolist(),
            "trusted_pose_translation_mm": self.trusted_pose.get_translation().detach().cpu().tolist(),
            "offset_rotation_std_radians": [float(torch.pi / 8), float(torch.pi / 10), float(torch.pi / 12)],
            "offset_translation_std_mm": [30.0, 50.0, 30.0],
            "ct_bone_attenuation_multiplier_uniform": [0.5, 8.0],
            "render_chunk_size": self.render_chunk_size,
            "normalization": "resize(256), per-image min-max, inversion, FOV radius=119, Normalize(0.3080, 0.1494)",
        }

    @staticmethod
    def _slice_pose(pose: RigidTransform, start: int, end: int) -> RigidTransform:
        return RigidTransform(
            pose.get_rotation("euler_angles", "ZYX")[start:end],
            pose.get_translation()[start:end],
            "euler_angles",
            "ZYX",
        )

    def _render_in_pose_chunks(self, renderer, poses: RigidTransform, batch_size: int) -> Tensor:
        chunks = []
        for start in range(0, batch_size, self.render_chunk_size):
            end = min(batch_size, start + self.render_chunk_size)
            chunks.append(renderer(None, None, None, pose=self._slice_pose(poses, start, end)))
        return torch.cat(chunks, dim=0)

    def render_poses(
        self,
        poses: RigidTransform,
        contrast_multiplier: float,
        include_bile_projection: bool = True,
        offsets: RigidTransform | None = None,
    ) -> SXHFeatureTrainingBatch:
        """Render a supplied batch of poses without changing their geometry."""
        batch_size = len(poses)
        if batch_size <= 0:
            raise ValueError("poses must contain at least one transform")
        if offsets is None:
            zeros = torch.zeros((batch_size, 3), dtype=torch.float32, device=self.device)
            offsets = RigidTransform(zeros, zeros.clone(), "euler_angles", "ZYX")

        with torch.no_grad():
            self._synchronize()
            start = time.perf_counter()
            self.adjuster.drr.set_bone_attenuation_multiplier(contrast_multiplier)
            ct_raw = self._render_in_pose_chunks(self.adjuster.drr, poses, batch_size)
            self._synchronize()
            ct_render_seconds = time.perf_counter() - start

            start = time.perf_counter()
            mrcp_raw = self._render_in_pose_chunks(self.adjuster.drr_mrcp, poses, batch_size)
            self._synchronize()
            mrcp_render_seconds = time.perf_counter() - start

            bile_projection = None
            if include_bile_projection:
                bile_projection = self.normalizer.resize(
                    self._render_in_pose_chunks(self.adjuster.drr_bone, poses, batch_size)
                )

            ct_drr = self.normalizer(ct_raw).to(torch.float32)
            mrcp_projection = self.normalizer(mrcp_raw).to(torch.float32)

        if not torch.isfinite(ct_drr).all() or not torch.isfinite(mrcp_projection).all():
            raise RuntimeError("Non-finite normalized CT-DRR or MRCP training image")
        valid_mask = self.normalizer.fov_mask.to(device=self.device, dtype=torch.bool).expand(batch_size, -1, -1, -1)
        return SXHFeatureTrainingBatch(
            ct_drr=ct_drr,
            mrcp_projection=mrcp_projection,
            valid_mask=valid_mask,
            poses=poses,
            offsets=offsets,
            contrast_multiplier=contrast_multiplier,
            ct_render_seconds=ct_render_seconds,
            mrcp_render_seconds=mrcp_render_seconds,
            bile_projection=bile_projection,
        )

    def render_batch(self, batch_size: int, include_bile_projection: bool = True) -> SXHFeatureTrainingBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        offsets = get_random_offset(batch_size, self.device)
        poses = compose_centered_perturbations(self.trusted_pose, self.specimen_center_pose, offsets)
        contrast_multiplier = self.contrast_distribution.sample().item()
        return self.render_poses(
            poses,
            contrast_multiplier=contrast_multiplier,
            include_bile_projection=include_bile_projection,
            offsets=offsets,
        )
