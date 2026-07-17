"""Refine the SXH CT-MRCP rigid transform from the trusted X-ray 031 guidewire result."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel.processing import resample_from_to

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CASE_ROOT = Path(__file__).resolve().parent
DIFFPOSE_ROOT = PROJECT_ROOT / "diffpose"
if str(DIFFPOSE_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFPOSE_ROOT))

from diffpose.calibration import RigidTransform

DEFAULT_CT_NII = PROJECT_ROOT / "mrct" / "data" / "孙新华" / "CT" / "3.nii"
DEFAULT_MRCP_NII = PROJECT_ROOT / "mrct" / "data" / "孙新华" / "MRCP" / "501.nii"
DEFAULT_BILE_DUCT_NII = PROJECT_ROOT / "mrct" / "data-duet" / "bile_duct" / "mrcp_006.nii.gz"
DEFAULT_CT_TO_MR = PROJECT_ROOT / "mrct" / "outputs" / "sxh_ct3_mrcp006_gallbladder_icp" / "ct_to_mr_icp_transform.txt"
DEFAULT_LEGACY_BILE_DUCT = (
    PROJECT_ROOT / "mrct" / "outputs" / "sxh_ct3_mrcp006_gallbladder_icp" / "mr_bile_duct_registered_to_ct.nii.gz"
)
DEFAULT_CT_XRAY_POSE = PROJECT_ROOT / "diffpose" / "ours" / "case" / "sxh" / "runs" / "mask" / "sxh_xray031_se3_log_map.csv"
DEFAULT_GUIDEWIRE_RESULT = CASE_ROOT / "outputs" / "guidewire_registration" / "xray031" / "result.json"
DEFAULT_OUTPUT_DIR = CASE_ROOT / "runs" / "refined_ct_mrcp_guidewire_xray031"
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

DRR_FACTORS = np.array([0.6, 0.6, 1.5], dtype=np.float64)


@dataclass(frozen=True)
class Refinement:
    drr_delta: np.ndarray
    ct_world_delta: np.ndarray
    ct_to_mr_original: np.ndarray
    mr_to_ct_original: np.ndarray
    ct_to_mr_refined: np.ndarray
    mr_to_ct_refined: np.ndarray
    closure_error: float


def column_pose_matrix(euler_zyx: list[float] | np.ndarray, translation: list[float] | np.ndarray) -> np.ndarray:
    """Return the conventional column-vector matrix for a DiffPose pose."""
    pose = RigidTransform(
        torch.as_tensor([euler_zyx], dtype=torch.float64),
        torch.as_tensor([translation], dtype=torch.float64),
        parameterization="euler_angles",
        convention="ZYX",
        dtype=torch.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = pose.get_rotation().detach().cpu().numpy()[0]
    matrix[:3, 3] = pose.get_translation().detach().cpu().numpy()[0]
    return matrix


def ct_voxels_to_drr_matrix(
    ct_nii: nib.Nifti1Image,
    factors: np.ndarray = DRR_FACTORS,
    drr_spacing: np.ndarray | None = None,
) -> np.ndarray:
    """Map native CT voxels to the actual SXH processed-DRR physical coordinates.

    ``IntubationDatasetMR`` resamples the volume before DRR rendering. Its
    resulting spacing must be used here; the native NIfTI zooms are only a
    fallback for standalone matrix tests.
    """
    factors = np.asarray(factors, dtype=np.float64)
    if factors.shape != (3,):
        raise ValueError(f"Expected three DRR factors, got {factors}")

    spacing = (
        np.asarray(ct_nii.header.get_zooms()[:3], dtype=np.float64)
        if drr_spacing is None
        else np.asarray(drr_spacing, dtype=np.float64)
    )
    if spacing.shape != (3,):
        raise ValueError(f"Expected three DRR spacing values, got {spacing}")
    shape = np.asarray(ct_nii.shape[:3], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = spacing[0] * factors[0]
    matrix[1, 1] = spacing[1] * factors[1]
    matrix[2, 2] = -spacing[2] * factors[2]
    matrix[2, 3] = (shape[2] - 1.0) * spacing[2] * factors[2]
    return matrix


def derive_refinement(
    ct_to_xray: np.ndarray,
    mrcp_to_xray: np.ndarray,
    ct_world_from_voxel: np.ndarray,
    drr_from_ct_voxel: np.ndarray,
    ct_to_mr_original: np.ndarray,
) -> Refinement:
    """Convert the trusted detector-pose correction into an updated CT-MRCP transform.

    DiffDRR poses move the source and detector around a fixed volume. To keep
    the optimized MRCP projection while substituting the CT-X-ray detector
    pose, the MRCP volume must receive ``CT_pose @ inverse(MRCP_pose)`` in
    DRR coordinates. The opposite order moves the detector correction in the
    wrong direction and produces a large 2D overlay error.
    """
    drr_delta = ct_to_xray @ np.linalg.inv(mrcp_to_xray)
    closure_error = float(np.abs(drr_delta @ mrcp_to_xray - ct_to_xray).max())

    ct_world_delta = (
        ct_world_from_voxel
        @ np.linalg.inv(drr_from_ct_voxel)
        @ drr_delta
        @ drr_from_ct_voxel
        @ np.linalg.inv(ct_world_from_voxel)
    )
    mr_to_ct_original = np.linalg.inv(ct_to_mr_original)
    mr_to_ct_refined = ct_world_delta @ mr_to_ct_original
    ct_to_mr_refined = np.linalg.inv(mr_to_ct_refined)
    return Refinement(
        drr_delta=drr_delta,
        ct_world_delta=ct_world_delta,
        ct_to_mr_original=ct_to_mr_original,
        mr_to_ct_original=mr_to_ct_original,
        ct_to_mr_refined=ct_to_mr_refined,
        mr_to_ct_refined=mr_to_ct_refined,
        closure_error=closure_error,
    )


def load_ct_xray_pose(path: Path) -> np.ndarray:
    row = pd.read_csv(path).iloc[-1]
    return column_pose_matrix(
        [float(row.alpha), float(row.beta), float(row.gamma)],
        [float(row.bx), float(row.by), float(row.bz)],
    )


def load_guidewire_pose(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    final_pose = payload["final_pose"]
    return column_pose_matrix(final_pose["euler_zyx"], final_pose["translation"])


def resample_mr_to_ct_grid(source_path: Path, ct_nii: nib.Nifti1Image, mr_to_ct: np.ndarray, order: int) -> nib.Nifti1Image:
    """Place a raw MRCP image in CT physical space and resample it onto the CT grid."""
    source = nib.load(str(source_path))
    data = np.asanyarray(source.dataobj)
    moved = nib.Nifti1Image(data, mr_to_ct @ source.affine, source.header)
    resampled = resample_from_to(moved, ct_nii, order=order)

    if order == 0:
        output_data = np.rint(np.asanyarray(resampled.dataobj)).astype(source.get_data_dtype())
    else:
        output_data = np.asanyarray(resampled.dataobj).astype(np.float32)
    return nib.Nifti1Image(output_data, ct_nii.affine, ct_nii.header)


def dice_score(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first) > 0
    second = np.asarray(second) > 0
    denominator = int(first.sum() + second.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(first, second).sum() / denominator)


def rotation_degrees(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(matrix[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, fmt="%.10f")


def render_xray031_overlay(refined_bile_duct_path: Path, output_dir: Path) -> dict[str, str]:
    """Render the refined bile-duct mask at the fixed CT-X-ray pose for visual review."""
    import cv2

    from ours.case.sxh.CT_dataset_MR import IntubationDatasetMR, apply_drr_axis_pipeline
    from ours.case.sxh.CT_dataset_nii import Transforms
    from ours.utils.drr_seg import DRRSeg

    frame_index = 31
    drr_params = dict(x_offset=20, y_offset=200, z_offset=100, z_cut=250, factors=DRR_FACTORS.tolist())
    specimen = IntubationDatasetMR(DEFAULT_CT_NII, DEFAULT_XRAY_ROOT, **drr_params)
    refined_native = np.asanyarray(nib.load(str(refined_bile_duct_path)).dataobj)
    refined_drr_grid = apply_drr_axis_pipeline(
        refined_native,
        DRR_FACTORS,
        z_cut=drr_params["z_cut"],
        order=0,
    )
    if refined_drr_grid.shape != specimen.volume.shape:
        raise ValueError(f"Refined mask shape {refined_drr_grid.shape} does not match CT DRR shape {specimen.volume.shape}")

    height = 256
    delx = specimen.delx * (512 / height)
    renderer = DRRSeg(
        refined_drr_grid,
        specimen.spacing,
        sdr=specimen.sdr,
        height=height,
        delx=delx,
        reverse_x_axis=True,
    )
    pose = load_ct_xray_pose(DEFAULT_CT_XRAY_POSE)
    pose_for_renderer = RigidTransform(
        torch.tensor(pose[:3, :3], dtype=torch.float32).unsqueeze(0),
        torch.tensor(pose[:3, 3], dtype=torch.float32).unsqueeze(0),
        parameterization="matrix",
    )
    transforms = Transforms(height)
    xray, _ = specimen[frame_index]
    with torch.no_grad():
        background = transforms(xray, reverse=False).squeeze().cpu().numpy()
        projected = transforms.resize(renderer(None, None, None, pose=pose_for_renderer)).squeeze().cpu().numpy()
    mask = projected > 0.01
    gray = np.clip(background * 255.0, 0, 255).astype(np.uint8)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[mask] = (0, 0, 255)

    xray_path = output_dir / "sxh_xray031_xray.png"
    overlay_path = output_dir / "sxh_xray031_refined_bile_overlay.png"
    cv2.imwrite(str(xray_path), gray)
    cv2.imwrite(str(overlay_path), overlay)
    return {"xray": str(xray_path), "refined_bile_overlay": str(overlay_path)}


def run(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ct_nii = nib.load(str(DEFAULT_CT_NII))
    from ours.case.sxh.CT_dataset_MR import IntubationDatasetMR

    specimen = IntubationDatasetMR(
        DEFAULT_CT_NII,
        DEFAULT_XRAY_ROOT,
        x_offset=20,
        y_offset=200,
        z_offset=100,
        z_cut=250,
        factors=DRR_FACTORS.tolist(),
    )
    refinement = derive_refinement(
        load_ct_xray_pose(DEFAULT_CT_XRAY_POSE),
        load_guidewire_pose(DEFAULT_GUIDEWIRE_RESULT),
        ct_nii.affine,
        ct_voxels_to_drr_matrix(ct_nii, drr_spacing=specimen.spacing),
        np.loadtxt(DEFAULT_CT_TO_MR, dtype=np.float64),
    )
    if refinement.closure_error > 1e-8:
        raise RuntimeError(f"Pose-chain closure failed: {refinement.closure_error}")

    refined_mrcp = resample_mr_to_ct_grid(DEFAULT_MRCP_NII, ct_nii, refinement.mr_to_ct_refined, order=1)
    refined_bile_duct = resample_mr_to_ct_grid(DEFAULT_BILE_DUCT_NII, ct_nii, refinement.mr_to_ct_refined, order=0)
    nib.save(refined_mrcp, str(output_dir / "mrcp_501_registered_to_ct_refined.nii.gz"))
    nib.save(refined_bile_duct, str(output_dir / "mr_bile_duct_registered_to_ct_refined.nii.gz"))
    visualizations = render_xray031_overlay(output_dir / "mr_bile_duct_registered_to_ct_refined.nii.gz", output_dir)

    original_bile_duct = resample_mr_to_ct_grid(
        DEFAULT_BILE_DUCT_NII,
        ct_nii,
        refinement.mr_to_ct_original,
        order=0,
    )
    legacy_bile_duct = nib.load(str(DEFAULT_LEGACY_BILE_DUCT))
    baseline_dice = dice_score(np.asanyarray(original_bile_duct.dataobj), np.asanyarray(legacy_bile_duct.dataobj))

    write_matrix(output_dir / "ct_to_mr_refined.txt", refinement.ct_to_mr_refined)
    write_matrix(output_dir / "mr_to_ct_refined.txt", refinement.mr_to_ct_refined)
    write_matrix(output_dir / "ct_world_delta_from_guidewire.txt", refinement.ct_world_delta)
    write_matrix(output_dir / "drr_delta_from_guidewire.txt", refinement.drr_delta)

    report = {
        "source": {
            "ct_to_mr_icp": str(DEFAULT_CT_TO_MR),
            "ct_xray_pose": str(DEFAULT_CT_XRAY_POSE),
            "trusted_guidewire_result": str(DEFAULT_GUIDEWIRE_RESULT),
            "mrcp": str(DEFAULT_MRCP_NII),
            "bile_duct": str(DEFAULT_BILE_DUCT_NII),
        },
        "validation": {
            "drr_pose_chain_max_abs_error": refinement.closure_error,
            "baseline_bile_duct_dice_against_legacy_resampling": baseline_dice,
            "visualizations": visualizations,
        },
        "guidewire_refinement": {
            "rotation_degrees": rotation_degrees(refinement.ct_world_delta),
            "translation_mm": refinement.ct_world_delta[:3, 3].tolist(),
            "drr_delta": refinement.drr_delta.tolist(),
            "ct_world_delta": refinement.ct_world_delta.tolist(),
        },
        "transforms": {
            "ct_to_mr_original": refinement.ct_to_mr_original.tolist(),
            "mr_to_ct_original": refinement.mr_to_ct_original.tolist(),
            "ct_to_mr_refined": refinement.ct_to_mr_refined.tolist(),
            "mr_to_ct_refined": refinement.mr_to_ct_refined.tolist(),
        },
    }
    report_path = output_dir / "refinement_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output_dir)
    print(json.dumps(report["validation"], indent=2))
    print(f"refined output: {args.output_dir}")


if __name__ == "__main__":
    main()
