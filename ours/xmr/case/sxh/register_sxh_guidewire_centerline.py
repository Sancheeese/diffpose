"""Register one SXH X-ray to a fixed MRCP centerline chain using its guidewire."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ours.xmr.case.sxh.guidewire_registration import (
    PoseOptimizationResult,
    fixed_chain_score,
    optimize_fixed_chain_pose,
    select_fixed_chain,
)
from ours.xmr.case.sxh.web_drr_server_nii_sxh_centerline import (
    DEFAULT_CENTERLINE_NII,
    DEFAULT_GUIDEWIRE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUNS_MASK_DIR,
    DRR,
    DRRSeg,
    DEFAULT_CT_NII,
    DEFAULT_XRAY_ROOT,
    IntubationDatasetMR,
    SXHCenterlineWebPoseAdjuster,
    SXH_DRR_PARAMS,
    Transforms,
    load_raw_centerline_graph_in_drr_coordinates,
    project_points_to_detector,
    render_chain_points_overlay,
)

SXH_CASE_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRATION_OUTPUT_DIR = SXH_CASE_ROOT / "outputs" / "guidewire_registration"


def _pose_payload(pose) -> dict[str, list[float]]:
    return {
        "euler_zyx": pose.get_rotation("euler_angles", "ZYX").detach().cpu().squeeze().tolist(),
        "translation": pose.get_translation().detach().cpu().squeeze().tolist(),
        "se3_log": pose.get_se3_log().detach().cpu().squeeze().tolist(),
    }


def save_registration_result(
    output_dir: str | Path,
    result: PoseOptimizationResult,
    *,
    selected_chain_index: int,
    selection_score: float,
    initial_pose=None,
) -> Path:
    """Save reproducible fixed-chain selection, losses, and global poses."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_chain_index": selected_chain_index,
        "selection_score": selection_score,
        "losses": {"initial": result.history[0], "final": result.history[-1]},
        "final_pose": _pose_payload(result.pose),
    }
    if initial_pose is not None:
        payload["initial_pose"] = _pose_payload(initial_pose)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result_path


def run_registration(index: int, n_iters: int, output_dir: Path, device: str) -> dict[str, Path]:
    """Run fixed-chain guidewire registration and export before/after overlays."""
    specimen = IntubationDatasetMR(DEFAULT_CT_NII, DEFAULT_XRAY_ROOT, **SXH_DRR_PARAMS)
    if not 0 <= index < len(specimen):
        raise ValueError(f"index {index} out of range [0, {len(specimen) - 1}]")

    vertices, edges = load_raw_centerline_graph_in_drr_coordinates(
        DEFAULT_CENTERLINE_NII, tuple(specimen.volume.shape), specimen.spacing
    )
    height = 256
    delx = specimen.delx * (512 / height)
    drr = DRR(specimen.volume, specimen.spacing, sdr=specimen.sdr, height=height, delx=delx, reverse_x_axis=True).to(device)
    bile_drr = DRRSeg(specimen.mr_mask, specimen.spacing, sdr=specimen.sdr, height=height, delx=delx, reverse_x_axis=True).to(device)
    adjuster = SXHCenterlineWebPoseAdjuster(
        drr,
        bile_drr,
        specimen,
        Transforms(height),
        device,
        centerline_vertices=vertices,
        centerline_edges=edges,
        guidewire_dir=DEFAULT_GUIDEWIRE_DIR,
        init_pose_mode="registered",
        runs_mask_dir=DEFAULT_RUNS_MASK_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        projection_mode="centerline",
    )
    adjuster.current_index = index
    adjuster.apply_initial_pose(index)
    initial_pose = adjuster.get_current_pose()
    guidewire = torch.as_tensor(adjuster.guidewire_points_for_index(index), device=device)
    if len(guidewire) < 2:
        raise ValueError(f"No usable guidewire chain found for index {index}")

    chains = adjuster.centerline_chains
    with torch.no_grad():
        projected_chains = [project_points_to_detector(vertices[chain].to(device), initial_pose, drr.detector) for chain in chains]
    selected_chain_index = select_fixed_chain(guidewire, projected_chains)
    selected_vertex_indices = chains[selected_chain_index]
    selected_vertices = vertices[selected_vertex_indices].to(device)
    selection_score = float(fixed_chain_score(guidewire, projected_chains[selected_chain_index]))
    result = optimize_fixed_chain_pose(
        selected_vertices,
        guidewire,
        drr.detector,
        initial_pose,
        project_points_to_detector,
        n_iters=n_iters,
    )

    run_dir = output_dir / f"xray{index:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        initial_projection = project_points_to_detector(vertices.to(device), initial_pose, drr.detector).cpu().numpy()
        final_projection = project_points_to_detector(vertices.to(device), result.pose.to(device), drr.detector).cpu().numpy()
    xray, _ = specimen[index]
    background = adjuster.transforms(xray, reverse=False).squeeze().cpu().numpy()
    guidewire_np = guidewire.detach().cpu().numpy()
    initial_overlay = run_dir / "initial_overlay.png"
    final_overlay = run_dir / "final_overlay.png"
    cv2.imwrite(str(initial_overlay), render_chain_points_overlay(background, initial_projection, chains, guidewire_np))
    cv2.imwrite(str(final_overlay), render_chain_points_overlay(background, final_projection, chains, guidewire_np))
    np.savez_compressed(
        run_dir / "optimization.npz",
        selected_chain_vertex_indices=selected_vertex_indices,
        guidewire_points_2d=guidewire_np,
        initial_tree_points_2d=initial_projection,
        final_tree_points_2d=final_projection,
        loss_history=np.asarray([[row[key] for key in ("total", "position", "tangent", "pose")] for row in result.history]),
    )
    result_path = save_registration_result(
        run_dir,
        result,
        selected_chain_index=selected_chain_index,
        selection_score=selection_score,
        initial_pose=initial_pose,
    )
    return {"result": result_path, "initial_overlay": initial_overlay, "final_overlay": final_overlay}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=31)
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REGISTRATION_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_registration(args.index, args.n_iters, args.output_dir, args.device)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
