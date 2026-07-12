"""SXH WebDRR server with an MRCP centerline-skeleton projection overlay.

Run from ``diffpose/ours``:
    conda run -n mr2ct python xmr/case/sxh/web_drr_server_nii_sxh_centerline.py

The centerline stays as a raw-MRCP graph. Its vertices undergo only fixed
coordinate transforms before a differentiable detector-plane projection.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from aiohttp import web

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SXH_CASE_ROOT = Path(__file__).resolve().parent
DIFFPOSE_ROOT = PROJECT_ROOT / "diffpose"
OURS_ROOT = DIFFPOSE_ROOT / "ours"
for path in (DIFFPOSE_ROOT, OURS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ours.case.sxh.CT_dataset_MR import (  # noqa: E402
    DEFAULT_ICP_TRANSFORM,
    load_ct_to_mr_transform,
)
from ours.case.sxh.CT_dataset_MR import IntubationDatasetMR  # noqa: E402
from ours.case.sxh.CT_dataset_nii import Transforms  # noqa: E402
from ours.utils.drr import DRR  # noqa: E402
from ours.utils.drr_seg import DRRSeg  # noqa: E402
from ours.web_drr_server_nii import generate_overlay_image_cv2, numpy_to_base64_cv2  # noqa: E402
from xmr.case.sxh.image_io import write_overlay_png  # noqa: E402
from xmr.case.sxh.pose_io import DEFAULT_RUNS_MASK_DIR, default_registered_index, discover_registered_indices  # noqa: E402
from xmr.case.sxh.web_drr_server_nii_sxh import (  # noqa: E402
    DEFAULT_CT_NII,
    DEFAULT_XRAY_ROOT,
    SXHWebPoseAdjuster,
)
from xmr.case.sxh.centerline.centerline import extract_centerline_tree  # noqa: E402


DEFAULT_CENTERLINE_NII = SXH_CASE_ROOT / "centerline" / "outputs" / "sxh_mrcp_006_centerline_tree.nii.gz"
DEFAULT_OUTPUT_DIR = SXH_CASE_ROOT / "outputs" / "web_centerline"
SXH_DRR_PARAMS = dict(x_offset=20, y_offset=200, z_offset=100, z_cut=250, factors=[0.6, 0.6, 1.5])


def project_points_to_detector(points: torch.Tensor, pose, detector) -> torch.Tensor:
    """Project volume-space 3D points to continuous ``(column, row)`` pixels.

    The detector plane is derived from the same DiffDRR source/target geometry
    used for CT DRRs. All operations after the fixed input points remain
    differentiable with respect to ``pose``.
    """
    source = pose.transform_points(detector.source)[0, 0]
    plane = pose.transform_points(detector.target[:, [0, 1, detector.width], :])[0]
    origin, next_column, next_row = plane
    column_axis = next_column - origin
    row_axis = next_row - origin
    column_step = torch.linalg.vector_norm(column_axis)
    row_step = torch.linalg.vector_norm(row_axis)
    column_axis = column_axis / column_step
    row_axis = row_axis / row_step
    plane_normal = torch.linalg.cross(column_axis, row_axis)

    rays = points - source
    denominator = torch.einsum("nd,d->n", rays, plane_normal)
    numerator = torch.dot(origin - source, plane_normal)
    epsilon = torch.finfo(points.dtype).eps
    safe_denominator = torch.where(
        denominator.abs() < epsilon,
        torch.where(denominator >= 0, torch.full_like(denominator, epsilon), torch.full_like(denominator, -epsilon)),
        denominator,
    )
    ray_scale = numerator / safe_denominator
    intersections = source + ray_scale[:, None] * rays
    offsets = intersections - origin
    columns = torch.einsum("nd,d->n", offsets, column_axis) / column_step
    rows = torch.einsum("nd,d->n", offsets, row_axis) / row_step
    return torch.stack((columns, rows), dim=1)


def _sample_projected_edges(projected_vertices: torch.Tensor, edges: torch.Tensor, step_px: float = 0.75) -> torch.Tensor:
    """Densely sample each projected graph edge while preserving endpoint gradients."""
    samples: list[torch.Tensor] = []
    for start_index, end_index in edges.tolist():
        start = projected_vertices[start_index]
        end = projected_vertices[end_index]
        length = torch.linalg.vector_norm(end - start).detach()
        count = max(2, int(torch.ceil(length / step_px).item()) + 1)
        alpha = torch.linspace(0.0, 1.0, count, dtype=start.dtype, device=start.device)[:, None]
        samples.append(start[None] * (1.0 - alpha) + end[None] * alpha)
    return torch.cat(samples, dim=0)


def render_soft_centerline(
    vertices: torch.Tensor,
    edges: torch.Tensor,
    pose,
    detector,
    *,
    sigma_px: float = 1.0,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Render a continuous differentiable 2D centerline image from a 3D graph."""
    projected_vertices = project_points_to_detector(vertices, pose, detector)
    margin = float(max(detector.height, detector.width))
    visible_vertices = torch.isfinite(projected_vertices).all(dim=1)
    visible_vertices &= (projected_vertices[:, 0].abs() <= detector.width + margin)
    visible_vertices &= (projected_vertices[:, 1].abs() <= detector.height + margin)
    visible_edges = visible_vertices[edges[:, 0]] & visible_vertices[edges[:, 1]]
    render_edges = edges[visible_edges]
    if len(render_edges) == 0:
        return vertices.sum().reshape(1, 1, 1, 1) * 0.0 + torch.zeros(
            (1, 1, detector.height, detector.width), dtype=vertices.dtype, device=vertices.device
        )
    samples = _sample_projected_edges(projected_vertices, render_edges)
    rows = torch.arange(detector.height, dtype=vertices.dtype, device=vertices.device)
    columns = torch.arange(detector.width, dtype=vertices.dtype, device=vertices.device)
    grid_row, grid_column = torch.meshgrid(rows, columns, indexing="ij")
    log_not_covered = torch.zeros((detector.height, detector.width), dtype=vertices.dtype, device=vertices.device)
    for sample_chunk in samples.split(chunk_size):
        distance_squared = (grid_column[None] - sample_chunk[:, 0, None, None]).square()
        distance_squared += (grid_row[None] - sample_chunk[:, 1, None, None]).square()
        coverage = torch.exp(-distance_squared / (2.0 * sigma_px**2)).clamp(max=1.0 - 1e-6)
        log_not_covered += torch.log1p(-coverage).sum(dim=0)
    return (1.0 - torch.exp(log_not_covered)).unsqueeze(0).unsqueeze(0)


def build_centerline_http_response(html: str) -> str:
    """Reuse the established four-panel UI, replacing MRCP intensity with centerline overlay."""
    return (
        html.replace("<title>SXH DRR + MRCP</title>", "<title>SXH DRR + Centerline</title>")
        .replace("MRCP Projection", "MRCP Centerline Overlay")
        .replace("SXH DRR + MRCP", "SXH DRR + Centerline")
    )


def ct_voxels_to_drr_mm(
    ct_voxels: np.ndarray,
    *,
    ct_shape: tuple[int, int, int],
    drr_spacing: np.ndarray,
    factors: np.ndarray,
) -> np.ndarray:
    """Map native CT voxels to Siddon's physical-volume coordinates.

    SXH flips axis 2 before zooming. Axis 0 must not be flipped here because
    ``DRR`` and ``DRRSeg`` flip their input volume internally before raycasting.
    """
    drr_voxels = np.asarray(ct_voxels, dtype=np.float64).copy()
    drr_voxels[:, 2] = ct_shape[2] - 1 - drr_voxels[:, 2]
    drr_voxels *= np.asarray(factors, dtype=np.float64)
    return drr_voxels * np.asarray(drr_spacing, dtype=np.float64)


def load_raw_centerline_graph_in_drr_coordinates(
    centerline_path: Path,
    drr_shape: tuple[int, int, int],
    drr_spacing: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raw-MRCP skeleton vertices to the legacy viewer's volume coordinates.

    This applies only fixed affine/rigid coordinate transforms. It never
    resamples the skeleton volume, so all raw graph vertices and edges survive.
    """
    centerline_nii = nib.load(str(centerline_path))
    tree = extract_centerline_tree(np.asanyarray(centerline_nii.dataobj) > 0, centerline_nii.affine)
    ct_nii = nib.load(str(DEFAULT_CT_NII))
    mr_to_ct = np.linalg.inv(load_ct_to_mr_transform(DEFAULT_ICP_TRANSFORM))
    vertices_h = np.c_[tree.vertices_mm, np.ones(len(tree.vertices_mm))]
    ct_world = (vertices_h @ mr_to_ct.T)[:, :3]
    ct_voxels = (np.c_[ct_world, np.ones(len(ct_world))] @ np.linalg.inv(ct_nii.affine).T)[:, :3]

    drr_mm = ct_voxels_to_drr_mm(
        ct_voxels,
        ct_shape=ct_nii.shape,
        drr_spacing=drr_spacing,
        factors=np.asarray(SXH_DRR_PARAMS["factors"], dtype=np.float64),
    )
    return torch.tensor(drr_mm, dtype=torch.float32), torch.tensor(tree.edges, dtype=torch.long)


class SXHCenterlineWebPoseAdjuster(SXHWebPoseAdjuster):
    """Extend the established SXH viewer with a centerline-projection panel."""

    def __init__(self, *args, centerline_vertices: torch.Tensor, centerline_edges: torch.Tensor, **kwargs):
        self.centerline_vertices = centerline_vertices
        self.centerline_edges = centerline_edges
        super().__init__(*args, drr_mrcp=None, **kwargs)

    def render_frame_arrays(self) -> dict[str, np.ndarray]:
        arrays = super().render_frame_arrays()
        centerline = render_soft_centerline(
            self.centerline_vertices.to(self.device),
            self.centerline_edges.to(self.device),
            self.get_current_pose(),
            self.drr.detector,
        )
        arrays["centerline"] = centerline.detach().cpu().squeeze().numpy()
        return arrays

    def generate_drr_and_overlay_images(self):
        try:
            arrays = self.render_frame_arrays()
            return (
                numpy_to_base64_cv2(arrays["ct_drr"]),
                generate_overlay_image_cv2(arrays["xray"], arrays["bile_mask"]),
                generate_overlay_image_cv2(arrays["xray"], arrays["centerline"], alpha=0.8),
            )
        except Exception as exc:
            print(f"generate centerline overlay error: {exc}")
            return "", "", ""

    def save_frame_pngs(self, tag: str, arrays: dict[str, np.ndarray]) -> dict[str, str]:
        paths = super().save_frame_pngs(tag, arrays)
        centerline_path = self.output_dir / tag / f"sxh_xray{self.current_index:03d}_centerline_overlay.png"
        write_overlay_png(centerline_path, arrays["xray"], arrays["centerline"])
        paths["centerline_overlay"] = str(centerline_path)
        return paths

    async def send_updates(self, websocket):
        self.running = True
        try:
            while self.running and not websocket.closed:
                start_time = time.time()
                if self.current_keys:
                    self.update_pose_continuous()
                drr_image, bile_overlay, centerline_overlay = self.generate_drr_and_overlay_images()
                if drr_image:
                    await websocket.send_json(
                        {
                            "type": "image_update",
                            "drr_image": drr_image,
                            "overlay_image": bile_overlay,
                            "mrcp_image": centerline_overlay,
                            "pose": self.pose_params.copy(),
                            "index": self.current_index,
                            "timestamp": time.time(),
                        }
                    )
                await asyncio.sleep(max(0.0, self.update_rate - (time.time() - start_time)))
        except Exception as exc:
            print(f"centerline streaming error: {exc}")
        finally:
            self.running = False

    async def http_handler(self, request):
        response = await super().http_handler(request)
        return web.Response(text=build_centerline_http_response(response.text), content_type="text/html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SXH DRR web server with MRCP centerline-point overlay")
    parser.add_argument("--init-pose", choices=["zero", "registered"], default="registered")
    parser.add_argument("--runs-mask-dir", type=Path, default=DEFAULT_RUNS_MASK_DIR)
    parser.add_argument("--centerline", type=Path, default=DEFAULT_CENTERLINE_NII)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-index", type=int, default=None)
    return parser.parse_args()


async def run_server(args: argparse.Namespace) -> None:
    if not args.centerline.is_file():
        raise FileNotFoundError(f"Centerline NIfTI not found: {args.centerline}")
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    start_index = args.start_index
    if start_index is None:
        start_index = default_registered_index(args.runs_mask_dir) if args.init_pose == "registered" else 0
        start_index = 0 if start_index is None else start_index

    specimen = IntubationDatasetMR(DEFAULT_CT_NII, DEFAULT_XRAY_ROOT, **SXH_DRR_PARAMS)
    centerline_vertices, centerline_edges = load_raw_centerline_graph_in_drr_coordinates(
        args.centerline,
        tuple(specimen.volume.shape),
        specimen.spacing,
    )

    height = 256
    delx = specimen.delx * (512 / height)
    drr = DRR(specimen.volume, specimen.spacing, sdr=specimen.sdr, height=height, delx=delx, reverse_x_axis=True, bone_attenuation_multiplier=3).to(device)
    drr_bile_duct = DRRSeg(specimen.mr_mask, specimen.spacing, sdr=specimen.sdr, height=height, delx=delx, reverse_x_axis=True).to(device)

    adjuster = SXHCenterlineWebPoseAdjuster(
        drr,
        drr_bile_duct,
        specimen,
        Transforms(height),
        device,
        centerline_vertices=centerline_vertices,
        centerline_edges=centerline_edges,
        init_pose_mode=args.init_pose,
        runs_mask_dir=args.runs_mask_dir,
        output_dir=args.output_dir,
        projection_mode="centerline",
        ws_port=8766,
        http_port=8081,
    )
    if not 0 <= start_index <= adjuster.max_index:
        raise ValueError(f"start_index {start_index} out of range [0, {adjuster.max_index}]")
    adjuster.current_index = start_index
    adjuster.apply_initial_pose(start_index)
    adjuster.reference_img_base64 = adjuster.generate_reference_image()

    http_app, ws_app = web.Application(), web.Application()
    http_app.router.add_get("/", adjuster.http_handler)
    ws_app.router.add_get("/ws", adjuster.websocket_handler)
    http_runner, ws_runner = web.AppRunner(http_app), web.AppRunner(ws_app)
    await http_runner.setup()
    await ws_runner.setup()
    await web.TCPSite(http_runner, adjuster.host, adjuster.http_port).start()
    await web.TCPSite(ws_runner, adjuster.host, adjuster.ws_port).start()

    print(f"HTTP server: http://{adjuster.host}:{adjuster.http_port}")
    print(f"WebSocket server: ws://{adjuster.host}:{adjuster.ws_port}/ws")
    print(f"Centerline source: {args.centerline}")
    print(f"Raw centerline graph: vertices={len(centerline_vertices)}, edges={len(centerline_edges)}")
    print(f"Registered CSV indices: {discover_registered_indices(args.runs_mask_dir)}")
    await asyncio.Future()


async def main() -> None:
    await run_server(parse_args())


if __name__ == "__main__":
    asyncio.run(main())
