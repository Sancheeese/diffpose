"""SXH WebDRR server with an MRCP centerline-skeleton projection overlay.

Run from ``diffpose/ours``:
    conda run -n mr2ct python xmr/case/sxh/web_drr_server_nii_sxh_centerline.py

The centerline stays as a raw-MRCP graph. Its vertices undergo only fixed
coordinate transforms before a differentiable detector-plane projection.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
import time
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch
from aiohttp import web
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

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
from xmr.case.sxh.image_io import to_zero_one, write_overlay_png  # noqa: E402
from xmr.case.sxh.pose_io import DEFAULT_RUNS_MASK_DIR, default_registered_index, discover_registered_indices  # noqa: E402
from xmr.case.sxh.web_drr_server_nii_sxh import (  # noqa: E402
    DEFAULT_CT_NII,
    DEFAULT_XRAY_ROOT,
    SXHWebPoseAdjuster,
)
from xmr.case.sxh.centerline.centerline import extract_centerline_tree  # noqa: E402


DEFAULT_CENTERLINE_NII = SXH_CASE_ROOT / "centerline" / "outputs" / "sxh_mrcp_006_centerline_tree.nii.gz"
DEFAULT_GUIDEWIRE_DIR = SXH_CASE_ROOT / "guidewire"
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


def matching_guidewire_mask_path(xray_filename: str | Path, guidewire_dir: str | Path) -> Path | None:
    """Find the guidewire segmentation whose NIfTI stem matches an X-ray DICOM stem."""
    xray_stem = Path(xray_filename).stem
    path = Path(guidewire_dir) / f"{xray_stem}.nii.gz"
    return path if path.is_file() else None


def guidewire_indices_for_specimen(specimen, guidewire_dir: str | Path) -> list[int]:
    """Return dataset indices whose X-ray DICOM stems have a guidewire segmentation."""
    return [
        index
        for index in range(len(specimen))
        if matching_guidewire_mask_path(specimen.get_x_filename(index), guidewire_dir) is not None
    ]


def extract_guidewire_points(mask: np.ndarray) -> np.ndarray:
    """Return ``(x, y)`` points from the largest 2D guidewire skeleton component."""
    binary = np.asarray(mask) > 0
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2D guidewire mask, got shape {binary.shape}")
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.empty((0, 2), dtype=np.float32)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = labels == sizes.argmax()
    skeleton = skeletonize(largest)
    yx = np.argwhere(skeleton)
    return yx[:, ::-1].astype(np.float32)


def _skeleton_graph(points_xy: np.ndarray) -> sparse.csr_matrix:
    point_to_index = {tuple(point.astype(int)): index for index, point in enumerate(points_xy)}
    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for index, point in enumerate(points_xy.astype(int)):
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            neighbor_index = point_to_index.get((point[0] + dx, point[1] + dy))
            if neighbor_index is not None:
                distance = float(np.hypot(dx, dy))
                rows.extend((index, neighbor_index))
                columns.extend((neighbor_index, index))
                weights.extend((distance, distance))
    return sparse.csr_matrix((weights, (rows, columns)), shape=(len(points_xy), len(points_xy)))


def extract_ordered_guidewire_chain(mask: np.ndarray) -> np.ndarray:
    """Trace the longest endpoint-to-endpoint path of a guidewire skeleton."""
    points_xy = extract_guidewire_points(mask)
    if len(points_xy) < 2:
        return points_xy
    graph = _skeleton_graph(points_xy)
    degrees = np.diff(graph.indptr)
    endpoints = np.flatnonzero(degrees == 1)
    if len(endpoints) >= 2:
        endpoint_distances = dijkstra(graph, directed=False, indices=endpoints)
        endpoint_distances[~np.isfinite(endpoint_distances)] = -np.inf
        source_row, target_column = np.unravel_index(endpoint_distances.argmax(), endpoint_distances.shape)
        source, target = int(endpoints[source_row]), int(target_column)
    else:
        distances = dijkstra(graph, directed=False, indices=0)
        distances[~np.isfinite(distances)] = -np.inf
        source, target = 0, int(distances.argmax())
    _, predecessors = dijkstra(graph, directed=False, indices=source, return_predecessors=True)
    path = [target]
    while path[-1] != source:
        predecessor = int(predecessors[path[-1]])
        if predecessor < 0:
            raise RuntimeError("Guidewire skeleton path reconstruction failed")
        path.append(predecessor)
    return points_xy[np.asarray(path[::-1])]


def decompose_graph_into_chains(num_vertices: int, edges: torch.Tensor) -> list[np.ndarray]:
    """Split a centerline graph into maximal paths between endpoints and junctions."""
    adjacency = [set() for _ in range(num_vertices)]
    for start, end in edges.detach().cpu().numpy().astype(int):
        adjacency[start].add(end)
        adjacency[end].add(start)
    degrees = [len(neighbors) for neighbors in adjacency]
    anchors = {index for index, degree in enumerate(degrees) if degree != 2}
    junctions = {index for index, degree in enumerate(degrees) if degree > 2}
    junction_group: dict[int, int] = {}
    group_id = 0
    for start in sorted(junctions):
        if start in junction_group:
            continue
        stack = [start]
        junction_group[start] = group_id
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in junctions and neighbor not in junction_group:
                    junction_group[neighbor] = group_id
                    stack.append(neighbor)
        group_id += 1

    def same_junction_cluster(first: int, second: int) -> bool:
        return first in junction_group and junction_group[first] == junction_group.get(second)

    used_edges: set[tuple[int, int]] = set()
    chains: list[np.ndarray] = []

    for anchor in sorted(anchors):
        for neighbor in sorted(adjacency[anchor]):
            edge = tuple(sorted((anchor, neighbor)))
            if edge in used_edges or same_junction_cluster(anchor, neighbor):
                used_edges.add(edge)
                continue
            path = [anchor]
            previous, current = anchor, neighbor
            while True:
                path.append(current)
                used_edges.add(tuple(sorted((previous, current))))
                if current in anchors:
                    break
                next_nodes = sorted(node for node in adjacency[current] if node != previous)
                if not next_nodes:
                    break
                previous, current = current, next_nodes[0]
            if len(path) >= 2:
                chains.append(np.asarray(path, dtype=np.int64))

    for start in range(num_vertices):
        for neighbor in sorted(adjacency[start]):
            edge = tuple(sorted((start, neighbor)))
            if edge in used_edges:
                continue
            path = [start]
            previous, current = start, neighbor
            while True:
                path.append(current)
                used_edges.add(tuple(sorted((previous, current))))
                next_nodes = sorted(node for node in adjacency[current] if node != previous)
                if not next_nodes or current == start:
                    break
                previous, current = current, next_nodes[0]
            if len(path) >= 2:
                chains.append(np.asarray(path, dtype=np.int64))
    return chains


def reflect_points_across_y_equals_x(points: np.ndarray) -> np.ndarray:
    """Apply the guidewire NIfTI-to-X-ray coordinate reflection ``(x, y) -> (y, x)``."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N, 2), got {points.shape}")
    return points[:, [1, 0]].copy()


def subsample_chain_points(points: np.ndarray, min_spacing_px: float) -> np.ndarray:
    """Retain visibly separated points from an ordered 2D chain for display only."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N, 2), got {points.shape}")
    if min_spacing_px <= 0:
        raise ValueError(f"min_spacing_px must be positive, got {min_spacing_px}")
    if len(points) <= 2:
        return points.copy()

    retained = [points[0]]
    for point in points[1:-1]:
        if np.linalg.norm(point - retained[-1]) >= min_spacing_px:
            retained.append(point)
    retained.append(points[-1])
    return np.asarray(retained, dtype=np.float32)


def render_chain_points_overlay(
    background: np.ndarray,
    centerline_points: np.ndarray,
    centerline_chains: list[np.ndarray],
    guidewire_chain: np.ndarray,
) -> np.ndarray:
    """Draw ordered MRCP branch-chain points and guidewire-chain points over one X-ray."""
    background_u8 = (to_zero_one(background) * 255).astype(np.uint8)
    overlay = cv2.cvtColor(background_u8, cv2.COLOR_GRAY2BGR)
    height, width = overlay.shape[:2]
    projected = np.asarray(centerline_points, dtype=np.float32)
    for chain in centerline_chains:
        display_points = subsample_chain_points(projected[np.asarray(chain, dtype=np.int64)], min_spacing_px=4.0)
        for column, row in display_points:
            if 0 <= column < width and 0 <= row < height:
                cv2.circle(
                    overlay,
                    (round(float(column)), round(float(row))),
                    1,
                    (128, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )
    guidewire_display_points = subsample_chain_points(guidewire_chain, min_spacing_px=4.0)
    for column, row in guidewire_display_points:
        if 0 <= column < width and 0 <= row < height:
            cv2.circle(overlay, (round(float(column)), round(float(row))), 1, (0, 0, 255), -1, cv2.LINE_AA)
    return overlay


def bgr_to_base64(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    return base64.b64encode(encoded).decode("ascii") if success else ""


def build_centerline_http_response(html: str) -> str:
    """Reuse the established four-panel UI, replacing MRCP intensity with centerline overlay."""
    return (
        html.replace("<title>SXH DRR + MRCP</title>", "<title>SXH DRR + Centerline</title>")
        .replace("MRCP Projection", "MRCP Branch Chains / Guidewire Chain")
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

    def __init__(
        self,
        *args,
        centerline_vertices: torch.Tensor,
        centerline_edges: torch.Tensor,
        guidewire_dir: Path = DEFAULT_GUIDEWIRE_DIR,
        **kwargs,
    ):
        self.centerline_vertices = centerline_vertices
        self.centerline_edges = centerline_edges
        self.centerline_chains = decompose_graph_into_chains(len(centerline_vertices), centerline_edges)
        self.guidewire_dir = Path(guidewire_dir)
        self._guidewire_points_cache: dict[Path, np.ndarray] = {}
        super().__init__(*args, drr_mrcp=None, **kwargs)

    def guidewire_points_for_index(self, index: int) -> np.ndarray:
        guidewire_path = matching_guidewire_mask_path(self.specimen.get_x_filename(index), self.guidewire_dir)
        if guidewire_path is None:
            return np.empty((0, 2), dtype=np.float32)
        cached = self._guidewire_points_cache.get(guidewire_path)
        if cached is not None:
            return cached
        guidewire_nii = nib.load(str(guidewire_path))
        guidewire_mask = np.squeeze(np.asanyarray(guidewire_nii.dataobj)).astype(np.float32)
        resized = self.transforms.resize(torch.from_numpy(guidewire_mask)[None, None]).squeeze().numpy()
        points = reflect_points_across_y_equals_x(extract_ordered_guidewire_chain(resized > 0.5))
        self._guidewire_points_cache[guidewire_path] = points
        return points

    def render_frame_arrays(self) -> dict[str, np.ndarray]:
        arrays = super().render_frame_arrays()
        pose = self.get_current_pose()
        centerline_points = project_points_to_detector(self.centerline_vertices.to(self.device), pose, self.drr.detector)
        centerline = render_soft_centerline(
            self.centerline_vertices.to(self.device),
            self.centerline_edges.to(self.device),
            pose,
            self.drr.detector,
        )
        arrays["centerline"] = centerline.detach().cpu().squeeze().numpy()
        arrays["centerline_points"] = centerline_points.detach().cpu().numpy()
        arrays["centerline_chains"] = self.centerline_chains
        arrays["guidewire_chain"] = self.guidewire_points_for_index(self.current_index)
        return arrays

    def generate_drr_and_overlay_images(self):
        try:
            arrays = self.render_frame_arrays()
            points_overlay = render_chain_points_overlay(
                arrays["xray"], arrays["centerline_points"], arrays["centerline_chains"], arrays["guidewire_chain"]
            )
            return (
                numpy_to_base64_cv2(arrays["ct_drr"]),
                generate_overlay_image_cv2(arrays["xray"], arrays["bile_mask"]),
                bgr_to_base64(points_overlay),
            )
        except Exception as exc:
            print(f"generate centerline overlay error: {exc}")
            return "", "", ""

    def save_frame_pngs(self, tag: str, arrays: dict[str, np.ndarray]) -> dict[str, str]:
        paths = super().save_frame_pngs(tag, arrays)
        centerline_path = self.output_dir / tag / f"sxh_xray{self.current_index:03d}_centerline_overlay.png"
        write_overlay_png(centerline_path, arrays["xray"], arrays["centerline"])
        paths["centerline_overlay"] = str(centerline_path)
        points_path = self.output_dir / tag / f"sxh_xray{self.current_index:03d}_registration_points_overlay.png"
        if not cv2.imwrite(
            str(points_path),
            render_chain_points_overlay(
                arrays["xray"], arrays["centerline_points"], arrays["centerline_chains"], arrays["guidewire_chain"]
            ),
        ):
            raise IOError(f"Failed to write registration points overlay: {points_path}")
        paths["registration_points_overlay"] = str(points_path)
        chain_offsets = np.cumsum([0, *[len(chain) for chain in arrays["centerline_chains"]]], dtype=np.int64)
        chain_indices = np.concatenate(arrays["centerline_chains"]) if arrays["centerline_chains"] else np.empty(0, dtype=np.int64)
        point_data_path = self.output_dir / tag / f"sxh_xray{self.current_index:03d}_registration_chains.npz"
        np.savez_compressed(
            point_data_path,
            centerline_points_2d=arrays["centerline_points"],
            centerline_chain_indices=chain_indices,
            centerline_chain_offsets=chain_offsets,
            guidewire_chain_2d=arrays["guidewire_chain"],
        )
        paths["registration_chains"] = str(point_data_path)
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
    parser.add_argument("--guidewire-dir", type=Path, default=DEFAULT_GUIDEWIRE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-index", type=int, default=None)
    return parser.parse_args()


async def run_server(args: argparse.Namespace) -> None:
    if not args.centerline.is_file():
        raise FileNotFoundError(f"Centerline NIfTI not found: {args.centerline}")
    if not args.guidewire_dir.is_dir():
        raise FileNotFoundError(f"Guidewire directory not found: {args.guidewire_dir}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    specimen = IntubationDatasetMR(DEFAULT_CT_NII, DEFAULT_XRAY_ROOT, **SXH_DRR_PARAMS)
    guidewire_indices = guidewire_indices_for_specimen(specimen, args.guidewire_dir)
    start_index = args.start_index
    if start_index is None:
        if args.init_pose == "registered":
            registered_indices = set(discover_registered_indices(args.runs_mask_dir))
            start_index = next((index for index in guidewire_indices if index in registered_indices), None)
            start_index = default_registered_index(args.runs_mask_dir) if start_index is None else start_index
        else:
            start_index = guidewire_indices[0] if guidewire_indices else 0
        start_index = 0 if start_index is None else start_index
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
        guidewire_dir=args.guidewire_dir,
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
    print(f"Guidewire directory: {args.guidewire_dir}")
    print(f"Guidewire dataset indices: {guidewire_indices}")
    print(f"Raw centerline graph: vertices={len(centerline_vertices)}, edges={len(centerline_edges)}")
    print(f"Registered CSV indices: {discover_registered_indices(args.runs_mask_dir)}")
    await asyncio.Future()


async def main() -> None:
    await run_server(parse_args())


if __name__ == "__main__":
    asyncio.run(main())
