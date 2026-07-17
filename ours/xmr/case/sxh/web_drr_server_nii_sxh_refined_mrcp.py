"""Visualize the guidewire-refined SXH CT-MRCP registration at X-ray 031.

The server follows the centerline viewer: it renders CT DRR, the refined
bile-duct overlay, and raw-MRCP centerline branches over the X-ray together
with the trusted automatic guidewire chain. It starts at X-ray 031 using the
automatic CT-X-ray registration result; it never reads manual poses.

Run from ``diffpose/ours`` or the project root:
    python xmr/case/sxh/web_drr_server_nii_sxh_refined_mrcp.py
    python -m xmr.case.sxh.web_drr_server_nii_sxh_refined_mrcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch
from aiohttp import web

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SXH_CASE_ROOT = Path(__file__).resolve().parent
DIFFPOSE_ROOT = PROJECT_ROOT / "diffpose"
OURS_ROOT = DIFFPOSE_ROOT / "ours"
for path in (DIFFPOSE_ROOT, OURS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ours.case.sxh.CT_dataset_MR import IntubationDatasetMR, apply_drr_axis_pipeline  # noqa: E402
from ours.case.sxh.CT_dataset_nii import Transforms  # noqa: E402
from ours.case.sxh.MRCP_dataset_nii import (  # noqa: E402
    DEFAULT_MRCP_NII,
    DEFAULT_XRAY_ROOT,
    IntubationDatasetMRCP,
)
from ours.utils.drr import DRR  # noqa: E402
from ours.utils.drr_mrcp import DRRMRCP  # noqa: E402
from ours.utils.drr_seg import DRRSeg  # noqa: E402
from ours.xmr.case.sxh.pose_io import DEFAULT_RUNS_MASK_DIR  # noqa: E402
from ours.xmr.case.sxh.centerline.centerline import extract_centerline_tree  # noqa: E402
from ours.xmr.case.sxh.web_drr_server_nii_sxh_centerline import (  # noqa: E402
    SXHCenterlineWebPoseAdjuster,
    ct_voxels_to_drr_mm,
    extract_ordered_guidewire_chain,
)
from ours.web_drr_server_nii import numpy_to_base64_cv2  # noqa: E402


DEFAULT_CT_NII = PROJECT_ROOT / "mrct" / "data" / "孙新华" / "CT" / "3.nii"
DEFAULT_REFINED_DIR = SXH_CASE_ROOT / "runs" / "refined_ct_mrcp_guidewire_xray031"
DEFAULT_REFINED_MRCP = DEFAULT_REFINED_DIR / "mrcp_501_registered_to_ct_refined.nii.gz"
DEFAULT_REFINED_BILE_DUCT = DEFAULT_REFINED_DIR / "mr_bile_duct_registered_to_ct_refined.nii.gz"
DEFAULT_REFINED_CT_TO_MR = DEFAULT_REFINED_DIR / "ct_to_mr_refined.txt"
DEFAULT_OUTPUT_DIR = SXH_CASE_ROOT / "runs" / "web_drr_refined_ct_mrcp_guidewire_xray031"
DEFAULT_START_INDEX = 31
DEFAULT_CENTERLINE_MASK = PROJECT_ROOT / "mrct" / "data-duet" / "bile_duct" / "mrcp_006.nii.gz"
DEFAULT_GUIDEWIRE_OVERLAY = SXH_CASE_ROOT / "outputs" / "guidewire_registration" / "xray031" / "final_overlay.png"

SXH_DRR_PARAMS = dict(
    x_offset=20,
    y_offset=200,
    z_offset=100,
    z_cut=250,
    factors=[0.6, 0.6, 1.5],
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SXH guidewire-refined CT DRR + centerline web server (X-ray 031 initialization)"
    )
    parser.add_argument(
        "--init-pose",
        choices=["zero", "registered"],
        default="registered",
        help="Initial CT-X-ray pose: zero or automatic registration CSV",
    )
    parser.add_argument(
        "--runs-mask-dir",
        type=Path,
        default=DEFAULT_RUNS_MASK_DIR,
        help="Directory containing automatic sxh_xrayNNN_se3_log_map.csv registration results",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=DEFAULT_START_INDEX,
        help="Initial X-ray index; default is the guidewire-refined frame 31",
    )
    parser.add_argument("--refined-bile-duct", type=Path, default=DEFAULT_REFINED_BILE_DUCT)
    parser.add_argument("--refined-ct-to-mr", type=Path, default=DEFAULT_REFINED_CT_TO_MR)
    parser.add_argument(
        "--centerline-mask",
        type=Path,
        default=DEFAULT_CENTERLINE_MASK,
        help="Raw MRCP bile-duct mask used to derive the complete 3D centerline tree",
    )
    parser.add_argument(
        "--guidewire-overlay",
        type=Path,
        default=DEFAULT_GUIDEWIRE_OVERLAY,
        help="Trusted automatic xray031 final overlay; its red chain is used as the guidewire",
    )
    parser.add_argument(
        "--projection",
        choices=["sum", "max"],
        default="max",
        help="MRCP DRR ray aggregation mode",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for rendering; defaults to cuda:1 when available",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=8081)
    parser.add_argument("--ws-port", type=int, default=8767)
    return parser.parse_args(argv)


def validate_refined_inputs(args: argparse.Namespace) -> None:
    """Fail before server startup instead of silently falling back to legacy ICP data."""
    for name, path in (
        ("refined bile duct", args.refined_bile_duct),
        ("refined CT-to-MR transform", args.refined_ct_to_mr),
        ("raw MRCP centerline mask", args.centerline_mask),
        ("trusted guidewire overlay", args.guidewire_overlay),
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")


def load_refined_centerline_graph_in_drr_coordinates(
    centerline_mask: Path,
    refined_ct_to_mr: Path,
    drr_spacing: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the raw MRCP skeleton tree onto the CT DRR grid via the refined rigid transform."""
    centerline_nii = nib.load(str(centerline_mask))
    tree = extract_centerline_tree(np.asanyarray(centerline_nii.dataobj) > 0, centerline_nii.affine)
    ct_nii = nib.load(str(DEFAULT_CT_NII))
    mr_to_ct = np.linalg.inv(np.loadtxt(refined_ct_to_mr, dtype=np.float64))
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


def load_guidewire_chain_from_overlay(path: Path) -> np.ndarray:
    """Recover the automatic guidewire chain from the red points saved by its trusted optimizer."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read guidewire overlay: {path}")
    blue, green, red = cv2.split(image)
    red_mask = ((red > 180) & (green < 100) & (blue < 100)).astype(np.uint8)
    # The optimizer export stores the guidewire as spaced red points. Connect
    # adjacent samples before extracting its longest ordered skeleton path.
    red_mask = cv2.dilate(red_mask, np.ones((3, 3), dtype=np.uint8))
    chain = extract_ordered_guidewire_chain(red_mask)
    if len(chain) < 2:
        raise ValueError(f"No usable red guidewire chain in overlay: {path}")
    return chain


class SXHRefinedMRCPWebPoseAdjuster(SXHCenterlineWebPoseAdjuster):
    """Centerline viewer whose MR-to-CT mapping is the guidewire-refined transform."""

    def __init__(
        self,
        *args,
        guidewire_overlay: Path,
        guidewire_index: int,
        drr_mrcp: DRRMRCP,
        mrcp_projection_mode: str,
        **kwargs,
    ):
        self.guidewire_overlay = Path(guidewire_overlay)
        self.guidewire_index = guidewire_index
        self._trusted_guidewire_chain: np.ndarray | None = None
        super().__init__(*args, **kwargs)
        self.drr_mrcp = drr_mrcp
        self.mrcp_projection_mode = mrcp_projection_mode

    def guidewire_points_for_index(self, index: int) -> np.ndarray:
        if index != self.guidewire_index:
            return np.empty((0, 2), dtype=np.float32)
        if self._trusted_guidewire_chain is None:
            self._trusted_guidewire_chain = load_guidewire_chain_from_overlay(self.guidewire_overlay)
        return self._trusted_guidewire_chain

    def render_frame_arrays(self) -> dict[str, np.ndarray]:
        arrays = super().render_frame_arrays()
        with torch.no_grad():
            projection = self.drr_mrcp(None, None, None, pose=self.get_current_pose())
            projection = self.transforms(projection).to(self.device).to(torch.float32)
        arrays["mrcp_projection"] = projection.squeeze().cpu().numpy()
        # Preserve the existing save/export convention for the MRCP projection.
        arrays["mrcp"] = arrays["mrcp_projection"]
        return arrays

    def centerline_and_mrcp_images(self) -> tuple[str, str, str, str]:
        try:
            arrays = self.render_frame_arrays()
            from ours.xmr.case.sxh.web_drr_server_nii_sxh_centerline import bgr_to_base64, render_chain_points_overlay
            from ours.web_drr_server_nii import generate_overlay_image_cv2

            centerline_overlay = render_chain_points_overlay(
                arrays["xray"], arrays["centerline_points"], arrays["centerline_chains"], arrays["guidewire_chain"]
            )
            return (
                numpy_to_base64_cv2(arrays["ct_drr"]),
                generate_overlay_image_cv2(arrays["xray"], arrays["bile_mask"]),
                bgr_to_base64(centerline_overlay),
                numpy_to_base64_cv2(arrays["mrcp_projection"]),
            )
        except Exception as exc:
            print(f"generate refined centerline/MRCP images error: {exc}")
            return "", "", "", ""

    async def send_updates(self, websocket):
        self.running = True
        try:
            while self.running and not websocket.closed:
                start_time = time.time()
                if self.current_keys:
                    self.update_pose_continuous()
                drr_image, bile_overlay, centerline_overlay, mrcp_projection = self.centerline_and_mrcp_images()
                if drr_image:
                    await websocket.send_json(
                        {
                            "type": "image_update",
                            "drr_image": drr_image,
                            "overlay_image": bile_overlay,
                            "mrcp_image": centerline_overlay,
                            "mrcp_projection_image": mrcp_projection,
                            "pose": self.pose_params.copy(),
                            "index": self.current_index,
                            "timestamp": time.time(),
                        }
                    )
                await asyncio.sleep(max(0.0, self.update_rate - (time.time() - start_time)))
        except Exception as exc:
            print(f"refined centerline streaming error: {exc}")
        finally:
            self.running = False

    async def http_handler(self, request):
        response = await super().http_handler(request)
        projection_panel = f'''                <div class="image-section">
                    <div class="image-container">
                        <div class="image-title">MRCP Projection ({self.mrcp_projection_mode})</div>
                        <img id="mrcpProjectionImage" src="" alt="MRCP Projection" class="drr-image" onerror="this.style.display='none'">
                        <div id="mrcpProjectionPlaceholder" style="color: #666; padding: 50px;">MRCP projection will appear here...</div>
                    </div>
                </div>
'''
        html = response.text.replace(
            "            </div>\n            <div class=\"controls-section\">",
            f"{projection_panel}            </div>\n            <div class=\"controls-section\">",
            1,
        )
        html = html.replace(
            "document.getElementById('mrcpPlaceholder').style.display = 'none';",
            "document.getElementById('mrcpPlaceholder').style.display = 'none';\n"
            "                    document.getElementById('mrcpProjectionPlaceholder').style.display = 'none';",
        )
        html = html.replace(
            "                        this.updatePoseInfo(data.pose);",
            "                        if (data.mrcp_projection_image) {\n"
            "                            document.getElementById('mrcpProjectionImage').src = `data:image/png;base64,${data.mrcp_projection_image}`;\n"
            "                            document.getElementById('mrcpProjectionImage').style.display = 'block';\n"
            "                        }\n"
            "                        this.updatePoseInfo(data.pose);",
            1,
        )
        html = html.replace("SXH DRR + Centerline", "SXH Refined Centerline + MRCP")
        return web.Response(text=html, content_type="text/html")

    def save_pose(self, subdir: str = "manual") -> dict[str, str]:
        try:
            # The tag is supplied by the browser, so keep all outputs under output_dir.
            tag = Path(subdir).name or "manual"
            save_path = self.output_dir / tag
            save_path.mkdir(parents=True, exist_ok=True)

            arrays = self.render_frame_arrays()
            current_pose = self.get_current_pose()
            rot = current_pose.get_rotation(parameterization="so3_log_map").detach().cpu().numpy()[0]
            xyz = current_pose.get_translation().detach().cpu().numpy()[0]
            pose_params = np.hstack([rot, xyz]).tolist()

            pose_filepath = save_path / f"pose_{self.current_index:04d}.json"
            pose_data = {
                "index": self.current_index,
                "pose_params": pose_params,
                "rotation": rot.tolist(),
                "translation": xyz.tolist(),
                "init_pose_mode": self.init_pose_mode,
                "projection_mode": self.projection_mode,
                "registration": "guidewire_refined_ct_mrcp_xray031",
                "timestamp": time.time(),
            }
            pose_filepath.write_text(json.dumps(pose_data, indent=2), encoding="utf-8")

            png_paths = self.save_frame_pngs(tag, arrays)
            self._last_save_paths = {"pose_json": str(pose_filepath), **png_paths}
            print(f"pose saved to: {pose_filepath}")
            return self._last_save_paths
        except Exception as exc:
            print(f"save pose error: {exc}")
            return {}


def build_adjuster(args: argparse.Namespace) -> SXHRefinedMRCPWebPoseAdjuster:
    """Build the CT DRR and refined centerline/guidewire overlay renderers."""
    validate_refined_inputs(args)
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    else:
        device = "cpu"

    specimen = IntubationDatasetMR(DEFAULT_CT_NII, DEFAULT_XRAY_ROOT, **SXH_DRR_PARAMS)
    mrcp_specimen = IntubationDatasetMRCP(
        DEFAULT_MRCP_NII,
        DEFAULT_XRAY_ROOT,
        registered_mrcp_path=DEFAULT_REFINED_MRCP,
        registered_mr_mask_path=args.refined_bile_duct,
        icp_transform_path=args.refined_ct_to_mr,
        **SXH_DRR_PARAMS,
    )
    refined_mask_native = np.asanyarray(nib.load(str(args.refined_bile_duct)).dataobj)
    refined_mask = apply_drr_axis_pipeline(
        refined_mask_native,
        SXH_DRR_PARAMS["factors"],
        z_cut=SXH_DRR_PARAMS["z_cut"],
        order=0,
    )
    if refined_mask.shape != specimen.volume.shape:
        raise ValueError(
            f"Refined bile-duct DRR shape {refined_mask.shape} does not match CT shape {specimen.volume.shape}"
        )
    centerline_vertices, centerline_edges = load_refined_centerline_graph_in_drr_coordinates(
        args.centerline_mask,
        args.refined_ct_to_mr,
        specimen.spacing,
    )

    height = 256
    delx = specimen.delx * (512 / height)
    drr = DRR(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.sdr,
        height=height,
        delx=delx,
        reverse_x_axis=True,
        bone_attenuation_multiplier=3,
    ).to(device)
    drr_bile_duct = DRRSeg(
        refined_mask,
        specimen.spacing,
        sdr=specimen.sdr,
        height=height,
        delx=delx,
        reverse_x_axis=True,
    ).to(device)
    drr_mrcp = DRRMRCP(
        mrcp_specimen.volume,
        mrcp_specimen.spacing,
        sdr=mrcp_specimen.sdr,
        height=height,
        delx=delx,
        reverse_x_axis=True,
        projection_mode=args.projection,
    ).to(device)
    return SXHRefinedMRCPWebPoseAdjuster(
        drr,
        drr_bile_duct,
        specimen,
        Transforms(height),
        device,
        centerline_vertices=centerline_vertices,
        centerline_edges=centerline_edges,
        guidewire_dir=SXH_CASE_ROOT,
        guidewire_overlay=args.guidewire_overlay,
        guidewire_index=DEFAULT_START_INDEX,
        drr_mrcp=drr_mrcp,
        mrcp_projection_mode=args.projection,
        init_pose_mode=args.init_pose,
        runs_mask_dir=args.runs_mask_dir,
        output_dir=args.output_dir,
        projection_mode="centerline",
        host=args.host,
        http_port=args.http_port,
        ws_port=args.ws_port,
    )


async def run_server(args: argparse.Namespace) -> None:
    adjuster = build_adjuster(args)
    if not 0 <= args.start_index <= adjuster.max_index:
        raise ValueError(f"start_index {args.start_index} out of range [0, {adjuster.max_index}]")

    adjuster.current_index = args.start_index
    adjuster.apply_initial_pose(args.start_index)
    adjuster.reference_img_base64 = adjuster.generate_reference_image()

    http_app = web.Application()
    ws_app = web.Application()
    http_app.router.add_get("/", adjuster.http_handler)
    ws_app.router.add_get("/ws", adjuster.websocket_handler)

    http_runner = web.AppRunner(http_app)
    ws_runner = web.AppRunner(ws_app)
    await http_runner.setup()
    await ws_runner.setup()
    await web.TCPSite(http_runner, adjuster.host, adjuster.http_port).start()
    await web.TCPSite(ws_runner, adjuster.host, adjuster.ws_port).start()

    print(f"HTTP server: http://{adjuster.host}:{adjuster.http_port}")
    print(f"WebSocket server: ws://{adjuster.host}:{adjuster.ws_port}/ws")
    print(f"initial frame: {args.start_index}")
    print(f"initial pose_params: {adjuster.pose_params}")
    print(f"refined bile duct: {args.refined_bile_duct}")
    print(f"centerline mask: {args.centerline_mask}")
    print(f"guidewire overlay: {args.guidewire_overlay}")
    print(f"MRCP projection mode: {args.projection}")
    print(f"saved reviews: {args.output_dir}")
    await asyncio.Future()


def main() -> None:
    asyncio.run(run_server(parse_args()))


if __name__ == "__main__":
    main()
