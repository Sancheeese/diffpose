"""SXH MRCP 501 NIfTI DRR web server.

MRCP 501 is resampled to the CT 3 DRR grid and rendered with the standard
``DRR`` class. This server shows MRCP DRR only (no bile duct overlay).

Run from ``diffpose/ours`` or the project root:
    python diffpose/ours/web_drr_server_nii_sxh_mrcp.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from aiohttp import web

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFPOSE_ROOT = PROJECT_ROOT / "diffpose"
OURS_ROOT = DIFFPOSE_ROOT / "ours"
for path in (DIFFPOSE_ROOT, OURS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ours.case.sxh.MRCP_dataset_nii import (  # noqa: E402
    DEFAULT_MRCP_NII,
    DEFAULT_XRAY_ROOT,
    IntubationDatasetMRCP,
)
from ours.case.sxh.CT_dataset_nii import Transforms  # noqa: E402
from ours.utils.drr import DRR  # noqa: E402
from ours.web_drr_server_nii import (  # noqa: E402
    WebPoseAdjuster,
    numpy_to_base64_cv2,
)


class SXHMRCPWebPoseAdjuster(WebPoseAdjuster):
    """MRCP-only DRR web UI with poses saved under diffpose/ours/gt_pose/sxh_mrcp."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = PROJECT_ROOT / "diffpose" / "ours" / "gt_pose" / "sxh_mrcp"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_pose(self, subdir="sxh_mrcp"):
        try:
            save_path = PROJECT_ROOT / "diffpose" / "ours" / "gt_pose" / subdir
            save_path.mkdir(parents=True, exist_ok=True)

            current_pose = self.get_current_pose()
            rot = current_pose.get_rotation(parameterization="so3_log_map").detach().cpu().numpy()[0]
            xyz = current_pose.get_translation().detach().cpu().numpy()[0]
            pose_params = np.hstack([rot, xyz]).tolist()

            filepath = save_path / f"pose_{self.current_index:04d}.json"
            pose_data = {
                "index": self.current_index,
                "pose_params": pose_params,
                "rotation": rot.tolist(),
                "translation": xyz.tolist(),
                "timestamp": time.time(),
            }

            with open(filepath, "w") as f:
                json.dump(pose_data, f, indent=2)

            print(f"pose saved to: {filepath}")
            return True
        except Exception as e:
            print(f"save pose error: {e}")
            return False

    def generate_drr_and_overlay_images(self):
        try:
            current_pose = self.get_current_pose()
            with torch.no_grad():
                pred_img = self.drr(None, None, None, pose=current_pose)
                pred_img_transformed = self.transforms(pred_img).to(self.device).to(torch.float32)
                drr_img_np = pred_img_transformed.squeeze().cpu().numpy()

            drr_base64 = numpy_to_base64_cv2(drr_img_np)
            return drr_base64, ""
        except Exception as e:
            print(f"generate SXH MRCP DRR error: {e}")
            return "", ""


async def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    specimen = IntubationDatasetMRCP(
        DEFAULT_MRCP_NII,
        DEFAULT_XRAY_ROOT,
        x_offset=20,
        y_offset=200,
        z_offset=100,
        z_cut=250,
        factors=[0.6, 0.6, 1.5],
    )

    height = 256
    subsample = 512 / height
    delx = specimen.delx * subsample
    delx = 3

    drr = DRR(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.sdr,
        height=height,
        delx=delx,
        reverse_x_axis=True,
        bone_attenuation_multiplier=3,
    ).to(device)

    transforms = Transforms(height)
    adjuster = SXHMRCPWebPoseAdjuster(
        drr,
        None,
        specimen,
        transforms,
        device,
        ws_port=8766,
        http_port=8080,
    )
    adjuster.pose_params = [0, 0, 0, 0, 0, 0]
    adjuster.pose_reset = adjuster.pose_params.copy()

    http_app = web.Application()
    ws_app = web.Application()

    http_app.router.add_get("/", adjuster.http_handler)
    ws_app.router.add_get("/ws", adjuster.websocket_handler)

    http_runner = web.AppRunner(http_app)
    await http_runner.setup()
    http_site = web.TCPSite(http_runner, adjuster.host, adjuster.http_port)

    ws_runner = web.AppRunner(ws_app)
    await ws_runner.setup()
    ws_site = web.TCPSite(ws_runner, adjuster.host, adjuster.ws_port)

    print(f"HTTP server: http://{adjuster.host}:{adjuster.http_port}")
    print(f"WebSocket server: ws://{adjuster.host}:{adjuster.ws_port}/ws")
    print(f"MRCP NIfTI: {DEFAULT_MRCP_NII}")
    print(f"X-ray DICOM root: {DEFAULT_XRAY_ROOT}")
    print("Projection volume: SXH MRCP 501 resampled to the CT 3 DRR grid")
    print("Rendering: MRCP 501 DRR only (no bile duct overlay)")

    await asyncio.gather(
        http_site.start(),
        ws_site.start(),
    )

    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
