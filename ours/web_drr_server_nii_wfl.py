"""WFL NIfTI DRR web server with MRCP 019 bile duct overlay.

Run from ``diffpose/ours`` or the project root:
    python diffpose/ours/web_drr_server_nii_wfl.py

The CT/MRCP transform comes from:
    mrct/outputs/wfl_ct306_mrcp019_gallbladder_icp/ct_to_mr_icp_transform.txt

The projected overlay target is the MRCP bile duct mask, already registered to
the CT 306 grid by the mrct pipeline.
"""

from __future__ import annotations

import asyncio
import json
import os
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

from diffpose.calibration import RigidTransform
from ours.case.wfl.CT_dataset_MR import IntubationDatasetMR
from ours.case.wfl.CT_dataset_nii import Transforms
from ours.utils.drr import DRR
from ours.utils.drr_seg import DRRSeg
from ours.web_drr_server_nii import WebPoseAdjuster, generate_overlay_image_cv2, numpy_to_base64_cv2

DEFAULT_CT_NII = PROJECT_ROOT / "mrct" / "data" / "王凤兰" / "CT" / "306.nii"
DEFAULT_XRAY_ROOT = (
    PROJECT_ROOT
    / "diffpose"
    / "ours"
    / "data"
    / "liwei"
    / "王凤兰"
    / "ERCP"
    / "FENGLAN^WANG^"
    / "20240313160330"
    / "1"
)


class WFLWebPoseAdjuster(WebPoseAdjuster):
    """Use the shared web UI while saving WFL poses under diffpose/ours/gt_pose."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = PROJECT_ROOT / "diffpose" / "ours" / "gt_pose" / "wfl"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_pose(self, subdir="wfl"):
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

                pred_mask = self.drr_bone(None, None, None, pose=current_pose)
                pred_mask = self.transforms.resize(pred_mask)
                mask_np = pred_mask.squeeze().detach().cpu().numpy()
                mask_np = (mask_np > 0.01).astype(np.float32)

            img, _ = self.specimen[self.current_index]
            background_img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
            background_np = background_img.cpu().squeeze().numpy()

            drr_base64 = numpy_to_base64_cv2(drr_img_np)
            overlay_base64 = generate_overlay_image_cv2(background_np, mask_np)

            return drr_base64, overlay_base64
        except Exception as e:
            print(f"generate WFL bile duct overlay error: {e}")
            return "", ""


async def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    specimen = IntubationDatasetMR(
        DEFAULT_CT_NII,
        DEFAULT_XRAY_ROOT,
        x_offset=20,
        z_offset=50,
        z_cut=30,
        z_cut_end=250,
        factors=[0.5, 0.5, 1.0],
    )

    height = 256
    subsample = 512 / height
    delx = specimen.delx * subsample

    drr = DRR(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.sdr,
        height=height,
        delx=delx,
        reverse_x_axis=True,
        bone_attenuation_multiplier=3,
    ).to(device)

    # The second renderer is intentionally the MRCP bile duct mask, not bone.
    # WebPoseAdjuster uses this channel to build the red overlay on the X-ray.
    drr_bile_duct = DRRSeg(
        specimen.mr_mask,
        specimen.spacing,
        sdr=specimen.sdr,
        height=height,
        delx=delx,
        reverse_x_axis=True,
    ).to(device)

    transforms = Transforms(height)
    adjuster = WFLWebPoseAdjuster(drr, drr_bile_duct, specimen, transforms, device)
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
    print(f"CT NIfTI: {DEFAULT_CT_NII}")
    print(f"X-ray DICOM root: {DEFAULT_XRAY_ROOT}")
    print("Overlay: WFL MRCP 019 bile duct registered to CT 306")

    await asyncio.gather(
        http_site.start(),
        ws_site.start(),
    )

    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
