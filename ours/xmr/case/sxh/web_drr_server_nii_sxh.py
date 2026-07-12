"""SXH NIfTI DRR web server with MRCP 006 bile duct overlay and MRCP 501 projection.

Run from ``diffpose/ours`` or the project root:
    python xmr/case/sxh/web_drr_server_nii_sxh.py
    python xmr/case/sxh/web_drr_server_nii_sxh.py --init-pose registered
    python xmr/case/sxh/web_drr_server_nii_sxh.py --init-pose zero --projection max
    python -m xmr.case.sxh.web_drr_server_nii_sxh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from aiohttp import web
from diffpose.calibration import RigidTransform

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SXH_CASE_ROOT = Path(__file__).resolve().parent
DIFFPOSE_ROOT = PROJECT_ROOT / "diffpose"
OURS_ROOT = DIFFPOSE_ROOT / "ours"
for path in (DIFFPOSE_ROOT, OURS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ours.case.sxh.CT_dataset_MR import IntubationDatasetMR  # noqa: E402
from ours.case.sxh.CT_dataset_nii import Transforms  # noqa: E402
from ours.case.sxh.MRCP_dataset_nii import (  # noqa: E402
    DEFAULT_MRCP_NII,
    IntubationDatasetMRCP,
)
from ours.utils.drr import DRR  # noqa: E402
from ours.utils.drr_mrcp import DRRMRCP  # noqa: E402
from ours.utils.drr_seg import DRRSeg  # noqa: E402
from ours.web_drr_server_nii import (  # noqa: E402
    WebPoseAdjuster,
    generate_overlay_image_cv2,
    numpy_to_base64_cv2,
)

from xmr.case.sxh.image_io import write_gray_png, write_overlay_png  # noqa: E402
from xmr.case.sxh.pose_io import (  # noqa: E402
    DEFAULT_RUNS_MASK_DIR,
    default_registered_index,
    discover_registered_indices,
    load_registered_pose,
)


DEFAULT_CT_NII = PROJECT_ROOT / "mrct" / "data" / "孙新华" / "CT" / "3.nii"
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
DEFAULT_OUTPUT_DIR = SXH_CASE_ROOT / "outputs" / "web_drr"
ZERO_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SXH CT DRR + bile duct + MRCP web server")
    parser.add_argument(
        "--init-pose",
        choices=["zero", "registered"],
        default="registered",
        help="Initial pose per frame: all zeros or register_sxh_mask CSV result",
    )
    parser.add_argument(
        "--runs-mask-dir",
        type=Path,
        default=DEFAULT_RUNS_MASK_DIR,
        help="Directory containing sxh_xrayNNN_se3_log_map.csv files",
    )
    parser.add_argument(
        "--projection",
        choices=["sum", "max"],
        default="sum",
        help="MRCP DRRMRCP ray aggregation mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for exported PNG frames",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Initial dataset index. Default: first index with registration CSV in registered mode, else 0",
    )
    return parser.parse_args()


class SXHWebPoseAdjuster(WebPoseAdjuster):
    """SXH web UI with optional registered pose init, MRCP projection, and PNG export."""

    def __init__(
        self,
        drr,
        drr_bone,
        specimen,
        transforms,
        device="cuda:0",
        *,
        drr_mrcp: DRRMRCP | None = None,
        init_pose_mode: Literal["zero", "registered"] = "registered",
        runs_mask_dir: Path = DEFAULT_RUNS_MASK_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        projection_mode: str = "sum",
        host="0.0.0.0",
        ws_port=8765,
        http_port=8080,
    ):
        self.drr_mrcp = drr_mrcp
        self.init_pose_mode = init_pose_mode
        self.runs_mask_dir = Path(runs_mask_dir)
        self.output_dir = Path(output_dir)
        self.projection_mode = projection_mode
        self._last_save_paths: dict[str, str] = {}

        super().__init__(
            drr,
            drr_bone,
            specimen,
            transforms,
            device,
            host=host,
            ws_port=ws_port,
            http_port=http_port,
        )
        self.save_dir = SXH_CASE_ROOT / "runs" / "gt_pose"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def registered_global_to_local_pose_params(self, global_params: list[float]) -> list[float]:
        """Convert a registered global CT-to-X-ray pose to the UI's local pose."""
        rotation = torch.tensor([global_params[:3]], dtype=torch.float32, device=self.device)
        translation = torch.tensor([global_params[3:]], dtype=torch.float32, device=self.device)
        global_pose = RigidTransform(rotation, translation, "euler_angles", "ZYX")

        local_pose = (
            self.isocenter_pose.compose(self.back_pose)
            .inverse()
            .compose(global_pose)
            .compose(self.center_pose.inverse())
        )
        local_rotation = local_pose.get_rotation("euler_angles", "ZYX").detach().cpu().numpy()[0]
        local_translation = local_pose.get_translation().detach().cpu().numpy()[0]
        return np.hstack([local_rotation, local_translation]).tolist()

    def apply_initial_pose(self, index: int) -> list[float]:
        if self.init_pose_mode == "registered":
            registered = load_registered_pose(index, self.runs_mask_dir)
            if registered is not None:
                self.pose_params = self.registered_global_to_local_pose_params(registered)
                self.pose_reset = self.pose_params.copy()
                return self.pose_params

        self.pose_params = ZERO_POSE.copy()
        self.pose_reset = ZERO_POSE.copy()
        if self.init_pose_mode == "registered":
            print(f"warning: falling back to zero pose for index {index}")
        return self.pose_params

    def change_index(self, new_index):
        if 0 <= new_index <= self.max_index:
            self.current_index = new_index
            self.apply_initial_pose(new_index)
            self.reference_img_base64 = self.generate_reference_image()
            return True
        return False

    def reset_pose_params(self):
        if self.init_pose_mode == "registered":
            self.apply_initial_pose(self.current_index)
        else:
            self.pose_params = ZERO_POSE.copy()
        print(f"pose reset for index {self.current_index}")

    def render_frame_arrays(self) -> dict[str, np.ndarray]:
        current_pose = self.get_current_pose()
        with torch.no_grad():
            pred_img = self.drr(None, None, None, pose=current_pose)
            pred_img_transformed = self.transforms(pred_img).to(self.device).to(torch.float32)
            drr_img_np = pred_img_transformed.squeeze().cpu().numpy()

            pred_mask = self.drr_bone(None, None, None, pose=current_pose)
            pred_mask = self.transforms.resize(pred_mask)
            mask_np = pred_mask.squeeze().detach().cpu().numpy()
            mask_np = (mask_np > 0.01).astype(np.float32)

            mrcp_img_np = np.zeros_like(drr_img_np, dtype=np.float32)
            if self.drr_mrcp is not None:
                pred_mrcp = self.drr_mrcp(None, None, None, pose=current_pose)
                pred_mrcp = self.transforms(pred_mrcp).to(self.device).to(torch.float32)
                mrcp_img_np = pred_mrcp.squeeze().detach().cpu().numpy()

        img, _ = self.specimen[self.current_index]
        background_img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
        background_np = background_img.cpu().squeeze().numpy()

        return {
            "xray": background_np,
            "ct_drr": drr_img_np,
            "bile_mask": mask_np,
            "mrcp": mrcp_img_np,
        }

    def generate_drr_and_overlay_images(self):
        try:
            arrays = self.render_frame_arrays()
            drr_base64 = numpy_to_base64_cv2(arrays["ct_drr"])
            overlay_base64 = generate_overlay_image_cv2(arrays["xray"], arrays["bile_mask"])
            mrcp_base64 = numpy_to_base64_cv2(arrays["mrcp"]) if self.drr_mrcp is not None else ""
            return drr_base64, overlay_base64, mrcp_base64
        except Exception as e:
            print(f"generate SXH bile duct/MRCP overlay error: {e}")
            return "", "", ""

    def save_frame_pngs(self, tag: str, arrays: dict[str, np.ndarray]) -> dict[str, str]:
        frame_dir = self.output_dir / tag
        frame_dir.mkdir(parents=True, exist_ok=True)
        idx = self.current_index
        paths = {
            "xray": frame_dir / f"sxh_xray{idx:03d}_xray.png",
            "ct_drr": frame_dir / f"sxh_xray{idx:03d}_ct_drr.png",
            "bile_overlay": frame_dir / f"sxh_xray{idx:03d}_bile_overlay.png",
            "mrcp": frame_dir / f"sxh_xray{idx:03d}_mrcp_{self.projection_mode}.png",
        }

        write_gray_png(paths["xray"], arrays["xray"])
        write_gray_png(paths["ct_drr"], arrays["ct_drr"])
        write_overlay_png(paths["bile_overlay"], arrays["xray"], arrays["bile_mask"])
        if self.drr_mrcp is not None:
            write_gray_png(paths["mrcp"], arrays["mrcp"])

        return {key: str(path) for key, path in paths.items()}

    def save_pose(self, subdir="gt_pose"):
        try:
            save_path = SXH_CASE_ROOT / "runs" / subdir
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
                "timestamp": time.time(),
            }

            with open(pose_filepath, "w") as f:
                json.dump(pose_data, f, indent=2)

            png_paths = self.save_frame_pngs(subdir, arrays)
            self._last_save_paths = {
                "pose_json": str(pose_filepath),
                **png_paths,
            }

            print(f"pose saved to: {pose_filepath}")
            for key, path in png_paths.items():
                print(f"{key} saved to: {path}")
            return self._last_save_paths
        except Exception as e:
            print(f"save pose error: {e}")
            return {}

    async def send_updates(self, websocket):
        self.running = True
        print("start streaming...")

        try:
            while self.running and not websocket.closed:
                start_time = time.time()

                if self.current_keys:
                    self.update_pose_continuous()

                drr_img_data, overlay_img_data, mrcp_img_data = self.generate_drr_and_overlay_images()

                if drr_img_data:
                    message = {
                        "type": "image_update",
                        "drr_image": drr_img_data,
                        "overlay_image": overlay_img_data,
                        "mrcp_image": mrcp_img_data,
                        "pose": self.pose_params.copy(),
                        "index": self.current_index,
                        "timestamp": time.time(),
                    }
                    await websocket.send_json(message)

                elapsed = time.time() - start_time
                await asyncio.sleep(max(0.0, self.update_rate - elapsed))
        except Exception as e:
            print(f"send update error: {e}")
        finally:
            self.running = False
            print("streaming stopped")

    async def websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        client_ip = request.remote
        print(f"websocket client connected: {client_ip}")

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")

                        if msg_type == "keydown":
                            self.current_keys.add(data.get("key"))
                        elif msg_type == "keyup":
                            self.current_keys.discard(data.get("key"))
                        elif msg_type == "start_stream":
                            asyncio.create_task(self.send_updates(ws))
                        elif msg_type == "stop_stream":
                            self.running = False
                        elif msg_type == "reset_pose":
                            self.reset_pose_params()
                        elif msg_type == "change_index":
                            new_index = data.get("index")
                            if new_index is not None and self.change_index(new_index):
                                await ws.send_json(
                                    {
                                        "type": "reference_update",
                                        "reference_image": self.reference_img_base64,
                                        "index": self.current_index,
                                        "pose": self.pose_params.copy(),
                                    }
                                )
                        elif msg_type == "index_prev":
                            if self.change_index(self.current_index - 1):
                                await ws.send_json(
                                    {
                                        "type": "reference_update",
                                        "reference_image": self.reference_img_base64,
                                        "index": self.current_index,
                                        "pose": self.pose_params.copy(),
                                    }
                                )
                        elif msg_type == "index_next":
                            if self.change_index(self.current_index + 1):
                                await ws.send_json(
                                    {
                                        "type": "reference_update",
                                        "reference_image": self.reference_img_base64,
                                        "index": self.current_index,
                                        "pose": self.pose_params.copy(),
                                    }
                                )
                        elif msg_type == "save_pose":
                            tag = data.get("tag", "gt_pose")
                            saved_paths = self.save_pose(tag)
                            await ws.send_json(
                                {
                                    "type": "save_result",
                                    "success": bool(saved_paths),
                                    "paths": saved_paths,
                                    "index": self.current_index,
                                    "tag": tag,
                                }
                            )
                    except json.JSONDecodeError as e:
                        print(f"json decode error: {e}")
                elif msg.type == web.WSMsgType.ERROR:
                    print(f"websocket error: {ws.exception()}")
                elif msg.type == web.WSMsgType.CLOSE:
                    print(f"websocket closed: {client_ip}")
                    break
        except Exception as e:
            print(f"websocket handler error: {e}")
        finally:
            self.running = False
            self.current_keys.clear()
            print(f"websocket client disconnected: {client_ip}")

        return ws

    async def http_handler(self, request):
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SXH DRR + MRCP</title>
    <meta charset="utf-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1800px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .header h1 {{ margin-bottom: 10px; font-size: 2.0em; }}
        .content {{ display: grid; grid-template-columns: 1.4fr 400px; gap: 20px; padding: 20px; }}
        .image-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
        .image-section {{ background: #1a1a1a; border-radius: 10px; padding: 15px; text-align: center; }}
        .image-container {{ display: flex; flex-direction: column; align-items: center; }}
        .image-title {{ color: white; font-size: 15px; font-weight: bold; margin-bottom: 10px; }}
        .drr-image {{ max-width: 100%; max-height: 360px; border: 2px solid #34495e; border-radius: 8px; background: #000; width: auto; height: 340px; object-fit: contain;}}
        .controls-section {{ display: flex; flex-direction: column; gap: 20px; }}
        .control-panel {{ background: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef; }}
        .status-panel {{ background: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3; }}
        .pose-info {{ background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; font-family: 'Courier New', monospace; }}
        .key-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 15px 0; }}
        .key-item {{ padding: 12px 8px; background: white; border: 2px solid #dee2e6; border-radius: 8px; text-align: center; cursor: pointer; user-select: none; transition: all 0.2s ease; font-weight: bold; }}
        .key-item:hover {{ border-color: #007bff; background: #f8f9fa; }}
        .key-item.active {{ background: #007bff; color: white; border-color: #0056b3; transform: scale(0.95); }}
        .button {{ padding: 12px 20px; background: #28a745; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold; transition: background 0.3s ease; margin: 5px; }}
        .button:hover {{ background: #218838; }}
        .button:disabled {{ background: #6c757d; cursor: not-allowed; }}
        .button.disconnect {{ background: #dc3545; }}
        .button.disconnect:hover {{ background: #c82333; }}
        .connection-status {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }}
        .connected {{ background: #28a745; }}
        .disconnected {{ background: #dc3545; }}
        .streaming {{ background: #ffc107; animation: pulse 1.5s infinite; }}
        @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} 100% {{ opacity: 1; }} }}
        .param-row {{ display: flex; justify-content: space-between; margin: 5px 0; padding: 3px 0; border-bottom: 1px solid #f1f1f1; }}
        .param-name {{ font-weight: bold; color: #495057; }}
        .param-value {{ font-family: 'Courier New', monospace; color: #e83e8c; }}
        .index-controls {{ display: flex; align-items: center; gap: 10px; margin: 10px 0; }}
        .index-input {{ width: 80px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; text-align: center; }}
        .index-button {{ padding: 8px 12px; background: #17a2b8; color: white; border: none; border-radius: 4px; cursor: pointer; }}
        .save-controls {{ display: flex; align-items: center; gap: 10px; margin: 10px 0; }}
        .save-input {{ flex: 1; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }}
        .save-button {{ padding: 8px 16px; background: #6f42c1; color: white; border: none; border-radius: 4px; cursor: pointer; }}
        .meta {{ color: #ddd; font-size: 13px; margin-top: 6px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SXH CT DRR + Bile Duct + MRCP</h1>
            <p>init_pose={self.init_pose_mode}, mrcp_projection={self.projection_mode}</p>
        </div>
        <div class="content">
            <div class="image-grid">
                <div class="image-section">
                    <div class="image-container">
                        <div class="image-title">Reference X-ray (index: <span id="currentIndex">{self.current_index}</span>)</div>
                        <img id="referenceImage" src="data:image/png;base64,{self.reference_img_base64}" alt="Reference Image" class="drr-image">
                        <div class="index-controls">
                            <button id="indexPrev" class="index-button">-1</button>
                            <input type="number" id="indexInput" class="index-input" value="{self.current_index}" min="0" max="{self.max_index}">
                            <button id="indexNext" class="index-button">+1</button>
                        </div>
                        <div class="save-controls">
                            <input type="text" id="saveTagInput" class="save-input" value="manual" placeholder="save tag">
                            <button id="savePoseButton" class="save-button">Save Pose + PNGs</button>
                        </div>
                    </div>
                </div>
                <div class="image-section">
                    <div class="image-container">
                        <div class="image-title">Bile Duct Overlay</div>
                        <img id="overlayImage" src="" alt="Overlay Image" class="drr-image" onerror="this.style.display='none'">
                        <div id="overlayPlaceholder" style="color: #666; padding: 50px;">Overlay will appear here...</div>
                    </div>
                </div>
                <div class="image-section">
                    <div class="image-container">
                        <div class="image-title">CT DRR</div>
                        <img id="drrImage" src="" alt="DRR Image" class="drr-image" onerror="this.style.display='none'">
                        <div id="imagePlaceholder" style="color: #666; padding: 50px;">CT DRR will appear here...</div>
                    </div>
                </div>
                <div class="image-section">
                    <div class="image-container">
                        <div class="image-title">MRCP Projection</div>
                        <img id="mrcpImage" src="" alt="MRCP Image" class="drr-image" onerror="this.style.display='none'">
                        <div id="mrcpPlaceholder" style="color: #666; padding: 50px;">MRCP projection will appear here...</div>
                    </div>
                </div>
            </div>
            <div class="controls-section">
                <div class="control-panel">
                    <h2>Server</h2>
                    <div class="status-panel">
                        <span id="statusIcon" class="connection-status disconnected"></span>
                        <span id="statusText">Disconnected</span>
                    </div>
                    <div style="margin: 15px 0;">
                        <input type="text" id="serverUrl" value="ws://SERVER_IP:{self.ws_port}/ws" style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 10px;">
                        <button id="connectBtn" class="button">Connect</button>
                        <button id="disconnectBtn" class="button disconnect" disabled>Disconnect</button>
                    </div>
                    <button id="startBtn" class="button" disabled>Start Streaming</button>
                    <button id="resetBtn" class="button" disabled>Reset Pose</button>
                </div>
                <div class="control-panel">
                    <h2>Keyboard</h2>
                    <div class="key-grid">
                        <div class="key-item" data-key="q">Q Alpha+</div>
                        <div class="key-item" data-key="a">A Alpha-</div>
                        <div class="key-item" data-key="w">W Beta+</div>
                        <div class="key-item" data-key="s">S Beta-</div>
                        <div class="key-item" data-key="e">E Gamma+</div>
                        <div class="key-item" data-key="d">D Gamma-</div>
                        <div class="key-item" data-key="u">U Bx+</div>
                        <div class="key-item" data-key="j">J Bx-</div>
                        <div class="key-item" data-key="i">I By+</div>
                        <div class="key-item" data-key="k">K By-</div>
                        <div class="key-item" data-key="o">O Bz+</div>
                        <div class="key-item" data-key="l">L Bz-</div>
                    </div>
                </div>
                <div class="pose-info">
                    <h2>Current Pose</h2>
                    <div id="poseInfo">
                        <div class="param-row"><span class="param-name">Alpha:</span><span class="param-value" id="alpha">{self.pose_params[0]:.4f}</span></div>
                        <div class="param-row"><span class="param-name">Beta:</span><span class="param-value" id="beta">{self.pose_params[1]:.4f}</span></div>
                        <div class="param-row"><span class="param-name">Gamma:</span><span class="param-value" id="gamma">{self.pose_params[2]:.4f}</span></div>
                        <div class="param-row"><span class="param-name">Bx:</span><span class="param-value" id="bx">{self.pose_params[3]:.1f}</span></div>
                        <div class="param-row"><span class="param-name">By:</span><span class="param-value" id="by">{self.pose_params[4]:.1f}</span></div>
                        <div class="param-row"><span class="param-name">Bz:</span><span class="param-value" id="bz">{self.pose_params[5]:.1f}</span></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        document.getElementById('serverUrl').value = 'ws://' + window.location.hostname + ':{self.ws_port}/ws';

        class DRRClient {{
            constructor() {{
                this.ws = null;
                this.isConnected = false;
                this.isStreaming = false;
                this.activeKeys = new Set();
                this.setupEventListeners();
            }}

            setupEventListeners() {{
                document.getElementById('connectBtn').addEventListener('click', () => this.connect());
                document.getElementById('disconnectBtn').addEventListener('click', () => this.disconnect());
                document.getElementById('startBtn').addEventListener('click', () => this.toggleStream());
                document.getElementById('resetBtn').addEventListener('click', () => this.resetPose());
                document.getElementById('indexPrev').addEventListener('click', () => this.changeIndex(-1));
                document.getElementById('indexNext').addEventListener('click', () => this.changeIndex(1));
                document.getElementById('indexInput').addEventListener('change', (e) => this.setIndex(parseInt(e.target.value)));
                document.getElementById('savePoseButton').addEventListener('click', () => this.savePose());
                document.addEventListener('keydown', (e) => this.handleKeyDown(e));
                document.addEventListener('keyup', (e) => this.handleKeyUp(e));
                document.querySelectorAll('.key-item').forEach(item => {{
                    item.addEventListener('mousedown', (e) => {{
                        this.simulateKeyPress(e.currentTarget.getAttribute('data-key'), 'keydown');
                    }});
                    item.addEventListener('mouseup', (e) => {{
                        this.simulateKeyPress(e.currentTarget.getAttribute('data-key'), 'keyup');
                    }});
                }});
            }}

            connect() {{
                this.ws = new WebSocket(document.getElementById('serverUrl').value);
                this.updateStatus('Connecting...', 'streaming');
                this.ws.onopen = () => {{
                    this.isConnected = true;
                    this.updateStatus('Connected', 'connected');
                    this.updateButtonStates();
                    document.getElementById('imagePlaceholder').style.display = 'none';
                    document.getElementById('overlayPlaceholder').style.display = 'none';
                    document.getElementById('mrcpPlaceholder').style.display = 'none';
                }};
                this.ws.onmessage = (event) => {{
                    const data = JSON.parse(event.data);
                    if (data.type === 'image_update') {{
                        document.getElementById('drrImage').src = `data:image/png;base64,${{data.drr_image}}`;
                        document.getElementById('drrImage').style.display = 'block';
                        if (data.overlay_image) {{
                            document.getElementById('overlayImage').src = `data:image/png;base64,${{data.overlay_image}}`;
                            document.getElementById('overlayImage').style.display = 'block';
                        }}
                        if (data.mrcp_image) {{
                            document.getElementById('mrcpImage').src = `data:image/png;base64,${{data.mrcp_image}}`;
                            document.getElementById('mrcpImage').style.display = 'block';
                        }}
                        this.updatePoseInfo(data.pose);
                    }} else if (data.type === 'reference_update') {{
                        document.getElementById('referenceImage').src = `data:image/png;base64,${{data.reference_image}}`;
                        document.getElementById('currentIndex').textContent = data.index;
                        document.getElementById('indexInput').value = data.index;
                        if (data.pose) this.updatePoseInfo(data.pose);
                    }} else if (data.type === 'save_result') {{
                        if (data.success) {{
                            alert('Saved pose and PNGs under tag: ' + data.tag);
                        }} else {{
                            alert('Save failed');
                        }}
                    }}
                }};
                this.ws.onclose = () => {{
                    this.isConnected = false;
                    this.isStreaming = false;
                    this.updateStatus('Disconnected', 'disconnected');
                    this.updateButtonStates();
                }};
            }}

            disconnect() {{
                if (this.ws) this.ws.close();
                this.ws = null;
            }}

            toggleStream() {{
                if (!this.isConnected) return;
                if (!this.isStreaming) {{
                    this.ws.send(JSON.stringify({{ type: 'start_stream' }}));
                    this.isStreaming = true;
                    document.getElementById('startBtn').textContent = 'Stop Streaming';
                }} else {{
                    this.ws.send(JSON.stringify({{ type: 'stop_stream' }}));
                    this.isStreaming = false;
                    document.getElementById('startBtn').textContent = 'Start Streaming';
                }}
            }}

            resetPose() {{
                if (this.isConnected) this.ws.send(JSON.stringify({{ type: 'reset_pose' }}));
            }}

            changeIndex(delta) {{
                const currentIndex = parseInt(document.getElementById('indexInput').value);
                this.setIndex(currentIndex + delta);
            }}

            setIndex(newIndex) {{
                if (!this.isConnected) return;
                this.ws.send(JSON.stringify({{ type: 'change_index', index: newIndex }}));
            }}

            savePose() {{
                if (!this.isConnected) return;
                const tag = document.getElementById('saveTagInput').value || 'manual';
                this.ws.send(JSON.stringify({{ type: 'save_pose', tag: tag }}));
            }}

            handleKeyDown(e) {{
                if (!this.isConnected || !this.isStreaming) return;
                const key = e.key.toLowerCase();
                if (this.isValidKey(key) && !this.activeKeys.has(key)) {{
                    e.preventDefault();
                    this.activeKeys.add(key);
                    this.ws.send(JSON.stringify({{ type: 'keydown', key: key }}));
                }}
            }}

            handleKeyUp(e) {{
                if (!this.isConnected || !this.isStreaming) return;
                const key = e.key.toLowerCase();
                if (this.activeKeys.has(key)) {{
                    e.preventDefault();
                    this.activeKeys.delete(key);
                    this.ws.send(JSON.stringify({{ type: 'keyup', key: key }}));
                }}
            }}

            simulateKeyPress(key, type) {{
                if (!this.isConnected || !this.isStreaming) return;
                if (type === 'keydown' && !this.activeKeys.has(key)) {{
                    this.activeKeys.add(key);
                    this.ws.send(JSON.stringify({{ type: 'keydown', key: key }}));
                }} else if (type === 'keyup' && this.activeKeys.has(key)) {{
                    this.activeKeys.delete(key);
                    this.ws.send(JSON.stringify({{ type: 'keyup', key: key }}));
                }}
            }}

            isValidKey(key) {{
                return ['q','a','w','s','e','d','u','j','i','k','o','l','shift'].includes(key);
            }}

            updateStatus(message, type) {{
                document.getElementById('statusText').textContent = message;
                document.getElementById('statusIcon').className = 'connection-status ' + type;
            }}

            updateButtonStates() {{
                document.getElementById('connectBtn').disabled = this.isConnected;
                document.getElementById('disconnectBtn').disabled = !this.isConnected;
                document.getElementById('startBtn').disabled = !this.isConnected;
                document.getElementById('resetBtn').disabled = !this.isConnected;
            }}

            updatePoseInfo(pose) {{
                if (!pose || pose.length !== 6) return;
                document.getElementById('alpha').textContent = pose[0].toFixed(4);
                document.getElementById('beta').textContent = pose[1].toFixed(4);
                document.getElementById('gamma').textContent = pose[2].toFixed(4);
                document.getElementById('bx').textContent = pose[3].toFixed(1);
                document.getElementById('by').textContent = pose[4].toFixed(1);
                document.getElementById('bz').textContent = pose[5].toFixed(1);
            }}
        }}

        new DRRClient();
    </script>
</body>
</html>
        """
        return web.Response(text=html_content, content_type="text/html")


async def run_server(args: argparse.Namespace):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    if args.start_index is not None:
        start_index = args.start_index
    elif args.init_pose == "registered":
        start_index = default_registered_index(args.runs_mask_dir)
        if start_index is None:
            print("warning: no registration CSV found; starting at index 0 with zero pose")
            start_index = 0
    else:
        start_index = 0

    specimen = IntubationDatasetMR(
        DEFAULT_CT_NII,
        DEFAULT_XRAY_ROOT,
        x_offset=20,
        y_offset=200,
        z_offset=100,
        z_cut=250,
        factors=[0.6, 0.6, 1.5],
    )

    mrcp_specimen = IntubationDatasetMRCP(
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
        specimen.mr_mask,
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

    transforms = Transforms(height)
    adjuster = SXHWebPoseAdjuster(
        drr,
        drr_bile_duct,
        specimen,
        transforms,
        device,
        drr_mrcp=drr_mrcp,
        init_pose_mode=args.init_pose,
        runs_mask_dir=args.runs_mask_dir,
        output_dir=args.output_dir,
        projection_mode=args.projection,
    )
    if not (0 <= start_index <= adjuster.max_index):
        raise ValueError(f"start_index {start_index} out of range [0, {adjuster.max_index}]")

    adjuster.current_index = start_index
    adjuster.apply_initial_pose(start_index)
    adjuster.reference_img_base64 = adjuster.generate_reference_image()

    registered_indices = discover_registered_indices(args.runs_mask_dir)
    print(f"start_index: {start_index}")
    print(f"initial pose_params: {adjuster.pose_params}")
    if args.init_pose == "registered":
        print(f"registered CSV indices available: {registered_indices[:5]}...{registered_indices[-3:] if len(registered_indices) > 5 else ''}")

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
    print(f"init_pose: {args.init_pose}")
    print(f"runs_mask_dir: {args.runs_mask_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"CT NIfTI: {DEFAULT_CT_NII}")
    print(f"MRCP NIfTI: {DEFAULT_MRCP_NII}")
    print(f"X-ray DICOM root: {DEFAULT_XRAY_ROOT}")
    print("Overlay: SXH MRCP 006 bile duct registered to CT 3")
    print(f"MRCP projection: DRRMRCP ({args.projection})")

    await asyncio.gather(
        http_site.start(),
        ws_site.start(),
    )

    await asyncio.Future()


async def main():
    args = parse_args()
    await run_server(args)


if __name__ == "__main__":
    asyncio.run(main())
