from time import process_time_ns

import cv2
import numpy as np
import torch
import matplotlib
import os
import json
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import asyncio
import websockets
import base64
import io
import time
from aiohttp import web
import threading
from diffpose.calibration import RigidTransform, convert
# from ours.utils.CT_dataset_PA import IntubationDataset
# from ours.case.cyx.CT_dataset import IntubationDataset
# from ours.case.cyx.CT_dataset_old import IntubationDataset
# from ours.case.xyl.CT_dataset import IntubationDataset
# from ours.case.jjl.CT_dataset import IntubationDataset
# from ours.case.zt.CT_dataset2 import IntubationDataset
# from ours.case.sjj.CT_dataset2 import IntubationDataset
# from ours.case.qbt.CT_dataset import IntubationDataset
# from ours.case.wfl.CT_dataset import IntubationDataset
# from ours.case.ysy.CT_dataset import IntubationDataset
from ours.case.ysy.CT_dataset_MR import IntubationDatasetMR

from ours.utils.CT_dataset_PA import Transforms, toZeroOne
from ours.utils.drr import DRR
from ours.utils.drr_seg import DRRSeg
from diffpose.deepfluoro import DeepFluoroDataset


def create_circle_mask(image_size, radius):
    """创建圆形掩码"""
    center = image_size // 2
    y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
    mask = ((x - center) ** 2 + (y - center) ** 2) <= radius ** 2
    return mask.float()


def get_edge(pred_img, circle_mask):
    """获取边缘（这里简化实现，您可以根据需要修改）"""
    # 简化版的边缘检测 - 您可以根据需要替换为实际的边缘检测算法
    edge = torch.abs(pred_img) * circle_mask
    return edge


def generate_overlay_image(background_np, edge_np, alpha=0.6):
    background_np = toZeroOne(background_np)
    edge_np = toZeroOne(edge_np)
    """生成叠加图像（方法2）"""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)

    # 显示背景灰度图
    ax.imshow(background_np, cmap='gray')

    # 创建红色半透明叠加
    if edge_np.max() > 1:
        normalized_edge = edge_np / 255.0
    else:
        normalized_edge = edge_np.copy()

    # 创建红色RGBA遮罩
    red_mask = np.zeros((256, 256, 4))
    red_mask[..., 0] = 1.0  # 红色通道
    red_mask[..., 3] = normalized_edge * alpha  # 透明度

    # 叠加显示
    ax.imshow(red_mask)
    ax.axis('off')
    ax.set_title('叠加图像 (边缘检测)', color='white', fontsize=12)
    fig.tight_layout(pad=0)
    fig.patch.set_facecolor('black')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=80, bbox_inches='tight',
                pad_inches=0, facecolor='black')
    buf.seek(0)
    overlay_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return overlay_base64


def generate_overlay_image_cv2(background_np, edge_np, alpha=0.3):
    """使用OpenCV生成叠加图像"""
    # 归一化
    background_np = toZeroOne(background_np)
    edge_np = toZeroOne(edge_np)

    # 转换为0-255范围的uint8
    bg_uint8 = (background_np * 255).astype(np.uint8)
    edge_uint8 = (edge_np * 255).astype(np.uint8)

    # 将灰度图转换为BGR彩色图
    bg_color = cv2.cvtColor(bg_uint8, cv2.COLOR_GRAY2BGR)

    # 创建红色遮罩
    red_mask = np.zeros_like(bg_color)
    red_mask[:, :, 2] = 255  # BGR格式中红色是[0,0,255]

    # 计算alpha通道
    alpha_mask = (edge_uint8 * alpha / 255.0).astype(np.float32)

    # 混合图像
    result = bg_color.astype(np.float32) * (1 - alpha_mask[:, :, None]) + \
             red_mask.astype(np.float32) * alpha_mask[:, :, None]

    # 限制范围并转换类型
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 编码为PNG
    success, encoded_image = cv2.imencode('.png', result)
    if success:
        img_base64 = base64.b64encode(encoded_image).decode('utf-8')
        return img_base64
    return ""


def numpy_to_base64_cv2(img_np, title=None):
    """使用OpenCV将numpy数组转为base64"""
    # 归一化到0-255
    img_np = toZeroOne(img_np)
    img_uint8 = (img_np * 255).astype(np.uint8)

    # 编码为PNG
    success, encoded_image = cv2.imencode('.png', img_uint8)
    if success:
        img_base64 = base64.b64encode(encoded_image).decode('utf-8')
        return img_base64
    return ""


class WebPoseAdjuster:
    def __init__(self, drr, drr_seg, specimen, transforms, device="cuda:0", host="0.0.0.0", ws_port=8765,
                 http_port=8080):
        self.drr = drr
        self.drr_seg = drr_seg
        self.transforms = transforms
        self.device = device
        self.specimen = specimen

        # 初始位姿参数
        self.pose_params = [0, 0, 0, 0, 0, 0]
        # zyl
        # self.pose_params = [0.32, 0.21, -0.83, 208, -2, 46]
        self.pose_reset = self.pose_params.copy()
        self.pose_reset = [0, 0, 0, 0, 0, 0]

        # 当前图像索引
        self.current_index = 0
        self.max_index = len(specimen) - 1  # 获取最大索引

        # # cyx
        # pose_params = [1.7459615468978882, 0.756145179271698, 0.22575482726097107, 278.4220275878906,
        #                236.57859802246094, 81.5419921875]
        # pose_params = [1.7092e+00, 7.3326e-01, 2.4428e-01, 3.1314e+02, 2.6345e+02, 7.2082e+01]
        # pose_params = [  1.6688,  -0.9184,  -0.8960, 193.6940,  93.6377, 113.5820]
        # pose_params = [ 1.4678, -1.0015, -0.9084, 160.0602, 189.4592,   6.5086]
        pose_params = [-0.9320164322853088, -0.023703558370471, 1.7940000295639038, 279.2518310546875, 69.84534454345703, 109.29673767089844]
        # pose_params = [0,0,0,0,0,0]
        rot = torch.tensor([pose_params[:3]], dtype=torch.float32, device=self.device)
        xyz = torch.tensor([pose_params[3:]], dtype=torch.float32, device=self.device)
        # self.pose = RigidTransform(rot, xyz, parameterization="so3_log_map", device=self.device)
        self.pose = RigidTransform(rot, xyz, "euler_angles", "ZYX", device=self.device)
        # self.pose = convert(
        #     [rot, xyz],
        #     input_parameterization="se3_log_map",
        #     output_parameterization="se3_exp_map",
        #     input_convention=None,
        # )

        self.a = torch.tensor([[174.0000, 104.1295, 135.2332]], dtype=torch.float64)
        self.b = torch.tensor([[[0.0, 0.0000, 1.5708]]])
        self.i = 0
        self.iso2 = RigidTransform(
            self.b, self.a, "euler_angles", "ZYX"
        ).to(device)

        rot = torch.tensor([[1.8524, -0.8056, -0.5666]])
        trans = torch.tensor([[256.6452, 3.0481, 128.6834]])
        self.gt_pose = RigidTransform(rot, trans, parameterization="so3_log_map").to(device)

        # 服务器配置
        self.host = host
        self.ws_port = ws_port
        self.http_port = http_port

        # 状态跟踪
        self.current_keys = set()
        self.update_rate = 0.05
        self.running = False
        self.rotation_step = 0.01
        self.translation_step = 2.0

        # 位姿组合
        self.center_pose = specimen.center_pose.to(device)
        self.back_pose = specimen.back_pose.to(device)
        self.isocenter_pose = specimen.isocenter_pose.to(device)
        _, self.gt_pose = specimen[20]
        self.gt_pose = self.gt_pose.to(device)
        t = self.gt_pose.get_translation().squeeze()
        self.gt_center = RigidTransform(torch.eye(3), t).to(device)
        self.gt_back = RigidTransform(torch.eye(3), -t).to(device)

        # 预加载参考图像
        self.reference_img_base64 = self.generate_reference_image()

        # 圆形掩码
        self.circle_mask = create_circle_mask(256, 118).to(self.device).unsqueeze(0).unsqueeze(0)

        # 保存目录
        self.save_dir = "gt_pose/ysy"
        os.makedirs(self.save_dir, exist_ok=True)

        print(f"🚀 WebPoseAdjuster 初始化完成")
        print(f"📡 WebSocket 服务: {self.host}:{self.ws_port}")
        print(f"🌐 HTTP 服务: {self.host}:{self.http_port}")
        print(f"📊 数据集大小: {len(specimen)}, 当前索引: {self.current_index}")
        if hasattr(specimen, "mr_mask"):
            fg = int((specimen.mr_mask > 0).sum())
            print(f"🫀 MRCP 胆管 mask 前景体素: {fg}, 路径: {specimen.mr_mask_path}")

    def generate_reference_image(self):
        """生成参考图像"""
        try:
            # 获取当前索引的数据
            img, pose = self.specimen[self.current_index]
            img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
            img_np = img.cpu().squeeze().numpy()

            # 生成图像
            fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
            ax.imshow(img_np, cmap='gray')
            ax.axis('off')
            ax.set_title(f'参考图像 (索引: {self.current_index})', color='white', fontsize=12)
            fig.tight_layout(pad=0)
            fig.patch.set_facecolor('black')

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=80, bbox_inches='tight',
                        pad_inches=0, facecolor='black')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)

            print(f"✅ 参考图像生成完成, 索引: {self.current_index}")
            return img_base64
        except Exception as e:
            print(f"生成参考图像错误: {e}")
            return ""

    def change_index(self, new_index):
        """改变当前索引"""
        if 0 <= new_index <= self.max_index:
            self.current_index = new_index
            # 重新生成参考图像
            self.reference_img_base64 = self.generate_reference_image()
            return True
        return False

    def save_pose(self, subdir="notag"):
        """保存当前位姿到gt_pose目录下的子目录"""
        try:
            # 基础目录是当前目录下的gt_pose
            base_dir = "gt_pose"

            # 完整的保存路径：gt_pose/子目录
            save_path = os.path.join(base_dir, subdir)

            # 确保目录存在
            os.makedirs(save_path, exist_ok=True)

            # 获取当前位姿
            current_pose = self.get_current_pose()
            rot = current_pose.get_rotation(parameterization="so3_log_map").detach().cpu().numpy()[0]
            xyz = current_pose.get_translation().detach().cpu().numpy()[0]
            pose_params = np.hstack([rot, xyz]).tolist()

            # 创建文件名
            filename = f"pose_{self.current_index:04d}.json"
            filepath = os.path.join(save_path, filename)

            # 保存为JSON格式
            pose_data = {
                "index": self.current_index,
                "pose_params": pose_params,  # 6个参数的列表
                "rotation": rot.tolist(),  # 旋转参数
                "translation": xyz.tolist(),  # 平移参数
                "timestamp": time.time()
            }

            with open(filepath, 'w') as f:
                json.dump(pose_data, f, indent=2)

            print(f"✅ 位姿已保存到: {filepath}")
            return True
        except Exception as e:
            print(f"保存位姿错误: {e}")
            return False

    def get_current_pose(self):
        rot = torch.tensor([self.pose_params[:3]], dtype=torch.float32, device=self.device)
        xyz = torch.tensor([self.pose_params[3:]], dtype=torch.float32, device=self.device)
        # pose = RigidTransform(rot, xyz, parameterization="so3_log_map")
        pose = RigidTransform(rot, xyz, "euler_angles", "ZYX")
        pose = self.isocenter_pose.compose(self.back_pose).compose(pose).compose(self.center_pose)
        # pose = self.pose.compose(pose)
        # pose = self.gt_pose.compose(self.gt_back).compose(pose).compose(self.gt_center)
        # pose = self.isocenter_pose.compose(pose)
        # pose = pose.compose(self.isocenter_pose)
        rot = pose.get_rotation(parameterization="so3_log_map").detach().cpu().numpy()[0]
        xyz = pose.get_translation().detach().cpu().numpy()[0]
        self.curr_pose_param = np.hstack([rot, xyz]).tolist()
        return pose

    def generate_drr_and_overlay_images(self):
        """生成DRR图像和叠加图像"""
        try:
            current_pose = self.get_current_pose()
            with torch.no_grad():
                # 生成DRR图像
                pred_img = self.drr(None, None, None, pose=current_pose)
                pred_img_transformed = self.transforms(pred_img).to(self.device).to(torch.float32)
                drr_img_np = pred_img_transformed.squeeze().cpu().numpy()

                # MRCP 胆管分割投影（binary any-hit DRR）
                pred_seg = self.drr_seg(None, None, None, pose=current_pose)
                pred_seg = self.transforms(pred_seg, reverse=False).to(self.device).to(torch.float32)
                pred_seg = pred_seg * self.circle_mask
                seg_np = pred_seg.squeeze().cpu().numpy()

            # 获取当前索引的参考图像作为叠加背景
            img, _ = self.specimen[self.current_index]
            background_img = self.transforms(img, reverse=False).to(self.device).to(torch.float32)
            background_np = background_img.cpu().squeeze().numpy()

            # 使用OpenCV生成DRR图像
            drr_base64 = numpy_to_base64_cv2(drr_img_np)

            # 使用OpenCV生成叠加图像
            overlay_base64 = generate_overlay_image_cv2(background_np, seg_np)

            return drr_base64, overlay_base64

        except Exception as e:
            print(f"生成图像错误: {e}")
            return "", ""

    def update_pose_continuous(self):
        """根据当前按下的键持续更新位姿"""
        step_multiplier = 6.0 if 'shift' in self.current_keys else 1.0

        key_actions = {
            'q': (0, self.rotation_step), 'a': (0, -self.rotation_step),
            'w': (1, self.rotation_step), 's': (1, -self.rotation_step),
            'e': (2, self.rotation_step), 'd': (2, -self.rotation_step),
            'u': (3, self.translation_step), 'j': (3, -self.translation_step),
            'i': (4, self.translation_step), 'k': (4, -self.translation_step),
            'o': (5, self.translation_step), 'l': (5, -self.translation_step),
        }

        for key, (idx, step) in key_actions.items():
            if key in self.current_keys:
                self.pose_params[idx] += step * step_multiplier

    async def send_updates(self, websocket):
        """持续发送图像更新"""
        self.running = True
        print("开始实时流传输...")

        try:
            while self.running and not websocket.closed:
                start_time = time.time()

                if self.current_keys:
                    self.update_pose_continuous()

                # 生成DRR图像和叠加图像
                drr_img_data, overlay_img_data = self.generate_drr_and_overlay_images()

                if drr_img_data and overlay_img_data:
                    message = {
                        'type': 'image_update',
                        'drr_image': drr_img_data,
                        'overlay_image': overlay_img_data,
                        'pose': self.pose_params.copy(),
                        # 'pose': self.curr_pose_param.copy(),
                        'index': self.current_index,
                        'timestamp': time.time()
                    }
                    await websocket.send_json(message)

                elapsed = time.time() - start_time
                await asyncio.sleep(max(0.0, self.update_rate - elapsed))

        except Exception as e:
            print(f"发送更新错误: {e}")
        finally:
            self.running = False
            print("实时流传输停止")

    async def websocket_handler(self, request):
        """WebSocket处理器"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        client_ip = request.remote
        print(f"WebSocket客户端连接: {client_ip}")

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        msg_type = data.get('type')

                        if msg_type == 'keydown':
                            self.current_keys.add(data.get('key'))
                        elif msg_type == 'keyup':
                            self.current_keys.discard(data.get('key'))
                        elif msg_type == 'start_stream':
                            # 启动实时更新
                            asyncio.create_task(self.send_updates(ws))
                        elif msg_type == 'stop_stream':
                            self.running = False
                        elif msg_type == 'reset_pose':
                            # self.pose_params = self.pose_reset
                            self.pose_params = [0,0,0,0,0,0]
                            print("位姿已重置")
                        elif msg_type == 'change_index':
                            # 改变索引
                            new_index = data.get('index')
                            if new_index is not None:
                                success = self.change_index(new_index)
                                if success:
                                    # 发送更新后的参考图像
                                    await ws.send_json({
                                        'type': 'reference_update',
                                        'reference_image': self.reference_img_base64,
                                        'index': self.current_index
                                    })
                        elif msg_type == 'index_prev':
                            # 索引减一
                            success = self.change_index(self.current_index - 1)
                            if success:
                                await ws.send_json({
                                    'type': 'reference_update',
                                    'reference_image': self.reference_img_base64,
                                    'index': self.current_index
                                })
                        elif msg_type == 'index_next':
                            # 索引加一
                            success = self.change_index(self.current_index + 1)
                            if success:
                                await ws.send_json({
                                    'type': 'reference_update',
                                    'reference_image': self.reference_img_base64,
                                    'index': self.current_index
                                })
                        elif msg_type == 'save_pose':
                            # 保存位姿
                            tag = data.get('tag', 'notag')
                            self.save_pose(tag)

                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {e}")

                elif msg.type == web.WSMsgType.ERROR:
                    print(f"WebSocket错误: {ws.exception()}")
                elif msg.type == web.WSMsgType.CLOSE:
                    print(f"WebSocket连接关闭: {client_ip}")
                    break

        except Exception as e:
            print(f"WebSocket处理错误: {e}")
        finally:
            self.running = False
            self.current_keys.clear()
            print(f"WebSocket客户端断开: {client_ip}")

        return ws

    async def http_handler(self, request):
        """HTTP请求处理器 - 返回HTML页面"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>实时DRR位姿调整 (MRCP胆管)</title>
    <meta charset="utf-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1600px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .header h1 {{ margin-bottom: 10px; font-size: 2.2em; }}
        .content {{ display: grid; grid-template-columns: 1fr 1fr 400px; gap: 20px; padding: 20px; }}
        .image-section {{ background: #1a1a1a; border-radius: 10px; padding: 15px; text-align: center; }}
        .image-container {{ display: flex; flex-direction: column; align-items: center; }}
        .image-title {{ color: white; font-size: 16px; font-weight: bold; margin-bottom: 10px; }}
        .drr-image {{ max-width: 100%; max-height: 500px; border: 2px solid #34495e; border-radius: 8px; background: #000; width: auto; height: 480px; object-fit: contain;}}
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
        .image-stack {{ display: flex; flex-direction: column; gap: 10px; }}
        .index-controls {{ display: flex; align-items: center; gap: 10px; margin: 10px 0; }}
        .index-input {{ width: 80px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; text-align: center; }}
        .index-button {{ padding: 8px 12px; background: #17a2b8; color: white; border: none; border-radius: 4px; cursor: pointer; }}
        .save-controls {{ display: flex; align-items: center; gap: 10px; margin: 10px 0; }}
        .save-input {{ flex: 1; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }}
        .save-button {{ padding: 8px 16px; background: #6f42c1; color: white; border: none; border-radius: 4px; cursor: pointer; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 实时DRR位姿调整 (MRCP 胆管叠加)</h1>
            <p>按住键盘按键实时调整3D位姿参数，叠加层为 ysy mrcp_003 胆管分割投影</p>
        </div>
        <div class="content">
            <!-- 参考图像和叠加图像 -->
            <div class="image-section">
                <div class="image-stack">
                    <div class="image-container">
                        <div class="image-title">📋 参考图像 (索引: <span id="currentIndex">{self.current_index}</span>)</div>
                        <img id="referenceImage" src="data:image/png;base64,{self.reference_img_base64}" alt="Reference Image" class="drr-image">

                        <!-- 索引控制 -->
                        <div class="index-controls">
                            <button id="indexPrev" class="index-button">-1</button>
                            <input type="number" id="indexInput" class="index-input" value="{self.current_index}" min="0" max="{self.max_index}">
                            <button id="indexNext" class="index-button">+1</button>
                        </div>

                        <!-- 保存控制 -->
                        <div class="save-controls">
                            <input type="text" id="saveTagInput" class="save-input" value="notag" placeholder="保存目录">
                            <button id="savePoseButton" class="save-button">保存位姿</button>
                        </div>
                    </div>
                    <div class="image-container">
                        <div class="image-title">🎯 MRCP 胆管分割叠加</div>
                        <img id="overlayImage" src="" alt="Overlay Image" class="drr-image" onerror="this.style.display='none'">
                        <div id="overlayPlaceholder" style="color: #666; padding: 50px;">叠加图像将在此显示...</div>
                    </div>
                </div>
            </div>

            <!-- 实时渲染图像 -->
            <div class="image-section">
                <div class="image-container">
                    <div class="image-title">🎮 实时渲染</div>
                    <img id="drrImage" src="" alt="DRR Image" class="drr-image" onerror="this.style.display='none'">
                    <div id="imagePlaceholder" style="color: #666; padding: 50px;">图像将在此显示... 请先连接服务器</div>
                </div>
            </div>

            <!-- 控制面板 -->
            <div class="controls-section">
                <div class="control-panel">
                    <h2>🔗 服务器连接</h2>
                    <div class="status-panel">
                        <span id="statusIcon" class="connection-status disconnected"></span>
                        <span id="statusText">未连接</span>
                    </div>
                    <div style="margin: 15px 0;">
                        <input type="text" id="serverUrl" value="ws://SERVER_IP:8765/ws" style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 10px;">
                        <button id="connectBtn" class="button">连接服务器</button>
                        <button id="disconnectBtn" class="button disconnect" disabled>断开连接</button>
                    </div>
                    <button id="startBtn" class="button" disabled>开始实时调整</button>
                    <button id="resetBtn" class="button" disabled>重置位姿</button>
                </div>
                <div class="control-panel">
                    <h2>⌨️ 键盘控制</h2>
                    <div class="instructions" style="background: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; font-size: 14px;">
                        <p><strong>使用说明:</strong> 连接服务器并开始实时调整后，按住以下按键进行控制：</p>
                        <p>🔄 <strong>Shift + 按键</strong> = 加速调整</p>
                    </div>
                    <h3>旋转控制 (弧度)</h3>
                    <div class="key-grid">
                        <div class="key-item" data-key="q">Q<br>Alpha+</div>
                        <div class="key-item" data-key="a">A<br>Alpha-</div>
                        <div class="key-item" data-key="w">W<br>Beta+</div>
                        <div class="key-item" data-key="s">S<br>Beta-</div>
                        <div class="key-item" data-key="e">E<br>Gamma+</div>
                        <div class="key-item" data-key="d">D<br>Gamma-</div>
                    </div>
                    <h3>平移控制 (毫米)</h3>
                    <div class="key-grid">
                        <div class="key-item" data-key="u">U<br>Bx+</div>
                        <div class="key-item" data-key="j">J<br>Bx-</div>
                        <div class="key-item" data-key="i">I<br>By+</div>
                        <div class="key-item" data-key="k">K<br>By-</div>
                        <div class="key-item" data-key="o">O<br>Bz+</div>
                        <div class="key-item" data-key="l">L<br>Bz-</div>
                    </div>
                </div>
                <div class="pose-info">
                    <h2>📊 当前位姿参数</h2>
                    <div id="poseInfo">
                        <div class="param-row"><span class="param-name">Alpha (X):</span><span class="param-value" id="alpha">0.0000</span></div>
                        <div class="param-row"><span class="param-name">Beta (Y):</span><span class="param-value" id="beta">0.0000</span></div>
                        <div class="param-row"><span class="param-name">Gamma (Z):</span><span class="param-value" id="gamma">0.0000</span></div>
                        <div class="param-row"><span class="param-name">Bx (mm):</span><span class="param-value" id="bx">0.0</span></div>
                        <div class="param-row"><span class="param-name">By (mm):</span><span class="param-value" id="by">0.0</span></div>
                        <div class="param-row"><span class="param-name">Bz (mm):</span><span class="param-value" id="bz">0.0</span></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 自动设置服务器IP
        document.getElementById('serverUrl').value = 'ws://' + window.location.hostname + ':8765/ws';

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

                // 索引控制
                document.getElementById('indexPrev').addEventListener('click', () => this.changeIndex(-1));
                document.getElementById('indexNext').addEventListener('click', () => this.changeIndex(1));
                document.getElementById('indexInput').addEventListener('change', (e) => this.setIndex(parseInt(e.target.value)));

                // 保存位姿
                document.getElementById('savePoseButton').addEventListener('click', () => this.savePose());

                document.addEventListener('keydown', (e) => this.handleKeyDown(e));
                document.addEventListener('keyup', (e) => this.handleKeyUp(e));

                document.querySelectorAll('.key-item').forEach(item => {{
                    item.addEventListener('mousedown', (e) => {{
                        const key = e.currentTarget.getAttribute('data-key');
                        this.simulateKeyPress(key, 'keydown');
                    }});
                    item.addEventListener('mouseup', (e) => {{
                        const key = e.currentTarget.getAttribute('data-key');
                        this.simulateKeyPress(key, 'keyup');
                    }});
                    item.addEventListener('mouseleave', (e) => {{
                        const key = e.currentTarget.getAttribute('data-key');
                        if (this.activeKeys.has(key)) {{
                            this.simulateKeyPress(key, 'keyup');
                        }}
                    }});
                }});
            }}

            connect() {{
                const wsUrl = document.getElementById('serverUrl').value;
                this.ws = new WebSocket(wsUrl);
                this.updateStatus('连接中...', 'streaming');

                this.ws.onopen = () => {{
                    this.isConnected = true;
                    this.updateStatus('连接成功', 'connected');
                    this.updateButtonStates();
                    document.getElementById('imagePlaceholder').style.display = 'none';
                    document.getElementById('overlayPlaceholder').style.display = 'none';
                }};

                this.ws.onmessage = (event) => {{
                    try {{
                        const data = JSON.parse(event.data);
                        if (data.type === 'image_update') {{
                            // 更新DRR图像
                            document.getElementById('drrImage').src = `data:image/png;base64,${{data.drr_image}}`;
                            document.getElementById('drrImage').style.display = 'block';

                            // 更新叠加图像
                            document.getElementById('overlayImage').src = `data:image/png;base64,${{data.overlay_image}}`;
                            document.getElementById('overlayImage').style.display = 'block';

                            this.updatePoseInfo(data.pose);
                        }} else if (data.type === 'reference_update') {{
                            // 更新参考图像
                            document.getElementById('referenceImage').src = `data:image/png;base64,${{data.reference_image}}`;
                            document.getElementById('currentIndex').textContent = data.index;
                            document.getElementById('indexInput').value = data.index;
                        }}
                    }} catch (e) {{
                        console.error('消息解析错误:', e);
                    }}
                }};

                this.ws.onclose = () => {{
                    this.isConnected = false;
                    this.isStreaming = false;
                    this.updateStatus('连接断开', 'disconnected');
                    this.updateButtonStates();
                    this.activeKeys.clear();
                    this.updateKeyVisuals();
                }};
            }}

            disconnect() {{
                if (this.ws) {{
                    this.ws.close();
                    this.ws = null;
                }}
                this.isConnected = false;
                this.isStreaming = false;
                this.updateStatus('未连接', 'disconnected');
                this.updateButtonStates();
            }}

            toggleStream() {{
                if (!this.isConnected) return;

                if (!this.isStreaming) {{
                    this.ws.send(JSON.stringify({{ type: 'start_stream' }}));
                    this.isStreaming = true;
                    document.getElementById('startBtn').textContent = '停止实时调整';
                    this.updateStatus('实时调整中...', 'streaming');
                }} else {{
                    this.ws.send(JSON.stringify({{ type: 'stop_stream' }}));
                    this.isStreaming = false;
                    document.getElementById('startBtn').textContent = '开始实时调整';
                    this.updateStatus('连接成功', 'connected');
                }}
            }}

            resetPose() {{
                if (this.isConnected) {{
                    this.ws.send(JSON.stringify({{ type: 'reset_pose' }}));
                }}
            }}

            changeIndex(delta) {{
                if (!this.isConnected) return;
                const currentIndex = parseInt(document.getElementById('indexInput').value);
                const newIndex = currentIndex + delta;
                this.setIndex(newIndex);
            }}

            setIndex(newIndex) {{
                if (!this.isConnected) return;
                this.ws.send(JSON.stringify({{ type: 'change_index', index: newIndex }}));
            }}

            savePose() {{
                if (!this.isConnected) return;
                const tag = document.getElementById('saveTagInput').value || 'notag';
                this.ws.send(JSON.stringify({{ type: 'save_pose', tag: tag }}));
                alert(`位姿已保存到 ${{tag}} 目录`);
            }}

            handleKeyDown(e) {{
                if (!this.isConnected || !this.isStreaming) return;

                const key = e.key.toLowerCase();
                if (this.isValidKey(key) && !this.activeKeys.has(key)) {{
                    e.preventDefault();
                    this.activeKeys.add(key);
                    this.ws.send(JSON.stringify({{ type: 'keydown', key: key }}));
                    this.updateKeyVisuals();
                }}
            }}

            handleKeyUp(e) {{
                if (!this.isConnected || !this.isStreaming) return;

                const key = e.key.toLowerCase();
                if (this.activeKeys.has(key)) {{
                    e.preventDefault();
                    this.activeKeys.delete(key);
                    this.ws.send(JSON.stringify({{ type: 'keyup', key: key }}));
                    this.updateKeyVisuals();
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
                this.updateKeyVisuals();
            }}

            isValidKey(key) {{
                const validKeys = ['q', 'a', 'w', 's', 'e', 'd', 'u', 'j', 'i', 'k', 'o', 'l', 'shift'];
                return validKeys.includes(key);
            }}

            updateKeyVisuals() {{
                document.querySelectorAll('.key-item').forEach(item => {{
                    const key = item.getAttribute('data-key');
                    item.classList.toggle('active', this.activeKeys.has(key));
                }});
            }}

            updateStatus(message, type) {{
                const statusEl = document.getElementById('statusText');
                const iconEl = document.getElementById('statusIcon');
                statusEl.textContent = message;
                iconEl.className = 'connection-status ' + type;
            }}

            updateButtonStates() {{
                document.getElementById('connectBtn').disabled = this.isConnected;
                document.getElementById('disconnectBtn').disabled = !this.isConnected;
                document.getElementById('startBtn').disabled = !this.isConnected;
                document.getElementById('resetBtn').disabled = !this.isConnected;
            }}

            updatePoseInfo(pose) {{
                if (pose && pose.length === 6) {{
                    const [alpha, beta, gamma, bx, by, bz] = pose;
                    document.getElementById('alpha').textContent = alpha.toFixed(4);
                    document.getElementById('beta').textContent = beta.toFixed(4);
                    document.getElementById('gamma').textContent = gamma.toFixed(4);
                    document.getElementById('bx').textContent = bx.toFixed(1);
                    document.getElementById('by').textContent = by.toFixed(1);
                    document.getElementById('bz').textContent = bz.toFixed(1);
                }}
            }}
        }}

        new DRRClient();
    </script>
</body>
</html>
        """
        return web.Response(text=html_content, content_type='text/html')


# main函数保持不变...
async def main():
    device = "cuda:0"

    nii_path = "/home/zsr/project/mrct/data/杨式瑜/CT/7.nii"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/杨式瑜/ERCP/SHIYU^YANG^/20240515170510/1"

    specimen = IntubationDatasetMR(
        nii_path,
        x_root,
        y_offset=50,
        z_offset=-50,
        z_cut=400,
        factors=[0.7, 1.5, 0.7],
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

    drr_seg = DRRSeg(
        specimen.mr_mask,
        specimen.spacing,
        sdr=specimen.sdr,
        height=height,
        delx=delx,
        reverse_x_axis=True,
    ).to(device)

    transforms = Transforms(height)

    adjuster = WebPoseAdjuster(drr, drr_seg, specimen, transforms, device)

    # 创建两个aiohttp应用
    http_app = web.Application()
    ws_app = web.Application()

    # 分别添加路由
    http_app.router.add_get('/', adjuster.http_handler)
    ws_app.router.add_get('/ws', adjuster.websocket_handler)

    # 启动HTTP服务器（8080端口）
    http_runner = web.AppRunner(http_app)
    await http_runner.setup()
    http_site = web.TCPSite(http_runner, adjuster.host, adjuster.http_port)

    # 启动WebSocket服务器（8765端口）
    ws_runner = web.AppRunner(ws_app)
    await ws_runner.setup()
    ws_site = web.TCPSite(ws_runner, adjuster.host, adjuster.ws_port)

    print(f"🌐 HTTP服务器启动在: http://{adjuster.host}:{adjuster.http_port}")
    print(f"📡 WebSocket服务器启动在: ws://{adjuster.host}:{adjuster.ws_port}/ws")
    print("📱 请在浏览器中访问以上地址")

    # 同时启动两个服务器
    await asyncio.gather(
        http_site.start(),
        ws_site.start()
    )

    # 保持运行
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())