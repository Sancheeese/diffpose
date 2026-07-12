# SXH：MRCP-X-ray 配准病例

这是孙新华（`sxh`）的 MRCP-X-ray 实验目录。该病例已有可验证的 CT-X-ray 配准和 MRCP-to-CT 刚性配准，可作为 XMR 的首个基准病例。

## 数据来源

| 角色 | 路径 |
| --- | --- |
| ERCP X-ray DICOM | `diffpose/ours/data/liwei/孙新华/ERCP/SUNXINHUA^^/20240712155050/1` |
| CT NIfTI | `mrct/data/孙新华/CT/3.nii` |
| MRCP NIfTI | `mrct/data/孙新华/MRCP/501.nii` |
| MRCP 006 胆管分割 | `mrct/data-duet/bile_duct/mrcp_006.nii.gz` |
| MRCP 501 注册到 CT 3 | `mrct/outputs/sxh_ct3_mrcp006_gallbladder_icp/mrcp_501_registered_to_ct.nii.gz` |
| MRCP 胆管注册到 CT 3 | `mrct/outputs/sxh_ct3_mrcp006_gallbladder_icp/mr_bile_duct_registered_to_ct.nii.gz` |
| 既有 CT-Xray 位姿结果 | `diffpose/ours/case/sxh/runs/mask/sxh_xray*_se3_log_map.csv` |

## 实验阶段

1. 用既有 CT-Xray 位姿把已注册到 CT 网格的 MRCP 投影到同一 detector 平面。
2. 读取对应 ERCP X-ray，统一方向、裁剪、像素 spacing 与 ROI。
3. 用 `xmr.mind_ssd` 和 `xmr.search_translation_mind` 估计投影空间的 MIND 粗偏移。
4. 以该偏移和既有 CT-Xray 位姿初始化后续 MRCP 3D/2D 6-DoF 优化。

## X-ray MIND 输出

对每个源 DICOM，仅输出两个同名文件到 `outputs/`：

```text
outputs/
└── 93968938_20240712_1_158/
    ├── 93968938_20240712_1_158.png       # 保持 DICOM 黑白极性的预处理 X-ray
    ├── 93968938_20240712_1_158.mind.pt   # shape=(24, H, W) 的 MIND 张量
    └── visualizations/
        ├── channel_00.png ... channel_23.png
        ├── mean.png                       # 24 通道逐像素均值
        ├── max.png                        # 24 通道逐像素最大值
        └── min.png                        # 24 通道逐像素最小值
```

X-ray PNG 保持 `MONOCHROME2` 的黑白极性，仅做 P1-P99 归一化和轻度平滑，不应用物理
`-log` 衰减变换。特征图按 MIND 原始 `[0, 1]` 范围映射为灰度，不对每张图单独拉伸对比度；因此不同
X-ray 的同类特征图可以直接比较亮度。

```bash
cd diffpose/ours
conda run -n mr2ct python -m xmr.case.sxh.extract_xray_mind \
  data/liwei/孙新华/ERCP/SUNXINHUA^^/20240712155050/1/93968938_20240712_1_158.dcm
```

固定参数为：3x3 patch、八方向偏移（上下左右和四个对角）乘以距离 `1/2/3 px`，共 24 通道、Gaussian `sigma=0.6 px`、P1-P99 归一化、ROI
腐蚀 2 px 和金字塔 `(1, 0.5, 0.25)`。当前导出阶段只应用前四项；ROI 腐蚀和金字塔在
后续 MIND-SSD 配准时使用。

## Web DRR 可视化

交互式 server：[web_drr_server_nii_sxh.py](web_drr_server_nii_sxh.py)

同时显示四路图像：

- 参考 ERCP X-ray
- CT DRR
- MRCP 006 胆管 overlay
- MRCP 501 投影（`DRRMRCP`）

位姿可来自 `register_sxh_mask` 的 CSV 结果，也可从全零开始手调。

```bash
cd diffpose/ours

# 默认：registered 初始化 + MRCP sum 投影
conda run -n mr2ct python xmr/case/sxh/web_drr_server_nii_sxh.py

# 全零初始化
conda run -n mr2ct python xmr/case/sxh/web_drr_server_nii_sxh.py --init-pose zero

# MRCP max 投影
conda run -n mr2ct python xmr/case/sxh/web_drr_server_nii_sxh.py --projection max
```

CLI 参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--init-pose` | `registered` | `zero` 或 `registered` |
| `--runs-mask-dir` | `case/sxh/runs/mask` | CT-Xray 配准 CSV 目录 |
| `--projection` | `sum` | MRCP `DRRMRCP` 模式 |
| `--output-dir` | `xmr/case/sxh/outputs/web_drr` | PNG 导出根目录 |

保存位姿时会同时写出 JSON 和 PNG：

```text
runs/gt_pose/{tag}/
└── pose_0006.json

outputs/web_drr/{tag}/
├── sxh_xray006_xray.png
├── sxh_xray006_ct_drr.png
├── sxh_xray006_bile_overlay.png
└── sxh_xray006_mrcp_sum.png
```

切帧时：

- `registered` 模式：自动加载 `sxh_xray{idx:03d}_se3_log_map.csv` 末行位姿
- `zero` 模式：每帧重置为全零
- Reset 按钮：`registered` 时回到当前帧 CSV 位姿，`zero` 时回到全零

**注意：** 配准 CSV 从 `sxh_xray003` 起才有（index 0–2 无结果）。`registered` 模式默认自动从 **index 3** 启动；也可手动指定 `--start-index 6`。

## 约束

- MRCP 射线累积图是 MRCP line-integral projection，不是物理 X-ray。
- MIND 只在共享的肝门/胆囊/胆管 ROI 内计算；骨、器械、图像边缘和不共享背景必须排除。
- 所有输出写入本目录的 `runs/` 或 `outputs/`，原始 DICOM/NIfTI 不复制到这里。
