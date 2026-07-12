# XMR: MRCP-X-ray 配准

`xmr/` 是 `diffpose/ours/` 内针对 **MRCP 与 ERCP/X-ray** 的独立研究模块。它不替代既有 CT-X-ray 病例脚本；目标是逐步建立 MRCP 投影到真实 X-ray 的可复现实验链路。

## 当前基线

当前实现提供一个可测试的 2D 初始化器：

1. 将 MRCP 体数据按 X-ray 相机几何投影为 2D 图像；直接射线累积的结果称为 **MRCP line-integral projection**，不是物理 X-ray。
2. 在固定 X-ray 和移动 MRCP 投影上计算 2D MIND 局部自相似特征。
3. 在指定 ROI 内最小化 MIND-SSD，网格搜索整数像素平移，作为后续 3D/2D 位姿优化的初始化。

当前的 `mind_descriptor` 是投影图像使用的 2D MIND-style 描述子；它不是 ConvexAdam 中的 3D MIND-SSC。

## 输入契约

- `fixed`：预处理后的真实 X-ray/ERCP 图像，形状 `(H, W)`。
- `moving`：由 MRCP 以同一相机几何生成的 2D 投影，形状 `(H, W)`。
- `mask`：可选的肝门、胆囊和胆管 ROI。必须排除图像边缘、器械和不共享的骨性结构。
- 两图必须在进入 MIND 前统一方向、像素坐标、视场和尺度。

## 病例入口

患者级实验位于 `xmr/case/<patient>/`。首个基准病例是
[`case/sxh/`](case/sxh/README.md)：它引用现有的孙新华 ERCP X-ray、MRCP 501、
MRCP 006 胆管分割以及已验证的 CT-X-ray 位姿结果。病例专用的日志和可视化写入
`runs/`、`outputs/`，不复制原始临床影像。

## 运行测试

```bash
cd diffpose/ours
conda run -n mr2ct python -m pytest xmr/tests -q
```

## 下一阶段

1. 接入已标定的 source-detector 几何和可微 MRCP 射线投影。
2. 以 2D 平移结果初始化 6-DoF 3D/2D 位姿优化。
3. 融合多器官分割的投影、边界/距离场和胆管中心线，而非只依赖原始 MRCP 强度。
4. 报告投影 TRE、中心线距离、失败率与不确定性；MIND-SSD 只作为中间优化指标。

## 模块接口

```python
from xmr import mind_ssd, search_translation_mind

result = search_translation_mind(fixed_xray, moving_mrcp_projection, max_shift=32, mask=roi)
```
