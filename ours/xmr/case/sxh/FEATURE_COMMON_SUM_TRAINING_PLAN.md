# SXH Sum-Projection Common Feature Training Plan

## 目标

新建一套不影响既有 `max` 版本的训练文件，用同样的 anti-shortcut CoMIR 训练策略，改为使用 MRCP `sum` 投影图训练 X-ray/MRCP 共同特征网络。

已有 `max` 版本保留：

- `train_feature_network_sxh_comir_antishortcut.py`
- `runs/feature_common_comir_antishortcut/`

新增 `sum` 版本计划使用：

- `train_feature_network_sxh_comir_antishortcut_sum.py`
- `runs/feature_common_comir_antishortcut_sum/`

## 保持不变的部分

训练框架保持和当前 anti-shortcut 版本一致：

- 网络：`CoMIRTwoBranchFeatureNetwork`
- 两个独立分支：
  - `xray_net`
  - `mrcp_net`
- 输入图像：`[B, 1, 256, 256]`
- 输出特征：`[B, 32, 256, 256]`
- feature channels：`32`
- patch size：`32 x 32`
- 每张渲染图采样 `24` 对 patch
- batch size：`8` 张渲染图
- 每 step 总 patch 对数：`8 x 24 = 192`
- 每个 query：`1` 个正样本，`23` 个同 batch 内跨模态负样本
- loss：双向 `CrossModalPatchInfoNCE`
  - `xray_to_mrcp`
  - `mrcp_to_xray`
- temperature：`0.07`
- optimizer：`AdamW`
- learning rate：`1e-4`
- weight decay：`1e-4`
- warmup steps：`500`
- min learning rate：`1e-6`
- gradient clip：`5.0`
- AMP：默认 `bf16`
- steps：默认 `100000`
- checkpoint interval：`1000`
- log interval：`10`

## 改动点

### 1. 新建 sum 训练脚本

从当前 max 训练脚本复制一份：

```text
train_feature_network_sxh_comir_antishortcut.py
-> train_feature_network_sxh_comir_antishortcut_sum.py
```

只改 sum 相关内容：

- `DEFAULT_OUTPUT_DIR` 改为：

```text
runs/feature_common_comir_antishortcut_sum
```

- renderer 初始化改为：

```python
SXHFeatureTrainingRenderer(
    projection_mode="sum",
    render_chunk_size=args.render_chunk_size,
    device=args.device,
)
```

- debug 图命名从 `mrcp_max` 改成 `mrcp_sum`

- config 里显式记录：

```json
"mrcp_projection_mode": "sum"
```

### 2. 不改旧 renderer 的默认行为

`SXHFeatureTrainingRenderer` 已经支持：

```python
projection_mode="max" 或 "sum"
```

因此 sum 版本只需要调用时传入 `"sum"`，不需要改旧训练入口。

### 3. 保持 refined MRCP-CT 链路

训练共同特征网络仍然用之前训练 `max` 网络时的 refined CT-MRCP 配准链路，不切换到 original。

原因：

- 训练数据需要同一 pose 下的 CT-DRR 和 MRCP projection 几何对应。
- refined 链路是我们目前最接近“正确配准”的 CT-MRCP 关系。
- original 链路主要用于后续模拟真实优化初始误差，不适合作为正样本训练对。

## 数据生成

每个 step：

1. 以 `xray031` refined pose 为中心。
2. 采样随机位姿扰动，扰动范围沿用 `get_random_offset`：
   - rotation std：
     - `pi/8`
     - `pi/10`
     - `pi/12`
   - translation std：
     - `30 mm`
     - `50 mm`
     - `30 mm`
3. 同一个 perturbed pose 渲染：
   - CT-DRR，作为 X-ray 分支输入
   - MRCP sum projection，作为 MRCP 分支输入
4. 逐张图像做 legacy normalization。
5. 裁剪圆形 FOV 最大内接正方形，再 resize 回 `256 x 256`。
6. 对两个模态独立做 anti-shortcut crop + C4 rotation 增强。
7. 在增强参数映射后的对应位置采样 patch，计算双向 InfoNCE。

## 输出文件

sum 训练输出目录：

```text
diffpose/ours/xmr/case/sxh/runs/feature_common_comir_antishortcut_sum/
```

包含：

- `config.json`
- `train.jsonl`
- `debug/`
  - smoke 时保存 CT-DRR、MRCP sum、裁剪图、增强图、胆管 overlay
- `checkpoints/`
  - `last.pt`
  - `step_001000.pt`
  - `step_002000.pt`
  - ...
  - `step_100000.pt`

## 验证步骤

### 1. 语法检查

```bash
conda run --no-capture-output -n mr2ct \
  python -m py_compile \
  diffpose/ours/xmr/case/sxh/train_feature_network_sxh_comir_antishortcut_sum.py
```

### 2. smoke 运行

```bash
conda run --no-capture-output -n mr2ct \
  python diffpose/ours/xmr/case/sxh/train_feature_network_sxh_comir_antishortcut_sum.py \
  --device cuda:0 \
  --smoke
```

检查：

- 能正常渲染 MRCP sum projection。
- `debug/` 图像存在。
- 输出特征 shape 是 `[8, 32, 256, 256]`。
- loss finite。
- checkpoint 正常保存。

### 3. 正式训练命令

```bash
conda run --no-capture-output -n mr2ct \
  python diffpose/ours/xmr/case/sxh/train_feature_network_sxh_comir_antishortcut_sum.py \
  --device cuda:0
```

## 后续比较

训练完成后，和 max 版本做三类比较：

1. 同一位姿：
   - virtual X-ray feature vs MRCP sum feature
   - real inverted X-ray feature vs MRCP sum feature

2. original 初始位姿配准：
   - virtual X-ray vs MRCP sum projection
   - real inverted X-ray vs MRCP sum projection

3. refined 位姿附近配准：
   - 检查 sum 网络是否比 max 网络更少受局部高亮/边界结构干扰。

## 预期风险

`sum` 投影可能比 `max` 投影更接近 X-ray 的线积分风格，但也可能带来：

- 背景和体厚累计响应更强。
- 胆管细结构对比降低。
- 特征网络更容易关注大范围灰度分布，而不是局部管状结构。

所以训练后需要重点看：

- feature PCA 是否仍然保留胆管/脊柱空间结构。
- real X-ray 和 virtual X-ray 的 feature gap 是否缩小。
- CMA-ES 是否仍然因为整幅图相似度跑向边缘/背景。
