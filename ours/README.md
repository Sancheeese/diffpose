# `ours` 研究索引

| 路径 | 用途 |
| --- | --- |
| `case/` | 按患者组织的 CT-X-ray/ERCP 配准、训练与评估脚本。 |
| `Bipose/`、`cnnnet/`、`deepfluoro/` | 既有位姿回归和优化实验。 |
| `web_drr_server_nii*.py` | CT/MRCP DRR 与可视化调试服务。 |
| `xmr/` | 新建：MRCP-X-ray 配准模块，含 MIND-SSD 图像空间初始化基线和 `case/sxh/` 首个病例入口。 |

新工作应优先在独立子目录中实现，并在对应 README 中说明输入数据、几何假设、评估指标和运行入口。
