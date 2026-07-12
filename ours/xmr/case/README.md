# XMR 病例目录

每个患者在此目录下拥有独立子目录：

```text
case/<patient>/
├── README.md       # 数据来源、几何约定、实验状态
├── config.py       # 病例路径与常量（后续加入）
├── runs/           # 配准日志、位姿和指标（不提交大体积中间结果）
└── outputs/        # 投影、叠加图和可视化（不提交原始临床影像）
```

病例脚本必须复用 `xmr` 的通用 MIND/投影接口；不能从 `ours/case/` 复制并修改 CT-X-ray 配准脚本后作为 MRCP-X-ray 方法。
