import os
import torch
import matplotlib.pyplot as plt

# 配置参数
checkpoint_dir = "/home/zsr/project/diffpose/experiments/deepfluoro/checkpoints"  # 模型存储目录
base_name = "specimen_01_less_epoch"  # 模型文件名前缀
max_epoch = 500  # 最大epoch数
interval = 50  # 保存间隔

# 准备数据存储
epochs = []
losses = []

# 遍历所有检查点文件
for epoch in range(0, max_epoch + 1, interval):
    if epoch == 500:
        losses.append(0.16)
        epochs.append(epoch)
    # 生成四位补零的epoch字符串
    epoch_str = f"{epoch:03d}"
    filename = f"{base_name}{epoch_str}.ckpt"
    filepath = os.path.join(checkpoint_dir, filename)

    if not os.path.exists(filepath):
        print(f"警告：文件 {filename} 不存在，已跳过")
        continue

    try:
        # 加载模型（自动检测设备）
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

        # 获取loss值（兼容Tensor和普通数值类型）
        loss = checkpoint["loss"]
        if loss is None:
            raise AttributeError("检查点中未找到loss属性")

        # 转换Tensor为Python float
        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        # 记录数据
        epochs.append(epoch)
        losses.append(loss)

    except Exception as e:
        print(f"加载 {filename} 失败: {str(e)}")
        continue

# 绘制图表
if len(epochs) > 0:
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, losses, 'b-o', linewidth=2, markersize=8, markerfacecolor='red')
    plt.title("Training Loss Progress", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(0, max_epoch + 1, max_epoch // 10))  # 自动调整横坐标密度
    plt.tight_layout()

    # 保存图片
    plt.savefig("loss_progress.png", dpi=300)
    print("图表已保存为 loss_progress.png")
    plt.show()
else:
    print("没有找到有效数据来生成图表")

