import torch

from diffpose.calibration import RigidTransform


def pose_to_so4(
        pose: RigidTransform,
        f: float = 1000.0,
) -> torch.Tensor:
    """
    将 RigidTransform (SE(3) 刚体位姿) 转换为对应的近似 SO(4) 矩阵。

    根据论文公式：
        [ R    t/(2f) ]
        [ -t^T R /(2f)   1 ]

    该映射是可微的、数值稳定的，用于计算双不变损失 ℒ_SO(4)。

    Args:
        pose: RigidTransform 对象（包含旋转 R 和平移 t）
        f:    scale factor，通常设为 C-arm 的 source-to-detector distance (mm)。
              常见值 1000~1500，论文中用该距离平衡平移与旋转尺度。

    Returns:
        so4: torch.Tensor [batch, 4, 4]，对应的 SO(4) 矩阵
    """
    R = pose.get_rotation()  # [B, 3, 3]，已经是右手旋转矩阵
    t = pose.get_translation()  # [B, 3]

    B = R.shape[0]
    device = R.device

    # 上半部分 3x4: [R | t/(2f)]
    top_left = R  # [B, 3, 3]
    top_right = t.unsqueeze(-1) / (2.0 * f)  # [B, 3, 1]
    top = torch.cat([top_left, top_right], dim=-1)  # [B, 3, 4]

    # 下半部分 1x4: [-t^T @ R /(2f) | 1]
    # 注意论文中的 -t^T • R，• 表示矩阵乘法
    bottom_left = -torch.bmm(t.unsqueeze(1), R) / (2.0 * f)  # [B, 1, 3]
    bottom_right = torch.ones(B, 1, 1, device=device, dtype=R.dtype)
    bottom = torch.cat([bottom_left, bottom_right], dim=-1)  # [B, 1, 4]

    # 拼接成 4x4
    so4 = torch.cat([top, bottom], dim=1)  # [B, 4, 4]

    return so4