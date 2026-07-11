import torch

from diffpose.calibration import RigidTransform, convert


def get_random_offset(batch_size: int, device) -> RigidTransform:
    # # 矢状面
    # r1 = torch.distributions.Normal(0, 0.00001).sample((batch_size,))
    # # 冠状面
    # r2 = torch.distributions.Normal(0, 0.00001).sample((batch_size,))
    # # 横断面
    # r3 = torch.distributions.Normal(0, 0.3).sample((batch_size,))
    r1 = torch.distributions.Normal(0, torch.pi / 8).sample((batch_size,))
    r2 = torch.distributions.Normal(0, torch.pi / 10).sample((batch_size,))
    # r3 = torch.distributions.Normal(0, torch.pi / 10).sample((batch_size,))
    r3 = torch.distributions.Normal(0, torch.pi / 12).sample((batch_size,))
    # # 前后
    t1 = torch.distributions.Normal(0, 30).sample((batch_size,))
    # 左右
    t2 = torch.distributions.Normal(0, 50).sample((batch_size,))
    # 上下
    t3 = torch.distributions.Normal(0, 30).sample((batch_size,))
    # 前后
    # t1 = torch.distributions.Normal(0, 40).sample((batch_size,))
    # # 左右
    # t2 = torch.distributions.Normal(0, 60).sample((batch_size,))
    # # 上下
    # t3 = torch.distributions.Normal(0, 40).sample((batch_size,))

    # r1 = torch.distributions.Normal(0, torch.pi / 15).sample((batch_size,))
    # r2 = torch.distributions.Normal(0, torch.pi / 20).sample((batch_size,))
    # r3 = torch.distributions.Normal(0, torch.pi / 20).sample((batch_size,))
    # # 前后
    # t1 = torch.distributions.Normal(0, 10).sample((batch_size,))
    # # 左右
    # t2 = torch.distributions.Normal(0, 10).sample((batch_size,))
    # # 上下
    # t3 = torch.distributions.Normal(0, 10).sample((batch_size,))
    log_R_vee = torch.stack([r1, r2, r3], dim=1).to(device)
    log_t_vee = torch.stack([t1, t2, t3], dim=1).to(device)

    isocenter_pose = RigidTransform(
        log_R_vee, log_t_vee, "euler_angles", "ZYX"
    )
    isocenter_pose = isocenter_pose.to(device)

    return isocenter_pose
