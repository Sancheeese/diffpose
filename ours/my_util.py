import torch

from diffpose.calibration import RigidTransform, convert


def get_random_offset(batch_size: int, device) -> RigidTransform:
    r1 = torch.distributions.Normal(0, 0.25).sample((batch_size,))
    r2 = torch.distributions.Normal(0, 0.1).sample((batch_size,))
    r3 = torch.distributions.Normal(0, 0.2).sample((batch_size,))
    t1 = torch.distributions.Normal(0, 60).sample((batch_size,))
    t2 = torch.distributions.Normal(0, 50).sample((batch_size,))
    t3 = torch.distributions.Normal(0, 8).sample((batch_size,))
    # t3 = -torch.abs(torch.distributions.Normal(0, 10).sample((batch_size,)))
    log_R_vee = torch.stack([r1, r2, r3], dim=1).to(device)
    log_t_vee = torch.stack([t1, t2, t3], dim=1).to(device)
    return convert(
        [log_R_vee, log_t_vee],
        "se3_log_map",
        "se3_exp_map",
    )
