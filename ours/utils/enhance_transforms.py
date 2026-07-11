
import torch

from torchvision.transforms import Compose, Lambda, Normalize, Resize, GaussianBlur, RandomErasing
from torchvision.transforms import ElasticTransform

class Transforms:
    def __init__(
        self,
        size: int,  # Dimension to resize image
        eps: float = 1e-6,
    ):
        """Transform X-rays and DRRs before inputting to CNN."""
        self.eps = eps
        self.transforms = Compose(
            [
                # Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + eps)),
                Resize((size, size), antialias=True),
                RandomErasing(p=1, scale=(0.03, 0.15), ratio=(0.1, 10.0), value=0, inplace=False),
                ElasticTransform(alpha=60.0, sigma=5.0),
                Normalize(mean=0.3080, std=0.1494),
            ]
        )

        y_coord = torch.arange(size) - size // 2
        x_coord = torch.arange(size) - size // 2
        Y, X = torch.meshgrid(y_coord, x_coord, indexing='ij')
        distance_sq = X ** 2 + Y ** 2  # 使用平方避免开根号
        mask = (distance_sq <= 121 ** 2).float()
        self.mask = mask.unsqueeze(0).unsqueeze(0)

    def __call__(self, x, reverse=True):
        x = (x - x.min()) / (x.max() - x.min() + self.eps)
        if reverse:
            x = 1 - x
        x = self.transforms(x)

        # 计算每个样本的最小值 (B,1,1,1)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        min_values = x.view(x.size(0), -1).min(dim=1)[0][:, None, None, None]

        # 应用蒙版：圆形区域保持原值，外围设为该样本最小值
        mask = self.mask.to(x.device)
        x = x * mask + min_values * (1 - mask)
        # x[x < -1] = min_values

        return x