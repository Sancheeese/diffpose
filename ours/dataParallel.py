import torch
from torch.nn import DataParallel

from diffpose.calibration import RigidTransform


class RigidTransformDataParallel(DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__(module, device_ids, output_device, dim)

    def gather(self, outputs, output_device):
        """自定义 gather 函数"""
        # 假设你希望按批量大小合并每个 GPU 上的输出
        # 你可以在这里修改输出的合并方式，按照你的需求进行自定义
        if isinstance(outputs[0], RigidTransform):
            R_list = []
            t_list = []

            # 合并每个 GPU 上的输出
            for output in outputs:
                R_list.append(output.get_rotation())
                t_list.append(output.get_translation())

            R_list = [r.to(output_device) for r in R_list]
            t_list = [t.to(output_device) for t in t_list]

            # 将旋转矩阵和位移向量合并
            R = torch.cat(R_list, dim=0)
            t = torch.cat(t_list, dim=0)

            return RigidTransform(R=R, t=t, device=outputs[0].device, dtype=outputs[0].dtype)
        else:
            # 默认的 gather 行为
            return super().gather(outputs, output_device)

