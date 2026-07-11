from diffpose.calibration import convert
from diffpose.registration import PoseRegressor
from torch.utils.checkpoint import checkpoint


class DiffPoseModel(PoseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        x = checkpoint(self.backbone, x, use_reentrant=True)
        rot = checkpoint(self.rot_regression, x, use_reentrant=True)
        xyz = checkpoint(self.xyz_regression, x, use_reentrant=True)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

