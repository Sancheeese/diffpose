"""Anti-shortcut CoMIR v2 preprocessing and training components."""

from .augment import AugmentedPair, CropC4Parameters, IndependentCropC4Augment
from .intensity import invert_legacy_standardized_intensity
from .loss import CrossModalPatchInfoNCE, CrossModalPatchLossOutput
from .model import CoMIRFeatureBranch, CoMIRTwoBranchFeatureNetwork
from .preprocess import CanonicalSquareCrop, canonical_square_bounds

__all__ = [
    "AugmentedPair",
    "CanonicalSquareCrop",
    "CoMIRFeatureBranch",
    "CoMIRTwoBranchFeatureNetwork",
    "CropC4Parameters",
    "CrossModalPatchInfoNCE",
    "CrossModalPatchLossOutput",
    "IndependentCropC4Augment",
    "canonical_square_bounds",
    "invert_legacy_standardized_intensity",
]
