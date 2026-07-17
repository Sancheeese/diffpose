"""Intensity variants for separate CoMIR experiments."""

from __future__ import annotations

from torch import Tensor


def invert_legacy_standardized_intensity(images: Tensor, mean: float = 0.3080, std: float = 0.1494) -> Tensor:
    """Invert black and white in the legacy transform's underlying [0, 1] space.

    ``PerImageLegacyTransform`` returns ``(u - mean) / std``. This function
    creates the separately standardized representation of ``1 - u``.
    """
    if std <= 0:
        raise ValueError("std must be positive")
    unit_intensity = images * std + mean
    return (1.0 - unit_intensity - mean) / std
