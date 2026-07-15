"""Differentiable fixed-chain guidewire-to-MRCP centerline registration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from diffpose.calibration import RigidTransform


def _as_points(points: torch.Tensor) -> torch.Tensor:
    points = torch.as_tensor(points, dtype=torch.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected 2D points with shape (N, 2), got {tuple(points.shape)}")
    finite = torch.isfinite(points).all(dim=1)
    points = points[finite]
    if len(points) == 0:
        raise ValueError("Points must contain at least one finite point")
    return points


def fixed_chain_score(guidewire_points: torch.Tensor, chain_points: torch.Tensor, max_distance_px: float = 32.0) -> torch.Tensor:
    """Return a robust one-way guidewire-to-chain distance used before locking a chain."""
    guidewire_points = _as_points(guidewire_points)
    chain_points = _as_points(chain_points).to(guidewire_points)
    distances = torch.cdist(guidewire_points, chain_points).amin(dim=1)
    return distances.clamp(max=max_distance_px).mean()


def select_fixed_chain(guidewire_points: torch.Tensor, projected_chains: list[torch.Tensor]) -> int:
    """Choose the projected bile-duct chain with the smallest initial robust score."""
    valid_scores = [
        (fixed_chain_score(guidewire_points, chain), index)
        for index, chain in enumerate(projected_chains)
        if len(chain) > 0
    ]
    if not valid_scores:
        raise ValueError("No non-empty projected bile-duct chains available")
    return min(valid_scores, key=lambda item: float(item[0].detach()))[1]


def polyline_tangents(points: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Return unit central-difference tangents for an ordered 2D point chain."""
    points = _as_points(points)
    if len(points) == 1:
        return torch.zeros_like(points)
    differences = torch.empty_like(points)
    differences[0] = points[1] - points[0]
    differences[-1] = points[-1] - points[-2]
    if len(points) > 2:
        differences[1:-1] = points[2:] - points[:-2]
    return differences / differences.norm(dim=1, keepdim=True).clamp_min(epsilon)


def soft_correspondence_weights(guidewire_points: torch.Tensor, chain_points: torch.Tensor, temperature_px: float = 2.0) -> torch.Tensor:
    """Return differentiable guidewire-to-chain soft nearest-point weights."""
    if temperature_px <= 0:
        raise ValueError(f"temperature_px must be positive, got {temperature_px}")
    guidewire_points = _as_points(guidewire_points)
    chain_points = _as_points(chain_points).to(guidewire_points)
    squared_distances = torch.cdist(guidewire_points, chain_points).square()
    return torch.softmax(-squared_distances / temperature_px**2, dim=1)


def one_way_position_loss(
    guidewire_points: torch.Tensor,
    chain_points: torch.Tensor,
    *,
    temperature_px: float = 2.0,
    huber_delta_px: float = 4.0,
) -> torch.Tensor:
    """Compute a robust one-way differentiable guidewire-to-chain distance loss."""
    guidewire_points = _as_points(guidewire_points)
    chain_points = _as_points(chain_points).to(guidewire_points)
    weights = soft_correspondence_weights(guidewire_points, chain_points, temperature_px)
    squared_distances = torch.cdist(guidewire_points, chain_points).square()
    soft_squared_distances = (weights * squared_distances).sum(dim=1)
    return (huber_delta_px**2 * (torch.sqrt(1.0 + soft_squared_distances / huber_delta_px**2) - 1.0)).mean()


def tangent_alignment_loss(
    guidewire_points: torch.Tensor,
    chain_points: torch.Tensor,
    *,
    temperature_px: float = 2.0,
) -> torch.Tensor:
    """Penalize nonparallel tangents without requiring known chain direction."""
    guidewire_points = _as_points(guidewire_points)
    chain_points = _as_points(chain_points).to(guidewire_points)
    guidewire_tangents = polyline_tangents(guidewire_points)
    chain_tangents = polyline_tangents(chain_points)
    weights = soft_correspondence_weights(guidewire_points, chain_points, temperature_px)
    matched_tangents = weights @ chain_tangents
    matched_tangents = matched_tangents / matched_tangents.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return (1.0 - (guidewire_tangents * matched_tangents).sum(dim=1).abs()).mean()


@dataclass
class PoseOptimizationResult:
    pose: RigidTransform
    history: list[dict[str, float]]

    @property
    def final_loss(self) -> torch.Tensor:
        return torch.tensor(self.history[-1]["total"])


def optimize_fixed_chain_pose(
    chain_vertices: torch.Tensor,
    guidewire_points: torch.Tensor,
    detector,
    initial_pose: RigidTransform,
    project_points: Callable[[torch.Tensor, RigidTransform, object], torch.Tensor],
    *,
    n_iters: int = 200,
    rotation_lr: float = 1e-4,
    translation_lr: float = 5e-2,
    tangent_weight: float = 2.0,
    pose_weight: float = 1e-3,
) -> PoseOptimizationResult:
    """Optimize a global pose against one fixed 3D bile-duct chain."""
    if n_iters < 1:
        raise ValueError(f"n_iters must be at least 1, got {n_iters}")
    device = chain_vertices.device
    guidewire_points = _as_points(guidewire_points).to(device)
    initial_pose = initial_pose.to(device)
    rotation = torch.nn.Parameter(initial_pose.get_rotation("so3_log_map").detach().clone())
    translation = torch.nn.Parameter(initial_pose.get_translation().detach().clone())
    optimizer = torch.optim.Adam(
        [{"params": [rotation], "lr": rotation_lr}, {"params": [translation], "lr": translation_lr}]
    )
    history: list[dict[str, float]] = []

    for _ in range(n_iters):
        optimizer.zero_grad()
        pose = RigidTransform(rotation, translation, parameterization="so3_log_map")
        projected_chain = project_points(chain_vertices, pose, detector)
        position = one_way_position_loss(guidewire_points, projected_chain)
        tangent = tangent_alignment_loss(guidewire_points, projected_chain)
        relative_log = initial_pose.inverse().compose(pose).get_se3_log()
        pose_penalty = (relative_log[..., :3].square().sum() / 25.0 + relative_log[..., 3:].square().sum() / 0.0025)
        total = position + tangent_weight * tangent + pose_weight * pose_penalty
        total.backward()
        optimizer.step()
        history.append(
            {
                "total": float(total.detach()),
                "position": float(position.detach()),
                "tangent": float(tangent.detach()),
                "pose": float(pose_penalty.detach()),
            }
        )

    final_pose = RigidTransform(rotation.detach(), translation.detach(), parameterization="so3_log_map")
    return PoseOptimizationResult(final_pose, history)
