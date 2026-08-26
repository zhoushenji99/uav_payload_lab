from __future__ import annotations

import torch


def uav_tilt_rad_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the angle between the body z-axis and world z-axis."""
    normalized = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    _, qx, qy, _ = normalized.unbind(dim=-1)
    body_z_world_z = 1.0 - 2.0 * (qx.square() + qy.square())
    return torch.acos(body_z_world_z.clamp(-1.0, 1.0))


def normalized_ctbr_terms(
    sent: torch.Tensor,
    delta: torch.Tensor,
    jerk: torch.Tensor,
    action_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return sent / action_scale, delta / action_scale, jerk / action_scale
