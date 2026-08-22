"""Pure tensor helpers for the real-hover Sim2Real gap profile."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def validate_inertia_diagonal(values: Sequence[float] | torch.Tensor) -> torch.Tensor:
    """Validate a diagonal rigid-body inertia and return it as float64."""
    diag = torch.as_tensor(values, dtype=torch.float64)
    if diag.shape != (3,) or not torch.isfinite(diag).all() or torch.any(diag <= 0.0):
        raise ValueError("inertia diagonal must contain three finite positive values")
    tolerance = 1e-9
    if (
        diag[0] + diag[1] + tolerance < diag[2]
        or diag[0] + diag[2] + tolerance < diag[1]
        or diag[1] + diag[2] + tolerance < diag[0]
    ):
        raise ValueError("inertia diagonal violates rigid-body triangle inequality")
    return diag


def diagonal_inertia_flat(
    values: Sequence[float] | torch.Tensor,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return a PhysX-compatible flattened 3x3 diagonal inertia tensor."""
    diag = validate_inertia_diagonal(values).to(device=device, dtype=torch.float32)
    return torch.diag(diag).transpose(0, 1).reshape(9)


def half_sine_profile(elapsed_s: torch.Tensor, duration_s: torch.Tensor) -> torch.Tensor:
    """Smooth startup pulse with zero value at and after both endpoints."""
    safe_duration = duration_s.clamp_min(1e-6)
    phase = math.pi * elapsed_s / safe_duration
    active = (elapsed_s >= 0.0) & (elapsed_s < duration_s)
    return torch.where(active, torch.sin(phase), torch.zeros_like(elapsed_s))


def select_delayed_actions(queue: torch.Tensor, delay_steps: torch.Tensor) -> torch.Tensor:
    """Select one action per row from an oldest-to-newest delay queue."""
    if queue.ndim != 3:
        raise ValueError("queue must have shape (num_envs, max_delay + 1, action_dim)")
    if delay_steps.shape != (queue.shape[0],):
        raise ValueError("delay_steps must have shape (num_envs,)")
    max_delay = queue.shape[1] - 1
    indices = (max_delay - delay_steps.to(device=queue.device, dtype=torch.long)).clamp(0, max_delay)
    rows = torch.arange(queue.shape[0], device=queue.device)
    return queue[rows, indices]
