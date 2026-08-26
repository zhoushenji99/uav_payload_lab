"""Pure tensor helpers for the real-hover Sim2Real gap profile."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def normalize_physical_context(
    payload_mass_kg: torch.Tensor | float,
    rope_length_m: torch.Tensor | float,
    *,
    payload_mass_range_kg: Sequence[float] | torch.Tensor,
    rope_length_range_m: Sequence[float] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize the hard-explicit mass and rope-length context to [0, 1]."""
    mass = torch.as_tensor(payload_mass_kg)
    if not mass.is_floating_point():
        mass = mass.to(dtype=torch.float32)
    length = torch.as_tensor(rope_length_m, device=mass.device, dtype=mass.dtype)
    mass_bounds = torch.as_tensor(
        payload_mass_range_kg, device=mass.device, dtype=mass.dtype
    )
    length_bounds = torch.as_tensor(
        rope_length_range_m, device=mass.device, dtype=mass.dtype
    )
    if (
        mass_bounds.shape != (2,)
        or length_bounds.shape != (2,)
        or not torch.isfinite(mass_bounds).all()
        or not torch.isfinite(length_bounds).all()
        or mass_bounds[1] <= mass_bounds[0]
        or length_bounds[1] <= length_bounds[0]
    ):
        raise ValueError("physical context ranges must be finite increasing pairs")
    z0 = ((mass - mass_bounds[0]) / (mass_bounds[1] - mass_bounds[0])).clamp(0.0, 1.0)
    z1 = ((length - length_bounds[0]) / (length_bounds[1] - length_bounds[0])).clamp(0.0, 1.0)
    return z0, z1


def rate_gain_from_time_constant(
    inertia_diag_kg_m2: torch.Tensor,
    time_constant_s: torch.Tensor,
) -> torch.Tensor:
    """Return the proportional moment gain that realizes I * dw/dt = Kp * e."""
    inertia = torch.as_tensor(inertia_diag_kg_m2)
    tau = torch.as_tensor(time_constant_s, device=inertia.device, dtype=inertia.dtype)
    if inertia.shape != tau.shape or inertia.shape[-1:] != (3,):
        raise ValueError("inertia and time constant must have matching (..., 3) shapes")
    if (
        not torch.isfinite(inertia).all()
        or not torch.isfinite(tau).all()
        or torch.any(inertia <= 0.0)
        or torch.any(tau <= 0.0)
    ):
        raise ValueError("inertia and time constant must be finite and positive")
    return inertia / tau


def compose_lumped_payload_mass(
    rope_length_m: torch.Tensor | float,
    ballast_mass_kg: torch.Tensor | float,
    *,
    rope_length_range_m: Sequence[float] | torch.Tensor,
    rope_mass_range_kg: Sequence[float] | torch.Tensor,
    fixed_moving_mass_kg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lumped payload mass and rope mass correlated with rope length."""
    length_bounds = torch.as_tensor(rope_length_range_m, dtype=torch.float64)
    mass_bounds = torch.as_tensor(rope_mass_range_kg, dtype=torch.float64)
    if (
        length_bounds.shape != (2,)
        or mass_bounds.shape != (2,)
        or not torch.isfinite(length_bounds).all()
        or not torch.isfinite(mass_bounds).all()
        or length_bounds[1] <= length_bounds[0]
        or mass_bounds[0] < 0.0
        or mass_bounds[1] < mass_bounds[0]
        or not math.isfinite(float(fixed_moving_mass_kg))
        or fixed_moving_mass_kg < 0.0
    ):
        raise ValueError("invalid lumped payload mass configuration")

    rope_length = torch.as_tensor(rope_length_m)
    if not rope_length.is_floating_point():
        rope_length = rope_length.to(dtype=torch.float32)
    ballast_mass = torch.as_tensor(
        ballast_mass_kg,
        device=rope_length.device,
        dtype=rope_length.dtype,
    )
    length_lo, length_hi = (float(item) for item in length_bounds)
    mass_lo, mass_hi = (float(item) for item in mass_bounds)
    length_fraction = ((rope_length - length_lo) / (length_hi - length_lo)).clamp(0.0, 1.0)
    rope_mass = mass_lo + length_fraction * (mass_hi - mass_lo)
    total_mass = ballast_mass + float(fixed_moving_mass_kg) + rope_mass
    return total_mass, rope_mass


def _normalized_quadratic_coefficients(
    coefficients: Sequence[float] | torch.Tensor,
) -> tuple[float, float]:
    """Return endpoint-normalized quadratic and linear coefficients.

    The measured polynomial may contain a constant sensor-fit offset.  Thrust
    ratio is therefore defined from ``T(u) - T(0)`` and normalized by
    ``T(1) - T(0)``, which guarantees zero and full thrust at u=0 and u=1.
    """
    coeffs = torch.as_tensor(coefficients, dtype=torch.float64)
    if coeffs.shape != (3,) or not torch.isfinite(coeffs).all():
        raise ValueError("thrust curve must contain three finite coefficients")
    quadratic = float(coeffs[0].item())
    linear = float(coeffs[1].item())
    endpoint_span = quadratic + linear
    if endpoint_span <= 0.0:
        raise ValueError("thrust curve must have positive T(1) - T(0)")
    quadratic /= endpoint_span
    linear /= endpoint_span
    if min(linear, 2.0 * quadratic + linear) < 0.0:
        raise ValueError("thrust curve must be monotonic on [0, 1]")
    return quadratic, linear


def normalized_quadratic_thrust_ratio(
    signal: torch.Tensor | float,
    coefficients: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Map normalized motor signal to endpoint-normalized collective thrust."""
    value = torch.as_tensor(signal)
    if not value.is_floating_point():
        value = value.to(dtype=torch.float32)
    value = value.clamp(0.0, 1.0)
    quadratic, linear = _normalized_quadratic_coefficients(coefficients)
    return (quadratic * value.square() + linear * value).clamp(0.0, 1.0)


def inverse_normalized_quadratic_thrust_ratio(
    thrust_ratio: torch.Tensor | float,
    coefficients: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Invert the normalized collective-thrust curve on the interval [0, 1]."""
    ratio = torch.as_tensor(thrust_ratio)
    if not ratio.is_floating_point():
        ratio = ratio.to(dtype=torch.float32)
    ratio = ratio.clamp(0.0, 1.0)
    quadratic, linear = _normalized_quadratic_coefficients(coefficients)
    if abs(quadratic) < 1e-12:
        return (ratio / linear).clamp(0.0, 1.0)
    discriminant = linear * linear + 4.0 * quadratic * ratio
    signal = (-linear + torch.sqrt(discriminant.clamp_min(0.0))) / (2.0 * quadratic)
    return signal.clamp(0.0, 1.0)


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


def select_delayed_ring(
    ring: torch.Tensor,
    write_index: torch.Tensor,
    delay_steps: torch.Tensor,
) -> torch.Tensor:
    """Select delayed samples from a circular buffer for each environment."""
    if ring.ndim != 3:
        raise ValueError("ring must have shape (num_envs, ring_length, sample_dim)")
    if write_index.shape != (ring.shape[0],) or delay_steps.shape != (ring.shape[0],):
        raise ValueError("write_index and delay_steps must have shape (num_envs,)")
    rows = torch.arange(ring.shape[0], device=ring.device)
    source_index = (
        write_index.to(device=ring.device, dtype=torch.long)
        - delay_steps.to(device=ring.device, dtype=torch.long)
    ) % ring.shape[1]
    return ring[rows, source_index]
