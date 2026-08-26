"""Small, testable helpers for Phase-II Teacher-to-Student shadow handover."""

from __future__ import annotations

import math

import torch


def validate_shadow_warmup(
    *,
    shadow_warmup_sec: float,
    policy_dt: float,
    history_len: int,
    slow_warmup_sec: float,
    mode: str,
) -> int:
    """Validate the opt-in shadow interval and return its policy-step count."""
    shadow_warmup_sec = float(shadow_warmup_sec)
    policy_dt = float(policy_dt)
    slow_warmup_sec = float(slow_warmup_sec)
    history_len = int(history_len)

    if not math.isfinite(shadow_warmup_sec) or shadow_warmup_sec < 0.0:
        raise ValueError("shadow warmup must be a finite non-negative duration")
    if not math.isfinite(policy_dt) or policy_dt <= 0.0:
        raise ValueError("policy_dt must be finite and positive")
    if history_len <= 0:
        raise ValueError("history_len must be positive")
    if not math.isfinite(slow_warmup_sec) or slow_warmup_sec < 0.0:
        raise ValueError("slow_warmup_sec must be finite and non-negative")
    if shadow_warmup_sec == 0.0:
        return 0
    if mode != "student":
        raise ValueError("positive shadow warmup is only valid in student mode")

    history_fill_sec = history_len * policy_dt
    tolerance = 1.0e-12
    if shadow_warmup_sec + tolerance < history_fill_sec:
        raise ValueError(
            "shadow warmup must fill the complete history window: "
            f"need at least {history_fill_sec:.6f}s"
        )
    if shadow_warmup_sec + tolerance < slow_warmup_sec:
        raise ValueError(
            "shadow warmup must finish the slow startup: "
            f"need at least {slow_warmup_sec:.6f}s"
        )

    return int(math.ceil(shadow_warmup_sec / policy_dt - tolerance))


def teacher_shadow_mask(episode_steps: torch.Tensor, shadow_steps: int) -> torch.Tensor:
    """Return a per-environment mask selecting Teacher control before handover."""
    if not isinstance(episode_steps, torch.Tensor) or episode_steps.ndim != 1:
        raise ValueError("episode_steps must be a one-dimensional torch.Tensor")
    shadow_steps = int(shadow_steps)
    if shadow_steps < 0:
        raise ValueError("shadow_steps must be non-negative")
    return episode_steps < shadow_steps


def select_shadow_actions(
    *,
    student_raw: torch.Tensor,
    student_clipped: torch.Tensor,
    teacher_raw: torch.Tensor,
    teacher_clipped: torch.Tensor,
    episode_steps: torch.Tensor,
    shadow_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select executed actions while retaining Student candidates for audit."""
    shapes = {
        tuple(student_raw.shape),
        tuple(student_clipped.shape),
        tuple(teacher_raw.shape),
        tuple(teacher_clipped.shape),
    }
    if len(shapes) != 1 or student_raw.ndim != 2:
        raise ValueError("all action tensors must share a two-dimensional shape")
    if episode_steps.shape != (student_raw.shape[0],):
        raise ValueError("episode_steps must contain one value per environment")

    mask = teacher_shadow_mask(episode_steps, shadow_steps)
    expanded_mask = mask.unsqueeze(-1)
    executed_raw = torch.where(expanded_mask, teacher_raw, student_raw)
    executed_clipped = torch.where(
        expanded_mask,
        teacher_clipped,
        student_clipped,
    )
    return executed_raw, executed_clipped, mask
