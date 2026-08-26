from __future__ import annotations

import torch


def select_control_actions(
    teacher: torch.Tensor,
    position: torch.Tensor,
    position_mask: torch.Tensor,
) -> torch.Tensor:
    if teacher.shape != position.shape:
        raise ValueError("teacher and position actions must have identical shape")
    if position_mask.shape != (teacher.shape[0],):
        raise ValueError("position mask must contain one value per environment")
    return torch.where(position_mask[:, None], position, teacher)
