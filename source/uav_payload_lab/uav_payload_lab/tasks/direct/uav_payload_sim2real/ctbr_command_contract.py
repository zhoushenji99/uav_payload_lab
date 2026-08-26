from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class CtbrLimits:
    low: tuple[float, float, float, float]
    high: tuple[float, float, float, float]
    max_delta: tuple[float, float, float, float]

    @classmethod
    def from_contract(cls, path: str | Path) -> "CtbrLimits":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))["ctbr_execution_contract"]
        return cls(
            tuple(raw["absolute_low"]),
            tuple(raw["absolute_high"]),
            tuple(raw["max_delta_per_60hz_step"]),
        )


def shape_ctbr_torch(
    target: torch.Tensor,
    previous: torch.Tensor,
    limits: CtbrLimits,
) -> torch.Tensor:
    low = torch.as_tensor(limits.low, dtype=target.dtype, device=target.device)
    high = torch.as_tensor(limits.high, dtype=target.dtype, device=target.device)
    delta = torch.as_tensor(limits.max_delta, dtype=target.dtype, device=target.device)
    absolute = torch.clamp(target, low, high)
    return torch.max(torch.min(absolute, previous + delta), previous - delta)


def shape_ctbr_numpy(
    target: np.ndarray,
    previous: np.ndarray,
    limits: CtbrLimits,
) -> np.ndarray:
    low = np.asarray(limits.low, dtype=target.dtype)
    high = np.asarray(limits.high, dtype=target.dtype)
    delta = np.asarray(limits.max_delta, dtype=target.dtype)
    absolute = np.clip(target, low, high)
    return np.maximum(np.minimum(absolute, previous + delta), previous - delta)
