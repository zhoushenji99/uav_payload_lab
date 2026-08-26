"""Isaac-Sim-free reference runtime for the exported fast/slow policy."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch


def causal_ema_alpha(policy_dt: float, tau_sec: float) -> float:
    """Return the causal EMA coefficient used by Phase-II play."""

    if policy_dt <= 0.0:
        raise ValueError(f"policy_dt must be positive, got {policy_dt}")
    if tau_sec < 0.0:
        raise ValueError(f"tau_sec must be non-negative, got {tau_sec}")
    if tau_sec == 0.0:
        return 1.0
    return 1.0 - math.exp(-policy_dt / tau_sec)


class FastSlowRuntime:
    """Stateful single-UAV runtime matching ``play_student_phase2.py``."""

    def __init__(
        self,
        slow_encoder: Any,
        fast_encoder: Any,
        actor: Any,
        *,
        history_len: int = 50,
        proprio_dim: int = 21,
        policy_hz: float = 60.0,
        slow_warmup_sec: float = 3.0,
        slow_update_hz: float = 1.0,
        fast_update_hz: float = 60.0,
        slow_filter_tau_sec: float = 0.25,
        device: str | torch.device = "cpu",
    ):
        if history_len <= 0 or proprio_dim <= 0:
            raise ValueError("history_len and proprio_dim must be positive")
        if policy_hz <= 0.0 or slow_update_hz <= 0.0 or fast_update_hz <= 0.0:
            raise ValueError("policy and context update frequencies must be positive")

        self.slow_encoder = slow_encoder
        self.fast_encoder = fast_encoder
        self.actor = actor
        self.history_len = int(history_len)
        self.proprio_dim = int(proprio_dim)
        self.policy_hz = float(policy_hz)
        self.policy_dt = 1.0 / self.policy_hz
        self.slow_warmup_steps = int(round(float(slow_warmup_sec) * self.policy_hz))
        self.slow_period_steps = max(1, int(round(self.policy_hz / float(slow_update_hz))))
        self.fast_period_steps = max(1, int(round(self.policy_hz / float(fast_update_hz))))
        self.slow_filter_alpha = causal_ema_alpha(
            self.policy_dt, float(slow_filter_tau_sec)
        )
        self.device = torch.device(device)
        self.action_low = torch.tensor(
            [-1.0, -2.5, -2.5, -1.5], dtype=torch.float32, device=self.device
        ).reshape(1, 4)
        self.action_high = torch.tensor(
            [0.0, 2.5, 2.5, 1.5], dtype=torch.float32, device=self.device
        ).reshape(1, 4)
        self.reset()

    def reset(self) -> None:
        self.history = torch.zeros(
            1,
            self.history_len,
            self.proprio_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self.z_slow_raw = torch.zeros(1, 2, device=self.device)
        self.z_slow_target = torch.zeros(1, 2, device=self.device)
        self.z_slow_cache = torch.zeros(1, 2, device=self.device)
        self.z_fast_cache = torch.zeros(1, 3, device=self.device)
        self.episode_step = 0
        self.slow_call_count = 0
        self.fast_call_count = 0

    def _prepare_proprio(self, proprio_21: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(
            proprio_21, dtype=torch.float32, device=self.device
        )
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if tuple(value.shape) != (1, self.proprio_dim):
            raise ValueError(
                f"Expected one proprio sample with shape [21] or [1, 21], got {tuple(value.shape)}"
            )
        return value

    def _slow_updates_now(self) -> bool:
        return self.episode_step < self.slow_warmup_steps or (
            self.episode_step >= self.slow_warmup_steps
            and (self.episode_step - self.slow_warmup_steps)
            % self.slow_period_steps
            == 0
        )

    def step(self, proprio_21: torch.Tensor) -> dict[str, Any]:
        proprio = self._prepare_proprio(proprio_21)
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1, :] = proprio

        fast_updated = self.episode_step % self.fast_period_steps == 0
        slow_updated = self._slow_updates_now()

        with torch.inference_mode():
            if fast_updated:
                self.z_fast_cache = self.fast_encoder(self.history).detach()
                self.fast_call_count += 1

            if slow_updated:
                self.z_slow_raw = self.slow_encoder(self.history).detach()
                self.z_slow_target = self.z_slow_raw.clone()
                self.slow_call_count += 1

            if self.episode_step < self.slow_warmup_steps:
                self.z_slow_cache = self.z_slow_target.clone()
            else:
                self.z_slow_cache = self.z_slow_cache + self.slow_filter_alpha * (
                    self.z_slow_target - self.z_slow_cache
                )

            context = torch.cat([self.z_slow_cache, self.z_fast_cache], dim=-1)
            actor_input = torch.cat([proprio, context], dim=-1)
            action_raw = self.actor(actor_input).detach()
            action_clamped = torch.minimum(
                torch.maximum(action_raw, self.action_low), self.action_high
            )

        result = {
            "episode_step": self.episode_step,
            "history": self.history.clone(),
            "z_slow_raw": self.z_slow_raw.clone(),
            "z_slow_target": self.z_slow_target.clone(),
            "z_slow_cache": self.z_slow_cache.clone(),
            "z_fast_cache": self.z_fast_cache.clone(),
            "context": context.clone(),
            "actor_input": actor_input.clone(),
            "action_raw": action_raw.clone(),
            "action_clamped": action_clamped.clone(),
            "slow_updated": slow_updated,
            "fast_updated": fast_updated,
        }
        self.episode_step += 1
        return result


def load_torchscript_runtime(
    bundle_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> FastSlowRuntime:
    """Load the three standard TorchScript artifacts from a deployment bundle."""

    root = Path(bundle_dir)
    models_root = root / "models" if (root / "models").is_dir() else root
    device_value = torch.device(device)
    required = {
        "slow_encoder": models_root / "slow_encoder.ts",
        "fast_encoder": models_root / "fast_encoder.ts",
        "actor": models_root / "actor.ts",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Deployment bundle is missing TorchScript models: {missing}")
    slow = torch.jit.load(str(required["slow_encoder"]), map_location=device_value).eval()
    fast = torch.jit.load(str(required["fast_encoder"]), map_location=device_value).eval()
    actor = torch.jit.load(str(required["actor"]), map_location=device_value).eval()
    return FastSlowRuntime(slow, fast, actor, device=device_value)
