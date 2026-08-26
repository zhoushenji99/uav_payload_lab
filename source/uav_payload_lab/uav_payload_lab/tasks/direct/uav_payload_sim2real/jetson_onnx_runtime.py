"""Isaac-Sim-free ONNX runtime for the deployable fast-slow Student policy."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_CONTEXT_LOW = (0.0, 0.0, -1.5873020887, -1.6908304691, -1.1491801739)
DEFAULT_CONTEXT_HIGH = (1.0, 1.0, 1.6039054394, 1.4842376709, 1.0464068651)
DEFAULT_CONTEXT_SEVERE_LOW = (-0.25, -0.25, -2.385104, -2.484597, -1.698077)
DEFAULT_CONTEXT_SEVERE_HIGH = (1.25, 1.25, 2.401708, 2.278005, 1.595304)


def causal_ema_alpha(policy_dt: float, tau_sec: float) -> float:
    if policy_dt <= 0.0:
        raise ValueError("policy_dt must be positive")
    if tau_sec < 0.0:
        raise ValueError("tau_sec must be non-negative")
    return 1.0 if tau_sec == 0.0 else 1.0 - math.exp(-policy_dt / tau_sec)


def _session_input_name(session: Any, fallback: str) -> str:
    get_inputs = getattr(session, "get_inputs", None)
    if get_inputs is None:
        return fallback
    inputs = get_inputs()
    return inputs[0].name if inputs else fallback


def create_onnx_session(path: str | Path) -> Any:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "onnxruntime is required on Jetson: python3 -m pip install onnxruntime"
        ) from error
    return ort.InferenceSession(
        str(Path(path).resolve()), providers=["CPUExecutionProvider"]
    )


class FastSlowOnnxRuntime:
    """Stateful single-UAV runtime with explicit history and handover readiness."""

    def __init__(
        self,
        slow_session: Any,
        fast_session: Any,
        actor_session: Any,
        *,
        input_names: Sequence[str] | None = None,
        history_len: int = 50,
        proprio_dim: int = 21,
        policy_hz: float = 60.0,
        slow_warmup_sec: float = 3.0,
        slow_update_hz: float = 1.0,
        slow_filter_tau_sec: float = 0.25,
        max_observation_age_sec: float = 0.1,
        action_low: Sequence[float] = (-1.0, -2.5, -2.5, -1.5),
        action_high: Sequence[float] = (0.0, 2.5, 2.5, 1.5),
        context_low: Sequence[float] = DEFAULT_CONTEXT_LOW,
        context_high: Sequence[float] = DEFAULT_CONTEXT_HIGH,
        context_severe_low: Sequence[float] = DEFAULT_CONTEXT_SEVERE_LOW,
        context_severe_high: Sequence[float] = DEFAULT_CONTEXT_SEVERE_HIGH,
    ):
        self.slow_session = slow_session
        self.fast_session = fast_session
        self.actor_session = actor_session
        defaults = ("history", "history", "actor_input")
        supplied = tuple(input_names) if input_names is not None else defaults
        if len(supplied) != 3:
            raise ValueError("input_names must contain slow, fast, and actor names")
        self.input_names = (
            _session_input_name(slow_session, supplied[0]),
            _session_input_name(fast_session, supplied[1]),
            _session_input_name(actor_session, supplied[2]),
        )
        self.history_len = int(history_len)
        self.proprio_dim = int(proprio_dim)
        self.policy_hz = float(policy_hz)
        self.slow_warmup_steps = int(round(slow_warmup_sec * policy_hz))
        self.slow_period_steps = max(1, int(round(policy_hz / slow_update_hz)))
        self.slow_filter_alpha = causal_ema_alpha(
            1.0 / policy_hz, slow_filter_tau_sec
        )
        self.max_observation_age_sec = float(max_observation_age_sec)
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(4)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(4)
        self.context_low = np.asarray(context_low, dtype=np.float32).reshape(5)
        self.context_high = np.asarray(context_high, dtype=np.float32).reshape(5)
        self.context_severe_low = np.asarray(
            context_severe_low, dtype=np.float32
        ).reshape(5)
        self.context_severe_high = np.asarray(
            context_severe_high, dtype=np.float32
        ).reshape(5)
        self.reset()

    @classmethod
    def from_bundle(cls, bundle_dir: str | Path) -> "FastSlowOnnxRuntime":
        root = Path(bundle_dir).resolve()
        manifest_path = root / "config" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_cfg = manifest["models"]
        runtime_cfg = manifest["runtime"]
        action_cfg = manifest["action"]
        context_cfg = manifest["context_bounds"]
        return cls(
            create_onnx_session(root / model_cfg["slow_encoder"]["onnx"]),
            create_onnx_session(root / model_cfg["fast_encoder"]["onnx"]),
            create_onnx_session(root / model_cfg["actor"]["onnx"]),
            history_len=runtime_cfg["history_len"],
            policy_hz=runtime_cfg["policy_hz"],
            slow_warmup_sec=runtime_cfg["slow_warmup_sec"],
            slow_update_hz=runtime_cfg["slow_update_hz_after_warmup"],
            slow_filter_tau_sec=runtime_cfg["slow_filter_tau_sec"],
            max_observation_age_sec=runtime_cfg["max_observation_age_sec"],
            action_low=action_cfg["low"],
            action_high=action_cfg["high"],
            context_low=context_cfg["low"],
            context_high=context_cfg["high"],
            context_severe_low=context_cfg["severe_low"],
            context_severe_high=context_cfg["severe_high"],
        )

    def reset(self) -> None:
        self.history = np.zeros(
            (1, self.history_len, self.proprio_dim), dtype=np.float32
        )
        self.history_fill_count = 0
        self.valid_step = 0
        self.z_slow_raw = np.zeros(2, dtype=np.float32)
        self.z_slow_target = np.zeros(2, dtype=np.float32)
        self.z_slow_cache = np.zeros(2, dtype=np.float32)
        self.z_fast = np.zeros(3, dtype=np.float32)
        self.slow_call_count = 0
        self.fast_call_count = 0

    def _slow_updates_now(self, step: int) -> bool:
        return step < self.slow_warmup_steps or (
            step >= self.slow_warmup_steps
            and (step - self.slow_warmup_steps) % self.slow_period_steps == 0
        )

    @staticmethod
    def _run(session: Any, input_name: str, value: np.ndarray) -> np.ndarray:
        return np.asarray(session.run(None, {input_name: value})[0], dtype=np.float32)

    def _base_record(
        self, observation: np.ndarray, observation_age_sec: float
    ) -> dict[str, Any]:
        return {
            "observation": observation.copy(),
            "observation_age_sec": float(observation_age_sec),
            "observation_valid": False,
            "reject_reason": "",
            "history_fill_count": self.history_fill_count,
            "valid_step": self.valid_step,
            "shadow_complete": False,
            "candidate_ready": False,
            "slow_updated": False,
            "fast_updated": False,
            "z_slow_raw": self.z_slow_raw.copy(),
            "z_slow_target": self.z_slow_target.copy(),
            "z_slow_cache": self.z_slow_cache.copy(),
            "z_fast": self.z_fast.copy(),
            "context_raw": np.concatenate((self.z_slow_cache, self.z_fast)),
            "context_clamped": np.clip(
                np.concatenate((self.z_slow_cache, self.z_fast)),
                self.context_low,
                self.context_high,
            ),
            "context": np.clip(
                np.concatenate((self.z_slow_cache, self.z_fast)),
                self.context_low,
                self.context_high,
            ),
            "context_out_of_range": False,
            "context_severe_out_of_range": False,
            "action_raw": np.full(4, np.nan, dtype=np.float32),
            "action_clamped": np.full(4, np.nan, dtype=np.float32),
            "previous_executed_ctbr": (
                observation[17:21].copy()
                if observation.shape == (21,)
                else np.full(4, np.nan, dtype=np.float32)
            ),
            "slow_latency_us": 0.0,
            "fast_latency_us": 0.0,
            "actor_latency_us": 0.0,
            "end_to_end_latency_us": 0.0,
        }

    def reject(
        self, reason: str, observation: Sequence[float] | None = None, *, age_sec: float = 0.0
    ) -> dict[str, Any]:
        value = (
            np.asarray(observation, dtype=np.float32).reshape(-1)
            if observation is not None
            else np.full(self.proprio_dim, np.nan, dtype=np.float32)
        )
        record = self._base_record(value, age_sec)
        self.reset()
        record.update(
            reject_reason=str(reason),
            history_fill_count=0,
            valid_step=0,
            shadow_complete=False,
            candidate_ready=False,
        )
        return record

    def step(
        self,
        observation_21: Sequence[float],
        *,
        observation_age_sec: float = 0.0,
    ) -> dict[str, Any]:
        total_start = time.perf_counter_ns()
        observation = np.asarray(observation_21, dtype=np.float32).reshape(-1)
        if observation.shape != (self.proprio_dim,):
            return self.reject(
                "observation_dimension", observation, age_sec=observation_age_sec
            )
        if not np.isfinite(observation).all():
            return self.reject(
                "nonfinite_observation", observation, age_sec=observation_age_sec
            )
        if observation_age_sec > self.max_observation_age_sec:
            return self.reject(
                "stale_observation", observation, age_sec=observation_age_sec
            )

        step = self.valid_step
        self.history[:, :-1, :] = self.history[:, 1:, :]
        self.history[:, -1, :] = observation
        self.history_fill_count = min(self.history_len, self.history_fill_count + 1)

        fast_start = time.perf_counter_ns()
        self.z_fast = self._run(
            self.fast_session, self.input_names[1], self.history
        ).reshape(3)
        fast_end = time.perf_counter_ns()
        self.fast_call_count += 1

        slow_updated = self._slow_updates_now(step)
        slow_us = 0.0
        if slow_updated:
            slow_start = time.perf_counter_ns()
            self.z_slow_raw = self._run(
                self.slow_session, self.input_names[0], self.history
            ).reshape(2)
            slow_us = (time.perf_counter_ns() - slow_start) / 1000.0
            self.z_slow_target = self.z_slow_raw.copy()
            self.slow_call_count += 1

        if step < self.slow_warmup_steps:
            self.z_slow_cache = self.z_slow_target.copy()
        else:
            self.z_slow_cache += self.slow_filter_alpha * (
                self.z_slow_target - self.z_slow_cache
            )

        context_raw = np.concatenate((self.z_slow_raw, self.z_fast)).astype(
            np.float32, copy=False
        )
        context_out_of_range = bool(
            np.any(context_raw < self.context_low)
            or np.any(context_raw > self.context_high)
        )
        context_severe_out_of_range = bool(
            np.any(context_raw < self.context_severe_low)
            or np.any(context_raw > self.context_severe_high)
        )
        actor_context_raw = np.concatenate(
            (self.z_slow_cache, self.z_fast)
        ).astype(np.float32, copy=False)
        context_clamped = np.clip(
            actor_context_raw, self.context_low, self.context_high
        ).astype(np.float32, copy=False)
        actor_input = np.concatenate((observation, context_clamped)).reshape(1, 26)
        actor_start = time.perf_counter_ns()
        action_raw = self._run(
            self.actor_session, self.input_names[2], actor_input
        ).reshape(4)
        actor_end = time.perf_counter_ns()
        action_clamped = np.clip(action_raw, self.action_low, self.action_high)

        self.valid_step += 1
        shadow_complete = self.valid_step >= self.slow_warmup_steps
        candidate_ready = (
            self.history_fill_count >= self.history_len and shadow_complete
            and not context_severe_out_of_range
        )
        record = self._base_record(observation, observation_age_sec)
        record.update(
            observation_valid=True,
            history_fill_count=self.history_fill_count,
            valid_step=self.valid_step,
            shadow_complete=shadow_complete,
            candidate_ready=candidate_ready,
            slow_updated=slow_updated,
            fast_updated=True,
            z_slow_raw=self.z_slow_raw.copy(),
            z_slow_target=self.z_slow_target.copy(),
            z_slow_cache=self.z_slow_cache.copy(),
            z_fast=self.z_fast.copy(),
            context_raw=context_raw.copy(),
            context_clamped=context_clamped.copy(),
            context=context_clamped.copy(),
            context_out_of_range=context_out_of_range,
            context_severe_out_of_range=context_severe_out_of_range,
            action_raw=action_raw.copy(),
            action_clamped=action_clamped.copy(),
            previous_executed_ctbr=observation[17:21].copy(),
            slow_latency_us=slow_us,
            fast_latency_us=(fast_end - fast_start) / 1000.0,
            actor_latency_us=(actor_end - actor_start) / 1000.0,
            end_to_end_latency_us=(time.perf_counter_ns() - total_start) / 1000.0,
        )
        if context_severe_out_of_range:
            record["reject_reason"] = "context_severe_out_of_range"
        return record
