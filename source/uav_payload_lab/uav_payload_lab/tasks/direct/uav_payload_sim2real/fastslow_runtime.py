"""Pure runtime and audit utilities for the Sim2Real fast/slow context path.

This module intentionally has no Isaac Lab dependency.  The rollout script uses
the same functions that the focused regression tests exercise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MultirateSchedule:
    policy_dt: float
    policy_hz: float
    slow_warmup_sec: float
    slow_warmup_steps: int
    slow_update_hz: float
    slow_period_steps: int
    fast_update_hz: float
    fast_period_steps: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def validate_evaluation_overrides(
    payload_mass_kg: float | None,
    rope_length_m: float | None,
    disable_wind: bool,
    payload_mass_range: tuple[float, float],
    rope_length_range: tuple[float, float],
) -> dict[str, float | bool | None]:
    """Validate fixed evaluation physics without changing training ranges."""

    def _validate_optional(
        value: float | None,
        bounds: tuple[float, float],
        label: str,
    ) -> float | None:
        if value is None:
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Fixed {label} must be finite, got {value!r}.")
        low, high = (float(bounds[0]), float(bounds[1]))
        if numeric < low or numeric > high:
            raise ValueError(
                f"Fixed {label} {numeric} is outside the training range "
                f"[{low}, {high}]."
            )
        return numeric

    return {
        "payload_mass_kg": _validate_optional(
            payload_mass_kg,
            payload_mass_range,
            "payload mass",
        ),
        "rope_length_m": _validate_optional(
            rope_length_m,
            rope_length_range,
            "rope length",
        ),
        "disable_wind": bool(disable_wind),
    }


def compute_multirate_schedule(
    history_len: int,
    policy_dt: float,
    slow_warmup_sec: float,
    slow_update_hz: float,
    fast_update_hz: float,
) -> MultirateSchedule:
    """Convert requested rates into deterministic integer policy-step periods."""

    if history_len <= 0:
        raise ValueError("history_len must be positive.")
    if policy_dt <= 0.0:
        raise ValueError("policy_dt must be positive.")
    if slow_warmup_sec < 0.0:
        raise ValueError("slow_warmup_sec must be non-negative.")
    if slow_update_hz <= 0.0 or fast_update_hz <= 0.0:
        raise ValueError("Slow and fast update rates must be positive.")

    policy_hz = 1.0 / float(policy_dt)
    if slow_update_hz > policy_hz + 1e-9 or fast_update_hz > policy_hz + 1e-9:
        raise ValueError(
            f"Requested context rate exceeds policy rate: policy={policy_hz:.6g} Hz, "
            f"slow={slow_update_hz:.6g} Hz, fast={fast_update_hz:.6g} Hz."
        )

    warmup_steps = max(int(history_len), int(math.ceil(float(slow_warmup_sec) / policy_dt)))
    slow_period_steps = max(1, int(round(1.0 / (float(slow_update_hz) * policy_dt))))
    fast_period_steps = max(1, int(round(1.0 / (float(fast_update_hz) * policy_dt))))

    return MultirateSchedule(
        policy_dt=float(policy_dt),
        policy_hz=policy_hz,
        slow_warmup_sec=float(slow_warmup_sec),
        slow_warmup_steps=warmup_steps,
        slow_update_hz=float(slow_update_hz),
        slow_period_steps=slow_period_steps,
        fast_update_hz=float(fast_update_hz),
        fast_period_steps=fast_period_steps,
    )


def causal_ema_alpha(policy_dt: float, time_constant_sec: float) -> float:
    """Return the exact discrete coefficient for a first-order causal filter."""

    if policy_dt <= 0.0:
        raise ValueError("policy_dt must be positive.")
    if time_constant_sec < 0.0:
        raise ValueError("time_constant_sec must be non-negative.")
    if time_constant_sec == 0.0:
        return 1.0
    return 1.0 - math.exp(-float(policy_dt) / float(time_constant_sec))


def summarize_latency_ms(values: Iterable[float]) -> dict[str, float | int | None]:
    """Summarize finite latency samples using NumPy's linear percentiles."""

    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    return {
        "count": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
    }


def _as_action_matrix(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected actions with shape (time, action_dim), got {arr.shape}.")
    if arr.shape[1] <= 0:
        raise ValueError("Action dimension must be positive.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Actions contain non-finite values.")
    return arr


def compute_action_total_variation(actions: np.ndarray) -> dict[str, float | list[float] | int]:
    """Compute L1 total variation of the executed CTBR command sequence."""

    arr = _as_action_matrix(actions)
    if arr.shape[0] < 2:
        per_channel = np.zeros(arr.shape[1], dtype=float)
    else:
        per_channel = np.sum(np.abs(np.diff(arr, axis=0)), axis=0)
    total = float(np.sum(per_channel))
    transitions = max(0, int(arr.shape[0] - 1))
    return {
        "samples": int(arr.shape[0]),
        "transitions": transitions,
        "per_channel": [float(x) for x in per_channel],
        "total": total,
        "mean_per_transition": float(total / transitions) if transitions else 0.0,
    }


def compute_action_band_energy(
    actions: np.ndarray,
    sample_rate_hz: float,
    low_hz: float = 5.0,
    high_hz: float = 30.0,
) -> dict[str, float | list[float] | int]:
    """Compute one-sided FFT action energy inside a frequency band.

    The absolute values use the squared-amplitude convention.  Fractional values
    divide band energy by all non-DC energy in the same action channel.
    """

    arr = _as_action_matrix(actions)
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    nyquist = 0.5 * float(sample_rate_hz)
    if low_hz < 0.0 or high_hz < low_hz or high_hz > nyquist + 1e-9:
        raise ValueError(
            f"Invalid band [{low_hz}, {high_hz}] Hz for Nyquist {nyquist} Hz."
        )
    n = int(arr.shape[0])
    if n < 2:
        zeros = [0.0] * int(arr.shape[1])
        return {
            "samples": n,
            "sample_rate_hz": float(sample_rate_hz),
            "low_hz": float(low_hz),
            "high_hz": float(high_hz),
            "per_channel": zeros,
            "total": 0.0,
            "fraction_per_channel": zeros,
            "fraction_total": 0.0,
        }

    centered = arr - np.mean(arr, axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0)
    power = (np.abs(spectrum) / float(n)) ** 2
    if power.shape[0] > 1:
        if n % 2 == 0:
            power[1:-1] *= 2.0
        else:
            power[1:] *= 2.0
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate_hz))
    band = (freqs >= float(low_hz) - 1e-12) & (freqs <= float(high_hz) + 1e-12)
    non_dc = freqs > 0.0

    per_channel = np.sum(power[band], axis=0)
    total_non_dc = np.sum(power[non_dc], axis=0)
    fractions = np.divide(
        per_channel,
        total_non_dc,
        out=np.zeros_like(per_channel),
        where=total_non_dc > 1e-18,
    )
    band_total = float(np.sum(per_channel))
    all_total = float(np.sum(total_non_dc))
    return {
        "samples": n,
        "sample_rate_hz": float(sample_rate_hz),
        "low_hz": float(low_hz),
        "high_hz": float(high_hz),
        "per_channel": [float(x) for x in per_channel],
        "total": band_total,
        "fraction_per_channel": [float(x) for x in fractions],
        "fraction_total": float(band_total / all_total) if all_total > 1e-18 else 0.0,
    }


def compute_gust_response_latency(
    gust: np.ndarray,
    z_fast: np.ndarray,
    policy_dt: float,
    gust_event_threshold: float = 0.05,
    fast_response_threshold: float = 0.05,
    response_window_sec: float = 0.5,
) -> dict[str, object]:
    """Measure an operational gust-to-fast-context response latency.

    A gust event is a step change in the logged piecewise-constant gust vector.
    Response is the first post-event fast-context value whose distance from the
    pre-event value crosses ``fast_response_threshold``.  This is a diagnostic
    channel latency, not a causal closed-loop disturbance-rejection time.
    """

    gust_arr = np.asarray(gust, dtype=float)
    fast_arr = np.asarray(z_fast, dtype=float)
    if gust_arr.ndim != 2 or fast_arr.ndim != 2:
        raise ValueError("gust and z_fast must both be two-dimensional time series.")
    if gust_arr.shape[0] != fast_arr.shape[0]:
        raise ValueError("gust and z_fast must contain the same number of time samples.")
    if policy_dt <= 0.0 or response_window_sec <= 0.0:
        raise ValueError("policy_dt and response_window_sec must be positive.")
    if gust_event_threshold < 0.0 or fast_response_threshold < 0.0:
        raise ValueError("Response thresholds must be non-negative.")
    if not np.all(np.isfinite(gust_arr)) or not np.all(np.isfinite(fast_arr)):
        raise ValueError("gust or z_fast contains non-finite values.")

    if gust_arr.shape[0] < 2:
        event_indices = np.asarray([], dtype=int)
    else:
        gust_delta = np.linalg.norm(np.diff(gust_arr, axis=0), axis=1)
        event_indices = np.flatnonzero(gust_delta >= float(gust_event_threshold)) + 1

    max_steps = max(1, int(math.ceil(float(response_window_sec) / float(policy_dt))))
    latencies: list[float] = []
    responded_events: list[int] = []
    missed_events: list[int] = []
    for event_idx in event_indices.tolist():
        baseline = fast_arr[event_idx - 1]
        stop = min(fast_arr.shape[0], event_idx + max_steps + 1)
        departure = np.linalg.norm(fast_arr[event_idx:stop] - baseline, axis=1)
        hits = np.flatnonzero(departure >= float(fast_response_threshold))
        if hits.size == 0:
            missed_events.append(int(event_idx))
            continue
        hit_idx = int(event_idx + hits[0])
        responded_events.append(int(event_idx))
        latencies.append(float((hit_idx - event_idx) * policy_dt))

    latency_summary = summarize_latency_ms([x * 1000.0 for x in latencies])
    return {
        "definition": (
            "piecewise gust step -> first fast-context departure from the pre-event "
            "value above threshold"
        ),
        "gust_event_threshold_mps2": float(gust_event_threshold),
        "fast_response_threshold": float(fast_response_threshold),
        "response_window_sec": float(response_window_sec),
        "event_count": int(event_indices.size),
        "responded_count": int(len(latencies)),
        "missed_count": int(len(missed_events)),
        "event_indices": [int(x) for x in event_indices.tolist()],
        "responded_event_indices": responded_events,
        "missed_event_indices": missed_events,
        "latencies_s": latencies,
        "latency_mean_s": (
            float(np.mean(np.asarray(latencies, dtype=float))) if latencies else None
        ),
        "latency_p95_s": (
            float(np.percentile(np.asarray(latencies, dtype=float), 95)) if latencies else None
        ),
        "latency_p99_s": (
            float(np.percentile(np.asarray(latencies, dtype=float), 99)) if latencies else None
        ),
        "latency_ms": latency_summary,
    }
