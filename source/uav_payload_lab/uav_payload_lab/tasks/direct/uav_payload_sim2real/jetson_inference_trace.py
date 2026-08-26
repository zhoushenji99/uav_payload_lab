"""Buffered CSV trace for reconstructing Jetson history and policy inference."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ACTION_NAMES = ("thrust", "roll_rate", "pitch_rate", "yaw_rate")
TRACE_FIELDS = [
    "wall_time_ns",
    "monotonic_time_ns",
    "px4_timestamp_us",
    "observation_timestamp_ns",
    "observation_age_sec",
    "observation_valid",
    "reject_reason",
    "history_fill_count",
    "valid_step",
    "shadow_complete",
    "candidate_ready",
    "slow_updated",
    "fast_updated",
    "input_sequence",
    "context_out_of_range",
    "context_severe_out_of_range",
    *[f"obs_{index}" for index in range(21)],
    *[f"z{index}" for index in range(5)],
    *[f"z_raw{index}" for index in range(5)],
    *[f"z_clamped{index}" for index in range(5)],
    "slow_raw_0",
    "slow_raw_1",
    "slow_target_0",
    "slow_target_1",
    "slow_cache_0",
    "slow_cache_1",
    *[f"action_raw_{name}" for name in ACTION_NAMES],
    *[f"action_clamped_{name}" for name in ACTION_NAMES],
    *[f"previous_ctbr_{name}" for name in ACTION_NAMES],
    "slow_latency_us",
    "fast_latency_us",
    "actor_latency_us",
    "end_to_end_latency_us",
]


def flatten_trace_record(record: dict[str, Any]) -> dict[str, Any]:
    row = {name: record.get(name, "") for name in TRACE_FIELDS}
    for index, value in enumerate(record["observation"]):
        row[f"obs_{index}"] = float(value)
    for index, value in enumerate(record["context"]):
        row[f"z{index}"] = float(value)
    for index, value in enumerate(record["context_raw"]):
        row[f"z_raw{index}"] = float(value)
    for index, value in enumerate(record["context_clamped"]):
        row[f"z_clamped{index}"] = float(value)
    for prefix, values in (
        ("slow_raw", record["z_slow_raw"]),
        ("slow_target", record["z_slow_target"]),
        ("slow_cache", record["z_slow_cache"]),
    ):
        for index, value in enumerate(values):
            row[f"{prefix}_{index}"] = float(value)
    for prefix, values in (
        ("action_raw", record["action_raw"]),
        ("action_clamped", record["action_clamped"]),
        ("previous_ctbr", record["previous_executed_ctbr"]),
    ):
        for name, value in zip(ACTION_NAMES, values, strict=True):
            row[f"{prefix}_{name}"] = float(value)
    return row


class InferenceTraceWriter:
    def __init__(self, path: str | Path, *, flush_every: int = 60):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=TRACE_FIELDS)
        self._writer.writeheader()
        self._flush_every = max(1, int(flush_every))
        self._pending = 0

    def write(self, record: dict[str, Any]) -> None:
        self._writer.writerow(flatten_trace_record(record))
        self._pending += 1
        if self._pending >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        self._stream.flush()
        self._pending = 0

    def close(self) -> None:
        if not self._stream.closed:
            self.flush()
            self._stream.close()

    def __enter__(self) -> "InferenceTraceWriter":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()
