#!/usr/bin/env python3
"""Replay recorded PX4 Position histories through a V8.9 Student and Actor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


try:
    import torch
except ModuleNotFoundError:
    # The immutable command document invokes this file with system Python, while
    # this workstation keeps PyTorch in Isaac Sim's interpreter. Re-exec once so
    # the documented command remains directly runnable.
    if __name__ == "__main__":
        isaac_python = Path.home() / "IsaacLab" / "_isaac_sim" / "python.sh"
        if isaac_python.is_file():
            os.execv(
                str(isaac_python),
                [str(isaac_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            )
    raise

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from ctbr_command_contract import CtbrLimits, shape_ctbr_numpy
from jetson_deployment import (
    DeploymentModels,
    load_deployment_models,
)


HISTORY_LEN = 50
OBSERVATION_DIM = 21


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_history_npy(path: str | Path) -> np.ndarray:
    history = np.asarray(np.load(Path(path), allow_pickle=False), dtype=np.float32)
    if history.shape != (HISTORY_LEN, OBSERVATION_DIM):
        raise ValueError(
            f"real Position history must be [50,21], got {history.shape}"
        )
    if not np.isfinite(history).all():
        raise ValueError("real Position history contains NaN or Inf")
    return history


def append_observation_to_fifo(history: np.ndarray, observation: np.ndarray) -> None:
    """Left-shift then append without ever writing a candidate into obs[17:21]."""
    if history.shape != (HISTORY_LEN, OBSERVATION_DIM):
        raise ValueError(f"FIFO must be [50,21], got {history.shape}")
    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape != (OBSERVATION_DIM,):
        raise ValueError(f"observation must be [21], got {observation.shape}")
    history[:-1] = history[1:]
    history[-1] = observation


def load_trace_position_prefix(
    path: str | Path,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Read only valid Position-shadow rows strictly before candidate-ready."""
    observations: list[np.ndarray] = []
    source_rows: list[dict[str, str]] = []
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if _as_bool(row.get("candidate_ready", False)):
                break
            if not _as_bool(row.get("observation_valid", False)):
                continue
            try:
                observation = np.asarray(
                    [float(row[f"obs_{index}"]) for index in range(OBSERVATION_DIM)],
                    dtype=np.float32,
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(observation).all():
                continue
            observations.append(observation)
            source_rows.append(row)
    if not observations:
        raise RuntimeError(f"no valid pre-handover Position observations in {path}")
    return np.stack(observations), source_rows


def _infer_sequence(
    observations: np.ndarray,
    models: DeploymentModels,
    limits: CtbrLimits,
    *,
    source: str,
    legacy_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    history = np.zeros((HISTORY_LEN, OBSERVATION_DIM), dtype=np.float32)
    low = np.asarray(limits.low, dtype=np.float32)
    high = np.asarray(limits.high, dtype=np.float32)
    max_delta = np.asarray(limits.max_delta, dtype=np.float32)
    records: list[dict[str, object]] = []
    with torch.inference_mode():
        for index, observation in enumerate(observations):
            observation = np.asarray(observation, dtype=np.float32)
            append_observation_to_fifo(history, observation)
            history_tensor = torch.from_numpy(history.copy()).unsqueeze(0)
            z_slow = models.slow_encoder(history_tensor).reshape(2).cpu().numpy()
            z_fast = models.fast_encoder(history_tensor).reshape(3).cpu().numpy()
            z_raw = np.concatenate((z_slow, z_fast)).astype(np.float32)
            # Only the two physical normalized dimensions have immutable bounds
            # in the V8.9 contract. Fast latent dimensions remain unmodified.
            z_clamped = z_raw.copy()
            z_clamped[:2] = np.clip(z_clamped[:2], 0.0, 1.0)
            actor_input = torch.from_numpy(
                np.concatenate((observation, z_clamped)).reshape(1, 26)
            )
            action_raw = models.actor(actor_input).reshape(4).cpu().numpy()
            previous = observation[17:21].copy()
            action_shaped = shape_ctbr_numpy(action_raw, previous, limits)
            delta = action_shaped - previous
            finite = bool(
                np.isfinite(z_raw).all()
                and np.isfinite(action_raw).all()
                and np.isfinite(action_shaped).all()
            )
            absolute_saturation = bool(
                np.any(action_raw < low) or np.any(action_raw > high)
            )
            delta_passed = bool(np.all(np.abs(delta) <= max_delta + 1.0e-6))
            record: dict[str, object] = {
                "source": source,
                "frame_index": index,
                "history_fill_count": min(index + 1, HISTORY_LEN),
                "finite": finite,
                "observation": observation.tolist(),
                "z_raw": z_raw.tolist(),
                "z_clamped": z_clamped.tolist(),
                "previous_ctbr": previous.tolist(),
                "action_raw": action_raw.tolist(),
                "action_shaped": action_shaped.tolist(),
                "delta": delta.tolist(),
                "absolute_saturation": absolute_saturation,
                "delta_contract_passed": delta_passed,
            }
            if legacy_rows is not None:
                legacy = legacy_rows[index]
                for key in (
                    "z_raw0", "z_raw1", "z_raw2", "z_raw3", "z_raw4",
                    "action_raw_thrust", "action_raw_roll_rate",
                    "action_raw_pitch_rate", "action_raw_yaw_rate",
                    "action_clamped_thrust", "action_clamped_roll_rate",
                    "action_clamped_pitch_rate", "action_clamped_yaw_rate",
                ):
                    record[f"legacy_{key}"] = legacy.get(key, "")
            records.append(record)
    if len(observations) == HISTORY_LEN and not np.array_equal(
        history, np.asarray(observations, dtype=np.float32)
    ):
        raise RuntimeError("FIFO order mismatch after the 50th real Position frame")
    return records


def evaluate_replay_gates(
    rows: Iterable[dict[str, object]], contract: dict[str, object]
) -> dict[str, object]:
    rows = list(rows)
    gates = contract["real_position_replay_gates"]
    failures: list[str] = []
    if not rows:
        return {"passed": False, "failures": ["no_full_history_candidate"]}
    finite = all(bool(row["finite"]) for row in rows)
    z1 = np.asarray([row["z_raw"][1] for row in rows], dtype=float)
    actions = np.asarray([row["action_shaped"] for row in rows], dtype=float)
    z1_median = float(np.median(z1))
    z1_error = abs(z1_median - float(gates["rope_z1_expected"]))
    rate_p95 = np.percentile(np.abs(actions[:, 1:]), 95, axis=0)
    any_saturation = any(bool(row["absolute_saturation"]) for row in rows)
    delta_passed = all(bool(row["delta_contract_passed"]) for row in rows)
    if bool(gates["candidate_must_be_finite"]) and not finite:
        failures.append("candidate_must_be_finite")
    if z1_error > float(gates["rope_z1_median_abs_error_max"]):
        failures.append("rope_z1_median_abs_error_max")
    rate_limits = np.asarray(
        [
            gates["roll_rate_p95_abs_max_rad_s"],
            gates["pitch_rate_p95_abs_max_rad_s"],
            gates["yaw_rate_p95_abs_max_rad_s"],
        ],
        dtype=float,
    )
    for axis, (value, limit) in enumerate(zip(rate_p95, rate_limits)):
        if float(value) > float(limit):
            failures.append(f"candidate_rate_p95_axis_{axis}")
    if not bool(gates["any_action_saturation_allowed"]) and any_saturation:
        failures.append("any_action_saturation_allowed")
    if bool(gates["candidate_must_respect_delta_contract"]) and not delta_passed:
        failures.append("candidate_must_respect_delta_contract")
    return {
        "passed": not failures,
        "failures": failures,
        "sample_count": len(rows),
        "z1_median": z1_median,
        "z1_median_abs_error": z1_error,
        "command_rate_p95_abs_rad_s": rate_p95.tolist(),
        "any_action_saturation": any_saturation,
        "all_delta_contract_passed": delta_passed,
        "all_finite": finite,
    }


def _flatten_record(record: dict[str, object]) -> dict[str, object]:
    flat = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "observation", "z_raw", "z_clamped", "previous_ctbr", "action_raw",
            "action_shaped", "delta",
        }
    }
    for name, length in (
        ("observation", 21), ("z_raw", 5), ("z_clamped", 5), ("previous_ctbr", 4),
        ("action_raw", 4), ("action_shaped", 4), ("delta", 4),
    ):
        for index, value in enumerate(record[name]):
            flat[f"{name}_{index}"] = value
    return flat


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    flattened = [_flatten_record(record) for record in records]
    fieldnames = list(flattened[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--history-npy", required=True)
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher = Path(args.teacher).expanduser().resolve()
    student = Path(args.student).expanduser().resolve()
    contract_path = Path(args.contract).expanduser().resolve()
    history_path = Path(args.history_npy).expanduser().resolve()
    trace_path = Path(args.trace_csv).expanduser().resolve()
    for path in (teacher, student, contract_path, history_path, trace_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    limits = CtbrLimits.from_contract(contract_path)
    models, model_metadata = load_deployment_models(teacher, student)
    history = load_history_npy(history_path)
    trace_observations, trace_source_rows = load_trace_position_prefix(trace_path)

    history_records = _infer_sequence(
        history, models, limits, source="fixed_real_position_50H"
    )
    trace_records = _infer_sequence(
        trace_observations,
        models,
        limits,
        source="legacy_crash_position_prefix",
        legacy_rows=trace_source_rows,
    )
    eligible_history = [
        row for row in history_records if row["history_fill_count"] == HISTORY_LEN
    ]
    eligible_trace = [
        row for row in trace_records if row["history_fill_count"] == HISTORY_LEN
    ]
    fixed_history_verdict = evaluate_replay_gates(eligible_history, contract)
    trace_position_verdict = evaluate_replay_gates(eligible_trace, contract)
    verdict = {
        "passed": bool(
            fixed_history_verdict["passed"] and trace_position_verdict["passed"]
        ),
        "cohorts": {
            "fixed_real_position_50H": fixed_history_verdict,
            "legacy_trace_position_prefix": trace_position_verdict,
        },
    }

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_records(output / "real_position_prefix_50H_replay.csv", history_records)
    _write_records(output / "legacy_trace_position_comparison.csv", trace_records)
    summary = {
        "passed": bool(verdict["passed"]),
        "verdict": verdict,
        "model_metadata": model_metadata,
        "inputs": {
            "teacher": str(teacher),
            "teacher_sha256": _sha256(teacher),
            "student": str(student),
            "student_sha256": _sha256(student),
            "contract": str(contract_path),
            "contract_sha256": _sha256(contract_path),
            "history_npy": str(history_path),
            "history_npy_sha256": _sha256(history_path),
            "trace_csv": str(trace_path),
            "trace_csv_sha256": _sha256(trace_path),
        },
        "fixed_history_frames": len(history_records),
        "trace_position_prefix_frames": len(trace_records),
        "full_history_candidate_count": len(eligible_history) + len(eligible_trace),
        "fifo_order": "oldest_to_newest_left_shift_then_append",
        "candidate_feedback_into_observation": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[V8.9 Real Replay] passed={summary['passed']} -> {output}")
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
