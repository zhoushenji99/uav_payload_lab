#!/usr/bin/env python3
"""Run and hard-gate V8.9 Teacher or Student checkpoints across paired scenarios."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Iterable

import numpy as np


def _metric_failures(summary: dict[str, object], contract: dict[str, object]):
    gates = contract["hard_acceptance_gates"]
    failures = []
    if not bool(summary.get("finite", False)):
        failures.append("no_nan_or_inf")
    if bool(summary.get("early_termination", True)):
        failures.append("no_early_termination")
    position_gate = (
        gates["payload_position_rmse_strong_wind_max_m"]
        if "strong_wind" in str(summary.get("scenario", ""))
        else gates["payload_position_rmse_nominal_max_m"]
    )
    scalar_limits = {
        "payload_position_rmse_m": position_gate,
        "uav_height_mean_abs_error_m": gates["uav_height_mean_abs_error_max_m"],
        "uav_tilt_p95_deg": gates["uav_tilt_p95_max_deg"],
        "uav_tilt_absolute_max_deg": gates["uav_tilt_absolute_max_deg"],
        "actual_body_rate_p95_rad_s": gates["actual_body_rate_p95_max_rad_s"],
        "actual_body_rate_absolute_max_rad_s": gates[
            "actual_body_rate_absolute_max_rad_s"
        ],
        "command_saturation_fraction": gates["command_saturation_fraction_max"],
        "ctbr_tv_mean": gates["command_total_variation_mean_per_step_max"],
        "command_high_frequency_5_30hz_fraction": gates[
            "command_high_frequency_5_30hz_fraction_max"
        ],
        "context_z0_rmse": gates["context_z0_rmse_max"],
        "context_z1_rmse": gates["context_z1_rmse_max"],
        "context_fast_rmse": gates["context_fast_rmse_max"],
        "context_severe_out_of_range_fraction": gates[
            "context_severe_out_of_range_fraction_max"
        ],
    }
    for key, limit in scalar_limits.items():
        value = float(summary.get(key, float("inf")))
        if not math.isfinite(value) or value > float(limit):
            failures.append(key)
    command_p95 = summary.get("command_rate_p95_abs_rad_s", [float("inf")] * 3)
    for axis, (value, limit) in enumerate(
        zip(command_p95, gates["command_rate_p95_abs_max_rad_s"])
    ):
        if not math.isfinite(float(value)) or float(value) > float(limit):
            failures.append(f"command_rate_p95_axis_{axis}")
    return failures


def evaluate_hard_gates(
    summaries: Iterable[dict[str, object]], contract: dict[str, object]
) -> dict[str, object]:
    """Reject a checkpoint if any individual seed/scenario violates any gate."""
    run_verdicts = []
    for summary in summaries:
        failures = _metric_failures(summary, contract)
        run_verdicts.append(
            {
                "seed": int(summary["seed"]),
                "scenario": str(summary["scenario"]),
                "passed": not failures,
                "failures": failures,
            }
        )
    failed = [item for item in run_verdicts if not item["passed"]]
    return {
        "passed": bool(run_verdicts and not failed),
        "failed_seeds": sorted({item["seed"] for item in failed}),
        "failed_scenarios": sorted({item["scenario"] for item in failed}),
        "runs": run_verdicts,
    }


def checkpoint_score(summaries: Iterable[dict[str, object]]) -> float:
    """Apply the immutable V8.9 checkpoint ranking formula."""
    scores = [
        float(item["payload_position_rmse_m"])
        + 0.01 * float(item["swing_rms_deg"])
        + 0.05 * float(item["uav_tilt_p95_deg"])
        + 0.10 * float(item["ctbr_tv_mean"])
        for item in summaries
    ]
    if not scores:
        return float("inf")
    return float(np.mean(scores))


def _matrix(rows: list[dict[str, str]], names: list[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)


def _high_frequency_fraction(actions: np.ndarray, sample_rate_hz: float = 60.0):
    fractions = []
    channels = []
    frequencies = np.fft.rfftfreq(actions.shape[0], d=1.0 / sample_rate_hz)
    denominator_mask = (frequencies >= 0.0) & (frequencies <= 30.0)
    numerator_mask = (frequencies >= 5.0) & (frequencies <= 30.0)
    for index in range(actions.shape[1]):
        values = actions[:, index]
        if float(np.std(values)) <= 0.02:
            channels.append(
                {"channel": index, "status": "not_applicable_pass", "fraction": None}
            )
            continue
        power = np.abs(np.fft.rfft(values - np.mean(values))) ** 2
        denominator = float(np.sum(power[denominator_mask]))
        fraction = float(np.sum(power[numerator_mask]) / max(denominator, 1e-12))
        fractions.append(fraction)
        channels.append({"channel": index, "status": "applicable", "fraction": fraction})
    return (max(fractions) if fractions else 0.0), channels


def summarize_rollout_csv(
    path: str | Path,
    *,
    seed: int,
    scenario: str,
    expected_steps: int,
    contract: dict[str, object],
) -> dict[str, object]:
    csv_path = Path(path)
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"rollout CSV has no rows: {csv_path}")

    numeric_names = [
        name
        for name in rows[0]
        if name not in {"mode", "control_source"}
    ]
    numeric = _matrix(rows, numeric_names)
    finite = bool(np.all(np.isfinite(numeric)))
    payload_error = _matrix(rows, ["payload_err_x", "payload_err_y", "payload_err_z"])
    uav_z = _matrix(rows, ["uav_pz"])[:, 0]
    swing = _matrix(rows, ["theta_x_deg", "theta_y_deg"])
    quaternion = _matrix(rows, ["uav_qw", "uav_qx", "uav_qy", "uav_qz"])
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-12)
    tilt = np.degrees(
        np.arccos(np.clip(1.0 - 2.0 * (quaternion[:, 1] ** 2 + quaternion[:, 2] ** 2), -1.0, 1.0))
    )
    actual_rates = _matrix(
        rows, ["actual_body_rate_x", "actual_body_rate_y", "actual_body_rate_z"]
    )
    actual_rate_norm = np.linalg.norm(actual_rates, axis=1)
    actions = _matrix(rows, ["a0_clamp", "a1_clamp", "a2_clamp", "a3_clamp"])
    rate_p95 = np.percentile(np.abs(actions[:, 1:]), 95, axis=0)
    limits = contract["ctbr_execution_contract"]
    low = np.asarray(limits["absolute_low"], dtype=float)
    high = np.asarray(limits["absolute_high"], dtype=float)
    saturation = np.isclose(actions, low[None, :], atol=1e-6) | np.isclose(
        actions, high[None, :], atol=1e-6
    )
    tv_mean = (
        float(np.abs(np.diff(actions, axis=0)).sum(axis=1).mean())
        if actions.shape[0] > 1
        else 0.0
    )
    hf_fraction, hf_channels = _high_frequency_fraction(actions)
    z_teacher = _matrix(rows, [f"zT{i}" for i in range(5)])
    z_hat = _matrix(rows, [f"zH{i}" for i in range(5)])
    z_error = z_hat - z_teacher
    severe = (
        (z_hat[:, 0] < -0.25)
        | (z_hat[:, 0] > 1.25)
        | (z_hat[:, 1] < -0.25)
        | (z_hat[:, 1] > 1.25)
        | (np.max(np.abs(z_error[:, 2:]), axis=1) > 1.0)
    )
    return {
        "seed": int(seed),
        "scenario": str(scenario),
        "csv_path": str(csv_path.resolve()),
        "samples": len(rows),
        "expected_steps": int(expected_steps),
        "finite": finite,
        "early_termination": len(rows) < int(expected_steps),
        "payload_position_rmse_m": float(np.sqrt(np.mean(np.sum(payload_error**2, axis=1)))),
        "uav_height_mean_abs_error_m": float(np.mean(np.abs(uav_z - 1.5))),
        "swing_rms_deg": float(np.sqrt(np.mean(np.sum(swing**2, axis=1)))),
        "uav_tilt_p95_deg": float(np.percentile(tilt, 95)),
        "uav_tilt_absolute_max_deg": float(np.max(tilt)),
        "actual_body_rate_p95_rad_s": float(np.percentile(actual_rate_norm, 95)),
        "actual_body_rate_absolute_max_rad_s": float(np.max(actual_rate_norm)),
        "command_rate_p95_abs_rad_s": [float(value) for value in rate_p95],
        "command_saturation_fraction": float(np.mean(saturation)),
        "ctbr_tv_mean": tv_mean,
        "command_high_frequency_5_30hz_fraction": hf_fraction,
        "command_high_frequency_channels": hf_channels,
        "context_z0_rmse": float(np.sqrt(np.mean(z_error[:, 0] ** 2))),
        "context_z1_rmse": float(np.sqrt(np.mean(z_error[:, 1] ** 2))),
        "context_fast_rmse": float(np.sqrt(np.mean(z_error[:, 2:] ** 2))),
        "context_severe_out_of_range_fraction": float(np.mean(severe)),
    }


def _scenario_arguments(scenario: str) -> list[str]:
    mapping = {
        "no_ambient_wind": ["--eval_disable_wind"],
        "training_wind": [],
        "training_wind_and_downwash": [],
        "strong_wind_1p25": ["--eval_wind_scale", "1.25"],
        "strong_wind_1p25x": ["--eval_wind_scale", "1.25"],
        "payload_sensor_nominal_gap": ["payload_sensor_tail_probability=0.0"],
        "payload_sensor_tail_gap": ["payload_sensor_tail_probability=1.0"],
        "body_rate_fast_bound": [
            "ctbr_rate_time_constant_range_s=[[0.08,0.08],[0.08,0.08],[0.12,0.12]]"
        ],
        "body_rate_slow_bound": [
            "ctbr_rate_time_constant_range_s=[[0.25,0.25],[0.25,0.25],[0.45,0.45]]"
        ],
        "direct_position_to_student_handover": [],
    }
    if scenario not in mapping:
        raise ValueError(f"unknown V8.9 evaluation scenario: {scenario}")
    return mapping[scenario]


def _checkpoint_candidates(args) -> list[Path]:
    if args.policy_kind == "teacher":
        return [
            Path(path).resolve()
            for path in sorted(
                glob.glob(str(Path(args.checkpoint_root) / args.checkpoint_glob))
            )
        ]
    report = json.loads(Path(args.student_candidates_from_report).read_text())
    return [
        Path(item["path"]).resolve()
        for item in report["top5_checkpoints"][: int(args.student_top_k)]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy-kind", choices=["teacher", "student"], required=True)
    parser.add_argument("--checkpoint-root", default="")
    parser.add_argument("--checkpoint-glob", default="model_*.pt")
    parser.add_argument("--teacher", default="")
    parser.add_argument("--student-root", default="")
    parser.add_argument("--student-candidates-from-report", default="")
    parser.add_argument("--student-top-k", type=int, default=5)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--precontrol", choices=["position", "teacher", "none"], default="position")
    parser.add_argument("--precontrol_sec", type=float, default=3.0)
    parser.add_argument("--student-control-sec", type=float, default=30.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    candidates = _checkpoint_candidates(args)
    if not candidates:
        raise RuntimeError("no checkpoint candidates found")
    for path in candidates:
        if not path.is_file():
            raise RuntimeError(f"candidate does not exist: {path}")
    if args.policy_kind == "student" and not Path(args.teacher).is_file():
        raise RuntimeError("Student evaluation requires an existing --teacher checkpoint")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    play_script = Path(__file__).with_name("play_student_phase2.py")
    isaaclab = Path.home() / "IsaacLab" / "isaaclab.sh"
    checkpoint_results = []
    for checkpoint in candidates:
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        candidate_dir = output / f"{checkpoint.stem}_{digest[:10]}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        run_summaries = []
        for seed in args.seeds:
            for scenario in args.scenarios:
                csv_path = candidate_dir / f"seed_{seed}_{scenario}.csv"
                if args.policy_kind == "teacher":
                    expected_steps = int(args.max_steps)
                    command = [
                        str(isaaclab), "-p", str(play_script),
                        "--task", args.task,
                        "--checkpoint", str(checkpoint),
                        "--mode", "teacher",
                        "--num_envs", "1",
                        "--seed", str(seed),
                        "--max_steps", str(expected_steps),
                        "--csv", str(csv_path),
                        "--headless",
                    ]
                else:
                    precontrol_steps = int(round(float(args.precontrol_sec) * 60.0))
                    expected_steps = precontrol_steps + int(
                        round(float(args.student_control_sec) * 60.0)
                    )
                    command = [
                        str(isaaclab), "-p", str(play_script),
                        "--task", args.task,
                        "--checkpoint", str(Path(args.teacher).resolve()),
                        "--mode", "student",
                        "--encoder", str(checkpoint),
                        "--num_envs", "1",
                        "--seed", str(seed),
                        "--precontrol", args.precontrol,
                        "--precontrol_sec", str(args.precontrol_sec),
                        "--slow_warmup_sec", "3.0",
                        "--slow_update_hz", "1.0",
                        "--fast_update_hz", "60.0",
                        "--slow_filter_tau_sec", "0.25",
                        "--max_steps", str(expected_steps),
                        "--csv", str(csv_path),
                        "--headless",
                    ]
                command.extend(_scenario_arguments(scenario))
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[6],
                    text=True,
                    capture_output=True,
                )
                (candidate_dir / f"seed_{seed}_{scenario}.log").write_text(
                    completed.stdout + "\n" + completed.stderr,
                    encoding="utf-8",
                )
                if completed.returncode != 0 or not csv_path.is_file():
                    run_summaries.append(
                        {
                            **{key: value for key, value in {
                                "seed": seed,
                                "scenario": scenario,
                                "finite": False,
                                "early_termination": True,
                            }.items()},
                            "execution_returncode": completed.returncode,
                        }
                    )
                    continue
                run_summaries.append(
                    summarize_rollout_csv(
                        csv_path,
                        seed=seed,
                        scenario=scenario,
                        expected_steps=expected_steps,
                        contract=contract,
                    )
                )
        verdict = evaluate_hard_gates(run_summaries, contract)
        result = {
            "checkpoint": str(checkpoint),
            "sha256": digest,
            "policy_kind": args.policy_kind,
            "runs": run_summaries,
            "verdict": verdict,
            "passed": verdict["passed"],
            "score": checkpoint_score(run_summaries) if verdict["passed"] else None,
        }
        (candidate_dir / "summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        checkpoint_results.append(result)

    overall = {
        "policy_kind": args.policy_kind,
        "contract": str(Path(args.contract).resolve()),
        "seeds": args.seeds,
        "scenarios": args.scenarios,
        "checkpoints": checkpoint_results,
    }
    (output / "summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(f"[V8.9 Eval] saved {output / 'summary.json'}")


if __name__ == "__main__":
    main()
