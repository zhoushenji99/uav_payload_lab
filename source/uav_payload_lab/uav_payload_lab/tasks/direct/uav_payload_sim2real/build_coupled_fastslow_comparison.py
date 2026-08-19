"""Build and audit a paired Coupled-RMA versus structured fast/slow package."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .build_fastslow_multiseed_package import (
        audit_fixed_nowind,
        compute_rollout_metrics,
    )
except ImportError:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from build_fastslow_multiseed_package import (
        audit_fixed_nowind,
        compute_rollout_metrics,
    )


EXOGENOUS_COLUMNS = [
    "time_s",
    "rope_length_m",
    "payload_mass_kg",
    "gust_x_mps2",
    "gust_y_mps2",
    "gust_z_mps2",
    "wind_acc_x_mps2",
    "wind_acc_y_mps2",
    "wind_acc_z_mps2",
]
Z_TEACHER_COLUMNS = [f"zT{index}" for index in range(5)]
Z_STUDENT_COLUMNS = [f"zH{index}" for index in range(5)]
POSITION_COLUMNS = ["payload_err_x", "payload_err_y", "payload_err_z"]
SWING_COLUMNS = ["theta_x_deg", "theta_y_deg"]
ACTION_COLUMNS = ["a0_clamp", "a1_clamp", "a2_clamp", "a3_clamp"]
REPOSITORY_ROOT = Path("/home/shenji/uav_payload_lab/uav_payload_lab")
ISAACLAB_LAUNCHER = Path("/home/shenji/IsaacLab/isaaclab.sh")
PLAY_SCRIPT = Path(
    "source/uav_payload_lab/uav_payload_lab/tasks/direct/"
    "uav_payload_sim2real/play_student_phase2.py"
)
COUPLED_MODES = ("teacher", "student")
COUPLED_LABELS = {
    "teacher": "Coupled Teacher",
    "student": "Coupled Student",
}
COMPARISON_METHODS = (
    "structured_teacher",
    "structured_student",
    "coupled_teacher",
    "coupled_student",
)
COMPARISON_LABELS = {
    "structured_teacher": "Structured Teacher",
    "structured_student": "Structured fast/slow Student",
    "coupled_teacher": "Coupled Teacher",
    "coupled_student": "Coupled Student",
}
METHOD_COLORS = {
    "structured_teacher": "#222222",
    "structured_student": "#e27d28",
    "coupled_teacher": "#6a3d9a",
    "coupled_student": "#2a6fbb",
}


def _as_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return frame[list(columns)].to_numpy(dtype=float)


def audit_cross_method_pair(
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    """Require exact scenario inputs across every compared closed-loop rollout."""

    if not frames:
        return {
            "strict_pair": False,
            "common_samples": 0,
            "missing_columns": {},
            "exact_columns": {},
            "max_abs_difference": {},
        }
    missing = {
        method: [column for column in EXOGENOUS_COLUMNS if column not in frame]
        for method, frame in frames.items()
    }
    missing = {method: columns for method, columns in missing.items() if columns}
    if missing:
        return {
            "strict_pair": False,
            "common_samples": 0,
            "missing_columns": missing,
            "exact_columns": {},
            "max_abs_difference": {},
        }

    lengths = {method: int(len(frame)) for method, frame in frames.items()}
    common = min(lengths.values())
    equal_lengths = len(set(lengths.values())) == 1
    reference_name = next(iter(frames))
    reference = frames[reference_name]
    exact: dict[str, dict[str, bool]] = {}
    maximum: dict[str, dict[str, float]] = {}
    for method, frame in frames.items():
        exact[method] = {}
        maximum[method] = {}
        for column in EXOGENOUS_COLUMNS:
            left = reference[column].to_numpy(dtype=float)[:common]
            right = frame[column].to_numpy(dtype=float)[:common]
            exact[method][column] = bool(np.array_equal(left, right))
            maximum[method][column] = (
                float(np.max(np.abs(left - right))) if common else math.nan
            )
    return {
        "strict_pair": bool(
            common > 0
            and equal_lengths
            and not missing
            and all(all(columns.values()) for columns in exact.values())
        ),
        "reference_method": reference_name,
        "common_samples": int(common),
        "lengths": lengths,
        "equal_lengths": bool(equal_lengths),
        "missing_columns": missing,
        "exact_columns": exact,
        "max_abs_difference": maximum,
    }


def compute_context_metrics(
    frame: pd.DataFrame,
    *,
    z_std: Sequence[float],
) -> dict[str, Any]:
    """Measure Student-to-Teacher context error in each method's own scale."""

    scale = np.asarray(z_std, dtype=float)
    if scale.shape != (5,) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("z_std must contain five finite positive values.")
    error = _as_matrix(frame, Z_STUDENT_COLUMNS) - _as_matrix(
        frame, Z_TEACHER_COLUMNS
    )
    rmse_dim = np.sqrt(np.mean(np.square(error), axis=0))
    nrmse_dim = rmse_dim / scale
    return {
        "context_rmse_raw": float(np.sqrt(np.mean(np.square(error)))),
        "context_rmse_dim": [float(value) for value in rmse_dim],
        "context_nrmse": float(np.sqrt(np.mean(np.square(error / scale)))),
        "context_nrmse_dim": [float(value) for value in nrmse_dim],
    }


def compute_deployment_gap(
    teacher: pd.DataFrame,
    student: pd.DataFrame,
) -> dict[str, float | int]:
    """Compute paired closed-loop Teacher-to-Student trace differences."""

    common = min(len(teacher), len(student))
    if common == 0:
        raise ValueError("Teacher and Student rollouts must be non-empty.")

    def rmse(columns: Sequence[str]) -> float:
        delta = (
            _as_matrix(student.iloc[:common], columns)
            - _as_matrix(teacher.iloc[:common], columns)
        )
        return float(np.sqrt(np.mean(np.sum(np.square(delta), axis=1))))

    action_delta = (
        _as_matrix(student.iloc[:common], ACTION_COLUMNS)
        - _as_matrix(teacher.iloc[:common], ACTION_COLUMNS)
    )
    return {
        "common_samples": int(common),
        "position_trace_gap_rmse_m": rmse(POSITION_COLUMNS),
        "swing_trace_gap_rmse_deg": rmse(SWING_COLUMNS),
        "action_trace_gap_rmse": float(
            np.sqrt(np.mean(np.square(action_delta)))
        ),
        "action_trace_gap_l1_mean": float(
            np.mean(np.sum(np.abs(action_delta), axis=1))
        ),
    }


def compute_method_metrics(
    frame: pd.DataFrame,
    *,
    z_std: Sequence[float],
    physical_context: bool,
) -> dict[str, Any]:
    """Compute comparable metrics without assigning false semantics to RMA z."""

    metrics = compute_rollout_metrics(frame)
    metrics.update(compute_context_metrics(frame, z_std=z_std))
    if not physical_context:
        metrics["mass_rmse_kg"] = math.nan
        metrics["rope_length_rmse_m"] = math.nan
    return metrics


def build_rollout_command(
    *,
    mode: str,
    scenario: Mapping[str, Any],
    csv_path: Path,
    checkpoint: Path,
    encoder: Path,
) -> list[str]:
    """Construct one copy-runnable Coupled Teacher or Student evaluation."""

    if mode not in {"teacher", "student"}:
        raise ValueError(f"Unsupported Coupled mode: {mode}")
    command = [
        str(ISAACLAB_LAUNCHER),
        "-p",
        str(PLAY_SCRIPT),
        "--task",
        "Isaac-Uav-Sim2Real-v0",
        "--checkpoint",
        str(checkpoint),
        "--rma_context_mode",
        "monolithic",
        "--num_envs",
        "1",
        "--trace_env",
        "0",
        "--seed",
        str(int(scenario["seed"])),
        "--max_steps",
        "2200",
        "--csv",
        str(csv_path),
        "--headless",
        "--mode",
        mode,
    ]
    if mode == "student":
        command.extend(
            [
                "--encoder",
                str(encoder),
                "--context_runtime_mode",
                "all_60hz",
            ]
        )
    mass = scenario.get("fixed_payload_mass_kg")
    rope = scenario.get("fixed_rope_length_m")
    if mass is not None:
        command.extend(["--eval_payload_mass_kg", str(float(mass))])
    if rope is not None:
        command.extend(["--eval_rope_length_m", str(float(rope))])
    if bool(scenario.get("disable_wind", False)):
        command.append("--eval_disable_wind")
    return command


def build_rollout_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Provide a terminal type accepted by IsaacLab's launcher script."""

    environment = dict(os.environ if base is None else base)
    environment["TERM"] = "xterm"
    return environment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_path(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("*play_summary.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one play summary in {run_dir}, found {len(candidates)}."
        )
    return candidates[0]


def prepare_package(
    *,
    package_dir: Path,
    coupled_run: Path,
    fastslow_package: Path,
) -> dict[str, Any]:
    """Create a provenance manifest using the existing five-scenario contract."""

    package_dir = package_dir.resolve()
    coupled_run = coupled_run.resolve()
    fastslow_package = fastslow_package.resolve()
    teacher_checkpoint = coupled_run / "model_19999.pt"
    student_dir = (
        coupled_run / "StudentMonolithic_sim2real_coupled_seed42_noprobe"
    )
    student_checkpoint = student_dir / "best_monolithic_student_encoder_z.pth"
    student_report = student_dir / "report.json"
    architecture = coupled_run / "context_architecture.json"
    source_manifest = _read_json(
        fastslow_package / "experiment_manifest.json"
    )
    for path in [
        teacher_checkpoint,
        student_checkpoint,
        student_report,
        architecture,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = {
        "package_name": package_dir.name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repository_root": str(REPOSITORY_ROOT),
        "source_run": coupled_run.name,
        "source_boundary": (
            "Only the July 27 monolithic Coupled Teacher and its matching "
            "monolithic Student are allowed."
        ),
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_checkpoint_sha256": _sha256(teacher_checkpoint),
        "student_checkpoint": str(student_checkpoint),
        "student_checkpoint_sha256": _sha256(student_checkpoint),
        "student_report": str(student_report),
        "context_architecture": str(architecture),
        "task": "Isaac-Uav-Sim2Real-v0",
        "teacher_context_mode": "monolithic",
        "student_context_mode": "monolithic",
        "student_runtime": "single history encoder at 60 Hz",
        "num_envs": 1,
        "trace_env": 0,
        "max_steps": 2200,
        "stop_on_done": True,
        "policy_hz": 60.0,
        "modes": list(COUPLED_MODES),
        "scenarios": source_manifest["scenarios"],
        "paired_fastslow_package": str(fastslow_package),
        "comparison_contract": {
            "scenario_source": str(
                fastslow_package / "experiment_manifest.json"
            ),
            "exact_exogenous_columns": EXOGENOUS_COLUMNS,
            "structured_modes": ["teacher", "fastslow"],
            "coupled_modes": list(COUPLED_MODES),
        },
    }
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def run_rollouts(
    *,
    package_dir: Path,
    manifest: Mapping[str, Any],
    force: bool = False,
) -> None:
    """Run the ten Coupled rollouts and preserve commands and console logs."""

    checkpoint = Path(str(manifest["teacher_checkpoint"]))
    encoder = Path(str(manifest["student_checkpoint"]))
    total = len(manifest["scenarios"]) * len(COUPLED_MODES)
    completed = 0
    for scenario in manifest["scenarios"]:
        for mode in COUPLED_MODES:
            completed += 1
            run_dir = package_dir / "raw" / scenario["id"] / mode
            run_dir.mkdir(parents=True, exist_ok=True)
            csv_path = run_dir / "rollout.csv"
            expected_summary = run_dir / (
                "phase2_teacher_play_summary.json"
                if mode == "teacher"
                else "phase2_student_play_summary.json"
            )
            command = build_rollout_command(
                mode=mode,
                scenario=scenario,
                csv_path=csv_path,
                checkpoint=checkpoint,
                encoder=encoder,
            )
            (run_dir / "command.txt").write_text(
                shlex.join(command) + "\n", encoding="utf-8"
            )
            if csv_path.is_file() and expected_summary.is_file() and not force:
                print(
                    f"[SKIP {completed}/{total}] "
                    f"{scenario['id']}/{mode} already exists.",
                    flush=True,
                )
                continue
            print(
                f"[RUN {completed}/{total}] {scenario['id']}/{mode}",
                flush=True,
            )
            environment = build_rollout_environment()
            with (run_dir / "console.log").open(
                "w", encoding="utf-8"
            ) as log_stream:
                result = subprocess.run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Rollout failed ({result.returncode}): "
                    f"{scenario['id']}/{mode}. See {run_dir / 'console.log'}"
                )
            if not csv_path.is_file() or not expected_summary.is_file():
                raise RuntimeError(
                    f"Rollout outputs missing: {scenario['id']}/{mode}"
                )
            print(
                f"[DONE {completed}/{total}] {scenario['id']}/{mode}",
                flush=True,
            )


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _safe_mean(values: Any) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if finite.size else math.nan


def _safe_std(values: Any) -> float:
    finite = _finite(values)
    return float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0


def _aggregate_metrics(
    metrics: pd.DataFrame,
    *,
    group_column: str,
) -> pd.DataFrame:
    numeric = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in {"seed"}
    ]
    rows: list[dict[str, Any]] = []
    for (scenario_group, method), subset in metrics.groupby(
        ["scenario_kind", group_column], sort=False
    ):
        row: dict[str, Any] = {
            "scenario_group": scenario_group,
            group_column: method,
            "count": int(len(subset)),
        }
        for column in numeric:
            row[f"{column}_mean"] = _safe_mean(subset[column])
            row[f"{column}_std"] = _safe_std(subset[column])
        rows.append(row)
    return pd.DataFrame(rows)


def _write_hash_manifest(package_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package_dir)
        if str(relative).startswith("data/sha256_manifest"):
            continue
        rows.append(
            {
                "relative_path": str(relative),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    result = pd.DataFrame(rows)
    data_dir = package_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(data_dir / "sha256_manifest.csv", index=False)
    return result


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _vector_norm(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return np.linalg.norm(_as_matrix(frame, columns), axis=1)


def _method_summary_metrics(
    summary: Mapping[str, Any],
) -> dict[str, float | int]:
    calls = summary.get("context_call_counts", {})
    full_calls = int(calls.get("full_batch_calls", 0))
    slow_calls = int(calls.get("slow_batch_calls", 0))
    fast_calls = int(calls.get("fast_batch_calls", 0))
    response = summary.get("gust_to_fast_context_response", {})
    return {
        "full_batch_calls": full_calls,
        "total_encoder_batch_calls": full_calls + slow_calls + fast_calls,
        "gust_event_count": int(response.get("event_count", 0)),
        "gust_responded_count": int(response.get("responded_count", 0)),
        "gust_response_mean_ms": float(
            response.get("latency_ms", {}).get("mean_ms")
            if response.get("latency_ms", {}).get("mean_ms") is not None
            else math.nan
        ),
        "gust_response_p95_ms": float(
            response.get("latency_ms", {}).get("p95_ms")
            if response.get("latency_ms", {}).get("p95_ms") is not None
            else math.nan
        ),
    }


def _read_rollout(
    package_dir: Path,
    scenario_id: str,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir = package_dir / "raw" / scenario_id / mode
    return pd.read_csv(run_dir / "rollout.csv"), _read_json(
        _summary_path(run_dir)
    )


def _plot_scenario_overview(
    *,
    scenarios: Sequence[Mapping[str, Any]],
    frames: Mapping[tuple[str, str], pd.DataFrame],
    reference_mode: str,
    output: Path,
) -> None:
    labels = [str(scenario["id"]) for scenario in scenarios]
    reference = [
        frames[(str(scenario["id"]), reference_mode)]
        for scenario in scenarios
    ]
    values = [
        [float(frame["payload_mass_kg"].median()) for frame in reference],
        [float(frame["rope_length_m"].median()) for frame in reference],
        [
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            _as_matrix(
                                frame,
                                [
                                    "wind_acc_x_mps2",
                                    "wind_acc_y_mps2",
                                    "wind_acc_z_mps2",
                                ],
                            )
                        )
                    )
                )
            )
            for frame in reference
        ],
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for axis, data, title, ylabel in zip(
        axes,
        values,
        ["Payload mass", "Rope length", "Total-wind component RMS"],
        ["kg", "m", "m/s²"],
    ):
        axis.bar(labels, data, color="#648fff")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, output)


def _plot_multiscenario(
    *,
    scenarios: Sequence[Mapping[str, Any]],
    frames: Mapping[tuple[str, str], pd.DataFrame],
    methods: Sequence[str],
    labels: Mapping[str, str],
    colors: Mapping[str, str],
    value_columns: Sequence[str],
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    rows = int(math.ceil(len(scenarios) / 2))
    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(14, 3.4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat = axes.reshape(-1)
    for axis, scenario in zip(flat, scenarios):
        scenario_id = str(scenario["id"])
        for method in methods:
            frame = frames[(scenario_id, method)]
            axis.plot(
                frame["time_s"],
                _vector_norm(frame, value_columns),
                label=labels[method],
                color=colors[method],
                linewidth=1.0,
            )
        axis.set_title(f"{title}: {scenario_id}")
        axis.set_xlabel("time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    for axis in flat[len(scenarios) :]:
        axis.axis("off")
    flat[0].legend(fontsize=8)
    _save_figure(fig, output)


def _grouped_metric_plot(
    *,
    table: pd.DataFrame,
    scenarios: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    method_column: str,
    labels: Mapping[str, str],
    colors: Mapping[str, str],
    metrics: Sequence[tuple[str, str, str]],
    output: Path,
) -> None:
    columns = 2
    rows = int(math.ceil(len(metrics) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    x = np.arange(len(scenarios), dtype=float)
    width = 0.82 / len(methods)
    for axis, (metric, title, ylabel) in zip(axes.reshape(-1), metrics):
        for index, method in enumerate(methods):
            values = []
            for scenario in scenarios:
                row = table[
                    (table["scenario"] == scenario["id"])
                    & (table[method_column] == method)
                ].iloc[0]
                values.append(float(row[metric]))
            offset = (index - (len(methods) - 1) / 2.0) * width
            axis.bar(
                x + offset,
                values,
                width,
                label=labels[method],
                color=colors[method],
            )
        axis.set_xticks(
            x,
            [str(scenario["id"]) for scenario in scenarios],
            rotation=35,
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    for axis in axes.reshape(-1)[len(metrics) :]:
        axis.axis("off")
    axes.reshape(-1)[0].legend(fontsize=8)
    _save_figure(fig, output)


def _training_report_row(
    *,
    architecture: str,
    report: Mapping[str, Any],
    semantics: str,
) -> dict[str, Any]:
    rmse = np.asarray(report["best_rmse_dim"], dtype=float)
    scale = np.asarray(report["z_std"], dtype=float)
    normalized = rmse / scale
    return {
        "architecture": architecture,
        "teacher_context_mode": report["teacher_context_mode"],
        "student_context_mode": report["student_context_mode"],
        "semantics": semantics,
        "epochs_ran": int(report["epochs_ran"]),
        "best_epoch": int(report["best_epoch"]),
        "reported_best_val": float(report["best_val"]),
        "derived_main_weighted_mse": float(
            np.mean(np.square(normalized))
        ),
        "derived_main_nrmse": float(
            np.sqrt(np.mean(np.square(normalized)))
        ),
        "dims01_nrmse": float(
            np.sqrt(np.mean(np.square(normalized[:2])))
        ),
        "dims234_nrmse": float(
            np.sqrt(np.mean(np.square(normalized[2:])))
        ),
        **{
            f"dim{index}_rmse": float(value)
            for index, value in enumerate(rmse)
        },
        **{
            f"dim{index}_nrmse": float(value)
            for index, value in enumerate(normalized)
        },
        "num_params": int(report["num_params"]),
        "aux_ml_coef": float(report["aux_ml_coef"]),
        "weighted_mse": bool(report["use_weighted_mse"]),
    }


def _relative_percent(proposed: float, baseline: float) -> float:
    if not np.isfinite(proposed) or abs(baseline) < 1e-12:
        return math.nan
    return float(100.0 * (proposed - baseline) / abs(baseline))


def _random_method_mean(
    metrics: pd.DataFrame,
    method: str,
    metric: str,
) -> float:
    values = metrics[
        (metrics["scenario_kind"] == "random_wind")
        & (metrics["method"] == method)
    ][metric]
    return _safe_mean(values)


def _write_conclusions(
    *,
    output: Path,
    metrics: pd.DataFrame,
    gaps: pd.DataFrame,
    training: pd.DataFrame,
    fastslow_internal: pd.DataFrame,
) -> None:
    structured = "structured_student"
    coupled = "coupled_student"

    def comparison(metric: str) -> float:
        return _relative_percent(
            _random_method_mean(metrics, structured, metric),
            _random_method_mean(metrics, coupled, metric),
        )

    def teacher_comparison(metric: str) -> float:
        return _relative_percent(
            _random_method_mean(metrics, "structured_teacher", metric),
            _random_method_mean(metrics, "coupled_teacher", metric),
        )

    random_student = metrics[
        (metrics["scenario_kind"] == "random_wind")
        & metrics["method"].isin([structured, coupled])
    ]

    def lower_count(metric: str) -> int:
        pivot = random_student.pivot(
            index="scenario", columns="method", values=metric
        )
        return int(np.sum(pivot[structured] < pivot[coupled]))

    gap_random = gaps[gaps["scenario_kind"] == "random_wind"]

    def gap_change(metric: str) -> float:
        values = gap_random.groupby("architecture")[metric].mean()
        return _relative_percent(
            float(values["structured"]), float(values["coupled"])
        )

    no_wind = metrics[metrics["scenario_kind"] == "fixed_nowind"].set_index(
        "method"
    )
    training_table = training.set_index("architecture")
    internal_random = fastslow_internal[
        fastslow_internal["scenario_kind"] == "random_wind"
    ].groupby("mode").mean(numeric_only=True)
    internal_latency = _relative_percent(
        float(internal_random.loc["fastslow", "end_to_end_mean_ms"]),
        float(internal_random.loc["all60", "end_to_end_mean_ms"]),
    )
    internal_calls = _relative_percent(
        float(internal_random.loc["fastslow", "slow_batch_calls"]),
        float(internal_random.loc["all60", "slow_batch_calls"]),
    )
    text = f"""# 原始 RMA（Coupled）与结构化快慢方法：结论

## 数据有效性

- 两个方法均使用 20k-iteration Teacher、1000-epoch Student。
- 4 个随机风扰场景和 1 个固定质量/绳长无风场景全部运行 2099 步。
- 四种闭环模式的质量、绳长、gust 和总风扰逐时刻完全相同。
- 这些是 **1 个训练 seed 的策略 + 5 个评估场景**，不是 5 次独立训练。

## 可直接支持的结论

1. **结构化上下文与可解释性成立。** Structured Teacher 的前两维由不可学习
   identity path 严格绑定到归一化质量和绳长；Coupled Teacher 的五维均为学习到的
   混合 latent，不能把 z0/z1 解释为质量和绳长。
2. **结构化 Student 对固定结构量的监督更干净。** 训练集上，Structured 的
   dim0/1 标准化 RMSE 为
   `{training_table.loc['structured', 'dims01_nrmse']:.4f}`，Coupled 的前两维
   latent 标准化 RMSE 为
   `{training_table.loc['coupled', 'dims01_nrmse']:.4f}`。注意 Coupled 前两维没有
   物理语义，这个数字只表示恢复其 Teacher latent 的难度。
3. **快慢调度的计算收益只对“双分支全 60 Hz”成立。** 相对于同一个双 Student
   的 all-60 Hz 消融，快慢调度平均端到端延迟变化 `{internal_latency:+.2f}%`，
   慢分支调用次数变化 `{internal_calls:+.2f}%`，同时保留 60 Hz 快分支。

## 当前数据呈现的性能取舍

- 与 Coupled Student 相比，Structured fast/slow Student 的随机风扰平均位置 RMSE
  变化 `{comparison('position_rmse_norm_m'):+.2f}%`，摆角 RMS 变化
  `{comparison('swing_rms_deg'):+.2f}%`（{lower_count('swing_rms_deg')}/4
  场景更低），摆角峰值变化
  `{comparison('swing_peak_deg'):+.2f}%`（{lower_count('swing_peak_deg')}/4
  场景更低）。
- CTBR 总变差变化 `{comparison('ctbr_total_variation_l1'):+.2f}%`，
  5–30 Hz 动作能量变化 `{comparison('ctbr_5_30hz_energy'):+.2f}%`。
- Teacher→Student 闭环摆角轨迹 gap 变化
  `{gap_change('swing_trace_gap_rmse_deg'):+.2f}%`，位置轨迹 gap 变化
  `{gap_change('position_trace_gap_rmse_m'):+.2f}%`，动作 gap 变化
  `{gap_change('action_trace_gap_rmse'):+.2f}%`。
- Teacher 替换为 Student 后，摆角 RMS 指标的绝对退化量变化
  `{gap_change('absolute_deployment_delta_swing_rms_deg'):+.2f}%`；这支持
  “结构化 Student 更好地保留 Teacher 的摆动性能”，但位置轨迹 gap 没有同步改善。
- 无风固定参数场景中，Structured Student 的标准化上下文误差为
  `{float(no_wind.loc[structured, 'context_nrmse']):.4f}`，Coupled Student 为
  `{float(no_wind.loc[coupled, 'context_nrmse']):.4f}`。
- Structured fast/slow 的平均端到端推理延迟相对 Coupled Student 变化
  `{comparison('end_to_end_mean_ms'):+.2f}%`；它有两个 Student encoder，总参数量
  约为 Coupled monolithic Student 的 2 倍。
- Structured Teacher 相对 Coupled Teacher 已经呈现摆角 RMS
  `{teacher_comparison('swing_rms_deg'):+.2f}%`、摆角峰值
  `{teacher_comparison('swing_peak_deg'):+.2f}%` 的变化。因此当前“摆动更小”的
  证据主要属于结构化 Teacher/Actor 学到的控制策略，不能单独归因于 1 Hz 调度；
  fast/slow Student 的作用是保留这一策略表现。

## 不能写成强结论的内容

- 当前数据不支持“结构化快慢方法在所有飞行指标上全面优于原始 RMA”。
- 当前数据不支持“相对原始 Coupled RMA 计算更少”；计算节省只发生在
  dual-60 Hz → dual fast/slow 这一内部调度消融中。
- Structured 训练的 reported best_val 包含 `aux_ml_coef=0.5`，Coupled 为 0，
  不能直接拿两个 reported best_val 宣称双 Student loss 更低。去除量纲并由
  per-dim RMSE 反算的 main NRMSE 分别为
  `{training_table.loc['structured', 'derived_main_nrmse']:.4f}` 与
  `{training_table.loc['coupled', 'derived_main_nrmse']:.4f}`。
- 真实系统 pipeline 的贡献必须由真实飞行实验支持，本仿真数据不能代替。
- 只有 1 个训练 seed；4 个有风 seed 是评估环境重复，不能替代多训练 seed
  的统计检验。

## 推荐论文表述

本轮最稳妥的主张是：**结构化方法把质量/绳长从不可解释的混合 latent 中剥离，
形成可检查的物理上下文；在严格配对的强风评估中，它以约 1.4% 的位置 RMSE
代价换取一致的摆动降低，Student 基本保留 Teacher 的运输品质；双速率调度则
抵消双编码器全频运行的部分额外开销。**
"""
    output.write_text(text, encoding="utf-8")


def build_analysis(
    *,
    package_dir: Path,
    fastslow_package: Path,
) -> dict[str, Any]:
    """Build Coupled-only artifacts and a strict comparison with fast/slow."""

    package_dir = package_dir.resolve()
    fastslow_package = fastslow_package.resolve()
    manifest = _read_json(package_dir / "experiment_manifest.json")
    fast_manifest = _read_json(
        fastslow_package / "experiment_manifest.json"
    )
    scenarios = list(manifest["scenarios"])
    coupled_report = _read_json(Path(manifest["student_report"]))
    structured_report_path = (
        Path(fast_manifest["student_checkpoint"]).parent / "report.json"
    )
    structured_report = _read_json(structured_report_path)

    data_dir = package_dir / "data"
    figure_dir = package_dir / "figures"
    comparison_dir = package_dir / "对比快慢结构"
    comparison_data = comparison_dir / "data"
    comparison_figures = comparison_dir / "figures"
    for directory in [
        data_dir,
        figure_dir,
        comparison_data,
        comparison_figures,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    coupled_frames: dict[tuple[str, str], pd.DataFrame] = {}
    coupled_summaries: dict[tuple[str, str], dict[str, Any]] = {}
    coupled_rows: list[dict[str, Any]] = []
    coupled_pair_audits: dict[str, Any] = {}
    fixed_audits: dict[str, Any] = {}
    frame_audits: dict[str, Any] = {}
    summary_audits: dict[str, Any] = {}
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        for mode in COUPLED_MODES:
            frame, summary = _read_rollout(package_dir, scenario_id, mode)
            coupled_frames[(scenario_id, mode)] = frame
            coupled_summaries[(scenario_id, mode)] = summary
            numeric = frame.select_dtypes(include=[np.number]).to_numpy()
            frame_audits[f"{scenario_id}/{mode}"] = {
                "rows": int(len(frame)),
                "expected_rows": 2099,
                "row_count_ok": bool(len(frame) == 2099),
                "numeric_nonfinite_count": int(
                    np.size(numeric) - np.isfinite(numeric).sum()
                ),
                "all_numeric_finite": bool(np.isfinite(numeric).all()),
            }
            overrides = summary.get("evaluation_overrides", {})
            expected_overrides = {
                "payload_mass_kg": scenario.get(
                    "fixed_payload_mass_kg"
                ),
                "rope_length_m": scenario.get(
                    "fixed_rope_length_m"
                ),
                "disable_wind": bool(
                    scenario.get("disable_wind", False)
                ),
            }
            actual_overrides = {
                "payload_mass_kg": overrides.get("payload_mass_kg"),
                "rope_length_m": overrides.get("rope_length_m"),
                "disable_wind": bool(
                    overrides.get("disable_wind", False)
                ),
            }
            summary_audits[f"{scenario_id}/{mode}"] = {
                "steps_ok": int(summary.get("steps", -1)) == 2099,
                "teacher_context_mode_ok": (
                    summary.get("teacher_context_mode") == "monolithic"
                ),
                "student_context_mode_ok": (
                    mode == "teacher"
                    or summary.get("student_context_mode") == "monolithic"
                ),
                "overrides_ok": actual_overrides == expected_overrides,
                "expected_overrides": expected_overrides,
                "actual_overrides": actual_overrides,
            }
            metrics = compute_method_metrics(
                frame,
                z_std=coupled_report["z_std"],
                physical_context=False,
            )
            metrics.update(_method_summary_metrics(summary))
            metrics.update(
                {
                    "scenario": scenario_id,
                    "seed": int(scenario["seed"]),
                    "scenario_kind": scenario["kind"],
                    "mode": mode,
                    "method": COUPLED_LABELS[mode],
                }
            )
            coupled_rows.append(metrics)
            if scenario["kind"] == "fixed_nowind":
                fixed_audits[f"{scenario_id}/{mode}"] = audit_fixed_nowind(
                    frame,
                    expected_mass_kg=float(
                        scenario["fixed_payload_mass_kg"]
                    ),
                    expected_rope_length_m=float(
                        scenario["fixed_rope_length_m"]
                    ),
                )
        coupled_pair_audits[scenario_id] = audit_cross_method_pair(
            {
                mode: coupled_frames[(scenario_id, mode)]
                for mode in COUPLED_MODES
            }
        )

    coupled_metrics = pd.DataFrame(coupled_rows)
    coupled_metrics.to_csv(data_dir / "rollout_metrics.csv", index=False)
    coupled_aggregate = _aggregate_metrics(
        coupled_metrics, group_column="mode"
    )
    coupled_aggregate.to_csv(
        data_dir / "aggregate_metrics.csv", index=False
    )
    coupled_gap_rows = []
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        row = compute_deployment_gap(
            coupled_frames[(scenario_id, "teacher")],
            coupled_frames[(scenario_id, "student")],
        )
        row.update(
            {
                "scenario": scenario_id,
                "seed": int(scenario["seed"]),
                "scenario_kind": scenario["kind"],
                "architecture": "coupled",
            }
        )
        coupled_gap_rows.append(row)
    coupled_gaps = pd.DataFrame(coupled_gap_rows)
    coupled_gaps.to_csv(data_dir / "deployment_gap.csv", index=False)

    comparison_frames: dict[tuple[str, str], pd.DataFrame] = {}
    comparison_summaries: dict[tuple[str, str], dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []
    cross_audits: dict[str, Any] = {}
    source_mapping = {
        "structured_teacher": (fastslow_package, "teacher"),
        "structured_student": (fastslow_package, "fastslow"),
        "coupled_teacher": (package_dir, "teacher"),
        "coupled_student": (package_dir, "student"),
    }
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        for method, (root, mode) in source_mapping.items():
            frame, summary = _read_rollout(root, scenario_id, mode)
            comparison_frames[(scenario_id, method)] = frame
            comparison_summaries[(scenario_id, method)] = summary
            structured = method.startswith("structured")
            report = structured_report if structured else coupled_report
            metrics = compute_method_metrics(
                frame,
                z_std=report["z_std"],
                physical_context=structured,
            )
            metrics.update(_method_summary_metrics(summary))
            metrics.update(
                {
                    "scenario": scenario_id,
                    "seed": int(scenario["seed"]),
                    "scenario_kind": scenario["kind"],
                    "method": method,
                    "method_label": COMPARISON_LABELS[method],
                    "architecture": (
                        "structured" if structured else "coupled"
                    ),
                    "role": (
                        "teacher" if method.endswith("teacher") else "student"
                    ),
                }
            )
            comparison_rows.append(metrics)
        cross_audits[scenario_id] = audit_cross_method_pair(
            {
                method: comparison_frames[(scenario_id, method)]
                for method in COMPARISON_METHODS
            }
        )

    comparison_metrics = pd.DataFrame(comparison_rows)
    comparison_metrics.to_csv(
        comparison_data / "comparison_rollout_metrics.csv", index=False
    )
    comparison_aggregate = _aggregate_metrics(
        comparison_metrics, group_column="method"
    )
    comparison_aggregate.to_csv(
        comparison_data / "comparison_aggregate_metrics.csv", index=False
    )
    paired_metrics = [
        "position_rmse_norm_m",
        "position_final_m",
        "swing_rms_deg",
        "swing_peak_deg",
        "swing_exposure_deg_s",
        "context_nrmse",
        "ctbr_total_variation_l1",
        "ctbr_5_30hz_energy",
        "end_to_end_mean_ms",
        "end_to_end_p95_ms",
        "end_to_end_p99_ms",
    ]
    paired_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        for role in ["teacher", "student"]:
            structured_method = f"structured_{role}"
            coupled_method = f"coupled_{role}"
            structured_row = comparison_metrics[
                (comparison_metrics["scenario"] == scenario_id)
                & (comparison_metrics["method"] == structured_method)
            ].iloc[0]
            coupled_row = comparison_metrics[
                (comparison_metrics["scenario"] == scenario_id)
                & (comparison_metrics["method"] == coupled_method)
            ].iloc[0]
            row: dict[str, Any] = {
                "scenario": scenario_id,
                "seed": int(scenario["seed"]),
                "scenario_kind": scenario["kind"],
                "role": role,
            }
            for metric in paired_metrics:
                structured_value = float(structured_row[metric])
                coupled_value = float(coupled_row[metric])
                row[f"structured_{metric}"] = structured_value
                row[f"coupled_{metric}"] = coupled_value
                row[f"delta_{metric}"] = (
                    structured_value - coupled_value
                )
                row[f"relative_change_{metric}"] = _relative_percent(
                    structured_value, coupled_value
                )
            paired_rows.append(row)
    paired_differences = pd.DataFrame(paired_rows)
    paired_differences.to_csv(
        comparison_data / "paired_method_differences.csv", index=False
    )
    paired_summary_rows = []
    for role in ["teacher", "student"]:
        subset = paired_differences[
            (paired_differences["scenario_kind"] == "random_wind")
            & (paired_differences["role"] == role)
        ]
        for metric in paired_metrics:
            deltas = subset[f"delta_{metric}"].to_numpy(dtype=float)
            relative = subset[
                f"relative_change_{metric}"
            ].to_numpy(dtype=float)
            paired_summary_rows.append(
                {
                    "role": role,
                    "metric": metric,
                    "eval_scenarios": int(len(subset)),
                    "mean_delta": _safe_mean(deltas),
                    "std_delta": _safe_std(deltas),
                    "mean_relative_change_percent": _safe_mean(relative),
                    "structured_lower_count": int(
                        np.sum(deltas < 0.0)
                    ),
                    "structured_higher_count": int(
                        np.sum(deltas > 0.0)
                    ),
                }
            )
    pd.DataFrame(paired_summary_rows).to_csv(
        comparison_data / "paired_method_difference_summary.csv",
        index=False,
    )
    gap_rows = []
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        for architecture, teacher_method, student_method in [
            (
                "structured",
                "structured_teacher",
                "structured_student",
            ),
            ("coupled", "coupled_teacher", "coupled_student"),
        ]:
            gap = compute_deployment_gap(
                comparison_frames[(scenario_id, teacher_method)],
                comparison_frames[(scenario_id, student_method)],
            )
            teacher_metrics = comparison_metrics[
                (comparison_metrics["scenario"] == scenario_id)
                & (comparison_metrics["method"] == teacher_method)
            ].iloc[0]
            student_metrics = comparison_metrics[
                (comparison_metrics["scenario"] == scenario_id)
                & (comparison_metrics["method"] == student_method)
            ].iloc[0]
            for metric in [
                "position_rmse_norm_m",
                "swing_rms_deg",
                "swing_peak_deg",
                "ctbr_total_variation_l1",
                "ctbr_5_30hz_energy",
            ]:
                teacher_value = float(teacher_metrics[metric])
                student_value = float(student_metrics[metric])
                gap[f"teacher_{metric}"] = teacher_value
                gap[f"student_{metric}"] = student_value
                gap[f"student_minus_teacher_{metric}"] = (
                    student_value - teacher_value
                )
                gap[f"absolute_deployment_delta_{metric}"] = abs(
                    student_value - teacher_value
                )
            gap.update(
                {
                    "scenario": scenario_id,
                    "seed": int(scenario["seed"]),
                    "scenario_kind": scenario["kind"],
                    "architecture": architecture,
                }
            )
            gap_rows.append(gap)
    comparison_gaps = pd.DataFrame(gap_rows)
    comparison_gaps.to_csv(
        comparison_data / "teacher_student_deployment_gap.csv",
        index=False,
    )

    training = pd.DataFrame(
        [
            _training_report_row(
                architecture="structured",
                report=structured_report,
                semantics=(
                    "dim0/1=normalized mass/rope; dim2-4=residual latent"
                ),
            ),
            _training_report_row(
                architecture="coupled",
                report=coupled_report,
                semantics="all five dimensions are learned mixed latent",
            ),
        ]
    )
    training.to_csv(
        comparison_data / "student_training_context_metrics.csv",
        index=False,
    )

    fastslow_internal = pd.read_csv(
        fastslow_package / "data" / "rollout_metrics.csv"
    )
    fastslow_internal.to_csv(
        comparison_data / "structured_internal_schedule_metrics.csv",
        index=False,
    )
    method_contract = pd.DataFrame(
        [
            {
                "method": "structured_fastslow",
                "teacher": "hard-explicit identity + residual encoder",
                "student": "independent slow and fast encoders",
                "runtime": "3 s all-60 Hz warm-up; then slow 1 Hz + fast 60 Hz",
                "student_parameters": int(structured_report["num_params"]),
                "physical_z0_z1": True,
            },
            {
                "method": "coupled_rma",
                "teacher": "single learned monolithic encoder",
                "student": "single monolithic history encoder",
                "runtime": "single encoder at 60 Hz",
                "student_parameters": int(coupled_report["num_params"]),
                "physical_z0_z1": False,
            },
        ]
    )
    method_contract.to_csv(
        comparison_data / "method_contract.csv", index=False
    )

    audit = {
        "expected_rollouts": int(len(scenarios) * len(COUPLED_MODES)),
        "observed_rollouts": int(len(coupled_metrics)),
        "coupled_teacher_student_pairs": coupled_pair_audits,
        "cross_method_pairs": cross_audits,
        "fixed_nowind": fixed_audits,
        "frames": frame_audits,
        "summaries": summary_audits,
        "checkpoint_hashes": {
            "coupled_teacher": manifest["teacher_checkpoint_sha256"],
            "coupled_student": manifest["student_checkpoint_sha256"],
            "structured_teacher": fast_manifest[
                "teacher_checkpoint_sha256"
            ],
            "structured_student": fast_manifest[
                "student_checkpoint_sha256"
            ],
        },
    }
    audit["passed"] = bool(
        audit["expected_rollouts"] == audit["observed_rollouts"]
        and all(
            item["strict_pair"]
            for item in coupled_pair_audits.values()
        )
        and all(item["strict_pair"] for item in cross_audits.values())
        and all(item["passed"] for item in fixed_audits.values())
        and all(
            item["row_count_ok"] and item["all_numeric_finite"]
            for item in frame_audits.values()
        )
        and all(
            item["steps_ok"]
            and item["teacher_context_mode_ok"]
            and item["student_context_mode_ok"]
            and item["overrides_ok"]
            for item in summary_audits.values()
        )
    )
    (data_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (comparison_data / "paired_inputs_audit.json").write_text(
        json.dumps(
            {
                "passed": bool(
                    all(item["strict_pair"] for item in cross_audits.values())
                ),
                "scenarios": cross_audits,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    coupled_colors = {"teacher": "#6a3d9a", "student": "#2a6fbb"}
    _plot_scenario_overview(
        scenarios=scenarios,
        frames=coupled_frames,
        reference_mode="teacher",
        output=figure_dir / "01_场景物理参数与风扰.png",
    )
    _plot_multiscenario(
        scenarios=scenarios,
        frames=coupled_frames,
        methods=COUPLED_MODES,
        labels=COUPLED_LABELS,
        colors=coupled_colors,
        value_columns=POSITION_COLUMNS,
        title="Payload position error",
        ylabel="error norm (m)",
        output=figure_dir / "02_位置误差多场景.png",
    )
    _plot_multiscenario(
        scenarios=scenarios,
        frames=coupled_frames,
        methods=COUPLED_MODES,
        labels=COUPLED_LABELS,
        colors=coupled_colors,
        value_columns=SWING_COLUMNS,
        title="Payload swing",
        ylabel="swing magnitude (deg)",
        output=figure_dir / "03_摆角多场景.png",
    )
    _grouped_metric_plot(
        table=coupled_metrics,
        scenarios=scenarios,
        methods=COUPLED_MODES,
        method_column="mode",
        labels=COUPLED_LABELS,
        colors=coupled_colors,
        metrics=[
            ("context_nrmse", "Student–Teacher context NRMSE", "NRMSE"),
            ("position_rmse_norm_m", "Position RMSE", "m"),
            ("swing_rms_deg", "Swing RMS", "deg"),
            ("ctbr_total_variation_l1", "CTBR total variation", "L1 TV"),
        ],
        output=figure_dir / "04_上下文恢复与控制指标.png",
    )
    _grouped_metric_plot(
        table=coupled_metrics,
        scenarios=scenarios,
        methods=COUPLED_MODES,
        method_column="mode",
        labels=COUPLED_LABELS,
        colors=coupled_colors,
        metrics=[
            ("ctbr_total_variation_l1", "CTBR total variation", "L1 TV"),
            ("ctbr_5_30hz_energy", "CTBR 5–30 Hz energy", "energy"),
        ],
        output=figure_dir / "05_CTBR动作连续性.png",
    )
    _grouped_metric_plot(
        table=coupled_metrics,
        scenarios=scenarios,
        methods=COUPLED_MODES,
        method_column="mode",
        labels=COUPLED_LABELS,
        colors=coupled_colors,
        metrics=[
            ("end_to_end_mean_ms", "End-to-end mean latency", "ms"),
            ("end_to_end_p95_ms", "End-to-end P95 latency", "ms"),
            ("end_to_end_p99_ms", "End-to-end P99 latency", "ms"),
            (
                "total_encoder_batch_calls",
                "Encoder batch calls",
                "calls",
            ),
        ],
        output=figure_dir / "06_计算开销.png",
    )
    _grouped_metric_plot(
        table=coupled_metrics,
        scenarios=scenarios,
        methods=COUPLED_MODES,
        method_column="mode",
        labels=COUPLED_LABELS,
        colors=coupled_colors,
        metrics=[
            ("position_peak_m", "Position-error peak", "m"),
            ("swing_peak_deg", "Swing peak", "deg"),
            ("position_final_m", "Final position error", "m"),
            ("context_nrmse", "Context NRMSE", "NRMSE"),
        ],
        output=figure_dir / "07_峰值与最终性能.png",
    )

    gap_plot = coupled_gaps.copy()
    gap_plot["mode"] = "student"
    _grouped_metric_plot(
        table=gap_plot,
        scenarios=scenarios,
        methods=("student",),
        method_column="mode",
        labels={"student": "Coupled Student–Teacher gap"},
        colors={"student": "#2a6fbb"},
        metrics=[
            (
                "position_trace_gap_rmse_m",
                "Closed-loop position trace gap",
                "m",
            ),
            (
                "swing_trace_gap_rmse_deg",
                "Closed-loop swing trace gap",
                "deg",
            ),
            ("action_trace_gap_rmse", "Closed-loop action gap", "RMSE"),
            (
                "action_trace_gap_l1_mean",
                "Closed-loop action L1 gap",
                "mean L1",
            ),
        ],
        output=figure_dir / "08_Teacher到Student部署差距.png",
    )
    fixed_scenario = next(
        scenario
        for scenario in scenarios
        if scenario["kind"] == "fixed_nowind"
    )
    fixed_scenario_id = str(fixed_scenario["id"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for mode in COUPLED_MODES:
        frame = coupled_frames[(fixed_scenario_id, mode)]
        axes[0, 0].plot(
            frame["time_s"],
            _vector_norm(frame, POSITION_COLUMNS),
            label=COUPLED_LABELS[mode],
            color=coupled_colors[mode],
        )
        axes[0, 1].plot(
            frame["time_s"],
            _vector_norm(frame, SWING_COLUMNS),
            label=COUPLED_LABELS[mode],
            color=coupled_colors[mode],
        )
        axes[1, 0].plot(
            frame["time_s"],
            np.linalg.norm(
                (
                    _as_matrix(frame, Z_STUDENT_COLUMNS)
                    - _as_matrix(frame, Z_TEACHER_COLUMNS)
                )
                / np.asarray(coupled_report["z_std"], dtype=float),
                axis=1,
            ),
            label=COUPLED_LABELS[mode],
            color=coupled_colors[mode],
        )
        axes[1, 1].plot(
            frame["time_s"],
            _vector_norm(frame, ACTION_COLUMNS),
            label=COUPLED_LABELS[mode],
            color=coupled_colors[mode],
        )
    for axis, title, ylabel in [
        (axes[0, 0], "No-wind position error", "m"),
        (axes[0, 1], "No-wind swing", "deg"),
        (axes[1, 0], "No-wind context error", "normalized norm"),
        (axes[1, 1], "No-wind CTBR action norm", "action norm"),
    ]:
        axis.set_title(title)
        axis.set_xlabel("time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    _save_figure(fig, figure_dir / "09_无风中等参数案例.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(
        np.arange(1, len(coupled_report["train_hist"]) + 1),
        coupled_report["train_hist"],
        label="train",
    )
    axes[0].plot(
        np.arange(1, len(coupled_report["val_hist"]) + 1),
        coupled_report["val_hist"],
        label="validation",
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Coupled Student training")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("weighted MSE")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].bar(
        [f"z{index}" for index in range(5)],
        np.asarray(coupled_report["best_rmse_dim"], dtype=float)
        / np.asarray(coupled_report["z_std"], dtype=float),
        color="#2a6fbb",
    )
    axes[1].set_title("Best per-dimension normalized RMSE")
    axes[1].set_ylabel("NRMSE")
    axes[1].grid(axis="y", alpha=0.25)
    _save_figure(fig, figure_dir / "10_Student训练曲线与恢复误差.png")

    _plot_multiscenario(
        scenarios=scenarios,
        frames=comparison_frames,
        methods=COMPARISON_METHODS,
        labels=COMPARISON_LABELS,
        colors=METHOD_COLORS,
        value_columns=POSITION_COLUMNS,
        title="Strictly paired position error",
        ylabel="error norm (m)",
        output=comparison_figures / "01_四方法位置误差.png",
    )
    _plot_multiscenario(
        scenarios=scenarios,
        frames=comparison_frames,
        methods=COMPARISON_METHODS,
        labels=COMPARISON_LABELS,
        colors=METHOD_COLORS,
        value_columns=SWING_COLUMNS,
        title="Strictly paired payload swing",
        ylabel="swing magnitude (deg)",
        output=comparison_figures / "02_四方法摆角.png",
    )
    student_methods = ("structured_student", "coupled_student")
    _grouped_metric_plot(
        table=comparison_metrics,
        scenarios=scenarios,
        methods=student_methods,
        method_column="method",
        labels=COMPARISON_LABELS,
        colors=METHOD_COLORS,
        metrics=[
            ("position_rmse_norm_m", "Student position RMSE", "m"),
            ("swing_rms_deg", "Student swing RMS", "deg"),
            ("swing_peak_deg", "Student swing peak", "deg"),
            ("position_final_m", "Student final error", "m"),
        ],
        output=comparison_figures / "03_Student飞行性能.png",
    )
    gap_colors = {"structured": "#e27d28", "coupled": "#2a6fbb"}
    gap_labels = {
        "structured": "Structured Teacher→Student",
        "coupled": "Coupled Teacher→Student",
    }
    _grouped_metric_plot(
        table=comparison_gaps,
        scenarios=scenarios,
        methods=("structured", "coupled"),
        method_column="architecture",
        labels=gap_labels,
        colors=gap_colors,
        metrics=[
            (
                "position_trace_gap_rmse_m",
                "Position trace deployment gap",
                "m",
            ),
            (
                "swing_trace_gap_rmse_deg",
                "Swing trace deployment gap",
                "deg",
            ),
            ("action_trace_gap_rmse", "Action deployment gap", "RMSE"),
            (
                "action_trace_gap_l1_mean",
                "Action deployment L1 gap",
                "mean L1",
            ),
        ],
        output=comparison_figures / "04_Teacher到Student部署差距.png",
    )
    _grouped_metric_plot(
        table=comparison_metrics,
        scenarios=scenarios,
        methods=student_methods,
        method_column="method",
        labels=COMPARISON_LABELS,
        colors=METHOD_COLORS,
        metrics=[
            (
                "context_nrmse",
                "Scale-normalized Student–Teacher context error",
                "NRMSE",
            ),
            (
                "ctbr_total_variation_l1",
                "CTBR total variation",
                "L1 TV",
            ),
            (
                "ctbr_5_30hz_energy",
                "CTBR 5–30 Hz energy",
                "energy",
            ),
            ("swing_exposure_deg_s", "Swing exposure", "deg·s"),
        ],
        output=comparison_figures / "05_上下文恢复与动作连续性.png",
    )
    _grouped_metric_plot(
        table=comparison_metrics,
        scenarios=scenarios,
        methods=student_methods,
        method_column="method",
        labels=COMPARISON_LABELS,
        colors=METHOD_COLORS,
        metrics=[
            ("end_to_end_mean_ms", "End-to-end mean latency", "ms"),
            ("end_to_end_p95_ms", "End-to-end P95 latency", "ms"),
            ("end_to_end_p99_ms", "End-to-end P99 latency", "ms"),
            (
                "total_encoder_batch_calls",
                "Encoder batch calls",
                "calls",
            ),
        ],
        output=comparison_figures / "06_相对原始RMA计算开销.png",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    x = np.arange(5)
    width = 0.36
    for index, architecture in enumerate(["structured", "coupled"]):
        row = training[training["architecture"] == architecture].iloc[0]
        values = [float(row[f"dim{dim}_nrmse"]) for dim in range(5)]
        axes[0].bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=architecture,
            color=gap_colors[architecture],
        )
    axes[0].set_xticks(x, [f"z{dim}" for dim in range(5)])
    axes[0].set_title("Training-set scale-normalized recovery error")
    axes[0].set_ylabel("NRMSE")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].bar(
        ["Structured", "Coupled"],
        [
            int(training.set_index("architecture").loc["structured", "num_params"]),
            int(training.set_index("architecture").loc["coupled", "num_params"]),
        ],
        color=[gap_colors["structured"], gap_colors["coupled"]],
    )
    axes[1].set_title("Student encoder parameters")
    axes[1].set_ylabel("parameters")
    axes[1].grid(axis="y", alpha=0.25)
    _save_figure(
        fig,
        comparison_figures / "07_训练恢复误差与模型规模.png",
    )

    representative = max(
        [
            scenario
            for scenario in scenarios
            if scenario["kind"] == "random_wind"
        ],
        key=lambda scenario: float(
            np.max(
                _vector_norm(
                    comparison_frames[
                        (str(scenario["id"]), "structured_teacher")
                    ],
                    [
                        "wind_acc_x_mps2",
                        "wind_acc_y_mps2",
                        "wind_acc_z_mps2",
                    ],
                )
            )
        ),
    )
    scenario_id = str(representative["id"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
    wind_frame = comparison_frames[(scenario_id, "structured_teacher")]
    axes[0].plot(
        wind_frame["time_s"],
        _vector_norm(
            wind_frame,
            [
                "wind_acc_x_mps2",
                "wind_acc_y_mps2",
                "wind_acc_z_mps2",
            ],
        ),
        color="#7b3294",
    )
    axes[0].set_title(f"Total wind: {scenario_id}")
    axes[0].set_ylabel("m/s²")
    for method, report in [
        ("structured_student", structured_report),
        ("coupled_student", coupled_report),
    ]:
        frame = comparison_frames[(scenario_id, method)]
        z = _as_matrix(frame, Z_STUDENT_COLUMNS)
        scale = np.asarray(report["z_std"], dtype=float)
        departure = np.linalg.norm(
            (z - np.vstack([z[0], z[:-1]])) / scale, axis=1
        )
        axes[1].plot(
            frame["time_s"],
            departure,
            label=COMPARISON_LABELS[method],
            color=METHOD_COLORS[method],
        )
    axes[1].set_title("Per-step normalized context response")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("normalized latent change")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    _save_figure(
        fig, comparison_figures / "08_阵风与上下文快速响应.png"
    )

    fixed = next(
        scenario
        for scenario in scenarios
        if scenario["kind"] == "fixed_nowind"
    )
    fixed_id = str(fixed["id"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for method in COMPARISON_METHODS:
        frame = comparison_frames[(fixed_id, method)]
        axes[0, 0].plot(
            frame["time_s"],
            _vector_norm(frame, POSITION_COLUMNS),
            label=COMPARISON_LABELS[method],
            color=METHOD_COLORS[method],
        )
        axes[0, 1].plot(
            frame["time_s"],
            _vector_norm(frame, SWING_COLUMNS),
            label=COMPARISON_LABELS[method],
            color=METHOD_COLORS[method],
        )
    for method, report in [
        ("structured_student", structured_report),
        ("coupled_student", coupled_report),
    ]:
        frame = comparison_frames[(fixed_id, method)]
        error = (
            _as_matrix(frame, Z_STUDENT_COLUMNS)
            - _as_matrix(frame, Z_TEACHER_COLUMNS)
        ) / np.asarray(report["z_std"], dtype=float)
        axes[1, 0].plot(
            frame["time_s"],
            np.linalg.norm(error, axis=1),
            label=COMPARISON_LABELS[method],
            color=METHOD_COLORS[method],
        )
        axes[1, 1].plot(
            frame["time_s"],
            _vector_norm(frame, ACTION_COLUMNS),
            label=COMPARISON_LABELS[method],
            color=METHOD_COLORS[method],
        )
    for axis, title, ylabel in [
        (axes[0, 0], "No-wind position error", "m"),
        (axes[0, 1], "No-wind swing", "deg"),
        (axes[1, 0], "No-wind normalized context error", "norm"),
        (axes[1, 1], "No-wind CTBR action norm", "action norm"),
    ]:
        axis.set_title(title)
        axis.set_xlabel("time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    _save_figure(
        fig, comparison_figures / "09_无风中等参数案例.png"
    )

    random_metrics = comparison_metrics[
        comparison_metrics["scenario_kind"] == "random_wind"
    ]
    overview_metrics = [
        ("position_rmse_norm_m", "Position RMSE"),
        ("swing_rms_deg", "Swing RMS"),
        ("swing_peak_deg", "Swing peak"),
        ("context_nrmse", "Context NRMSE"),
        ("ctbr_total_variation_l1", "CTBR TV"),
        ("end_to_end_mean_ms", "Latency"),
    ]
    changes = []
    labels = []
    for metric, label in overview_metrics:
        proposed = _safe_mean(
            random_metrics[
                random_metrics["method"] == "structured_student"
            ][metric]
        )
        baseline = _safe_mean(
            random_metrics[
                random_metrics["method"] == "coupled_student"
            ][metric]
        )
        changes.append(_relative_percent(proposed, baseline))
        labels.append(label)
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.barh(
        labels,
        changes,
        color=["#2ca02c" if value < 0 else "#d62728" for value in changes],
    )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel(
        "Structured fast/slow relative to Coupled Student (%)\n"
        "negative is lower; interpretation depends on metric"
    )
    axis.set_title("Random-wind evaluation: relative metric change")
    axis.grid(axis="x", alpha=0.25)
    _save_figure(
        fig, comparison_figures / "10_贡献点证据总览.png"
    )

    _write_conclusions(
        output=comparison_dir / "结论与论文边界.md",
        metrics=comparison_metrics,
        gaps=comparison_gaps,
        training=training,
        fastslow_internal=fastslow_internal,
    )
    readme = f"""# 原始 RMA Coupled 7.30 数据整理

- Source run: `{manifest['source_run']}`
- Teacher: `{manifest['teacher_checkpoint']}`
- Student: `{manifest['student_checkpoint']}`
- Scenarios: {len(scenarios)}
- Coupled rollouts: {len(coupled_metrics)}
- Cross-method paired audit passed: `{audit['passed']}`

`raw/` 保存 Coupled Teacher/Student 原始 CSV、命令和控制台日志。
`data/` 与 `figures/` 是 Coupled 自身分析。
`对比快慢结构/` 是与 7.24 Structured fast/slow 数据的严格配对分析。

注意：五个 seed 是评估场景，不是五次独立训练。
"""
    (package_dir / "README.md").write_text(readme, encoding="utf-8")
    comparison_readme = """# Structured fast/slow versus Coupled RMA

本目录只使用逐时刻完全一致的质量、绳长与风扰场景。

- `data/`: 逐次指标、聚合指标、部署 gap、训练恢复指标和配对审计。
- `figures/`: 四方法轨迹与贡献点证据图。
- `结论与论文边界.md`: 可支持、不可支持和推荐表述。

Coupled z 没有质量/绳长物理语义，因此未把 Coupled z0/z1 画成物理量。
"""
    (comparison_dir / "README.md").write_text(
        comparison_readme, encoding="utf-8"
    )
    hash_frame = _write_hash_manifest(package_dir)
    return {
        "package_dir": str(package_dir),
        "audit_passed": bool(audit["passed"]),
        "coupled_rollouts": int(len(coupled_metrics)),
        "comparison_rollouts": int(len(comparison_metrics)),
        "hash_count": int(len(hash_frame)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package_dir", type=Path, required=True)
    parser.add_argument("--coupled_run", type=Path, required=True)
    parser.add_argument("--fastslow_package", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run_rollouts", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = args.package_dir / "experiment_manifest.json"
    if args.prepare or not manifest_path.is_file():
        manifest = prepare_package(
            package_dir=args.package_dir,
            coupled_run=args.coupled_run,
            fastslow_package=args.fastslow_package,
        )
        print(f"[PREPARED] {manifest_path}", flush=True)
    else:
        manifest = _read_json(manifest_path)
    if args.run_rollouts:
        run_rollouts(
            package_dir=args.package_dir.resolve(),
            manifest=manifest,
            force=bool(args.force),
        )
    if args.build:
        result = build_analysis(
            package_dir=args.package_dir,
            fastslow_package=args.fastslow_package,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
