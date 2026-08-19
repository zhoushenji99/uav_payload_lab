"""Build a reproducible multi-seed package for the July 24 fast/slow run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .fastslow_runtime import (
        compute_action_band_energy,
        compute_action_total_variation,
    )
except ImportError:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from fastslow_runtime import (
        compute_action_band_energy,
        compute_action_total_variation,
    )


EXPECTED_SOURCE_RUN = (
    "2026-07-24_06-06-18_hardexplicit_teacher_fastslow_seed42"
)
MODES = ("teacher", "all60", "fastslow")
MODE_LABELS = {
    "teacher": "Hard-explicit Teacher",
    "all60": "Dual Student, all 60 Hz",
    "fastslow": "Dual Student, fast/slow",
}
MODE_COLORS = {
    "teacher": "#222222",
    "all60": "#2a6fbb",
    "fastslow": "#e27d28",
}
WIND_COLUMNS = [
    "wind_acc_x_mps2",
    "wind_acc_y_mps2",
    "wind_acc_z_mps2",
]
GUST_COLUMNS = ["gust_x_mps2", "gust_y_mps2", "gust_z_mps2"]
ACTION_COLUMNS = ["a0_clamp", "a1_clamp", "a2_clamp", "a3_clamp"]
FAST_TEACHER_COLUMNS = ["zT2", "zT3", "zT4"]
FAST_STUDENT_COLUMNS = ["zH2", "zH3", "zH4"]
EXOGENOUS_COLUMNS = [
    "time_s",
    "rope_length_m",
    "payload_mass_kg",
    *GUST_COLUMNS,
    *WIND_COLUMNS,
]
REQUIRED_COLUMNS = [
    "time_s",
    "payload_err_x",
    "payload_err_y",
    "payload_err_z",
    "theta_x_deg",
    "theta_y_deg",
    "rope_length_m",
    "payload_mass_kg",
    "zT0",
    "zT1",
    "zH0",
    "zH1",
    *FAST_TEACHER_COLUMNS,
    *FAST_STUDENT_COLUMNS,
    "slow_batch_calls",
    "fast_batch_calls",
    "end_to_end_inference_ms",
    *GUST_COLUMNS,
    *WIND_COLUMNS,
    *ACTION_COLUMNS,
]


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _safe_mean(values: Any) -> float:
    array = _finite(values)
    return float(np.mean(array)) if array.size else float("nan")


def _safe_percentile(values: Any, percentile: float) -> float:
    array = _finite(values)
    return (
        float(np.percentile(array, percentile))
        if array.size
        else float("nan")
    )


def _norm(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.linalg.norm(frame[columns].to_numpy(dtype=float), axis=1)


def compute_rollout_metrics(
    frame: pd.DataFrame,
    *,
    sample_rate_hz: float = 60.0,
) -> dict[str, float | int]:
    """Compute control, recovery, action, wind, and latency metrics."""

    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Rollout is missing required columns: {missing}")
    if len(frame) == 0:
        raise ValueError("Rollout is empty.")

    position_error = _norm(
        frame, ["payload_err_x", "payload_err_y", "payload_err_z"]
    )
    swing = _norm(frame, ["theta_x_deg", "theta_y_deg"])
    actions = frame[ACTION_COLUMNS].to_numpy(dtype=float)
    wind = _norm(frame, WIND_COLUMNS)

    teacher_fast = frame[FAST_TEACHER_COLUMNS].to_numpy(dtype=float)
    student_fast = frame[FAST_STUDENT_COLUMNS].to_numpy(dtype=float)
    mass_hat = 0.3 + 0.5 * frame["zH0"].to_numpy(dtype=float)
    rope_hat = 0.25 + 0.55 * frame["zH1"].to_numpy(dtype=float)
    mass_true = frame["payload_mass_kg"].to_numpy(dtype=float)
    rope_true = frame["rope_length_m"].to_numpy(dtype=float)

    total_variation = compute_action_total_variation(actions)
    band_energy = compute_action_band_energy(
        actions,
        sample_rate_hz=sample_rate_hz,
        low_hz=5.0,
        high_hz=min(30.0, 0.5 * sample_rate_hz),
    )

    end_to_end = frame["end_to_end_inference_ms"].to_numpy(dtype=float)
    actor = (
        frame["actor_inference_ms"].to_numpy(dtype=float)
        if "actor_inference_ms" in frame
        else np.asarray([], dtype=float)
    )
    duration = (
        float(frame["time_s"].iloc[-1] - frame["time_s"].iloc[0])
        if len(frame) > 1
        else 0.0
    )

    return {
        "samples": int(len(frame)),
        "duration_s": duration,
        "position_rmse_norm_m": float(
            np.sqrt(np.mean(np.square(position_error)))
        ),
        "position_final_m": float(position_error[-1]),
        "position_peak_m": float(np.max(position_error)),
        "swing_rms_deg": float(np.sqrt(np.mean(np.square(swing)))),
        "swing_peak_deg": float(np.max(swing)),
        "swing_exposure_deg_s": float(
            np.trapz(swing, frame["time_s"].to_numpy(dtype=float))
        ),
        "z_rmse_mean": (
            _safe_mean(frame["z_rmse"]) if "z_rmse" in frame else float("nan")
        ),
        "mass_rmse_kg": float(
            np.sqrt(np.mean(np.square(mass_hat - mass_true)))
        ),
        "rope_length_rmse_m": float(
            np.sqrt(np.mean(np.square(rope_hat - rope_true)))
        ),
        "fast_context_rmse": float(
            np.sqrt(np.mean(np.square(student_fast - teacher_fast)))
        ),
        "ctbr_total_variation_l1": float(total_variation["total"]),
        "ctbr_mean_variation_l1": float(
            total_variation["mean_per_transition"]
        ),
        "ctbr_5_30hz_energy": float(band_energy["total"]),
        "ctbr_5_30hz_fraction": float(band_energy["fraction_total"]),
        "end_to_end_mean_ms": _safe_mean(end_to_end),
        "end_to_end_p95_ms": _safe_percentile(end_to_end, 95),
        "end_to_end_p99_ms": _safe_percentile(end_to_end, 99),
        "actor_mean_ms": _safe_mean(actor),
        "slow_batch_calls": int(
            np.nanmax(frame["slow_batch_calls"].to_numpy(dtype=float))
        ),
        "fast_batch_calls": int(
            np.nanmax(frame["fast_batch_calls"].to_numpy(dtype=float))
        ),
        "wind_rms_mps2": float(np.sqrt(np.mean(np.square(wind)))),
        "wind_peak_mps2": float(np.max(wind)),
        "payload_mass_kg": float(np.median(mass_true)),
        "rope_length_m": float(np.median(rope_true)),
    }


def audit_schedule_pair(
    all60: pd.DataFrame,
    fastslow: pd.DataFrame,
) -> dict[str, Any]:
    """Require exact exogenous inputs over the common rollout prefix."""

    missing = [
        column
        for column in EXOGENOUS_COLUMNS
        if column not in all60 or column not in fastslow
    ]
    if missing:
        return {
            "strict_pair": False,
            "common_samples": 0,
            "missing_columns": sorted(set(missing)),
            "exact_columns": {},
            "max_abs_difference": {},
        }

    common = min(len(all60), len(fastslow))
    exact_columns: dict[str, bool] = {}
    max_abs_difference: dict[str, float] = {}
    for column in EXOGENOUS_COLUMNS:
        left = all60[column].to_numpy(dtype=float)[:common]
        right = fastslow[column].to_numpy(dtype=float)[:common]
        exact_columns[column] = bool(np.array_equal(left, right))
        max_abs_difference[column] = (
            float(np.max(np.abs(left - right))) if common else float("nan")
        )
    return {
        "strict_pair": bool(common > 0 and all(exact_columns.values())),
        "common_samples": int(common),
        "missing_columns": [],
        "exact_columns": exact_columns,
        "max_abs_difference": max_abs_difference,
    }


def audit_fixed_nowind(
    frame: pd.DataFrame,
    *,
    expected_mass_kg: float,
    expected_rope_length_m: float,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Verify the fixed medium-physics, zero-total-wind scenario."""

    mass_error = float(
        np.max(
            np.abs(
                frame["payload_mass_kg"].to_numpy(dtype=float)
                - float(expected_mass_kg)
            )
        )
    )
    rope_error = float(
        np.max(
            np.abs(
                frame["rope_length_m"].to_numpy(dtype=float)
                - float(expected_rope_length_m)
            )
        )
    )
    wind_max = float(np.max(np.abs(frame[WIND_COLUMNS].to_numpy(dtype=float))))
    return {
        "passed": bool(
            mass_error <= tolerance
            and rope_error <= tolerance
            and wind_max <= tolerance
        ),
        "expected_mass_kg": float(expected_mass_kg),
        "expected_rope_length_m": float(expected_rope_length_m),
        "max_mass_abs_error_kg": mass_error,
        "max_rope_length_abs_error_m": rope_error,
        "max_total_wind_abs_mps2": wind_max,
        "tolerance": float(tolerance),
    }


def _summary_path(run_dir: Path) -> Path:
    direct = run_dir / "summary.json"
    if direct.exists():
        return direct
    candidates = sorted(run_dir.glob("*play_summary.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one summary JSON in {run_dir}, found {len(candidates)}"
        )
    return candidates[0]


def _read_rollout(package_dir: Path, scenario_id: str, mode: str):
    run_dir = package_dir / "raw" / scenario_id / mode
    csv_path = run_dir / "rollout.csv"
    summary_path = _summary_path(run_dir)
    return (
        pd.read_csv(csv_path),
        json.loads(summary_path.read_text(encoding="utf-8")),
        csv_path,
        summary_path,
    )


def _relative_change(new: float, baseline: float) -> float:
    if not np.isfinite(new) or not np.isfinite(baseline) or abs(baseline) < 1e-12:
        return float("nan")
    return float((new - baseline) / abs(baseline))


def _write_hash_manifest(package_dir: Path, data_dir: Path) -> pd.DataFrame:
    paths = [package_dir / "experiment_manifest.json"]
    paths.extend(
        path
        for path in sorted((package_dir / "raw").rglob("*"))
        if path.is_file()
    )
    rows = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "relative_path": str(path.relative_to(package_dir)),
                "bytes": int(path.stat().st_size),
                "sha256": digest,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / "sha256_manifest.csv", index=False)
    return frame


def _new_figure(rows: int, columns: int, width: float = 7.0):
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(width * columns, 3.2 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    return fig, axes


def _save(fig: plt.Figure, path: Path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_scenario_audit(
    scenarios: list[dict[str, Any]],
    frames: dict[tuple[str, str], pd.DataFrame],
    out: Path,
):
    labels = [scenario["id"] for scenario in scenarios]
    teacher_frames = [frames[(scenario["id"], "teacher")] for scenario in scenarios]
    mass = [float(frame["payload_mass_kg"].median()) for frame in teacher_frames]
    rope = [float(frame["rope_length_m"].median()) for frame in teacher_frames]
    wind = [
        float(np.sqrt(np.mean(np.square(_norm(frame, WIND_COLUMNS)))))
        for frame in teacher_frames
    ]
    fig, axes = _new_figure(1, 3, width=5.2)
    for axis, values, title, ylabel in [
        (axes[0, 0], mass, "Payload mass", "kg"),
        (axes[0, 1], rope, "Rope length", "m"),
        (axes[0, 2], wind, "Total wind RMS", "m/s²"),
    ]:
        axis.bar(labels, values, color="#648fff")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    _save(fig, out / "01_场景物理参数与风扰.png")


def _plot_multiscenario_trace(
    scenarios: list[dict[str, Any]],
    frames: dict[tuple[str, str], pd.DataFrame],
    out: Path,
    *,
    filename: str,
    title_prefix: str,
    value_function,
    ylabel: str,
):
    rows = int(math.ceil(len(scenarios) / 2))
    fig, axes = _new_figure(rows, 2)
    flat = axes.reshape(-1)
    for axis, scenario in zip(flat, scenarios):
        for mode in MODES:
            frame = frames[(scenario["id"], mode)]
            axis.plot(
                frame["time_s"],
                value_function(frame),
                label=MODE_LABELS[mode],
                color=MODE_COLORS[mode],
                linewidth=1.1,
            )
        axis.set_title(f"{title_prefix}: {scenario['id']}")
        axis.set_xlabel("time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    for axis in flat[len(scenarios) :]:
        axis.axis("off")
    flat[0].legend(fontsize=8)
    _save(fig, out / filename)


def _grouped_bars(
    axis: plt.Axes,
    table: pd.DataFrame,
    metric: str,
    scenarios: list[dict[str, Any]],
    modes: tuple[str, ...],
    title: str,
    ylabel: str,
):
    x = np.arange(len(scenarios), dtype=float)
    width = 0.8 / len(modes)
    for index, mode in enumerate(modes):
        values = []
        for scenario in scenarios:
            row = table[
                (table["scenario"] == scenario["id"])
                & (table["mode"] == mode)
            ].iloc[0]
            values.append(float(row[metric]))
        axis.bar(
            x + (index - (len(modes) - 1) / 2.0) * width,
            values,
            width=width,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
        )
    axis.set_xticks(x, [scenario["id"] for scenario in scenarios], rotation=35)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)


def _plot_context(
    scenarios: list[dict[str, Any]],
    metrics: pd.DataFrame,
    out: Path,
):
    fig, axes = _new_figure(1, 3, width=5.5)
    for axis, metric, title, ylabel in [
        (axes[0, 0], "mass_rmse_kg", "Mass recovery RMSE", "kg"),
        (
            axes[0, 1],
            "rope_length_rmse_m",
            "Rope-length recovery RMSE",
            "m",
        ),
        (
            axes[0, 2],
            "fast_context_rmse",
            "Fast-context Student–Teacher RMSE",
            "latent units",
        ),
    ]:
        _grouped_bars(
            axis,
            metrics,
            metric,
            scenarios,
            ("all60", "fastslow"),
            title,
            ylabel,
        )
    axes[0, 0].legend(fontsize=8)
    _save(fig, out / "04_Student上下文恢复.png")


def _plot_action(
    scenarios: list[dict[str, Any]],
    metrics: pd.DataFrame,
    out: Path,
):
    fig, axes = _new_figure(1, 2)
    _grouped_bars(
        axes[0, 0],
        metrics,
        "ctbr_total_variation_l1",
        scenarios,
        MODES,
        "CTBR total variation",
        "L1 total variation",
    )
    _grouped_bars(
        axes[0, 1],
        metrics,
        "ctbr_5_30hz_energy",
        scenarios,
        MODES,
        "CTBR 5–30 Hz energy",
        "squared amplitude",
    )
    axes[0, 0].legend(fontsize=8)
    _save(fig, out / "05_CTBR动作连续性.png")


def _plot_compute(
    scenarios: list[dict[str, Any]],
    metrics: pd.DataFrame,
    out: Path,
):
    fig, axes = _new_figure(1, 2)
    _grouped_bars(
        axes[0, 0],
        metrics,
        "end_to_end_mean_ms",
        scenarios,
        ("all60", "fastslow"),
        "End-to-end inference mean",
        "ms",
    )
    _grouped_bars(
        axes[0, 1],
        metrics,
        "slow_batch_calls",
        scenarios,
        ("all60", "fastslow"),
        "Slow-branch calls",
        "calls",
    )
    axes[0, 0].legend(fontsize=8)
    _save(fig, out / "06_计算开销与调用次数.png")


def _plot_gust_response(
    scenarios: list[dict[str, Any]],
    frames: dict[tuple[str, str], pd.DataFrame],
    out: Path,
):
    candidates = [
        scenario for scenario in scenarios if not scenario.get("disable_wind")
    ]
    representative = max(
        candidates,
        key=lambda scenario: float(
            np.max(_norm(frames[(scenario["id"], "teacher")], WIND_COLUMNS))
        ),
    )
    fig, axes = _new_figure(2, 1, width=10.0)
    teacher = frames[(representative["id"], "teacher")]
    axes[0, 0].plot(
        teacher["time_s"],
        _norm(teacher, WIND_COLUMNS),
        color="#7b3294",
    )
    axes[0, 0].set_title(
        f"Total wind acceleration: {representative['id']}"
    )
    axes[0, 0].set_ylabel("m/s²")
    axes[0, 0].grid(alpha=0.25)
    for mode in ("all60", "fastslow"):
        frame = frames[(representative["id"], mode)]
        fast = frame[FAST_STUDENT_COLUMNS].to_numpy(dtype=float)
        departure = np.linalg.norm(
            fast - np.vstack([fast[0], fast[:-1]]), axis=1
        )
        axes[1, 0].plot(
            frame["time_s"],
            departure,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
        )
    axes[1, 0].set_title("Per-step fast-context response")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("latent change")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()
    _save(fig, out / "07_阵风与快速上下文响应.png")


def _plot_aggregate(
    aggregate: pd.DataFrame,
    out: Path,
):
    random_rows = aggregate[aggregate["scenario_group"] == "random_wind"]
    metrics = [
        ("position_rmse_norm_m", "Position RMSE", "m"),
        ("swing_rms_deg", "Swing RMS", "deg"),
        ("ctbr_total_variation_l1", "CTBR total variation", "L1 TV"),
        ("end_to_end_mean_ms", "Inference mean", "ms"),
    ]
    fig, axes = _new_figure(2, 2)
    for axis, (metric, title, ylabel) in zip(axes.reshape(-1), metrics):
        means = []
        stds = []
        labels = []
        colors = []
        for mode in MODES:
            row = random_rows[random_rows["mode"] == mode]
            if row.empty:
                continue
            means.append(float(row[f"{metric}_mean"].iloc[0]))
            stds.append(float(row[f"{metric}_std"].iloc[0]))
            labels.append(MODE_LABELS[mode])
            colors.append(MODE_COLORS[mode])
        axis.bar(labels, means, yerr=stds, capsize=4, color=colors)
        axis.set_title(f"{title}: random-wind seeds")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    _save(fig, out / "08_整体性能汇总.png")


def _plot_teacher_gap(
    scenarios: list[dict[str, Any]],
    metrics: pd.DataFrame,
    out: Path,
):
    fig, axes = _new_figure(1, 2)
    x = np.arange(len(scenarios))
    width = 0.35
    for index, mode in enumerate(("all60", "fastslow")):
        position_gap = []
        swing_gap = []
        for scenario in scenarios:
            teacher = metrics[
                (metrics["scenario"] == scenario["id"])
                & (metrics["mode"] == "teacher")
            ].iloc[0]
            student = metrics[
                (metrics["scenario"] == scenario["id"])
                & (metrics["mode"] == mode)
            ].iloc[0]
            position_gap.append(
                float(
                    student["position_rmse_norm_m"]
                    - teacher["position_rmse_norm_m"]
                )
            )
            swing_gap.append(
                float(student["swing_rms_deg"] - teacher["swing_rms_deg"])
            )
        offset = (index - 0.5) * width
        axes[0, 0].bar(
            x + offset,
            position_gap,
            width,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
        )
        axes[0, 1].bar(
            x + offset,
            swing_gap,
            width,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
        )
    for axis, title, ylabel in [
        (axes[0, 0], "Student–Teacher position gap", "RMSE difference (m)"),
        (axes[0, 1], "Student–Teacher swing gap", "RMS difference (deg)"),
    ]:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(
            x, [scenario["id"] for scenario in scenarios], rotation=35
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    _save(fig, out / "09_Teacher到Student部署差距.png")


def _plot_fixed_case(
    scenarios: list[dict[str, Any]],
    frames: dict[tuple[str, str], pd.DataFrame],
    out: Path,
):
    fixed = next(
        scenario for scenario in scenarios if scenario["kind"] == "fixed_nowind"
    )
    fig, axes = _new_figure(2, 2)
    for mode in MODES:
        frame = frames[(fixed["id"], mode)]
        axes[0, 0].plot(
            frame["time_s"],
            _norm(
                frame,
                ["payload_err_x", "payload_err_y", "payload_err_z"],
            ),
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
        )
        axes[0, 1].plot(
            frame["time_s"],
            _norm(frame, ["theta_x_deg", "theta_y_deg"]),
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
        )
        mass_hat = 0.3 + 0.5 * frame["zH0"].to_numpy(dtype=float)
        rope_hat = 0.25 + 0.55 * frame["zH1"].to_numpy(dtype=float)
        axes[1, 0].plot(
            frame["time_s"],
            mass_hat,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
        )
        axes[1, 1].plot(
            frame["time_s"],
            rope_hat,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
        )
    axes[1, 0].axhline(0.55, color="black", linestyle="--", label="true")
    axes[1, 1].axhline(0.525, color="black", linestyle="--", label="true")
    for axis, title, ylabel in [
        (axes[0, 0], "No-wind position error", "m"),
        (axes[0, 1], "No-wind swing magnitude", "deg"),
        (axes[1, 0], "Recovered payload mass", "kg"),
        (axes[1, 1], "Recovered rope length", "m"),
    ]:
        axis.set_title(title)
        axis.set_xlabel("time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    _save(fig, out / "10_无风中等参数案例.png")


def _build_aggregate(
    metrics: pd.DataFrame,
    scenarios: list[dict[str, Any]],
) -> pd.DataFrame:
    scenario_kind = {
        scenario["id"]: scenario["kind"] for scenario in scenarios
    }
    table = metrics.copy()
    table["scenario_group"] = table["scenario"].map(scenario_kind)
    numeric = [
        column
        for column in table.select_dtypes(include=[np.number]).columns
        if column not in {"seed"}
    ]
    rows = []
    for (group, mode), subset in table.groupby(
        ["scenario_group", "mode"], sort=False
    ):
        row: dict[str, Any] = {
            "scenario_group": group,
            "mode": mode,
            "count": int(len(subset)),
        }
        for column in numeric:
            values = subset[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = _safe_mean(values)
            finite = _finite(values)
            row[f"{column}_std"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_readme(
    package_dir: Path,
    manifest: dict[str, Any],
    audit: dict[str, Any],
):
    text = f"""# 快慢结构7.27数据整理

本文件夹只整理 `{manifest['source_run']}` 的 Hard-explicit Teacher、双分支
All-60 Hz Student 和快慢 Student 数据。

- Teacher checkpoint: `{manifest.get('teacher_checkpoint', 'see manifest')}`
- Student checkpoint: `{manifest.get('student_checkpoint', 'see manifest')}`
- Evaluation scenarios: {len(manifest['scenarios'])}
- Rollouts: {len(manifest['scenarios']) * len(manifest['modes'])}
- Data audit passed: `{audit['passed']}`

`raw/` 保存原始 CSV、运行 summary 与控制台日志；`data/` 保存逐次指标、
配对审计、聚合统计和 SHA256；`figures/` 保存分析图片。

注意：这里的五个 seed 是五个评估场景，不是五次独立 Teacher/Student 训练。
以后 Coupled 必须复用 `experiment_manifest.json` 中完全相同的场景设置。
"""
    (package_dir / "README.md").write_text(text, encoding="utf-8")


def build_package(package_dir: str | Path) -> dict[str, Any]:
    package_dir = Path(package_dir).resolve()
    manifest_path = package_dir / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenarios = list(manifest["scenarios"])
    modes = tuple(manifest["modes"])
    if modes != MODES:
        raise ValueError(f"Expected modes {MODES}, got {modes}")

    data_dir = package_dir / "data"
    figure_dir = package_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    metric_rows = []
    required_column_audit: dict[str, Any] = {}
    summary_override_audit: dict[str, Any] = {}
    for scenario in scenarios:
        scenario_id = scenario["id"]
        for mode in modes:
            frame, summary, _, _ = _read_rollout(
                package_dir, scenario_id, mode
            )
            frames[(scenario_id, mode)] = frame
            summaries[(scenario_id, mode)] = summary
            missing = [
                column for column in REQUIRED_COLUMNS if column not in frame
            ]
            required_column_audit[f"{scenario_id}/{mode}"] = {
                "passed": not missing,
                "missing": missing,
                "rows": int(len(frame)),
            }

            overrides = summary.get("evaluation_overrides", {})
            expected_override = {
                "payload_mass_kg": scenario.get("fixed_payload_mass_kg"),
                "rope_length_m": scenario.get("fixed_rope_length_m"),
                "disable_wind": bool(scenario.get("disable_wind", False)),
            }
            actual_override = {
                "payload_mass_kg": overrides.get("payload_mass_kg"),
                "rope_length_m": overrides.get("rope_length_m"),
                "disable_wind": bool(overrides.get("disable_wind", False)),
            }
            summary_override_audit[f"{scenario_id}/{mode}"] = {
                "passed": actual_override == expected_override,
                "expected": expected_override,
                "actual": actual_override,
            }

            metrics = compute_rollout_metrics(frame)
            metrics.update(
                {
                    "scenario": scenario_id,
                    "seed": int(scenario["seed"]),
                    "scenario_kind": scenario["kind"],
                    "mode": mode,
                    "method": MODE_LABELS[mode],
                }
            )
            metric_rows.append(metrics)

    metrics_frame = pd.DataFrame(metric_rows)
    first_columns = [
        "scenario",
        "seed",
        "scenario_kind",
        "mode",
        "method",
    ]
    metrics_frame = metrics_frame[
        first_columns
        + [column for column in metrics_frame if column not in first_columns]
    ]
    metrics_frame.to_csv(data_dir / "rollout_metrics.csv", index=False)

    pair_audits: dict[str, Any] = {}
    pair_rows = []
    for scenario in scenarios:
        scenario_id = scenario["id"]
        pair_audit = audit_schedule_pair(
            frames[(scenario_id, "all60")],
            frames[(scenario_id, "fastslow")],
        )
        pair_audits[scenario_id] = pair_audit
        all60_metrics = metrics_frame[
            (metrics_frame["scenario"] == scenario_id)
            & (metrics_frame["mode"] == "all60")
        ].iloc[0]
        fastslow_metrics = metrics_frame[
            (metrics_frame["scenario"] == scenario_id)
            & (metrics_frame["mode"] == "fastslow")
        ].iloc[0]
        row = {
            "scenario": scenario_id,
            "seed": int(scenario["seed"]),
            "strict_pair": bool(pair_audit["strict_pair"]),
            "common_samples": int(pair_audit["common_samples"]),
        }
        for metric in [
            "position_rmse_norm_m",
            "swing_rms_deg",
            "swing_peak_deg",
            "mass_rmse_kg",
            "rope_length_rmse_m",
            "fast_context_rmse",
            "ctbr_total_variation_l1",
            "ctbr_5_30hz_energy",
            "end_to_end_mean_ms",
            "end_to_end_p95_ms",
            "end_to_end_p99_ms",
            "slow_batch_calls",
        ]:
            baseline = float(all60_metrics[metric])
            proposed = float(fastslow_metrics[metric])
            row[f"all60_{metric}"] = baseline
            row[f"fastslow_{metric}"] = proposed
            row[f"delta_{metric}"] = proposed - baseline
            row[f"relative_change_{metric}"] = _relative_change(
                proposed, baseline
            )
        pair_rows.append(row)
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(data_dir / "paired_schedule_metrics.csv", index=False)

    fixed_audits: dict[str, Any] = {}
    for scenario in scenarios:
        if scenario["kind"] != "fixed_nowind":
            continue
        for mode in modes:
            fixed_audits[f"{scenario['id']}/{mode}"] = audit_fixed_nowind(
                frames[(scenario["id"], mode)],
                expected_mass_kg=float(scenario["fixed_payload_mass_kg"]),
                expected_rope_length_m=float(
                    scenario["fixed_rope_length_m"]
                ),
            )

    aggregate = _build_aggregate(metrics_frame, scenarios)
    aggregate.to_csv(data_dir / "aggregate_metrics.csv", index=False)
    hash_frame = _write_hash_manifest(package_dir, data_dir)

    audit = {
        "source_run": manifest.get("source_run"),
        "source_run_matches_july24": (
            manifest.get("source_run") == EXPECTED_SOURCE_RUN
        ),
        "expected_rollouts": int(len(scenarios) * len(modes)),
        "observed_rollouts": int(len(metrics_frame)),
        "required_columns": required_column_audit,
        "summary_overrides": summary_override_audit,
        "schedule_pairs": pair_audits,
        "fixed_nowind": fixed_audits,
        "hash_count": int(len(hash_frame)),
    }
    audit["passed"] = bool(
        audit["source_run_matches_july24"]
        and audit["expected_rollouts"] == audit["observed_rollouts"]
        and all(
            item["passed"] for item in required_column_audit.values()
        )
        and all(
            item["passed"] for item in summary_override_audit.values()
        )
        and all(item["strict_pair"] for item in pair_audits.values())
        and all(item["passed"] for item in fixed_audits.values())
    )
    (data_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _plot_scenario_audit(scenarios, frames, figure_dir)
    _plot_multiscenario_trace(
        scenarios,
        frames,
        figure_dir,
        filename="02_位置误差多场景.png",
        title_prefix="Payload position error",
        value_function=lambda frame: _norm(
            frame, ["payload_err_x", "payload_err_y", "payload_err_z"]
        ),
        ylabel="error norm (m)",
    )
    _plot_multiscenario_trace(
        scenarios,
        frames,
        figure_dir,
        filename="03_摆角多场景.png",
        title_prefix="Payload swing",
        value_function=lambda frame: _norm(
            frame, ["theta_x_deg", "theta_y_deg"]
        ),
        ylabel="swing magnitude (deg)",
    )
    _plot_context(scenarios, metrics_frame, figure_dir)
    _plot_action(scenarios, metrics_frame, figure_dir)
    _plot_compute(scenarios, metrics_frame, figure_dir)
    _plot_gust_response(scenarios, frames, figure_dir)
    _plot_aggregate(aggregate, figure_dir)
    _plot_teacher_gap(scenarios, metrics_frame, figure_dir)
    _plot_fixed_case(scenarios, frames, figure_dir)
    _write_readme(package_dir, manifest, audit)

    return {
        "package_dir": str(package_dir),
        "metrics": metrics_frame,
        "pairs": pair_frame,
        "aggregate": aggregate,
        "audit": audit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package_dir", required=True)
    args = parser.parse_args()
    result = build_package(args.package_dir)
    print(
        json.dumps(
            {
                "package_dir": result["package_dir"],
                "audit_passed": result["audit"]["passed"],
                "rollouts": result["audit"]["observed_rollouts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not result["audit"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
