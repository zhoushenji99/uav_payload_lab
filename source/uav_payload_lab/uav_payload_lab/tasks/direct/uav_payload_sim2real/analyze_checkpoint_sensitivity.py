"""Analyze paired Teacher checkpoint sensitivity for structured and Coupled RMA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POSITION_COLUMNS = ["payload_err_x", "payload_err_y", "payload_err_z"]
SWING_COLUMNS = ["theta_x_deg", "theta_y_deg"]
RAW_ACTION_COLUMNS = ["a0_raw", "a1_raw", "a2_raw", "a3_raw"]
CLAMP_ACTION_COLUMNS = ["a0_clamp", "a1_clamp", "a2_clamp", "a3_clamp"]
WIND_COLUMNS = [
    "wind_acc_x_mps2",
    "wind_acc_y_mps2",
    "wind_acc_z_mps2",
]
EXOGENOUS_COLUMNS = [
    "time_s",
    "payload_mass_kg",
    "rope_length_m",
    "gust_x_mps2",
    "gust_y_mps2",
    "gust_z_mps2",
    *WIND_COLUMNS,
]
METHOD_LABELS = {
    "fastslow": "Structured Fast-Slow Teacher",
    "coupled": "Coupled RMA Teacher",
}
METHOD_COLORS = {"fastslow": "#d55e00", "coupled": "#0072b2"}
LOWER_IS_BETTER_METRICS = [
    "position_rmse_m",
    "position_peak_m",
    "position_final_m",
    "swing_rms_deg",
    "swing_peak_deg",
    "swing_exposure_deg_s",
    "ctbr_tv_mean_per_transition",
    "ctbr_5_30hz_fraction",
    "raw_action_clip_fraction",
]
LOSS_METRICS = ("best_val", "final_val", "final_train")


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def _gradient(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if len(values) < 3:
        return np.zeros_like(values)
    return np.gradient(values, time_s)


def compute_swing_energy_evidence(
    frame: pd.DataFrame,
    *,
    smooth_window_s: float = 0.15,
) -> pd.DataFrame:
    """Compute the three energy-evidence traces used by the paper figure."""

    required = [
        "time_s",
        "uav_px",
        "uav_py",
        "uav_pz",
        "theta_x_deg",
        "theta_y_deg",
        "theta_dot_x_deg_s",
        "theta_dot_y_deg_s",
        "rope_length_m",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Energy evidence is missing columns: {missing}")
    if len(frame) < 3:
        raise ValueError("Energy evidence requires at least three samples.")

    time_s = frame["time_s"].to_numpy(dtype=float)
    delta_t = float(np.nanmedian(np.diff(time_s)))
    window = (
        max(1, int(round(float(smooth_window_s) / delta_t)))
        if delta_t > 0.0
        else 1
    )
    length_m = max(
        float(np.nanmedian(frame["rope_length_m"].to_numpy(dtype=float))),
        1e-6,
    )
    uav_acceleration = []
    for column in ("uav_px", "uav_py", "uav_pz"):
        position = frame[column].to_numpy(dtype=float)
        velocity = _gradient(position, time_s)
        uav_acceleration.append(_gradient(velocity, time_s))
    acceleration_x, acceleration_y, acceleration_z = uav_acceleration

    theta_x = np.deg2rad(frame["theta_x_deg"].to_numpy(dtype=float))
    theta_y = np.deg2rad(frame["theta_y_deg"].to_numpy(dtype=float))
    theta_dot_x = np.deg2rad(
        frame["theta_dot_x_deg_s"].to_numpy(dtype=float)
    )
    theta_dot_y = np.deg2rad(
        frame["theta_dot_y_deg_s"].to_numpy(dtype=float)
    )
    theta_squared = np.square(theta_x) + np.square(theta_y)
    theta_dot_squared = np.square(theta_dot_x) + np.square(theta_dot_y)
    omega_squared = (9.81 + acceleration_z) / length_m
    omega_squared_dot = _gradient(omega_squared, time_s)

    energy = 0.5 * (
        theta_dot_squared + omega_squared * theta_squared
    )
    power_xy = (
        acceleration_x * theta_dot_x + acceleration_y * theta_dot_y
    ) / length_m
    power_parameter = 0.5 * omega_squared_dot * theta_squared
    power_model = power_xy + power_parameter
    energy_dot = _gradient(energy, time_s)

    return pd.DataFrame(
        {
            "time_s": time_s,
            "E_hat": _moving_average(energy, window),
            "E_dot_num": _moving_average(energy_dot, window),
            "P_xy": _moving_average(power_xy, window),
            "P_param": _moving_average(power_parameter, window),
            "P_model": _moving_average(power_model, window),
        }
    )


def build_loss_comparison_table(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Attach exact within-generation loss changes to report-derived records."""

    table = pd.DataFrame(records).copy()
    required = {
        "generation",
        "method",
        "is_reference",
        *LOSS_METRICS,
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Loss records are missing columns: {missing}")

    table["reference_method"] = ""
    for metric in LOSS_METRICS:
        table[f"{metric}_relative_change_pct"] = np.nan
    for generation, indices in table.groupby("generation").groups.items():
        subset = table.loc[indices]
        references = subset[subset["is_reference"].astype(bool)]
        if len(references) != 1:
            raise ValueError(
                f"Generation {generation!r} must have exactly one reference."
            )
        reference = references.iloc[0]
        table.loc[indices, "reference_method"] = str(reference["method"])
        for metric in LOSS_METRICS:
            reference_value = float(reference[metric])
            if abs(reference_value) <= 1e-15:
                continue
            table.loc[indices, f"{metric}_relative_change_pct"] = (
                (
                    table.loc[indices, metric].astype(float)
                    - reference_value
                )
                / abs(reference_value)
                * 100.0
            )
    return table


def _norm(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.linalg.norm(frame[columns].to_numpy(dtype=float), axis=1)


def _band_energy(
    actions: np.ndarray,
    *,
    sample_rate_hz: float,
    low_hz: float = 5.0,
    high_hz: float = 30.0,
) -> tuple[float, float]:
    if actions.shape[0] < 4:
        return 0.0, 0.0
    centered = actions - np.mean(actions, axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0)
    frequencies = np.fft.rfftfreq(actions.shape[0], d=1.0 / sample_rate_hz)
    power = np.square(np.abs(spectrum))
    usable_high = min(float(high_hz), 0.5 * float(sample_rate_hz))
    band = (frequencies >= float(low_hz)) & (frequencies <= usable_high)
    non_dc = frequencies > 0.0
    band_energy = float(np.sum(power[band]))
    total_energy = float(np.sum(power[non_dc]))
    fraction = band_energy / total_energy if total_energy > 0.0 else 0.0
    return band_energy, float(fraction)


def _settling_time(
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    threshold: float,
) -> float:
    above = np.flatnonzero(values > float(threshold))
    if above.size == 0:
        return float(time_s[0])
    last = int(above[-1])
    if last + 1 >= len(time_s):
        return float("nan")
    return float(time_s[last + 1])


def compute_teacher_metrics(
    frame: pd.DataFrame,
    *,
    sample_rate_hz: float = 60.0,
) -> dict[str, float | int]:
    """Compute transport, action, wind, and latency metrics for one rollout."""

    required = [
        "time_s",
        *POSITION_COLUMNS,
        *SWING_COLUMNS,
        *RAW_ACTION_COLUMNS,
        *CLAMP_ACTION_COLUMNS,
        "payload_mass_kg",
        "rope_length_m",
        *WIND_COLUMNS,
        "actor_inference_ms",
        "end_to_end_inference_ms",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Rollout is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Rollout is empty.")

    position = _norm(frame, POSITION_COLUMNS)
    swing = _norm(frame, SWING_COLUMNS)
    raw_actions = frame[RAW_ACTION_COLUMNS].to_numpy(dtype=float)
    actions = frame[CLAMP_ACTION_COLUMNS].to_numpy(dtype=float)
    action_delta = np.diff(actions, axis=0)
    tv_total = float(np.sum(np.abs(action_delta)))
    transitions = max(int(len(actions) - 1), 0)
    tv_mean = tv_total / transitions if transitions else 0.0
    band_energy, band_fraction = _band_energy(
        actions, sample_rate_hz=sample_rate_hz
    )
    clipped = ~np.isclose(raw_actions, actions, rtol=0.0, atol=1e-9)
    wind = _norm(frame, WIND_COLUMNS)
    time_s = frame["time_s"].to_numpy(dtype=float)
    actor_ms = frame["actor_inference_ms"].to_numpy(dtype=float)
    end_to_end_ms = frame["end_to_end_inference_ms"].to_numpy(dtype=float)

    return {
        "samples": int(len(frame)),
        "duration_s": (
            float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0
        ),
        "position_rmse_m": float(np.sqrt(np.mean(np.square(position)))),
        "position_peak_m": float(np.max(position)),
        "position_final_m": float(position[-1]),
        "swing_rms_deg": float(np.sqrt(np.mean(np.square(swing)))),
        "swing_peak_deg": float(np.max(swing)),
        "swing_exposure_deg_s": float(np.trapz(swing, time_s)),
        "swing_below_2deg_fraction": float(np.mean(swing <= 2.0)),
        "swing_settling_time_2deg_s": _settling_time(
            time_s, swing, threshold=2.0
        ),
        "ctbr_tv_total_l1": tv_total,
        "ctbr_tv_mean_per_transition": float(tv_mean),
        "ctbr_5_30hz_energy": band_energy,
        "ctbr_5_30hz_fraction": band_fraction,
        "raw_action_clip_fraction": float(np.mean(clipped)),
        "wind_rms_mps2": float(np.sqrt(np.mean(np.square(wind)))),
        "wind_peak_mps2": float(np.max(wind)),
        "payload_mass_kg": float(np.median(frame["payload_mass_kg"])),
        "rope_length_m": float(np.median(frame["rope_length_m"])),
        "actor_mean_ms": float(np.mean(actor_ms)),
        "actor_p95_ms": float(np.percentile(actor_ms, 95)),
        "actor_p99_ms": float(np.percentile(actor_ms, 99)),
        "end_to_end_mean_ms": float(np.mean(end_to_end_ms)),
        "end_to_end_p95_ms": float(np.percentile(end_to_end_ms, 95)),
        "end_to_end_p99_ms": float(np.percentile(end_to_end_ms, 99)),
    }


def audit_exact_exogenous(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> dict[str, Any]:
    """Audit exact physics and wind pairing over the common rollout prefix."""

    missing = [
        column
        for column in EXOGENOUS_COLUMNS
        if column not in left or column not in right
    ]
    if missing:
        return {
            "passed": False,
            "common_samples": 0,
            "missing_columns": sorted(set(missing)),
            "exact_columns": {},
            "max_abs_difference": {},
        }

    common = min(len(left), len(right))
    exact: dict[str, bool] = {}
    maximum: dict[str, float] = {}
    for column in EXOGENOUS_COLUMNS:
        left_values = left[column].to_numpy(dtype=float)[:common]
        right_values = right[column].to_numpy(dtype=float)[:common]
        exact[column] = bool(np.array_equal(left_values, right_values))
        maximum[column] = (
            float(np.max(np.abs(left_values - right_values)))
            if common
            else float("nan")
        )
    return {
        "passed": bool(
            common > 0
            and len(left) == len(right)
            and all(exact.values())
        ),
        "common_samples": int(common),
        "left_samples": int(len(left)),
        "right_samples": int(len(right)),
        "missing_columns": [],
        "exact_columns": exact,
        "max_abs_difference": maximum,
    }


def _summary_path(run_dir: Path) -> Path:
    direct = run_dir / "phase2_teacher_play_summary.json"
    if direct.is_file():
        return direct
    candidates = sorted(run_dir.glob("*teacher*summary.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one Teacher summary in {run_dir}, found {len(candidates)}"
        )
    return candidates[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rollouts(
    package_roots: dict[str, Path],
    *,
    steps: tuple[int, ...],
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[tuple[str, int, int], pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    frames: dict[tuple[str, int, int], pd.DataFrame] = {}
    for method, package_root in package_roots.items():
        for step in steps:
            for seed in seeds:
                run_dir = (
                    package_root
                    / "raw"
                    / f"step_{step}"
                    / f"seed_{seed}"
                    / "teacher"
                )
                csv_path = run_dir / "rollout.csv"
                summary_path = _summary_path(run_dir)
                frame = pd.read_csv(csv_path)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                frames[(method, step, seed)] = frame
                metrics = compute_teacher_metrics(frame)
                metrics.update(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "checkpoint_step": int(step),
                        "seed": int(seed),
                        "csv_path": str(csv_path),
                        "summary_path": str(summary_path),
                        "summary_steps": int(summary.get("steps", -1)),
                    }
                )
                rows.append(metrics)
    return pd.DataFrame(rows), frames


def _aggregate_by_checkpoint(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in {"checkpoint_step", "seed"}
    ]
    rows: list[dict[str, Any]] = []
    for (method, step), subset in metrics.groupby(
        ["method", "checkpoint_step"], sort=True
    ):
        row: dict[str, Any] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "checkpoint_step": int(step),
            "seed_count": int(subset["seed"].nunique()),
        }
        for column in numeric:
            values = subset[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{column}_mean"] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
            row[f"{column}_std"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_checkpoint_sensitivity(
    checkpoint_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHOD_LABELS:
        method_rows = checkpoint_summary[
            checkpoint_summary["method"] == method
        ]
        for metric in LOWER_IS_BETTER_METRICS:
            values = method_rows[
                ["checkpoint_step", f"{metric}_mean"]
            ].dropna()
            if values.empty:
                continue
            metric_values = values[f"{metric}_mean"].to_numpy(dtype=float)
            overall = float(np.mean(metric_values))
            best_index = int(np.argmin(metric_values))
            worst_index = int(np.argmax(metric_values))
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "best_checkpoint_step": int(
                        values.iloc[best_index]["checkpoint_step"]
                    ),
                    "worst_checkpoint_step": int(
                        values.iloc[worst_index]["checkpoint_step"]
                    ),
                    "best_mean": float(metric_values[best_index]),
                    "worst_mean": float(metric_values[worst_index]),
                    "checkpoint_range": float(
                        np.max(metric_values) - np.min(metric_values)
                    ),
                    "checkpoint_range_fraction_of_mean": (
                        float(
                            (np.max(metric_values) - np.min(metric_values))
                            / abs(overall)
                        )
                        if abs(overall) > 1e-12
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _build_paired_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = metrics.set_index(["method", "checkpoint_step", "seed"])
    rows: list[dict[str, Any]] = []
    for step in sorted(metrics["checkpoint_step"].unique()):
        for seed in sorted(metrics["seed"].unique()):
            fastslow = indexed.loc[("fastslow", step, seed)]
            coupled = indexed.loc[("coupled", step, seed)]
            row: dict[str, Any] = {
                "checkpoint_step": int(step),
                "seed": int(seed),
            }
            for metric in LOWER_IS_BETTER_METRICS:
                proposed = float(fastslow[metric])
                baseline = float(coupled[metric])
                row[f"fastslow_{metric}"] = proposed
                row[f"coupled_{metric}"] = baseline
                row[f"delta_{metric}"] = proposed - baseline
                row[f"relative_change_{metric}"] = (
                    (proposed - baseline) / abs(baseline)
                    if abs(baseline) > 1e-12
                    else float("nan")
                )
                row[f"fastslow_lower_{metric}"] = bool(proposed < baseline)
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize_paired(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, int | None, pd.DataFrame]] = [
        ("all_checkpoint_seed_cells", None, paired)
    ]
    groups.extend(
        (
            "per_checkpoint",
            int(step),
            subset,
        )
        for step, subset in paired.groupby("checkpoint_step", sort=True)
    )
    for group, step, subset in groups:
        for metric in LOWER_IS_BETTER_METRICS:
            changes = subset[f"relative_change_{metric}"].to_numpy(dtype=float)
            deltas = subset[f"delta_{metric}"].to_numpy(dtype=float)
            finite_changes = changes[np.isfinite(changes)]
            finite_deltas = deltas[np.isfinite(deltas)]
            rows.append(
                {
                    "group": group,
                    "checkpoint_step": step,
                    "metric": metric,
                    "cell_count": int(len(subset)),
                    "fastslow_lower_count": int(
                        subset[f"fastslow_lower_{metric}"].sum()
                    ),
                    "mean_delta": (
                        float(np.mean(finite_deltas))
                        if finite_deltas.size
                        else float("nan")
                    ),
                    "median_delta": (
                        float(np.median(finite_deltas))
                        if finite_deltas.size
                        else float("nan")
                    ),
                    "mean_relative_change": (
                        float(np.mean(finite_changes))
                        if finite_changes.size
                        else float("nan")
                    ),
                    "median_relative_change": (
                        float(np.median(finite_changes))
                        if finite_changes.size
                        else float("nan")
                    ),
                    "relative_change_std": (
                        float(np.std(finite_changes, ddof=1))
                        if finite_changes.size > 1
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _build_pair_audit(
    frames: dict[tuple[str, int, int], pd.DataFrame],
    *,
    steps: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    cross_method: dict[str, Any] = {}
    within_method: dict[str, Any] = {}
    for step in steps:
        for seed in seeds:
            key = f"step_{step}/seed_{seed}"
            cross_method[key] = audit_exact_exogenous(
                frames[("fastslow", step, seed)],
                frames[("coupled", step, seed)],
            )
    reference_step = steps[0]
    for method in METHOD_LABELS:
        for step in steps[1:]:
            for seed in seeds:
                key = (
                    f"{method}/step_{reference_step}_vs_{step}/seed_{seed}"
                )
                within_method[key] = audit_exact_exogenous(
                    frames[(method, reference_step, seed)],
                    frames[(method, step, seed)],
                )

    hard_explicit: dict[str, Any] = {}
    for step in steps:
        for seed in seeds:
            frame = frames[("fastslow", step, seed)]
            expected_z0 = (
                frame["payload_mass_kg"].to_numpy(dtype=float) - 0.3
            ) / 0.5
            expected_z1 = (
                frame["rope_length_m"].to_numpy(dtype=float) - 0.25
            ) / 0.55
            error_z0 = float(
                np.max(np.abs(frame["zT0"].to_numpy(dtype=float) - expected_z0))
            )
            error_z1 = float(
                np.max(np.abs(frame["zT1"].to_numpy(dtype=float) - expected_z1))
            )
            hard_explicit[f"step_{step}/seed_{seed}"] = {
                "passed": bool(error_z0 <= 1e-6 and error_z1 <= 1e-6),
                "zT0_max_abs_error": error_z0,
                "zT1_max_abs_error": error_z1,
            }
    passed = bool(
        all(item["passed"] for item in cross_method.values())
        and all(item["passed"] for item in within_method.values())
        and all(item["passed"] for item in hard_explicit.values())
    )
    return {
        "passed": passed,
        "cross_method_exact_exogenous": cross_method,
        "within_method_checkpoint_exact_exogenous": within_method,
        "hard_explicit_identity_path": hard_explicit,
    }


def _plot_metric_grid(
    metrics: pd.DataFrame,
    *,
    method: str,
    output: Path,
    metrics_to_plot: list[tuple[str, str, str]],
):
    subset = metrics[metrics["method"] == method]
    fig, axes = plt.subplots(
        1,
        len(metrics_to_plot),
        figsize=(5.2 * len(metrics_to_plot), 4.0),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, (metric, title, ylabel) in zip(axes, metrics_to_plot):
        for seed, seed_rows in subset.groupby("seed", sort=True):
            seed_rows = seed_rows.sort_values("checkpoint_step")
            axis.plot(
                seed_rows["checkpoint_step"],
                seed_rows[metric],
                marker="o",
                alpha=0.45,
                linewidth=1.0,
                label=f"seed {seed}",
            )
        means = (
            subset.groupby("checkpoint_step", sort=True)[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        axis.errorbar(
            means["checkpoint_step"],
            means["mean"],
            yerr=means["std"].fillna(0.0),
            color=METHOD_COLORS[method],
            marker="s",
            linewidth=2.2,
            capsize=4,
            label="mean ± SD",
        )
        axis.set_title(title)
        axis.set_xlabel("checkpoint step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.ticklabel_format(style="plain", axis="x")
    axes[0].legend(fontsize=8)
    fig.suptitle(METHOD_LABELS[method])
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_cross_method(
    checkpoint_summary: pd.DataFrame,
    output: Path,
):
    plot_metrics = [
        ("position_rmse_m", "Position RMSE", "m"),
        ("swing_rms_deg", "Swing RMS", "deg"),
        (
            "ctbr_tv_mean_per_transition",
            "CTBR total variation / step",
            "L1",
        ),
        ("ctbr_5_30hz_fraction", "CTBR 5–30 Hz fraction", "fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (metric, title, ylabel) in zip(axes.flat, plot_metrics):
        for method in METHOD_LABELS:
            rows = checkpoint_summary[
                checkpoint_summary["method"] == method
            ].sort_values("checkpoint_step")
            axis.errorbar(
                rows["checkpoint_step"],
                rows[f"{metric}_mean"],
                yerr=rows[f"{metric}_std"],
                marker="o",
                capsize=4,
                linewidth=2.0,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(title)
        axis.set_xlabel("checkpoint step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.ticklabel_format(style="plain", axis="x")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Teacher checkpoint sensitivity: 3 seeds per checkpoint")
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_paired_deltas(paired: pd.DataFrame, output: Path):
    plot_metrics = [
        ("position_rmse_m", "Position RMSE delta", "m"),
        ("swing_rms_deg", "Swing RMS delta", "deg"),
        (
            "ctbr_tv_mean_per_transition",
            "CTBR variation delta",
            "L1 / step",
        ),
        ("ctbr_5_30hz_fraction", "5–30 Hz fraction delta", "fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (metric, title, ylabel) in zip(axes.flat, plot_metrics):
        for seed, rows in paired.groupby("seed", sort=True):
            rows = rows.sort_values("checkpoint_step")
            axis.plot(
                rows["checkpoint_step"],
                rows[f"delta_{metric}"],
                marker="o",
                label=f"seed {seed}",
            )
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.set_title(title)
        axis.set_xlabel("checkpoint step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.ticklabel_format(style="plain", axis="x")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fast-Slow minus Coupled; negative favors Fast-Slow")
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_representative_traces(
    frames: dict[tuple[str, int, int], pd.DataFrame],
    *,
    steps: tuple[int, ...],
    seed: int,
    output: Path,
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for method in METHOD_LABELS:
        for step in steps:
            frame = frames[(method, step, seed)]
            style = "-" if method == "fastslow" else "--"
            label = f"{METHOD_LABELS[method]} / {step}"
            axes[0].plot(
                frame["time_s"],
                _norm(frame, POSITION_COLUMNS),
                linestyle=style,
                linewidth=1.0,
                label=label,
            )
            axes[1].plot(
                frame["time_s"],
                _norm(frame, SWING_COLUMNS),
                linestyle=style,
                linewidth=1.0,
                label=label,
            )
    axes[0].set_title(f"Position-error trajectories, seed {seed}")
    axes[0].set_ylabel("m")
    axes[1].set_title(f"Payload-swing trajectories, seed {seed}")
    axes[1].set_ylabel("deg")
    axes[1].set_xlabel("time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=7)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _task_frame_coordinates(
    frame: pd.DataFrame,
    *,
    goal_reference: np.ndarray = np.array([4.0, 0.0, 2.0]),
) -> dict[str, np.ndarray]:
    goal_columns = ["goal_px", "goal_py", "goal_pz"]
    payload_columns = ["payload_px", "payload_py", "payload_pz"]
    missing = [
        column
        for column in [*goal_columns, *payload_columns]
        if column not in frame
    ]
    if missing:
        raise ValueError(f"Position figure is missing columns: {missing}")
    goal_world = np.nanmedian(
        frame[goal_columns].to_numpy(dtype=float), axis=0
    )
    origin_world = goal_world - np.asarray(goal_reference, dtype=float)
    payload_task = (
        frame[payload_columns].to_numpy(dtype=float) - origin_world
    )
    goal_task = frame[goal_columns].to_numpy(dtype=float) - origin_world
    return {
        "payload_x": payload_task[:, 0],
        "payload_y": payload_task[:, 1],
        "payload_z": payload_task[:, 2],
        "goal_x": goal_task[:, 0],
        "goal_y": goal_task[:, 1],
        "goal_z": goal_task[:, 2],
    }


def _plot_payload_xyz_swingxy(
    frames: dict[str, pd.DataFrame],
    output: Path,
    *,
    step: int,
    seed: int,
):
    fig, axes = plt.subplots(
        2, 3, figsize=(14.5, 7.5), constrained_layout=True
    )
    styles = {
        "fastslow": {"linestyle": "-", "linewidth": 1.7},
        "coupled": {"linestyle": "--", "linewidth": 1.5},
    }
    for method in ("fastslow", "coupled"):
        frame = frames[method]
        time_s = frame["time_s"].to_numpy(dtype=float)
        task = _task_frame_coordinates(frame)
        for column_index, axis_name in enumerate(("x", "y", "z")):
            values = (
                pd.Series(task[f"payload_{axis_name}"])
                .rolling(7, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )
            axes[0, column_index].plot(
                time_s,
                values,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                **styles[method],
            )
        for column_index, source in enumerate(
            ("theta_x_deg", "theta_y_deg")
        ):
            values = (
                frame[source]
                .rolling(7, center=True, min_periods=1)
                .mean()
                .to_numpy(dtype=float)
            )
            axes[1, column_index].plot(
                time_s,
                values,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                **styles[method],
            )
        error_norm = _norm(frame, POSITION_COLUMNS)
        axes[1, 2].plot(
            time_s,
            error_norm,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            **styles[method],
        )

    reference_frame = frames["fastslow"]
    reference_task = _task_frame_coordinates(reference_frame)
    reference_time = reference_frame["time_s"].to_numpy(dtype=float)
    for column_index, axis_name in enumerate(("x", "y", "z")):
        axes[0, column_index].plot(
            reference_time,
            reference_task[f"goal_{axis_name}"],
            color="black",
            linestyle=":",
            linewidth=1.0,
            label="reference",
        )
        axes[0, column_index].set_title(
            f"Payload {axis_name.upper()} (task frame)"
        )
        axes[0, column_index].set_ylabel("position (m)")
    for axis, title in zip(
        axes[1, :2],
        ("Payload swing θx", "Payload swing θy"),
    ):
        axis.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(title)
        axis.set_ylabel("angle (deg)")
    axes[1, 2].set_title("Payload position-error norm")
    axes[1, 2].set_ylabel("error (m)")
    for axis in axes.flat:
        axis.set_xlabel("time (s)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Paired Teacher rollout: checkpoint {step}, seed {seed}"
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_swing_energy_3panel(
    frames: dict[str, pd.DataFrame],
    output: Path,
    *,
    step: int,
    seed: int,
    time_window_s: float = 5.0,
):
    fig, axes = plt.subplots(
        3, 1, figsize=(12.5, 10.0), sharex=True, constrained_layout=True
    )
    for method in ("fastslow", "coupled"):
        energy = compute_swing_energy_evidence(frames[method])
        mask = energy["time_s"].to_numpy(dtype=float) <= float(time_window_s)
        trace = energy.loc[mask]
        style = "-" if method == "fastslow" else "--"
        label = METHOD_LABELS[method]
        axes[0].plot(
            trace["time_s"],
            trace["E_hat"],
            color=METHOD_COLORS[method],
            linestyle=style,
            linewidth=1.6,
            label=label,
        )
        axes[1].plot(
            trace["time_s"],
            trace["E_dot_num"],
            color=METHOD_COLORS[method],
            linestyle=style,
            linewidth=1.2,
            label=f"{label}: dE/dt",
        )
        axes[1].plot(
            trace["time_s"],
            trace["P_model"],
            color=METHOD_COLORS[method],
            linestyle=":",
            linewidth=1.3,
            label=f"{label}: P_model",
        )
        axes[2].plot(
            trace["time_s"],
            trace["P_xy"],
            color=METHOD_COLORS[method],
            linestyle=style,
            linewidth=1.2,
            label=f"{label}: P_xy",
        )
        axes[2].plot(
            trace["time_s"],
            trace["P_param"],
            color=METHOD_COLORS[method],
            linestyle=":",
            linewidth=1.2,
            label=f"{label}: P_param",
        )
        axes[2].plot(
            trace["time_s"],
            trace["P_model"],
            color=METHOD_COLORS[method],
            linestyle="-.",
            linewidth=1.0,
            alpha=0.75,
            label=f"{label}: P_model",
        )
    axes[0].set_title("Normalized swing energy")
    axes[0].set_ylabel("E_hat")
    axes[1].set_title("Energy-rate verification: dE/dt vs P_model")
    axes[1].set_ylabel("power")
    axes[2].set_title("Power decomposition: P_model = P_xy + P_param")
    axes[2].set_ylabel("power")
    axes[2].set_xlabel("time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    fig.suptitle(
        f"Swing-energy evidence, checkpoint {step}, seed {seed}, "
        f"first {time_window_s:g} s"
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _read_loss_record(
    path: Path,
    *,
    generation: str,
    method: str,
    is_reference: bool,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    train_history = report.get("train_hist", [])
    validation_history = report.get("val_hist", [])
    if not train_history or not validation_history:
        raise ValueError(f"Loss report has no complete histories: {path}")
    return {
        "generation": generation,
        "method": method,
        "is_reference": bool(is_reference),
        "best_val": float(report["best_val"]),
        "final_val": float(validation_history[-1]),
        "final_train": float(train_history[-1]),
        "best_epoch": int(report.get("best_epoch", -1)),
        "epochs_ran": int(report.get("epochs_ran", len(train_history))),
        "num_params": int(report.get("num_params", -1)),
        "aux_ml_coef": float(report.get("aux_ml_coef", float("nan"))),
        "report_path": str(path.resolve()),
    }


def _plot_loss_comparison(table: pd.DataFrame, output: Path):
    generations = [
        ("old", "Old April experiment"),
        ("current", "Current July experiment"),
    ]
    labels = {
        "best_val": "Best validation",
        "final_val": "Final validation",
        "final_train": "Final training",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.20,
        top=0.76,
        wspace=0.16,
    )
    for axis, (generation, title) in zip(axes, generations):
        subset = table[table["generation"] == generation].reset_index(
            drop=True
        )
        x_positions = np.arange(len(LOSS_METRICS), dtype=float)
        width = 0.36
        for method_index, row in subset.iterrows():
            offsets = x_positions + (method_index - 0.5) * width
            values = [float(row[metric]) for metric in LOSS_METRICS]
            color = (
                METHOD_COLORS["coupled"]
                if bool(row["is_reference"])
                else METHOD_COLORS["fastslow"]
            )
            bars = axis.bar(
                offsets,
                values,
                width=width,
                color=color,
                alpha=0.88,
                label=str(row["method"]),
            )
            for metric, bar, value in zip(LOSS_METRICS, bars, values):
                change = float(
                    row[f"{metric}_relative_change_pct"]
                )
                annotation = f"{value:.5f}"
                if not bool(row["is_reference"]):
                    annotation += f"\n({change:+.1f}%)"
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    annotation,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        axis.set_xticks(
            x_positions,
            [labels[metric] for metric in LOSS_METRICS],
        )
        axis.set_ylabel("reported weighted MSE")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
        axis.set_ylim(bottom=0.0)
    fig.suptitle(
        "Phase-II Student loss from original report.json files\n"
        "Percentage is relative to Coupled within the same experiment",
        y=0.96,
    )
    fig.text(
        0.5,
        0.045,
        (
            "Raw loss is shown exactly. The July pair changes Teacher labels, "
            "Student architecture, and aux coefficient, so it is not a "
            "strict architecture-only loss ablation."
        ),
        ha="center",
        fontsize=8,
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_requested_paper_figures(
    fastslow_package: str | Path,
    coupled_package: str | Path,
    *,
    loss_report_paths: dict[str, str | Path],
    step: int = 19500,
    seed: int = 42,
) -> list[str]:
    """Create the two requested dynamics figures and the raw-loss evidence."""

    package_roots = {
        "fastslow": Path(fastslow_package).resolve(),
        "coupled": Path(coupled_package).resolve(),
    }
    frames = {
        method: pd.read_csv(
            package_root
            / "raw"
            / f"step_{step}"
            / f"seed_{seed}"
            / "teacher"
            / "rollout.csv"
        )
        for method, package_root in package_roots.items()
    }
    for package_root in package_roots.values():
        (package_root / "figures").mkdir(parents=True, exist_ok=True)
        (package_root / "data").mkdir(parents=True, exist_ok=True)

    fast_figure_dir = package_roots["fastslow"] / "figures"
    figure_names = [
        f"06_step{step}_seed{seed}_payload_xyz_swingxy.png",
        f"07_step{step}_seed{seed}_swing_energy_3panel.png",
        "08_PhaseII_loss_raw_comparison.png",
    ]
    _plot_payload_xyz_swingxy(
        frames,
        fast_figure_dir / figure_names[0],
        step=step,
        seed=seed,
    )
    _plot_swing_energy_3panel(
        frames,
        fast_figure_dir / figure_names[1],
        step=step,
        seed=seed,
    )

    report_specification = [
        ("old_coupled", "old", "Coupled", True),
        ("old_decoupled", "old", "Decoupled", False),
        ("current_coupled", "current", "Coupled", True),
        (
            "current_structured",
            "current",
            "Structured Fast-Slow",
            False,
        ),
    ]
    records = [
        _read_loss_record(
            Path(loss_report_paths[key]),
            generation=generation,
            method=method,
            is_reference=is_reference,
        )
        for key, generation, method, is_reference in report_specification
    ]
    loss_table = build_loss_comparison_table(records)
    _plot_loss_comparison(
        loss_table,
        fast_figure_dir / figure_names[2],
    )
    for package_root in package_roots.values():
        loss_table.to_csv(
            package_root / "data" / "phase2_loss_detailed.csv",
            index=False,
        )
    coupled_figure_dir = package_roots["coupled"] / "figures"
    for figure_name in figure_names:
        shutil.copyfile(
            fast_figure_dir / figure_name,
            coupled_figure_dir / figure_name,
        )
    return [str(fast_figure_dir / name) for name in figure_names]


def _write_manifests(
    package_roots: dict[str, Path],
    *,
    steps: tuple[int, ...],
    seeds: tuple[int, ...],
):
    for method, package_root in package_roots.items():
        run_root = package_root.parent
        checkpoints = []
        for step in steps:
            checkpoint = run_root / f"model_{step}.pt"
            checkpoints.append(
                {
                    "step": int(step),
                    "path": str(checkpoint),
                    "sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
                }
            )
        manifest = {
            "experiment": "Teacher checkpoint sensitivity",
            "method": method,
            "method_label": METHOD_LABELS[method],
            "run_root": str(run_root),
            "package_root": str(package_root),
            "checkpoint_steps": list(steps),
            "evaluation_seeds": list(seeds),
            "rollout_count": int(len(steps) * len(seeds)),
            "task": "Isaac-Uav-Sim2Real-v0",
            "num_envs": 1,
            "max_steps": 2200,
            "mode": "teacher",
            "checkpoints": checkpoints,
            "statistical_note": (
                "Checkpoint-seed cells are correlated sensitivity measurements, "
                "not independent training runs."
            ),
        }
        (package_root / "experiment_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _write_hashes(package_root: Path):
    paths = [
        path
        for path in sorted((package_root / "raw").rglob("*"))
        if path.is_file()
    ]
    rows = [
        {
            "relative_path": str(path.relative_to(package_root)),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    pd.DataFrame(rows).to_csv(
        package_root / "data" / "sha256_manifest.csv", index=False
    )


def _write_readme(
    package_root: Path,
    *,
    method: str,
    audit: dict[str, Any],
):
    text = f"""# 3steps3seeds — {METHOD_LABELS[method]}

本目录保存 19000、19500、19999 三个 Teacher checkpoint 在 seed
38、40、42 下的敏感性评估。

- 原始数据：`raw/step_*/seed_*/teacher/`
- 逐 rollout 指标：`data/rollout_metrics.csv`
- checkpoint 均值与标准差：`data/checkpoint_summary.csv`
- 跨结构逐格配对比较：`data/cross_method_paired_metrics.csv`
- 数据审计：`data/data_audit.json`
- 审计通过：`{audit['passed']}`

统计边界：这九格来自同一次 Teacher 训练的三个相邻 checkpoint，不是九次
独立训练，因而只能判断 checkpoint 敏感性，不能代替多训练 seed 统计。
"""
    (package_root / "README.md").write_text(text, encoding="utf-8")


def build_analysis(
    fastslow_package: str | Path,
    coupled_package: str | Path,
    *,
    steps: tuple[int, ...] = (19000, 19500, 19999),
    seeds: tuple[int, ...] = (38, 40, 42),
) -> dict[str, Any]:
    """Build audited per-method and paired checkpoint-sensitivity artifacts."""

    package_roots = {
        "fastslow": Path(fastslow_package).resolve(),
        "coupled": Path(coupled_package).resolve(),
    }
    for package_root in package_roots.values():
        (package_root / "data").mkdir(parents=True, exist_ok=True)
        (package_root / "figures").mkdir(parents=True, exist_ok=True)

    metrics, frames = _read_rollouts(
        package_roots, steps=steps, seeds=seeds
    )
    checkpoint_summary = _aggregate_by_checkpoint(metrics)
    sensitivity = _build_checkpoint_sensitivity(checkpoint_summary)
    paired = _build_paired_metrics(metrics)
    paired_summary = _summarize_paired(paired)
    audit = _build_pair_audit(frames, steps=steps, seeds=seeds)

    for method, package_root in package_roots.items():
        method_metrics = metrics[metrics["method"] == method]
        method_metrics.to_csv(
            package_root / "data" / "rollout_metrics.csv", index=False
        )
        checkpoint_summary[
            checkpoint_summary["method"] == method
        ].to_csv(
            package_root / "data" / "checkpoint_summary.csv", index=False
        )
        sensitivity[sensitivity["method"] == method].to_csv(
            package_root / "data" / "checkpoint_sensitivity.csv", index=False
        )
        paired.to_csv(
            package_root / "data" / "cross_method_paired_metrics.csv",
            index=False,
        )
        paired_summary.to_csv(
            package_root / "data" / "cross_method_summary.csv", index=False
        )
        (package_root / "data" / "data_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _plot_metric_grid(
            metrics,
            method=method,
            output=package_root
            / "figures"
            / "01_checkpoint位置与摆角.png",
            metrics_to_plot=[
                ("position_rmse_m", "Position RMSE", "m"),
                ("swing_rms_deg", "Swing RMS", "deg"),
                ("swing_peak_deg", "Swing peak", "deg"),
            ],
        )
        _plot_metric_grid(
            metrics,
            method=method,
            output=package_root
            / "figures"
            / "02_checkpoint动作连续性.png",
            metrics_to_plot=[
                (
                    "ctbr_tv_mean_per_transition",
                    "CTBR variation / step",
                    "L1",
                ),
                (
                    "ctbr_5_30hz_fraction",
                    "CTBR 5–30 Hz fraction",
                    "fraction",
                ),
                (
                    "raw_action_clip_fraction",
                    "Raw-action clipping",
                    "fraction",
                ),
            ],
        )

    fast_figure_dir = package_roots["fastslow"] / "figures"
    coupled_figure_dir = package_roots["coupled"] / "figures"
    cross_figure = fast_figure_dir / "03_两结构checkpoint均值对比.png"
    _plot_cross_method(checkpoint_summary, cross_figure)
    shutil.copyfile(
        cross_figure,
        coupled_figure_dir / cross_figure.name,
    )
    delta_figure = fast_figure_dir / "04_逐seed配对差值.png"
    _plot_paired_deltas(paired, delta_figure)
    shutil.copyfile(
        delta_figure,
        coupled_figure_dir / delta_figure.name,
    )
    trace_figure = fast_figure_dir / "05_seed42轨迹对比.png"
    _plot_representative_traces(
        frames,
        steps=steps,
        seed=seeds[-1],
        output=trace_figure,
    )
    shutil.copyfile(
        trace_figure,
        coupled_figure_dir / trace_figure.name,
    )

    _write_manifests(package_roots, steps=steps, seeds=seeds)
    for method, package_root in package_roots.items():
        _write_readme(package_root, method=method, audit=audit)
        _write_hashes(package_root)

    return {
        "audit_passed": bool(audit["passed"]),
        "rollout_count": int(len(metrics)),
        "paired_cell_count": int(len(paired)),
        "fastslow_package": str(package_roots["fastslow"]),
        "coupled_package": str(package_roots["coupled"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze 3-checkpoint x 3-seed Teacher sensitivity."
    )
    parser.add_argument("--fastslow_package", required=True)
    parser.add_argument("--coupled_package", required=True)
    args = parser.parse_args()
    result = build_analysis(
        args.fastslow_package,
        args.coupled_package,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
