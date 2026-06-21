#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze phase2 teacher/student CSV and reproduce plots in the style of IsaaclabPlot12.5.py:
  Fig1  payload position (task frame) + swing angles   (teacher vs student)
  Fig2  payload swing angles (task frame)              (teacher vs student)
  Fig3  latent z comparison                            (student zH vs zT, optionally teacher zT)
  Fig4  energy dissipation verification (E_hat, dE/dt, P_xy, P_param, P_model)

Usage example:
  python scripts/rsl_rl/analyze_phase2_csv_v2.py \
    --teacher /path/to/phase2_teacher.csv \
    --student /path/to/phase2_student.csv \
    --goal 2 0 2 --start -2 0 2 \
    --time_window 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------

def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    x = np.asarray(x, dtype=float)
    k = np.ones(win, dtype=float) / float(win)
    return np.convolve(x, k, mode="same")

def _safe_unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.array([1.0, 0.0], dtype=float)
    return (v / n).astype(float)

def _gradient(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Robust gradient with non-uniform dt support."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(y) < 3:
        return np.zeros_like(y)
    return np.gradient(y, t)

def _infer_origin_from_goal(df: pd.DataFrame, goal_ref: np.ndarray) -> np.ndarray:
    """
    If CSV contains goal_px/py/pz in WORLD coordinates and goal_ref is TASK offset (e.g. [2,0,2]),
    then env_origin ≈ goal_world - goal_ref (constant per run/trace env).
    """
    for c in ["goal_px", "goal_py", "goal_pz"]:
        if c not in df.columns:
            return np.zeros(3, dtype=float)
    ox = np.nanmedian(df["goal_px"].to_numpy(dtype=float) - float(goal_ref[0]))
    oy = np.nanmedian(df["goal_py"].to_numpy(dtype=float) - float(goal_ref[1]))
    oz = np.nanmedian(df["goal_pz"].to_numpy(dtype=float) - float(goal_ref[2]))
    return np.array([ox, oy, oz], dtype=float)

def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

def _save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep current phase2 CSV column names, but allow some aliases for older scripts.
    """
    rename = {}

    # common aliases
    alias_groups = [
        (["time", "t", "t_sec"], "time_s"),
        (["uav_x", "uav_px_w"], "uav_px"),
        (["uav_y", "uav_py_w"], "uav_py"),
        (["uav_z", "uav_pz_w"], "uav_pz"),
        (["payload_x", "payload_px_w"], "payload_px"),
        (["payload_y", "payload_py_w"], "payload_py"),
        (["payload_z", "payload_pz_w"], "payload_pz"),
        (["theta_x", "swing_x_deg"], "theta_x_deg"),
        (["theta_y", "swing_y_deg"], "theta_y_deg"),
        (["theta_dot_x", "swing_vel_x_deg_s"], "theta_dot_x_deg_s"),
        (["theta_dot_y", "swing_vel_y_deg_s"], "theta_dot_y_deg_s"),
        (["rope_len", "L", "rope_length"], "rope_length_m"),
        (["payload_mass"], "payload_mass_kg"),
    ]

    cols_lower = {c.lower(): c for c in df.columns}
    for aliases, target in alias_groups:
        if target in df.columns:
            continue
        for a in aliases:
            if a in cols_lower:
                rename[cols_lower[a]] = target
                break

    if rename:
        df = df.rename(columns=rename)

    return df

@dataclass
class Series:
    name: str
    df: pd.DataFrame
    origin_w: np.ndarray
    start_ref: np.ndarray
    goal_ref: np.ndarray

    def t(self) -> np.ndarray:
        return self.df["time_s"].to_numpy(dtype=float)

    def add_task_frame(self) -> None:
        """Add *_x/_y/_z columns in TASK frame (subtract env origin)."""
        o = self.origin_w
        for prefix in ["uav", "payload", "goal"]:
            for axis, idx in [("x", 0), ("y", 1), ("z", 2)]:
                wcol = f"{prefix}_p{axis}"
                if wcol in self.df.columns:
                    self.df[f"{prefix}_{axis}"] = self.df[wcol].to_numpy(dtype=float) - float(o[idx])

    def add_derivatives(self) -> None:
        """Add uav_vx/vy/vz and uav_ax/ay/az in TASK frame by numerical differentiation."""
        t = self.t()
        for axis in ["x", "y", "z"]:
            pcol = f"uav_{axis}"
            if pcol not in self.df.columns:
                continue
            p = self.df[pcol].to_numpy(dtype=float)
            v = _gradient(p, t)
            a = _gradient(v, t)
            self.df[f"uav_v{axis}"] = v
            self.df[f"uav_a{axis}"] = a

    def clip_time(self, t_max: float | None) -> "Series":
        if t_max is None:
            return self
        m = self.t() <= float(t_max)
        df2 = self.df.loc[m].copy()
        return Series(self.name, df2, self.origin_w, self.start_ref, self.goal_ref)

def load_series(path: Path, name: str, start_ref: np.ndarray, goal_ref: np.ndarray) -> Series:
    df = pd.read_csv(path)
    df = _canon_cols(df)
    if "time_s" not in df.columns:
        raise ValueError(f"[{name}] CSV missing time_s column: {path}")
    origin = _infer_origin_from_goal(df, goal_ref)
    s = Series(name=name, df=df, origin_w=origin, start_ref=start_ref, goal_ref=goal_ref)
    s.add_task_frame()
    s.add_derivatives()
    return s


# ----------------------------
# Plots
# ----------------------------

def plot_fig1_payload_pos_and_swing(teacher: Series, student: Series, out_dir: Path) -> None:
    # match time range
    tmax = min(float(teacher.t()[-1]), float(student.t()[-1]))
    teacher = teacher.clip_time(tmax)
    student = student.clip_time(tmax)

    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    axs = axs.reshape(2, 3)
    # --- helper: only for plotting, not for metrics ---
    def _smooth(y, win=7):
        import numpy as np
        if win <= 1:
            return y
        k = np.ones(win) / win
        return np.convolve(y, k, mode="same")
    # payload position x/y/z
    refs = {
        "x": (teacher.goal_ref[0], "goal_x"),
        "y": (teacher.goal_ref[1], "goal_y"),
        "z": (teacher.goal_ref[2], "goal_z"),
    }
    for j, axis in enumerate(["x", "y", "z"]):
        ax = axs[0, j]
        yT = teacher.df[f"payload_{axis}"].to_numpy(dtype=float)
        yS = student.df[f"payload_{axis}"].to_numpy(dtype=float)

        # optional display smoothing
        yT_plot = _smooth(yT, win=7)
        yS_plot = _smooth(yS, win=7)

        ax.plot(teacher.t(), yT_plot, label="teacher")
        ax.plot(student.t(), yS_plot, label="student")
        ax.axhline(float(refs[axis][0]), linestyle="--", linewidth=1.0, label="ref")

        ax.set_title(f"Payload {axis.upper()} (task frame)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{axis} (m)")
        ax.grid(True, linestyle=":", linewidth=0.6)

        # ---- nicer axis limits ----
        if axis == "y":
            ax.set_ylim(-3.0, 3.0)

    # swing angles theta_x / theta_y
    for j, (col, title) in enumerate([("theta_x_deg", "Swing θx (deg)"), ("theta_y_deg", "Swing θy (deg)")]):
        ax = axs[1, j]

        thT = teacher.df[col].to_numpy(dtype=float)
        thS = student.df[col].to_numpy(dtype=float)

        # optional display smoothing
        thT_plot = _smooth(thT, win=7)
        thS_plot = _smooth(thS, win=7)

        ax.plot(teacher.t(), thT_plot, label="teacher")
        ax.plot(student.t(), thS_plot, label="student")
        ax.axhline(0.0, linestyle="--", linewidth=1.0, label="ref")

        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("deg")
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.set_ylim(-30.0, 30.0)

    # last panel: payload error norm (optional quick sanity)
    ax = axs[1, 2]
    if all(c in teacher.df.columns for c in ["payload_err_x", "payload_err_y", "payload_err_z"]):
        eT = np.linalg.norm(teacher.df[["payload_err_x","payload_err_y","payload_err_z"]].to_numpy(dtype=float), axis=1)
        eS = np.linalg.norm(student.df[["payload_err_x","payload_err_y","payload_err_z"]].to_numpy(dtype=float), axis=1)
        ax.plot(teacher.t(), eT, label="teacher")
        ax.plot(student.t(), eS, label="student")
        ax.set_ylabel("||e|| (m)")
    ax.set_title("Payload error norm")
    ax.set_xlabel("Time (s)")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()

    _save_fig(fig, out_dir / "fig1_payload_pos_and_swing.png")

def plot_fig2_swing_angles(teacher: Series, student: Series, out_dir: Path) -> None:
    tmax = min(float(teacher.t()[-1]), float(student.t()[-1]))
    teacher = teacher.clip_time(tmax)
    student = student.clip_time(tmax)

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axs[0].plot(teacher.t(), teacher.df["theta_x_deg"], label="teacher")
    axs[0].plot(student.t(), student.df["theta_x_deg"], label="student")
    axs[0].axhline(0.0, linestyle="--", linewidth=1.0, label="ref")
    axs[0].set_ylabel("deg")
    axs[0].set_title("Payload swing angle θx")
    axs[0].grid(True, linestyle=":", linewidth=0.6)
    axs[0].legend()

    axs[1].plot(teacher.t(), teacher.df["theta_y_deg"], label="teacher")
    axs[1].plot(student.t(), student.df["theta_y_deg"], label="student")
    axs[1].axhline(0.0, linestyle="--", linewidth=1.0, label="ref")
    axs[1].set_ylabel("deg")
    axs[1].set_title("Payload swing angle θy")
    axs[1].grid(True, linestyle=":", linewidth=0.6)
    axs[1].legend()

    # magnitude
    thT = np.sqrt(teacher.df["theta_x_deg"].to_numpy(dtype=float)**2 + teacher.df["theta_y_deg"].to_numpy(dtype=float)**2)
    thS = np.sqrt(student.df["theta_x_deg"].to_numpy(dtype=float)**2 + student.df["theta_y_deg"].to_numpy(dtype=float)**2)
    axs[2].plot(teacher.t(), thT, label="teacher")
    axs[2].plot(student.t(), thS, label="student")
    axs[2].axhline(0.0, linestyle="--", linewidth=1.0, label="ref")
    axs[2].set_ylabel("deg")
    axs[2].set_title("Swing magnitude sqrt(θx^2+θy^2)")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True, linestyle=":", linewidth=0.6)
    axs[2].legend()

    _save_fig(fig, out_dir / "fig2_swing_angles.png")

def plot_fig3_z_compare(teacher: Series, student: Series, out_dir: Path) -> None:
    # student must have zT* and zH*
    need = [f"zT{i}" for i in range(5)] + [f"zH{i}" for i in range(5)]
    for c in need:
        if c not in student.df.columns:
            raise ValueError(f"[student] CSV missing {c}")

    tS = student.t()
    tmax = float(tS[-1])

    # teacher zT optional
    have_teacher_zT = all(f"zT{i}" in teacher.df.columns for i in range(5))

    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    for i in range(5):
        ax = axs[i]
        ax.plot(tS, student.df[f"zT{i}"], label="student:zT (teacher mu(priv))")
        ax.plot(tS, student.df[f"zH{i}"], label="student:zH (encoder)")
        if have_teacher_zT:
            # align by time index (same dt) – good enough for visual reference
            tT = teacher.t()
            m = tT <= tmax
            ax.plot(tT[m], teacher.df.loc[m, f"zT{i}"], linestyle="--", linewidth=1.0, label="teacher:zT")
        ax.set_ylabel(f"z[{i}]")
        ax.grid(True, linestyle=":", linewidth=0.6)
        if i == 0:
            ax.legend()

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Latent z comparison (student encoder vs teacher priv->mu)", y=0.995)
    _save_fig(fig, out_dir / "fig3_z_compare.png")

def _compute_energy(df: pd.DataFrame, goal_ref: np.ndarray, smooth_window_s: float = 0.15):
    """
    Re-implement the core of IsaaclabPlot12.5 Figure-6 combined plots, using
    columns available in phase2 CSV (derive UAV accel from position).
    """
    g0 = 9.81

    t = df["time_s"].to_numpy(dtype=float)
    if len(t) < 3:
        return None

    # estimate dt and smoothing window
    dt = float(np.nanmedian(np.diff(t)))
    win = max(1, int(round(float(smooth_window_s) / dt)))

    # rope length: prefer rope_length_m in CSV
    if "rope_length_m" in df.columns:
        L_est = float(np.nanmedian(df["rope_length_m"].to_numpy(dtype=float)))
    else:
        L_est = 1.0

    # direction in XY: use UAV displacement in TASK frame
    dx = float(df["uav_x"].iloc[-1] - df["uav_x"].iloc[0])
    dy = float(df["uav_y"].iloc[-1] - df["uav_y"].iloc[0])
    d_xy = _safe_unit(np.array([dx, dy], dtype=float))

    # swing angle and swing velocity (rad / rad/s)
    thx = np.deg2rad(df["theta_x_deg"].to_numpy(dtype=float))
    thy = np.deg2rad(df["theta_y_deg"].to_numpy(dtype=float))
    thdx = np.deg2rad(df["theta_dot_x_deg_s"].to_numpy(dtype=float))
    thdy = np.deg2rad(df["theta_dot_y_deg_s"].to_numpy(dtype=float))

    th2 = thx**2 + thy**2
    thd2 = thdx**2 + thdy**2

    # UAV accelerations (m/s^2) in TASK frame – derived
    ax = df.get("uav_ax", pd.Series(np.zeros_like(t))).to_numpy(dtype=float)
    ay = df.get("uav_ay", pd.Series(np.zeros_like(t))).to_numpy(dtype=float)
    az = df.get("uav_az", pd.Series(np.zeros_like(t))).to_numpy(dtype=float)

    # parallel components (for mechanism plot)
    v_swing_parallel = df["theta_dot_x_deg_s"].to_numpy(dtype=float) * d_xy[0] + df["theta_dot_y_deg_s"].to_numpy(dtype=float) * d_xy[1]
    a_uav_parallel = ax * d_xy[0] + ay * d_xy[1]

    # w2 and energy
    w2 = (g0 + az) / max(L_est, 1e-6)
    w2_dot = _gradient(w2, t)
    E_hat = 0.5 * (thd2 + w2 * th2)

    # power terms (verification / decomposition)
    P_xy = (ax * thdx + ay * thdy) / max(L_est, 1e-6)
    P_param = 0.5 * w2_dot * th2
    P_model = P_xy + P_param
    E_dot_num = _gradient(E_hat, t)

    # smoothing for plots
    out = dict(
        t=t,
        dt=dt,
        L_est=L_est,
        d_xy=d_xy,
        v_swing_parallel=_moving_average(v_swing_parallel, win),
        a_uav_parallel=_moving_average(a_uav_parallel, win),
        E_hat=_moving_average(E_hat, win),
        E_dot_num=_moving_average(E_dot_num, win),
        P_xy=_moving_average(P_xy, win),
        P_param=_moving_average(P_param, win),
        P_model=_moving_average(P_model, win),
    )
    return out

def plot_fig4_energy_combined(teacher: Series, student: Series, out_dir: Path, time_window_s: float | None, smooth_window_s: float) -> None:
    # clip time (energy plot focuses on early window, like IsaaclabPlot12.5)
    teacher_c = teacher.clip_time(time_window_s)
    student_c = student.clip_time(time_window_s)

    ET = _compute_energy(teacher_c.df, teacher_c.goal_ref, smooth_window_s=smooth_window_s)
    ES = _compute_energy(student_c.df, student_c.goal_ref, smooth_window_s=smooth_window_s)
    if ET is None or ES is None:
        print("[WARN] Not enough samples for energy plot.")
        return

    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # (1) mechanism: v_swing_parallel & a_uav_parallel (twin y)
    ax1 = axs[0]
    ax1.plot(ET["t"], ET["v_swing_parallel"], label="teacher: v_swing_parallel (deg/s)")
    ax1.plot(ES["t"], ES["v_swing_parallel"], label="student: v_swing_parallel (deg/s)")
    ax1.set_ylabel("deg/s")
    ax1.grid(True, linestyle=":", linewidth=0.6)
    ax1.set_title("Mechanism: swing vel parallel vs UAV accel parallel")

    ax1b = ax1.twinx()
    ax1b.plot(ET["t"], ET["a_uav_parallel"], linestyle="--", linewidth=1.0, label="teacher: a_uav_parallel (m/s^2)")
    ax1b.plot(ES["t"], ES["a_uav_parallel"], linestyle="--", linewidth=1.0, label="student: a_uav_parallel (m/s^2)")
    ax1b.set_ylabel("m/s²")

    # merge legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    # (2) energy
    axs[1].plot(ET["t"], ET["E_hat"], label="teacher: E_hat")
    axs[1].plot(ES["t"], ES["E_hat"], label="student: E_hat")
    axs[1].set_ylabel("E_hat (arb.)")
    axs[1].set_title("Energy E_hat")
    axs[1].grid(True, linestyle=":", linewidth=0.6)
    axs[1].legend()

    # (3) verification: dE/dt vs P_model
    axs[2].plot(ET["t"], ET["E_dot_num"], label="teacher: dE/dt")
    axs[2].plot(ET["t"], ET["P_model"], linestyle="--", linewidth=1.0, label="teacher: P_model")
    axs[2].plot(ES["t"], ES["E_dot_num"], label="student: dE/dt")
    axs[2].plot(ES["t"], ES["P_model"], linestyle="--", linewidth=1.0, label="student: P_model")
    axs[2].set_ylabel("rate (arb.)")
    axs[2].set_title("Verification: dE/dt ≈ P_model")
    axs[2].grid(True, linestyle=":", linewidth=0.6)
    axs[2].legend(ncol=2)

    # (4) decomposition
    axs[3].plot(ET["t"], ET["P_model"], label="teacher: P_model")
    axs[3].plot(ET["t"], ET["P_xy"], linestyle="--", linewidth=1.0, label="teacher: P_xy")
    axs[3].plot(ET["t"], ET["P_param"], linestyle=":", linewidth=1.0, label="teacher: P_param")

    axs[3].plot(ES["t"], ES["P_model"], label="student: P_model")
    axs[3].plot(ES["t"], ES["P_xy"], linestyle="--", linewidth=1.0, label="student: P_xy")
    axs[3].plot(ES["t"], ES["P_param"], linestyle=":", linewidth=1.0, label="student: P_param")

    axs[3].set_ylabel("power (arb.)")
    axs[3].set_title("Decomposition: P_model = P_xy + P_param")
    axs[3].set_xlabel("Time (s)")
    axs[3].grid(True, linestyle=":", linewidth=0.6)
    axs[3].legend(ncol=2)

    _save_fig(fig, out_dir / "fig4_energy_combined.png")




# ----------------------------
# Cross-method Phase-II comparison
# ----------------------------

def _payload_error_norm(df: pd.DataFrame, goal_ref: np.ndarray | None = None) -> np.ndarray:
    """Return payload-to-goal error norm. Prefer logged payload_err_* columns."""
    if all(c in df.columns for c in ["payload_err_x", "payload_err_y", "payload_err_z"]):
        return np.linalg.norm(df[["payload_err_x", "payload_err_y", "payload_err_z"]].to_numpy(dtype=float), axis=1)
    if goal_ref is None:
        raise ValueError("goal_ref is required when payload_err_* columns are missing")
    p = df[["payload_x", "payload_y", "payload_z"]].to_numpy(dtype=float)
    return np.linalg.norm(p - goal_ref.reshape(1, 3), axis=1)


def _swing_mag_deg(df: pd.DataFrame) -> np.ndarray:
    return np.sqrt(
        df["theta_x_deg"].to_numpy(dtype=float) ** 2
        + df["theta_y_deg"].to_numpy(dtype=float) ** 2
    )


def _payload_speed(df: pd.DataFrame) -> np.ndarray:
    t = df["time_s"].to_numpy(dtype=float)
    p = df[["payload_x", "payload_y", "payload_z"]].to_numpy(dtype=float)
    v = np.gradient(p, t, axis=0)
    return np.linalg.norm(v, axis=1)


def _interp_to(t_ref: np.ndarray, t_src: np.ndarray, y_src: np.ndarray) -> np.ndarray:
    return np.interp(t_ref, t_src, y_src)


def _method_metrics(method: str, role: str, s: Series) -> dict:
    df = s.df
    t = df["time_s"].to_numpy(dtype=float)
    err = _payload_error_norm(df, s.goal_ref)
    swing = _swing_mag_deg(df)
    speed = _payload_speed(df)

    def first_hit(thresh: float):
        idx = np.where(err <= thresh)[0]
        if len(idx) == 0:
            return np.nan, np.nan
        i = int(idx[0])
        return float(t[i]), float(speed[i])

    t02, v02 = first_hit(0.2)
    t01, v01 = first_hit(0.1)

    row = {
        "method": method,
        "role": role,
        "hit_t_0p2_s": t02,
        "speed_at_0p2_mps": v02,
        "hit_t_0p1_s": t01,
        "speed_at_0p1_mps": v01,
        "final_error_m": float(err[-1]),
        "tail5_error_m": float(np.nanmean(err[t >= max(0.0, t[-1] - 5.0)])),
        "max_swing_deg": float(np.nanmax(swing)),
        "mean_swing_deg": float(np.nanmean(swing)),
    }

    energy = _compute_energy(df, s.goal_ref, smooth_window_s=0.15)
    if energy is not None:
        row["E_hat_mean"] = float(np.nanmean(energy["E_hat"]))
        row["E_hat_peak"] = float(np.nanmax(energy["E_hat"]))
        e5_mask = energy["t"] >= 5.0
        row["E_hat_after5_mean"] = float(np.nanmean(energy["E_hat"][e5_mask])) if np.any(e5_mask) else np.nan
    return row


def _pair_gap_metrics(method: str, teacher: Series, student: Series) -> dict:
    """Student-teacher realization gap for task-level curves."""
    tT = teacher.t()
    tS = student.t()
    t0 = max(float(tT[0]), float(tS[0]))
    t1 = min(float(tT[-1]), float(tS[-1]))
    m = (tT >= t0) & (tT <= t1)
    t = tT[m]

    eT = _payload_error_norm(teacher.df, teacher.goal_ref)[m]
    eS = _interp_to(t, tS, _payload_error_norm(student.df, student.goal_ref))
    thT = _swing_mag_deg(teacher.df)[m]
    thS = _interp_to(t, tS, _swing_mag_deg(student.df))

    return {
        "method": method,
        "mean_abs_error_gap_m": float(np.nanmean(np.abs(eS - eT))),
        "max_abs_error_gap_m": float(np.nanmax(np.abs(eS - eT))),
        "mean_abs_swing_gap_deg": float(np.nanmean(np.abs(thS - thT))),
        "max_abs_swing_gap_deg": float(np.nanmax(np.abs(thS - thT))),
    }


def plot_phase2_cross_method_compare(
    dec_teacher: Series,
    dec_student: Series,
    coup_teacher: Series,
    coup_student: Series,
    out_dir: Path,
    time_window_s: float | None,
    smooth_window_s: float,
    labels: tuple[str, str] = ("Decoupled", "Coupled"),
) -> None:
    """
    Compare Decoupled and Coupled Phase-II results in one seed.

    Important: do not directly compare z0/z1 between methods. Decoupled z0/z1 are semantic
    mass/length contexts, while Coupled z dimensions are black-box latent coordinates.
    Therefore this function compares task curves and student-teacher realization gaps.
    """
    _ensure_out_dir(out_dir)

    series = [dec_teacher, dec_student, coup_teacher, coup_student]
    names = [f"{labels[0]} teacher", f"{labels[0]} student", f"{labels[1]} teacher", f"{labels[1]} student"]

    # clip all to common duration and optional requested window
    common_tmax = min(float(s.t()[-1]) for s in series)
    if time_window_s is not None:
        common_tmax = min(common_tmax, float(time_window_s))
    series = [s.clip_time(common_tmax) for s in series]
    dec_teacher, dec_student, coup_teacher, coup_student = series

    # -------------------- Fig A: task curves --------------------
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Phase-II cross-method comparison: task curves", fontsize=14, weight="bold")

    for s, name in zip(series, names):
        t = s.t()
        axs[0].plot(t, _payload_error_norm(s.df, s.goal_ref), linewidth=1.8, label=name)
        axs[1].plot(t, _swing_mag_deg(s.df), linewidth=1.8, label=name)

    axs[0].axhline(0.2, linestyle=":", linewidth=1.0, label="0.2 m")
    axs[0].axhline(0.1, linestyle="--", linewidth=1.0, label="0.1 m")
    axs[0].set_ylabel("Payload error (m)")
    axs[0].set_title("Payload-to-goal error")

    axs[1].set_ylabel("Swing magnitude (deg)")
    axs[1].set_title("Payload swing magnitude")

    # z_rmse only compares teacher-vs-student within each method, not raw latent dimensions across methods
    if "z_rmse" in dec_student.df.columns:
        axs[2].plot(dec_student.t(), dec_student.df["z_rmse"].to_numpy(dtype=float), linewidth=1.8, label=f"{labels[0]} z RMSE")
    if "z_rmse" in coup_student.df.columns:
        axs[2].plot(coup_student.t(), coup_student.df["z_rmse"].to_numpy(dtype=float), linewidth=1.8, label=f"{labels[1]} z RMSE")
    axs[2].set_ylabel("z RMSE")
    axs[2].set_title("Student latent realization error")
    axs[2].set_xlabel("Time (s)")

    for ax in axs:
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.legend(fontsize="small", ncol=2)

    _save_fig(fig, out_dir / "phase2_compare_task_curves.png")

    # -------------------- Fig B: student-teacher gap curves --------------------
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Phase-II student-teacher realization gap", fontsize=14, weight="bold")

    def plot_gap(ax_err, ax_swing, teacher: Series, student: Series, label: str):
        tT = teacher.t()
        tS = student.t()
        t0 = max(float(tT[0]), float(tS[0]))
        t1 = min(float(tT[-1]), float(tS[-1]))
        m = (tT >= t0) & (tT <= t1)
        t = tT[m]
        eT = _payload_error_norm(teacher.df, teacher.goal_ref)[m]
        eS = _interp_to(t, tS, _payload_error_norm(student.df, student.goal_ref))
        thT = _swing_mag_deg(teacher.df)[m]
        thS = _interp_to(t, tS, _swing_mag_deg(student.df))
        ax_err.plot(t, np.abs(eS - eT), linewidth=1.8, label=label)
        ax_swing.plot(t, np.abs(thS - thT), linewidth=1.8, label=label)

    plot_gap(axs[0], axs[1], dec_teacher, dec_student, labels[0])
    plot_gap(axs[0], axs[1], coup_teacher, coup_student, labels[1])

    axs[0].set_ylabel("|student-teacher| error gap (m)")
    axs[0].set_title("Payload error realization gap")
    axs[1].set_ylabel("|student-teacher| swing gap (deg)")
    axs[1].set_title("Swing realization gap")
    axs[1].set_xlabel("Time (s)")
    for ax in axs:
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.legend()
    _save_fig(fig, out_dir / "phase2_compare_realization_gap.png")

    # -------------------- Fig C: decoupled semantic z only --------------------
    # This is for the proposed method only. Coupled latent dimensions are not semantically aligned.
    if all(c in dec_student.df.columns for c in ["zH0", "zH1", "priv0", "priv1"]):
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle("Decoupled semantic context recovery", fontsize=14, weight="bold")
        t = dec_student.t()
        axs[0].plot(t, dec_student.df["zH0"].to_numpy(dtype=float), label="student zH0")
        if "zT0" in dec_student.df.columns:
            axs[0].plot(t, dec_student.df["zT0"].to_numpy(dtype=float), linestyle="--", label="teacher zT0")
        axs[0].axhline(float(np.nanmedian(dec_student.df["priv0"].to_numpy(dtype=float))), linestyle=":", label="true mass")
        axs[0].set_ylim(0.0, 1.0)
        axs[0].set_ylabel("mass context")
        axs[0].grid(True, linestyle=":", linewidth=0.6)
        axs[0].legend()

        axs[1].plot(t, dec_student.df["zH1"].to_numpy(dtype=float), label="student zH1")
        if "zT1" in dec_student.df.columns:
            axs[1].plot(t, dec_student.df["zT1"].to_numpy(dtype=float), linestyle="--", label="teacher zT1")
        axs[1].axhline(float(np.nanmedian(dec_student.df["priv1"].to_numpy(dtype=float))), linestyle=":", label="true length")
        axs[1].set_ylim(0.0, 1.0)
        axs[1].set_ylabel("length context")
        axs[1].set_xlabel("Time (s)")
        axs[1].grid(True, linestyle=":", linewidth=0.6)
        axs[1].legend()
        _save_fig(fig, out_dir / "phase2_decoupled_semantic_z01.png")

    # -------------------- CSV summary --------------------
    rows = []
    rows.append(_method_metrics(labels[0], "teacher", dec_teacher))
    rows.append(_method_metrics(labels[0], "student", dec_student))
    rows.append(_method_metrics(labels[1], "teacher", coup_teacher))
    rows.append(_method_metrics(labels[1], "student", coup_student))
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "phase2_compare_metrics.csv", index=False)

    gaps_df = pd.DataFrame([
        _pair_gap_metrics(labels[0], dec_teacher, dec_student),
        _pair_gap_metrics(labels[1], coup_teacher, coup_student),
    ])
    gaps_df.to_csv(out_dir / "phase2_compare_realization_gap_metrics.csv", index=False)

    print("[Compare] saved:")
    print("  - phase2_compare_task_curves.png")
    print("  - phase2_compare_realization_gap.png")
    print("  - phase2_decoupled_semantic_z01.png (if columns exist)")
    print("  - phase2_compare_metrics.csv")
    print("  - phase2_compare_realization_gap_metrics.csv")




def plot_phase2_multiseed_compare(
    dec_teachers: list[Series],
    dec_students: list[Series],
    coup_teachers: list[Series],
    coup_students: list[Series],
    seed_labels: list[str],
    out_dir: Path,
    time_window_s: float | None,
    labels: tuple[str, str] = ("Decoupled", "Coupled"),
) -> None:
    """
    Plot Decoupled vs Coupled Phase-II results for multiple evaluation seeds in ONE figure.

    This is the correct multi-seed visualization for the paper:
      - rows = seeds
      - columns = task-level quantities
      - curves = dec/coup teacher/student within each seed

    We do NOT directly compare decoupled z0/z1 against coupled z0/z1, because
    decoupled z0/z1 are semantic mass/length variables, while coupled z dimensions
    are black-box latent coordinates.
    """
    _ensure_out_dir(out_dir)

    n = len(seed_labels)
    if not (len(dec_teachers) == len(dec_students) == len(coup_teachers) == len(coup_students) == n):
        raise ValueError("multi-seed lists must have the same length")

    # -------------------- Figure 1: student/teacher task curves across seeds --------------------
    fig, axs = plt.subplots(n, 2, figsize=(14, max(3.2 * n, 4.0)), sharex=False)
    if n == 1:
        axs = np.asarray([axs])

    fig.suptitle("Phase-II multi-seed comparison: Decoupled vs Coupled", fontsize=14, weight="bold")

    for i, seed in enumerate(seed_labels):
        dT, dS, cT, cS = dec_teachers[i], dec_students[i], coup_teachers[i], coup_students[i]
        common_tmax = min(float(s.t()[-1]) for s in [dT, dS, cT, cS])
        if time_window_s is not None:
            common_tmax = min(common_tmax, float(time_window_s))
        dT, dS, cT, cS = [s.clip_time(common_tmax) for s in [dT, dS, cT, cS]]

        ax_e = axs[i, 0]
        ax_s = axs[i, 1]

        # Error norm: teacher dashed, student solid
        ax_e.plot(dT.t(), _payload_error_norm(dT.df, dT.goal_ref), linestyle="--", linewidth=1.2, label=f"{labels[0]} teacher")
        ax_e.plot(dS.t(), _payload_error_norm(dS.df, dS.goal_ref), linestyle="-", linewidth=1.8, label=f"{labels[0]} student")
        ax_e.plot(cT.t(), _payload_error_norm(cT.df, cT.goal_ref), linestyle="--", linewidth=1.2, label=f"{labels[1]} teacher")
        ax_e.plot(cS.t(), _payload_error_norm(cS.df, cS.goal_ref), linestyle="-", linewidth=1.8, label=f"{labels[1]} student")
        ax_e.axhline(0.2, linestyle=":", linewidth=0.9)
        ax_e.axhline(0.1, linestyle=":", linewidth=0.9)
        ax_e.set_title(f"{seed}: payload error")
        ax_e.set_ylabel("error (m)")
        ax_e.grid(True, linestyle=":", linewidth=0.6)

        # Swing magnitude
        ax_s.plot(dT.t(), _swing_mag_deg(dT.df), linestyle="--", linewidth=1.2, label=f"{labels[0]} teacher")
        ax_s.plot(dS.t(), _swing_mag_deg(dS.df), linestyle="-", linewidth=1.8, label=f"{labels[0]} student")
        ax_s.plot(cT.t(), _swing_mag_deg(cT.df), linestyle="--", linewidth=1.2, label=f"{labels[1]} teacher")
        ax_s.plot(cS.t(), _swing_mag_deg(cS.df), linestyle="-", linewidth=1.8, label=f"{labels[1]} student")
        ax_s.set_title(f"{seed}: swing magnitude")
        ax_s.set_ylabel("swing (deg)")
        ax_s.grid(True, linestyle=":", linewidth=0.6)

        if i == n - 1:
            ax_e.set_xlabel("Time (s)")
            ax_s.set_xlabel("Time (s)")
        if i == 0:
            ax_e.legend(fontsize="small", ncol=2)
            ax_s.legend(fontsize="small", ncol=2)

    _save_fig(fig, out_dir / "phase2_multiseed_task_curves.png")

    # -------------------- Figure 2: student-teacher realization gap across seeds --------------------
    fig, axs = plt.subplots(n, 2, figsize=(14, max(3.2 * n, 4.0)), sharex=False)
    if n == 1:
        axs = np.asarray([axs])

    fig.suptitle("Phase-II multi-seed student-teacher realization gap", fontsize=14, weight="bold")

    def _gap_arrays(teacher: Series, student: Series):
        tT = teacher.t()
        tS = student.t()
        t0 = max(float(tT[0]), float(tS[0]))
        t1 = min(float(tT[-1]), float(tS[-1]))
        if time_window_s is not None:
            t1 = min(t1, float(time_window_s))
        m = (tT >= t0) & (tT <= t1)
        t = tT[m]
        eT = _payload_error_norm(teacher.df, teacher.goal_ref)[m]
        eS = _interp_to(t, tS, _payload_error_norm(student.df, student.goal_ref))
        thT = _swing_mag_deg(teacher.df)[m]
        thS = _interp_to(t, tS, _swing_mag_deg(student.df))
        return t, np.abs(eS - eT), np.abs(thS - thT)

    gap_rows = []
    metric_rows = []
    for i, seed in enumerate(seed_labels):
        dT, dS, cT, cS = dec_teachers[i], dec_students[i], coup_teachers[i], coup_students[i]

        td, ed, sd = _gap_arrays(dT, dS)
        tc, ec, sc = _gap_arrays(cT, cS)

        axs[i, 0].plot(td, ed, linewidth=1.8, label=labels[0])
        axs[i, 0].plot(tc, ec, linewidth=1.8, label=labels[1])
        axs[i, 0].set_title(f"{seed}: payload error gap")
        axs[i, 0].set_ylabel("|student-teacher| (m)")
        axs[i, 0].grid(True, linestyle=":", linewidth=0.6)

        axs[i, 1].plot(td, sd, linewidth=1.8, label=labels[0])
        axs[i, 1].plot(tc, sc, linewidth=1.8, label=labels[1])
        axs[i, 1].set_title(f"{seed}: swing gap")
        axs[i, 1].set_ylabel("|student-teacher| (deg)")
        axs[i, 1].grid(True, linestyle=":", linewidth=0.6)

        if i == n - 1:
            axs[i, 0].set_xlabel("Time (s)")
            axs[i, 1].set_xlabel("Time (s)")
        if i == 0:
            axs[i, 0].legend()
            axs[i, 1].legend()

        r = _pair_gap_metrics(labels[0], dT, dS); r["seed"] = seed; gap_rows.append(r)
        r = _pair_gap_metrics(labels[1], cT, cS); r["seed"] = seed; gap_rows.append(r)

        for method, role, s in [
            (labels[0], "teacher", dT),
            (labels[0], "student", dS),
            (labels[1], "teacher", cT),
            (labels[1], "student", cS),
        ]:
            row = _method_metrics(method, role, s)
            row["seed"] = seed
            metric_rows.append(row)

    _save_fig(fig, out_dir / "phase2_multiseed_realization_gap.png")

    metrics_df = pd.DataFrame(metric_rows)
    # Put seed first for readability
    metrics_df = metrics_df[["seed"] + [c for c in metrics_df.columns if c != "seed"]]
    metrics_df.to_csv(out_dir / "phase2_multiseed_metrics.csv", index=False)

    gaps_df = pd.DataFrame(gap_rows)
    gaps_df = gaps_df[["seed"] + [c for c in gaps_df.columns if c != "seed"]]
    gaps_df.to_csv(out_dir / "phase2_multiseed_gap_metrics.csv", index=False)

    # Mean/std summary for paper tables
    summary_cols = [
        "hit_t_0p2_s", "speed_at_0p2_mps", "hit_t_0p1_s", "speed_at_0p1_mps",
        "final_error_m", "tail5_error_m", "max_swing_deg", "mean_swing_deg",
        "E_hat_mean", "E_hat_peak", "E_hat_after5_mean",
    ]
    exist = [c for c in summary_cols if c in metrics_df.columns]
    summary = metrics_df.groupby(["method", "role"])[exist].agg(["mean", "std"])
    summary.to_csv(out_dir / "phase2_multiseed_metrics_mean_std.csv")

    gap_summary_cols = [
        "mean_abs_error_gap_m", "max_abs_error_gap_m",
        "mean_abs_swing_gap_deg", "max_abs_swing_gap_deg",
    ]
    exist_g = [c for c in gap_summary_cols if c in gaps_df.columns]
    gap_summary = gaps_df.groupby(["method"])[exist_g].agg(["mean", "std"])
    gap_summary.to_csv(out_dir / "phase2_multiseed_gap_mean_std.csv")

    print("[MultiSeed] saved:")
    print("  - phase2_multiseed_task_curves.png")
    print("  - phase2_multiseed_realization_gap.png")
    print("  - phase2_multiseed_metrics.csv")
    print("  - phase2_multiseed_gap_metrics.csv")
    print("  - phase2_multiseed_metrics_mean_std.csv")
    print("  - phase2_multiseed_gap_mean_std.csv")



def _available_z_dim(student: Series, max_dim: int = 5) -> int:
    """Infer available z dimensions from zT*/zH* columns in a phase2 student CSV."""
    dims = []
    for i in range(max_dim):
        if (f"zH{i}" in student.df.columns) or (f"zT{i}" in student.df.columns):
            dims.append(i)
    return (max(dims) + 1) if dims else 0


def _get_z_teacher_student_for_plot(teacher: Series, student: Series, i: int):
    """
    Return (t, z_teacher, z_student) for z_i.
    Prefer student CSV's zT_i and zH_i because they are logged on the same time base.
    Fallback to teacher CSV's zT_i interpolated onto student time.
    """
    tS = student.t()

    # student-estimated latent
    if f"zH{i}" in student.df.columns:
        z_student = student.df[f"zH{i}"].to_numpy(dtype=float)
    elif f"z{i}" in student.df.columns:
        z_student = student.df[f"z{i}"].to_numpy(dtype=float)
    else:
        z_student = np.full_like(tS, np.nan, dtype=float)

    # teacher/reference latent on student time base
    if f"zT{i}" in student.df.columns:
        z_teacher = student.df[f"zT{i}"].to_numpy(dtype=float)
    elif f"zT{i}" in teacher.df.columns:
        z_teacher = _interp_to(tS, teacher.t(), teacher.df[f"zT{i}"].to_numpy(dtype=float))
    elif f"z{i}" in teacher.df.columns:
        z_teacher = _interp_to(tS, teacher.t(), teacher.df[f"z{i}"].to_numpy(dtype=float))
    else:
        z_teacher = np.full_like(tS, np.nan, dtype=float)

    return tS, z_teacher, z_student


def _plot_one_method_multiseed_z(
    method_label: str,
    teachers: list[Series],
    students: list[Series],
    seed_labels: list[str],
    out_path: Path,
    time_window_s: float | None,
    z_dim: int = 5,
    semantic_z01: bool = False,
) -> None:
    """
    One method, all seeds, teacher/student z in ONE compact figure.

    Correct layout for Phase-II latent comparison:
      rows = z dimensions
      one subplot per z dimension
      within each subplot: all seeds are overlaid
        - same color = same seed
        - dashed = teacher zT
        - solid  = student zH

    This matches the intended figure: seed38/40/42 are all shown inside the same z0 subplot,
    not split into separate rows.
    """
    n = len(seed_labels)
    if n == 0:
        return

    # infer actual z dim, but cap at z_dim
    actual = max([_available_z_dim(stu, max_dim=z_dim) for stu in students] + [0])
    if actual <= 0:
        print(f"[WARN] {method_label}: no z columns found, skip {out_path.name}")
        return
    z_dim = min(z_dim, actual)

    # Taller figure: one row per z dimension. This is compact and paper-friendly.
    fig, axs = plt.subplots(
        z_dim,
        1,
        figsize=(12.0, max(2.0 * z_dim, 4.0)),
        sharex=True,
        squeeze=False,
    )
    axs = axs[:, 0]

    fig.suptitle(f"{method_label}: Phase-II latent z, multi-seed teacher/student overlay", fontsize=14, weight="bold")

    # Use matplotlib default color cycle, but bind color to seed.
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for c in range(z_dim):
        ax = axs[c]
        for r, seed in enumerate(seed_labels):
            teacher = teachers[r]
            student = students[r]
            t, zT, zH = _get_z_teacher_student_for_plot(teacher, student, c)
            if time_window_s is not None:
                m = t <= float(time_window_s)
                t_plot, zT_plot, zH_plot = t[m], zT[m], zH[m]
            else:
                t_plot, zT_plot, zH_plot = t, zT, zH

            color = color_cycle[r % len(color_cycle)] if color_cycle else None

            # teacher: dashed; student: solid. Same color = same seed.
            ax.plot(
                t_plot,
                zT_plot,
                linestyle="--",
                linewidth=1.3,
                color=color,
                label=f"{seed} teacher zT" if c == 0 else None,
            )
            ax.plot(
                t_plot,
                zH_plot,
                linestyle="-",
                linewidth=1.7,
                color=color,
                label=f"{seed} student zH" if c == 0 else None,
            )

        # Semantic y-axis for decoupled mass/length only.
        if semantic_z01 and c == 0:
            ax.set_title("Mass latent z0 (normalized)", fontsize=11)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("z0 norm")
        elif semantic_z01 and c == 1:
            ax.set_title("Length latent z1 (normalized)", fontsize=11)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("z1 norm")
        else:
            ax.set_title(f"Latent z{c} (raw)", fontsize=11)
            ax.set_ylabel(f"z{c}")

        ax.grid(True, linestyle=":", linewidth=0.6)

    axs[-1].set_xlabel("Time (s)")

    # Put legend only once, outside the first subplot to reduce clutter.
    handles, labels_ = axs[0].get_legend_handles_labels()
    if handles:
        axs[0].legend(handles, labels_, fontsize="small", ncol=min(3, max(1, n)), loc="best")

    _save_fig(fig, out_path)

def plot_phase2_multiseed_z_outputs(
    dec_teachers: list[Series],
    dec_students: list[Series],
    coup_teachers: list[Series],
    coup_students: list[Series],
    seed_labels: list[str],
    out_dir: Path,
    time_window_s: float | None,
    labels: tuple[str, str] = ("Decoupled", "Coupled"),
    z_dim: int = 5,
) -> None:
    """
    Plot z outputs across three seeds.

    Outputs:
      1) phase2_multiseed_z_decoupled.png: Decoupled 3 seeds, teacher/student zT/zH.
      2) phase2_multiseed_z_coupled.png: Coupled 3 seeds, teacher/student zT/zH.
      3) phase2_multiseed_z_all_methods.png: Decoupled and Coupled in one large figure.

    This compares teacher-vs-student within each method. It does NOT claim that
    Decoupled z0 is semantically comparable to Coupled z0.
    """
    _ensure_out_dir(out_dir)

    _plot_one_method_multiseed_z(
        method_label=labels[0],
        teachers=dec_teachers,
        students=dec_students,
        seed_labels=seed_labels,
        out_path=out_dir / "phase2_multiseed_z_decoupled.png",
        time_window_s=time_window_s,
        z_dim=z_dim,
        semantic_z01=True,
    )
    _plot_one_method_multiseed_z(
        method_label=labels[1],
        teachers=coup_teachers,
        students=coup_students,
        seed_labels=seed_labels,
        out_path=out_dir / "phase2_multiseed_z_coupled.png",
        time_window_s=time_window_s,
        z_dim=z_dim,
        semantic_z01=False,
    )

    # Combined large figure: rows = method × seed, cols = z dims.
    rows = []
    for i, seed in enumerate(seed_labels):
        rows.append((f"{labels[0]} {seed}", dec_teachers[i], dec_students[i], True))
    for i, seed in enumerate(seed_labels):
        rows.append((f"{labels[1]} {seed}", coup_teachers[i], coup_students[i], False))

    actual = max([_available_z_dim(stu, max_dim=z_dim) for _, _, stu, _ in rows] + [0])
    if actual <= 0:
        print("[WARN] no z columns found, skip combined z plot")
        return
    z_dim = min(z_dim, actual)

    fig, axs = plt.subplots(len(rows), z_dim, figsize=(3.0 * z_dim, max(1.9 * len(rows), 5.0)), sharex=False)
    if len(rows) == 1 and z_dim == 1:
        axs = np.asarray([[axs]])
    elif len(rows) == 1:
        axs = np.asarray([axs])
    elif z_dim == 1:
        axs = np.asarray([[a] for a in axs])

    fig.suptitle("Phase-II latent z across methods and seeds", fontsize=14, weight="bold")
    for r, (row_label, teacher, student, semantic_z01) in enumerate(rows):
        for c in range(z_dim):
            ax = axs[r, c]
            t, zT, zH = _get_z_teacher_student_for_plot(teacher, student, c)
            if time_window_s is not None:
                m = t <= float(time_window_s)
                t_plot, zT_plot, zH_plot = t[m], zT[m], zH[m]
            else:
                t_plot, zT_plot, zH_plot = t, zT, zH
            ax.plot(t_plot, zT_plot, linestyle="--", linewidth=1.2, label="teacher zT")
            ax.plot(t_plot, zH_plot, linestyle="-", linewidth=1.5, label="student zH")
            ax.set_title(f"z{c}", fontsize=9)
            ax.grid(True, linestyle=":", linewidth=0.6)
            if c == 0:
                ax.set_ylabel(row_label)
            if r == len(rows) - 1:
                ax.set_xlabel("Time (s)")
            if r == 0 and c == 0:
                ax.legend(fontsize="small")
            if semantic_z01 and c in (0, 1):
                ax.set_ylim(0.0, 1.0)

    _save_fig(fig, out_dir / "phase2_multiseed_z_all_methods.png")
    print("[MultiSeed-Z] saved:")
    print("  - phase2_multiseed_z_decoupled.png")
    print("  - phase2_multiseed_z_coupled.png")
    print("  - phase2_multiseed_z_all_methods.png")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=str, default=None, help="Path to phase2_teacher.csv")
    ap.add_argument("--student", type=str, default=None, help="Path to phase2_student.csv")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory for figures (default: dir of student csv)")
    ap.add_argument("--start", type=float, nargs=3, default=[-2.0, 0.0, 2.0], help="TASK start offset (x y z)")
    ap.add_argument("--goal", type=float, nargs=3, default=[2.0, 0.0, 2.0], help="TASK goal offset (x y z)")
    ap.add_argument("--time_window", type=float, default=5.0, help="Energy plot time window in seconds (like IsaaclabPlot12.5). Use <=0 to disable clipping.")
    ap.add_argument("--smooth_window", type=float, default=0.15, help="Smoothing window seconds for energy curves.")
    # Cross-method Phase-II compare mode. If these four paths are provided, the script ignores --teacher/--student.
    ap.add_argument("--dec_teacher", type=str, default=None, help="Decoupled phase2_teacher.csv")
    ap.add_argument("--dec_student", type=str, default=None, help="Decoupled phase2_student.csv")
    ap.add_argument("--coup_teacher", type=str, default=None, help="Coupled phase2_teacher.csv")
    ap.add_argument("--coup_student", type=str, default=None, help="Coupled phase2_student.csv")
    ap.add_argument("--compare_labels", type=str, nargs=2, default=["Decoupled", "Coupled"], help="Labels for compare mode")
    # Multi-seed compare mode: put three seeds of Decoupled/Coupled Phase-II into ONE figure.
    ap.add_argument("--dec_teachers", type=str, nargs="+", default=None, help="Decoupled phase2_teacher.csv list, one per seed")
    ap.add_argument("--dec_students", type=str, nargs="+", default=None, help="Decoupled phase2_student.csv list, one per seed")
    ap.add_argument("--coup_teachers", type=str, nargs="+", default=None, help="Coupled phase2_teacher.csv list, one per seed")
    ap.add_argument("--coup_students", type=str, nargs="+", default=None, help="Coupled phase2_student.csv list, one per seed")
    ap.add_argument("--seed_labels", type=str, nargs="+", default=None, help="Labels for seeds, e.g. seed38 seed40 seed42")
    ap.add_argument("--z_dim", type=int, default=5, help="Number of latent dimensions to plot in multi-seed z figures")
    args = ap.parse_args()

    start_ref = np.array(args.start, dtype=float)
    goal_ref = np.array(args.goal, dtype=float)

    # ----------------------------
    # Multi-seed compare mode: Decoupled/Coupled Phase-II over multiple seeds in one figure
    # ----------------------------
    multi_lists = [args.dec_teachers, args.dec_students, args.coup_teachers, args.coup_students]
    if any(x is not None for x in multi_lists):
        if not all(x is not None for x in multi_lists):
            raise SystemExit(
                "[ERROR] Multi-seed mode requires all four lists: "
                "--dec_teachers --dec_students --coup_teachers --coup_students"
            )
        n = len(args.dec_teachers)
        if not (len(args.dec_students) == len(args.coup_teachers) == len(args.coup_students) == n):
            raise SystemExit("[ERROR] Multi-seed path lists must have the same length")
        seed_labels = args.seed_labels if args.seed_labels is not None else [f"seed{i}" for i in range(n)]
        if len(seed_labels) != n:
            raise SystemExit("[ERROR] --seed_labels length must match the number of CSV paths")

        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path("./phase2_multiseed_compare").resolve()
        _ensure_out_dir(out_dir)

        dec_teachers = [load_series(Path(p).expanduser().resolve(), f"dec_teacher_{seed_labels[i]}", start_ref, goal_ref) for i, p in enumerate(args.dec_teachers)]
        dec_students = [load_series(Path(p).expanduser().resolve(), f"dec_student_{seed_labels[i]}", start_ref, goal_ref) for i, p in enumerate(args.dec_students)]
        coup_teachers = [load_series(Path(p).expanduser().resolve(), f"coup_teacher_{seed_labels[i]}", start_ref, goal_ref) for i, p in enumerate(args.coup_teachers)]
        coup_students = [load_series(Path(p).expanduser().resolve(), f"coup_student_{seed_labels[i]}", start_ref, goal_ref) for i, p in enumerate(args.coup_students)]

        tw = float(args.time_window)
        tmax = None if tw <= 0 else tw
        plot_phase2_multiseed_compare(
            dec_teachers=dec_teachers,
            dec_students=dec_students,
            coup_teachers=coup_teachers,
            coup_students=coup_students,
            seed_labels=seed_labels,
            out_dir=out_dir,
            time_window_s=tmax,
            labels=(args.compare_labels[0], args.compare_labels[1]),
        )
        plot_phase2_multiseed_z_outputs(
            dec_teachers=dec_teachers,
            dec_students=dec_students,
            coup_teachers=coup_teachers,
            coup_students=coup_students,
            seed_labels=seed_labels,
            out_dir=out_dir,
            time_window_s=tmax,
            labels=(args.compare_labels[0], args.compare_labels[1]),
            z_dim=int(args.z_dim),
        )
        print(f"[DONE] Multi-seed Phase-II figures saved to: {out_dir}")
        return

    # ----------------------------
    # Compare mode: Decoupled Phase-II vs Coupled Phase-II in one seed
    # ----------------------------
    compare_paths = [args.dec_teacher, args.dec_student, args.coup_teacher, args.coup_student]
    if any(p is not None for p in compare_paths):
        if not all(p is not None for p in compare_paths):
            raise SystemExit(
                "[ERROR] Compare mode requires all four paths: "
                "--dec_teacher --dec_student --coup_teacher --coup_student"
            )
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path(args.dec_student).expanduser().resolve().parent / "phase2_compare"
        _ensure_out_dir(out_dir)

        dec_teacher = load_series(Path(args.dec_teacher).expanduser().resolve(), "dec_teacher", start_ref, goal_ref)
        dec_student = load_series(Path(args.dec_student).expanduser().resolve(), "dec_student", start_ref, goal_ref)
        coup_teacher = load_series(Path(args.coup_teacher).expanduser().resolve(), "coup_teacher", start_ref, goal_ref)
        coup_student = load_series(Path(args.coup_student).expanduser().resolve(), "coup_student", start_ref, goal_ref)

        print(f"[INFO] dec_teacher origin_w: {dec_teacher.origin_w}")
        print(f"[INFO] dec_student origin_w: {dec_student.origin_w}")
        print(f"[INFO] coup_teacher origin_w: {coup_teacher.origin_w}")
        print(f"[INFO] coup_student origin_w: {coup_student.origin_w}")

        tw = float(args.time_window)
        tmax = None if tw <= 0 else tw
        plot_phase2_cross_method_compare(
            dec_teacher=dec_teacher,
            dec_student=dec_student,
            coup_teacher=coup_teacher,
            coup_student=coup_student,
            out_dir=out_dir,
            time_window_s=tmax,
            smooth_window_s=float(args.smooth_window),
            labels=(args.compare_labels[0], args.compare_labels[1]),
        )
        print(f"[DONE] Cross-method Phase-II figures saved to: {out_dir}")
        return

    if args.teacher is None or args.student is None:
        raise SystemExit("[ERROR] Single-method mode requires --teacher and --student")

    teacher_path = Path(args.teacher).expanduser().resolve()
    student_path = Path(args.student).expanduser().resolve()

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else student_path.parent
    _ensure_out_dir(out_dir)

    teacher = load_series(teacher_path, "teacher", start_ref, goal_ref)
    student = load_series(student_path, "student", start_ref, goal_ref)

    # Optional: print inferred origins so you can sanity-check the -9..11 issue.
    print(f"[INFO] teacher origin_w inferred: {teacher.origin_w} (goal_world - goal_ref)")
    print(f"[INFO] student origin_w inferred: {student.origin_w} (goal_world - goal_ref)")

    # Plots
    plot_fig1_payload_pos_and_swing(teacher, student, out_dir)
    plot_fig2_swing_angles(teacher, student, out_dir)
    plot_fig3_z_compare(teacher, student, out_dir)

    tw = float(args.time_window)
    tmax = None if tw <= 0 else tw
    plot_fig4_energy_combined(teacher, student, out_dir, tmax, float(args.smooth_window))

    print(f"[DONE] Figures saved to: {out_dir}")

if __name__ == "__main__":
    main()