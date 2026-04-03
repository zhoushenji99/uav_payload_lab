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
        """Add uav/payload velocities, and uav accelerations, in TASK frame."""
        t = self.t()

        # UAV velocity + acceleration
        for axis in ["x", "y", "z"]:
            pcol = f"uav_{axis}"
            if pcol in self.df.columns:
                p = self.df[pcol].to_numpy(dtype=float)
                v = _gradient(p, t)
                a = _gradient(v, t)
                self.df[f"uav_v{axis}"] = v
                self.df[f"uav_a{axis}"] = a

        # payload velocity
        for axis in ["x", "y", "z"]:
            pcol = f"payload_{axis}"
            if pcol in self.df.columns:
                p = self.df[pcol].to_numpy(dtype=float)
                v = _gradient(p, t)
                self.df[f"payload_v{axis}"] = v

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
def _first_true(mask: np.ndarray):
    idx = np.where(mask)[0]
    return None if len(idx) == 0 else int(idx[0])

def _first_sustained(mask: np.ndarray, min_len: int):
    if min_len <= 1:
        return _first_true(mask)
    run = 0
    for i, m in enumerate(mask):
        if m:
            run += 1
            if run >= min_len:
                return i - min_len + 1
        else:
            run = 0
    return None

def compute_summary(series: Series, goal_radius: float = 0.1, stable_time: float = 1.0):
    df = series.df
    t = series.t()

    if all(c in df.columns for c in ["payload_err_x", "payload_err_y", "payload_err_z"]):
        err = np.linalg.norm(df[["payload_err_x", "payload_err_y", "payload_err_z"]].to_numpy(dtype=float), axis=1)
    else:
        err = np.linalg.norm(
            df[["payload_x", "payload_y", "payload_z"]].to_numpy(dtype=float)
            - series.goal_ref.reshape(1, 3),
            axis=1,
        )

    theta = np.sqrt(
        df["theta_x_deg"].to_numpy(dtype=float) ** 2
        + df["theta_y_deg"].to_numpy(dtype=float) ** 2
    )

    payload_speed = np.sqrt(
        df.get("payload_vx", pd.Series(np.zeros_like(t))).to_numpy(dtype=float) ** 2
        + df.get("payload_vy", pd.Series(np.zeros_like(t))).to_numpy(dtype=float) ** 2
        + df.get("payload_vz", pd.Series(np.zeros_like(t))).to_numpy(dtype=float) ** 2
    )

    uav_speed = np.sqrt(
        df.get("uav_vx", pd.Series(np.zeros_like(t))).to_numpy(dtype=float) ** 2
        + df.get("uav_vy", pd.Series(np.zeros_like(t))).to_numpy(dtype=float) ** 2
        + df.get("uav_vz", pd.Series(np.zeros_like(t))).to_numpy(dtype=float) ** 2
    )

    dt = float(np.nanmedian(np.diff(t))) if len(t) >= 2 else 0.0
    stable_steps = max(1, int(round(stable_time / max(dt, 1e-9))))

    hit_mask = err <= goal_radius
    idx_hit = _first_true(hit_mask)
    idx_stable = _first_sustained(hit_mask, stable_steps)

    def pick(arr, idx):
        return None if idx is None else float(arr[idx])

    summary = {
        "final_error_m": float(err[-1]),
        "tail2_mean_error_m": float(np.mean(err[t >= (t[-1] - 2.0)])) if len(t) else None,
        "tail5_mean_error_m": float(np.mean(err[t >= (t[-1] - 5.0)])) if len(t) else None,
        "peak_theta_deg": float(np.max(theta)),
        "theta_p95_deg": float(np.percentile(theta, 95)),
        "final_theta_deg": float(theta[-1]),
        "first_hit_time_s": pick(t, idx_hit),
        "first_hit_payload_speed_mps": pick(payload_speed, idx_hit),
        "first_hit_uav_speed_mps": pick(uav_speed, idx_hit),
        "stable_hit_time_s": pick(t, idx_stable),
        "stable_hit_payload_speed_mps": pick(payload_speed, idx_stable),
        "stable_hit_uav_speed_mps": pick(uav_speed, idx_stable),
    }

    if all(c in df.columns for c in ["a0_raw","a1_raw","a2_raw","a3_raw"]):
        a_raw = df[["a0_raw","a1_raw","a2_raw","a3_raw"]].to_numpy(dtype=float)
        summary["action_abs_mean"] = float(np.mean(np.abs(a_raw)))
        summary["action_sat_ratio"] = float(np.mean(np.abs(a_raw) >= 0.999))

    if "z_rmse" in df.columns:
        summary["z_rmse_mean"] = float(df["z_rmse"].mean())

    return summary

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
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=str, required=True, help="Path to phase2_teacher.csv")
    ap.add_argument("--student", type=str, required=True, help="Path to phase2_student.csv")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory for figures (default: dir of student csv)")
    ap.add_argument("--start", type=float, nargs=3, default=[-2.0, 0.0, 2.0], help="TASK start offset (x y z)")
    ap.add_argument("--goal", type=float, nargs=3, default=[2.0, 0.0, 2.0], help="TASK goal offset (x y z)")
    ap.add_argument("--time_window", type=float, default=5.0, help="Energy plot time window in seconds (like IsaaclabPlot12.5). Use <=0 to disable clipping.")
    ap.add_argument("--smooth_window", type=float, default=0.15, help="Smoothing window seconds for energy curves.")
    args = ap.parse_args()

    teacher_path = Path(args.teacher).expanduser().resolve()
    student_path = Path(args.student).expanduser().resolve()

    start_ref = np.array(args.start, dtype=float)
    goal_ref = np.array(args.goal, dtype=float)

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
    teacher_summary = compute_summary(teacher)
    student_summary = compute_summary(student)

    import json
    with open(out_dir / "summary_teacher.json", "w") as f:
        json.dump(teacher_summary, f, indent=2)
    with open(out_dir / "summary_student.json", "w") as f:
        json.dump(student_summary, f, indent=2)

    print("[SUMMARY][teacher]")
    for k, v in teacher_summary.items():
        print(f"  {k}: {v}")

    print("[SUMMARY][student]")
    for k, v in student_summary.items():
        print(f"  {k}: {v}")
    print(f"[DONE] Figures saved to: {out_dir}")

if __name__ == "__main__":
    main()