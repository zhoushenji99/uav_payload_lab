#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase2 CSV key-figures generator (paper-focused).

Required figures:
Fig1) Teacher vs Student: pos & theta_xy swing (5 subplots in one big figure).
Fig2) Teacher z0..z4 vs Student z0..z4 (5 subplots in one big figure).
Fig3) priv0..priv4 vs z0..z4 (scatter; 5 subplots in one big figure).
Fig4) Teacher vs Student: anti-sway energy dissipation rate statistics (one figure).

Run:
  python3 analyze_phase2_csv.py --csv <payload_phase2_student_full_v2.csv> --out_dir <dir>

Notes:
- We try to auto-detect teacher/student column pairs with common suffix/prefix patterns.
- If some columns are missing, we will warn and either fallback or skip that subplot.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def find_pair(df, base_name, *, prefer_student_base=True):
    """
    Find teacher/student columns for a metric.

    Teacher candidates:
      {base}_T, {base}_teacher, teacher_{base}, T_{base}

    Student candidates:
      {base}_S, {base}_student, student_{base}, S_{base}, {base}

    If prefer_student_base=True, we use {base} as student when present.
    """
    t_cands = [
        f"{base_name}_T", f"{base_name}_teacher", f"teacher_{base_name}", f"T_{base_name}"
    ]
    s_cands = [
        f"{base_name}_S", f"{base_name}_student", f"student_{base_name}", f"S_{base_name}"
    ]
    if prefer_student_base:
        s_cands.append(base_name)

    col_T = _first_existing(df, t_cands)
    col_S = _first_existing(df, s_cands)

    return col_T, col_S

def find_dim_cols(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    def key(c):
        try:
            return int(c[len(prefix):])
        except Exception:
            return 10**9
    return sorted(cols, key=key)

def safe_series(df, col):
    if col is None or col not in df.columns:
        return None
    x = df[col].to_numpy(dtype=np.float64)
    x[~np.isfinite(x)] = np.nan
    return x

def finite_mask(*arrs):
    m = None
    for a in arrs:
        if a is None:
            continue
        mm = np.isfinite(a)
        m = mm if m is None else (m & mm)
    if m is None:
        return None
    return m

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def get_time(df):
    # common time column names
    for c in ["time", "t", "t_sec", "t_s", "sim_time"]:
        if c in df.columns:
            t = df[c].to_numpy(dtype=np.float64)
            t[~np.isfinite(t)] = np.nan
            return t, c
    # fallback: step index
    t = np.arange(len(df), dtype=np.float64)
    return t, "step"


# -----------------------------
# Main plotting functions
# -----------------------------
def plot_fig1_pos_theta(df, out_png):
    """
    Fig1: 5 subplots in a big figure:
      1) pos_err teacher vs student
      2) theta_x_deg teacher vs student
      3) theta_y_deg teacher vs student
      4) theta_dot_x_deg_s teacher vs student
      5) theta_dot_y_deg_s teacher vs student
    """
    t, tname = get_time(df)

    # metric bases you asked for
    bases = ["pos_err", "theta_x_deg", "theta_y_deg", "theta_dot_x_deg_s", "theta_dot_y_deg_s"]

    # fallback options (in case your CSV uses slightly different names)
    # e.g., theta_x_deg may be tilt_x_deg, swing_x_deg, etc.
    fallbacks = {
        "theta_x_deg": ["theta_x_deg", "tilt_x_deg", "swing_x_deg", "theta_x"],
        "theta_y_deg": ["theta_y_deg", "tilt_y_deg", "swing_y_deg", "theta_y"],
        "theta_dot_x_deg_s": ["theta_dot_x_deg_s", "theta_x_dot_deg_s", "swing_dot_x_deg_s", "theta_dot_x"],
        "theta_dot_y_deg_s": ["theta_dot_y_deg_s", "theta_y_dot_deg_s", "swing_dot_y_deg_s", "theta_dot_y"],
        "pos_err": ["pos_err", "final_distance_to_goal", "dist_to_goal", "distance_to_goal"],
    }

    series_pairs = []
    used_titles = []
    warnings = []

    for b in bases:
        # try base first
        colT, colS = find_pair(df, b, prefer_student_base=True)

        # if not found, try fallback list for the BASE NAME (we still keep teacher/student pattern)
        if colT is None or colS is None:
            for alt in fallbacks.get(b, [b]):
                colT2, colS2 = find_pair(df, alt, prefer_student_base=True)
                if colS2 is not None:  # student metric must exist
                    colT, colS = colT2, colS2
                    b = alt
                    break

        yT = safe_series(df, colT) if colT is not None else None
        yS = safe_series(df, colS) if colS is not None else None

        if yS is None:
            warnings.append(f"[Fig1] Missing student column for '{b}' (tried {fallbacks.get(b,[b])}). Skip subplot.")
            series_pairs.append((None, None, b))
            used_titles.append(b)
            continue

        if yT is None:
            warnings.append(f"[Fig1] Missing teacher column for '{b}'. Will plot student only.")
        series_pairs.append((yT, yS, b))
        used_titles.append(b)

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    for i, (yT, yS, name) in enumerate(series_pairs):
        ax = axes[i]
        ax.plot(t, yS, label="student")
        if yT is not None:
            ax.plot(t, yT, label="teacher")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")

    axes[-1].set_xlabel(f"{tname}")
    fig.suptitle("Fig1: Teacher vs Student - Position and Swing (theta_xy)", y=0.995)
    savefig(out_png)

    if warnings:
        print("\n".join(warnings))


def plot_fig2_z_overlay(df, out_png):
    """
    Fig2: zT0..zT4 vs zH0..zH4 overlays (5 subplots)
    """
    t, tname = get_time(df)

    zH_cols = find_dim_cols(df, "zH")
    zT_cols = find_dim_cols(df, "zT")
    if len(zH_cols) < 5 or len(zT_cols) < 5:
        raise RuntimeError(f"[Fig2] Need zH0..zH4 and zT0..zT4. Found zH={zH_cols}, zT={zT_cols}")

    zH = df[zH_cols[:5]].to_numpy(dtype=np.float64)
    zT = df[zT_cols[:5]].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    for i in range(5):
        ax = axes[i]
        ax.plot(t, zH[:, i], label=f"student zH{i}")
        ax.plot(t, zT[:, i], label=f"teacher zT{i}")
        ax.set_ylabel(f"z{i}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel(f"{tname}")
    fig.suptitle("Fig2: Teacher zT0-4 vs Student zH0-4", y=0.995)
    savefig(out_png)


def plot_fig3_priv_vs_z(df, out_png):
    """
    Fig3: priv0..priv4 vs z0..z4 (scatter, 5 subplots)

    We will plot:
      x = priv_i
      y1 = teacher zT_i (if exists)
      y2 = student zH_i (if exists)
    """
    # priv
    priv_cols = [f"priv{i}" for i in range(5)]
    # sometimes your dataset might use "priv0..4" or "mlw0..4"
    if not all(c in df.columns for c in priv_cols):
        alt = [f"mlw{i}" for i in range(5)]
        if all(c in df.columns for c in alt):
            priv_cols = alt
        else:
            raise RuntimeError(f"[Fig3] Need priv0..priv4 (or mlw0..mlw4). Missing: {priv_cols}")

    # z
    zH_cols = find_dim_cols(df, "zH")
    zT_cols = find_dim_cols(df, "zT")
    has_student = len(zH_cols) >= 5
    has_teacher = len(zT_cols) >= 5
    if not (has_student or has_teacher):
        raise RuntimeError("[Fig3] Need at least zH0..zH4 or zT0..zT4.")

    priv = df[priv_cols].to_numpy(dtype=np.float64)
    zH = df[zH_cols[:5]].to_numpy(dtype=np.float64) if has_student else None
    zT = df[zT_cols[:5]].to_numpy(dtype=np.float64) if has_teacher else None

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=False)
    for i in range(5):
        ax = axes[i]
        x = priv[:, i]
        m = np.isfinite(x)
        if zT is not None:
            y = zT[:, i]
            mm = m & np.isfinite(y)
            ax.scatter(x[mm], y[mm], s=6, alpha=0.35, label=f"teacher zT{i}")
        if zH is not None:
            y = zH[:, i]
            mm = m & np.isfinite(y)
            ax.scatter(x[mm], y[mm], s=6, alpha=0.35, label=f"student zH{i}")

        ax.set_xlabel(priv_cols[i])
        ax.set_ylabel(f"z{i}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")

    fig.suptitle("Fig3: Privileged vars (priv/mlw) vs z (teacher/student)", y=0.995)
    savefig(out_png)


def plot_fig4_energy_dissipation(df, out_png):
    """
    Fig4: energy dissipation rate statistics (teacher vs student).

    We need E_hat for teacher and student (if teacher missing, plot student only).
    Candidate bases: E_hat, E_hat_mean, Ehat, energy_hat

    We compute dissipation rate:
      dE/dt via finite difference, then dissipation = -dE/dt (positive means energy decreases)

    We output one figure with:
      - top: E_hat vs time (teacher/student)
      - bottom: histogram of dissipation rate (teacher/student)
    """
    t, tname = get_time(df)

    # Find energy columns
    # prefer E_hat
    base_candidates = ["E_hat", "E_hat_mean", "Ehat", "energy_hat"]
    colT = None
    colS = None
    used_base = None
    for b in base_candidates:
        cT, cS = find_pair(df, b, prefer_student_base=True)
        if cS is not None:
            colT, colS = cT, cS
            used_base = b
            break

    if colS is None:
        raise RuntimeError(f"[Fig4] Cannot find student energy column. Tried: {base_candidates}")

    E_S = safe_series(df, colS)
    E_T = safe_series(df, colT) if colT is not None else None

    # time diffs
    dt = np.diff(t)
    dt[dt == 0] = np.nan
    # compute rate arrays aligned to length N (pad first with nan)
    dE_S = np.diff(E_S) / dt
    diss_S = -dE_S
    diss_S = np.concatenate([[np.nan], diss_S])

    if E_T is not None:
        dE_T = np.diff(E_T) / dt
        diss_T = -dE_T
        diss_T = np.concatenate([[np.nan], diss_T])
    else:
        diss_T = None

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    # top: E_hat time series
    ax0 = axes[0]
    ax0.plot(t, E_S, label="student")
    if E_T is not None:
        ax0.plot(t, E_T, label="teacher")
    ax0.set_title(f"Energy proxy over time ({used_base})")
    ax0.set_xlabel(tname)
    ax0.set_ylabel("E_hat")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")

    # bottom: dissipation histogram
    ax1 = axes[1]
    mS = np.isfinite(diss_S)
    ax1.hist(diss_S[mS], bins=80, alpha=0.55, label="student")
    if diss_T is not None:
        mT = np.isfinite(diss_T)
        ax1.hist(diss_T[mT], bins=80, alpha=0.55, label="teacher")
    ax1.set_title("Energy dissipation rate distribution (positive=energy decreasing)")
    ax1.set_xlabel("dissipation = -dE/dt")
    ax1.set_ylabel("count")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    fig.suptitle("Fig4: Teacher vs Student - Anti-sway energy dissipation statistics", y=0.995)
    savefig(out_png)


# -----------------------------
# Entry
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    df = pd.read_csv(args.csv)
    print(f"[INFO] Loaded CSV: {args.csv} rows={len(df)} cols={len(df.columns)}")

    # Figure 1
    fig1 = os.path.join(args.out_dir, "Fig1_pos_theta_teacher_vs_student.png")
    plot_fig1_pos_theta(df, fig1)
    print(f"[OK] Saved: {fig1}")

    # Figure 2
    fig2 = os.path.join(args.out_dir, "Fig2_z_teacher_vs_student.png")
    plot_fig2_z_overlay(df, fig2)
    print(f"[OK] Saved: {fig2}")

    # Figure 3
    fig3 = os.path.join(args.out_dir, "Fig3_priv_vs_z.png")
    plot_fig3_priv_vs_z(df, fig3)
    print(f"[OK] Saved: {fig3}")

    # Figure 4
    fig4 = os.path.join(args.out_dir, "Fig4_energy_dissipation_teacher_vs_student.png")
    plot_fig4_energy_dissipation(df, fig4)
    print(f"[OK] Saved: {fig4}")

    print("[DONE] Key figures generated.")


if __name__ == "__main__":
    main()
