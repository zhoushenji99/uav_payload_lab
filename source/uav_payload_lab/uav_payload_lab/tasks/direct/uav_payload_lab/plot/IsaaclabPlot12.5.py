import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from scipy.spatial.transform import Rotation as R
import math

# ============================================================
# IsaaclabPlot12.5.py
#
# 单 CSV：
#   按你原来的 Figure 1-6 详细分析逻辑跑
#
# 多 CSV：
#   2 个 CSV：Decoupled vs Coupled
#   3 个 CSV：PPO vs Coupled vs Decoupled
#   自动进入对比模式，把多条轨迹画在同一张图里
# ============================================================

# 默认路径：不传 --csv 时保持你原来的行为
DEFAULT_SIM_CSV = "/home/shenji/uav_payload_lab/uav_payload_lab/logs/rsl_rl/Encoder_DataCollectionMLW/2026-04-19_23-47-50/payload_data.csv"

# === Analysis window (seconds) ===
# CLI 里 --time_window -1 表示不裁剪
DEFAULT_TIME_WINDOW_S = 5.0

# 坐标系校正 (World -> Task)
OFFSET_X = 0.0
OFFSET_Y = 0.0

# === Task reference ===
REF_X = 2.0
REF_Y = 0.0
REF_Z = 2.0
REF_THETA_X = 0.0
REF_THETA_Y = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot IsaacLab UAV-payload CSV. One CSV = original mode; two/three CSVs = comparison mode."
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        default=[DEFAULT_SIM_CSV],
        help="One CSV for original mode, or 2/3 CSVs for comparison mode, e.g. PPO Coupled Decoupled.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Labels for comparison mode, e.g. --labels PPO Coupled Decoupled",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output directory for comparison figures and metrics.",
    )
    parser.add_argument(
        "--time_window",
        type=float,
        default=DEFAULT_TIME_WINDOW_S,
        help="Crop to [0, time_window] seconds. Use -1 to disable.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show matplotlib windows after saving.",
    )

    # 用 parse_known_args，避免 IsaacLab / shell 额外参数导致崩溃
    args, _ = parser.parse_known_args()
    return args


ARGS = parse_args()
TIME_WINDOW_S = None if ARGS.time_window is not None and ARGS.time_window < 0 else ARGS.time_window

# 单 CSV 模式下，后面旧代码继续使用 simulation_data_path
simulation_data_path = ARGS.csv[0]


def preprocess_payload_csv(path: str) -> pd.DataFrame:
    """读取一个 payload CSV，并统一列名，供双 CSV 对比模式使用。"""
    df = pd.read_csv(path)

    # 兼容你的旧列名
    rename_map = {}
    if "Time" in df.columns and "time" not in df.columns:
        rename_map["Time"] = "time"
    if "Swing_Deg_X" in df.columns and "theta_x_deg" not in df.columns:
        rename_map["Swing_Deg_X"] = "theta_x_deg"
    if "Swing_Deg_Y" in df.columns and "theta_y_deg" not in df.columns:
        rename_map["Swing_Deg_Y"] = "theta_y_deg"

    df = df.rename(columns=rename_map)

    if "time" not in df.columns:
        raise ValueError(f"[ERROR] CSV 缺少 time/Time 列: {path}")

    # payload position
    if "Payload_X" in df.columns:
        df["payload_x"] = df["Payload_X"] - OFFSET_X
        df["payload_y"] = df["Payload_Y"] - OFFSET_Y
        df["payload_z"] = df["Payload_Z"]
    elif all(c in df.columns for c in ["payload_x", "payload_y", "payload_z"]):
        pass
    else:
        raise ValueError(
            f"[ERROR] CSV 缺少 Payload_X/Y/Z 或 payload_x/y/z: {path}\n"
            f"columns={list(df.columns)}"
        )

    # swing columns
    if "theta_x_deg" not in df.columns or "theta_y_deg" not in df.columns:
        raise ValueError(
            f"[ERROR] CSV 缺少 theta_x_deg/theta_y_deg 或 Swing_Deg_X/Y: {path}\n"
            f"columns={list(df.columns)}"
        )

    # 如果有四元数，顺便算欧拉角，方便后续扩展
    quat_cols = ["UAV_quat_1", "UAV_quat_2", "UAV_quat_3", "UAV_quat_0"]
    if all(c in df.columns for c in quat_cols):
        try:
            quats = df[quat_cols].to_numpy()
            euler = R.from_quat(quats).as_euler("xyz", degrees=True)
            df["roll"] = euler[:, 0]
            df["pitch"] = euler[:, 1]
            df["yaw"] = euler[:, 2]
        except Exception as e:
            print(f"[WARN] 四元数转欧拉角失败，可忽略: {e}")

    # 派生指标
    df["dist_to_goal"] = np.sqrt(
        (df["payload_x"] - REF_X) ** 2
        + (df["payload_y"] - REF_Y) ** 2
        + (df["payload_z"] - REF_Z) ** 2
    )

    df["swing_mag_deg"] = np.sqrt(
        df["theta_x_deg"] ** 2 + df["theta_y_deg"] ** 2
    )

    # payload speed：优先读现成速度，没有就用位置差分
    vel_candidates = [
        ("Payload_vx", "Payload_vy", "Payload_vz"),
        ("payload_vx", "payload_vy", "payload_vz"),
        ("Payload_VX", "Payload_VY", "Payload_VZ"),
    ]

    found_vel = None
    for cols in vel_candidates:
        if all(c in df.columns for c in cols):
            found_vel = cols
            break

    if found_vel is not None:
        vx, vy, vz = [df[c].to_numpy(dtype=float) for c in found_vel]
    else:
        t = df["time"].to_numpy(dtype=float)
        vx = np.gradient(df["payload_x"].to_numpy(dtype=float), t)
        vy = np.gradient(df["payload_y"].to_numpy(dtype=float), t)
        vz = np.gradient(df["payload_z"].to_numpy(dtype=float), t)

    df["payload_speed"] = np.sqrt(vx * vx + vy * vy + vz * vz)

    # swing energy surrogate
    if "SwingVel_DegS_X" in df.columns and "SwingVel_DegS_Y" in df.columns:
        thx = np.deg2rad(df["theta_x_deg"].to_numpy(dtype=float))
        thy = np.deg2rad(df["theta_y_deg"].to_numpy(dtype=float))
        thdx = np.deg2rad(df["SwingVel_DegS_X"].to_numpy(dtype=float))
        thdy = np.deg2rad(df["SwingVel_DegS_Y"].to_numpy(dtype=float))

        L_est = 0.8
        if "UAV_Z" in df.columns and "Payload_Z" in df.columns:
            dz = np.abs(
                df["UAV_Z"].to_numpy(dtype=float)
                - df["Payload_Z"].to_numpy(dtype=float)
            )
            dz = dz[np.isfinite(dz)]
            if len(dz) > 20:
                L_est = float(np.median(dz))

        df["E_hat"] = 0.5 * (
            thdx ** 2
            + thdy ** 2
            + (9.81 / max(L_est, 1e-6)) * (thx ** 2 + thy ** 2)
        )

    return df


def crop_time(df: pd.DataFrame, tmax):
    if tmax is None:
        return df
    return df[(df["time"] >= 0.0) & (df["time"] <= float(tmax))].copy()


def first_hit_metric(df: pd.DataFrame, threshold: float):
    hit = df[df["dist_to_goal"] <= threshold]
    if len(hit) == 0:
        return np.nan, np.nan
    idx = hit.index[0]
    return float(df.loc[idx, "time"]), float(df.loc[idx, "payload_speed"])


def compute_summary(df: pd.DataFrame) -> dict:
    t02, v02 = first_hit_metric(df, 0.2)
    t01, v01 = first_hit_metric(df, 0.1)

    out = {
        "hit_t_0p2_s": t02,
        "speed_at_0p2_mps": v02,
        "hit_t_0p1_s": t01,
        "speed_at_0p1_mps": v01,
        "final_error_m": float(df["dist_to_goal"].iloc[-1]),
        "max_swing_deg": float(df["swing_mag_deg"].max()),
        "mean_swing_deg": float(df["swing_mag_deg"].mean()),
    }

    if "E_hat" in df.columns:
        out["E_hat_mean"] = float(df["E_hat"].mean())
        out["E_hat_peak"] = float(df["E_hat"].max())

    return out


def plot_phase1_teacher_compare(csv_paths, labels, out_dir, time_window):
    """
    多 CSV 对比模式：
    支持 2 个或 3 个方法画在同一张图里。
    典型用法：PPO vs Coupled vs Decoupled。
    """
    os.makedirs(out_dir, exist_ok=True)

    dfs = []
    for p in csv_paths:
        df = preprocess_payload_csv(p)
        df = crop_time(df, time_window)
        dfs.append(df)
        print(f"[Compare] loaded {p}: {len(df)} rows after crop")

    # ============================================================
    # Figure A: payload position + swing angles
    # ============================================================
    fig, axs = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    fig.suptitle(
        "Ablation Comparison: " + " vs ".join(labels),
        fontsize=16,
        weight="bold",
    )

    items = [
        ("payload_x", "Payload X", REF_X, "Position (m)"),
        ("payload_y", "Payload Y", REF_Y, "Position (m)"),
        ("payload_z", "Payload Z", REF_Z, "Position (m)"),
        ("theta_x_deg", "Swing theta_x", REF_THETA_X, "Angle (deg)"),
        ("theta_y_deg", "Swing theta_y", REF_THETA_Y, "Angle (deg)"),
        ("swing_mag_deg", "Swing magnitude", 0.0, "Angle (deg)"),
    ]

    for ax, (col, title, ref, ylabel) in zip(axs.flat, items):
        for df, lab in zip(dfs, labels):
            ax.plot(df["time"], df[col], linewidth=2.2, label=lab)

        if ref is not None:
            ax.axhline(ref, linestyle=":", linewidth=1.2, alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(fontsize="small")

        if time_window is not None:
            ax.set_xlim(0.0, float(time_window))

    fig_path = os.path.join(out_dir, "ablation_compare_payload_swing.png")
    fig.savefig(fig_path, dpi=300)
    print(f"[Saved] {fig_path}")

    # ============================================================
    # Figure B: distance / speed / swing magnitude
    # ============================================================
    fig2, axs2 = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True, sharex=True)
    fig2.suptitle(
        "Ablation Comparison: Error, Speed, and Swing",
        fontsize=16,
        weight="bold",
    )

    for df, lab in zip(dfs, labels):
        axs2[0].plot(df["time"], df["dist_to_goal"], linewidth=2.2, label=lab)
        axs2[1].plot(df["time"], df["payload_speed"], linewidth=2.2, label=lab)
        axs2[2].plot(df["time"], df["swing_mag_deg"], linewidth=2.2, label=lab)

    axs2[0].axhline(0.2, linestyle=":", alpha=0.8, label="0.2 m")
    axs2[0].axhline(0.1, linestyle="--", alpha=0.8, label="0.1 m")
    axs2[0].set_ylabel("Distance to goal (m)")
    axs2[1].set_ylabel("Payload speed (m/s)")
    axs2[2].set_ylabel("Swing magnitude (deg)")
    axs2[2].set_xlabel("Time (s)")

    for ax in axs2:
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(fontsize="small")
        if time_window is not None:
            ax.set_xlim(0.0, float(time_window))

    fig2_path = os.path.join(out_dir, "ablation_compare_error_speed_swing.png")
    fig2.savefig(fig2_path, dpi=300)
    print(f"[Saved] {fig2_path}")

    # ============================================================
    # Metrics CSV
    # ============================================================
    rows = []
    for p, df, lab in zip(csv_paths, dfs, labels):
        row = {"label": lab, "csv": p}
        row.update(compute_summary(df))
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics_path = os.path.join(out_dir, "ablation_compare_metrics.csv")
    metrics.to_csv(metrics_path, index=False)

    print("\n[Metrics]")
    print(metrics.to_string(index=False))
    print(f"[Saved] {metrics_path}")

    if ARGS.show:
        plt.show()
    else:
        plt.close("all")


# ============================================================
# 多 CSV：进入对比模式，然后提前退出
#   1 个 CSV：走原始单文件详细图 Figure 1-6
#   2 个 CSV：两方法对比
#   3 个 CSV：三方法对比，例如 PPO / Coupled / Decoupled
# ============================================================
if len(ARGS.csv) in (2, 3):
    if ARGS.labels is not None and len(ARGS.labels) == len(ARGS.csv):
        labels = ARGS.labels
    elif len(ARGS.csv) == 2:
        labels = ["Decoupled", "Coupled"]
    else:
        labels = ["PPO", "Coupled", "Decoupled"]

    plot_phase1_teacher_compare(
        csv_paths=ARGS.csv,
        labels=labels,
        out_dir=ARGS.out_dir,
        time_window=TIME_WINDOW_S,
    )
    sys.exit(0)

elif len(ARGS.csv) != 1:
    raise SystemExit(
        "[ERROR] --csv 只能给 1 个、2 个或 3 个路径。"
        "1 个=原图模式；2/3 个=对比模式。"
    )
# --- 2. 加载仿真数据 ---
try:
    df_sim = pd.read_csv(simulation_data_path)
    print(f"[Sim] 成功加载 '{simulation_data_path}'，共 {len(df_sim)} 行")
except FileNotFoundError:
    print(f"错误: 找不到文件 '{simulation_data_path}'，请确认文件路径。")
    raise SystemExit

# 列名标准化与预处理
if "Payload_X" in df_sim.columns:
    df_sim.rename(columns={
        "Time": "time",
        "Swing_Deg_X": "theta_x_deg",
        "Swing_Deg_Y": "theta_y_deg"
    }, inplace=True)

    # 坐标转换 (用于 Payload 对比图)
    df_sim["payload_x"] = df_sim["Payload_X"] - OFFSET_X
    df_sim["payload_y"] = df_sim["Payload_Y"] - OFFSET_Y
    df_sim["payload_z"] = df_sim["Payload_Z"]

    # [新增] 计算无人机姿态 (四元数 -> 欧拉角)
    quats = df_sim[['UAV_quat_1', 'UAV_quat_2', 'UAV_quat_3', 'UAV_quat_0']].to_numpy()
    r = R.from_quat(quats)
    euler = r.as_euler('xyz', degrees=True) # 使用 xyz 顺序 (Roll, Pitch, Yaw)
    df_sim['roll'] = euler[:, 0]
    df_sim['pitch'] = euler[:, 1]
    df_sim['yaw'] = euler[:, 2]

    print(f"[Sim] 数据预处理完成: 坐标已校正, 欧拉角已计算。")

# # --- 3. 加载并清洗论文数据 (保持原有逻辑) ---
# paper_data = {}
# paper_available = False

# try:
#     df_paper = pd.read_csv(paper_data_path, header=None)
#     print(f"[Paper] 成功加载 '{paper_data_path}'")
#     paper_available = True

#     groups = {
#         "px": (0, 1), "py": (2, 3), "pz": (4, 5),
#         "thetax": (6, 7), "thetay": (8, 9)
#     }

#     for key, (t_col, v_col) in groups.items():
#         if v_col < df_paper.shape[1]:
#             sub_df = df_paper[[t_col, v_col]].dropna()
#             sub_df.columns = ["time", "value"]
#             sub_df = sub_df.sort_values(by="time")
#             paper_data[key] = sub_df
#             print(f"  - {key}: {len(sub_df)} 点")

# except Exception as e:
#     print(f"[Paper] 未加载论文数据或读取失败 (可忽略): {e}")

# ==========================================
#               绘图部分
# ==========================================

# ------------------------------------------
# Figure 1: Payload 轨迹对比
# ------------------------------------------
fig1, axs1 = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
fig1.suptitle("Figure 1: Payload Control Performance (Sim Only)", fontsize=16, weight="bold")

def plot_curve(ax, sim_x, sim_y, title, ref=None, ylabel="Position (m)"):
    ax.set_title(title, fontsize=12)
    ax.plot(sim_x, sim_y, label="My RL (Sim)", linewidth=2.5, color='tab:blue')
    if ref is not None:
        ax.axhline(y=ref, linestyle=":", color='gray', alpha=0.8, label=f"Ref: {ref}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

plot_curve(axs1[0, 0], df_sim["time"], df_sim["payload_x"], "Payload X", REF_X)
plot_curve(axs1[0, 1], df_sim["time"], df_sim["payload_y"], "Payload Y", REF_Y)
plot_curve(axs1[0, 2], df_sim["time"], df_sim["payload_z"], "Payload Z", REF_Z)
plot_curve(axs1[1, 0], df_sim["time"], df_sim["theta_x_deg"], "Swing Theta X", REF_THETA_X, "Angle (deg)")
plot_curve(axs1[1, 1], df_sim["time"], df_sim["theta_y_deg"], "Swing Theta Y", REF_THETA_Y, "Angle (deg)")
axs1[1, 2].axis('off')

# ------------------------------------------
# Figure 2: UAV 姿态与状态详情 (Body Frame)
# ------------------------------------------
fig2, axs2 = plt.subplots(4, 3, figsize=(18, 16), constrained_layout=True)
fig2.suptitle("Figure 2: UAV States Detail (Position, Velocity, Attitude, Angular Rate)", fontsize=16, weight="bold")

# 第一行：位置 (Raw World Position)
axs2[0, 0].plot(df_sim["time"], df_sim["UAV_X"], color='tab:green')
axs2[0, 0].set_title("UAV Position X (World)")
axs2[0, 0].set_ylabel("m")

axs2[0, 1].plot(df_sim["time"], df_sim["UAV_Y"], color='tab:green')
axs2[0, 1].set_title("UAV Position Y (World)")

axs2[0, 2].plot(df_sim["time"], df_sim["UAV_Z"], color='tab:green')
axs2[0, 2].set_title("UAV Position Z (World)")

# 第二行：线速度 (Body Frame)
axs2[1, 0].plot(df_sim["time"], df_sim["UAV_v_bx"], color='tab:purple')
axs2[1, 0].set_title("UAV Linear Vel X (Body)")
axs2[1, 0].set_ylabel("m/s")

axs2[1, 1].plot(df_sim["time"], df_sim["UAV_v_by"], color='tab:purple')
axs2[1, 1].set_title("UAV Linear Vel Y (Body)")

axs2[1, 2].plot(df_sim["time"], df_sim["UAV_v_bz"], color='tab:purple')
axs2[1, 2].set_title("UAV Linear Vel Z (Body)")

# 第三行：姿态 (Euler Angles)
axs2[2, 0].plot(df_sim["time"], df_sim["roll"], color='tab:red')
axs2[2, 0].set_title("UAV Roll (Euler X)")
axs2[2, 0].set_ylabel("deg")

axs2[2, 1].plot(df_sim["time"], df_sim["pitch"], color='tab:red')
axs2[2, 1].set_title("UAV Pitch (Euler Y)")

axs2[2, 2].plot(df_sim["time"], df_sim["yaw"], color='tab:red')
axs2[2, 2].set_title("UAV Yaw (Euler Z)")

# 第四行：角速度 (Body Rates)
axs2[3, 0].plot(df_sim["time"], df_sim["UAV_w_bx"], color='tab:brown')
axs2[3, 0].set_title("UAV Angular Rate X")
axs2[3, 0].set_ylabel("rad/s")

axs2[3, 1].plot(df_sim["time"], df_sim["UAV_w_by"], color='tab:brown')
axs2[3, 1].set_title("UAV Angular Rate Y")

axs2[3, 2].plot(df_sim["time"], df_sim["UAV_w_bz"], color='tab:brown')
axs2[3, 2].set_title("UAV Angular Rate Z")

for ax in axs2.flat:
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlabel("Time (s)")

# ------------------------------------------
# Figure 3: Action & Commands 分析
# ------------------------------------------
fig3, axs3 = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
fig3.suptitle("Figure 3: Neural Network Actions & Environment Commands", fontsize=16, weight="bold")

def plot_action(ax, idx):
    ax.plot(df_sim["time"], df_sim[f"Policy_a{idx}"], label="Policy (Net)", alpha=0.6, linewidth=1)
    # 兼容性检查：如果CSV里没有Env_raw列，就不画
    if f"Env_raw_a{idx}" in df_sim.columns:
        ax.plot(df_sim["time"], df_sim[f"Env_raw_a{idx}"], label="Env Raw", alpha=0.6, linestyle="--", linewidth=1)
    if f"Env_clamp_a{idx}" in df_sim.columns:
        ax.plot(df_sim["time"], df_sim[f"Env_clamp_a{idx}"], label="Env Clamp", color='k', linewidth=1.5)

    ax.set_title(f"Action {idx}")
    ax.set_ylim([-1.5, 1.5])
    ax.grid(True, linestyle=":")
    if idx == 0: ax.legend(loc='upper right', fontsize='small')

# 第一行：Action
for i in range(4):
    plot_action(axs3[0, i], i)

# 第二行：Commands
axs3[1, 0].plot(df_sim["time"], df_sim["Thrust_Cmd"], color='tab:olive')
axs3[1, 0].set_title("Thrust Command")
axs3[1, 0].set_ylabel("Force (N)")

axs3[1, 1].plot(df_sim["time"], df_sim["Moment_Cmd_X"], color='tab:cyan')
axs3[1, 1].set_title("Moment Cmd X")
axs3[1, 1].set_ylabel("Torque (Nm)")

axs3[1, 2].plot(df_sim["time"], df_sim["Moment_Cmd_Y"], color='tab:cyan')
axs3[1, 2].set_title("Moment Cmd Y")

axs3[1, 3].plot(df_sim["time"], df_sim["Moment_Cmd_Z"], color='tab:cyan')
axs3[1, 3].set_title("Moment Cmd Z")

for ax in axs3.flat:
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlabel("Time (s)")

# ------------------------------------------
# [新增] Figure 4: World Frame Dynamics (Velocity & Acceleration)
# ------------------------------------------
# 只有当CSV包含加速度数据时才画
if "UAV_a_wz" in df_sim.columns:
    fig4, axs4 = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig4.suptitle("Figure 4: World Frame Dynamics (Velocity & Acceleration)", fontsize=16, weight="bold")

    # 第一行：世界系速度
    # 检查是否存在 UAV_v_wx (如果是旧版 play.py 生成的可能没有)
    if "UAV_v_wx" in df_sim.columns:
        axs4[0, 0].plot(df_sim["time"], df_sim["UAV_v_wx"], color='tab:orange', label='World Vel X')
        axs4[0, 1].plot(df_sim["time"], df_sim["UAV_v_wy"], color='tab:orange', label='World Vel Y')
        axs4[0, 2].plot(df_sim["time"], df_sim["UAV_v_wz"], color='tab:orange', label='World Vel Z')
    else:
        # 如果没有世界系速度，可以用 Body Velocity 近似占位，或者留白
        axs4[0, 0].text(0.5, 0.5, "World Vel Data Missing", ha='center')

    axs4[0, 0].set_title("World Linear Vel X")
    axs4[0, 0].set_ylabel("m/s")
    axs4[0, 1].set_title("World Linear Vel Y")
    axs4[0, 2].set_title("World Linear Vel Z")

    # 第二行：世界系加速度 (这就是你要的图)
    axs4[1, 0].plot(df_sim["time"], df_sim["UAV_a_wx"], color='tab:red')
    axs4[1, 1].plot(df_sim["time"], df_sim["UAV_a_wy"], color='tab:red')
    axs4[1, 2].plot(df_sim["time"], df_sim["UAV_a_wz"], color='tab:red')

    axs4[1, 0].set_title("World Accel X")
    axs4[1, 0].set_ylabel("m/s^2")
    axs4[1, 1].set_title("World Accel Y")
    # 重点标注 Z 加速度
    axs4[1, 2].set_title("World Accel Z (The 'Artificial Gravity' Indicator)")
    axs4[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    for ax in axs4.flat:
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlabel("Time (s)")

    plt.figure(fig4.number)
    plt.savefig("plot_4_world_dynamics.png", dpi=300)
    print("  - plot_4_world_dynamics.png")
else:
    print("[Info] CSV 中未发现加速度数据 (UAV_a_wz)，跳过 Figure 4。")


# ------------------------------------------
# [新增] Figure 5: Input Shaping Evidence (Tilt/Accel vs Swing)
# ------------------------------------------
def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Simple moving average (same length)."""
    x = np.asarray(x, dtype=np.float32)
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, kernel, mode="same")

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson corr ignoring NaNs; returns nan if insufficient data."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 10:
        return float("nan")
    aa = a[m] - float(np.mean(a[m]))
    bb = b[m] - float(np.mean(b[m]))
    sa = float(np.std(aa))
    sb = float(np.std(bb))
    if sa < 1e-9 or sb < 1e-9:
        return float("nan")
    return float(np.mean(aa * bb) / (sa * sb))

# 只有当CSV包含世界系加速度 (UAV_a_wx/UAV_a_wy) 时才画
if "UAV_a_wx" in df_sim.columns and "UAV_a_wy" in df_sim.columns:
    # 1) 估计“任务主运动方向” (XY 平面)：从 UAV 起点到终点的位移方向
    dx = float(df_sim["UAV_X"].iloc[-1] - df_sim["UAV_X"].iloc[0]) if "UAV_X" in df_sim.columns else 1.0
    dy = float(df_sim["UAV_Y"].iloc[-1] - df_sim["UAV_Y"].iloc[0]) if "UAV_Y" in df_sim.columns else 0.0
    dnorm = float(np.sqrt(dx * dx + dy * dy))
    if dnorm < 1e-6:
        d_xy = np.array([1.0, 0.0], dtype=np.float32)
    else:
        d_xy = np.array([dx / dnorm, dy / dnorm], dtype=np.float32)

    a_wx = df_sim["UAV_a_wx"].to_numpy(dtype=np.float32)
    a_wy = df_sim["UAV_a_wy"].to_numpy(dtype=np.float32)
    a_parallel = a_wx * d_xy[0] + a_wy * d_xy[1]

    # 2) 选择“更像沿运动方向倾斜”的欧拉角轴 (roll or pitch)
    candidates = []
    if "pitch" in df_sim.columns:
        candidates.append(("pitch", df_sim["pitch"].to_numpy(dtype=np.float32)))
    if "roll" in df_sim.columns:
        candidates.append(("roll", df_sim["roll"].to_numpy(dtype=np.float32)))

    if len(candidates) == 0:
        print("[Warn] 未发现 roll/pitch（欧拉角列），跳过 Figure 5。")
    else:
        corr_list = [(name, _safe_corr(arr, a_parallel)) for name, arr in candidates]
        best_name, best_corr = max(corr_list, key=lambda t: (abs(t[1]) if np.isfinite(t[1]) else -1))
        tilt_deg = dict(candidates)[best_name]

        # 3) 摆角投影到主运动方向（诊断用）：theta_parallel ≈ theta_x * dx + theta_y * dy
        #    注意：这不是严格的坐标变换，但作为“相位证据”的可视化非常有用。
        theta_parallel = None
        if "theta_x_deg" in df_sim.columns and "theta_y_deg" in df_sim.columns:
            thx = df_sim["theta_x_deg"].to_numpy(dtype=np.float32)
            thy = df_sim["theta_y_deg"].to_numpy(dtype=np.float32)
            theta_parallel = thx * d_xy[0] + thy * d_xy[1]

        # 4) 平滑（避免差分加速度的噪声）—— 默认 0.15s 窗口
        t_all = df_sim["time"].to_numpy(dtype=np.float32)
        dt = float(np.median(np.diff(t_all))) if len(t_all) > 2 else 1 / 60.0

        # --- Crop to early-window for input-shaping analysis (e.g., 0-5s) ---
        if TIME_WINDOW_S is not None:
            m_win = (t_all >= 0.0) & (t_all <= float(TIME_WINDOW_S))
        else:
            m_win = np.ones_like(t_all, dtype=bool)

        win = max(1, int(round(0.15 / max(dt, 1e-6))))
        a_parallel_f_all = _moving_average(a_parallel, win)
        tilt_f_all = _moving_average(tilt_deg, win)
        theta_parallel_f_all = _moving_average(theta_parallel, win) if theta_parallel is not None else None

        # Apply window mask
        t = t_all[m_win]
        a_parallel_f = a_parallel_f_all[m_win]
        tilt_f = tilt_f_all[m_win]
        theta_parallel_f = theta_parallel_f_all[m_win] if theta_parallel_f_all is not None else None

        # 5) 找到“明显的加速度符号翻转”（对应：加速→减速→再加速）
        eps = 0.30  # m/s^2，小于这个认为是 0（过滤噪声）
        s = np.sign(a_parallel_f)
        s[np.abs(a_parallel_f) < eps] = 0
        flip_idx = np.where((s[1:] * s[:-1] < 0))[0] + 1
        flip_t = [float(t[i]) for i in flip_idx[:10]]  # 取前几个足够标注阶段

        # 6) 估计绳长 L（若 CSV 有 UAV_Z & Payload_Z），用于半周期参考线
        L_est = 0.8
        if "UAV_Z" in df_sim.columns and "Payload_Z" in df_sim.columns:
            dz = (df_sim["UAV_Z"].to_numpy(dtype=np.float32) - df_sim["Payload_Z"].to_numpy(dtype=np.float32))
            dz_w = dz[m_win]
            if np.isfinite(dz_w).sum() > 20:
                L_est = float(np.median(dz_w[np.isfinite(dz_w)]))
        g0 = 9.81
        T_half = math.pi * math.sqrt(max(L_est, 1e-3) / g0)

        # 7) “峰值对齐”的相位差标注：t_a_peak 与 t_theta_peak（直观证据，不是严格系统辨识）
        t_a_peak = None
        t_theta_peak = None
        phase_lag = None

        # a_peak：在 0-2s 内找第一段正加速度峰（更像“第一次加速脉冲”）
        m_a = (t <= 2.0) & np.isfinite(a_parallel_f)
        if int(m_a.sum()) >= 5:
            a_seg = a_parallel_f[m_a]
            t_seg = t[m_a]
            if np.any(a_seg > 0):
                j = int(np.argmax(a_seg))
            else:
                j = int(np.argmax(np.abs(a_seg)))
            t_a_peak = float(t_seg[j])
        else:
            j = int(np.argmax(np.abs(a_parallel_f)))
            t_a_peak = float(t[j])

        # theta_peak：在 a_peak 之后 0~1.5s 内找 |theta| 最大点（响应滞后）
        if theta_parallel_f is not None and t_a_peak is not None:
            m_th = (t >= t_a_peak) & (t <= t_a_peak + 1.5) & np.isfinite(theta_parallel_f)
            if int(m_th.sum()) >= 5:
                th_seg = theta_parallel_f[m_th]
                t_th = t[m_th]
                k = int(np.argmax(np.abs(th_seg)))
                t_theta_peak = float(t_th[k])
                phase_lag = float(t_theta_peak - t_a_peak)

        # 8) 绘图：只画 0-5s；并用竖线标注“阶段”与“相位差”
        fig5, axs5 = plt.subplots(2, 1, figsize=(18, 8), constrained_layout=True, sharex=True)
        fig5.suptitle("Figure 5 (0-5s): Input-Shaping Evidence (Tilt/Accel & Swing/Accel)", fontsize=16, weight="bold")

        # (a) tilt(t) 与 a_parallel(t) 同相性检查
        ax = axs5[0]
        ax2 = ax.twinx()
        ax.plot(t, tilt_f, label=f"{best_name}(t) [deg] (smoothed)", linewidth=2)
        ax2.plot(t, a_parallel_f, label="a_parallel(t) [m/s^2] (smoothed)", linewidth=2, linestyle="--")
        ax.set_title("1) tilt(t) vs a_parallel(t): should be roughly in-phase (small-angle approx)")
        ax.set_ylabel("Tilt (deg)")
        ax2.set_ylabel("a_parallel (m/s^2)")
        ax.grid(True, linestyle=":", alpha=0.5)

        # 阶段标注（由 a_parallel 的符号翻转定义）
        stage_bounds = [float(t[0])] + flip_t + [float(t[-1])]
        stage_labels = ["Accel", "Brake", "Catch", "Trim"]
        for i in range(min(len(stage_bounds) - 1, len(stage_labels))):
            t0, t1 = stage_bounds[i], stage_bounds[i + 1]
            tm = 0.5 * (t0 + t1)
            ax.text(tm, 0.95, stage_labels[i], transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=10, alpha=0.85)
            ax.axvline(t0, color="k", alpha=0.20, linestyle=":")
        ax.axvline(stage_bounds[-1], color="k", alpha=0.20, linestyle=":")
        for tt in flip_t:
            ax.axvline(tt, color="k", alpha=0.25, linestyle=":")

        # 峰值/半周期参考线
        if t_a_peak is not None:
            ax.axvline(t_a_peak, alpha=0.45, linestyle="--")
            ax2.axvline(t_a_peak, alpha=0.25, linestyle="--")
            ax.text(t_a_peak, 0.02, "a_peak", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, alpha=0.95)
            t_half_ref = t_a_peak + T_half
            if t_half_ref <= float(t[-1]):
                ax.axvline(t_half_ref, alpha=0.35, linestyle="--")
                ax.text(t_half_ref, 0.02, "a_peak + T/2", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=9, alpha=0.95)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize="small")

        # (b) theta(t) 与 a_parallel(t) 的“翻转 → 包络下降”现象
        ax = axs5[1]
        ax2 = ax.twinx()
        if theta_parallel_f is not None:
            ax.plot(t, theta_parallel_f, label="theta_parallel(t) [deg] (smoothed)", linewidth=2)
            ax.set_ylabel("theta_parallel (deg)")
            ax.set_title("2) swing vs a_parallel: braking (sign flips) should reduce swing envelope (input-shaping intuition)")
        else:
            if "theta_x_deg" in df_sim.columns:
                ax.plot(t, df_sim["theta_x_deg"].to_numpy(dtype=np.float32)[m_win], label="theta_x_deg", linewidth=1.5)
            if "theta_y_deg" in df_sim.columns:
                ax.plot(t, df_sim["theta_y_deg"].to_numpy(dtype=np.float32)[m_win], label="theta_y_deg", linewidth=1.5)
            ax.set_ylabel("theta (deg)")
            ax.set_title("2) swing vs a_parallel: (theta_parallel missing; plotting theta_x/theta_y)")

        ax2.plot(t, a_parallel_f, label="a_parallel(t) [m/s^2] (smoothed)", linewidth=2, linestyle="--")
        ax2.set_ylabel("a_parallel (m/s^2)")

        for tt in flip_t:
            ax.axvline(tt, color="k", alpha=0.25, linestyle=":")

        # 相位差（峰值对齐的直观标注）
        if t_a_peak is not None:
            ax.axvline(t_a_peak, alpha=0.35, linestyle="--")
            ax2.axvline(t_a_peak, alpha=0.20, linestyle="--")
        if t_theta_peak is not None:
            ax.axvline(t_theta_peak, alpha=0.50, linestyle="-.")
            ax.text(t_theta_peak, 0.02, "theta_peak", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, alpha=0.95)
        if phase_lag is not None and t_theta_peak is not None and t_a_peak is not None:
            y0, y1 = ax.get_ylim()
            y_arrow = y0 + 0.80 * (y1 - y0)
            ax.annotate("", xy=(t_theta_peak, y_arrow), xytext=(t_a_peak, y_arrow),
                        arrowprops=dict(arrowstyle="<->", alpha=0.75))
            ax.text(0.5 * (t_a_peak + t_theta_peak), y_arrow, f"Δt≈{phase_lag:.2f}s",
                    ha="center", va="bottom", fontsize=10, alpha=0.95)

        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlabel("Time (s)")
        if TIME_WINDOW_S is not None:
            ax.set_xlim(0.0, float(TIME_WINDOW_S))

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize="small")

        # 保存
        print(f"[Figure5] travel_dir_xy=({d_xy[0]:+.3f},{d_xy[1]:+.3f}), best_tilt={best_name}, corr={best_corr:+.3f}, smooth_win={win} (~{win*dt:.3f}s)")
        print(f"[Figure5] L_est≈{L_est:.3f} m => T/2≈{T_half:.3f} s; flip_t={flip_t}; phase_lag≈{phase_lag}")
        plt.figure(fig5.number)
        plt.savefig("plot_5_input_shaping_evidence_0_5s.png", dpi=300)
        print("  - plot_5_input_shaping_evidence_0_5s.png")

else:
    print("[Info] CSV 中未发现 UAV_a_wx/UAV_a_wy，跳过 Figure 5。")

# --- Crop view to TIME_WINDOW_S (if enabled) for other figures ---
if TIME_WINDOW_S is not None:
    for _fig in [fig1, fig2, fig3]:
        for _ax in _fig.axes:
            _ax.set_xlim(0.0, float(TIME_WINDOW_S))
    # fig4 只在加速度数据存在时创建
    if "fig4" in globals():
        for _ax in fig4.axes:
            _ax.set_xlim(0.0, float(TIME_WINDOW_S))




# ==========================================
# [修改] Figure 6 Combined: 综合分析 (Style-Matched)
# 风格已修正：统一使用点状网格(:)，调整线宽，对齐原图风格
# ==========================================
print("\n[Plotting] Figure 6 Combined (Mechanism + Energy Flow)...")

# 检查所需列
_need_cols = ["time", "theta_x_deg", "theta_y_deg", "SwingVel_DegS_X", "SwingVel_DegS_Y", "UAV_a_wx", "UAV_a_wy"]

if all([c in df_sim.columns for c in _need_cols]):
    # --- 1. 数据准备 (共用) ---
    t_all = df_sim["time"].to_numpy(dtype=np.float32)
    dt = float(np.median(np.diff(t_all))) if len(t_all) > 2 else 0.02

    # Window
    if TIME_WINDOW_S is not None:
        m_win = (t_all >= 0.0) & (t_all <= float(TIME_WINDOW_S))
    else:
        m_win = np.ones_like(t_all, dtype=bool)

    t = t_all[m_win]

    # Rope length estimate
    L_est = 0.8
    if "UAV_Z" in df_sim.columns and "Payload_Z" in df_sim.columns:
        dz = (df_sim["UAV_Z"].to_numpy(dtype=np.float32) - df_sim["Payload_Z"].to_numpy(dtype=np.float32))
        L_est = float(np.median(np.abs(dz)))
    g0 = 9.81

    # --- 2. 计算部分 ---

    # (A) 投影与平滑 (Mechanism)
    if "UAV_X" in df_sim.columns and "UAV_Y" in df_sim.columns:
        dx = float(df_sim["UAV_X"].iloc[-1] - df_sim["UAV_X"].iloc[0])
        dy = float(df_sim["UAV_Y"].iloc[-1] - df_sim["UAV_Y"].iloc[0])
        dnorm = float(np.sqrt(dx*dx + dy*dy))
        d_xy = np.array([dx/dnorm, dy/dnorm]) if dnorm > 1e-6 else np.array([1.0, 0.0])
    else:
        d_xy = np.array([1.0, 0.0])

    v_swing_x = df_sim["SwingVel_DegS_X"].to_numpy(dtype=np.float32)
    v_swing_y = df_sim["SwingVel_DegS_Y"].to_numpy(dtype=np.float32)
    a_uav_x = df_sim["UAV_a_wx"].to_numpy(dtype=np.float32)
    a_uav_y = df_sim["UAV_a_wy"].to_numpy(dtype=np.float32)

    v_swing_parallel = v_swing_x * d_xy[0] + v_swing_y * d_xy[1]
    a_uav_parallel   = a_uav_x * d_xy[0] + a_uav_y * d_xy[1]

    win = max(1, int(round(0.15 / dt)))
    def _smooth(arr, w):
        return np.convolve(arr, np.ones(w)/w, mode='same')

    v_plot = _smooth(v_swing_parallel, win)[m_win]
    a_plot = _smooth(a_uav_parallel, win)[m_win]

    # (B) 能量与功率 (Energy)
    thx = df_sim["theta_x_deg"].to_numpy(dtype=np.float32) * (np.pi/180.0)
    thy = df_sim["theta_y_deg"].to_numpy(dtype=np.float32) * (np.pi/180.0)
    thdx = df_sim["SwingVel_DegS_X"].to_numpy(dtype=np.float32) * (np.pi/180.0)
    thdy = df_sim["SwingVel_DegS_Y"].to_numpy(dtype=np.float32) * (np.pi/180.0)

    th2 = thx*thx + thy*thy
    thd2 = thdx*thdx + thdy*thdy

    a_z = df_sim["UAV_a_wz"].to_numpy(dtype=np.float32) if "UAV_a_wz" in df_sim.columns else np.zeros_like(thx)
    w2 = (g0 + a_z) / max(L_est, 1e-6)
    w2 = np.clip(w2, 0.1, 200.0)

    E_hat_all = 0.5 * (thd2 + w2 * th2)
    E_hat_f = _smooth(E_hat_all, win)
    E_dot = np.gradient(E_hat_f, t_all)

    P_xy = (a_uav_x * thdx + a_uav_y * thdy) / max(L_est, 1e-6)
    w2_f = _smooth(w2, win)
    w2_dot = np.gradient(w2_f, t_all)
    P_param = 0.5 * w2_dot * th2
    P_model = P_xy + P_param

    E_hat = E_hat_f[m_win]
    E_dot_w = E_dot[m_win]
    P_xy_w = _smooth(P_xy, win)[m_win]
    P_param_w = _smooth(P_param, win)[m_win]
    P_model_w = _smooth(P_model, win)[m_win]

    # --- 3. 绘图 (合并) ---
    fig6, axs6 = plt.subplots(4, 1, figsize=(18, 16), constrained_layout=True, sharex=True)
    fig6.suptitle("Figure 6: Comprehensive Energy Analysis (Mechanism -> Result)", fontsize=16, weight="bold")

    # [Subplot 1] Mechanism
    ax1 = axs6[0]
    color_vel = 'tab:blue'
    ax1.set_title("1. Mechanism: In-Phase Coupling (UAV Accelerates with Swing)", fontsize=12, loc='left')
    ax1.set_ylabel("Swing Vel (deg/s)", color=color_vel, weight='bold')
    l1, = ax1.plot(t, v_plot, color=color_vel, linewidth=2.0, label="Swing Velocity ($v_{swing}$)")
    ax1.tick_params(axis='y', labelcolor=color_vel)
    ax1.grid(True, linestyle=":", alpha=0.5)  # 修正：使用冒号虚线

    ax1_r = ax1.twinx()
    color_acc = 'tab:red'
    ax1_r.set_ylabel("UAV Accel (m/s$^2$)", color=color_acc, weight='bold')
    l2, = ax1_r.plot(t, a_plot, color=color_acc, linestyle='--', linewidth=2.0, label="UAV Acceleration ($a_{uav}$)")
    ax1_r.tick_params(axis='y', labelcolor=color_acc)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    # [Subplot 2] Energy State
    axs6[1].set_title("2. Energy State: Monotonic Decay", fontsize=12, loc='left')
    axs6[1].plot(t, E_hat, label="E_hat(t) (smoothed)", color='k', linewidth=2.0)
    axs6[1].set_ylabel("E_hat [rad^2/s^2]")
    axs6[1].grid(True, linestyle=":", alpha=0.5) # 修正
    axs6[1].legend(loc="upper right")

    # [Subplot 3] Verification
    axs6[2].set_title("3. Verification: Power Matches Energy Rate", fontsize=12, loc='left')
    axs6[2].plot(t, E_dot_w, label="dE_hat/dt (numeric)", color='gray', alpha=0.6)
    axs6[2].plot(t, P_model_w, label="P_model (analytical)", color='tab:purple', linewidth=2.0)
    axs6[2].axhline(0.0, linestyle="--", linewidth=1, color='k')
    axs6[2].set_ylabel("Rate [rad^2/s^3]")
    axs6[2].grid(True, linestyle=":", alpha=0.5) # 修正
    axs6[2].legend(loc="upper right")

    # [Subplot 4] Decomposition
    axs6[3].set_title("4. Decomposition: Contribution of Horizontal Motion", fontsize=12, loc='left')
    axs6[3].plot(t, P_xy_w, label="P_xy = -(a_xy·theta_dot)/L", color='tab:red', linewidth=2.0)
    axs6[3].plot(t, P_param_w, label="P_param (Vertical/Gravity)", color='tab:green', alpha=0.7, linewidth=1.5)
    axs6[3].plot(t, P_model_w, label="P_model (Total)", color='k', linestyle='--', alpha=0.5)
    axs6[3].axhline(0.0, linestyle="--", linewidth=1, color='k')
    axs6[3].set_xlabel("Time (s)")
    axs6[3].set_ylabel("Power [rad^2/s^3]")
    axs6[3].grid(True, linestyle=":", alpha=0.5) # 修正
    axs6[3].legend(loc="upper right")

    # 阴影标注
    axs6[3].fill_between(t, 0.0, P_model_w, where=(P_model_w < 0.0), alpha=0.10, color='green', interpolate=True)

    # 保存
    plt.figure(fig6.number)
    plt.savefig("plot_6_combined.png", dpi=300)
    print("  - plot_6_combined.png (Saved!)")

else:
    print(f"[Warn] 缺少必要列 {_need_cols}，跳过 Figure 6 Combined。")
# 保存其他图片
plt.figure(fig1.number)
plt.savefig("plot_1_payload.png", dpi=300)

plt.figure(fig2.number)
plt.savefig("plot_2_uav_states.png", dpi=300)

plt.figure(fig3.number)
plt.savefig("plot_3_actions.png", dpi=300)

print("\n[Output] 所有图像已保存：")
print("  - plot_1_payload.png")
print("  - plot_2_uav_states.png")
print("  - plot_3_actions.png")

plt.show()