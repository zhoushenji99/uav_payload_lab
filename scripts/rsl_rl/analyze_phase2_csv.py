# analyze_phase2_csv.py (Final Version with Advanced Energy Mechanism)
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 基础工具与数据增强
# ==========================================

def load_data(path):
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return None
    return pd.read_csv(path)

def augment_data_for_energy_analysis(df):
    """
    预处理数据：计算导数（速度、加速度），以满足高级能量分析的需求。
    """
    # 1. 确定时间步长 dt
    t_all = df['time_s'].to_numpy()
    dt = np.median(np.diff(t_all)) if len(t_all) > 1 else 0.02
    if dt <= 0: dt = 0.02
    
    # 2. 计算摆动角速度 (SwingVel_DegS)
    # 使用 gradient 进行中心差分
    df['SwingVel_DegS_X'] = np.gradient(df['theta_x_deg'], dt)
    df['SwingVel_DegS_Y'] = np.gradient(df['theta_y_deg'], dt)
    
    # 3. 计算无人机加速度 (UAV_a_w)
    # 先算速度
    vx = np.gradient(df['uav_px'], dt)
    vy = np.gradient(df['uav_py'], dt)
    vz = np.gradient(df['uav_pz'], dt)
    # 再算加速度
    df['UAV_a_wx'] = np.gradient(vx, dt)
    df['UAV_a_wy'] = np.gradient(vy, dt)
    df['UAV_a_wz'] = np.gradient(vz, dt)
    
    # 4. 确保有 UAV_Z 和 Payload_Z (用于估算绳长)
    if 'uav_pz' in df.columns: df['UAV_Z'] = df['uav_pz']
    if 'payload_pz' in df.columns: df['Payload_Z'] = df['payload_pz']
    
    # 5. 确保有 UAV_X/Y 用于计算投影方向
    df['UAV_X'] = df['uav_px']
    df['UAV_Y'] = df['uav_py']
    
    return df

# ==========================================
# 2. 核心绘图逻辑 (Figure 6 Combined 迁移)
# ==========================================

def plot_advanced_energy_mechanism(df_sim, out_path, title_prefix):
    """
    [移植自 IsaaclabPlot12.5.py] Figure 6 Combined: 综合分析
    包含: Mechanism (耦合), Energy State, Verification, Decomposition
    """
    print(f"[Plotting] Energy Mechanism Analysis for {title_prefix}...")

    # --- 数据准备 ---
    t = df_sim["time_s"].to_numpy(dtype=np.float32)
    t_all = t # 别名
    
    # 绳长估算
    L_est = 0.8
    if "rope_length_m" in df_sim.columns:
        L_est = df_sim["rope_length_m"].mean()
    elif "UAV_Z" in df_sim.columns and "Payload_Z" in df_sim.columns:
        dz = (df_sim["UAV_Z"] - df_sim["Payload_Z"]).to_numpy()
        L_est = float(np.median(np.abs(dz)))
    g0 = 9.81

    # --- 计算部分 ---
    
    # (A) 投影与平滑 (Mechanism)
    # 计算运动主方向 (起始点到终点)
    dx = float(df_sim["UAV_X"].iloc[-1] - df_sim["UAV_X"].iloc[0])
    dy = float(df_sim["UAV_Y"].iloc[-1] - df_sim["UAV_Y"].iloc[0])
    dnorm = float(np.sqrt(dx*dx + dy*dy))
    d_xy = np.array([dx/dnorm, dy/dnorm]) if dnorm > 1e-6 else np.array([1.0, 0.0])

    # 提取数据
    v_swing_x = df_sim["SwingVel_DegS_X"].to_numpy(dtype=np.float32)
    v_swing_y = df_sim["SwingVel_DegS_Y"].to_numpy(dtype=np.float32)
    a_uav_x = df_sim["UAV_a_wx"].to_numpy(dtype=np.float32)
    a_uav_y = df_sim["UAV_a_wy"].to_numpy(dtype=np.float32)

    # 投影到主运动方向
    v_swing_parallel = v_swing_x * d_xy[0] + v_swing_y * d_xy[1]
    a_uav_parallel   = a_uav_x * d_xy[0] + a_uav_y * d_xy[1]

    # 平滑函数
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.02
    win = max(1, int(round(0.15 / dt))) # 0.15s 窗口平滑
    def _smooth(arr, w):
        return np.convolve(arr, np.ones(w)/w, mode='same')

    v_plot = _smooth(v_swing_parallel, win)
    a_plot = _smooth(a_uav_parallel, win)

    # (B) 能量与功率 (Energy)
    # 弧度转换
    thx = df_sim["theta_x_deg"].to_numpy(dtype=np.float32) * (np.pi/180.0)
    thy = df_sim["theta_y_deg"].to_numpy(dtype=np.float32) * (np.pi/180.0)
    thdx = df_sim["SwingVel_DegS_X"].to_numpy(dtype=np.float32) * (np.pi/180.0)
    thdy = df_sim["SwingVel_DegS_Y"].to_numpy(dtype=np.float32) * (np.pi/180.0)

    th2 = thx*thx + thy*thy
    thd2 = thdx*thdx + thdy*thdy

    a_z = df_sim["UAV_a_wz"].to_numpy(dtype=np.float32) if "UAV_a_wz" in df_sim.columns else np.zeros_like(thx)
    
    # 等效重力频率 w^2 = (g + az) / L
    w2 = (g0 + a_z) / max(L_est, 1e-6)
    w2 = np.clip(w2, 0.1, 200.0)

    # 能量估计 E_hat
    E_hat_all = 0.5 * (thd2 + w2 * th2)
    E_hat_f = _smooth(E_hat_all, win)
    E_dot = np.gradient(E_hat_f, t) # 数值微分求功率

    # 理论功率模型 P_model = P_xy + P_param
    # P_xy: 水平运动做的功
    P_xy = (a_uav_x * thdx + a_uav_y * thdy) / max(L_est, 1e-6)
    # P_param: 垂直运动/变参数做的功
    w2_f = _smooth(w2, win)
    w2_dot = np.gradient(w2_f, t)
    P_param = 0.5 * w2_dot * th2
    P_model = P_xy + P_param

    # 平滑后用于绘图
    E_hat = E_hat_f
    E_dot_w = E_dot
    P_xy_w = _smooth(P_xy, win)
    P_param_w = _smooth(P_param, win)
    P_model_w = _smooth(P_model, win)

    # --- 3. 绘图 (合并) ---
    fig6, axs6 = plt.subplots(4, 1, figsize=(18, 16), constrained_layout=True, sharex=True)
    fig6.suptitle(f"{title_prefix}: Comprehensive Energy Analysis (Mechanism -> Result)", fontsize=16, weight="bold")

    # [Subplot 1] Mechanism
    ax1 = axs6[0]
    color_vel = 'tab:blue'
    ax1.set_title("1. Mechanism: In-Phase Coupling (UAV Accelerates with Swing)", fontsize=12, loc='left')
    ax1.set_ylabel("Swing Vel (deg/s)", color=color_vel, weight='bold')
    l1, = ax1.plot(t, v_plot, color=color_vel, linewidth=2.0, label="Swing Velocity ($v_{swing}$)")
    ax1.tick_params(axis='y', labelcolor=color_vel)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax1_r = ax1.twinx()
    color_acc = 'tab:red'
    ax1_r.set_ylabel("UAV Accel (m/s$^2$)", color=color_acc, weight='bold')
    l2, = ax1_r.plot(t, a_plot, color=color_acc, linestyle='--', linewidth=2.0, label="UAV Acceleration ($a_{uav}$)")
    ax1_r.tick_params(axis='y', labelcolor=color_acc)
    
    # Legend合并
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    # [Subplot 2] Energy State
    axs6[1].set_title("2. Energy State: Monotonic Decay", fontsize=12, loc='left')
    axs6[1].plot(t, E_hat, label="E_hat(t) (smoothed)", color='k', linewidth=2.0)
    axs6[1].set_ylabel("E_hat [rad^2/s^2]")
    axs6[1].grid(True, linestyle=":", alpha=0.5)
    axs6[1].legend(loc="upper right")

    # [Subplot 3] Verification
    axs6[2].set_title("3. Verification: Power Matches Energy Rate", fontsize=12, loc='left')
    axs6[2].plot(t, E_dot_w, label="dE_hat/dt (numeric)", color='gray', alpha=0.6)
    axs6[2].plot(t, P_model_w, label="P_model (analytical)", color='tab:purple', linewidth=2.0)
    axs6[2].axhline(0.0, linestyle="--", linewidth=1, color='k')
    axs6[2].set_ylabel("Rate [rad^2/s^3]")
    axs6[2].grid(True, linestyle=":", alpha=0.5)
    axs6[2].legend(loc="upper right")

    # [Subplot 4] Decomposition
    axs6[3].set_title("4. Decomposition: Contribution of Horizontal Motion", fontsize=12, loc='left')
    axs6[3].plot(t, P_xy_w, label="P_xy = -(a_xy·theta_dot)/L", color='tab:red', linewidth=2.0)
    axs6[3].plot(t, P_param_w, label="P_param (Vertical/Gravity)", color='tab:green', alpha=0.7, linewidth=1.5)
    axs6[3].plot(t, P_model_w, label="P_model (Total)", color='k', linestyle='--', alpha=0.5)
    axs6[3].axhline(0.0, linestyle="--", linewidth=1, color='k')
    axs6[3].set_xlabel("Time (s)")
    axs6[3].set_ylabel("Power [rad^2/s^3]")
    axs6[3].grid(True, linestyle=":", alpha=0.5)
    axs6[3].legend(loc="upper right")

    # 阴影标注负功区域 (Dissipation)
    axs6[3].fill_between(t, 0.0, P_model_w, where=(P_model_w < 0.0), alpha=0.10, color='green', interpolate=True)
    
    plt.savefig(out_path, dpi=300)
    print(f"  - {out_path} (Saved!)")
    plt.close()

# ==========================================
# 3. 其他基础绘图 (保留之前的 Fig1-3)
# ==========================================

def plot_fig1_position(teacher, student, out_dir):
    """Fig1: 位置跟踪"""
    t_min = min(teacher['time_s'].max(), student['time_s'].max())
    t_teacher = teacher[teacher['time_s'] <= t_min].copy()
    t_student = student[student['time_s'] <= t_min].copy()

    # 坐标系校准 (World -> Task Frame)
    task_goal_x = 2.0
    offset_x = t_teacher['goal_px'].mean() - task_goal_x
    offset_y = t_teacher['goal_py'].mean() - 0.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # X
    axes[0].plot(t_teacher['time_s'], t_teacher['payload_px'] - offset_x, 'g-', label='Teacher', linewidth=2, alpha=0.6)
    axes[0].plot(t_student['time_s'], t_student['payload_px'] - offset_x, 'r--', label='Student', linewidth=2)
    axes[0].plot(t_teacher['time_s'], t_teacher['goal_px'] - offset_x, 'k:', label='Goal', alpha=0.5)
    axes[0].set_ylabel('Pos X (m)')
    axes[0].set_title('Figure 1: Position Tracking (Task Frame)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # Y
    axes[1].plot(t_teacher['time_s'], t_teacher['payload_py'] - offset_y, 'g-', label='Teacher', linewidth=2, alpha=0.6)
    axes[1].plot(t_student['time_s'], t_student['payload_py'] - offset_y, 'r--', label='Student', linewidth=2)
    axes[1].set_ylabel('Pos Y (m)')
    axes[1].grid(True)

    # Z
    axes[2].plot(t_teacher['time_s'], t_teacher['payload_pz'], 'g-', label='Teacher', linewidth=2, alpha=0.6)
    axes[2].plot(t_student['time_s'], t_student['payload_pz'], 'r--', label='Student', linewidth=2)
    axes[2].set_ylabel('Pos Z (m)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Fig1_Position.png"))
    plt.close()

def plot_fig2_swing(teacher, student, out_dir):
    """Fig2: 摆角"""
    t_min = min(teacher['time_s'].max(), student['time_s'].max())
    t_teacher = teacher[teacher['time_s'] <= t_min]
    t_student = student[student['time_s'] <= t_min]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(t_teacher['time_s'], t_teacher['theta_x_deg'], 'g-', label='Teacher', alpha=0.6)
    axes[0].plot(t_student['time_s'], t_student['theta_x_deg'], 'r--', label='Student')
    axes[0].set_ylabel('Swing X (deg)')
    axes[0].set_title('Figure 2: Swing Suppression')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t_teacher['time_s'], t_teacher['theta_y_deg'], 'g-', label='Teacher', alpha=0.6)
    axes[1].plot(t_student['time_s'], t_student['theta_y_deg'], 'r--', label='Student')
    axes[1].set_ylabel('Swing Y (deg)')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Fig2_Swing.png"))
    plt.close()

def plot_fig3_latent(student, out_dir):
    """Fig3: Latent"""
    df = student
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    labels = ["z0 (Mass)", "z1 (Length)", "z2 (Env)", "z3 (Wind)", "z4 (Env)"]
    
    for i in range(5):
        axes[i].plot(df['time_s'], df[f'zT{i}'], 'k-', linewidth=2, label='Teacher (GT)', alpha=0.5)
        axes[i].plot(df['time_s'], df[f'zH{i}'], 'r--', linewidth=2, label='Student (Pred)')
        axes[i].set_ylabel(f'Latent {i}')
        axes[i].set_title(labels[i])
        axes[i].grid(True)
        if i == 0: axes[i].legend(loc='upper right')
            
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Fig3_Latent_Full.png"))
    plt.close()
def plot_fig_physical_decoder(student, out_dir):
    """
    Fig_Physical_Decoder: 物理参数解码验证 (修正版：反归一化 + 双轴显示)
    """
    if 'phys_pred_mass' not in student.columns:
        print("[Warn] 缺少 phys_pred_mass 列")
        return

    print("[Plotting] Physical Decoder Verification...")
    
    # === 1. 核心修正：反归一化 ===
    # 你的 Config 定义范围：Mass=[0.05, 0.15], Length=[0.3, 0.8]
    # 公式：Physical = Min + Norm * (Max - Min)
    mass_min, mass_max = 0.05, 0.15
    pred_mass_kg = mass_min + student['phys_pred_mass'] * (mass_max - mass_min)
    
    rope_min, rope_max = 0.3, 0.8
    pred_len_m = rope_min + student['phys_pred_len'] * (rope_max - rope_min)
    
    # === 2. 滤波 (去除风扰动的高频噪声) ===
    # 窗口 100 (约2秒)，这对于质量这种常数估计非常合理
    pred_mass_smooth = pred_mass_kg.rolling(window=100, min_periods=1).mean()
    pred_len_smooth = pred_len_m.rolling(window=100, min_periods=1).mean()

    # === 3. 绘图 ===
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- 质量图 ---
    ax1 = axes[0]
    # 画真值 (黑线)
    l1, = ax1.plot(student['time_s'], student['payload_mass_kg'], 'k-', linewidth=3, label='True Mass (GT)')
    # 画滤波后的预测 (红虚线)
    l2, = ax1.plot(student['time_s'], pred_mass_smooth, 'r--', linewidth=2.5, label='Predicted (Filtered)')
    # 画原始噪声预测 (浅红线，展示真实波动)
    ax1.plot(student['time_s'], pred_mass_kg, 'r-', alpha=0.15, linewidth=1)
    
    ax1.set_ylabel('Mass (kg)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Mass Identification: Error ≈ {abs(pred_mass_smooth.mean() - student["payload_mass_kg"].mean()):.4f} kg', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_ylim(0.0, 0.25) # 聚焦在真值附近

    # --- 绳长图 ---
    ax2 = axes[1]
    l1, = ax2.plot(student['time_s'], student['rope_length_m'], 'k-', linewidth=3, label='True Length (GT)')
    l2, = ax2.plot(student['time_s'], pred_len_smooth, 'b--', linewidth=2.5, label='Predicted (Filtered)')
    ax2.plot(student['time_s'], pred_len_m, 'b-', alpha=0.15, linewidth=1)
    
    ax2.set_ylabel('Rope Length (m)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_ylim(0.0, 1.0)
    
    save_path = os.path.join(out_dir, "Fig_Physical_Decoder.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"  - Saved {save_path}")
    plt.close()
# ==========================================
# 4. 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, required=True, help="Path to teacher csv")
    parser.add_argument("--student", type=str, required=True, help="Path to student csv")
    parser.add_argument("--out_dir", type=str, default=".")
    args, unknown = parser.parse_known_args()
    
    t_df = load_data(args.teacher)
    s_df = load_data(args.student)
    
    if t_df is not None and s_df is not None:
        # 1. 数据增强：计算速度、加速度
        print("[Info] Augmenting data with derivatives...")
        t_df = augment_data_for_energy_analysis(t_df)
        s_df = augment_data_for_energy_analysis(s_df)
        
        # 2. 绘制基础图表
        plot_fig1_position(t_df, s_df, args.out_dir)
        plot_fig2_swing(t_df, s_df, args.out_dir)
        plot_fig3_latent(s_df, args.out_dir)
        # === [新增调用] ===
        # 绘制物理参数解码验证图
        plot_fig_physical_decoder(s_df, args.out_dir)
        # 3. 绘制高级能量机理图 (Figure 6 Combined)
        # 为 Teacher 生成
        plot_advanced_energy_mechanism(t_df, 
                                     os.path.join(args.out_dir, "Fig4_Teacher_Energy_Mechanism.png"), 
                                     "Teacher")
        
        # 为 Student 生成
        plot_advanced_energy_mechanism(s_df, 
                                     os.path.join(args.out_dir, "Fig5_Student_Energy_Mechanism.png"), 
                                     "Student")
        
    else:
        print("Failed to load CSV files.")

if __name__ == "__main__":
    main()