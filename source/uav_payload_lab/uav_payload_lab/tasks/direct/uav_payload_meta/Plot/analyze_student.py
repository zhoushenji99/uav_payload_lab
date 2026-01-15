import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# ================= 配置区域 =================
# 你提供的绝对路径
FILE_STUDENT = "/home/shenji/uav_payload_lab/uav_payload_lab/source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_meta/Plot/payload_data_student.csv"
FILE_ORACLE  = "/home/shenji/uav_payload_lab/uav_payload_lab/source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_meta/Plot/payload_data_oracle.csv"
SAVE_DIR = "." # 图片保存路径
TIME_WINDOW_S = 5 # 设为 None 显示全过程，或设为 20.0 只看前20秒
# ===========================================

def process_dataframe(df, label):
    """
    完全复刻 IsaaclabPlot12.5.py 的处理逻辑
    包括列名映射、欧拉角计算、特别是能量计算
    """
    print(f"[{label}] Processing {len(df)} frames...")
    
    # 1. 欧拉角 (Quaternion -> Euler ZYX)
    if 'UAV_quat_1' in df.columns:
        quats = df[['UAV_quat_1', 'UAV_quat_2', 'UAV_quat_3', 'UAV_quat_0']].to_numpy()
        r = R.from_quat(quats)
        euler = r.as_euler('xyz', degrees=True)
        df['roll'] = euler[:, 0]
        df['pitch'] = euler[:, 1]
        df['yaw'] = euler[:, 2]

    # 2. 列名标准化 (兼容新旧 CSV)
    rename_map = {
        'Time': 'time', 
        'Tilt_Deg_X': 'theta_x_deg', 
        'Tilt_Deg_Y': 'theta_y_deg',
        'Swing_Deg_X': 'theta_x_deg',
        'Swing_Deg_Y': 'theta_y_deg',
        'SwingVel_DegS_X': 'theta_dot_x_deg', # 如果 CSV 里有这个更好
        'SwingVel_DegS_Y': 'theta_dot_y_deg'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # 3. 计算角速度 (如果 CSV 里没有直接记录，就用差分)
    # IsaaclabPlot12.5.py 逻辑：优先用 SwingVel_DegS_X
    if 'theta_dot_x_deg' not in df.columns:
        dt = np.diff(df['time'], prepend=df['time'].iloc[0])
        dt[dt == 0] = 0.02 # 防止除以0
        df['theta_dot_x_deg'] = np.diff(df['theta_x_deg'], prepend=df['theta_x_deg'].iloc[0]) / dt
        df['theta_dot_y_deg'] = np.diff(df['theta_y_deg'], prepend=df['theta_y_deg'].iloc[0]) / dt
    
    # ================= [核心：完全复刻能量算法] =================
    # 参考 IsaaclabPlot12.5.py: (B) 能量与功率 (Energy)
    
    # 准备数据 (转弧度)
    thx = df['theta_x_deg'].to_numpy(dtype=np.float32) * (np.pi/180.0)
    thy = df['theta_y_deg'].to_numpy(dtype=np.float32) * (np.pi/180.0)
    thdx = df['theta_dot_x_deg'].to_numpy(dtype=np.float32) * (np.pi/180.0)
    thdy = df['theta_dot_y_deg'].to_numpy(dtype=np.float32) * (np.pi/180.0)

    # 模长平方
    th2 = thx*thx + thy*thy
    thd2 = thdx*thdx + thdy*thdy

    # 获取物理参数
    g0 = 9.81
    # 垂直加速度修正
    a_z = df['UAV_a_wz'].to_numpy(dtype=np.float32) if 'UAV_a_wz' in df.columns else np.zeros_like(thx)
    
    # 绳长 L (优先用真值，如果没有就用默认)
    if 'True_Len' in df.columns:
        L_est = df['True_Len'].to_numpy(dtype=np.float32)
    else:
        L_est = 0.5 * np.ones_like(thx) # Default
    
    # 计算固有频率平方 w^2 = (g + a_z) / L
    # Clip L to avoid div by zero
    L_safe = np.clip(L_est, 0.1, 10.0)
    w2 = (g0 + a_z) / L_safe
    w2 = np.clip(w2, 0.1, 1000.0) # 防止负值导致虚数

    # E_hat_all = 0.5 * (thd2 + w2 * th2)
    # 这是单位质量的归一化能量 (J/kg/L^2 这种量纲，但作为指标足够了)
    df['Energy_Hat'] = 0.5 * (thd2 + w2 * th2)
    
    return df

def plot_comparison():
    # 1. 加载
    if not os.path.exists(FILE_STUDENT) or not os.path.exists(FILE_ORACLE):
        print(f"Error: 找不到 CSV 文件。\nStudent: {FILE_STUDENT}\nOracle: {FILE_ORACLE}")
        return

    df_stu = pd.read_csv(FILE_STUDENT)
    df_ora = pd.read_csv(FILE_ORACLE)

    # 2. 处理
    df_stu = process_dataframe(df_stu, "Student")
    df_ora = process_dataframe(df_ora, "Oracle")
    
    # 时间窗口裁剪
    if TIME_WINDOW_S is not None:
        df_stu = df_stu[df_stu['time'] <= TIME_WINDOW_S]
        df_ora = df_ora[df_ora['time'] <= TIME_WINDOW_S]

    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Figure 1: 轨迹与误差对比 (基本功) ---
    fig1, axs1 = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig1.suptitle("Figure 1: Trajectory & Tracking (Student vs Oracle)", fontsize=16, weight="bold")

    # X轴轨迹
    ax = axs1[0, 0]
    ax.plot(df_stu['time'], df_stu['Payload_X'], label='Student', color='tab:blue', linewidth=2)
    ax.plot(df_ora['time'], df_ora['Payload_X'], label='Oracle', color='tab:orange', linestyle='--', linewidth=2)
    if 'Target_X' in df_stu.columns:
        ax.plot(df_stu['time'], df_stu['Target_X'], label='Target', color='green', linestyle=':', linewidth=2)
    ax.set_title("Payload Position X")
    ax.legend()

    # Z轴轨迹
    ax = axs1[0, 1]
    ax.plot(df_stu['time'], df_stu['Payload_Z'], label='Student', color='tab:blue')
    ax.plot(df_ora['time'], df_ora['Payload_Z'], label='Oracle', color='tab:orange', linestyle='--')
    if 'Target_Z' in df_stu.columns:
        ax.plot(df_stu['time'], df_stu['Target_Z'], label='Target', color='green', linestyle=':')
    ax.set_title("Payload Position Z (Height)")
    ax.legend()

    # 误差对比
    ax = axs1[1, 0]
    ax.plot(df_stu['time'], df_stu['Pos_Error_m'], label='Student', color='tab:blue')
    ax.plot(df_ora['time'], df_ora['Pos_Error_m'], label='Oracle', color='tab:orange', linestyle='--')
    ax.set_title("Position Tracking Error (m)")
    ax.set_ylabel("Error (m)")
    ax.legend()

    # 摆角大小
    ax = axs1[1, 1]
    ax.plot(df_stu['time'], df_stu['Swing_Mag_Deg'], label='Student', color='tab:blue')
    ax.plot(df_ora['time'], df_ora['Swing_Mag_Deg'], label='Oracle', color='tab:orange', linestyle='--')
    ax.set_title("Swing Magnitude (deg)")
    ax.legend()
    
    fig1.savefig(os.path.join(SAVE_DIR, 'compare_1_trajectory.png'), dpi=150)
    print("生成: compare_1_trajectory.png")

    # --- Figure 2: 能量统计对比 (你的核心需求) ---
    fig2, axs2 = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    fig2.suptitle("Figure 2: Residual Oscillation Energy (E_hat) Comparison", fontsize=16, weight="bold")

    # 1. 绝对能量曲线
    ax = axs2[0]
    ax.plot(df_stu['time'], df_stu['Energy_Hat'], label='Student Energy', color='tab:blue', linewidth=1.5)
    ax.plot(df_ora['time'], df_ora['Energy_Hat'], label='Oracle Energy', color='tab:orange', linestyle='--', linewidth=1.5)
    ax.set_title("Residual Energy (Kinetic + Potential)")
    ax.set_ylabel("Energy (Normalized)")
    ax.legend()
    ax.grid(True, which='both', linestyle='--')

    # 2. 能量差值 (Student - Oracle)
    ax = axs2[1]
    # 对齐长度
    min_len = min(len(df_stu), len(df_ora))
    diff = df_stu['Energy_Hat'].iloc[:min_len].values - df_ora['Energy_Hat'].iloc[:min_len].values
    t_common = df_stu['time'].iloc[:min_len]
    
    ax.plot(t_common, diff, color='purple', label='Diff (Student - Oracle)')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    
    # 填充颜色
    ax.fill_between(t_common, diff, 0, where=(diff>0), color='red', alpha=0.1, label='Student Worse')
    ax.fill_between(t_common, diff, 0, where=(diff<0), color='green', alpha=0.1, label='Student Better')
    
    ax.set_title("Energy Gap Analysis (Lower is Better)")
    ax.set_ylabel("Energy Delta")
    ax.legend()

    fig2.savefig(os.path.join(SAVE_DIR, 'compare_2_energy.png'), dpi=150)
    print("生成: compare_2_energy.png")

    # --- Figure 3: UAV 动力学 (状态空间) ---
    fig3, axs3 = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig3.suptitle("Figure 3: UAV Dynamics Comparison", fontsize=16, weight="bold")

    # Vel X
    ax = axs3[0, 0]
    ax.plot(df_stu['time'], df_stu['UAV_v_wx'], label='Student', color='tab:blue')
    ax.plot(df_ora['time'], df_ora['UAV_v_wx'], label='Oracle', color='tab:orange', linestyle='--')
    ax.set_title("World Velocity X")
    ax.legend()

    # Accel X
    ax = axs3[0, 1]
    ax.plot(df_stu['time'], df_stu['UAV_a_wx'], label='Student', color='tab:blue')
    ax.plot(df_ora['time'], df_ora['UAV_a_wx'], label='Oracle', color='tab:orange', linestyle='--')
    ax.set_title("World Acceleration X")

    # Pitch Angle (姿态响应)
    ax = axs3[1, 0]
    ax.plot(df_stu['time'], df_stu['pitch'], label='Student', color='tab:blue')
    ax.plot(df_ora['time'], df_ora['pitch'], label='Oracle', color='tab:orange', linestyle='--')
    ax.set_title("Pitch Angle (deg)")
    
    # Thrust Command (控制量)
    ax = axs3[1, 1]
    ax.plot(df_stu['time'], df_stu['Thrust_Cmd'], label='Student', color='tab:blue')
    ax.plot(df_ora['time'], df_ora['Thrust_Cmd'], label='Oracle', color='tab:orange', linestyle='--')
    ax.set_title("Thrust Command")

    fig3.savefig(os.path.join(SAVE_DIR, 'compare_3_dynamics.png'), dpi=150)
    print("生成: compare_3_dynamics.png")
    
    # --- Figure 5: 估计性能 (Student Focus) ---
    # 这张图主要展示 Student 到底猜得准不准，Oracle 在这里只提供 True 值基准
    if 'True_Mass' in df_stu.columns:
        fig5, axs5 = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        fig5.suptitle("Figure 5: Student Estimation Performance", fontsize=16, weight="bold")

        # Mass
        ax = axs5[0, 0]
        ax.plot(df_stu['time'], df_stu['True_Mass'], 'k--', label='True Mass', linewidth=2)
        ax.plot(df_stu['time'], df_stu['Pred_Mass'], 'b-', label='Student Pred', linewidth=1.5)
        ax.set_title("Mass Estimation")
        ax.legend()
        
        ax = axs5[0, 1]
        ax.plot(df_stu['time'], df_stu['Mass_Err'], 'r-', label='Abs Error')
        ax.set_title("Mass Prediction Error")
        
        # Length
        ax = axs5[1, 0]
        ax.plot(df_stu['time'], df_stu['True_Len'], 'k--', label='True Len', linewidth=2)
        ax.plot(df_stu['time'], df_stu['Pred_Len'], 'g-', label='Student Pred', linewidth=1.5)
        ax.set_title("Length Estimation")
        ax.legend()
        
        ax = axs5[1, 1]
        ax.plot(df_stu['time'], df_stu['Len_Err'], 'r-', label='Abs Error')
        ax.set_title("Length Prediction Error")
        
        fig5.savefig(os.path.join(SAVE_DIR, 'compare_5_estimation.png'), dpi=150)
        print("生成: compare_5_estimation.png")

    plt.show()

if __name__ == "__main__":
    plot_comparison()