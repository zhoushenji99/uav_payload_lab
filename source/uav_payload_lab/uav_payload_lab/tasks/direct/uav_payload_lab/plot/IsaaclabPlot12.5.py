import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R  # [新增] 用于四元数转欧拉角

# --- 1. 配置 ---
# [修改] 默认读取当前目录下的 payload_data.csv (根据你的实际文件名修改)
simulation_data_path = "/home/shenji/uav_payload_lab/uav_payload_lab/source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_lab/plot/payload_data copy 5.csv" 
paper_data_path = "/home/shenji/uav_payload_lab/uav_payload_lab/source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_lab/plot/普通控制器vs.heanhua.csv" 

# 坐标系校正 (World -> Task)
OFFSET_X = 21.0
OFFSET_Y = -15.0

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

# --- 3. 加载并清洗论文数据 (保持原有逻辑) ---
paper_data = {} 
paper_available = False

try:
    df_paper = pd.read_csv(paper_data_path, header=None)
    print(f"[Paper] 成功加载 '{paper_data_path}'")
    paper_available = True
    
    groups = {
        "px": (0, 1), "py": (2, 3), "pz": (4, 5),
        "thetax": (6, 7), "thetay": (8, 9)
    }
    
    for key, (t_col, v_col) in groups.items():
        if v_col < df_paper.shape[1]:
            sub_df = df_paper[[t_col, v_col]].dropna()
            sub_df.columns = ["time", "value"]
            sub_df = sub_df.sort_values(by="time")
            paper_data[key] = sub_df
            print(f"  - {key}: {len(sub_df)} 点")

except Exception as e:
    print(f"[Paper] 未加载论文数据或读取失败 (可忽略): {e}")

# ==========================================
#               绘图部分
# ==========================================

# ------------------------------------------
# Figure 1: Payload 轨迹对比
# ------------------------------------------
fig1, axs1 = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
fig1.suptitle(f"Figure 1: Payload Control Performance (Sim vs Paper)", fontsize=16, weight="bold")

def plot_curve(ax, sim_x, sim_y, key_paper, title, ref=None, ylabel="Position (m)"):
    ax.set_title(title, fontsize=12)
    ax.plot(sim_x, sim_y, label="My RL (Sim)", linewidth=2.5, color='tab:blue')
    if paper_available and key_paper in paper_data:
        pdf = paper_data[key_paper]
        ax.plot(pdf["time"], pdf["value"], linestyle="--", label="Hean Hua (Paper)", linewidth=2, color='tab:orange')
    if ref is not None:
        ax.axhline(y=ref, linestyle=":", color='gray', alpha=0.8, label=f"Ref: {ref}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

plot_curve(axs1[0, 0], df_sim["time"], df_sim["payload_x"], "px", "Payload X", -0.5)
plot_curve(axs1[0, 1], df_sim["time"], df_sim["payload_y"], "py", "Payload Y", 0.0)
plot_curve(axs1[0, 2], df_sim["time"], df_sim["payload_z"], "pz", "Payload Z", 1.2)
plot_curve(axs1[1, 0], df_sim["time"], df_sim["theta_x_deg"], "thetax", "Swing Theta X", 0.0, "Angle (deg)")
plot_curve(axs1[1, 1], df_sim["time"], df_sim["theta_y_deg"], "thetay", "Swing Theta Y", 0.0, "Angle (deg)")
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