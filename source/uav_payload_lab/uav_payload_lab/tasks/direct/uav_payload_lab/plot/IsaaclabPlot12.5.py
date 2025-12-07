import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 配置 ---
simulation_data_path = "/home/shenji/uav_payload_lab/uav_payload_lab/source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_lab/plot/payload_data copy.csv"
paper_data_path = "/home/shenji/uav_payload_lab/uav_payload_lab/source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_lab/plot/普通控制器vs.heanhua.csv" 

# 坐标系校正 (World -> Task)
# 你的仿真起点 X=21.5, 论文起点 X=0.5 -> Offset = 21.0
# 你的仿真起点 Y=-14.0, 论文起点 Y=1.0 -> Offset = -15.0
OFFSET_X = 21.0
OFFSET_Y = -15.0

# --- 2. 加载仿真数据 ---
try:
    df_sim = pd.read_csv(simulation_data_path)
    print(f"[Sim] 成功加载 '{simulation_data_path}'，共 {len(df_sim)} 行")
except FileNotFoundError:
    print(f"错误: 找不到文件 '{simulation_data_path}'")
    raise SystemExit

# 列名标准化
if "Payload_X" in df_sim.columns:
    df_sim.rename(columns={
        "Time": "time",
        "Swing_Deg_X": "theta_x_deg",
        "Swing_Deg_Y": "theta_y_deg"
    }, inplace=True)
    
    # 坐标转换
    df_sim["payload_x"] = df_sim["Payload_X"] - OFFSET_X
    df_sim["payload_y"] = df_sim["Payload_Y"] - OFFSET_Y
    df_sim["payload_z"] = df_sim["Payload_Z"]
    
    print(f"[Sim] 坐标已校正: X -= {OFFSET_X}, Y -= {OFFSET_Y}")

# --- 3. 加载并清洗论文数据 ---
paper_data = {} # 存储清洗后的 DataFrame
paper_available = False

try:
    df_paper = pd.read_csv(paper_data_path, header=None)
    print(f"[Paper] 成功加载 '{paper_data_path}'")
    paper_available = True
    
    # 定义每组数据的列索引 (Time列, Value列)
    # 假设顺序: X, Y, Z, ThetaX, ThetaY
    groups = {
        "px": (0, 1),
        "py": (2, 3),
        "pz": (4, 5),
        "thetax": (6, 7),
        "thetay": (8, 9)
    }
    
    for key, (t_col, v_col) in groups.items():
        if v_col < df_paper.shape[1]:
            # 提取非空数据
            sub_df = df_paper[[t_col, v_col]].dropna()
            sub_df.columns = ["time", "value"]
            
            # [关键修复] 按时间排序，消除“倒流”
            sub_df = sub_df.sort_values(by="time")
            
            paper_data[key] = sub_df
            print(f"  - {key}: {len(sub_df)} 点 (已按时间排序)")

except Exception as e:
    print(f"[Paper] 读取警告: {e}")

# --- 4. 绘图 ---
fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
fig.suptitle(f"Payload Control: RL (Sigma=0.3) vs Hean Hua Paper", fontsize=18, weight="bold")

def plot_curve(ax, sim_x, sim_y, key_paper, title, ref=None, ylabel="Position (m)"):
    ax.set_title(title, fontsize=14)
    # 仿真曲线
    ax.plot(sim_x, sim_y, label="My RL (Sim)", linewidth=2.5, color='tab:blue')
    
    # 论文曲线
    if paper_available and key_paper in paper_data:
        pdf = paper_data[key_paper]
        ax.plot(pdf["time"], pdf["value"], linestyle="--", label="Hean Hua (Paper)", linewidth=2, color='tab:orange')
    
    # 参考线
    if ref is not None:
        ax.axhline(y=ref, linestyle=":", color='gray', alpha=0.8, label=f"Ref: {ref}")
        
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

# 绘制各子图
plot_curve(axs[0, 0], df_sim["time"], df_sim["payload_x"], "px", "Payload X", -0.5)
plot_curve(axs[0, 1], df_sim["time"], df_sim["payload_y"], "py", "Payload Y", 0.0)
plot_curve(axs[0, 2], df_sim["time"], df_sim["payload_z"], "pz", "Payload Z", 1.2)
plot_curve(axs[1, 0], df_sim["time"], df_sim["theta_x_deg"], "thetax", "Swing Theta X", 0.0, "Angle (deg)")
plot_curve(axs[1, 1], df_sim["time"], df_sim["theta_y_deg"], "thetay", "Swing Theta Y", 0.0, "Angle (deg)")

# 隐藏多余子图
axs[1, 2].axis('off')

# 保存
plt.savefig("final_comparison_fixed.png", dpi=300)
print("\n[Output] 图像已保存为 final_comparison_fixed.png")
plt.show()