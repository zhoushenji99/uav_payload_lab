# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play with Student Policy and Record Comprehensive Metrics (CSV)."""

import argparse
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import csv
import time
import random

from isaaclab.app import AppLauncher

# ================= 配置区域 =================
# 请确保这里填的是你 best_cnn_encoder.pth 的绝对路径
ENCODER_PATH = "/home/shenji/uav_payload_lab/uav_payload_lab/best_cnn_encoder.pth"
# ===========================================

# add argparse arguments
parser = argparse.ArgumentParser(description="Play with Student Policy.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Uav-Meta-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the RSL-RL policy checkpoint.")
parser.add_argument("--use_oracle", action="store_true", default=False, help="If True, use Ground Truth (Cheating).")
parser.add_argument("--max_steps", type=int, default=3000, help="Number of steps to run for evaluation.")
# [核心修复] 注册 --seed 参数
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

# 引入必要的库
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# 注册你的自定义环境
import uav_payload_lab.tasks

# === 手动定义 set_seed 函数 ===
def set_seed(seed: int):
    """Set the seed for random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"[INFO] Random Seed set to: {seed}")

# === Encoder 定义 ===
class CNNEncoder(nn.Module):
    def __init__(self, input_dim=21, history_len=50, output_dim=2):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(input_dim, 64, 5, 1, 2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Flatten()
        )
        flatten_dim = 64 * history_len
        self.regressor = nn.Sequential(
            nn.Linear(flatten_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        return self.regressor(self.cnn_layers(x))

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Play with RSL-RL agent."""
    
    # 1. 设置随机种子
    current_seed = args_cli.seed if args_cli.seed is not None else 42
    set_seed(current_seed)

    # 2. 覆盖环境参数
    env_cfg.scene.num_envs = args_cli.num_envs
    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    device = env.unwrapped.device
    base_env = env.unwrapped

    # 3. 加载 Teacher Policy
    print(f"[INFO] Loading Policy from: {args_cli.checkpoint}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)

    # 4. 加载 Student Encoder
    print(f"[INFO] Loading Encoder from: {ENCODER_PATH}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder file not found: {ENCODER_PATH}")
        
    encoder = CNNEncoder().to(device)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    encoder.eval()

    # 5. 初始化 Buffer
    history_len = 50
    obs_history = torch.zeros((env.num_envs, history_len, 21), device=device)
    last_actions = torch.zeros((env.num_envs, 4), device=device)
    
    print(f"\n[INFO] Starting Simulation...")
    mode_name = 'Oracle' if args_cli.use_oracle else 'Student'
    print(f"[MODE] {mode_name}")
    
    # === Metrics 记录列表 ===
    csv_data = []
    
    obs, _ = env.reset()
    
    # 初始化阶段先清零一次计数器
    if hasattr(env.unwrapped, "episode_length_buf"):
        env.unwrapped.episode_length_buf[:] = 0

    timestep = 0
    dt = env.unwrapped.step_dt
    prev_uav_v_w = None
    
    try:
        with torch.inference_mode():
            while simulation_app.is_running():
                # 防止超时重置
                if hasattr(env.unwrapped, "episode_length_buf"):
                    env.unwrapped.episode_length_buf[:] = 0

                # ================= [Encoder 推理] =================
                obs_tensor = obs
                # 稳健的 Tensor 提取
                if not isinstance(obs_tensor, torch.Tensor):
                    if hasattr(obs_tensor, "get") and obs_tensor.get("policy") is not None:
                         obs_tensor = obs_tensor["policy"]
                    elif hasattr(obs_tensor, "keys") and "policy" in obs_tensor.keys():
                         obs_tensor = obs_tensor["policy"]
                if not isinstance(obs_tensor, torch.Tensor):
                    if hasattr(obs_tensor, "keys") and "policy" in obs_tensor.keys():
                         obs_tensor = obs_tensor["policy"]

                if not isinstance(obs_tensor, torch.Tensor):
                     print("Warning: Obs is not tensor, skipping frame.")
                     continue

                # 更新历史 Buffer
                obs_proprio = obs_tensor[:, :17] 
                current_feat = torch.cat([obs_proprio, last_actions], dim=1)
                obs_history = torch.roll(obs_history, shifts=-1, dims=1)
                obs_history[:, -1, :] = current_feat

                # Encoder 预测
                pred_params = encoder(obs_history) # [Mass_Norm, Len_Norm]
                
                # ================= [构建 Policy Input] =================
                policy_input_tensor = obs_tensor.clone()
                
                if not args_cli.use_oracle:
                    # Student 模式：用预测值覆盖
                    policy_input_tensor[:, 17:19] = pred_params
                # Oracle 模式：obs_tensor 本身就包含真值，不需要动

                # ================= [Policy 执行] =================
                if hasattr(obs, "clone"):
                    policy_input_dict = obs.clone()
                    policy_input_dict["policy"] = policy_input_tensor
                    actions = policy(policy_input_dict)
                elif isinstance(obs, dict):
                    policy_input_dict = obs.copy()
                    policy_input_dict["policy"] = policy_input_tensor
                    actions = policy(policy_input_dict)
                else:
                    actions = policy(policy_input_tensor)
                
                # 保存这一帧的动作
                actions_curr = actions.clone()

                # 环境步进
                obs, _, dones, _ = env.step(actions)
                last_actions = actions.clone()
                
                # Reset 历史
                if torch.any(dones):
                    obs_history[dones] = 0.0
                    last_actions[dones] = 0.0

                # ================= [详细数据记录] =================
                # 只记录第 0 个环境的数据
                env_idx = 0 

                # 1. 基础时间
                current_time = timestep * dt

                # 2. UAV 状态
                uav_pos = base_env._robot.data.root_pos_w[env_idx].cpu().numpy()
                uav_quat = base_env._robot.data.root_quat_w[env_idx].cpu().numpy()
                uav_v_b = base_env._robot.data.root_lin_vel_b[env_idx].cpu().numpy()
                uav_w_b = base_env._robot.data.root_ang_vel_b[env_idx].cpu().numpy()
                
                # 3. 计算世界系加速度
                uav_v_w = base_env._robot.data.root_lin_vel_w[env_idx].cpu().numpy()
                if prev_uav_v_w is None:
                    uav_a_w = np.zeros(3)
                else:
                    uav_a_w = (uav_v_w - prev_uav_v_w) / dt
                prev_uav_v_w = uav_v_w.copy()

                # 4. Payload 位置
                if hasattr(base_env, "_payload_id"):
                    payload_idx = base_env._payload_id
                    p_load = base_env._robot.data.body_pos_w[env_idx, payload_idx, :].cpu().numpy()
                else:
                    p_load = np.array([0., 0., 0.])

                # 5. 直接读取目标位置 (Target Position)
                if hasattr(base_env, "_desired_pos_w"):
                    target_pos = base_env._desired_pos_w[env_idx].cpu().numpy()
                else:
                    target_pos = np.array([0., 0., 0.])

                # 6. 摆角信息
                tilt_deg = obs_tensor[env_idx, 3:5].cpu().numpy() 
                swing_mag = np.linalg.norm(tilt_deg)

                # 7. 误差与估计
                pos_error = np.linalg.norm(target_pos - p_load)
                
                true_mass_norm = obs_tensor[env_idx, 17].item()
                true_len_norm = obs_tensor[env_idx, 18].item()
                pred_mass_norm = pred_params[env_idx, 0].item()
                pred_len_norm = pred_params[env_idx, 1].item()
                
                mass_err = abs(true_mass_norm - pred_mass_norm)
                len_err = abs(true_len_norm - pred_len_norm)

                # 8. 动作指令
                a_policy = actions_curr[env_idx].cpu().numpy()
                if hasattr(base_env, "_raw_actions"):
                    a_env_raw = base_env._raw_actions[env_idx].cpu().numpy()
                else:
                    a_env_raw = a_policy
                
                if hasattr(base_env, "_actions"):
                    a_env_clamp = base_env._actions[env_idx].cpu().numpy()
                else:
                    a_env_clamp = a_policy

                if hasattr(base_env, "_thrust"):
                    thrust_cmd = float(base_env._thrust[env_idx, 0, 2].cpu())
                else:
                    thrust_cmd = 0.0
                
                if hasattr(base_env, "_moment"):
                    moment_cmd = base_env._moment[env_idx, 0, :].cpu().numpy()
                else:
                    moment_cmd = np.array([0., 0., 0.])

                # === 打包 CSV ===
                csv_data.append([
                    current_time,
                    # UAV State
                    uav_pos[0], uav_pos[1], uav_pos[2],
                    uav_quat[0], uav_quat[1], uav_quat[2], uav_quat[3],
                    uav_v_b[0], uav_v_b[1], uav_v_b[2],
                    uav_w_b[0], uav_w_b[1], uav_w_b[2],
                    uav_v_w[0], uav_v_w[1], uav_v_w[2],
                    uav_a_w[0], uav_a_w[1], uav_a_w[2],
                    # Payload State
                    p_load[0], p_load[1], p_load[2],
                    # Target Position
                    target_pos[0], target_pos[1], target_pos[2],
                    # Metrics
                    pos_error, swing_mag,
                    tilt_deg[0], tilt_deg[1],
                    # Estimation
                    true_mass_norm, pred_mass_norm, mass_err,
                    true_len_norm, pred_len_norm, len_err,
                    # Actions
                    a_policy[0], a_policy[1], a_policy[2], a_policy[3],
                    a_env_raw[0], a_env_raw[1], a_env_raw[2], a_env_raw[3],
                    a_env_clamp[0], a_env_clamp[1], a_env_clamp[2], a_env_clamp[3],
                    thrust_cmd, moment_cmd[0], moment_cmd[1], moment_cmd[2]
                ])

                timestep += 1
                if timestep % 100 == 0:
                    print(f"Step {timestep:04d} | "
                          f"PosErr: {pos_error:.3f} | "
                          f"Mass(T/P): {true_mass_norm:.2f}/{pred_mass_norm:.2f} | "
                          f"Len(T/P): {true_len_norm:.2f}/{pred_len_norm:.2f}")

                if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
                    print(f"\n[INFO] Reached max steps ({args_cli.max_steps}). Finishing...")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    # ================= [保存 CSV] =================
    csv_filename = f"payload_data_{mode_name.lower()}.csv"
    csv_path = os.path.join(os.getcwd(), csv_filename)
    
    print(f"\n{'='*20} Saving Data {'='*20}")
    print(f"Saving {len(csv_data)} frames to: {csv_path}")
    
    try:
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Time", 
                "UAV_X","UAV_Y","UAV_Z", 
                "UAV_quat_0","UAV_quat_1","UAV_quat_2","UAV_quat_3", 
                "UAV_v_bx","UAV_v_by","UAV_v_bz", 
                "UAV_w_bx","UAV_w_by","UAV_w_bz", 
                "UAV_v_wx","UAV_v_wy","UAV_v_wz", 
                "UAV_a_wx","UAV_a_wy","UAV_a_wz",
                "Payload_X","Payload_Y","Payload_Z", 
                "Target_X", "Target_Y", "Target_Z", 
                "Pos_Error_m", "Swing_Mag_Deg",
                "Tilt_Deg_X", "Tilt_Deg_Y",
                "True_Mass", "Pred_Mass", "Mass_Err",
                "True_Len", "Pred_Len", "Len_Err",
                "Policy_a0","Policy_a1","Policy_a2","Policy_a3", 
                "Env_raw_a0","Env_raw_a1","Env_raw_a2","Env_raw_a3", 
                "Env_clamp_a0","Env_clamp_a1","Env_clamp_a2","Env_clamp_a3", 
                "Thrust_Cmd","Moment_Cmd_X","Moment_Cmd_Y","Moment_Cmd_Z"
            ])
            writer.writerows(csv_data)
        print("[INFO] CSV saved successfully.")
    except Exception as e:
        print(f"[ERROR] CSV Save failed: {e}")

    # 简单统计
    if len(csv_data) > 0:
        data_np = np.array(csv_data)
        avg_pos = np.mean(data_np[:, 26]) # Pos_Error_m
        avg_mass = np.mean(data_np[:, 32]) # Mass_Err
        print(f"Average Position Error: {avg_pos:.4f} m")
        if not args_cli.use_oracle:
            print(f"Average Mass Est Error: {avg_mass:.4f} (norm)")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()