# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play with Student Policy (Encoder + Teacher) - Fixed Reset Logic."""

import argparse
import sys
import os
import torch
import torch.nn as nn
import numpy as np

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
    
    # 1. 覆盖环境参数
    env_cfg.scene.num_envs = args_cli.num_envs
    # 2. 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    device = env.unwrapped.device

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
    print(f"[MODE] {'USING ORACLE (Cheating)' if args_cli.use_oracle else 'USING ENCODER (Student Inference)'}")
    
    obs, _ = env.reset()
    
    # [关键修复1] 初始化阶段先清零一次计数器
    if hasattr(env.unwrapped, "episode_length_buf"):
        env.unwrapped.episode_length_buf[:] = 0

    with torch.inference_mode():
        while simulation_app.is_running():
            # [关键修复2] 在循环中每一步都清零计数器，防止超时重置
            # 这样 Mass 就永远不会因为时间到了而改变，只有撞机才会 Reset
            if hasattr(env.unwrapped, "episode_length_buf"):
                env.unwrapped.episode_length_buf[:] = 0

            # ================= [Encoder 推理环节] =================
            # [关键修复3] 采用 play.py 中更健壮的 Tensor 提取逻辑
            obs_tensor = obs
            # 如果是 TensorDict，取 policy
            if not isinstance(obs_tensor, torch.Tensor):
                if hasattr(obs_tensor, "get") and obs_tensor.get("policy") is not None:
                     obs_tensor = obs_tensor["policy"]
                elif hasattr(obs_tensor, "keys") and "policy" in obs_tensor.keys():
                     obs_tensor = obs_tensor["policy"]
            
            # 确保一定是 Tensor
            if not isinstance(obs_tensor, torch.Tensor):
                 # 极端情况下的 fallback
                 print("Warning: Obs is not tensor, skipping frame.")
                 continue

            # 2. 更新历史 Buffer
            obs_proprio = obs_tensor[:, :17] 
            current_feat = torch.cat([obs_proprio, last_actions], dim=1)
            
            obs_history = torch.roll(obs_history, shifts=-1, dims=1)
            obs_history[:, -1, :] = current_feat

            # 3. Encoder 预测
            pred_params = encoder(obs_history) # 输出: [Mass_Norm, Len_Norm]
            
            # ================= [欺骗 Policy] =================
            policy_input_tensor = obs_tensor.clone()
            
            if not args_cli.use_oracle:
                # 用预测值覆盖
                policy_input_tensor[:, 17:19] = pred_params
                
                # 可视化打印
                if env.unwrapped.common_step_counter % 50 == 0:
                    try:
                        true_mass_obs = obs_tensor[0, 17].item()
                        pred_mass_obs = pred_params[0, 0].item()
                        err = abs(true_mass_obs - pred_mass_obs)
                        print(f"Step {env.unwrapped.common_step_counter:04d} | Mass(Norm) True: {true_mass_obs:.3f} vs Pred: {pred_mass_obs:.3f} | Err: {err:.3f}")
                    except:
                        pass

            # 4. Policy 执行
            # 重新封装回字典 (RSL-RL 格式要求)
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
            
            # 5. 环境步进
            obs, _, dones, _ = env.step(actions)
            last_actions = actions.clone()
            
            # Reset 处理 (只有撞机/坠毁时才会进入这里)
            if torch.any(dones):
                obs_history[dones] = 0.0
                last_actions[dones] = 0.0
                # 注意：这里我们允许它 Reset 历史，因为坠机了必须要重新开始

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()