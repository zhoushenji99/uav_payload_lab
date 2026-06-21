# play.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
# [修正1] 添加必要的库导入
import csv
import os
import time
import numpy as np
import torch
import gymnasium as gym
import random
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import uav_payload_lab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    # ===== force deterministic eval seed =====
    if args_cli.seed is not None:
        agent_cfg.seed = int(args_cli.seed)

    seed = int(agent_cfg.seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env_cfg.seed = seed

    print(f"[DEBUG] play seed = {seed}")
    print(f"[DEBUG] env_cfg.seed = {env_cfg.seed}")
    print(f"[DEBUG] agent_cfg.seed = {agent_cfg.seed}")
    
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cfg = agent_cfg.to_dict()
    runner_cfg["policy"].update(
        {
            "proprio_obs_dim": int(getattr(env_cfg, "proprio_obs_dim", 21)),
            "privileged_obs_dim": int(getattr(env_cfg, "privileged_obs_dim", 5)),
            "z_dim": int(getattr(env_cfg, "rma_z_dim", 5)),
            "z_exp_dim": int(getattr(env_cfg, "rma_z_exp_dim", 2)),
            "use_mu": bool(getattr(env_cfg, "rma_use_mu", True)),
            "use_physics_anchor": bool(getattr(env_cfg, "rma_use_physics_anchor", False)),
        }
    )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # --- robust load: model weights only (ignore optimizer mismatch) ---
    try:
        runner.load(resume_path)
    except Exception as e:
        print(f"[WARN] runner.load failed: {e}")
        print("[WARN] Fallback: load model_state_dict only (skip optimizer).")
        ckpt = torch.load(resume_path, map_location=agent_cfg.device)

        policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        policy_nn.load_state_dict(ckpt["model_state_dict"], strict=False)
    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    
    # [新增] 准备 CSV 数据存储
    csv_data = []
    print("[INFO] Starting simulation loop and recording payload data...")
    
    # reset environment
    obs = env.get_observations()
    # [新增] 强制清零时间计数器，防止 play 时发生随机 reset
    if hasattr(env.unwrapped, "episode_length_buf"):
        env.unwrapped.episode_length_buf[:] = 0
    timestep = 0
    # [新增] 初始化上一帧速度变量，用于计算加速度
    prev_uav_v_w = None
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # 保存策略原始输出 (Policy Output)
            actions_policy = actions.clone()
            
            # env stepping
            obs, _, dones, _ = env.step(actions)
            
            # [新增] 如果这一步触发了重置，就不记录数据了，停止录制以保证数据是一条完整轨迹
            if dones.any():
                print("[INFO] Episode finished (reset triggered). Stopping recording.")
                break
            
            # =========================================================================
            # [新增] 详细数据记录逻辑 (Data Collection)
            # =========================================================================
            base_env = env.unwrapped
            
            # 1. [关键修复] 获取 Obs Tensor，解决 IndexError
            # 无论 obs 是 dict, TensorDict 还是 Tensor，都强制拆包到底层 Tensor
            obs_tensor = obs
            if not isinstance(obs_tensor, torch.Tensor):
                # 尝试取 'policy' 键
                if hasattr(obs_tensor, "get") and obs_tensor.get("policy") is not None:
                     obs_tensor = obs_tensor["policy"]
                elif hasattr(obs_tensor, "keys") and "policy" in obs_tensor.keys():
                     obs_tensor = obs_tensor["policy"]

            # 二次检查：防止嵌套 (TensorDict 里包 TensorDict)
            if not isinstance(obs_tensor, torch.Tensor):
                if hasattr(obs_tensor, "keys") and "policy" in obs_tensor.keys():
                     obs_tensor = obs_tensor["policy"]
            
            # 如果此时还不是 Tensor，说明结构非常特殊，打印错误但不崩溃
            if not isinstance(obs_tensor, torch.Tensor):
                if timestep == 0: print(f"[WARNING] Could not extract Tensor from obs. Type: {type(obs)}")
                # 使用默认零值防止报错
                tilt_deg = np.array([0.0, 0.0])
                swing_vel_deg_s = np.array([0.0, 0.0])
            else:
                # 正常提取 [0: num_envs, 3:5] -> 摆角
                tilt_deg = obs_tensor[0, 3:5].cpu().numpy() # [theta_x, theta_y]
                swing_vel_deg_s = obs_tensor[0, 5:7].cpu().numpy()

            # 2. 获取 Payload 真实位置 (World Frame)
            if hasattr(base_env, "_payload_id"):
                payload_idx = base_env._payload_id
                # 取第 0 个环境的 payload 位置
                p_load = base_env._robot.data.body_pos_w[0, payload_idx, :].cpu().numpy()
            else:
                p_load = np.array([0, 0, 0]) 

            # 3. 获取 UAV 状态
            uav_pos = base_env._robot.data.root_pos_w[0].cpu().numpy()
            uav_quat = base_env._robot.data.root_quat_w[0].cpu().numpy()
            uav_v_b = base_env._robot.data.root_lin_vel_b[0].cpu().numpy()
            uav_w_b = base_env._robot.data.root_ang_vel_b[0].cpu().numpy()
            # [新增] === 计算世界系加速度 (关键修改) ===
            # 1. 获取世界系线速度 (root_lin_vel_w)
            uav_v_w = base_env._robot.data.root_lin_vel_w[0].cpu().numpy()
            # 2. 差分计算加速度: a = (v_now - v_prev) / dt
            if prev_uav_v_w is None:
                uav_a_w = np.zeros(3) # 第一帧没法算，给0
            else:
                uav_a_w = (uav_v_w - prev_uav_v_w) / dt
            
            # 3. 更新历史变量
            prev_uav_v_w = uav_v_w.copy()
            # [结束新增] ============================
            # 4. 获取 动作信息
            a_policy = actions_policy[0].cpu().numpy()
            
            # 获取 Env 里的 Raw Action (未 Clamp)
            if hasattr(base_env, "_raw_actions"):
                a_env_raw = base_env._raw_actions[0].cpu().numpy()
            else:
                a_env_raw = a_policy # Fallback

            # 获取 Env 里的 Clamped Action (实际执行)
            if hasattr(base_env, "_actions"):
                a_env_clamp = base_env._actions[0].cpu().numpy()
            else:
                a_env_clamp = a_policy # Fallback

            # 5. 获取 推力与力矩指令
            # _thrust shape: (num_envs, 1, 3)
            if hasattr(base_env, "_thrust"):
                thrust_cmd = float(base_env._thrust[0, 0, 2].cpu()) 
            else:
                thrust_cmd = 0.0
            
            if hasattr(base_env, "_moment"):
                moment_cmd = base_env._moment[0, 0, :].cpu().numpy()
            else:
                moment_cmd = np.array([0, 0, 0])

            # 6. 记录时间
            current_time = timestep * dt
            # --- add near where you already have policy_nn and obs_tensor ---

            # privileged e is the tail of raw obs in Phase-1: [proprio_dim : proprio_dim + priv_dim]
            e_priv = np.zeros(5, dtype=np.float32)
            priv_start = int(getattr(base_env.cfg, "proprio_obs_dim", 21))
            priv_dim = int(getattr(base_env.cfg, "privileged_obs_dim", 5))
            if isinstance(obs_tensor, torch.Tensor) and priv_dim > 0 and obs_tensor.shape[1] >= priv_start + priv_dim:
                e_raw = obs_tensor[0, priv_start : priv_start + min(priv_dim, 5)].cpu().numpy()
                e_priv[: e_raw.shape[0]] = e_raw

            # z from teacher encoder μ(e). PPO baselines may have rma_z_dim=0,
            # but the CSV schema keeps five z columns for plotting compatibility.
            z_raw = np.zeros(getattr(base_env.cfg, "rma_z_dim", 5), dtype=np.float32)
            if hasattr(policy_nn, "last_z") and (policy_nn.last_z is not None):
                z_raw = policy_nn.last_z[0].detach().cpu().numpy()

            z = np.zeros(5, dtype=np.float32)
            z_len = min(5, z_raw.shape[0])
            if z_len > 0:
                z[:z_len] = z_raw[:z_len]

            zexp_dim = getattr(base_env.cfg, "rma_z_exp_dim", 2)
            z_exp = z[:zexp_dim]
            z_imp = z[zexp_dim:]

            # 7. 打包数据: [Time, UAV..., Payload..., Swing..., Actions..., Commands...]
            csv_data.append([
                current_time,
                uav_pos[0], uav_pos[1], uav_pos[2],
                uav_quat[0], uav_quat[1], uav_quat[2], uav_quat[3],
                uav_v_b[0], uav_v_b[1], uav_v_b[2],
                uav_w_b[0], uav_w_b[1], uav_w_b[2],
                # [新增] 世界系速度 (3维) & 世界系加速度 (3维)
                uav_v_w[0], uav_v_w[1], uav_v_w[2],
                uav_a_w[0], uav_a_w[1], uav_a_w[2],
                p_load[0], p_load[1], p_load[2],
                tilt_deg[0], tilt_deg[1],
                swing_vel_deg_s[0], swing_vel_deg_s[1],
                a_policy[0], a_policy[1], a_policy[2], a_policy[3],
                a_env_raw[0], a_env_raw[1], a_env_raw[2], a_env_raw[3],
                a_env_clamp[0], a_env_clamp[1], a_env_clamp[2], a_env_clamp[3],
                thrust_cmd, moment_cmd[0], moment_cmd[1], moment_cmd[2],
                e_priv[0], e_priv[1], e_priv[2], e_priv[3], e_priv[4],
                z[0], z[1], z[2], z[3], z[4],
            ])
            # ==========================

            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1
            if timestep >= 2100: # 35秒
                 print("[INFO] Reached max steps for recording.")
                 break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # [新增] 循环结束后写入 CSV
    csv_filename = os.path.join(log_dir, "payload_data.csv")
    print(f"[INFO] Saving payload data to: {csv_filename}")
    try:
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入详细表头
            writer.writerow([
                "Time", 
                "UAV_X","UAV_Y","UAV_Z", 
                "UAV_quat_0","UAV_quat_1","UAV_quat_2","UAV_quat_3", 
                "UAV_v_bx","UAV_v_by","UAV_v_bz", 
                "UAV_w_bx","UAV_w_by","UAV_w_bz", 
                # [新增] 对应的表头名
                "UAV_v_wx","UAV_v_wy","UAV_v_wz", 
                "UAV_a_wx","UAV_a_wy","UAV_a_wz",
                "Payload_X","Payload_Y","Payload_Z", 
                "Swing_Deg_X","Swing_Deg_Y", 
                "SwingVel_DegS_X","SwingVel_DegS_Y", 
                "Policy_a0","Policy_a1","Policy_a2","Policy_a3", 
                "Env_raw_a0","Env_raw_a1","Env_raw_a2","Env_raw_a3", 
                "Env_clamp_a0","Env_clamp_a1","Env_clamp_a2","Env_clamp_a3", 
                "Thrust_Cmd","Moment_Cmd_X","Moment_Cmd_Y","Moment_Cmd_Z",
                "e_m", "e_l", "e_wx", "e_wy", "e_wz",
                "z0","z1","z2","z3","z4"

            ])
            # 写入数据
            writer.writerows(csv_data)
        print("[INFO] Data saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save CSV: {e}")

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
