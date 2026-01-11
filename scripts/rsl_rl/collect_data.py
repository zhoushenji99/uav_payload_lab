# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os          # <--- 新增
import torch       # <--- 新增
import time        # <--- 确保有这个
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
# ==================== [请插入下面这一行] ====================
parser.add_argument("--steps", type=int, default=1000, help="Number of steps to collect data.")
# ==========================================================
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

import gymnasium as gym
import os
import time
import torch

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

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
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

    # reset environment
    obs = env.get_observations()
    # ================= [新增代码 START] 初始化数据采集容器 =================
    print("[Data Collection] Initializing buffers...")
    
    # 1. 配置参数
    history_len = 50       # 历史窗口长度 (Student 输入是一段视频流)
    proprio_dim = 17       # 你的状态里，去掉 Oracle 后的维度 (19 - 2 = 17)
    action_dim = 4         # 动作维度
    
    # 2. 核心 Buffer：Encoder 的输入历史
    # Shape: [num_envs, 50, 21]  (21 = 17状态 + 4动作)
    # 使用 device 也就是 GPU 上的内存，速度最快
    obs_history = torch.zeros((env.num_envs, history_len, proprio_dim + action_dim), device=env.device)
    
    # 3. 记录“上一次动作”的变量 (因为 Encoder 输入通常包含上一帧动作)
    last_actions = torch.zeros((env.num_envs, action_dim), device=env.device)
    
    # 4. 最终的大列表 (用来存所有数据，最后再一次性写入硬盘)
    dataset_inputs = []
    dataset_labels = []
    
    # 5. 采集总步数控制
    collect_total_steps = args_cli.steps # 采集多少步？(4096环境 * 1000步 = 400万条数据)
    current_step_count = 0
    # ================= [新增代码 END] =================
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # ================= [修正代码 START] =================
            # 1. 【新增步骤】先从 TensorDict 里把真正的观测 Tensor 拿出来
            # 这里的 "policy" 对应你日志里的 Resolved observation sets: policy
            if isinstance(obs, dict) or hasattr(obs, "keys"):
                obs_tensor = obs["policy"] 
            else:
                obs_tensor = obs
#           2. 现在用 obs_tensor 来切片（不要用 obs 了）
            # 构造 Student (Encoder) 的输入：切掉 Oracle，拼上 Action
            obs_proprio = obs_tensor[:, :17]  # <--- 改了这里
            
            # 将 (当前状态17维 + 上一帧动作4维) 拼接 -> 21维
            current_feat = torch.cat([obs_proprio, last_actions], dim=1)
            
            # 更新滑动窗口 (FIFO队列)
            # 把旧数据往左移，把新数据填到最右边
            obs_history = torch.roll(obs_history, shifts=-1, dims=1)
            obs_history[:, -1, :] = current_feat
            
            # 2. 获取 Label (真值)
            # 注意：这里需要你的 env 里能访问 payload_mass。
            # 如果报错 'AttributeError'，请检查 meta_uav_env.py 是否有 self.payload_mass
            try:
                # 尝试直接访问 (假设 env.unwrapped 能拿到)
                # 必须 reshape 成 (N, 1) 方便拼接
                mass_true = env.unwrapped.payload_mass.unsqueeze(1) 
                rope_true = env.unwrapped.rope_length.unsqueeze(1)
                
                # 拼接成 (N, 2) 的标签
                true_params = torch.cat([mass_true, rope_true], dim=1)
            except AttributeError:
                # 如果代码报错找不到 payload_mass，暂时用 obs 里的 oracle 代替测试
                # print("Warning: Can't find real mass, using oracle obs instead.")
                true_params = obs_tensor[:, 17:19]

            # 3. 存入列表 (为了防止爆显存，每一步的数据转到 CPU 或者保留在 GPU 看你内存大小)
            # 建议：如果内存够大 (64G+)，保留在 GPU 也可以；为了稳妥，.cpu()
            # 只有当历史数据填满(>50步)后才开始存，否则前面都是0没意义
            if current_step_count > history_len:
                # 使用 float16 节省一半空间
                dataset_inputs.append(obs_history.clone().cpu().to(torch.float16))
                dataset_labels.append(true_params.clone().cpu().to(torch.float32))
            
            # 4. 更新 last_actions (供下一帧使用)
            last_actions = actions.clone()
            
            current_step_count += 1
            # ================= [新增代码 END] =================
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # ================= [新增代码 START] 处理 Reset =================
            # 如果某个环境重置了(dones=True)，它的历史 buffer 应该清空，防止把上一局的数据带到下一局
            if torch.any(dones):
                obs_history[dones] = 0.0
                last_actions[dones] = 0.0
            # ================= [新增代码 END] =================
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        # ================= [新增代码 START] 采集够了就退出并保存 =================
        if current_step_count >= collect_total_steps + history_len:
            print(f"Collected {len(dataset_inputs)} steps of data. Saving...")
            
            # 拼接所有列表 -> 大 Tensor
            # inputs shape: (Total_Steps * Num_Envs, 50, 21)
            # labels shape: (Total_Steps * Num_Envs, 2)
            full_inputs = torch.cat(dataset_inputs, dim=0)
            full_labels = torch.cat(dataset_labels, dim=0)
            
            save_path = os.path.join(os.path.dirname(resume_path), "encoder_dataset.pt")
            torch.save({
                "inputs": full_inputs,
                "labels": full_labels
            }, save_path)
            
            print(f"Saved dataset to {save_path}")
            print(f"Input Shape: {full_inputs.shape}, Label Shape: {full_labels.shape}")
            break # 退出循环
        # ================= [新增代码 END] =================
        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()