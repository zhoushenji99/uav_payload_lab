# play_student_phase2.py
# Phase-2 closed-loop rollout for Teacher vs Student (RMA):
# - Teacher: z = mu(priv)  (oracle context)
# - Student: z_hat = encoder(history(proprio + last_action))
# Logs env{trace_env} to CSV for paper-grade plots (UAV pos, payload pos, swing, z, energy proxies).
#
# Key fixes vs your "Ultimate" script:
# 1) Teacher action input MUST be [proprio(17), z_teacher(5)], NOT [proprio(17), priv(5)].
# 2) Student encoder input MUST match collect_z_dataset.py: feature_t = [proprio(17), last_action(4)] => 21 dims.
# 3) Use IsaacLab's hydra_task_config + parse_known_args pattern (same as official play.py) so CLI works.

from __future__ import annotations

import argparse
import sys
import os
import csv
import time
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from isaaclab.app import AppLauncher
import cli_args  # isort: skip


# ----------------------------
# args
# ----------------------------
parser = argparse.ArgumentParser(description="Phase-2 Play: Teacher(mu(priv)) vs Student(encoder(history)) with CSV logging.")

# phase-2 specific
parser.add_argument("--mode", type=str, default="student", choices=["student", "teacher"])
parser.add_argument("--encoder", type=str, default="", help="Student encoder .pth (required if --mode student).")
parser.add_argument("--history_len", type=int, default=50)
parser.add_argument("--trace_env", type=int, default=0, help="Which env index to log to CSV.")
parser.add_argument("--stop_on_done", action="store_true", default=True, help="Stop when trace_env episode ends (default: True).")
parser.add_argument("--no_stop_on_done", dest="stop_on_done", action="store_false", help="Do not stop on done; keep running until max_steps.")
parser.add_argument("--max_steps", type=int, default=2000, help="Max env steps to run (avoid crossing episode boundary).")
parser.add_argument("--csv", type=str, default="", help="CSV output path. If empty, write into checkpoint folder.")

# standard play args
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--real-time", action="store_true", default=False, help="Sleep to match wall-clock dt.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------
# safe imports after SimulationApp
# ----------------------------
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks  # noqa: F401
import uav_payload_lab.tasks  # noqa: F401


# ----------------------------
# helpers
# ----------------------------
def _get_obs_tensor(obs_td):
    if isinstance(obs_td, torch.Tensor):
        return obs_td
    if hasattr(obs_td, "get") and obs_td.get("policy") is not None:
        return obs_td["policy"]
    if isinstance(obs_td, dict) and "policy" in obs_td:
        return obs_td["policy"]
    raise RuntimeError(f"Unsupported obs type: {type(obs_td)}")


def _safe_load_model_only(runner: OnPolicyRunner, ckpt_path: str):
    """Load ONLY model (and normalizers if exist). Avoid optimizer mismatch."""
    ckpt = torch.load(ckpt_path, map_location=runner.device)

    # common keys in rsl_rl / isaaclab
    if "model_state_dict" in ckpt:
        runner.alg.policy.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        runner.alg.policy.load_state_dict(ckpt["model"], strict=False)
    elif "state_dict" in ckpt:
        runner.alg.policy.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        # fall back: assume whole ckpt is state_dict
        runner.alg.policy.load_state_dict(ckpt, strict=False)

    # optional normalizers
    if "actor_obs_normalizer_state_dict" in ckpt and hasattr(runner.alg.policy, "actor_obs_normalizer"):
        runner.alg.policy.actor_obs_normalizer.load_state_dict(ckpt["actor_obs_normalizer_state_dict"])
    if "critic_obs_normalizer_state_dict" in ckpt and hasattr(runner.alg.policy, "critic_obs_normalizer"):
        runner.alg.policy.critic_obs_normalizer.load_state_dict(ckpt["critic_obs_normalizer_state_dict"])


def _default_csv_path(resume_path: str, mode: str) -> str:
    run_dir = os.path.dirname(resume_path)
    name = "phase2_teacher.csv" if mode == "teacher" else "phase2_student.csv"
    return os.path.join(run_dir, name)


# ----------------------------
# student encoder (must match train_student_z.py)
# ----------------------------
class CNNStudentEncoder(nn.Module):
    def __init__(self, input_dim=21, history_len=50, output_dim=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, 5, 1, 2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Flatten(),
        )
        flat = 64 * history_len
        self.mlp = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        # === [必须新增] decoder 定义 ===
        # 否则 load_state_dict 会报错 "Unexpected key: decoder..."
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2) 
        )
        # ============================
    def forward(self, x):
        x = x.permute(0, 2, 1)
        feat = self.cnn(x)
        z = self.mlp(feat)
        
        # === [必须新增] 计算物理预测 ===
        z_exp = z[:, :2]
        phys_pred = self.decoder(z_exp)
        
        return z, phys_pred  # 返回 Tuple (z, phys_pred)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    # ---- cfg wiring (mirror official play.py + collect_z_dataset.py) ----
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    if args_cli.seed is not None:
        agent_cfg.seed = int(args_cli.seed)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(agent_cfg.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # where to load checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    # === 【修改位置在这里】 ===
    # 强制把最大时长设为 10000秒，防止环境自动 Reset
    if hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = 10000.0
    # ========================
    # ---- env ----
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    # ---- runner + policy ----
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading checkpoint (model-only): {resume_path}")
    _safe_load_model_only(runner, resume_path)

    policy_nn = runner.alg.policy  # expected RMAActorCritic in your setup
    policy_nn.eval()

    # teacher must use mu(priv)
    if hasattr(policy_nn, "use_mu"):
        policy_nn.use_mu = True

    # ---- student encoder ----
    encoder = None
    if args_cli.mode == "student":
        if not args_cli.encoder:
            raise ValueError("--mode student requires --encoder path")
        encoder = CNNStudentEncoder(input_dim=21, history_len=int(args_cli.history_len), output_dim=5).to(env.device)
        encoder.load_state_dict(torch.load(args_cli.encoder, map_location=env.device))
        encoder.eval()
        print(f"[INFO] Loaded student encoder: {args_cli.encoder}")

    # ---- buffers ----
    history_len = int(args_cli.history_len)
    proprio_dim = 17
    action_dim = 4
    z_dim = getattr(policy_nn, "z_dim", 5)
    priv_dim = z_dim  # current env appends mlw(5): m_norm, l_norm, wind_norm(3)

    obs = env.get_observations()
    dt = float(base_env.step_dt)
    obs_history = torch.zeros((env.num_envs, history_len, proprio_dim + action_dim), device=env.device)
    last_actions = torch.zeros((env.num_envs, action_dim), device=env.device)

    trace_env = int(args_cli.trace_env)
    if trace_env < 0 or trace_env >= env.num_envs:
        raise ValueError(f"--trace_env {trace_env} out of range. num_envs={env.num_envs}")

    # ---- CSV ----
    csv_path = args_cli.csv.strip() if args_cli.csv.strip() else _default_csv_path(resume_path, args_cli.mode)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    header = [
        "time_s",
        "mode",
        # UAV / payload / goal positions
        "uav_px","uav_py","uav_pz",
        "payload_px","payload_py","payload_pz",
        "goal_px","goal_py","goal_pz",
        # payload error (also obs[0:3])
        "payload_err_x","payload_err_y","payload_err_z",
        # swing metrics (from obs)
        "theta_x_deg","theta_y_deg",
        "theta_dot_x_deg_s","theta_dot_y_deg_s",
        # true physical params (from env buffers)
        "rope_length_m","payload_mass_kg",
        # priv (obs tail)
        "priv0","priv1","priv2","priv3","priv4",
        # z teacher / z hat
        "zT0","zT1","zT2","zT3","zT4",
        "zH0","zH1","zH2","zH3","zH4",
        "z_rmse",
        "phys_pred_mass", "phys_pred_len",
        # actions
        "a0_raw","a1_raw","a2_raw","a3_raw",
        "a0_clamp","a1_clamp","a2_clamp","a3_clamp",
    ]

    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(header)

    print(f"[INFO] mode={args_cli.mode} num_envs={env.num_envs} max_steps={args_cli.max_steps} stop_on_done={args_cli.stop_on_done}")
    print(f"[INFO] CSV -> {csv_path}")

    step_count = 0
    t0 = time.time()
    # === 【在这里插入代码】 ===
    # 强制重置随机种子，消除模型加载带来的 RNG 漂移
    print(f"[INFO] Re-seeding RNG to {args_cli.seed} to ensure env alignment...")
    import random
    import numpy as np
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    # ==========================

    obs, _ = env.reset()  # <--- 这行代码非常关键，必须插在它前面！
    # 确保 e 在循环前定义，避免报错
    e = int(args_cli.trace_env)

    while simulation_app.is_running() and step_count < int(args_cli.max_steps):
        start = time.time()
        with torch.inference_mode():
            # 1. 解析观测数据
            obs_tensor = _get_obs_tensor(obs)  # (N, 22)
            obs_proprio = obs_tensor[:, :proprio_dim]
            priv = obs_tensor[:, proprio_dim:proprio_dim + priv_dim]

            # 2. 更新历史信息 (用于 Student Encoder)
            feat = torch.cat([obs_proprio, last_actions], dim=1)
            obs_history = torch.roll(obs_history, shifts=-1, dims=1)
            obs_history[:, -1, :] = feat

            # 3. 计算 Latent Z 和 物理预测
            #    Teacher: z = mu(priv), phys_pred = 0 (占位)
            #    Student: z = encoder(history), phys_pred = decoder(z)
            if args_cli.mode == "teacher":
                if hasattr(policy_nn, "mu"):
                    z_teacher = policy_nn.mu(priv)
                else:
                    z_teacher = priv
                
                # Teacher 模式下的变量赋值
                z_hat = z_teacher
                phys_pred_val = [0.0, 0.0] # 占位符
                
                # 策略配置：Teacher 使用 mu(priv)
                if hasattr(policy_nn, "use_mu"):
                    policy_nn.use_mu = True
                    
            else: # Student Mode
                # 调用 Encoder (返回 tuple)
                z_hat, phys_pred = encoder(obs_history)
                
                # 提取当前环境的物理预测值 (用于 Log)
                phys_pred_val = phys_pred[e].detach().cpu().numpy().tolist()
                
                # 同时也需要计算 z_teacher 用于对比 (RMSE)
                if hasattr(policy_nn, "mu"):
                    z_teacher = policy_nn.mu(priv)
                else:
                    z_teacher = priv

                # 策略配置：Student 直接使用 z_hat
                if hasattr(policy_nn, "use_mu"):
                    policy_nn.use_mu = False

            # 4. 构建策略输入 (raw_in)
            #    Teacher: [proprio, priv] (策略内部会自己做 mu(priv))
            #    Student: [proprio, z_hat] (我们要手动替换 tail)
            raw_in = obs_tensor.clone()
            
            if args_cli.mode == "student":
                # 关键：用 Student 预测的 z 替换掉观测中的 priv
                raw_in[:, proprio_dim:proprio_dim + priv_dim] = z_hat

            # 5. 策略推理
            obs_in = {}
            obs_in["policy"] = raw_in
            if isinstance(obs, dict) and ("critic" in obs):
                obs_in["critic"] = raw_in

            actions_raw = policy_nn.act_inference(obs_in)
            actions_clamp = actions_raw.clamp(-1.0, 1.0)

            # step env
            obs, _, dones, _ = env.step(actions_clamp)
            last_actions = actions_clamp.detach()

            # reset history for done envs
            if torch.any(dones):
                obs_history[dones] = 0.0
                last_actions[dones] = 0.0
                if hasattr(policy_nn, "reset"):
                    policy_nn.reset(dones)

            # ---- log trace env ----
            e = trace_env
            # positions from env (more trustworthy than reconstructing)
            uav_pos = base_env._robot.data.root_pos_w[e].detach().cpu().numpy().tolist()
            payload_pos = base_env._robot.data.body_pos_w[e, base_env._payload_id, :].detach().cpu().numpy().tolist()
            goal_pos = base_env._desired_pos_w[e].detach().cpu().numpy().tolist()
            payload_err = obs_tensor[e, 0:3].detach().cpu().numpy().tolist()

            theta_x = float(obs_tensor[e, 3].detach().cpu())
            theta_y = float(obs_tensor[e, 4].detach().cpu())
            theta_dx = float(obs_tensor[e, 5].detach().cpu())
            theta_dy = float(obs_tensor[e, 6].detach().cpu())

            rope_L = float(base_env._rope_lengths[e].detach().cpu())
            pay_m = float(base_env._payload_mass[e].detach().cpu())

            priv_e = priv[e].detach().cpu().numpy().tolist()
            zT = z_teacher[e].detach().cpu().numpy().tolist()
            zH = z_hat[e].detach().cpu().numpy().tolist()
            z_rmse = float(torch.sqrt(torch.mean((z_teacher[e] - z_hat[e]) ** 2)).detach().cpu())

            a_raw = actions_raw[e].detach().cpu().numpy().tolist()
            a_clp = actions_clamp[e].detach().cpu().numpy().tolist()

            w.writerow([
                step_count * dt,
                args_cli.mode,
                *uav_pos,
                *payload_pos,
                *goal_pos,
                *payload_err,
                theta_x, theta_y,
                theta_dx, theta_dy,
                rope_L, pay_m,
                *priv_e,
                *zT,
                *zH,
                z_rmse,
                phys_pred_val[0], phys_pred_val[1],
                *a_raw,
                *a_clp,
            ])

            # stop if traced env ended (for single-episode comparability)
            if bool(args_cli.stop_on_done) and bool(dones[e].item()):
                print(f"[INFO] trace_env done at step={step_count}. stop_on_done=True -> exit.")
                break

        step_count += 1

        # realtime option
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    f.close()
    env.close()
    print(f"[DONE] steps={step_count} wall={time.time()-t0:.2f}s -> {csv_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
