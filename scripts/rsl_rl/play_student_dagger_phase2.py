# play_student_dagger_phase2.py
# Evaluate a DAgger-trained Phase-2 student encoder.

from __future__ import annotations

import argparse
import sys
import os
import csv
import time

import torch
import torch.nn as nn
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play Phase-2 DAgger student encoder.")
parser.add_argument("--mode", type=str, default="student", choices=["teacher", "student"])
parser.add_argument("--encoder", type=str, default="")
parser.add_argument("--history_len", type=int, default=-1)
parser.add_argument("--trace_env", type=int, default=0)
parser.add_argument("--max_steps", type=int, default=3500)
parser.add_argument("--csv", type=str, default="")
parser.add_argument("--prefill_history", action="store_true", default=True)
parser.add_argument("--no_prefill_history", dest="prefill_history", action="store_false")
parser.add_argument("--prefill_steps", type=int, default=-1)
parser.add_argument("--stop_on_done", action="store_true", default=True)
parser.add_argument("--no_stop_on_done", dest="stop_on_done", action="store_false")

parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks  # noqa: F401
import uav_payload_lab.tasks  # noqa: F401


def _get_obs_tensor(obs_td):
    if isinstance(obs_td, torch.Tensor):
        return obs_td
    if hasattr(obs_td, "get") and obs_td.get("policy") is not None:
        return obs_td["policy"]
    if isinstance(obs_td, dict) and "policy" in obs_td:
        return obs_td["policy"]
    raise RuntimeError(f"Unsupported obs type: {type(obs_td)}")


def _safe_load_model_only(runner: OnPolicyRunner, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=runner.device)
    if "model_state_dict" in ckpt:
        runner.alg.policy.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        runner.alg.policy.load_state_dict(ckpt["model"], strict=False)
    elif "state_dict" in ckpt:
        runner.alg.policy.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        runner.alg.policy.load_state_dict(ckpt, strict=False)
    if "actor_obs_normalizer_state_dict" in ckpt and hasattr(runner.alg.policy, "actor_obs_normalizer"):
        runner.alg.policy.actor_obs_normalizer.load_state_dict(ckpt["actor_obs_normalizer_state_dict"])
    if "critic_obs_normalizer_state_dict" in ckpt and hasattr(runner.alg.policy, "critic_obs_normalizer"):
        runner.alg.policy.critic_obs_normalizer.load_state_dict(ckpt["critic_obs_normalizer_state_dict"])


def _default_csv_path(resume_path: str, mode: str) -> str:
    run_dir = os.path.dirname(resume_path)
    return os.path.join(run_dir, "phase2_teacher_dagger.csv" if mode == "teacher" else "phase2_student_dagger.csv")


class RMAStateHistoryEncoder(nn.Module):
    def __init__(self, input_dim: int, history_len: int, output_dim: int):
        super().__init__()
        self.frame_encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.LeakyReLU())
        if history_len == 50:
            self.temporal = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=8, stride=4), nn.LeakyReLU(),
                nn.Conv1d(32, 32, kernel_size=5, stride=1), nn.LeakyReLU(),
                nn.Conv1d(32, 32, kernel_size=5, stride=1), nn.LeakyReLU(), nn.Flatten())
            flat_dim = 32 * 3
        else:
            self.temporal = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(),
                nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(),
                nn.AdaptiveAvgPool1d(3), nn.Flatten())
            flat_dim = 32 * 3
        self.head = nn.Linear(flat_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, d = x.shape
        y = self.frame_encoder(x.reshape(b * h, d)).reshape(b, h, 32)
        return self.head(self.temporal(y.permute(0, 2, 1)))


class CNNStudentEncoder(nn.Module):
    def __init__(self, input_dim=21, history_len=50, output_dim=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, 5, 1, 2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(64), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(64 * history_len, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, output_dim))

    def forward(self, x):
        return self.mlp(self.cnn(x.permute(0, 2, 1)))


def _make_encoder(kind: str, input_dim: int, history_len: int, z_dim: int):
    return RMAStateHistoryEncoder(input_dim, history_len, z_dim) if kind == "rma" else CNNStudentEncoder(input_dim, history_len, z_dim)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.seed is not None:
        agent_cfg.seed = int(args_cli.seed)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(agent_cfg.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    if hasattr(base_env, "episode_length_buf"):
        base_env.episode_length_buf[:] = 0

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading checkpoint (model-only): {resume_path}")
    _safe_load_model_only(runner, resume_path)
    policy_nn = runner.alg.policy
    policy_nn.eval()

    if args_cli.mode == "student":
        if not args_cli.encoder:
            raise ValueError("--mode student requires --encoder")
        ckpt = torch.load(args_cli.encoder, map_location=env.device)
        state_dict = ckpt["state_dict"]
        encoder_type = ckpt.get("encoder_type", "rma")
        history_len = int(args_cli.history_len) if args_cli.history_len > 0 else int(ckpt["history_len"])
        proprio_dim = int(ckpt["proprio_dim"])
        priv_dim = int(ckpt["priv_dim"])
        z_dim = int(ckpt["z_dim"])
        encoder = _make_encoder(encoder_type, proprio_dim, history_len, z_dim).to(env.device)
        encoder.load_state_dict(state_dict)
        encoder.eval()
        print(f"[INFO] Loaded encoder: {args_cli.encoder}")
    else:
        obs0 = _get_obs_tensor(env.get_observations())
        z_dim = int(getattr(policy_nn, "z_dim", 5))
        proprio_dim = int(getattr(policy_nn, "proprio_dim", obs0.shape[1] - z_dim))
        priv_dim = z_dim
        history_len = int(args_cli.history_len) if args_cli.history_len > 0 else 50
        encoder = None

    obs = env.get_observations()
    dt = float(base_env.step_dt)
    obs_history = torch.zeros((env.num_envs, history_len, proprio_dim), device=env.device)
    valid_hist_len = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    trace_env = int(args_cli.trace_env)
    csv_path = args_cli.csv.strip() if args_cli.csv.strip() else _default_csv_path(resume_path, args_cli.mode)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    prefill_steps = int(args_cli.prefill_steps) if int(args_cli.prefill_steps) > 0 else history_len

    print(f"[Dims] mode={args_cli.mode} proprio_dim={proprio_dim} priv_dim={priv_dim} z_dim={z_dim} history_len={history_len} prefill={args_cli.prefill_history}")
    print(f"[CSV] {csv_path}")

    header = ["time_s", "mode", "phase", "uav_px","uav_py","uav_pz", "payload_px","payload_py","payload_pz", "goal_px","goal_py","goal_pz", "payload_err_x","payload_err_y","payload_err_z", "theta_x_deg","theta_y_deg", "theta_dot_x_deg_s","theta_dot_y_deg_s", "rope_length_m","payload_mass_kg", *[f"priv{i}" for i in range(z_dim)], *[f"zT{i}" for i in range(z_dim)], *[f"zH{i}" for i in range(z_dim)], "z_rmse", "a0_raw","a1_raw","a2_raw","a3_raw", "a0_clamp","a1_clamp","a2_clamp","a3_clamp"]

    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(header)

    step_count, eval_step = 0, 0
    t0 = time.time()
    while simulation_app.is_running() and step_count < int(args_cli.max_steps):
        start = time.time()
        with torch.inference_mode():
            obs_tensor = _get_obs_tensor(obs)
            current_proprio = obs_tensor[:, :proprio_dim]
            priv = obs_tensor[:, proprio_dim:proprio_dim + priv_dim]

            obs_history = torch.roll(obs_history, shifts=-1, dims=1)
            obs_history[:, -1, :] = current_proprio
            valid_hist_len += 1
            valid_mask = valid_hist_len >= history_len
            z_teacher = policy_nn.mu(priv).detach() if hasattr(policy_nn, "mu") else priv.detach()

            phase = "eval"
            if args_cli.mode == "teacher":
                if hasattr(policy_nn, "use_mu"):
                    policy_nn.use_mu = True
                z_hat = z_teacher
                policy_in = torch.cat([current_proprio, priv], dim=1)
            else:
                use_prefill = bool(args_cli.prefill_history) and (step_count < prefill_steps)
                if use_prefill or not bool(valid_mask[trace_env].item()):
                    phase = "prefill"
                    if hasattr(policy_nn, "use_mu"):
                        policy_nn.use_mu = True
                    z_hat = z_teacher
                    policy_in = torch.cat([current_proprio, priv], dim=1)
                else:
                    if hasattr(policy_nn, "use_mu"):
                        policy_nn.use_mu = False
                    z_hat = encoder(obs_history).detach()
                    policy_in = torch.cat([current_proprio, z_hat], dim=1)

            actions_raw = policy_nn.act_inference({"policy": policy_in})
            actions = actions_raw.clamp(-1.0, 1.0)

            should_log = args_cli.mode == "teacher" or phase == "eval"
            if should_log:
                e = trace_env
                uav_pos = base_env._robot.data.root_pos_w[e].detach().cpu().numpy().tolist()
                payload_pos = base_env._robot.data.body_pos_w[e, base_env._payload_id, :].detach().cpu().numpy().tolist()
                goal_pos = base_env._desired_pos_w[e].detach().cpu().numpy().tolist()
                payload_err = obs_tensor[e, 0:3].detach().cpu().numpy().tolist()
                theta_x = float(obs_tensor[e, 3].detach().cpu())
                theta_y = float(obs_tensor[e, 4].detach().cpu())
                theta_dx = float(obs_tensor[e, 5].detach().cpu())
                theta_dy = float(obs_tensor[e, 6].detach().cpu())
                rope_L = float(base_env._rope_lengths[e].detach().cpu()) if hasattr(base_env, "_rope_lengths") else float("nan")
                pay_m = float(base_env._payload_mass[e].detach().cpu()) if hasattr(base_env, "_payload_mass") else float("nan")
                z_rmse = float(torch.sqrt(torch.mean((z_teacher[e] - z_hat[e]) ** 2)).detach().cpu())
                writer.writerow([eval_step * dt, args_cli.mode, phase, *uav_pos, *payload_pos, *goal_pos, *payload_err, theta_x, theta_y, theta_dx, theta_dy, rope_L, pay_m, *priv[e].detach().cpu().numpy().tolist(), *z_teacher[e].detach().cpu().numpy().tolist(), *z_hat[e].detach().cpu().numpy().tolist(), z_rmse, *actions_raw[e].detach().cpu().numpy().tolist(), *actions[e].detach().cpu().numpy().tolist()])
                eval_step += 1

            obs, _, dones, _ = env.step(actions)

            # RslRl/IsaacLab wrappers sometimes return dones as 0/1 tensor.
            # Always convert to a real boolean mask before tensor indexing.
            done_mask = dones.to(dtype=torch.bool).reshape(-1)

            if done_mask.numel() != env.num_envs:
                raise RuntimeError(
                    f"Bad dones shape: got {tuple(dones.shape)}, "
                    f"after reshape={tuple(done_mask.shape)}, expected num_envs={env.num_envs}"
                )

            if torch.any(done_mask):
                obs_history[done_mask] = 0.0
                valid_hist_len[done_mask] = 0
                if hasattr(policy_nn, "reset"):
                    policy_nn.reset(done_mask)

            if bool(args_cli.stop_on_done) and bool(done_mask[trace_env].item()):
                print(f"[INFO] trace_env done at step={step_count}; stop.")
                break

        step_count += 1
        if args_cli.real_time:
            time.sleep(max(0.0, dt - (time.time() - start)))

    f.close()
    env.close()
    print(f"[DONE] steps={step_count} eval_steps={eval_step} wall={time.time()-t0:.1f}s -> {csv_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
