# train_student_dagger_phase2.py
# Phase-2 DAgger-style training for RMA student encoder.
# Keeps Phase-1 teacher/policy checkpoint unchanged; trains only phi(history)->z_teacher.

from __future__ import annotations

import argparse
import sys
import os
import time
import csv
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="DAgger-style Phase-2 training: student history encoder -> teacher latent.")

parser.add_argument("--history_len", type=int, default=50)
parser.add_argument("--num_iterations", type=int, default=200)
parser.add_argument("--steps_per_iter", type=int, default=600)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--num_learning_epochs", type=int, default=2)
parser.add_argument("--num_mini_batches", type=int, default=8)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--teacher_prob_start", type=float, default=1.0)
parser.add_argument("--teacher_prob_end", type=float, default=0.0)
parser.add_argument("--teacher_prob_decay_iters", type=int, default=100)
parser.add_argument("--loss_mode", type=str, default="weighted", choices=["mse", "weighted"])
parser.add_argument("--encoder_type", type=str, default="rma", choices=["rma", "cnn"])
parser.add_argument("--out_dir", type=str, default="")
parser.add_argument("--save_name", type=str, default="best_student_encoder_z.pth")
parser.add_argument("--trace_env", type=int, default=0)
parser.add_argument("--proprio_dim", type=int, default=-1)
parser.add_argument("--z_dim", type=int, default=-1)

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


def _infer_dims(policy_nn, obs_tensor: torch.Tensor, args) -> Tuple[int, int, int]:
    z_dim = int(args.z_dim) if int(args.z_dim) > 0 else int(getattr(policy_nn, "z_dim", 5))
    if int(args.proprio_dim) > 0:
        proprio_dim = int(args.proprio_dim)
    elif hasattr(policy_nn, "proprio_dim"):
        proprio_dim = int(getattr(policy_nn, "proprio_dim"))
    elif hasattr(policy_nn, "proprio_obs_dim"):
        proprio_dim = int(getattr(policy_nn, "proprio_obs_dim"))
    else:
        proprio_dim = int(obs_tensor.shape[1] - z_dim)
    priv_dim = z_dim
    if obs_tensor.shape[1] < proprio_dim + priv_dim:
        raise RuntimeError(
            f"Bad dims: obs_dim={obs_tensor.shape[1]}, proprio_dim={proprio_dim}, priv_dim={priv_dim}. "
            "Use --proprio_dim/--z_dim override if needed."
        )
    return proprio_dim, priv_dim, z_dim


def _linear_teacher_prob(it: int, start: float, end: float, decay_iters: int) -> float:
    if decay_iters <= 0:
        return float(end)
    alpha = min(1.0, max(0.0, it / float(decay_iters)))
    return float(start + alpha * (end - start))


class RMAStateHistoryEncoder(nn.Module):
    """RMA-style history encoder: per-frame Linear -> temporal Conv1d -> raw z."""

    def __init__(self, input_dim: int, history_len: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.output_dim = int(output_dim)
        self.frame_encoder = nn.Sequential(nn.Linear(self.input_dim, 32), nn.LeakyReLU())
        if self.history_len == 50:
            self.temporal = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=8, stride=4), nn.LeakyReLU(),
                nn.Conv1d(32, 32, kernel_size=5, stride=1), nn.LeakyReLU(),
                nn.Conv1d(32, 32, kernel_size=5, stride=1), nn.LeakyReLU(),
                nn.Flatten(),
            )
            flat_dim = 32 * 3
        else:
            self.temporal = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(),
                nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(),
                nn.AdaptiveAvgPool1d(3), nn.Flatten(),
            )
            flat_dim = 32 * 3
        self.head = nn.Linear(flat_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, d = x.shape
        if h != self.history_len or d != self.input_dim:
            raise RuntimeError(f"Bad input shape {tuple(x.shape)}, expected (B,{self.history_len},{self.input_dim})")
        y = self.frame_encoder(x.reshape(b * h, d)).reshape(b, h, 32)
        return self.head(self.temporal(y.permute(0, 2, 1)))


class CNNStudentEncoder(nn.Module):
    def __init__(self, input_dim=21, history_len=50, output_dim=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, 5, 1, 2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(64 * history_len, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.mlp(self.cnn(x.permute(0, 2, 1)))


def _make_encoder(kind: str, input_dim: int, history_len: int, z_dim: int) -> nn.Module:
    if kind == "rma":
        return RMAStateHistoryEncoder(input_dim, history_len, z_dim)
    if kind == "cnn":
        return CNNStudentEncoder(input_dim, history_len, z_dim)
    raise ValueError(kind)


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
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    out_dir = args_cli.out_dir.strip() or os.path.join(log_dir, "student_dagger_phase2")
    os.makedirs(out_dir, exist_ok=True)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading checkpoint (model-only): {resume_path}")
    _safe_load_model_only(runner, resume_path)
    policy_nn = runner.alg.policy
    policy_nn.eval()
    for p in policy_nn.parameters():
        p.requires_grad_(False)

    obs = env.get_observations()
    obs_tensor = _get_obs_tensor(obs)
    proprio_dim, priv_dim, z_dim = _infer_dims(policy_nn, obs_tensor, args_cli)

    history_len = int(args_cli.history_len)
    encoder = _make_encoder(args_cli.encoder_type, proprio_dim, history_len, z_dim).to(env.device)
    optimizer = optim.Adam(encoder.parameters(), lr=float(args_cli.lr))

    obs_history = torch.zeros((env.num_envs, history_len, proprio_dim), device=env.device)
    valid_hist_len = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    trace_env = int(args_cli.trace_env)
    print(f"[Dims] obs_dim={obs_tensor.shape[1]} proprio_dim={proprio_dim} priv_dim={priv_dim} z_dim={z_dim} history_len={history_len} encoder={args_cli.encoder_type}")
    print(f"[DAgger] num_envs={env.num_envs}, iterations={args_cli.num_iterations}, steps_per_iter={args_cli.steps_per_iter}")
    print(f"[OUT] {out_dir}")

    metrics_path = os.path.join(out_dir, "dagger_train_metrics.csv")
    best_loss = float("inf")
    dt = float(base_env.step_dt)
    t0 = time.time()

    with open(metrics_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["iter", "teacher_prob", "num_samples", "train_loss", *[f"rmse_dim{i}" for i in range(z_dim)], "trace_z_rmse", "trace_valid_hist_len"])

        for it in range(int(args_cli.num_iterations)):
            encoder.eval()
            rollout_x, rollout_y = [], []
            sum_sq = torch.zeros(z_dim, device=env.device)
            count = 0
            trace_z_rmse = float("nan")
            teacher_prob = _linear_teacher_prob(it, float(args_cli.teacher_prob_start), float(args_cli.teacher_prob_end), int(args_cli.teacher_prob_decay_iters))

            obs = env.get_observations()

            # recreate buffers each iteration to avoid PyTorch inference tensor in-place update error
            obs_history = torch.zeros(
                (env.num_envs, history_len, proprio_dim),
                device=env.device,
                dtype=obs_tensor.dtype,
            )
            valid_hist_len = torch.zeros(
                env.num_envs,
                dtype=torch.long,
                device=env.device,
            )

            if hasattr(policy_nn, "reset"):
                policy_nn.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))

            for _step in range(int(args_cli.steps_per_iter)):
                with torch.inference_mode():
                    obs_tensor = _get_obs_tensor(obs)
                    current_proprio = obs_tensor[:, :proprio_dim]
                    priv = obs_tensor[:, proprio_dim:proprio_dim + priv_dim]

                    obs_history = torch.roll(obs_history, shifts=-1, dims=1)
                    obs_history[:, -1, :] = current_proprio
                    valid_hist_len += 1
                    valid_mask = valid_hist_len >= history_len

                    z_teacher = policy_nn.mu(priv).detach() if hasattr(policy_nn, "mu") else priv.detach()
                    z_student = encoder(obs_history).detach()
                    z_used = torch.where(valid_mask[:, None], z_student, z_teacher)

                    # teacher action
                    if hasattr(policy_nn, "use_mu"):
                        policy_nn.use_mu = True
                    teacher_in = torch.cat([current_proprio, priv], dim=1)
                    a_teacher = policy_nn.act_inference({"policy": teacher_in}).clamp(-1.0, 1.0)

                    # student action
                    if hasattr(policy_nn, "use_mu"):
                        policy_nn.use_mu = False
                    student_in = torch.cat([current_proprio, z_used], dim=1)
                    a_student = policy_nn.act_inference({"policy": student_in}).clamp(-1.0, 1.0)

                    use_teacher = (torch.rand(env.num_envs, 1, device=env.device) < teacher_prob) | (~valid_mask[:, None])
                    actions = torch.where(use_teacher, a_teacher, a_student)

                    if torch.any(valid_mask):
                        x_valid = obs_history[valid_mask].detach()
                        y_valid = z_teacher[valid_mask].detach()
                        rollout_x.append(x_valid.cpu().to(torch.float32))
                        rollout_y.append(y_valid.cpu().to(torch.float32))
                        err = z_student[valid_mask] - z_teacher[valid_mask]
                        sum_sq += (err * err).sum(dim=0)
                        count += err.shape[0]

                    if 0 <= trace_env < env.num_envs and bool(valid_mask[trace_env].item()):
                        trace_z_rmse = float(torch.sqrt(torch.mean((z_student[trace_env] - z_teacher[trace_env]) ** 2)).detach().cpu())

                    obs, _, dones, _ = env.step(actions)

                    # IsaacLab/RSL-RL wrapper may return dones as 0/1 tensor.
                    # Always convert to a boolean mask before tensor indexing.
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

                if args_cli.real_time:
                    time.sleep(max(0.0, dt))

            if len(rollout_x) == 0:
                print(f"[Iter {it:04d}] no valid samples; skip")
                continue

            x = torch.cat(rollout_x, dim=0)
            y = torch.cat(rollout_y, dim=0)
            n_samples = x.shape[0]
            z_std = y.std(dim=0).clamp_min(1e-6).to(env.device)
            weights = (1.0 / (z_std * z_std)).detach()

            encoder.train()
            perm = torch.randperm(n_samples)
            total_loss, total_updates = 0.0, 0
            chunks = torch.chunk(perm, max(1, int(args_cli.num_mini_batches)))
            for _epoch in range(int(args_cli.num_learning_epochs)):
                for mb in chunks:
                    bx = x[mb].to(env.device, non_blocking=True)
                    by = y[mb].to(env.device, non_blocking=True)
                    pred = encoder(bx)
                    loss = torch.mean(((pred - by) ** 2) * weights) if args_cli.loss_mode == "weighted" else torch.mean((pred - by) ** 2)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += float(loss.detach().cpu())
                    total_updates += 1

            train_loss = total_loss / max(1, total_updates)
            rmse_dim = torch.sqrt(sum_sq / max(1, count)).detach().cpu()
            writer.writerow([it, teacher_prob, n_samples, train_loss, *[float(v) for v in rmse_dim], trace_z_rmse, int(valid_hist_len[trace_env].detach().cpu()) if 0 <= trace_env < env.num_envs else -1])
            fcsv.flush()
            print(f"[Iter {it:04d}] teacher_prob={teacher_prob:.3f} samples={n_samples} loss={train_loss:.6e} rmse_dim={rmse_dim.numpy().round(3)} trace_z_rmse={trace_z_rmse:.3f}")

            if train_loss < best_loss:
                best_loss = train_loss
                save_path = os.path.join(out_dir, args_cli.save_name)
                torch.save({
                    "state_dict": encoder.state_dict(),
                    "encoder_type": args_cli.encoder_type,
                    "history_len": history_len,
                    "input_dim": proprio_dim,
                    "proprio_dim": proprio_dim,
                    "priv_dim": priv_dim,
                    "z_dim": z_dim,
                    "loss_mode": args_cli.loss_mode,
                    "checkpoint": resume_path,
                    "best_loss": best_loss,
                }, save_path)
                print(f"[Save] {save_path} best_loss={best_loss:.6e}")

    env.close()
    print(f"[DONE] wall={time.time() - t0:.1f}s metrics={metrics_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
