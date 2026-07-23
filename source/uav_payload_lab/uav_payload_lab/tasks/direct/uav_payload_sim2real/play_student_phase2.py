# play_student_phase2.py
# Phase-2 closed-loop rollout for Teacher vs Student (RMA):
# - Teacher: z = mu(priv)  (oracle context)
# - Student: z_exp_hat is cached at low rate; z_imp_hat is refreshed at policy rate.
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
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[6]
_RSL_RL_DIR = _REPO_ROOT / "scripts" / "rsl_rl"
if str(_RSL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_RSL_RL_DIR))

import cli_args  # isort: skip


# ----------------------------
# args
# ----------------------------
parser = argparse.ArgumentParser(description="Phase-2 Play: Teacher(mu(priv)) vs Student(encoder(history)) with CSV logging.")

# phase-2 specific
parser.add_argument("--mode", type=str, default="student", choices=["student", "teacher"])
parser.add_argument("--encoder", type=str, default="", help="Student encoder .pth (required if --mode student).")
parser.add_argument("--history_len", type=int, default=50)
parser.add_argument("--slow_warmup_sec", type=float, default=1.0, help="Run the slow encoder at policy rate during startup.")
parser.add_argument("--slow_update_hz", type=float, default=1.0, help="Slow encoder rate after startup.")
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


def _get_action_bounds(base_env, device):
    """Return execution action bounds for the current env interface."""
    if getattr(base_env.cfg, "action_interface", "") == "px4_ctbr" or hasattr(base_env, "_ctbr_body_rate_limit"):
        if hasattr(base_env, "_ctbr_body_rate_limit"):
            rate_limit = base_env._ctbr_body_rate_limit.to(device=device, dtype=torch.float32)
        else:
            rate_limit = torch.tensor(base_env.cfg.ctbr_body_rate_limit, device=device, dtype=torch.float32)
        low = torch.cat((torch.tensor([-1.0], device=device, dtype=torch.float32), -rate_limit))
        high = torch.cat((torch.tensor([0.0], device=device, dtype=torch.float32), rate_limit))
        return low, high
    low = torch.full((4,), -1.0, device=device, dtype=torch.float32)
    high = torch.full((4,), 1.0, device=device, dtype=torch.float32)
    return low, high


def _clip_actions_to_bounds(actions: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return torch.minimum(torch.maximum(actions, low), high)


def _safe_load_model_only(runner: OnPolicyRunner, ckpt_path: str):
    """Load ONLY model (and normalizers if exist). Avoid optimizer mismatch."""
    ckpt = torch.load(ckpt_path, map_location=runner.device)

    # common keys in rsl_rl / isaaclab
    if "model_state_dict" in ckpt:
        runner.alg.policy.load_state_dict(ckpt["model_state_dict"], strict=True)
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        runner.alg.policy.load_state_dict(ckpt["model"], strict=True)
    elif "state_dict" in ckpt:
        runner.alg.policy.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        # fall back: assume whole ckpt is state_dict
        runner.alg.policy.load_state_dict(ckpt, strict=True)

    # optional normalizers
    if "actor_obs_normalizer_state_dict" in ckpt and hasattr(runner.alg.policy, "actor_obs_normalizer"):
        runner.alg.policy.actor_obs_normalizer.load_state_dict(ckpt["actor_obs_normalizer_state_dict"])
    if "critic_obs_normalizer_state_dict" in ckpt and hasattr(runner.alg.policy, "critic_obs_normalizer"):
        runner.alg.policy.critic_obs_normalizer.load_state_dict(ckpt["critic_obs_normalizer_state_dict"])


def _default_csv_path(resume_path: str, mode: str) -> str:
    run_dir = os.path.dirname(resume_path)
    name = "phase2_teacher.csv" if mode == "teacher" else "phase2_student.csv"
    return os.path.join(run_dir, name)


def _compute_slow_schedule(history_len, policy_dt, slow_warmup_sec, slow_update_hz):
    if history_len <= 0 or policy_dt <= 0.0 or slow_update_hz <= 0.0 or slow_warmup_sec < 0.0:
        raise ValueError("Invalid fast/slow schedule parameters.")
    warmup_steps = max(history_len, int(math.ceil(slow_warmup_sec / policy_dt)))
    period_steps = max(1, int(round(1.0 / (slow_update_hz * policy_dt))))
    return warmup_steps, period_steps


def _slow_update_mask(episode_steps, warmup_steps, period_steps):
    warmup_mask = episode_steps < warmup_steps
    periodic_mask = (episode_steps >= warmup_steps) & (
        (episode_steps - warmup_steps) % period_steps == 0
    )
    return warmup_mask | periodic_mask


# ----------------------------
# student encoder (must match train_student_z.py)
# ----------------------------
class CNNContextEncoder(nn.Module):
    def __init__(self, input_dim=21, history_len=50, output_dim=2):
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

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,H,21)->(B,21,H)
        return self.mlp(self.cnn(x))


class FastSlowStudentEncoder(nn.Module):
    def __init__(self, input_dim=21, history_len=50, z_slow_dim=2, z_fast_dim=3):
        super().__init__()
        self.slow_encoder = CNNContextEncoder(input_dim, history_len, z_slow_dim)
        self.fast_encoder = CNNContextEncoder(input_dim, history_len, z_fast_dim)

    def encode_slow(self, x):
        return self.slow_encoder(x)

    def encode_fast(self, x):
        return self.fast_encoder(x)

    def forward(self, x):
        return self.encode_slow(x), self.encode_fast(x)


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

    # ---- env ----
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    action_low, action_high = _get_action_bounds(base_env, env.device)
    print(f"[INFO] action_low={action_low.detach().cpu().tolist()} action_high={action_high.detach().cpu().tolist()}")
    # if hasattr(base_env, "episode_length_buf"):
    #     base_env.episode_length_buf[:] = 0
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
    student_input_dim = 21
    student_history_len = int(args_cli.history_len)
    student_z_dim = 5
    student_z_slow_dim = 2
    student_z_fast_dim = 3

    if args_cli.mode == "student":
        if not args_cli.encoder:
            raise ValueError("--mode student requires --encoder path")

        enc_ckpt = torch.load(args_cli.encoder, map_location=env.device)

        if not isinstance(enc_ckpt, dict) or enc_ckpt.get("model_type") != "fast_slow_context":
            raise RuntimeError("Student checkpoint is not a fast/slow context checkpoint.")
        if "state_dict" not in enc_ckpt:
            raise RuntimeError("Fast/slow student checkpoint is missing state_dict.")
        enc_state_dict = enc_ckpt["state_dict"]
        student_history_len = int(enc_ckpt.get("history_len", student_history_len))
        student_input_dim = int(enc_ckpt.get("input_dim", student_input_dim))
        student_z_dim = int(enc_ckpt.get("z_dim", student_z_dim))
        student_z_slow_dim = int(enc_ckpt.get("z_slow_dim", student_z_slow_dim))
        student_z_fast_dim = int(enc_ckpt.get("z_fast_dim", student_z_fast_dim))
        if student_z_slow_dim + student_z_fast_dim != student_z_dim:
            raise RuntimeError(
                f"Bad student context dims: slow={student_z_slow_dim}, "
                f"fast={student_z_fast_dim}, total={student_z_dim}."
            )
        if (
            student_z_dim != int(policy_nn.z_dim)
            or student_z_slow_dim != int(policy_nn.z_exp_dim)
            or student_z_fast_dim != int(policy_nn.z_imp_dim)
        ):
            raise RuntimeError(
                "Student/Teacher context mismatch: "
                f"student=({student_z_slow_dim},{student_z_fast_dim}), "
                f"teacher=({policy_nn.z_exp_dim},{policy_nn.z_imp_dim})."
            )

        encoder = FastSlowStudentEncoder(
            input_dim=student_input_dim,
            history_len=student_history_len,
            z_slow_dim=student_z_slow_dim,
            z_fast_dim=student_z_fast_dim,
        ).to(env.device)

        encoder.load_state_dict(enc_state_dict, strict=True)
        encoder.eval()
        print(
            f"[INFO] Loaded student encoder: {args_cli.encoder} | "
            f"input_dim={student_input_dim} history_len={student_history_len} "
            f"z_slow_dim={student_z_slow_dim} z_fast_dim={student_z_fast_dim}"
        )

    # ---- buffers ----
    # ---- buffers ----
    history_len = student_history_len if args_cli.mode == "student" else int(args_cli.history_len)
    proprio_dim = 21
    action_dim = 4
    z_dim = student_z_dim if args_cli.mode == "student" else int(getattr(policy_nn, "z_dim", 5))
    priv_dim = int(getattr(policy_nn, "z_dim", 5))

    obs = env.get_observations()
    dt = float(base_env.step_dt)
    obs_history = torch.zeros((env.num_envs, history_len, proprio_dim), device=env.device)
    last_actions = torch.zeros((env.num_envs, action_dim), device=env.device)  # 只用于日志/重置，不参与feat
    episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    z_slow_cache = torch.zeros((env.num_envs, student_z_slow_dim), device=env.device)
    slow_warmup_steps, slow_period_steps = _compute_slow_schedule(
        history_len,
        dt,
        float(args_cli.slow_warmup_sec),
        float(args_cli.slow_update_hz),
    )

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
        "slow_updated","episode_step",
        # actions
        "a0_raw","a1_raw","a2_raw","a3_raw",
        "a0_clamp","a1_clamp","a2_clamp","a3_clamp",
    ]

    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(header)

    print(f"[INFO] mode={args_cli.mode} num_envs={env.num_envs} max_steps={args_cli.max_steps} stop_on_done={args_cli.stop_on_done}")
    print(
        f"[INFO] policy_hz={1.0 / dt:.1f} fast_hz={1.0 / dt:.1f} "
        f"slow_warmup_steps={slow_warmup_steps} slow_period_steps={slow_period_steps}"
    )
    print(f"[INFO] CSV -> {csv_path}")

    step_count = 0
    t0 = time.time()

    # ----------------------------
    # Main rollout loop (single-step, aligned logging)
    # ----------------------------
    while simulation_app.is_running() and step_count < int(args_cli.max_steps):
        start = time.time()
        with torch.inference_mode():

            # ---- 0) current obs(t) -> tensor ----
            obs_tensor = _get_obs_tensor(obs)           # (N,26)
            obs_proprio = obs_tensor[:, :proprio_dim]   # (N,21)
            priv = obs_tensor[:, proprio_dim: proprio_dim + priv_dim]   # 21:26

            # ---- 1) update history with (s_t, a_{t-1}) ----
            # feature_t = [proprio(t), last_action(t-1)]  (MUST match collect_z_dataset.py)
            feat = obs_proprio
            obs_history = torch.roll(obs_history, shifts=-1, dims=1)
            obs_history[:, -1, :] = feat

            # ---- 2) compute z_teacher (for logging/label) and z_hat (student prediction) ----
            if hasattr(policy_nn, "mu"):
                z_teacher = policy_nn.mu(priv)  # (N, z_dim)
            else:
                z_teacher = priv  # fallback (shouldn't happen if RMAActorCritic is used)

            if args_cli.mode == "teacher":
                # teacher: policy sees e(t) and internally uses z = mu(e)
                if hasattr(policy_nn, "use_mu"):
                    policy_nn.use_mu = True
                z_hat = z_teacher  # for logging only
                slow_update_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
                policy_in = obs_tensor  # tail stays as priv=e
            else:
                # Student: fast context is refreshed every policy step. Slow context
                # is refreshed at policy rate during startup, then sample-and-held.
                if hasattr(policy_nn, "use_mu"):
                    policy_nn.use_mu = False
                z_fast = encoder.encode_fast(obs_history).detach()
                slow_update_mask = _slow_update_mask(
                    episode_steps,
                    slow_warmup_steps,
                    slow_period_steps,
                )
                if torch.any(slow_update_mask):
                    z_slow_cache[slow_update_mask] = encoder.encode_slow(
                        obs_history[slow_update_mask]
                    ).detach()
                z_hat = torch.cat([z_slow_cache, z_fast], dim=-1)
                policy_in = torch.cat([obs_proprio, z_hat], dim=1)  # (N, 21+z_dim)

            # ---- 3) build obs dict for policy ----
            obs_in = {"policy": policy_in}
            if isinstance(obs, dict) and ("critic" in obs):
                obs_in["critic"] = policy_in

            # ---- 4) action inference (t) ----
            if not hasattr(policy_nn, "act_inference"):
                raise RuntimeError("policy_nn has no act_inference(); unexpected policy type.")
            actions_raw = policy_nn.act_inference(obs_in)             # (N, act_dim)
            actions = _clip_actions_to_bounds(actions_raw, action_low, action_high)

            # ---- 5) LOG at time t (BEFORE env.step) ----
            e = trace_env

            # state (t): from base_env (aligned with obs_tensor)
            uav_pos = base_env._robot.data.root_pos_w[e].detach().cpu().numpy().tolist()
            payload_pos = base_env._robot.data.body_pos_w[e, base_env._payload_id, :].detach().cpu().numpy().tolist()
            goal_pos = base_env._desired_pos_w[e].detach().cpu().numpy().tolist()

            # obs-derived (t)
            payload_err = obs_tensor[e, 0:3].detach().cpu().numpy().tolist()
            theta_x = float(obs_tensor[e, 3].detach().cpu())
            theta_y = float(obs_tensor[e, 4].detach().cpu())
            theta_dx = float(obs_tensor[e, 5].detach().cpu())
            theta_dy = float(obs_tensor[e, 6].detach().cpu())

            # env params (t)
            rope_L = float(base_env._rope_lengths[e].detach().cpu())
            pay_m = float(base_env._payload_mass[e].detach().cpu())

            # z compare (t)
            priv_e = priv[e].detach().cpu().numpy().tolist()
            zT = z_teacher[e].detach().cpu().numpy().tolist()
            zH = z_hat[e].detach().cpu().numpy().tolist()
            z_rmse = float(torch.sqrt(torch.mean((z_teacher[e] - z_hat[e]) ** 2)).detach().cpu())

            # actions (t)
            a_raw = actions_raw[e].detach().cpu().numpy().tolist()
            a_clp = actions[e].detach().cpu().numpy().tolist()

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
                int(slow_update_mask[e].item()), int(episode_steps[e].item()),
                *a_raw,
                *a_clp,
            ])

            # ---- 6) SINGLE env.step: advance to obs(t+1) ----
            obs, _, dones, _ = env.step(actions)

            last_actions = actions.detach()
            episode_steps += 1

            done_mask = dones.to(dtype=torch.bool).reshape(-1)
            if done_mask.numel() != env.num_envs:
                raise RuntimeError(
                    f"Bad dones shape: got {tuple(dones.shape)}, "
                    f"after reshape={tuple(done_mask.shape)}, expected num_envs={env.num_envs}"
                )

            if torch.any(done_mask):
                obs_history[done_mask] = 0.0
                last_actions[done_mask] = 0.0
                episode_steps[done_mask] = 0
                z_slow_cache[done_mask] = 0.0
                if hasattr(policy_nn, "reset"):
                    policy_nn.reset(done_mask)

            # stop exactly at episode boundary of trace_env
            if bool(args_cli.stop_on_done) and bool(dones[e].item()):
                print(f"[INFO] trace_env done at step={step_count}. stop_on_done=True -> exit.")
                break

        step_count += 1

        # realtime option
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ----------------------------
    # Cleanup
    # ----------------------------
    f.close()
    env.close()

    wall_time_sec = float(time.time() - t0)
    print(f"[DONE] steps={step_count} wall={wall_time_sec:.2f}s -> {csv_path}")

    summary = {
        "mode": args_cli.mode,
        "checkpoint": resume_path,
        "encoder": args_cli.encoder if args_cli.mode == "student" else "",
        "csv_path": csv_path,
        "steps": int(step_count),
        "wall_time_sec": wall_time_sec,
        "dt": float(dt),
        "num_envs": int(env.num_envs),
        "seed": int(args_cli.seed) if args_cli.seed is not None else None,
        "max_steps": int(args_cli.max_steps),
        "stop_on_done": bool(args_cli.stop_on_done),
        "policy_hz": float(1.0 / dt),
        "fast_update_hz": float(1.0 / dt),
        "slow_warmup_sec": float(args_cli.slow_warmup_sec),
        "slow_update_hz": float(args_cli.slow_update_hz),
        "slow_warmup_steps": int(slow_warmup_steps),
        "slow_period_steps": int(slow_period_steps),
    }

    summary_name = "phase2_teacher_play_summary.json" if args_cli.mode == "teacher" else "phase2_student_play_summary.json"
    summary_path = os.path.join(os.path.dirname(csv_path), summary_name)
    with open(summary_path, "w") as fsum:
        json.dump(summary, fsum, indent=2)
    print(f"[Summary] saved {summary_path}")



if __name__ == "__main__":
    main()
    simulation_app.close()
