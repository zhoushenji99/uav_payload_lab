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
from pathlib import Path

import numpy as np
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
parser.add_argument(
    "--student_shadow_warmup_sec",
    type=float,
    default=0.0,
    help=(
        "Student-mode only. Let the Teacher execute actions for this duration while "
        "the Student runs in shadow and fills its history/context caches; 0 keeps "
        "the legacy immediate-Student behavior."
    ),
)
parser.add_argument("--slow_warmup_sec", type=float, default=3.0, help="Run the slow encoder at policy rate during startup.")
parser.add_argument("--slow_update_hz", type=float, default=1.0, help="Slow encoder rate after startup.")
parser.add_argument("--fast_update_hz", type=float, default=60.0, help="Fast encoder update rate.")
parser.add_argument(
    "--slow_filter_tau_sec",
    type=float,
    default=0.25,
    help="Post-startup causal slow-context filter time constant; 0 disables filtering.",
)
parser.add_argument(
    "--context_runtime_mode",
    type=str,
    default="fast_slow",
    choices=["fast_slow", "all_60hz"],
    help="Proposed fast/slow schedule or a future all-branches-at-policy-rate ablation.",
)
parser.add_argument(
    "--rma_context_mode",
    type=str,
    default=None,
    choices=["split_hard", "split_soft", "monolithic"],
    help="Teacher checkpoint architecture. Defaults to the task configuration.",
)
parser.add_argument("--profile_inference", action="store_true", default=True)
parser.add_argument("--no_profile_inference", dest="profile_inference", action="store_false")
parser.add_argument("--gust_event_threshold", type=float, default=0.05)
parser.add_argument("--fast_response_threshold", type=float, default=0.05)
parser.add_argument("--gust_response_window_sec", type=float, default=0.5)
parser.add_argument("--trace_env", type=int, default=0, help="Which env index to log to CSV.")
parser.add_argument("--stop_on_done", action="store_true", default=True, help="Stop when trace_env episode ends (default: True).")
parser.add_argument("--no_stop_on_done", dest="stop_on_done", action="store_false", help="Do not stop on done; keep running until max_steps.")
parser.add_argument("--max_steps", type=int, default=2000, help="Max env steps to run (avoid crossing episode boundary).")
parser.add_argument("--csv", type=str, default="", help="CSV output path. If empty, write into checkpoint folder.")
parser.add_argument(
    "--eval_payload_mass_kg",
    type=float,
    default=None,
    help="Evaluation-only fixed payload mass; keeps the training normalization range unchanged.",
)
parser.add_argument(
    "--eval_rope_length_m",
    type=float,
    default=None,
    help="Evaluation-only fixed rope length; keeps the training normalization range unchanged.",
)
parser.add_argument(
    "--eval_disable_wind",
    action="store_true",
    default=False,
    help="Disable mean, gust, and OU wind for this evaluation rollout.",
)
parser.add_argument(
    "--eval_wind_scale",
    type=float,
    default=1.0,
    help=(
        "Evaluation-only multiplier applied to the physical wind after the "
        "training-range clamp. Values above 1 are physical OOD while the "
        "Teacher normalization denominator remains unchanged."
    ),
)
parser.add_argument(
    "--eval_wind_mode",
    type=str,
    default="training",
    choices=["training", "sinusoid"],
    help="Use the original stochastic training wind or a deterministic evaluation sinusoid.",
)
parser.add_argument("--eval_wind_amplitude_mps2", type=float, default=1.0)
parser.add_argument("--eval_wind_frequency_hz", type=float, default=1.0)
parser.add_argument("--eval_wind_start_sec", type=float, default=3.0)
parser.add_argument("--eval_wind_axis", type=str, default="x", choices=["x", "y"])
parser.add_argument("--eval_wind_phase_rad", type=float, default=0.0)

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
from uav_payload_lab.tasks.direct.uav_payload_sim2real.fastslow_runtime import (
    causal_ema_alpha,
    compute_action_band_energy,
    compute_action_total_variation,
    compute_gust_response_latency,
    compute_multirate_schedule,
    summarize_latency_ms,
    validate_evaluation_overrides,
)
from uav_payload_lab.tasks.direct.uav_payload_sim2real.phase2_shadow_handover import (
    select_shadow_actions,
    validate_shadow_warmup,
)


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


def _build_rma_runner_cfg(agent_cfg, env_cfg):
    runner_cfg = agent_cfg.to_dict()
    fields = (
        "proprio_obs_dim",
        "privileged_obs_dim",
        "rma_z_dim",
        "rma_z_exp_dim",
        "rma_use_mu",
        "rma_context_mode",
        "rma_use_physics_anchor",
        "rma_mu_hidden_dims",
        "rma_activation",
    )
    if all(hasattr(env_cfg, field) for field in fields):
        runner_cfg["policy"].update(
            {
                "proprio_obs_dim": int(env_cfg.proprio_obs_dim),
                "privileged_obs_dim": int(env_cfg.privileged_obs_dim),
                "z_dim": int(env_cfg.rma_z_dim),
                "z_exp_dim": int(env_cfg.rma_z_exp_dim),
                "use_mu": bool(env_cfg.rma_use_mu),
                "context_mode": str(env_cfg.rma_context_mode),
                "use_physics_anchor": bool(env_cfg.rma_use_physics_anchor),
                "mu_hidden_dims": list(env_cfg.rma_mu_hidden_dims),
                "mu_activation": env_cfg.rma_activation,
            }
        )
    return runner_cfg


def _default_csv_path(resume_path: str, mode: str) -> str:
    run_dir = os.path.dirname(resume_path)
    name = "phase2_teacher.csv" if mode == "teacher" else "phase2_student.csv"
    return os.path.join(run_dir, name)


def _slow_update_mask(episode_steps, warmup_steps, period_steps):
    warmup_mask = episode_steps < warmup_steps
    periodic_mask = (episode_steps >= warmup_steps) & (
        (episode_steps - warmup_steps) % period_steps == 0
    )
    return warmup_mask | periodic_mask


def _periodic_update_mask(episode_steps, period_steps):
    return (episode_steps % int(period_steps)) == 0


def _synchronize(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _timed_call(function, device, enabled):
    if not enabled:
        return function(), 0.0
    _synchronize(device)
    start = time.perf_counter()
    result = function()
    _synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0


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


class MonolithicStudentEncoder(nn.Module):
    def __init__(self, input_dim=21, history_len=50, z_dim=5):
        super().__init__()
        self.encoder = CNNContextEncoder(input_dim, history_len, z_dim)

    def forward(self, x):
        return self.encoder(x)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    # ---- cfg wiring (mirror official play.py + collect_z_dataset.py) ----
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.rma_context_mode is not None:
        env_cfg.rma_context_mode = str(args_cli.rma_context_mode)
    if str(getattr(env_cfg, "rma_context_mode", "split_hard")) != "split_soft":
        env_cfg.rma_use_physics_anchor = False
        env_cfg.rma_phys_anchor_coef = 0.0

    if args_cli.seed is not None:
        agent_cfg.seed = int(args_cli.seed)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(agent_cfg.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    evaluation_overrides = validate_evaluation_overrides(
        payload_mass_kg=args_cli.eval_payload_mass_kg,
        rope_length_m=args_cli.eval_rope_length_m,
        disable_wind=args_cli.eval_disable_wind,
        wind_scale=args_cli.eval_wind_scale,
        wind_mode=args_cli.eval_wind_mode,
        wind_amplitude_mps2=args_cli.eval_wind_amplitude_mps2,
        wind_frequency_hz=args_cli.eval_wind_frequency_hz,
        wind_start_sec=args_cli.eval_wind_start_sec,
        wind_axis=args_cli.eval_wind_axis,
        wind_phase_rad=args_cli.eval_wind_phase_rad,
        payload_mass_range=tuple(env_cfg.payload_mass_range),
        rope_length_range=tuple(env_cfg.rope_length_range),
    )
    env_cfg.eval_fixed_payload_mass_kg = evaluation_overrides["payload_mass_kg"]
    env_cfg.eval_fixed_rope_length_m = evaluation_overrides["rope_length_m"]
    env_cfg.eval_disable_wind = evaluation_overrides["disable_wind"]
    env_cfg.eval_wind_scale = evaluation_overrides["wind_scale"]
    env_cfg.eval_wind_mode = evaluation_overrides["wind_mode"]
    env_cfg.eval_wind_amplitude_mps2 = evaluation_overrides["wind_amplitude_mps2"]
    env_cfg.eval_wind_frequency_hz = evaluation_overrides["wind_frequency_hz"]
    env_cfg.eval_wind_start_sec = evaluation_overrides["wind_start_sec"]
    env_cfg.eval_wind_axis = evaluation_overrides["wind_axis"]
    env_cfg.eval_wind_phase_rad = evaluation_overrides["wind_phase_rad"]
    print(f"[INFO] evaluation_overrides={evaluation_overrides}")

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
    runner_cfg = _build_rma_runner_cfg(agent_cfg, env_cfg)
    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
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
    student_context_mode = "split"
    student_teacher_context_mode = None

    if args_cli.mode == "student":
        if not args_cli.encoder:
            raise ValueError("--mode student requires --encoder path")

        enc_ckpt = torch.load(args_cli.encoder, map_location=env.device)

        if not isinstance(enc_ckpt, dict) or enc_ckpt.get("model_type") not in {
            "fast_slow_context",
            "monolithic_context",
        }:
            raise RuntimeError(
                "Student checkpoint must be fast_slow_context or monolithic_context."
            )
        if "state_dict" not in enc_ckpt:
            raise RuntimeError("Student checkpoint is missing state_dict.")
        student_context_mode = (
            "split"
            if enc_ckpt.get("model_type") == "fast_slow_context"
            else "monolithic"
        )
        student_teacher_context_mode = enc_ckpt.get("teacher_context_mode")
        policy_context_mode = str(getattr(policy_nn, "context_mode", "unknown"))
        if policy_context_mode == "split_hard" and str(
            student_teacher_context_mode
        ) != "split_hard":
            raise RuntimeError(
                "Hard-explicit Teacher requires a Student checkpoint with explicit "
                "teacher_context_mode='split_hard' lineage. Retrain the Student from "
                "the new audited hard-explicit dataset."
            )
        if student_teacher_context_mode not in (None, "unknown") and str(
            student_teacher_context_mode
        ) != policy_context_mode:
            raise RuntimeError(
                "Student/Teacher architecture lineage mismatch: "
                f"student was trained from {student_teacher_context_mode!r}, "
                f"loaded Teacher policy is {policy_context_mode!r}."
            )
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

        if student_context_mode == "split":
            encoder = FastSlowStudentEncoder(
                input_dim=student_input_dim,
                history_len=student_history_len,
                z_slow_dim=student_z_slow_dim,
                z_fast_dim=student_z_fast_dim,
            ).to(env.device)
        else:
            encoder = MonolithicStudentEncoder(
                input_dim=student_input_dim,
                history_len=student_history_len,
                z_dim=student_z_dim,
            ).to(env.device)

        encoder.load_state_dict(enc_state_dict, strict=True)
        encoder.eval()
        print(
            f"[INFO] Loaded student encoder: {args_cli.encoder} | "
            f"context_mode={student_context_mode} "
            f"input_dim={student_input_dim} history_len={student_history_len} "
            f"z_slow_dim={student_z_slow_dim} z_fast_dim={student_z_fast_dim}"
        )

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
    z_slow_raw = torch.zeros((env.num_envs, student_z_slow_dim), device=env.device)
    z_slow_target = torch.zeros((env.num_envs, student_z_slow_dim), device=env.device)
    z_slow_cache = torch.zeros((env.num_envs, student_z_slow_dim), device=env.device)
    z_fast_cache = torch.zeros((env.num_envs, student_z_fast_dim), device=env.device)
    schedule = compute_multirate_schedule(
        history_len=history_len,
        policy_dt=dt,
        slow_warmup_sec=float(args_cli.slow_warmup_sec),
        slow_update_hz=float(args_cli.slow_update_hz),
        fast_update_hz=float(args_cli.fast_update_hz),
    )
    slow_warmup_steps = schedule.slow_warmup_steps
    slow_period_steps = schedule.slow_period_steps
    fast_period_steps = schedule.fast_period_steps
    slow_filter_alpha = causal_ema_alpha(dt, float(args_cli.slow_filter_tau_sec))
    student_shadow_warmup_steps = validate_shadow_warmup(
        shadow_warmup_sec=float(args_cli.student_shadow_warmup_sec),
        policy_dt=dt,
        history_len=history_len,
        slow_warmup_sec=float(args_cli.slow_warmup_sec),
        mode=str(args_cli.mode),
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
        "control_source",
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
        # slow context audit: raw CNN output, held filter target, Actor-visible cache
        "z_slow_raw0","z_slow_raw1",
        "z_slow_target0","z_slow_target1",
        "z_slow_cache0","z_slow_cache1",
        "slow_updated","fast_updated","full_updated","episode_step",
        "slow_batch_calls","fast_batch_calls","full_batch_calls",
        "slow_env_samples","fast_env_samples","full_env_samples",
        "slow_inference_ms","fast_inference_ms","full_inference_ms",
        "actor_inference_ms","end_to_end_inference_ms",
        "gust_x_mps2","gust_y_mps2","gust_z_mps2","gust_event",
        "wind_acc_x_mps2","wind_acc_y_mps2","wind_acc_z_mps2",
        "context_refresh_action_l1","context_refresh_action_l2","context_refresh_action_max",
        "executed_action_delta_l1",
        # Student Actor candidate is logged even while Teacher controls shadow warm-up.
        "student_candidate_a0_raw","student_candidate_a1_raw","student_candidate_a2_raw","student_candidate_a3_raw",
        "student_candidate_a0_clamp","student_candidate_a1_clamp","student_candidate_a2_clamp","student_candidate_a3_clamp",
        # actions
        "a0_raw","a1_raw","a2_raw","a3_raw",
        "a0_clamp","a1_clamp","a2_clamp","a3_clamp",
    ]

    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(header)

    print(f"[INFO] mode={args_cli.mode} num_envs={env.num_envs} max_steps={args_cli.max_steps} stop_on_done={args_cli.stop_on_done}")
    print(
        f"[INFO] policy_hz={schedule.policy_hz:.1f} fast_hz={schedule.fast_update_hz:.1f} "
        f"slow_warmup_steps={slow_warmup_steps} slow_period_steps={slow_period_steps} "
        f"slow_filter_tau={args_cli.slow_filter_tau_sec:.3f}s alpha={slow_filter_alpha:.8f} "
        f"runtime_mode={args_cli.context_runtime_mode} "
        f"student_shadow_warmup_steps={student_shadow_warmup_steps}"
    )
    print(f"[INFO] CSV -> {csv_path}")

    step_count = 0
    t0 = time.time()
    call_counts = {
        "slow_batch_calls": 0,
        "fast_batch_calls": 0,
        "full_batch_calls": 0,
        "slow_env_samples": 0,
        "fast_env_samples": 0,
        "full_env_samples": 0,
    }
    latency_samples = {"slow": [], "fast": [], "full": [], "actor": [], "end_to_end": []}
    trace_actions = []
    trace_gust = []
    trace_z_fast = []
    context_refresh_action_l1 = []
    context_refresh_action_l2 = []
    context_refresh_action_max = []
    previous_trace_action = None
    previous_trace_gust = None

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

            slow_update_mask = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
            fast_update_mask = torch.zeros_like(slow_update_mask)
            full_update_mask = torch.zeros_like(slow_update_mask)
            slow_inference_ms = 0.0
            fast_inference_ms = 0.0
            full_inference_ms = 0.0
            context_delta_l1 = 0.0
            context_delta_l2 = 0.0
            context_delta_max = 0.0
            old_slow_cache = z_slow_cache.clone()

            if args_cli.profile_inference:
                _synchronize(env.device)
                end_to_end_start = time.perf_counter()

            if args_cli.mode == "teacher":
                # teacher: policy sees e(t) and internally uses z = mu(e)
                if hasattr(policy_nn, "use_mu"):
                    policy_nn.use_mu = True
                z_hat = z_teacher  # for logging only
                z_slow_raw[:] = z_teacher[:, :student_z_slow_dim]
                z_slow_target[:] = z_teacher[:, :student_z_slow_dim]
                z_slow_cache[:] = z_teacher[:, :student_z_slow_dim]
                z_fast_cache[:] = z_teacher[:, student_z_slow_dim:]
                policy_in = obs_tensor  # tail stays as priv=e
            else:
                if hasattr(policy_nn, "use_mu"):
                    policy_nn.use_mu = False

                if student_context_mode == "monolithic":
                    full_update_mask[:] = True
                    z_all, full_inference_ms = _timed_call(
                        lambda: encoder(obs_history).detach(),
                        env.device,
                        bool(args_cli.profile_inference),
                    )
                    z_slow_raw[:] = z_all[:, :student_z_slow_dim]
                    z_slow_target[:] = z_slow_raw
                    z_slow_cache[:] = z_slow_raw
                    z_fast_cache[:] = z_all[:, student_z_slow_dim:]
                    call_counts["full_batch_calls"] += 1
                    call_counts["full_env_samples"] += int(env.num_envs)
                    latency_samples["full"].append(full_inference_ms)
                else:
                    fast_update_mask = _periodic_update_mask(
                        episode_steps, fast_period_steps
                    )
                    if torch.any(fast_update_mask):
                        z_fast_new, fast_inference_ms = _timed_call(
                            lambda: encoder.encode_fast(
                                obs_history[fast_update_mask]
                            ).detach(),
                            env.device,
                            bool(args_cli.profile_inference),
                        )
                        z_fast_cache[fast_update_mask] = z_fast_new
                        call_counts["fast_batch_calls"] += 1
                        call_counts["fast_env_samples"] += int(
                            fast_update_mask.sum().item()
                        )
                        latency_samples["fast"].append(fast_inference_ms)

                    if args_cli.context_runtime_mode == "all_60hz":
                        slow_update_mask[:] = True
                    else:
                        slow_update_mask = _slow_update_mask(
                            episode_steps,
                            slow_warmup_steps,
                            slow_period_steps,
                        )
                    if torch.any(slow_update_mask):
                        z_slow_new, slow_inference_ms = _timed_call(
                            lambda: encoder.encode_slow(
                                obs_history[slow_update_mask]
                            ).detach(),
                            env.device,
                            bool(args_cli.profile_inference),
                        )
                        z_slow_raw[slow_update_mask] = z_slow_new
                        z_slow_target[slow_update_mask] = z_slow_new
                        call_counts["slow_batch_calls"] += 1
                        call_counts["slow_env_samples"] += int(
                            slow_update_mask.sum().item()
                        )
                        latency_samples["slow"].append(slow_inference_ms)

                    if args_cli.context_runtime_mode == "all_60hz":
                        z_slow_cache[:] = z_slow_target
                    else:
                        startup_mask = episode_steps < slow_warmup_steps
                        z_slow_cache[startup_mask] = z_slow_target[startup_mask]
                        post_startup_mask = ~startup_mask
                        z_slow_cache[post_startup_mask] += slow_filter_alpha * (
                            z_slow_target[post_startup_mask]
                            - z_slow_cache[post_startup_mask]
                        )

                z_hat = torch.cat([z_slow_cache, z_fast_cache], dim=-1)
                policy_in = torch.cat([obs_proprio, z_hat], dim=1)  # (N, 21+z_dim)

            # ---- 3) build obs dict for policy ----
            obs_in = {"policy": policy_in}
            if isinstance(obs, dict) and ("critic" in obs):
                obs_in["critic"] = policy_in

            # ---- 4) action inference (t) ----
            if not hasattr(policy_nn, "act_inference"):
                raise RuntimeError("policy_nn has no act_inference(); unexpected policy type.")
            student_actions_raw, actor_inference_ms = _timed_call(
                lambda: policy_nn.act_inference(obs_in),
                env.device,
                bool(args_cli.profile_inference),
            )
            student_actions = _clip_actions_to_bounds(
                student_actions_raw, action_low, action_high
            )
            latency_samples["actor"].append(actor_inference_ms)

            if args_cli.profile_inference:
                _synchronize(env.device)
                end_to_end_inference_ms = (
                    time.perf_counter() - end_to_end_start
                ) * 1000.0
            else:
                end_to_end_inference_ms = 0.0
            latency_samples["end_to_end"].append(end_to_end_inference_ms)

            # During the opt-in shadow interval, the Student encoder, caches, and
            # Actor still run normally, but the privileged Teacher action is the
            # one executed by the environment. At handover no runtime state is
            # reset: only this per-environment action selector changes source.
            actions_raw = student_actions_raw
            actions = student_actions
            teacher_shadow_active = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
            if args_cli.mode == "student" and student_shadow_warmup_steps > 0:
                if not hasattr(policy_nn, "use_mu"):
                    raise RuntimeError(
                        "Teacher shadow warm-up requires policy_nn.use_mu support."
                    )
                policy_nn.use_mu = True
                try:
                    teacher_actions_raw = policy_nn.act_inference(
                        {"policy": obs_tensor}
                    )
                finally:
                    policy_nn.use_mu = False
                teacher_actions = _clip_actions_to_bounds(
                    teacher_actions_raw, action_low, action_high
                )
                actions_raw, actions, teacher_shadow_active = select_shadow_actions(
                    student_raw=student_actions_raw,
                    student_clipped=student_actions,
                    teacher_raw=teacher_actions_raw,
                    teacher_clipped=teacher_actions,
                    episode_steps=episode_steps,
                    shadow_steps=student_shadow_warmup_steps,
                )

            # Counterfactual refresh diagnostic: under the same observation and
            # current fast context, compare old cache vs newly refreshed raw target.
            # These two extra Actor calls are deliberately outside profiled latency.
            e = trace_env
            if (
                args_cli.mode == "student"
                and student_context_mode == "split"
                and bool(slow_update_mask[e].item())
            ):
                z_old_e = torch.cat(
                    [old_slow_cache[e : e + 1], z_fast_cache[e : e + 1]], dim=-1
                )
                z_raw_e = torch.cat(
                    [z_slow_target[e : e + 1], z_fast_cache[e : e + 1]], dim=-1
                )
                old_policy_in = torch.cat(
                    [obs_proprio[e : e + 1], z_old_e], dim=-1
                )
                raw_policy_in = torch.cat(
                    [obs_proprio[e : e + 1], z_raw_e], dim=-1
                )
                old_action = _clip_actions_to_bounds(
                    policy_nn.act_inference({"policy": old_policy_in}),
                    action_low,
                    action_high,
                )
                raw_action = _clip_actions_to_bounds(
                    policy_nn.act_inference({"policy": raw_policy_in}),
                    action_low,
                    action_high,
                )
                context_action_delta = torch.abs(raw_action - old_action)[0]
                context_delta_l1 = float(context_action_delta.sum().cpu())
                context_delta_l2 = float(
                    torch.linalg.vector_norm(context_action_delta).cpu()
                )
                context_delta_max = float(context_action_delta.max().cpu())
                if int(episode_steps[e].item()) >= slow_warmup_steps:
                    context_refresh_action_l1.append(context_delta_l1)
                    context_refresh_action_l2.append(context_delta_l2)
                    context_refresh_action_max.append(context_delta_max)

            # ---- 5) LOG at time t (BEFORE env.step) ----
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
            student_a_raw = (
                student_actions_raw[e].detach().cpu().numpy().tolist()
            )
            student_a_clp = student_actions[e].detach().cpu().numpy().tolist()
            if args_cli.mode == "teacher":
                control_source = "teacher"
            elif bool(teacher_shadow_active[e].item()):
                control_source = "teacher_shadow"
            else:
                control_source = "student"
            slow_raw_e = z_slow_raw[e].detach().cpu().numpy().tolist()
            slow_target_e = z_slow_target[e].detach().cpu().numpy().tolist()
            slow_cache_e = z_slow_cache[e].detach().cpu().numpy().tolist()

            if hasattr(base_env, "_wind_gust"):
                gust_e = (
                    base_env._wind_gust[e].detach().cpu().numpy().astype(float)
                )
            else:
                gust_e = np.zeros(3, dtype=float)
            if hasattr(base_env, "_wind_acc_w"):
                wind_acc_e = (
                    base_env._wind_acc_w[e].detach().cpu().numpy().astype(float)
                )
            else:
                wind_acc_e = np.zeros(3, dtype=float)
            gust_event = (
                previous_trace_gust is not None
                and float(np.linalg.norm(gust_e - previous_trace_gust))
                >= float(args_cli.gust_event_threshold)
            )
            action_e = np.asarray(a_clp, dtype=float)
            executed_action_delta_l1 = (
                float(np.sum(np.abs(action_e - previous_trace_action)))
                if previous_trace_action is not None
                else 0.0
            )
            trace_actions.append(action_e.copy())
            trace_gust.append(gust_e.copy())
            trace_z_fast.append(np.asarray(zH[student_z_slow_dim:], dtype=float))
            previous_trace_action = action_e.copy()
            previous_trace_gust = gust_e.copy()

            w.writerow([
                step_count * dt,
                args_cli.mode,
                control_source,
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
                *slow_raw_e,
                *slow_target_e,
                *slow_cache_e,
                int(slow_update_mask[e].item()),
                int(fast_update_mask[e].item()),
                int(full_update_mask[e].item()),
                int(episode_steps[e].item()),
                int(call_counts["slow_batch_calls"]),
                int(call_counts["fast_batch_calls"]),
                int(call_counts["full_batch_calls"]),
                int(call_counts["slow_env_samples"]),
                int(call_counts["fast_env_samples"]),
                int(call_counts["full_env_samples"]),
                slow_inference_ms,
                fast_inference_ms,
                full_inference_ms,
                actor_inference_ms,
                end_to_end_inference_ms,
                *gust_e.tolist(),
                int(gust_event),
                *wind_acc_e.tolist(),
                context_delta_l1,
                context_delta_l2,
                context_delta_max,
                executed_action_delta_l1,
                *student_a_raw,
                *student_a_clp,
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
                z_slow_raw[done_mask] = 0.0
                z_slow_target[done_mask] = 0.0
                z_slow_cache[done_mask] = 0.0
                z_fast_cache[done_mask] = 0.0
                if hasattr(policy_nn, "reset"):
                    policy_nn.reset(done_mask)

            step_count += 1
            # stop exactly at episode boundary of trace_env
            if bool(args_cli.stop_on_done) and bool(dones[e].item()):
                print(
                    f"[INFO] trace_env done after {step_count} logged steps. "
                    "stop_on_done=True -> exit."
                )
                break

        # realtime option
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ----------------------------
    # Cleanup
    # ----------------------------
    f.close()
    num_envs = int(env.num_envs)
    env.close()

    wall_time_sec = float(time.time() - t0)
    print(f"[DONE] steps={step_count} wall={wall_time_sec:.2f}s -> {csv_path}")

    actions_array = np.asarray(trace_actions, dtype=float).reshape(-1, action_dim)
    gust_array = np.asarray(trace_gust, dtype=float).reshape(-1, 3)
    z_fast_array = np.asarray(trace_z_fast, dtype=float).reshape(
        -1, student_z_fast_dim
    )
    action_tv = compute_action_total_variation(actions_array)
    action_band_energy = compute_action_band_energy(
        actions_array,
        sample_rate_hz=schedule.policy_hz,
        low_hz=5.0,
        high_hz=min(30.0, 0.5 * schedule.policy_hz),
    )
    gust_response = compute_gust_response_latency(
        gust_array,
        z_fast_array,
        policy_dt=dt,
        gust_event_threshold=float(args_cli.gust_event_threshold),
        fast_response_threshold=float(args_cli.fast_response_threshold),
        response_window_sec=float(args_cli.gust_response_window_sec),
    )

    def _summarize_refresh(values):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return {"count": 0, "mean": None, "p95": None, "p99": None, "max": None}
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }

    summary = {
        "mode": args_cli.mode,
        "teacher_context_mode": str(getattr(policy_nn, "context_mode", "unknown")),
        "student_teacher_context_mode": student_teacher_context_mode,
        "student_context_mode": student_context_mode if args_cli.mode == "student" else None,
        "context_runtime_mode": str(args_cli.context_runtime_mode),
        "checkpoint": resume_path,
        "encoder": args_cli.encoder if args_cli.mode == "student" else "",
        "csv_path": csv_path,
        "steps": int(step_count),
        "wall_time_sec": wall_time_sec,
        "dt": float(dt),
        "num_envs": num_envs,
        "seed": int(args_cli.seed) if args_cli.seed is not None else None,
        "max_steps": int(args_cli.max_steps),
        "stop_on_done": bool(args_cli.stop_on_done),
        "policy_hz": float(schedule.policy_hz),
        "fast_update_hz": float(schedule.fast_update_hz),
        "fast_period_steps": int(fast_period_steps),
        "slow_warmup_sec": float(args_cli.slow_warmup_sec),
        "slow_update_hz": float(args_cli.slow_update_hz),
        "slow_warmup_steps": int(slow_warmup_steps),
        "slow_period_steps": int(slow_period_steps),
        "student_shadow_warmup_sec": float(args_cli.student_shadow_warmup_sec),
        "student_shadow_warmup_steps": int(student_shadow_warmup_steps),
        "slow_filter_tau_sec": float(args_cli.slow_filter_tau_sec),
        "slow_filter_alpha": float(slow_filter_alpha),
        "profile_inference": bool(args_cli.profile_inference),
        "evaluation_overrides": {
            **evaluation_overrides,
            "payload_mass_training_range_kg": [
                float(env_cfg.payload_mass_range[0]),
                float(env_cfg.payload_mass_range[1]),
            ],
            "rope_length_training_range_m": [
                float(env_cfg.rope_length_range[0]),
                float(env_cfg.rope_length_range[1]),
            ],
            "wind_enabled_effective": bool(
                getattr(base_env, "_wind_enabled", False)
            ),
        },
        "context_call_counts": call_counts,
        "post_startup_slow_call_reduction_fraction": float(
            1.0 - 1.0 / slow_period_steps
        ),
        "observed_overall_slow_batch_call_reduction_fraction": (
            float(1.0 - call_counts["slow_batch_calls"] / step_count)
            if step_count > 0 and args_cli.mode == "student"
            and student_context_mode == "split"
            else None
        ),
        "observed_overall_slow_env_sample_reduction_fraction": (
            float(
                1.0
                - call_counts["slow_env_samples"]
                / max(1, step_count * num_envs)
            )
            if step_count > 0
            and args_cli.mode == "student"
            and student_context_mode == "split"
            else None
        ),
        "inference_latency": {
            name: summarize_latency_ms(values)
            for name, values in latency_samples.items()
        },
        "ctbr_action_total_variation": action_tv,
        "ctbr_action_5_30hz_energy": action_band_energy,
        "gust_to_fast_context_response": gust_response,
        "context_refresh_counterfactual_action_change": {
            "definition": (
                "same-observation Actor output difference between old slow cache "
                "and newly refreshed unfiltered slow target"
            ),
            "l1": _summarize_refresh(context_refresh_action_l1),
            "l2": _summarize_refresh(context_refresh_action_l2),
            "max_abs_channel": _summarize_refresh(context_refresh_action_max),
        },
    }

    summary_name = "phase2_teacher_play_summary.json" if args_cli.mode == "teacher" else "phase2_student_play_summary.json"
    summary_path = os.path.join(os.path.dirname(csv_path), summary_name)
    with open(summary_path, "w") as fsum:
        json.dump(summary, fsum, indent=2)
    print(f"[Summary] saved {summary_path}")



if __name__ == "__main__":
    main()
    simulation_app.close()
