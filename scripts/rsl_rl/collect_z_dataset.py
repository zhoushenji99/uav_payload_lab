# collect_z_dataset.py
# Collect dataset for Phase-2 student: inputs = history(proprio+last_action), labels = z_teacher (mu(priv))
# PLUS: write env0 trace CSV + meta statistics for paper-quality analysis.

import argparse
import sys
import os
import time
import csv
import math
import json
import torch
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect Phase-2 dataset: history -> z_teacher + trace + meta stats.")
parser.add_argument("--steps", type=int, default=1000, help="Steps to collect (per-env steps).")
parser.add_argument("--history_len", type=int, default=50)
parser.add_argument("--save_every", type=int, default=25, help="Save a shard every N steps.")
parser.add_argument("--out_name", type=str, default="student_z_dataset", help="Output folder name under log_dir.")

# extra logging
parser.add_argument("--trace_csv", action="store_true", help="Write env0 trace csv for plotting.")
parser.add_argument("--trace_env", type=int, default=0, help="Which env index to trace into csv.")
parser.add_argument("--trace_max_steps", type=int, default=0, help="If >0, limit trace rows (debug).")

# standard IsaacLab play args
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--real-time", action="store_true", default=False)

parser.add_argument("--probe_sec", type=float, default=2.0,
                    help="Seconds of probing excitation at episode start (0 disables).")
parser.add_argument("--probe_amp", type=float, default=0.35,
                    help="Amplitude of probing lateral command in action space.")
parser.add_argument("--probe_freq", type=float, default=1.0,
                    help="Hz of probing sine.")
parser.add_argument("--sample_stride", type=int, default=5,
                    help="Store one sample every K env steps after warmup.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# SAFE imports after SimulationApp
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks  # noqa: F401
import uav_payload_lab.tasks  # noqa: F401


def _safe_load_model_only(runner: OnPolicyRunner, ckpt_path: str):
    """Load ONLY model (and normalizers if exist). Avoid optimizer mismatch."""
    ckpt = torch.load(ckpt_path, map_location=runner.device)
    runner.alg.policy.load_state_dict(ckpt["model_state_dict"], strict=False)
    if "actor_obs_normalizer_state_dict" in ckpt and hasattr(runner.alg.policy, "actor_obs_normalizer"):
        runner.alg.policy.actor_obs_normalizer.load_state_dict(ckpt["actor_obs_normalizer_state_dict"])
    if "critic_obs_normalizer_state_dict" in ckpt and hasattr(runner.alg.policy, "critic_obs_normalizer"):
        runner.alg.policy.critic_obs_normalizer.load_state_dict(ckpt["critic_obs_normalizer_state_dict"])


def _get_obs_tensor(obs_td):
    if isinstance(obs_td, torch.Tensor):
        return obs_td
    # tensordict / dict
    if hasattr(obs_td, "get") and obs_td.get("policy") is not None:
        return obs_td["policy"]
    if isinstance(obs_td, dict) and "policy" in obs_td:
        return obs_td["policy"]
    raise RuntimeError(f"Unsupported obs type: {type(obs_td)}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    print(f"[INFO] Loading checkpoint (model-only): {resume_path}")
    _safe_load_model_only(runner, resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy  # RMAActorCritic

    # teacher collection must use_mu=True so z_teacher = mu(priv)
    if hasattr(policy_nn, "use_mu"):
        policy_nn.use_mu = True
    history_len = int(args_cli.history_len)
    input_dim = 21            # 当前 teacher 真正看到的 proprio 维度，已经包含 prev_actions(4)
    z_dim = getattr(policy_nn, "z_dim", 5)
    priv_dim = z_dim          # 当前 teacher 的 privileged tail = 5

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    probe_steps = int(args_cli.probe_sec / dt)

    # history buffer: (N, H, 21)
    obs_history = torch.zeros((env.num_envs, history_len, input_dim), device=env.device)

    out_dir = os.path.join(log_dir, args_cli.out_name)
    os.makedirs(out_dir, exist_ok=True)
    collect_t0 = time.time()
    # shard buffers
    shard_inputs = []
    shard_labels = []
    shard_labels_ml = []
    shard_idx = 0
    step_count = 0
    warmup = history_len
    stored_steps = 0
    # meta stats accumulators (CPU)
    z_sum = torch.zeros(z_dim)
    z_sumsq = torch.zeros(z_dim)
    z_min = torch.full((z_dim,), float("inf"))
    z_max = torch.full((z_dim,), -float("inf"))

    priv_sum = torch.zeros(priv_dim)
    priv_sumsq = torch.zeros(priv_dim)
    priv_min = torch.full((priv_dim,), float("inf"))
    priv_max = torch.full((priv_dim,), -float("inf"))

    # trace csv
    trace_rows = []
    trace_env = int(args_cli.trace_env)
    trace_path = os.path.join(out_dir, "trace_env0.csv")  # name kept stable for paper scripts

    print(f"[Collect] num_envs={env.num_envs}, steps={args_cli.steps}, history_len={history_len}, z_dim={z_dim}")
    print(f"[Collect] saving shards to: {out_dir}")

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # ---- 0) compute teacher action from current obs(t) ----
            actions_raw = policy(obs)                          # (N,4)
            actions_clamp = actions_raw.clamp(-1.0, 1.0)       # (N,4)

            # ---- 1) OPTIONAL probe: override action for first probe_steps ----
            actions_to_step = actions_clamp

            probe_start = warmup
            probe_end = warmup + probe_steps   # probe_steps 仍然由 probe_sec/dt 决定

            if probe_steps > 0 and (probe_start <= step_count < probe_end):
                t = (step_count - probe_start) * float(dt)
                s = math.sin(2.0 * math.pi * float(args_cli.probe_freq) * t)

                actions_to_step = actions_clamp.clone()
                actions_to_step[:, 1] = float(args_cli.probe_amp) * s
                actions_to_step[:, 2] = -float(args_cli.probe_amp) * s
                actions_to_step[:, 3] = 0.0
                actions_to_step = actions_to_step.clamp(-1.0, 1.0)


            # ---- 2) build history input from obs(t) and last_action(t-1) ----
            obs_tensor = _get_obs_tensor(obs)      # (N,26)
            feat = obs_tensor[:, :input_dim]       # (N,21)

            obs_history = torch.roll(obs_history, shifts=-1, dims=1)
            obs_history[:, -1, :] = feat

            # ---- 3) teacher label z_teacher: use mu(priv) (robust; no dependency on last_z timing) ----
            priv = obs_tensor[:, 21:26]             # (N,5)
            if hasattr(policy_nn, "mu"):
                z_teacher = policy_nn.mu(priv).detach()        # (N,5)
            else:
                # fallback (shouldn't happen in your setup)
                z_teacher = priv.detach()

            # ---- 4) store sample after warmup ----
            if step_count >= warmup and ((step_count - warmup) % args_cli.sample_stride == 0):
                shard_inputs.append(obs_history.detach().clone().cpu().to(torch.float16))
                shard_labels.append(z_teacher.detach().clone().cpu().to(torch.float32))
                shard_labels_ml.append(priv[:, :2].detach().clone().cpu().to(torch.float32))
                stored_steps += 1

                # stats (CPU): accumulate per stored sample
                z_cpu = z_teacher.detach().cpu()
                z_sum += z_cpu.sum(dim=0)
                z_sumsq += (z_cpu * z_cpu).sum(dim=0)
                z_min = torch.minimum(z_min, z_cpu.min(dim=0).values)
                z_max = torch.maximum(z_max, z_cpu.max(dim=0).values)

                p_cpu = priv.detach().cpu()
                priv_sum += p_cpu.sum(dim=0)
                priv_sumsq += (p_cpu * p_cpu).sum(dim=0)
                priv_min = torch.minimum(priv_min, p_cpu.min(dim=0).values)
                priv_max = torch.maximum(priv_max, p_cpu.max(dim=0).values)

                # trace
                if args_cli.trace_csv and trace_env < env.num_envs:
                    e = trace_env
                    payload_err = obs_tensor[e, 0:3].detach().cpu().numpy()
                    theta_deg = obs_tensor[e, 3:5].detach().cpu().numpy()
                    theta_dot_deg_s = obs_tensor[e, 5:7].detach().cpu().numpy()
                    pos_err = float(torch.norm(obs_tensor[e, 0:3]).cpu())

                    trace_rows.append([
                        (step_count - warmup) * float(dt),
                        *priv[e].detach().cpu().numpy().tolist(),
                        *z_teacher[e].detach().cpu().numpy().tolist(),
                        pos_err,
                        theta_deg[0], theta_deg[1],
                        theta_dot_deg_s[0], theta_dot_deg_s[1],
                        *actions_raw[e].detach().cpu().numpy().tolist(),
                        *actions_to_step[e].detach().cpu().numpy().tolist(),
                    ])

            # ---- 5) step env ONCE using actions_to_step ----
            obs, _, dones, _ = env.step(actions_to_step)

            # ---- 6) update last_actions & reset buffers for done envs ----
            done_mask = dones.to(dtype=torch.bool).reshape(-1)
            if torch.any(done_mask):
                obs_history[done_mask] = 0.0
                policy_nn.reset(done_mask)

            step_count += 1

            # ---- 7) save shards & stop ----
            if (step_count >= warmup) and ((step_count - warmup + 1) % args_cli.save_every == 0):
                if len(shard_inputs) > 0:
                    inputs = torch.stack(shard_inputs, dim=0).reshape(-1, history_len, input_dim)
                    labels = torch.stack(shard_labels, dim=0).reshape(-1, z_dim)
                    labels_ml = torch.stack(shard_labels_ml, dim=0).reshape(-1, 2)

                    shard_path = os.path.join(out_dir, f"shard_{shard_idx:04d}.pt")
                    torch.save(
                        {
                            "inputs": inputs,
                            "labels": labels,
                            "labels_ml": labels_ml,
                        },
                        shard_path,
                    )
                    print(
                        f"[Collect] saved {shard_path} | "
                        f"inputs={inputs.shape} labels={labels.shape} labels_ml={labels_ml.shape}"
                    )

                    shard_inputs.clear()
                    shard_labels.clear()
                    shard_labels_ml.clear()
                    shard_idx += 1

            # stop condition
            if step_count >= (args_cli.steps + warmup):
                if len(shard_inputs) > 0:
                    inputs = torch.stack(shard_inputs, dim=0).reshape(-1, history_len, input_dim)
                    labels = torch.stack(shard_labels, dim=0).reshape(-1, z_dim)
                    labels_ml = torch.stack(shard_labels_ml, dim=0).reshape(-1, 2)

                    shard_path = os.path.join(out_dir, f"shard_{shard_idx:04d}.pt")
                    torch.save(
                        {
                            "inputs": inputs,
                            "labels": labels,
                            "labels_ml": labels_ml,
                        },
                        shard_path,
                    )
                    print(
                        f"[Collect] saved {shard_path} | "
                        f"inputs={inputs.shape} labels={labels.shape} labels_ml={labels_ml.shape}"
                    )

                # write meta + stat
                total_samples = stored_steps * env.num_envs
                z_mean = z_sum / max(1, total_samples)
                z_var = z_sumsq / max(1, total_samples) - z_mean * z_mean
                z_std = torch.sqrt(torch.clamp(z_var, min=1e-12))

                p_mean = priv_sum / max(1, total_samples)
                p_var = priv_sumsq / max(1, total_samples) - p_mean * p_mean
                p_std = torch.sqrt(torch.clamp(p_var, min=1e-12))

                meta = {
                    "checkpoint": resume_path,
                    "num_envs": env.num_envs,
                    "history_len": history_len,
                    "input_dim": input_dim,
                    "sample_stride": args_cli.sample_stride,
                    "priv_dim": priv_dim,
                    "z_dim": z_dim,
                    "steps": args_cli.steps,
                    "save_every": args_cli.save_every,
                    "obs_layout": {
                        "policy_obs(21)": "err(3),theta(2),theta_dot(2),quat(4),lin_vel(3),ang_vel(3),prev_actions(4)",
                        "priv(5)": "m_norm(1), l_norm(1), wind_norm(3)",
                        "student_input": "history of policy_obs(21)",
                    },
                    "aux_target": "labels_ml = priv[:, :2] = [m_norm, l_norm]",
                    "z_stats": {
                        "mean": z_mean.tolist(),
                        "std": z_std.tolist(),
                        "min": z_min.tolist(),
                        "max": z_max.tolist(),
                    },
                    "priv_stats": {
                        "mean": p_mean.tolist(),
                        "std": p_std.tolist(),
                        "min": priv_min.tolist(),
                        "max": priv_max.tolist(),
                    },
                }
                torch.save(meta, os.path.join(out_dir, "meta.pt"))
                wall_time_sec = time.time() - collect_t0
                num_shards = shard_idx + (1 if len(shard_inputs) > 0 else 0)
                total_samples = stored_steps * env.num_envs

                collect_report = {
                    "checkpoint": resume_path,
                    "out_dir": out_dir,
                    "wall_time_sec": wall_time_sec,
                    "num_envs": int(env.num_envs),
                    "steps_requested": int(args_cli.steps),
                    "warmup_steps": int(warmup),
                    "env_steps_total": int(step_count),
                    "stored_steps": int(stored_steps),
                    "total_samples": int(total_samples),
                    "num_shards": int(num_shards),
                    "samples_per_sec": float(total_samples / max(wall_time_sec, 1e-9)),
                    "history_len": int(history_len),
                    "input_dim": int(input_dim),
                    "z_dim": int(z_dim),
                    "sample_stride": int(args_cli.sample_stride),
                    "save_every": int(args_cli.save_every),
                    "probe_sec": float(args_cli.probe_sec),
                    "probe_amp": float(args_cli.probe_amp),
                    "probe_freq": float(args_cli.probe_freq),
                    "trace_csv": bool(args_cli.trace_csv),
                }

                collect_report_path = os.path.join(out_dir, "collect_report.json")
                with open(collect_report_path, "w") as f:
                    json.dump(collect_report, f, indent=2)
                print(f"[Collect] saved collect_report.json -> {collect_report_path}")
                print("[Collect] saved meta.pt")

                if args_cli.trace_csv and len(trace_rows) > 0:
                    with open(trace_path, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow([
                            "time",
                            "priv0","priv1","priv2","priv3","priv4",
                            "zT0","zT1","zT2","zT3","zT4",
                            "pos_err",
                            "theta_x_deg","theta_y_deg",
                            "theta_dot_x_deg_s","theta_dot_y_deg_s",
                            "a0_raw","a1_raw","a2_raw","a3_raw",
                            "a0_clamp","a1_clamp","a2_clamp","a3_clamp",
                        ])
                        w.writerows(trace_rows)
                    print(f"[Collect] saved trace csv: {trace_path} rows={len(trace_rows)}")

                print("[Collect] DONE.")
                break

        # realtime if needed
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()