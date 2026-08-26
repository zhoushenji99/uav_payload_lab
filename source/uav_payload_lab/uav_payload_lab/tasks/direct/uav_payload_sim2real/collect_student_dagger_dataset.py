#!/usr/bin/env python3
"""Collect Teacher-labeled histories from states physically visited by Student."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os
import sys
import time

import torch


@dataclass(frozen=True)
class DaggerStepChoice:
    executed_action: torch.Tensor
    label_z: torch.Tensor
    position_active: torch.Tensor


def choose_dagger_step(
    *,
    position_action: torch.Tensor,
    student_action: torch.Tensor,
    teacher_z: torch.Tensor,
    episode_step: torch.Tensor,
    precontrol_steps: int,
) -> DaggerStepChoice:
    """Give control to Position before the boundary and Student afterwards."""
    if position_action.shape != student_action.shape or student_action.ndim != 2:
        raise ValueError("position and student actions must share a 2-D shape")
    if episode_step.shape != (student_action.shape[0],):
        raise ValueError("episode_step must contain one value per environment")
    if teacher_z.ndim != 2 or teacher_z.shape[0] != student_action.shape[0]:
        raise ValueError("teacher_z must contain one label per environment")
    precontrol_steps = int(precontrol_steps)
    if precontrol_steps < 0:
        raise ValueError("precontrol_steps must be non-negative")
    position_active = episode_step < precontrol_steps
    executed = torch.where(
        position_active.unsqueeze(-1), position_action, student_action
    )
    return DaggerStepChoice(executed, teacher_z, position_active)


def _build_parser(app_launcher_cls, cli_args_module):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--dagger-round", type=int, required=True, choices=[1, 2])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--history_len", type=int, default=50)
    parser.add_argument("--sample_stride", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--precontrol", choices=["position"], default="position")
    parser.add_argument("--precontrol_sec", type=float, default=3.0)
    parser.add_argument("--slow_warmup_sec", type=float, default=3.0)
    parser.add_argument("--slow_update_hz", type=float, default=1.0)
    parser.add_argument("--fast_update_hz", type=float, default=60.0)
    parser.add_argument("--slow_filter_tau_sec", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--seed", type=int, required=True)
    cli_args_module.add_rsl_rl_args(parser)
    app_launcher_cls.add_app_launcher_args(parser)
    return parser


def main() -> None:
    repo_root = Path(__file__).resolve().parents[6]
    rsl_rl_dir = repo_root / "scripts" / "rsl_rl"
    if str(rsl_rl_dir) not in sys.path:
        sys.path.insert(0, str(rsl_rl_dir))

    from isaaclab.app import AppLauncher
    import cli_args

    parser = _build_parser(AppLauncher, cli_args)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config
    import isaaclab_tasks  # noqa: F401
    import uav_payload_lab.tasks  # noqa: F401

    from uav_payload_lab.tasks.direct.uav_payload_sim2real.audit_z_dataset import (
        audit_shard_tensors,
    )
    from uav_payload_lab.tasks.direct.uav_payload_sim2real.ctbr_command_contract import (
        CtbrLimits,
        shape_ctbr_torch,
    )
    from uav_payload_lab.tasks.direct.uav_payload_sim2real.fastslow_runtime import (
        causal_ema_alpha,
        compute_multirate_schedule,
        update_fastslow_context,
    )
    from uav_payload_lab.tasks.direct.uav_payload_sim2real.train_student_z import (
        FastSlowStudentEncoder,
    )

    def safe_load_teacher(runner, checkpoint_path: str) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=runner.device,
            weights_only=False,
        )
        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get("model", checkpoint.get("state_dict", checkpoint)),
        )
        runner.alg.policy.load_state_dict(state, strict=True)

    def runner_cfg(agent_cfg, env_cfg):
        config = agent_cfg.to_dict()
        for field in (
            "proprio_obs_dim",
            "privileged_obs_dim",
            "rma_z_dim",
            "rma_z_exp_dim",
            "rma_use_mu",
            "rma_context_mode",
            "rma_use_physics_anchor",
            "rma_mu_hidden_dims",
            "rma_activation",
        ):
            if not hasattr(env_cfg, field):
                raise RuntimeError(f"environment lacks RMA field: {field}")
        config["policy"].update(
            {
                "proprio_obs_dim": int(env_cfg.proprio_obs_dim),
                "privileged_obs_dim": int(env_cfg.privileged_obs_dim),
                "z_dim": int(env_cfg.rma_z_dim),
                "z_exp_dim": int(env_cfg.rma_z_exp_dim),
                "use_mu": bool(env_cfg.rma_use_mu),
                "context_mode": str(env_cfg.rma_context_mode),
                "use_physics_anchor": bool(env_cfg.rma_use_physics_anchor),
                "mu_hidden_dims": list(env_cfg.rma_mu_hidden_dims),
                "activation": str(env_cfg.rma_activation),
            }
        )
        return config

    def observation_tensor(observation):
        if isinstance(observation, torch.Tensor):
            return observation
        if hasattr(observation, "get") and observation.get("policy") is not None:
            return observation["policy"]
        return observation["policy"]

    @hydra_task_config(args_cli.task, args_cli.agent)
    def collect(env_cfg, agent_cfg):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.rma_context_mode = "split_hard"
        env_cfg.rma_use_physics_anchor = False
        env_cfg.rma_phys_anchor_coef = 0.0
        env_cfg.scene.num_envs = int(args_cli.num_envs or env_cfg.scene.num_envs)
        env_cfg.seed = int(args_cli.seed)
        env_cfg.sim.device = args_cli.device or env_cfg.sim.device
        env_cfg.log_dir = str(Path(args_cli.teacher).expanduser().resolve().parent)

        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None,
        )
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        base_env = env.unwrapped
        runner = OnPolicyRunner(
            env,
            runner_cfg(agent_cfg, env_cfg),
            log_dir=None,
            device=agent_cfg.device,
        )
        teacher_path = str(Path(args_cli.teacher).expanduser().resolve())
        safe_load_teacher(runner, teacher_path)
        policy = runner.alg.policy
        if str(getattr(policy, "context_mode", "")) != "split_hard":
            raise RuntimeError("DAgger requires a split_hard Teacher")

        student_path = Path(args_cli.student).expanduser().resolve()
        student_checkpoint = torch.load(
            student_path,
            map_location=env.device,
            weights_only=False,
        )
        if student_checkpoint.get("student_context_mode") != "split":
            raise RuntimeError("DAgger requires a split Fast/Slow Student")
        history_len = int(student_checkpoint.get("history_len", args_cli.history_len))
        if history_len != int(args_cli.history_len):
            raise RuntimeError("Student checkpoint history length differs from CLI")
        encoder = FastSlowStudentEncoder(
            input_dim=int(student_checkpoint.get("input_dim", 21)),
            history_len=history_len,
            z_slow_dim=int(student_checkpoint.get("z_slow_dim", 2)),
            z_fast_dim=int(student_checkpoint.get("z_fast_dim", 3)),
        ).to(env.device)
        encoder.load_state_dict(student_checkpoint["state_dict"], strict=True)
        encoder.eval()

        dt = float(base_env.step_dt)
        schedule = compute_multirate_schedule(
            history_len,
            dt,
            float(args_cli.slow_warmup_sec),
            float(args_cli.slow_update_hz),
            float(args_cli.fast_update_hz),
        )
        precontrol_steps = int(round(float(args_cli.precontrol_sec) / dt))
        if precontrol_steps != 180:
            raise RuntimeError(
                f"V8.9 requires exactly 180 Position precontrol steps, got {precontrol_steps}"
            )
        slow_alpha = causal_ema_alpha(dt, float(args_cli.slow_filter_tau_sec))
        limits = CtbrLimits.from_contract(
            repo_root / "configs" / "v89_training_acceptance_contract.json"
        )

        num_envs = int(env.num_envs)
        obs = env.get_observations()
        history = torch.zeros(num_envs, history_len, 21, device=env.device)
        history_fill = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        episode_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        env_ids = torch.arange(num_envs, dtype=torch.long, device=env.device)
        generation = torch.zeros_like(env_ids)
        episode_ids = env_ids.clone()
        z_slow_raw = torch.zeros(num_envs, 2, device=env.device)
        z_slow_target = torch.zeros_like(z_slow_raw)
        z_slow_cache = torch.zeros_like(z_slow_raw)
        z_fast_cache = torch.zeros(num_envs, 3, device=env.device)

        output = Path(args_cli.output).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        buffers = {
            key: []
            for key in (
                "inputs",
                "labels",
                "labels_ml",
                "history_source",
                "seed",
                "env_id",
                "episode_id",
                "episode_step",
                "student_action_raw",
                "student_action_shaped",
                "executed_action_shaped",
            )
        }
        shard_idx = 0
        samples = 0
        z_sum = torch.zeros(5)
        z_sumsq = torch.zeros(5)
        z_min = torch.full((5,), float("inf"))
        z_max = torch.full((5,), -float("inf"))

        def save_shard() -> None:
            nonlocal shard_idx
            if not buffers["inputs"]:
                return
            data = {key: torch.cat(parts, dim=0) for key, parts in buffers.items()}
            data["episode_keys"] = torch.unique(
                torch.stack((data["seed"], data["episode_id"]), dim=1), dim=0
            ).tolist()
            audit = audit_shard_tensors(
                data["inputs"],
                data["labels"],
                data["labels_ml"],
                history_len=history_len,
            )
            if not audit["all_finite"] or audit["slow_identity_max_abs"] != 0.0:
                raise RuntimeError(f"DAgger shard failed audit: {audit}")
            path = output / f"shard_{shard_idx:04d}.pt"
            torch.save(data, path)
            print(f"[DAgger] saved {path} samples={data['inputs'].shape[0]}")
            for parts in buffers.values():
                parts.clear()
            shard_idx += 1

        total_steps = precontrol_steps + int(args_cli.steps)
        source_id = 1 + int(args_cli.dagger_round)
        started = time.time()
        with torch.inference_mode():
            for global_step in range(total_steps):
                obs_tensor = observation_tensor(obs)
                proprio = obs_tensor[:, :21]
                privileged = obs_tensor[:, 21:26]
                history = torch.roll(history, shifts=-1, dims=1)
                history[:, -1] = proprio
                history_fill = torch.clamp(history_fill + 1, max=history_len)

                context = update_fastslow_context(
                    encoder=encoder,
                    obs_history=history,
                    episode_steps=episode_steps,
                    schedule=schedule,
                    slow_filter_alpha=slow_alpha,
                    context_runtime_mode="fast_slow",
                    z_slow_raw=z_slow_raw,
                    z_slow_target=z_slow_target,
                    z_slow_cache=z_slow_cache,
                    z_fast_cache=z_fast_cache,
                )
                if hasattr(policy, "use_mu"):
                    policy.use_mu = False
                student_raw = policy.act_inference(
                    {"policy": torch.cat((proprio, context.z_hat), dim=1)}
                )
                previous = base_env._last_transmitted_actions.clone()
                student_shaped = shape_ctbr_torch(student_raw, previous, limits)
                position_raw = base_env.compute_position_hold_ctbr()
                choice = choose_dagger_step(
                    position_action=position_raw,
                    student_action=student_raw,
                    teacher_z=policy.mu(privileged).detach(),
                    episode_step=episode_steps,
                    precontrol_steps=precontrol_steps,
                )
                executed_shaped = shape_ctbr_torch(
                    choice.executed_action,
                    previous,
                    limits,
                )

                valid = (
                    (~choice.position_active)
                    & (history_fill >= history_len)
                    & (episode_steps.remainder(int(args_cli.sample_stride)) == 0)
                )
                count = int(valid.sum().item())
                if count:
                    labels = choice.label_z[valid].detach().cpu().to(torch.float32)
                    buffers["inputs"].append(
                        history[valid].detach().cpu().to(torch.float16)
                    )
                    buffers["labels"].append(labels)
                    buffers["labels_ml"].append(
                        privileged[valid, :2].detach().cpu().to(torch.float32)
                    )
                    buffers["history_source"].append(
                        torch.full((count,), source_id, dtype=torch.uint8)
                    )
                    buffers["seed"].append(
                        torch.full((count,), int(args_cli.seed), dtype=torch.int64)
                    )
                    buffers["env_id"].append(env_ids[valid].detach().cpu())
                    buffers["episode_id"].append(episode_ids[valid].detach().cpu())
                    buffers["episode_step"].append(episode_steps[valid].detach().cpu())
                    buffers["student_action_raw"].append(
                        student_raw[valid].detach().cpu().to(torch.float32)
                    )
                    buffers["student_action_shaped"].append(
                        student_shaped[valid].detach().cpu().to(torch.float32)
                    )
                    buffers["executed_action_shaped"].append(
                        executed_shaped[valid].detach().cpu().to(torch.float32)
                    )
                    samples += count
                    z_sum += labels.sum(dim=0)
                    z_sumsq += labels.square().sum(dim=0)
                    z_min = torch.minimum(z_min, labels.min(dim=0).values)
                    z_max = torch.maximum(z_max, labels.max(dim=0).values)

                obs, _, dones, _ = env.step(executed_shaped)
                episode_steps += 1
                done = dones.to(dtype=torch.bool).reshape(-1)
                if torch.any(done):
                    history[done] = 0
                    history_fill[done] = 0
                    episode_steps[done] = 0
                    generation[done] += 1
                    episode_ids[done] = generation[done] * num_envs + env_ids[done]
                    z_slow_raw[done] = 0
                    z_slow_target[done] = 0
                    z_slow_cache[done] = 0
                    z_fast_cache[done] = 0
                    if hasattr(policy, "reset"):
                        policy.reset(done)

                if (global_step + 1) % int(args_cli.save_every) == 0:
                    save_shard()
        save_shard()

        z_mean = z_sum / max(samples, 1)
        z_std = torch.sqrt(
            torch.clamp(z_sumsq / max(samples, 1) - z_mean.square(), min=1e-12)
        )
        source_name = f"student_dagger_round_{int(args_cli.dagger_round)}"
        meta = {
            "checkpoint": teacher_path,
            "student_checkpoint": str(student_path),
            "teacher_context_mode": "split_hard",
            "num_envs": num_envs,
            "history_len": history_len,
            "input_dim": 21,
            "z_dim": 5,
            "z_exp_dim": 2,
            "total_samples": samples,
            "seed": int(args_cli.seed),
            "dagger_round": int(args_cli.dagger_round),
            "precontrol": "position",
            "precontrol_steps": precontrol_steps,
            "history_source_encoding": {str(source_id): source_name},
            "z_stats": {
                "mean": z_mean.tolist(),
                "std": z_std.tolist(),
                "min": z_min.tolist(),
                "max": z_max.tolist(),
            },
        }
        torch.save(meta, output / "meta.pt")
        report = {
            "passed": samples > 0,
            "total_samples": samples,
            "num_shards": shard_idx,
            "teacher": teacher_path,
            "student": str(student_path),
            "history_source": source_name,
            "seed": int(args_cli.seed),
            "wall_time_sec": time.time() - started,
        }
        (output / "collect_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (output / "dataset_audit.json").write_text(
            json.dumps(
                {
                    "passed": samples > 0,
                    "total_samples": samples,
                    "hard_identity_ok": True,
                    "all_finite": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        env.close()
        if samples <= 0:
            raise RuntimeError("DAgger collection produced no complete Student histories")
        print(f"[DAgger] DONE samples={samples} output={output}")

    try:
        collect()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
