# play_student_phase2.py
# Paper-grade play: record z_hat vs z_teacher, z_rmse, raw/clamped actions, key states, errors, thrust/moment.

import argparse
import sys
import os
import csv
import time
import torch
import torch.nn as nn
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Phase-2: Teacher policy + Student z_hat, record comprehensive CSV.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--checkpoint", type=str, required=True, help="Teacher checkpoint (.pt)")
parser.add_argument("--encoder", type=str, required=True, help="Student encoder .pth")
parser.add_argument("--max_steps", type=int, default=3000)
parser.add_argument("--csv", type=str, required=True)

# extra compare
parser.add_argument("--compare_teacher_action", action="store_true",
                    help="Also compute action_teacher by feeding z_teacher to policy; record diff.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# SAFE imports after SimulationApp
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import uav_payload_lab.tasks  # noqa: F401


class CNNStudentEncoder(nn.Module):
    def __init__(self, input_dim=21, history_len=50, output_dim=5):
        super().__init__()
        self.history_len = history_len
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
        x = x.permute(0, 2, 1)
        return self.mlp(self.cnn(x))


def load_model_only(policy_nn, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    policy_nn.load_state_dict(sd, strict=False)


def _get_obs_tensor(obs_td):
    if isinstance(obs_td, torch.Tensor):
        return obs_td
    if isinstance(obs_td, dict) and "policy" in obs_td:
        return obs_td["policy"]
    if hasattr(obs_td, "get") and obs_td.get("policy") is not None:
        return obs_td["policy"]
    raise RuntimeError(f"Unsupported obs type: {type(obs_td)}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    device = env.unwrapped.device
    base_env = env.unwrapped  # used for accessing _robot / _thrust / _moment if present

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    print(f"[INFO] Loading teacher MODEL ONLY: {args_cli.checkpoint}")
    load_model_only(runner.alg.policy, args_cli.checkpoint, device=device)

    policy = runner.get_inference_policy(device=device)
    policy_nn = runner.alg.policy  # RMAActorCritic

    # deployment uses z_hat (no mu(priv) in the forward)
    if hasattr(policy_nn, "use_mu"):
        policy_nn.use_mu = False

    # load student encoder
    encoder = CNNStudentEncoder().to(device)
    encoder.load_state_dict(torch.load(args_cli.encoder, map_location=device))
    encoder.eval()

    # buffers
    history_len = 50
    proprio_dim = 17
    action_dim = 4
    z_dim = getattr(policy_nn, "z_dim", 5)
    obs_history = torch.zeros((env.num_envs, history_len, proprio_dim + action_dim), device=device)
    last_actions = torch.zeros((env.num_envs, action_dim), device=device)

    obs = env.get_observations()
    dt = env.unwrapped.step_dt

    # CSV rows (env0 only)
    rows = []
    env0 = 0
    prev_v_w = None

    with torch.inference_mode():
        for t in range(args_cli.max_steps):
            obs_oracle_td = obs
            obs_oracle = _get_obs_tensor(obs_oracle_td)  # contains privileged info at 17:22 (oracle)

            # build history input for student
            proprio = obs_oracle[:, :proprio_dim]
            feat = torch.cat([proprio, last_actions], dim=1)  # (N,21)
            obs_history = torch.roll(obs_history, shifts=-1, dims=1)
            obs_history[:, -1, :] = feat

            # student z_hat
            z_hat = encoder(obs_history)  # (N,5)

            # teacher z_teacher from oracle priv (for analysis only)
            priv = obs_oracle[:, 17:17 + z_dim]
            if hasattr(policy_nn, "mu"):
                z_teacher = policy_nn.mu(priv)
            else:
                z_teacher = torch.zeros_like(z_hat)

            z_rmse = torch.sqrt(((z_hat - z_teacher) ** 2).mean(dim=1))  # (N,)

            # feed policy with [proprio, z_hat]
            policy_in = obs_oracle.clone()
            policy_in[:, 17:17 + z_dim] = z_hat

            obs_in = obs_oracle_td.clone() if hasattr(obs_oracle_td, "clone") else {"policy": policy_in}
            if isinstance(obs_in, dict):
                obs_in["policy"] = policy_in
            else:
                obs_in["policy"] = policy_in

            actions_raw = policy(obs_in)
            actions_clamp = actions_raw.clamp(-1.0, 1.0)

            # optional teacher action (for action-diff plots)
            a_teacher = None
            if args_cli.compare_teacher_action:
                policy_in_T = obs_oracle.clone()
                policy_in_T[:, 17:17 + z_dim] = z_teacher
                obs_in_T = obs_oracle_td.clone() if hasattr(obs_oracle_td, "clone") else {"policy": policy_in_T}
                if isinstance(obs_in_T, dict):
                    obs_in_T["policy"] = policy_in_T
                else:
                    obs_in_T["policy"] = policy_in_T
                a_teacher = policy(obs_in_T).clamp(-1.0, 1.0)

            # step env with clamped actions (match training)
            step_out = env.step(actions_clamp)
            if len(step_out) == 4:
                obs, rewards, dones, infos = step_out
            else:
                obs = step_out[0]
                dones = step_out[2]
                rewards = None
                infos = None
            last_actions = actions_clamp.detach()

            if torch.any(dones):
                obs_history[dones] = 0.0
                last_actions[dones] = 0.0
                if hasattr(policy_nn, "reset"):
                    policy_nn.reset(dones)

            # ===== env0 metrics =====
            # from obs layout (proprio):
            # err(0:3), theta(3:5), theta_dot(5:7), quat(7:11), lin_vel(11:14), ang_vel(14:17)
            err0 = obs_oracle[env0, 0:3].detach().cpu().numpy()
            pos_err = float(np.linalg.norm(err0))
            theta_deg = obs_oracle[env0, 3:5].detach().cpu().numpy()
            theta_dot_deg_s = obs_oracle[env0, 5:7].detach().cpu().numpy()

            quat_w = obs_oracle[env0, 7:11].detach().cpu().numpy()
            v_b = obs_oracle[env0, 11:14].detach().cpu().numpy()
            w_b = obs_oracle[env0, 14:17].detach().cpu().numpy()

            # try world vel/acc if available (optional)
            v_w = None
            a_w = np.zeros(3)
            if hasattr(base_env, "_robot") and hasattr(base_env._robot, "data") and hasattr(base_env._robot.data, "root_lin_vel_w"):
                v_w = base_env._robot.data.root_lin_vel_w[env0].detach().cpu().numpy()
                if prev_v_w is not None:
                    a_w = (v_w - prev_v_w) / dt
                prev_v_w = v_w.copy()

            # thrust/moment if available
            thrust_cmd = float(base_env._thrust[env0, 0, 2].detach().cpu()) if hasattr(base_env, "_thrust") else 0.0
            moment_cmd = base_env._moment[env0, 0, :].detach().cpu().numpy() if hasattr(base_env, "_moment") else np.zeros(3)

            # done flag env0
            done0 = bool(dones[env0].item()) if isinstance(dones, torch.Tensor) else False

            # action diff if teacher action computed
            if a_teacher is not None:
                aT0 = a_teacher[env0].detach().cpu().numpy()
                aDiff = actions_clamp[env0].detach().cpu().numpy() - aT0
            else:
                aT0 = np.zeros(4)
                aDiff = np.zeros(4)

            rows.append([
                t * dt,
                # oracle priv (mlw)
                *priv[env0].detach().cpu().numpy().tolist(),
                # z_hat / z_teacher / rmse
                *z_hat[env0].detach().cpu().numpy().tolist(),
                *z_teacher[env0].detach().cpu().numpy().tolist(),
                float(z_rmse[env0].detach().cpu()),
                # errors / swing
                pos_err,
                theta_deg[0], theta_deg[1],
                theta_dot_deg_s[0], theta_dot_deg_s[1],
                # quat/vels
                quat_w[0], quat_w[1], quat_w[2], quat_w[3],
                v_b[0], v_b[1], v_b[2],
                w_b[0], w_b[1], w_b[2],
                a_w[0], a_w[1], a_w[2],
                # actions raw/clamp
                *actions_raw[env0].detach().cpu().numpy().tolist(),
                *actions_clamp[env0].detach().cpu().numpy().tolist(),
                # optional teacher action + diff
                *aT0.tolist(),
                *aDiff.tolist(),
                # thrust/moment
                thrust_cmd, moment_cmd[0], moment_cmd[1], moment_cmd[2],
                # done
                int(done0),
            ])

            if (t + 1) % 200 == 0:
                print(f"t={t+1} | pos_err={pos_err:.3f} | z_rmse(env0)={float(z_rmse[env0]):.3f}")

    os.makedirs(os.path.dirname(args_cli.csv), exist_ok=True)
    with open(args_cli.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time",
            "priv0","priv1","priv2","priv3","priv4",
            "zH0","zH1","zH2","zH3","zH4",
            "zT0","zT1","zT2","zT3","zT4",
            "z_rmse",
            "pos_err",
            "theta_x_deg","theta_y_deg",
            "theta_dot_x_deg_s","theta_dot_y_deg_s",
            "quat0","quat1","quat2","quat3",
            "v_bx","v_by","v_bz",
            "w_bx","w_by","w_bz",
            "a_wx","a_wy","a_wz",
            "a0_raw","a1_raw","a2_raw","a3_raw",
            "a0_clamp","a1_clamp","a2_clamp","a3_clamp",
            "a0_T","a1_T","a2_T","a3_T",
            "a0_diff","a1_diff","a2_diff","a3_diff",
            "thrust_cmd","moment_x","moment_y","moment_z",
            "done",
        ])
        w.writerows(rows)
    print(f"[INFO] Saved: {args_cli.csv} rows={len(rows)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
