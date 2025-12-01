# uav_payload_lab_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, CUBOID_MARKER_CFG
from isaaclab.utils.math import subtract_frame_transforms

from .uav_payload_lab_env_cfg import UavPayloadLabEnvCfg



class UavPayloadLabEnv(DirectRLEnv):
    """完全照搬 QuadcopterEnv 的实现，只是改了类名 & cfg 类型。"""

    cfg: UavPayloadLabEnvCfg

    def __init__(self, cfg: UavPayloadLabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # ★ 任务：起点 / 终点（相对 env_origin 的偏移）
        self._start_offset = torch.tensor(cfg.start_pos_w, dtype=torch.float, device=self.device)
        self._goal_offset  = torch.tensor(cfg.goal_pos_w,  dtype=torch.float, device=self.device)
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "r_pos",         # 位置主项
                "r_tilt",        # 摆角 shaping 项
                "r_swing",       # 摆速 shaping 项
                "time_penalty",  # 时间惩罚
                "death_penalty", # 摔机惩罚
                "total",         # 总 reward
            ]
        }
        # Get specific body indices
        # Get specific body indices ---------------------------------------
        # 根 body：官方案例也是这样写，用到的是 ids 列表
        body_ids, body_names = self._robot.find_bodies("body")
        self._body_id = body_ids  # list[int]，用于 set_external_force_and_torque 的 body_ids

        # payload 刚体：这里只需要「一个具体 body index」用于 body_pos_w 的第二维索引
        payload_ids, payload_names = self._robot.find_bodies("link")
        if len(payload_ids) == 0:
            raise RuntimeError("UavPayloadLabEnv: cannot find payload body named 'link'.")
        self._payload_id = payload_ids[0]  # int，用来写 p_load_w = body_pos_w[:, self._payload_id, :]


        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # 摆角历史（deg）用于计算角速度（deg/s）
        # 摆角历史缓冲区，延迟到首次 _get_observations 再按实际形状创建
        self._prev_tilt_deg = None
        self._tilt_vel_deg = None
        self._has_prev_tilt = None
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        # 期望角速度（rad/s）
        body_rate_des = self.cfg.body_rate_max * self._actions[:, 1:]  # (num_envs, 3)

        # P 环：τ_raw = Kp * (ω_des - ω_meas)
        tau_raw = self.cfg.rate_kp * (body_rate_des - self._robot.data.root_ang_vel_b)

        # 力矩限幅：|τ_i| <= moment_scale
        self._moment[:, 0, :] = torch.clamp(
            tau_raw,
            -self.cfg.moment_scale,
            self.cfg.moment_scale,
        )

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        """构造 13 维 obs，语义尽量和老版 SimpleHoverEnv 对齐：

        obs = [
            0:3   r_load_uav = p_load - p_uav          （世界系）
            3:6   e_load     = goal_payload - p_load   （世界系）
            6:9   e_uav      = goal_uav     - p_uav    （世界系）
            9:11  [tx_deg, ty_deg]                     （摆角，deg）
            11:13[wx_deg_s, wy_deg_s]                  （摆角角速度，deg/s）
        ]
        """
        # === UAV & payload 世界坐标 ===
        # UAV 根（世界系）
        p_uav_w = self._robot.data.root_pos_w              # (N,3)

        # payload body（世界系）
        # body_pos_w 形状通常为 (N, num_bodies, 3)，这里用你之前定义的 _payload_id
        p_load_w = self._robot.data.body_pos_w[:, self._payload_id, :]  # (N,3)

        # (1) payload 相对 UAV
        r_load_uav = p_load_w - p_uav_w                     # (N,3)

        # === 目标点 ===
        # 先沿用 Stage0：goal_uav = _desired_pos_w
        goal_uav_w = self._desired_pos_w                    # (N,3)

        # payload 目标点：简单假设在 UAV 目标点正下方 rope_length
        goal_payload_w = goal_uav_w.clone()
        goal_payload_w[:, 2] -= self.cfg.rope_length

        # (2) payload 相对 payload 目标
        e_load = goal_payload_w - p_load_w                  # (N,3)

        # (3) UAV 相对 UAV 目标
        e_uav = goal_uav_w - p_uav_w                        # (N,3)

        # === 摆角（deg）：由 UAV->payload 向量几何计算 ===
        # r = p_load - p_uav，理想情况下 r ≈ (0, 0, -L)
        r = p_load_w - p_uav_w                              # (N,3)
        dx = r[:, 0]
        dy = r[:, 1]
        dz = r[:, 2]

        # 避免除零：den ≈ -dz ≈ L (>0)
        den = torch.clamp(-dz, min=1e-3)

        # 定义：
        #   tx = atan2(dx, -dz)
        #   ty = atan2(dy, -dz)
        # 小角度时 tx ≈ dx/L, ty ≈ dy/L，和你老工程保持一致
        tx_rad = torch.atan2(dx, den)
        ty_rad = torch.atan2(dy, den)

        deg = 180.0 / math.pi
        tx_deg = tx_rad * deg
        ty_deg = ty_rad * deg

        tilt_deg = torch.stack([tx_deg, ty_deg], dim=-1)    # (N,2)

        dt = max(self.step_dt, 1e-6)

        # lazy init / 形状自检：以当前 tilt_deg 的 shape 为准
        if (
            self._prev_tilt_deg is None
            or self._prev_tilt_deg.shape != tilt_deg.shape
        ):
            self._prev_tilt_deg = torch.zeros_like(tilt_deg)
            self._tilt_vel_deg = torch.zeros_like(tilt_deg)
            self._has_prev_tilt = torch.zeros(
                tilt_deg.shape[0], dtype=torch.bool, device=self.device
            )

        delta_tilt = tilt_deg - self._prev_tilt_deg   # (N,2)

        # 先全置零，再对 has_prev_tilt=True 的 env 用差分更新
        w_deg = torch.zeros_like(delta_tilt)          # (N,2)
        mask = self._has_prev_tilt                    # (N,)
        if mask.any():
            w_deg[mask] = delta_tilt[mask] / dt       # 只对 True 的 env 算角速度

        # 更新历史
        self._prev_tilt_deg = tilt_deg
        self._tilt_vel_deg = w_deg
        self._has_prev_tilt[:] = True

        # === 拼 obs 向量 ===
        obs = torch.cat(
            [
                r_load_uav,          # 0:3
                e_load,              # 3:6
                e_uav,               # 6:9
                tilt_deg,            # 9:11
                w_deg,               # 11:13
            ],
            dim=-1,
        )

        observations = {"policy": obs}
        return observations


    def _get_rewards(self) -> torch.Tensor:
        """基于 payload 位置 + 摆角 + 摆角速度的高斯 shaping reward."""

        # === 任务几何：payload 到目标点的距离 ===
        # UAV / payload / 目标点
        p_uav_w = self._robot.data.root_pos_w                          # (N, 3)
        p_load_w = self._robot.data.body_pos_w[:, self._payload_id, :] # (N, 3)

        goal_uav_w = self._desired_pos_w                               # (N, 3)
        goal_payload_w = goal_uav_w.clone()
        goal_payload_w[:, 2] -= self.cfg.rope_length

        # payload 相对 payload 目标
        e_load = goal_payload_w - p_load_w                              # (N, 3)
        dist = torch.linalg.norm(e_load, dim=1)                         # (N,)

        # === 摆角（deg）：和 _get_observations 保持一致 ===
        r = p_load_w - p_uav_w                                          # (N, 3)
        dx = r[:, 0]
        dy = r[:, 1]
        dz = r[:, 2]

        den = torch.clamp(-dz, min=1e-3)
        tx_rad = torch.atan2(dx, den)
        ty_rad = torch.atan2(dy, den)

        deg = 180.0 / math.pi
        tx_deg = tx_rad * deg
        ty_deg = ty_rad * deg
        theta_deg = torch.sqrt(tx_deg * tx_deg + ty_deg * ty_deg)       # (N,)

        # === 摆角角速度（deg/s）：直接用 _get_observations 差分出来的结果 ===
        wx_deg = self._tilt_vel_deg[:, 0]
        wy_deg = self._tilt_vel_deg[:, 1]
        swing_deg_s = torch.sqrt(wx_deg * wx_deg + wy_deg * wy_deg)     # (N,)

        # === 单维高斯打分 r_pos_raw / r_tilt_raw / r_swing_raw ===
        eps = 1e-6
        sigma_pos = max(float(self.cfg.sigma_pos), eps)
        sigma_tilt = max(float(self.cfg.sigma_tilt_deg), eps)
        sigma_swing = max(float(self.cfg.sigma_swing_deg_s), eps)

        z_pos = dist / sigma_pos
        z_tilt = theta_deg / sigma_tilt
        z_swing = swing_deg_s / sigma_swing

        r_pos_raw = torch.exp(-0.5 * z_pos * z_pos)       # [0,1]
        r_tilt_raw = torch.exp(-0.5 * z_tilt * z_tilt)    # [0,1]
        r_swing_raw = torch.exp(-0.5 * z_swing * z_swing) # [0,1]

        # === FlyThrough 风格：pos + pos * (tilt + swing) ===
        pos_w = float(self.cfg.pos_weight)
        tilt_w = float(self.cfg.tilt_weight)

        shaping = tilt_w * r_tilt_raw + 0.5 * tilt_w * r_swing_raw
        r_pos_term = pos_w * r_pos_raw
        r_shape_term = pos_w * r_pos_raw * shaping

        # 轻微时间惩罚：随 step_dt 缩放，保证不同 dt 下语义一致
        time_penalty = float(self.cfg.time_penalty)
        r_time = -time_penalty * self.step_dt
        r_time_vec = torch.full_like(dist, r_time)

        reward = r_pos_term + r_shape_term + r_time_vec
        # ---- 死亡惩罚：高度 <0.1 或 >2.0 就一次性扣 death_penalty ----
        z = self._robot.data.root_pos_w[:, 2]
        died = torch.logical_or(z < 0.1, z > 5.0)
        death_penalty_vec = -self.cfg.death_penalty * died.float()
        reward = reward + death_penalty_vec

        # === Logging：分项累计，便于 tensorboard / CSV ===
        rewards = {
            "r_pos": r_pos_term,
            "r_tilt": pos_w * r_pos_raw * tilt_w * r_tilt_raw,
            "r_swing": pos_w * r_pos_raw * (0.5 * tilt_w * r_swing_raw),
            "time_penalty": r_time_vec,
            "death_penalty": death_penalty_vec,

        }
        # 顺便记录总 reward
        rewards["total"] = reward

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 5.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        p_load_w = self._robot.data.body_pos_w[env_ids, self._payload_id, :]
        goal_uav_w = self._desired_pos_w[env_ids]
        goal_payload_w = goal_uav_w.clone()
        goal_payload_w[:, 2] -= self.cfg.rope_length

        final_distance_to_goal = torch.linalg.norm(
            goal_payload_w - p_load_w, dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        # self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        # self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        # self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        # 每个 env 的原点（平铺用）
        env_origins = self._terrain.env_origins[env_ids]
        default_root_state[:, :3] = env_origins + self._start_offset
        # UAV 目标点（世界系）= env_origin + goal_offset
        self._desired_pos_w[env_ids] = env_origins + self._goal_offset
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # reset 摆角历史，用于新 episode 的角速度差分
        if isinstance(self._prev_tilt_deg, torch.Tensor):
            self._prev_tilt_deg[env_ids] = 0.0
            self._tilt_vel_deg[env_ids] = 0.0
            self._has_prev_tilt[env_ids] = False

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)


