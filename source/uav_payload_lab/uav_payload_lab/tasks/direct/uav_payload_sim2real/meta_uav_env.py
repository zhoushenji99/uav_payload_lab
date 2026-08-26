# uav_payload_lab_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, CUBOID_MARKER_CFG
from isaaclab.utils.math import subtract_frame_transforms

from .meta_uav_env_cfg import UavPayloadMetaEnvCfg
from .ctbr_command_contract import CtbrLimits, shape_ctbr_torch
from .hover_reward_terms import normalized_ctbr_terms, uav_tilt_rad_wxyz
from .real_hover_gap import (
    compose_lumped_payload_mass,
    diagonal_inertia_flat,
    half_sine_profile,
    inverse_normalized_quadratic_thrust_ratio,
    normalize_physical_context,
    normalized_quadratic_thrust_ratio,
    rate_gain_from_time_constant,
    select_delayed_actions,
    select_delayed_ring,
    validate_inertia_diagonal,
)



class UavPayloadMetaEnv(DirectRLEnv):
    """完全照搬 QuadcopterEnv 的实现，只是改了类名 & cfg 类型。"""

    cfg: UavPayloadMetaEnvCfg

    def __init__(self, cfg: UavPayloadMetaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._raw_actions = torch.zeros_like(self._actions)
        self._policy_actions = torch.zeros_like(self._actions)
        self._prev_policy_actions = torch.zeros_like(self._actions)
        self._prev_actions = torch.zeros_like(self._actions)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # ★ 任务：起点 / 终点（相对 env_origin 的偏移）
        # 我们可以给它一个随机初始值，防止4096个环境在同一帧同时换目标（造成卡顿）
        self._start_offset = torch.tensor(cfg.start_pos_w, dtype=torch.float, device=self.device)
        self._goal_offset  = torch.tensor(cfg.goal_pos_w,  dtype=torch.float, device=self.device)
                # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "r_pos",         # 位置主项
                "r_tilt",        # 摆角 shaping 项
                "r_swing",       # 摆速 shaping 项
                "time_penalty",  # 时间惩罚（你现在 reward 里没用到的话，可以以后删掉）
                "death_penalty", # 摔机惩罚
                "total",         # 总 reward
                "dist",          # payload 到目标的距离（m）
                "theta_deg",     # payload 合摆角（deg）
                "swing_deg_s",   # payload 合角速度（deg/s）
                "r_action_raw",
                "r_action_l2",
                "r_action_smooth",
                "r_action_jerk",
                "r_uav_tilt",
                "r_actual_rate",
                "action_raw_sum",
                "E_hat_mean",  # [新增] 摆能量(归一化)的时间积分，用于TB里做E_hat_mean
            ]
        }
        # Get specific body indices
        # Get specific body indices ---------------------------------------
        # 根 body：官方案例也是这样写，用到的是 ids 列表
        body_ids, body_names = self._robot.find_bodies("body")
        self._body_id = body_ids  # list[int]，用于 set_external_force_and_torque 的 body_ids

        # payload 刚体：这里只需要「一个具体 body index」用于 body_pos_w 的第二维索引
        payload_ids, payload_names = self._robot.find_bodies(r"^(?!.*winch).*link")
        if len(payload_ids) == 0:
            raise RuntimeError("UavPayloadLabEnv: cannot find payload body named 'link'.")
        self._payload_id = payload_ids[0]  # int，用来写 p_load_w = body_pos_w[:, self._payload_id, :]
        # --- [新增] 获取 Prismatic Joint (绳长关节) 的索引 ---
        # 你的 USD 里关节名字叫 "PrismaticJoint"
        rope_joint_indices, _ = self._robot.find_joints("rope_joint")
        if len(rope_joint_indices) == 0:
            # 容错：防止你 USD 里还没改名成功，保留一个旧名字的查找
            rope_joint_indices, _ = self._robot.find_joints("PrismaticJoint")

        if len(rope_joint_indices) == 0:
            raise RuntimeError("Cannot find 'rope_joint' in USD! Please check joint name.")
        self._rope_joint_idx = rope_joint_indices[0]

        # --- [新增] 初始化记录绳长的 tensor ---
        self._rope_lengths = torch.zeros(self.num_envs, device=self.device)
        # 1) 找关节（建议你把 USD 里的 PrismaticJoint prim 改名 rope_joint）
        self._rope_joint_id = self._robot.find_joints("rope_joint")[0][0]   # 取第一个匹配到的 joint index

        # Apply the measured UAV rigid-body properties at runtime.  This keeps
        # the source USD reusable and makes the deployed nominal values auditable.
        self._apply_uav_physics(self._robot._ALL_INDICES, randomize=False)

        # 2) 计算 F_max（固定：只看 UAV body 质量）
        g = float(self.cfg.sim.gravity[2]) if hasattr(self.cfg.sim, "gravity") else -9.81
        g = abs(g)

        masses0 = self._robot.root_physx_view.get_masses()[0]  # (num_bodies,)
        uav_mass = float(masses0[self._body_id[0]].item())     # 只取 body 的质量
        self._uav_mass = uav_mass  # <<< [新增] 缓存 UAV 质量，给风扰用
        self._F_max = float(
            getattr(self.cfg, "ctbr_total_max_thrust_n", self.cfg.thrust_to_weight * (uav_mass * g))
        )

        # 3) 每个 env 的绳长 / 悬停推力 buffer
        # self._rope_len = torch.full((self.num_envs,), self.cfg.rope_length, device=self.device)
        self._F_hover = torch.full((self.num_envs,), uav_mass * g, device=self.device)  # 先给个初值

        # 缓存默认 mass / COM / inertia（用于每次reset从标称值开始随机）
        self._default_masses_cpu = self._robot.root_physx_view.get_masses().clone()     # (num_envs, num_bodies) on CPU
        self._default_coms_cpu = self._robot.root_physx_view.get_coms().clone()
        self._default_inertias_cpu = self._robot.root_physx_view.get_inertias().clone() # (num_envs, num_bodies, 9) on CPU

        # 记录每个env当前payload质量（放在env device上用于log）
        self._payload_mass = torch.zeros(self.num_envs, device=self.device)
        self._payload_ballast_mass = torch.zeros(self.num_envs, device=self.device)
        self._rope_mass = torch.zeros(self.num_envs, device=self.device)


        # gravity (float)
        self._gravity_magnitude = float(torch.tensor(self.sim.cfg.gravity).norm().item())

        # per-env weight (num_envs,)
        masses_cpu = self._robot.root_physx_view.get_masses()  # (num_envs, num_bodies) on CPU
        self._robot_weight = masses_cpu.sum(dim=1).to(self.device) * self._gravity_magnitude

        # 摆角历史（deg）用于计算角速度（deg/s）
        # 摆角历史缓冲区，延迟到首次 _get_observations 再按实际形状创建
        self._prev_tilt_deg = None
        self._tilt_vel_deg = None
        self._has_prev_tilt = None
        # noisy obs history (only for policy observation)
        self._obs_prev_tilt_deg = None
        self._obs_has_prev_tilt = None
        self._obs_w_deg_filt = None
        self._obs_has_prev_w = None
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        # Wind disturbance module (optional)
        self._init_wind_module()
        self._init_payload_sensor_gap()

        # _policy_actions 是网络真 raw 输出，用于 reward 惩罚；_raw_actions 保留为执行边界内的动作缓存。
        self._raw_actions = torch.zeros_like(self._actions)
        self._prev_raw_actions = torch.zeros_like(self._actions)
        self._policy_actions = torch.zeros_like(self._actions)
        self._prev_policy_actions = torch.zeros_like(self._actions)
        # Last CTBR actually transmitted across the deployment interface after
        # command clipping, before unobservable actuator delay/LPF dynamics.
        self._last_transmitted_actions = torch.zeros_like(self._actions)
        self._prev_transmitted_actions = torch.zeros_like(self._actions)
        self._transmitted_delta = torch.zeros_like(self._actions)
        self._prev_transmitted_delta = torch.zeros_like(self._actions)

        # Per-environment action transport: link delay, first-order actuator
        # response, and conservative thrust/moment effectiveness.
        delay_lo, delay_hi = getattr(
            self.cfg,
            "action_delay_steps_range",
            (int(self.cfg.action_delay_steps), int(self.cfg.action_delay_steps)),
        )
        self._action_max_delay_steps = int(max(delay_lo, delay_hi))
        self._action_queue = torch.zeros(
            (self.num_envs, self._action_max_delay_steps + 1, self._actions.shape[-1]),
            device=self.device,
        )
        self._action_delay_steps_per_env = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._action_lpf_alpha_per_env = torch.ones(self.num_envs, 1, device=self.device)
        self._collective_efficiency = torch.ones(self.num_envs, device=self.device)
        self._moment_efficiency = torch.ones(self.num_envs, 1, device=self.device)

        # [新增] 低通滤波内部状态
        self._filtered_actions = torch.zeros_like(self._actions)
        self._ctbr_body_rate_limit = torch.tensor(self.cfg.ctbr_body_rate_limit, dtype=torch.float, device=self.device)
        contract_path = Path(__file__).resolve().parents[6] / "configs/v89_training_acceptance_contract.json"
        self._ctbr_limits = CtbrLimits.from_contract(contract_path)
        self._ctbr_rate_kp = torch.tensor(self.cfg.ctbr_rate_kp, dtype=torch.float, device=self.device)
        self._ctbr_rate_kp_per_env = self._ctbr_rate_kp.unsqueeze(0).repeat(self.num_envs, 1)
        self._ctbr_rate_time_constant_s = torch.zeros(self.num_envs, 3, device=self.device)
        self._ctbr_moment_limit = torch.tensor(self.cfg.ctbr_moment_limit, dtype=torch.float, device=self.device)
        self._ctbr_rate_sign = torch.tensor(self.cfg.ctbr_px4_to_isaac_rate_sign, dtype=torch.float, device=self.device)
        self._ctbr_action_scale = torch.cat(
            (torch.ones(1, device=self.device), self._ctbr_body_rate_limit)
        )
        self._ctbr_thrust_body_z = torch.zeros(self.num_envs, device=self.device)
        self._ctbr_rate_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._ctbr_rate_meas = torch.zeros(self.num_envs, 3, device=self.device)
        self._ctbr_rate_error = torch.zeros(self.num_envs, 3, device=self.device)
        self._position_history_rate_bias_px4 = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._sample_body_rate_dynamics(self._robot._ALL_INDICES)

    def _sample_body_rate_dynamics(self, env_ids: torch.Tensor) -> None:
        """Sample a per-axis first-order PX4 rate-loop approximation."""
        count = int(env_ids.numel())
        if count == 0:
            return
        ranges = torch.as_tensor(
            getattr(self.cfg, "ctbr_rate_time_constant_range_s", ()),
            dtype=torch.float32,
            device=self.device,
        )
        if (
            ranges.shape != (3, 2)
            or not torch.isfinite(ranges).all()
            or torch.any(ranges[:, 0] <= 0.0)
            or torch.any(ranges[:, 1] < ranges[:, 0])
        ):
            raise ValueError(
                "ctbr_rate_time_constant_range_s must contain three positive [low, high] pairs"
            )
        tau = ranges[:, 0].unsqueeze(0) + torch.rand(
            count, 3, device=self.device
        ) * (ranges[:, 1] - ranges[:, 0]).unsqueeze(0)
        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        body_id = int(self._body_id[0])
        inertia_flat = self._robot.root_physx_view.get_inertias()[env_ids_cpu, body_id]
        inertia_diag = inertia_flat[:, [0, 4, 8]].to(self.device)
        self._ctbr_rate_time_constant_s[env_ids] = tau
        self._ctbr_rate_kp_per_env[env_ids] = rate_gain_from_time_constant(
            inertia_diag, tau
        )

    def _apply_uav_physics(self, env_ids: torch.Tensor, *, randomize: bool) -> None:
        """Apply measured UAV mass, COM, and a physical diagonal inertia."""
        if not getattr(self.cfg, "enable_real_hover_gap", False):
            return

        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        count = int(env_ids_cpu.numel())
        if count == 0:
            return

        nominal_inertia = validate_inertia_diagonal(self.cfg.uav_inertia_diag_kg_m2)
        masses = self._robot.root_physx_view.get_masses().clone()
        coms = self._robot.root_physx_view.get_coms().clone()
        inertias = self._robot.root_physx_view.get_inertias().clone()
        body_id = int(self._body_id[0])

        mass_scale = torch.ones(count, dtype=masses.dtype, device="cpu")
        inertia_scale = torch.ones(count, dtype=inertias.dtype, device="cpu")
        com_offset = torch.zeros(count, 3, dtype=coms.dtype, device="cpu")
        if randomize:
            mass_lo, mass_hi = self.cfg.uav_mass_scale_range
            inertia_lo, inertia_hi = self.cfg.uav_inertia_scale_range
            com_lo, com_hi = self.cfg.uav_com_offset_range_m
            mass_scale.uniform_(float(mass_lo), float(mass_hi))
            inertia_scale.uniform_(float(inertia_lo), float(inertia_hi))
            com_offset.uniform_(float(com_lo), float(com_hi))

        masses[env_ids_cpu, body_id] = float(self.cfg.uav_mass_kg) * mass_scale
        nominal_com = torch.tensor(self.cfg.uav_com_m, dtype=coms.dtype, device="cpu")
        coms[env_ids_cpu, body_id, :3] = nominal_com.unsqueeze(0) + com_offset
        inertia_flat = diagonal_inertia_flat(nominal_inertia, device="cpu").to(inertias.dtype)
        inertias[env_ids_cpu, body_id] = inertia_flat.unsqueeze(0) * inertia_scale.unsqueeze(1)

        if not (
            torch.isfinite(masses[env_ids_cpu, body_id]).all()
            and torch.isfinite(coms[env_ids_cpu, body_id]).all()
            and torch.isfinite(inertias[env_ids_cpu, body_id]).all()
        ):
            raise RuntimeError("Real Hover Gap produced non-finite UAV rigid-body properties.")

        self._robot.root_physx_view.set_masses(masses, env_ids_cpu)
        self._robot.root_physx_view.set_coms(coms, env_ids_cpu)
        self._robot.root_physx_view.set_inertias(inertias, env_ids_cpu)
        if hasattr(self, "_uav_mass_tensor"):
            self._uav_mass_tensor[env_ids] = masses[env_ids_cpu, body_id].to(self.device)

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

    def _decode_px4_ctbr_action(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PX4-native CTBR: thrust_body[2] plus body-rate setpoints."""
        decoded = torch.empty_like(actions)
        decoded[:, 0] = actions[:, 0].clamp(-1.0, 0.0)
        decoded[:, 1:4] = torch.clamp(
            actions[:, 1:4],
            min=-self._ctbr_body_rate_limit,
            max=self._ctbr_body_rate_limit,
        )
        thrust_body_z = decoded[:, 0]
        rate_sp_isaac = decoded[:, 1:4] * self._ctbr_rate_sign
        return decoded, thrust_body_z, rate_sp_isaac

    def compute_position_hold_ctbr(self) -> torch.Tensor:
        """Return the Position CTBR target that can directly drive the plant."""
        root_pos_w = self._robot.data.root_pos_w
        root_vel_w = self._robot.data.root_lin_vel_w
        root_quat_w = self._robot.data.root_quat_w
        target_pos_w = self._terrain.env_origins + self._start_offset

        pos_kp = torch.as_tensor(
            self.cfg.position_history_pos_kp, device=self.device, dtype=root_pos_w.dtype
        )
        vel_kd = torch.as_tensor(
            self.cfg.position_history_vel_kd, device=self.device, dtype=root_pos_w.dtype
        )
        accel_limit = torch.as_tensor(
            self.cfg.position_history_accel_limit_mps2,
            device=self.device,
            dtype=root_pos_w.dtype,
        )
        accel_cmd_w = pos_kp * (target_pos_w - root_pos_w) - vel_kd * root_vel_w
        accel_cmd_w = torch.clamp(accel_cmd_w, -accel_limit, accel_limit)

        w, x, y, z = root_quat_w.unbind(dim=-1)
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
        pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))

        vertical_accel = (self._gravity_magnitude + accel_cmd_w[:, 2]).clamp_min(1.0)
        # This USD's +roll produces +world-y acceleration under the existing
        # PX4-to-Isaac action convention (verified by closed-loop trace).
        roll_des = torch.atan2(accel_cmd_w[:, 1], vertical_accel)
        pitch_des = torch.atan2(
            accel_cmd_w[:, 0],
            torch.sqrt(vertical_accel.square() + accel_cmd_w[:, 1].square()),
        )
        yaw_error = torch.atan2(torch.sin(-yaw), torch.cos(-yaw))
        attitude_error = torch.stack((roll_des - roll, pitch_des - pitch, yaw_error), dim=-1)
        attitude_kp = torch.as_tensor(
            self.cfg.position_history_attitude_kp,
            device=self.device,
            dtype=root_pos_w.dtype,
        )
        rate_sp_isaac = torch.clamp(
            attitude_kp * attitude_error,
            -self._ctbr_body_rate_limit,
            self._ctbr_body_rate_limit,
        )

        total_mass = self._robot_weight / max(self._gravity_magnitude, 1e-6)
        specific_force_w = accel_cmd_w.clone()
        specific_force_w[:, 2] += self._gravity_magnitude
        required_force = total_mass * torch.linalg.norm(specific_force_w, dim=-1)
        thrust_ratio = (
            required_force / (float(self._F_max) * self._collective_efficiency)
        ).clamp(0.0, 1.0)

        actions = torch.zeros(self.num_envs, 4, device=self.device)
        actions[:, 0] = -self._thrust_ratio_to_collective_signal(thrust_ratio)
        rate_sp_px4 = rate_sp_isaac / self._ctbr_rate_sign
        rate_sp_px4 = rate_sp_px4 + self._position_history_rate_bias_px4
        position_rate_limit = torch.as_tensor(
            self.cfg.position_history_rate_limit_rps,
            device=self.device,
            dtype=root_pos_w.dtype,
        )
        actions[:, 1:4] = torch.clamp(
            rate_sp_px4, -position_rate_limit, position_rate_limit
        )
        decoded, _, _ = self._decode_px4_ctbr_action(actions)
        return decoded

    def _collective_signal_to_thrust_ratio(self, signal: torch.Tensor) -> torch.Tensor:
        """Map normalized PX4 collective signal to realized thrust ratio."""
        model = str(getattr(self.cfg, "ctbr_thrust_model", "linear"))
        if model == "linear":
            return signal.clamp(0.0, 1.0)
        if model == "normalized_quadratic":
            return normalized_quadratic_thrust_ratio(
                signal,
                self.cfg.ctbr_thrust_curve_coeffs,
            )
        raise ValueError(f"Unsupported ctbr_thrust_model: {model!r}")

    def _thrust_ratio_to_collective_signal(self, thrust_ratio: torch.Tensor) -> torch.Tensor:
        """Invert the configured collective-thrust model for reset seeding."""
        model = str(getattr(self.cfg, "ctbr_thrust_model", "linear"))
        if model == "linear":
            return thrust_ratio.clamp(0.0, 1.0)
        if model == "normalized_quadratic":
            return inverse_normalized_quadratic_thrust_ratio(
                thrust_ratio,
                self.cfg.ctbr_thrust_curve_coeffs,
            )
        raise ValueError(f"Unsupported ctbr_thrust_model: {model!r}")

    def _pre_physics_step(self, actions: torch.Tensor):
        # 1. 记录上一帧动作缓存
        self._prev_raw_actions = self._raw_actions.clone()
        self._prev_policy_actions = self._policy_actions.clone()

        # 2. 记录当前网络输出，并按统一PX4 CTBR绝对/变化率合同整形。
        self._policy_actions = actions.clone()
        decoded_target, _, _ = self._decode_px4_ctbr_action(self._policy_actions)
        self._prev_transmitted_actions = self._last_transmitted_actions.clone()
        self._last_transmitted_actions = shape_ctbr_torch(
            decoded_target,
            self._prev_transmitted_actions,
            self._ctbr_limits,
        )
        self._transmitted_delta = (
            self._last_transmitted_actions - self._prev_transmitted_actions
        )
        self._raw_actions = self._last_transmitted_actions.clone()

        # 3. Per-environment link delay.
        self._action_queue = torch.roll(self._action_queue, shifts=-1, dims=1)
        self._action_queue[:, -1, :] = self._last_transmitted_actions
        delayed_actions = select_delayed_actions(
            self._action_queue, self._action_delay_steps_per_env
        )

        # 4. 经过一阶低通滤波 (LPF)
        alpha = self._action_lpf_alpha_per_env
        self._filtered_actions = (1.0 - alpha) * self._filtered_actions + alpha * delayed_actions

        # 5. PX4 CTBR -> Isaac thrust and moment.
        self._actions, thrust_body_z, rate_sp_isaac = self._decode_px4_ctbr_action(self._filtered_actions)
        self._ctbr_thrust_body_z = thrust_body_z
        collective_signal = (-thrust_body_z).clamp(0.0, 1.0)
        thrust_ratio = self._collective_signal_to_thrust_ratio(collective_signal)
        self._thrust[:, 0, 2] = (
            thrust_ratio
            * self._F_max
            * self._collective_efficiency
        )

        rate_meas_b = self._robot.data.root_ang_vel_b
        rate_error = rate_sp_isaac - rate_meas_b
        moment_cmd = self._ctbr_rate_kp_per_env * rate_error * self._moment_efficiency
        self._moment[:, 0, :] = torch.clamp(moment_cmd, -self._ctbr_moment_limit, self._ctbr_moment_limit)
        self._ctbr_rate_cmd = rate_sp_isaac
        self._ctbr_rate_meas = rate_meas_b
        self._ctbr_rate_error = rate_error

        # [新增] wind state update (OU + gust), store wind accel in world frame
        self._wind_step(self.step_dt)

    def _apply_action(self):
        # Keep the original direct wrench path when all disturbance gaps are off.
        if not getattr(self, "_disturbance_enabled", False):
            self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
            return

        # 组合外力/外矩：body_ids = [uav_body, payload_body(可选)]
        body_ids = self._ext_body_ids
        forces = self._ext_forces_buf
        torques = self._ext_torques_buf
        forces.zero_()
        torques.zero_()

        # --- UAV: thrust/moment (body frame) ---
        forces[:, 0, :] = self._thrust[:, 0, :]
        torques[:, 0, :] = self._moment[:, 0, :]

        # --- Wind: compute in world frame -> rotate into each body's frame ---
        ambient_acc_w = self._wind_acc_w  # (N,3) world
        startup_acc_w = self._startup_acc_w  # (N,3) world
        if self._wind_apply_to_uav:
            F_uav_w = self._uav_mass_tensor.unsqueeze(-1) * (
                ambient_acc_w * self._wind_scale_uav
                + startup_acc_w * self._startup_gust_uav_scale
            )
        else:
            F_uav_w = self._uav_mass_tensor.unsqueeze(-1) * (
                startup_acc_w * self._startup_gust_uav_scale
            )

        # payload force uses per-env payload mass buffer
        if len(body_ids) > 1:
            F_pay_w = self._payload_mass.unsqueeze(-1) * (
                ambient_acc_w * (self._wind_scale_payload if self._wind_apply_to_payload else 0.0)
                + startup_acc_w * self._startup_gust_payload_scale
            )
        else:
            F_pay_w = None

        # quaternions (world)
        quat_uav_w = self._robot.data.root_quat_w  # (N,4) wxyz

        # UAV wind: world -> uav body frame
        F_uav_b = self._quat_rotate_inverse(quat_uav_w, F_uav_w)
        forces[:, 0, :] = forces[:, 0, :] + F_uav_b

        # payload wind: world -> payload body frame
        if F_pay_w is not None:
            body_quat_w = getattr(self._robot.data, "body_quat_w", None)
            if body_quat_w is not None:
                quat_pay_w = body_quat_w[:, self._payload_id, :]
            else:
                quat_pay_w = quat_uav_w  # fallback (不会崩，但不够准)

            # Downwash is sampled in the UAV frame. Convert it to world and
            # then to the payload frame before adding it to the payload only.
            downwash_w = self._quat_rotate(quat_uav_w, self._downwash_force_b)
            F_pay_b = self._quat_rotate_inverse(quat_pay_w, F_pay_w + downwash_w)
            forces[:, 1, :] = forces[:, 1, :] + F_pay_b

        # apply (forces/torques are in body frame, consistent with your thrust)
        self._robot.set_external_force_and_torque(forces, torques, body_ids=body_ids)

    def _init_payload_sensor_gap(self) -> None:
        """Allocate per-environment payload-camera transport state."""
        self._payload_sensor_enabled = bool(
            getattr(self.cfg, "enable_payload_sensor_gap", False)
        )
        max_delay_s = max(
            float(getattr(self.cfg, "payload_sensor_nominal_delay_s", (0.0, 0.0))[1]),
            float(getattr(self.cfg, "payload_sensor_tail_delay_s", (0.0, 0.0))[1]),
        )
        self._payload_ring_len = max(2, int(math.ceil(max_delay_s / self.step_dt)) + 2)
        self._payload_clean_ring = torch.zeros(
            self.num_envs, self._payload_ring_len, 5, device=self.device
        )
        self._payload_ring_step = torch.full(
            (self.num_envs, self._payload_ring_len),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._payload_ring_write_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._payload_sensor_elapsed_s = torch.zeros(self.num_envs, device=self.device)
        self._payload_sensor_period_s = torch.full(
            (self.num_envs,), self.step_dt, device=self.device
        )
        self._payload_sensor_delay_s = torch.zeros(self.num_envs, device=self.device)
        self._payload_sensor_delay_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._payload_sensor_next_update_s = torch.zeros(self.num_envs, device=self.device)
        self._payload_sensor_valid_probability = torch.ones(self.num_envs, device=self.device)
        self._payload_sensor_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._payload_sensor_held = torch.zeros(self.num_envs, 5, device=self.device)
        self._payload_sensor_held_rate = torch.zeros(self.num_envs, 2, device=self.device)
        self._payload_sensor_last_source_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._payload_sensor_last_valid_s = torch.zeros(self.num_envs, device=self.device)
        self._payload_sensor_valid_updates = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._payload_sensor_dropouts = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._payload_position_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._payload_angle_bias = torch.zeros(self.num_envs, 2, device=self.device)
        self._attitude_trim_bias_rad = torch.zeros(self.num_envs, 2, device=self.device)
        self._linear_velocity_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._body_rate_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._reset_payload_sensor_gap(self._robot._ALL_INDICES)

    def _reset_payload_sensor_gap(self, env_ids: torch.Tensor) -> None:
        """Reset and sample only the requested payload-camera rows."""
        count = int(env_ids.numel())
        if count == 0:
            return
        device = self.device
        tail = torch.rand(count, device=device) < float(
            getattr(self.cfg, "payload_sensor_tail_probability", 0.0)
        )
        nominal_hz = getattr(self.cfg, "payload_sensor_nominal_hz", (60.0, 60.0))
        tail_hz = getattr(self.cfg, "payload_sensor_tail_hz", nominal_hz)
        nominal_delay = getattr(self.cfg, "payload_sensor_nominal_delay_s", (0.0, 0.0))
        tail_delay = getattr(self.cfg, "payload_sensor_tail_delay_s", nominal_delay)

        hz_lo = torch.where(
            tail,
            torch.full((count,), float(tail_hz[0]), device=device),
            torch.full((count,), float(nominal_hz[0]), device=device),
        )
        hz_hi = torch.where(
            tail,
            torch.full((count,), float(tail_hz[1]), device=device),
            torch.full((count,), float(nominal_hz[1]), device=device),
        )
        sampled_hz = hz_lo + (hz_hi - hz_lo) * torch.rand(count, device=device)
        delay_lo = torch.where(
            tail,
            torch.full((count,), float(tail_delay[0]), device=device),
            torch.full((count,), float(nominal_delay[0]), device=device),
        )
        delay_hi = torch.where(
            tail,
            torch.full((count,), float(tail_delay[1]), device=device),
            torch.full((count,), float(nominal_delay[1]), device=device),
        )
        sampled_delay = delay_lo + (delay_hi - delay_lo) * torch.rand(count, device=device)
        valid_lo, valid_hi = getattr(
            self.cfg, "payload_sensor_valid_probability", (1.0, 1.0)
        )

        self._payload_sensor_period_s[env_ids] = 1.0 / sampled_hz.clamp_min(1e-6)
        self._payload_sensor_delay_s[env_ids] = sampled_delay
        self._payload_sensor_delay_steps[env_ids] = torch.round(
            sampled_delay / self.step_dt
        ).to(torch.long).clamp(0, self._payload_ring_len - 1)
        self._payload_sensor_valid_probability[env_ids] = torch.empty(
            count, device=device
        ).uniform_(float(valid_lo), float(valid_hi))
        self._payload_sensor_elapsed_s[env_ids] = 0.0
        self._payload_sensor_next_update_s[env_ids] = 0.0
        self._payload_sensor_initialized[env_ids] = False
        self._payload_sensor_last_valid_s[env_ids] = 0.0
        self._payload_sensor_valid_updates[env_ids] = 0
        self._payload_sensor_dropouts[env_ids] = 0
        self._payload_sensor_held[env_ids] = 0.0
        self._payload_sensor_held_rate[env_ids] = 0.0
        self._payload_sensor_last_source_step[env_ids] = -1
        self._payload_clean_ring[env_ids] = 0.0
        self._payload_ring_step[env_ids] = -1
        self._payload_ring_write_idx[env_ids] = 0

        pos_lo, pos_hi = getattr(self.cfg, "payload_position_bias_range_m", (0.0, 0.0))
        angle_lo, angle_hi = getattr(self.cfg, "payload_angle_bias_range_deg", (0.0, 0.0))
        trim_lo, trim_hi = getattr(self.cfg, "attitude_trim_bias_range_deg", (0.0, 0.0))
        vel_lo, vel_hi = getattr(self.cfg, "linear_velocity_bias_range_mps", (0.0, 0.0))
        rate_lo, rate_hi = getattr(self.cfg, "body_rate_bias_range_rps", (0.0, 0.0))
        self._payload_position_bias[env_ids] = torch.empty(count, 3, device=device).uniform_(
            float(pos_lo), float(pos_hi)
        )
        self._payload_angle_bias[env_ids] = torch.empty(count, 2, device=device).uniform_(
            float(angle_lo), float(angle_hi)
        )
        self._attitude_trim_bias_rad[env_ids] = torch.empty(count, 2, device=device).uniform_(
            math.radians(float(trim_lo)), math.radians(float(trim_hi))
        )
        self._linear_velocity_bias[env_ids] = torch.empty(count, 3, device=device).uniform_(
            float(vel_lo), float(vel_hi)
        )
        self._body_rate_bias[env_ids] = torch.empty(count, 3, device=device).uniform_(
            float(rate_lo), float(rate_hi)
        )

    def _transport_payload_observation(
        self,
        e_load: torch.Tensor,
        tilt_deg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply camera timing, latency, dropout, hold, and fixed episode bias."""
        if not self._payload_sensor_enabled:
            return e_load, tilt_deg, self._tilt_vel_deg

        rows = torch.arange(self.num_envs, device=self.device)
        write_idx = self._payload_ring_write_idx
        clean = torch.cat([e_load, tilt_deg], dim=-1)
        self._payload_clean_ring[rows, write_idx] = clean
        current_step = torch.round(self._payload_sensor_elapsed_s / self.step_dt).to(torch.long)
        self._payload_ring_step[rows, write_idx] = current_step
        self._payload_sensor_elapsed_s += self.step_dt

        due = self._payload_sensor_elapsed_s >= self._payload_sensor_next_update_s
        due = due | (~self._payload_sensor_initialized)
        valid = torch.rand(self.num_envs, device=self.device) <= self._payload_sensor_valid_probability
        stale = (
            self._payload_sensor_elapsed_s - self._payload_sensor_last_valid_s
            >= float(getattr(self.cfg, "payload_sensor_hold_cap_s", 0.50))
        )
        update = due & (valid | stale | (~self._payload_sensor_initialized))
        dropout = due & (~update)

        delayed = select_delayed_ring(
            self._payload_clean_ring,
            write_idx,
            self._payload_sensor_delay_steps,
        )
        source_idx = (write_idx - self._payload_sensor_delay_steps) % self._payload_ring_len
        source_step = self._payload_ring_step[rows, source_idx]
        source_unavailable = source_step < 0
        first = ~self._payload_sensor_initialized
        use_current = first | source_unavailable
        delayed = torch.where(use_current.unsqueeze(-1), clean, delayed)
        source_step = torch.where(use_current, current_step, source_step)

        measured = delayed.clone()
        measured[:, :3] += self._payload_position_bias
        measured[:, 3:5] += self._payload_angle_bias
        if getattr(self.cfg, "enable_obs_noise", False):
            measured[:, :3] += torch.randn_like(measured[:, :3]) * float(
                self.cfg.obs_noise_e_load_std_m
            )
            measured[:, 3:5] += torch.randn_like(measured[:, 3:5]) * float(
                self.cfg.obs_noise_tilt_std_deg
            )

        source_delta_s = (
            source_step - self._payload_sensor_last_source_step
        ).to(torch.float32) * self.step_dt
        new_rate = torch.where(
            (self._payload_sensor_initialized & (source_delta_s > 1e-6)).unsqueeze(-1),
            (measured[:, 3:5] - self._payload_sensor_held[:, 3:5])
            / source_delta_s.clamp_min(1e-6).unsqueeze(-1),
            torch.zeros_like(self._payload_sensor_held_rate),
        )
        self._payload_sensor_held[update] = measured[update]
        self._payload_sensor_held_rate[update] = new_rate[update]
        self._payload_sensor_last_source_step[update] = source_step[update]
        self._payload_sensor_last_valid_s[update] = self._payload_sensor_elapsed_s[update]
        self._payload_sensor_initialized[update] = True
        self._payload_sensor_valid_updates[update] += 1
        self._payload_sensor_dropouts[dropout] += 1
        self._payload_sensor_next_update_s[due] = (
            self._payload_sensor_elapsed_s[due] + self._payload_sensor_period_s[due]
        )
        self._payload_ring_write_idx = (write_idx + 1) % self._payload_ring_len

        return (
            self._payload_sensor_held[:, :3],
            self._payload_sensor_held[:, 3:5],
            self._payload_sensor_held_rate,
        )

    def _get_observations(self) -> dict:
        """构造 17 维的观察量:

        obs = [
            e_load[0:3],       # 0-2  payload 到目标位置误差 (世界系)
            tilt_deg[0:2],     # 3-4  摆角 θx, θy (deg)
            w_deg[0:2],        # 5-6  摆角角速度 θ̇x, θ̇y (deg/s)
            root_quat_w[0:4],  # 7-10 UAV 根 body 姿态四元数 (世界系)
            v_b[0:3],          # 11-13 UAV 根 body 线速度 (机体系)
            w_b[0:3],          # 14-16 UAV 根 body 角速度 (机体系)
        ]
        """
        # --- 1) 位置相关：UAV / payload / 目标 ------------------------------
        body_pos_w = self._robot.data.body_pos_w  # (num_envs, num_bodies, 3)
        # UAV 根 body 位置（这里 body 取第一个 body_id）
        p_uav_w = body_pos_w[:, self._body_id[0], :]  # (num_envs, 3)
        # payload 位置
        p_load_w = body_pos_w[:, self._payload_id, :]  # (num_envs, 3)

        # payload 到 UAV 的向量（世界系），用于计算摆角
        r_load_uav = p_uav_w - p_load_w  # (num_envs, 3)

        # payload 到目标点的误差（世界系）
        e_load = self._desired_pos_w - p_load_w  # (num_envs, 3)

        # --- 2) 摆角 + 摆角角速度 ----------------------------------------
        # rope 长度，用 cfg 中的参数（标量）
        L = self._rope_lengths.clamp(min=1e-3)

        # 近似：摆角 θx, θy（世界系）—— 和你原来的定义保持一致
        ex = r_load_uav[:, 0]
        ey = r_load_uav[:, 1]
        ez = r_load_uav[:, 2].clamp(min=1e-6)

        # 这里假设绳长 ≈ L，且摆角较小，用水平分量 / L 近似
        theta_x = torch.asin((ex / L).clamp(-1.0, 1.0))
        theta_y = torch.asin((ey / L).clamp(-1.0, 1.0))

        tilt_rad = torch.stack([theta_x, theta_y], dim=-1)  # (num_envs, 2)
        tilt_deg = tilt_rad * (180.0 / math.pi)

        # 初始化历史 buffer（第一次调用时）
        if self._prev_tilt_deg is None:
            self._prev_tilt_deg = torch.zeros_like(tilt_deg)
            self._tilt_vel_deg = torch.zeros_like(tilt_deg)
            self._has_prev_tilt = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

        # 计算角速度（deg/s），用上一帧的摆角做差分
        dt = self.step_dt  # DirectRLEnv 里定义好的 "每次 RL step 对应的物理时间"
        mask_has_prev = self._has_prev_tilt

        delta_tilt = tilt_deg - self._prev_tilt_deg  # (num_envs, 2)
        w_deg = torch.where(
            mask_has_prev.unsqueeze(-1),
            delta_tilt / max(dt, 1e-6),
            torch.zeros_like(delta_tilt),
        )

        # 更新历史
        self._prev_tilt_deg = tilt_deg.clone()
        self._tilt_vel_deg = w_deg.clone()
        self._has_prev_tilt[:] = True

        # --- 3) UAV 姿态 + 线速度 + 角速度 -------------------------------
        # 姿态四元数（世界系）
        root_quat_w = self._robot.data.root_quat_w  # (num_envs, 4)

        # 线速度、角速度（机体系）
        v_b = self._robot.data.root_lin_vel_b  # (num_envs, 3)
        w_b = self._robot.data.root_ang_vel_b  # (num_envs, 3)

        # --- 4) 打包 obs ---------------------------------------------------
        # Policy-only sensor transport. Reward and critic stay on clean state.
        if self._payload_sensor_enabled:
            e_load_obs, tilt_deg_obs, w_deg_obs = self._transport_payload_observation(
                e_load, tilt_deg
            )
            trim_quat = self._quat_from_roll_pitch(self._attitude_trim_bias_rad)
            root_quat_w_obs = self._quat_multiply(root_quat_w, trim_quat)
            root_quat_w_obs = root_quat_w_obs / torch.linalg.norm(
                root_quat_w_obs, dim=-1, keepdim=True
            ).clamp_min(1e-6)
            v_b_obs = v_b + self._linear_velocity_bias
            w_b_obs = w_b + self._body_rate_bias
            if getattr(self.cfg, "enable_obs_noise", False):
                v_b_obs = v_b_obs + torch.randn_like(v_b_obs) * float(
                    self.cfg.obs_noise_v_b_std_mps
                )
                w_b_obs = w_b_obs + torch.randn_like(w_b_obs) * float(
                    self.cfg.obs_noise_w_b_std_rps
                )
        elif getattr(self.cfg, "enable_obs_noise", False):
            e_load_obs = e_load + torch.randn_like(e_load) * float(self.cfg.obs_noise_e_load_std_m)
            tilt_deg_obs = tilt_deg + torch.randn_like(tilt_deg) * float(self.cfg.obs_noise_tilt_std_deg)
            v_b_obs = v_b + torch.randn_like(v_b) * float(self.cfg.obs_noise_v_b_std_mps)
            w_b_obs = w_b + torch.randn_like(w_b) * float(self.cfg.obs_noise_w_b_std_rps)
            root_quat_w_obs = root_quat_w
            w_deg_obs = w_deg
        else:
            e_load_obs = e_load
            tilt_deg_obs = tilt_deg
            w_deg_obs = w_deg
            root_quat_w_obs = root_quat_w
            v_b_obs = v_b
            w_b_obs = w_b

        # noisy obs for policy
        obs_policy = torch.cat(
            [
                e_load_obs,       # 0-2
                tilt_deg_obs,     # 3-4
                w_deg_obs,        # 5-6
                root_quat_w_obs,  # 7-10
                v_b_obs,          # 11-13
                w_b_obs,          # 14-16
                self._last_transmitted_actions, # 17-20: last transmitted clamped CTBR
            ],
            dim=-1,
        )

        # clean obs for critic
        obs_critic = torch.cat(
            [
                e_load,        # 0-2
                tilt_deg,      # 3-4
                w_deg,         # 5-6
                root_quat_w,   # 7-10
                v_b,           # 11-13
                w_b,           # 14-16
                self._last_transmitted_actions, # 17-20: last transmitted clamped CTBR
            ],
            dim=-1,
        )

        # --- Oracle: append true payload mass / rope length / wind ---
        if getattr(self.cfg, "use_oracle_mass_obs", False):
            # A. Mass
            m_norm, l_norm = normalize_physical_context(
                self._payload_mass,
                self._rope_lengths,
                payload_mass_range_kg=self.cfg.payload_mass_range,
                rope_length_range_m=self.cfg.rope_length_range,
            )
            m_norm = m_norm.unsqueeze(-1)
            l_norm = l_norm.unsqueeze(-1)

            # C. Control-relevant residual acceleration in the UAV body frame.
            # It includes ambient/startup acceleration and payload-only downwash.
            downwash_w = self._quat_rotate(root_quat_w, self._downwash_force_b)
            residual_w = (
                self._wind_acc_w * self._wind_scale_payload
                + self._startup_acc_w * self._startup_gust_payload_scale
                + downwash_w / self._payload_mass.clamp_min(1e-6).unsqueeze(-1)
            )
            residual_b = self._quat_rotate_inverse(root_quat_w, residual_w)
            max_residual = float(getattr(self.cfg, "residual_accel_norm_max", 5.5))
            wind_norm = residual_b / max(max_residual, 1e-6)

            obs_policy = torch.cat([obs_policy, m_norm, l_norm, wind_norm], dim=-1)
            obs_critic = torch.cat([obs_critic, m_norm, l_norm, wind_norm], dim=-1)

        return {"policy": obs_policy, "critic": obs_critic}



    def _get_rewards(self) -> torch.Tensor:
        """
        混合奖励函数：位置 + 摆角 + 摆速 + 动作幅值 + 动作连续性 + 自旋惩罚
        与 _get_observations() 使用同源几何定义：
        - UAV 位置: body_pos_w[:, self._body_id[0], :]
        - payload 位置: body_pos_w[:, self._payload_id, :]
        - 摆角: theta_x/theta_y
        - 摆速: self._tilt_vel_deg
        """

        # === 1) 基础数据 ===
        body_pos_w = self._robot.data.body_pos_w
        p_uav_w = body_pos_w[:, self._body_id[0], :]      # 与 obs 对齐
        p_load_w = body_pos_w[:, self._payload_id, :]     # 与 obs 对齐
        goal_payload_w = self._desired_pos_w

        # payload 到目标点误差 (m)
        e_load = goal_payload_w - p_load_w
        dist = torch.linalg.norm(e_load, dim=1)

        # === 2) 摆角 / 摆速（与 obs 完全同源）===
        r_load_uav = p_uav_w - p_load_w
        L = self._rope_lengths.clamp(min=1e-3)

        theta_x = torch.asin((r_load_uav[:, 0] / L).clamp(-1.0, 1.0))
        theta_y = torch.asin((r_load_uav[:, 1] / L).clamp(-1.0, 1.0))

        # 合摆角：rad / deg
        theta_rad = torch.sqrt(theta_x * theta_x + theta_y * theta_y)
        theta_deg = theta_rad * (180.0 / math.pi)

        # 合摆速：clean 的 deg/s（来自 clean 摆角差分）
        wx_deg = self._tilt_vel_deg[:, 0]
        wy_deg = self._tilt_vel_deg[:, 1]
        swing_deg_s = torch.sqrt(wx_deg * wx_deg + wy_deg * wy_deg)

        # 归一化摆能量诊断量（不进 reward，只做 logging）
        # E_hat = 0.5 * (||theta_dot||^2 + (g/L) * ||theta||^2)
        g = 9.81
        theta_dot_rad_s = swing_deg_s * (math.pi / 180.0)
        E_hat = 0.5 * (
            theta_dot_rad_s * theta_dot_rad_s
            + (g / torch.clamp(L, min=1e-6)) * (theta_rad * theta_rad)
        )
        E_hat_mean_dt = E_hat * self.step_dt

        # === 3) 位置奖励 ===
        r_alive = 4.0
        r_dist_dense = -1.0 * dist
        r_dist_gauss = torch.exp(-0.5 * (dist / self.cfg.sigma_pos) ** 2)

        r_pos_val = float(self.cfg.pos_weight) * (
            r_alive + r_dist_dense + 2.0 * r_dist_gauss
        )

        # === 4) 摆角 / 摆速惩罚 ===
        r_tilt_val = -1.0 * float(self.cfg.tilt_weight) * (
            theta_deg / self.cfg.sigma_tilt_deg
        ) ** 2

        r_swing_val = -0.1 * float(self.cfg.tilt_weight) * (
            swing_deg_s / self.cfg.sigma_swing_deg_s
        ) ** 2

        # === 5) 机体状态与实际发送动作惩罚 ===
        tilt_deg_uav = torch.rad2deg(uav_tilt_rad_wxyz(self._robot.data.root_quat_w))
        actual_rate_norm = self._robot.data.root_ang_vel_b / self._ctbr_body_rate_limit
        r_uav_tilt = -float(self.cfg.uav_tilt_penalty_scale) * torch.square(
            tilt_deg_uav / float(self.cfg.uav_tilt_normalization_deg)
        )
        r_actual_rate = -float(self.cfg.actual_body_rate_penalty_scale) * torch.sum(
            actual_rate_norm.square(), dim=1
        )

        transmitted_jerk = self._transmitted_delta - self._prev_transmitted_delta
        sent_norm, delta_norm, jerk_norm = normalized_ctbr_terms(
            self._last_transmitted_actions,
            self._transmitted_delta,
            transmitted_jerk,
            self._ctbr_action_scale,
        )
        r_action_l2 = -float(self.cfg.action_l2_penalty_scale) * torch.sum(
            sent_norm.square(), dim=1
        )
        r_action_smooth = -float(self.cfg.action_smooth_penalty_scale) * torch.sum(
            delta_norm.square(), dim=1
        )
        r_action_jerk = -float(self.cfg.action_jerk_penalty_scale) * torch.sum(
            jerk_norm.square(), dim=1
        )

        a0_low_excess = torch.relu(-1.0 - self._policy_actions[:, 0])
        a0_high_excess = torch.relu(self._policy_actions[:, 0])
        rate_excess = torch.relu(torch.abs(self._policy_actions[:, 1:4]) - self._ctbr_body_rate_limit)
        rate_excess = rate_excess / self._ctbr_body_rate_limit
        raw_excess = torch.cat((a0_low_excess.unsqueeze(1), a0_high_excess.unsqueeze(1), rate_excess), dim=1)
        r_action_raw_val = -float(self.cfg.action_raw_excess_penalty_scale) * torch.sum(torch.square(raw_excess), dim=1)

        r_action_total = r_action_l2 + r_action_smooth + r_action_jerk + r_action_raw_val

        # === 6) 死亡惩罚 ===
        root_pos = self._robot.data.root_pos_w
        env_origins = self._terrain.env_origins.to(root_pos.device)

        height_fail = torch.logical_or(root_pos[:, 2] < 0.1, root_pos[:, 2] > 6.0)
        rel_pos = root_pos - env_origins
        out_of_box = torch.any(torch.abs(rel_pos) > 6.0, dim=1)

        tilt_fail = tilt_deg_uav > float(self.cfg.uav_tilt_termination_deg)
        died = torch.logical_or(torch.logical_or(height_fail, out_of_box), tilt_fail)
        death_penalty_vec = -1.0 * float(self.cfg.death_penalty) * died.float()

        # === 7) 自旋惩罚 ===
        quat = self._robot.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw_angle = torch.atan2(
            2 * (w * z + x * y),
            1 - 2 * (y * y + z * z)
        )
        yaw_rate = self._robot.data.root_ang_vel_b[:, 2]

        r_spin_val = -1.0 * float(self.cfg.spin_weight) * (
            torch.square(yaw_rate) + torch.square(yaw_angle)
        )

        # === 8) 总奖励 ===
        reward = (
            r_pos_val
            + r_tilt_val
            + r_swing_val
            + death_penalty_vec
            + r_action_total
            + r_spin_val
            + r_uav_tilt
            + r_actual_rate
        )

        # === 9) Logging ===
        rewards_dict = {
            "r_pos": r_pos_val,
            "r_tilt": r_tilt_val,
            "r_spin": r_spin_val,
            "r_swing": r_swing_val,
            "death_penalty": death_penalty_vec,
            "r_action_raw": r_action_raw_val,
            "r_action_l2": r_action_l2,
            "r_action_smooth": r_action_smooth,
            "r_action_jerk": r_action_jerk,
            "r_uav_tilt": r_uav_tilt,
            "r_actual_rate": r_actual_rate,
            "action_raw_sum": torch.sum(torch.abs(self._raw_actions), dim=1),
            "action_policy_raw_sum": torch.sum(torch.abs(self._policy_actions), dim=1),
            "dist": dist,
            "theta_deg": theta_deg,
            "swing_deg_s": swing_deg_s,
            "E_hat_mean": E_hat_mean_dt,
            "time_penalty": torch.zeros_like(reward),
            "total": reward,
        }

        for key, value in rewards_dict.items():
            if key in self._episode_sums:
                self._episode_sums[key] += value

        # === 10) 更新 smoothness 历史动作 ===
        self._prev_actions = self._actions.clone()
        self._prev_transmitted_delta = self._transmitted_delta.clone()

        return reward



    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            # 时间到：和原来一样
            time_out = self.episode_length_buf >= self.max_episode_length - 1

            # UAV 当前世界坐标
            root_pos = self._robot.data.root_pos_w  # (num_envs, 3)

            # 1) 高度越界：低于 0.1 或 高于 5.0（保留原规则）
            height_fail = torch.logical_or(root_pos[:, 2] < 0.1, root_pos[:, 2] > 6.0)

            # 2) 相对各自 env 原点的越界：任一坐标绝对值 > 5.0 m
            #    env_spacing = 6.0，因此 ±5m 仍然在自己这一格内
            env_origins = self._terrain.env_origins.to(root_pos.device)  # (num_envs, 3)
            rel_pos = root_pos - env_origins                              # 以各自 env 原点为参考
            out_of_box = torch.any(torch.abs(rel_pos) > 6.0, dim=1)

            tilt_deg_uav = torch.rad2deg(
                uav_tilt_rad_wxyz(self._robot.data.root_quat_w)
            )
            tilt_fail = tilt_deg_uav > float(self.cfg.uav_tilt_termination_deg)

            # died = 高度越界、出盒子或机体倾角超过安全边界
            died = torch.logical_or(torch.logical_or(height_fail, out_of_box), tilt_fail)

            return died, time_out

    def _sample_action_transport(self, env_ids: torch.Tensor) -> None:
        """Sample provisional actuator/link parameters once per episode."""
        count = int(env_ids.numel())
        if count == 0:
            return
        delay_lo, delay_hi = getattr(
            self.cfg,
            "action_delay_steps_range",
            (int(self.cfg.action_delay_steps), int(self.cfg.action_delay_steps)),
        )
        self._action_delay_steps_per_env[env_ids] = torch.randint(
            int(delay_lo), int(delay_hi) + 1, (count,), device=self.device
        )
        alpha_lo, alpha_hi = getattr(
            self.cfg, "action_lpf_alpha_range", (self.cfg.action_lpf_alpha, self.cfg.action_lpf_alpha)
        )
        collective_lo, collective_hi = getattr(
            self.cfg, "collective_efficiency_range", (1.0, 1.0)
        )
        moment_lo, moment_hi = getattr(self.cfg, "moment_efficiency_range", (1.0, 1.0))
        self._action_lpf_alpha_per_env[env_ids] = torch.empty(
            count, 1, device=self.device
        ).uniform_(float(alpha_lo), float(alpha_hi))
        self._collective_efficiency[env_ids] = torch.empty(count, device=self.device).uniform_(
            float(collective_lo), float(collective_hi)
        )
        self._moment_efficiency[env_ids] = torch.empty(count, 1, device=self.device).uniform_(
            float(moment_lo), float(moment_hi)
        )
        position_bias_center = torch.as_tensor(
            self.cfg.position_history_rate_bias_center_rps,
            device=self.device,
            dtype=torch.float32,
        )
        position_bias_jitter = torch.as_tensor(
            self.cfg.position_history_rate_bias_jitter_rps,
            device=self.device,
            dtype=torch.float32,
        )
        self._position_history_rate_bias_px4[env_ids] = (
            position_bias_center.unsqueeze(0)
            + (2.0 * torch.rand(count, 3, device=self.device) - 1.0)
            * position_bias_jitter.unsqueeze(0)
        )

    def _fill_action_transport(self, env_ids: torch.Tensor, hover_actions: torch.Tensor) -> None:
        """Initialize every selected queue slot and filter state at hover."""
        self._action_queue[env_ids] = hover_actions.unsqueeze(1).expand(
            -1, self._action_max_delay_steps + 1, -1
        )
        self._filtered_actions[env_ids] = hover_actions


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        # 必须在 reset 物理状态之前记录，否则位置都被重置了，永远查不到死因
        self._log_termination_stats(env_ids)

        # 1. 重置 Robot (清除之前的速度、力等)
        self._robot.reset(env_ids)

        # Episode-level UAV property randomization around the measured nominal.
        self._apply_uav_physics(env_ids, randomize=True)
        self._sample_body_rate_dynamics(env_ids)

        # 2. 获取默认状态
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        # 3. 绳长随机化 & 设置关节目标 (核心修复!)
        if hasattr(self.cfg, "rope_length_range"):
            lo_len, hi_len = self.cfg.rope_length_range
            fixed_rope_length = getattr(
                self.cfg, "eval_fixed_rope_length_m", None
            )
            if fixed_rope_length is None:
                L = (
                    torch.rand(len(env_ids), device=self.device)
                    * (hi_len - lo_len)
                    + lo_len
                )
            else:
                L = torch.full(
                    (len(env_ids),),
                    float(fixed_rope_length),
                    device=self.device,
                )
            self._rope_lengths[env_ids] = L

            # (A) 设置状态：告诉物理引擎现在绳子有多长
            target_pos = -1.0 * L
            joint_pos[:, self._rope_joint_idx] = target_pos
            joint_vel[:, self._rope_joint_idx] = 0.0

            # (B) 【关键修复】设置 Drive Target：告诉弹簧“你就停在这个长度，别乱拉”
            # 注意：set_joint_position_target 需要 (num_envs, 1) 的维度
            self._robot.set_joint_position_target(
                target_pos.view(-1, 1),
                env_ids=env_ids,
                joint_ids=[self._rope_joint_idx]
            )
        else:
            # 如果没有随机化配置，给个默认值防止报错
            self._rope_lengths[env_ids] = 0.8
        # 5. Lumped suspended mass: moving gimbal + tag + rope(L) + ballast.
        if hasattr(self.cfg, "payload_mass_range"):
            env_ids_cpu = env_ids.to("cpu")
            # Read the current arrays so the UAV randomization above is retained.
            masses = self._robot.root_physx_view.get_masses().clone()
            inertias = self._robot.root_physx_view.get_inertias().clone()
            fixed_payload_mass = getattr(
                self.cfg, "eval_fixed_payload_mass_kg", None
            )
            if fixed_payload_mass is None:
                ballast_lo, ballast_hi = self.cfg.payload_ballast_mass_range
                ballast_mass = torch.empty(
                    (len(env_ids),), device=self.device
                ).uniform_(float(ballast_lo), float(ballast_hi))
                new_mass_device, rope_mass = compose_lumped_payload_mass(
                    self._rope_lengths[env_ids],
                    ballast_mass,
                    rope_length_range_m=self.cfg.rope_length_range,
                    rope_mass_range_kg=self.cfg.rope_mass_range_kg,
                    fixed_moving_mass_kg=float(self.cfg.payload_fixed_moving_mass_kg),
                )
            else:
                new_mass_device = torch.full(
                    (len(env_ids),),
                    float(fixed_payload_mass),
                    device=self.device,
                )
                _, rope_mass = compose_lumped_payload_mass(
                    self._rope_lengths[env_ids],
                    torch.zeros_like(new_mass_device),
                    rope_length_range_m=self.cfg.rope_length_range,
                    rope_mass_range_kg=self.cfg.rope_mass_range_kg,
                    fixed_moving_mass_kg=float(self.cfg.payload_fixed_moving_mass_kg),
                )
                ballast_mass = (
                    new_mass_device
                    - float(self.cfg.payload_fixed_moving_mass_kg)
                    - rope_mass
                )
            new_mass = new_mass_device.to("cpu")
            masses[env_ids_cpu, self._payload_id] = new_mass
            self._robot.root_physx_view.set_masses(masses, env_ids_cpu)
            # Preserve payload shape and density assumption by scaling inertia
            # linearly from the nominal payload mass.
            default_payload_mass = self._default_masses_cpu[env_ids_cpu, self._payload_id].clamp_min(1e-6)
            mass_ratio = new_mass / default_payload_mass
            inertias[env_ids_cpu, self._payload_id] = (
                self._default_inertias_cpu[env_ids_cpu, self._payload_id]
                * mass_ratio.unsqueeze(1)
            )
            self._robot.root_physx_view.set_inertias(inertias, env_ids_cpu)
            # 记录到 buffer
            self._payload_mass[env_ids] = new_mass_device
            self._payload_ballast_mass[env_ids] = ballast_mass
            self._rope_mass[env_ids] = rope_mass
            self._robot_weight[env_ids] = masses[env_ids_cpu].sum(dim=1).to(self.device) * self._gravity_magnitude

        # === 6.【关键修改】随机化完参数后，立刻记本局参数 ===
        # 确保记录的是新一局的真实质量和绳长
        self._log_task_config(env_ids)
        self._reset_wind(env_ids)
        self._reset_payload_sensor_gap(env_ids)
        # 4. 计算出生位置 (使用 Config 里的 start_pos_w)
        # 加上 env_origins，让无人机分散开，不要叠在一起
        env_origins = self._terrain.env_origins[env_ids]
        default_root_state[:, :3] = env_origins + self._start_offset
        # # 设置目标点
        self._desired_pos_w[env_ids] = env_origins + self._goal_offset
        if self.cfg.goal_z_subtract_rope_length:
            self._desired_pos_w[env_ids, 2] -= self._rope_lengths[env_ids]

        # 5. 写入物理引擎
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 6. 重置历史 Buffer (用于摆角速度计算)
        if isinstance(self._prev_tilt_deg, torch.Tensor):
            self._prev_tilt_deg[env_ids] = 0.0
            self._tilt_vel_deg[env_ids] = 0.0
            self._has_prev_tilt[env_ids] = False
        if isinstance(self._obs_prev_tilt_deg, torch.Tensor):
            self._obs_prev_tilt_deg[env_ids] = 0.0
        if isinstance(self._obs_has_prev_tilt, torch.Tensor):
            self._obs_has_prev_tilt[env_ids] = False
        if isinstance(self._obs_w_deg_filt, torch.Tensor):
            self._obs_w_deg_filt[env_ids] = 0.0
        if isinstance(self._obs_has_prev_w, torch.Tensor):
            self._obs_has_prev_w[env_ids] = False
        # 8. 父类逻辑 (必须调用)
        super()._reset_idx(env_ids)

        # 9. 错峰 Reset (Spread out)
        if len(env_ids) == self.num_envs and self.num_envs > 1:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf,
                high=int(self.max_episode_length),
            )
        else:
            self.episode_length_buf[env_ids] = 0
        # [新增]warm
        self._sample_action_transport(env_ids)
        hover_actions = torch.zeros((len(env_ids), self._actions.shape[-1]), device=self.device)
        F_hover = self._robot_weight[env_ids]
        hover_thrust_ratio = (
            F_hover / (float(self._F_max) * self._collective_efficiency[env_ids])
        ).clamp(0.0, 1.0)
        hover_actions[:, 0] = -self._thrust_ratio_to_collective_signal(hover_thrust_ratio)
        hover_actions[:, 1:4] = 0.0

        self._raw_actions[env_ids] = hover_actions
        self._prev_raw_actions[env_ids] = hover_actions
        self._policy_actions[env_ids] = hover_actions
        self._prev_policy_actions[env_ids] = hover_actions
        self._last_transmitted_actions[env_ids] = hover_actions
        self._prev_transmitted_actions[env_ids] = hover_actions
        self._transmitted_delta[env_ids] = 0.0
        self._prev_transmitted_delta[env_ids] = 0.0

        self._fill_action_transport(env_ids, hover_actions)
        self._actions[env_ids] = hover_actions
        self._prev_actions[env_ids] = hover_actions
        self._ctbr_thrust_body_z[env_ids] = hover_actions[:, 0]
        self._ctbr_rate_cmd[env_ids] = 0.0
        self._ctbr_rate_meas[env_ids] = 0.0
        self._ctbr_rate_error[env_ids] = 0.0

    # --- 新增辅助函数 1：记录结束状态 (放 Reset 前) ---
    def _log_termination_stats(self, env_ids):
        extras = dict()
        # 1. Final Distance
        p_load_w = self._robot.data.body_pos_w[env_ids, self._payload_id, :]
        goal_w = self._desired_pos_w[env_ids]
        final_dist = torch.linalg.norm(goal_w - p_load_w, dim=1).mean()

        # 2. Died Reason
        root_pos = self._robot.data.root_pos_w[env_ids]
        env_origins = self._terrain.env_origins[env_ids]

        height_fail = torch.logical_or(root_pos[:, 2] < 0.1, root_pos[:, 2] > 6.0)
        out_of_box = torch.any(torch.abs(root_pos - env_origins) > 6.0, dim=1)
        # 遍历所有累加器，计算平均值 (Total Sum / Max Time)
        for key in self._episode_sums.keys():
            # 取出这些 env 的累加值
            avg = torch.mean(self._episode_sums[key][env_ids])

            # 写入 Log (加前缀 Episode_Reward 以便在 TB 里分类显示)
            extras["Episode_Reward/" + key] = avg / self.max_episode_length_s

            # 【重要】清零！否则下个 episode 会无限累加
            self._episode_sums[key][env_ids] = 0.0

        extras["Metrics/final_distance_to_goal"] = final_dist.item()
        extras["Debug/died_height"] = int(torch.count_nonzero(height_fail).item())
        extras["Debug/died_out_of_box"] = int(torch.count_nonzero(out_of_box).item())

        if "log" not in self.extras: self.extras["log"] = dict()
        self.extras["log"].update(extras)

    # --- 新增辅助函数 2：记录任务配置 (放 Reset 后) ---
    def _log_task_config(self, env_ids):
        m = self._payload_mass[env_ids]
        l = self._rope_lengths[env_ids]
        ballast = self._payload_ballast_mass[env_ids]
        rope_mass = self._rope_mass[env_ids]

        extras = dict()
        extras["Metrics/payload_mass_true_mean"] = float(m.mean().item())
        extras["Metrics/payload_mass_true_min"]  = float(m.min().item())
        extras["Metrics/payload_mass_true_max"]  = float(m.max().item())
        extras["Metrics/payload_ballast_mass_mean"] = float(ballast.mean().item())
        extras["Metrics/rope_mass_mean"] = float(rope_mass.mean().item())

        extras["Metrics/rope_length_mean"] = float(l.mean().item())
        extras["Metrics/rope_length_min"]  = float(l.min().item())
        extras["Metrics/rope_length_max"]  = float(l.max().item())

        if "log" not in self.extras: self.extras["log"] = dict()
        self.extras["log"].update(extras)

    def get_real_hover_gap_audit(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of configured and realized gaps."""
        def _list(name: str, default):
            value = getattr(self.cfg, name, default)
            return [float(item) for item in value]

        def _scalar(name: str, default: float) -> float:
            return float(getattr(self.cfg, name, default))

        audit = {
            "real_hover_gap_profile": str(
                getattr(self.cfg, "real_hover_gap_profile", "disabled")
            ),
            "enable_real_hover_gap": bool(
                getattr(self.cfg, "enable_real_hover_gap", False)
            ),
            "observation_history_action_source": "last_transmitted_clamped_ctbr",
            "uav_mass_kg": float(getattr(self.cfg, "uav_mass_kg", self._uav_mass)),
            "uav_com_m": _list("uav_com_m", (0.0, 0.0, 0.0)),
            "uav_inertia_diag_kg_m2": _list(
                "uav_inertia_diag_kg_m2", (0.0, 0.0, 0.0)
            ),
            "uav_mass_scale_range": _list("uav_mass_scale_range", (1.0, 1.0)),
            "uav_com_offset_range_m": _list("uav_com_offset_range_m", (0.0, 0.0)),
            "uav_inertia_scale_range": _list("uav_inertia_scale_range", (1.0, 1.0)),
            "wind_mean_accel_max": _scalar("wind_mean_accel_max", 0.0),
            "wind_gust_accel_max": _scalar("wind_gust_accel_max", 0.0),
            "wind_total_accel_max": _scalar("wind_total_accel_max", 0.0),
            "wind_gust_duration_s": [
                _scalar("wind_gust_dt_min", 0.0),
                _scalar("wind_gust_dt_max", 0.0),
            ],
            "wind_ou_theta": _scalar("wind_ou_theta", 0.0),
            "wind_ou_sigma": _scalar("wind_ou_sigma", 0.0),
            "enable_wind": bool(getattr(self.cfg, "enable_wind", False)),
            "payload_sensor_nominal_hz": _list(
                "payload_sensor_nominal_hz", (60.0, 60.0)
            ),
            "payload_sensor_tail_hz": _list(
                "payload_sensor_tail_hz", (60.0, 60.0)
            ),
            "payload_sensor_tail_probability": _scalar(
                "payload_sensor_tail_probability", 0.0
            ),
            "payload_sensor_nominal_delay_s": _list(
                "payload_sensor_nominal_delay_s", (0.0, 0.0)
            ),
            "payload_sensor_tail_delay_s": _list(
                "payload_sensor_tail_delay_s", (0.0, 0.0)
            ),
            "payload_sensor_valid_probability": _list(
                "payload_sensor_valid_probability", (1.0, 1.0)
            ),
            "payload_sensor_hold_cap_s": _scalar("payload_sensor_hold_cap_s", 0.0),
            "enable_payload_sensor_gap": bool(
                getattr(self.cfg, "enable_payload_sensor_gap", False)
            ),
            "payload_position_bias_range_m": _list(
                "payload_position_bias_range_m", (0.0, 0.0)
            ),
            "payload_angle_bias_range_deg": _list(
                "payload_angle_bias_range_deg", (0.0, 0.0)
            ),
            "attitude_trim_bias_range_deg": _list(
                "attitude_trim_bias_range_deg", (0.0, 0.0)
            ),
            "linear_velocity_bias_range_mps": _list(
                "linear_velocity_bias_range_mps", (0.0, 0.0)
            ),
            "body_rate_bias_range_rps": _list(
                "body_rate_bias_range_rps", (0.0, 0.0)
            ),
            "startup_gust_accel_range_mps2": _list(
                "startup_gust_accel_range_mps2", (0.0, 0.0)
            ),
            "enable_startup_gust": bool(
                getattr(self.cfg, "enable_startup_gust", False)
            ),
            "startup_gust_duration_range_s": _list(
                "startup_gust_duration_range_s", (0.0, 0.0)
            ),
            "downwash_bias_force_range_n": _list(
                "downwash_bias_force_range_n", (0.0, 0.0)
            ),
            "enable_payload_downwash": bool(
                getattr(self.cfg, "enable_payload_downwash", False)
            ),
            "downwash_ou_sigma_n_sqrt_s": _scalar(
                "downwash_ou_sigma_n_sqrt_s", 0.0
            ),
            "downwash_force_clip_n": _scalar("downwash_force_clip_n", 0.0),
            "residual_accel_norm_max": _scalar("residual_accel_norm_max", 1.0),
            "payload_ballast_mass_range": _list(
                "payload_ballast_mass_range", (0.0, 0.0)
            ),
            "payload_mass_range": _list("payload_mass_range", (0.0, 0.0)),
            "rope_length_range": _list("rope_length_range", (0.0, 0.0)),
            "payload_fixed_moving_mass_kg": float(
                getattr(self.cfg, "payload_fixed_moving_mass_kg", 0.0)
            ),
            "rope_mass_range_kg": _list("rope_mass_range_kg", (0.0, 0.0)),
            "ctbr_thrust_model": str(getattr(self.cfg, "ctbr_thrust_model", "linear")),
            "ctbr_thrust_curve_coeffs": _list(
                "ctbr_thrust_curve_coeffs", (0.0, 1.0, 0.0)
            ),
            "ctbr_rate_time_constant_range_s": [
                [float(item) for item in pair]
                for pair in getattr(self.cfg, "ctbr_rate_time_constant_range_s", ())
            ],
            "action_delay_steps_range": [
                int(item)
                for item in getattr(self.cfg, "action_delay_steps_range", (0, 0))
            ],
            "action_lpf_alpha_range": _list("action_lpf_alpha_range", (1.0, 1.0)),
            "collective_efficiency_range": _list(
                "collective_efficiency_range", (1.0, 1.0)
            ),
            "moment_efficiency_range": _list(
                "moment_efficiency_range", (1.0, 1.0)
            ),
            "realized": {
                "payload_sensor_hz_mean": float(
                    (1.0 / self._payload_sensor_period_s.clamp_min(1e-6)).mean().item()
                ),
                "payload_sensor_hz_min": float(
                    (1.0 / self._payload_sensor_period_s.clamp_min(1e-6)).min().item()
                ),
                "payload_sensor_hz_max": float(
                    (1.0 / self._payload_sensor_period_s.clamp_min(1e-6)).max().item()
                ),
                "payload_sensor_delay_s_mean": float(
                    self._payload_sensor_delay_s.mean().item()
                ),
                "payload_sensor_delay_s_min": float(self._payload_sensor_delay_s.min().item()),
                "payload_sensor_delay_s_max": float(self._payload_sensor_delay_s.max().item()),
                "payload_sensor_valid_probability_mean": float(
                    self._payload_sensor_valid_probability.mean().item()
                ),
                "payload_sensor_valid_updates": int(
                    self._payload_sensor_valid_updates.sum().item()
                ),
                "payload_sensor_dropouts": int(self._payload_sensor_dropouts.sum().item()),
                "startup_amplitude_mean_mps2": float(self._startup_amplitude.mean().item()),
                "startup_duration_mean_s": float(self._startup_duration_s.mean().item()),
                "downwash_force_mean_n": float(
                    torch.linalg.norm(self._downwash_force_b, dim=-1).mean().item()
                ),
                "action_delay_steps_mean": float(
                    self._action_delay_steps_per_env.float().mean().item()
                ),
                "action_lpf_alpha_mean": float(self._action_lpf_alpha_per_env.mean().item()),
                "collective_efficiency_mean": float(self._collective_efficiency.mean().item()),
                "moment_efficiency_mean": float(self._moment_efficiency.mean().item()),
                "ctbr_rate_time_constant_mean_s": [
                    float(item)
                    for item in self._ctbr_rate_time_constant_s.mean(dim=0).tolist()
                ],
                "ctbr_rate_time_constant_min_s": [
                    float(item)
                    for item in self._ctbr_rate_time_constant_s.min(dim=0).values.tolist()
                ],
                "ctbr_rate_time_constant_max_s": [
                    float(item)
                    for item in self._ctbr_rate_time_constant_s.max(dim=0).values.tolist()
                ],
                "payload_ballast_mass_mean_kg": float(
                    self._payload_ballast_mass.mean().item()
                ),
                "rope_mass_mean_kg": float(self._rope_mass.mean().item()),
            },
        }
        return audit


    # ---------------------------------------------------------------------
    # Wind disturbance module (optional)
    #   - OU smooth noise + piecewise constant gust
    #   - wind modeled as world-frame acceleration (m/s^2)
    #   - converted to force via F = m * a, then rotated into body frame
    # ---------------------------------------------------------------------

    def _init_wind_module(self):
        self._wind_enabled = bool(
            getattr(self.cfg, "enable_wind", False)
            and not getattr(self.cfg, "eval_disable_wind", False)
        )
        self._startup_gust_enabled = bool(getattr(self.cfg, "enable_startup_gust", False))
        self._downwash_enabled = bool(getattr(self.cfg, "enable_payload_downwash", False))
        self._disturbance_enabled = bool(
            self._wind_enabled or self._startup_gust_enabled or self._downwash_enabled
        )

        # always create buffer to avoid attribute errors
        self._wind_acc_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._eval_wind_elapsed_s = torch.zeros(self.num_envs, device=self.device)
        self._startup_acc_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._startup_direction_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._startup_amplitude = torch.zeros(self.num_envs, device=self.device)
        self._startup_duration_s = torch.zeros(self.num_envs, device=self.device)
        self._startup_elapsed_s = torch.zeros(self.num_envs, device=self.device)
        self._downwash_bias_force_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._downwash_ou_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._downwash_force_b = torch.zeros(self.num_envs, 3, device=self.device)

        # switches
        self._wind_apply_to_uav = bool(getattr(self.cfg, "wind_apply_to_uav", True))
        self._wind_apply_to_payload = bool(getattr(self.cfg, "wind_apply_to_payload", True))
        self._wind_axis = str(getattr(self.cfg, "wind_axis", "xy")).lower()

        # params
        self._wind_mean_accel_max = float(getattr(self.cfg, "wind_mean_accel_max", 0.5))
        self._wind_gust_accel_max = float(getattr(self.cfg, "wind_gust_accel_max", 1.5))
        self._wind_total_accel_max = float(getattr(self.cfg, "wind_total_accel_max", 3.0))
        self._wind_gust_dt_min = float(getattr(self.cfg, "wind_gust_dt_min", 0.5))
        self._wind_gust_dt_max = float(getattr(self.cfg, "wind_gust_dt_max", 2.0))
        self._wind_ou_theta = float(getattr(self.cfg, "wind_ou_theta", 1.0))
        self._wind_ou_sigma = float(getattr(self.cfg, "wind_ou_sigma", 1.0))
        self._wind_scale_uav = float(getattr(self.cfg, "wind_scale_uav", 0.4))
        self._wind_scale_payload = float(getattr(self.cfg, "wind_scale_payload", 1.0))
        self._startup_gust_uav_scale = float(getattr(self.cfg, "startup_gust_uav_scale", 0.4))
        self._startup_gust_payload_scale = float(getattr(self.cfg, "startup_gust_payload_scale", 1.0))
        self._downwash_ou_sigma = float(getattr(self.cfg, "downwash_ou_sigma_n_sqrt_s", 0.15))
        self._downwash_ou_theta = float(getattr(self.cfg, "downwash_ou_theta", 1.0))
        self._downwash_force_clip = float(getattr(self.cfg, "downwash_force_clip_n", 1.2))
        self._eval_wind_scale = float(getattr(self.cfg, "eval_wind_scale", 1.0))
        if not math.isfinite(self._eval_wind_scale) or self._eval_wind_scale < 0.0:
            raise ValueError(
                "eval_wind_scale must be finite and non-negative, got "
                f"{self._eval_wind_scale!r}."
            )
        self._eval_wind_mode = str(
            getattr(self.cfg, "eval_wind_mode", "training")
        ).strip().lower()
        if self._eval_wind_mode not in {"training", "sinusoid"}:
            raise ValueError(
                "eval_wind_mode must be 'training' or 'sinusoid', got "
                f"{self._eval_wind_mode!r}."
            )
        self._eval_wind_amplitude_mps2 = float(
            getattr(self.cfg, "eval_wind_amplitude_mps2", 1.0)
        )
        self._eval_wind_frequency_hz = float(
            getattr(self.cfg, "eval_wind_frequency_hz", 1.0)
        )
        self._eval_wind_start_sec = float(
            getattr(self.cfg, "eval_wind_start_sec", 3.0)
        )
        self._eval_wind_axis = str(
            getattr(self.cfg, "eval_wind_axis", "x")
        ).strip().lower()
        self._eval_wind_phase_rad = float(
            getattr(self.cfg, "eval_wind_phase_rad", 0.0)
        )
        if (
            not math.isfinite(self._eval_wind_amplitude_mps2)
            or self._eval_wind_amplitude_mps2 < 0.0
        ):
            raise ValueError("eval_wind_amplitude_mps2 must be finite and non-negative.")
        if (
            not math.isfinite(self._eval_wind_frequency_hz)
            or self._eval_wind_frequency_hz <= 0.0
        ):
            raise ValueError("eval_wind_frequency_hz must be finite and positive.")
        if (
            not math.isfinite(self._eval_wind_start_sec)
            or self._eval_wind_start_sec < 0.0
        ):
            raise ValueError("eval_wind_start_sec must be finite and non-negative.")
        if self._eval_wind_axis not in {"x", "y"}:
            raise ValueError("eval_wind_axis must be 'x' or 'y'.")
        if not math.isfinite(self._eval_wind_phase_rad):
            raise ValueError("eval_wind_phase_rad must be finite.")

        # body ids for external wrench application
        self._uav_body_idx = int(self._body_id[0])
        self._ext_body_ids = [self._uav_body_idx]
        if self._wind_apply_to_payload or self._startup_gust_enabled or self._downwash_enabled:
            self._ext_body_ids.append(int(self._payload_id))

        self._ext_forces_buf = torch.zeros(self.num_envs, len(self._ext_body_ids), 3, device=self.device)
        self._ext_torques_buf = torch.zeros_like(self._ext_forces_buf)

        # per-env wind states
        self._wind_mean = torch.zeros(self.num_envs, 3, device=self.device)
        self._wind_gust = torch.zeros(self.num_envs, 3, device=self.device)
        self._wind_ou = torch.zeros(self.num_envs, 3, device=self.device)
        self._wind_t = torch.zeros(self.num_envs, device=self.device)
        self._wind_t_next = torch.zeros(self.num_envs, device=self.device)

        # UAV mass tensor (constant)
        self._uav_mass_tensor = torch.full((self.num_envs,), float(getattr(self, "_uav_mass", 1.0)), device=self.device)

        # init for all envs
        self._reset_wind(self._robot._ALL_INDICES)

    def _reset_wind(self, env_ids: torch.Tensor):
        if not getattr(self, "_disturbance_enabled", getattr(self, "_wind_enabled", False)):
            return
        dev = self.device
        m = int(env_ids.shape[0])

        # episode-constant mean wind direction (XY)
        if getattr(self, "_wind_enabled", False):
            ang = 2.0 * math.pi * torch.rand(m, device=dev)
            dir_xy = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)  # (m,2)
            mag = self._wind_mean_accel_max * torch.rand(m, device=dev)
            mean_xy = dir_xy * mag.unsqueeze(-1)
            self._wind_mean[env_ids] = torch.cat([mean_xy, torch.zeros(m, 1, device=dev)], dim=-1)
        else:
            self._wind_mean[env_ids] = 0.0

        # clear gust/ou and timers
        self._wind_gust[env_ids] = 0.0
        self._wind_ou[env_ids] = 0.0
        self._wind_acc_w[env_ids] = 0.0
        self._wind_t[env_ids] = 0.0
        self._eval_wind_elapsed_s[env_ids] = 0.0

        dt_next = self._wind_gust_dt_min + (self._wind_gust_dt_max - self._wind_gust_dt_min) * torch.rand(m, device=dev)
        self._wind_t_next[env_ids] = dt_next

        # Smooth random startup pulse in a horizontal world-frame direction.
        if getattr(self, "_startup_gust_enabled", False):
            startup_ang = 2.0 * math.pi * torch.rand(m, device=dev)
            self._startup_direction_w[env_ids] = torch.stack(
                [torch.cos(startup_ang), torch.sin(startup_ang), torch.zeros(m, device=dev)],
                dim=-1,
            )
            accel_lo, accel_hi = self.cfg.startup_gust_accel_range_mps2
            duration_lo, duration_hi = self.cfg.startup_gust_duration_range_s
            self._startup_amplitude[env_ids] = torch.empty(m, device=dev).uniform_(
                float(accel_lo), float(accel_hi)
            )
            self._startup_duration_s[env_ids] = torch.empty(m, device=dev).uniform_(
                float(duration_lo), float(duration_hi)
            )
            self._startup_elapsed_s[env_ids] = 0.0
            self._startup_acc_w[env_ids] = 0.0

        # Episode-constant payload downwash bias plus a zero-initialized OU term.
        if getattr(self, "_downwash_enabled", False):
            downwash_ang = 2.0 * math.pi * torch.rand(m, device=dev)
            force_lo, force_hi = self.cfg.downwash_bias_force_range_n
            downwash_mag = torch.empty(m, device=dev).uniform_(float(force_lo), float(force_hi))
            self._downwash_bias_force_b[env_ids] = torch.stack(
                [torch.cos(downwash_ang) * downwash_mag, torch.sin(downwash_ang) * downwash_mag, torch.zeros(m, device=dev)],
                dim=-1,
            )
            self._downwash_ou_b[env_ids] = 0.0
            self._downwash_force_b[env_ids] = self._downwash_bias_force_b[env_ids]

    def _wind_step(self, dt: float):
        """Update ambient wind, startup pulse, and payload-only downwash."""
        if not getattr(self, "_disturbance_enabled", getattr(self, "_wind_enabled", False)):
            return

        dev = self.device
        n = self.num_envs
        self._eval_wind_elapsed_s += dt

        if getattr(self, "_wind_enabled", False) and getattr(self, "_eval_wind_mode", "training") == "sinusoid":
            elapsed = self._eval_wind_elapsed_s
            active = elapsed >= self._eval_wind_start_sec
            phase = (
                2.0
                * math.pi
                * self._eval_wind_frequency_hz
                * (elapsed - self._eval_wind_start_sec)
                + self._eval_wind_phase_rad
            )
            value = self._eval_wind_amplitude_mps2 * torch.sin(phase)
            value = torch.where(active, value, torch.zeros_like(value))
            self._wind_acc_w.zero_()
            axis_index = 0 if self._eval_wind_axis == "x" else 1
            self._wind_acc_w[:, axis_index] = value * self._eval_wind_scale
        elif getattr(self, "_wind_enabled", False):
            # Existing piecewise gust + OU ambient-wind model.
            self._wind_t += dt
            mask = self._wind_t >= self._wind_t_next
            if mask.any():
                idx = torch.nonzero(mask).squeeze(-1)
                m = int(idx.shape[0])
                ang = 2.0 * math.pi * torch.rand(m, device=dev)
                dir_xy = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
                mag = self._wind_gust_accel_max * (2.0 * torch.rand(m, device=dev) - 1.0)
                gust_xy = dir_xy * mag.unsqueeze(-1)
                self._wind_gust[idx] = torch.cat([gust_xy, torch.zeros(m, 1, device=dev)], dim=-1)
                self._wind_t[idx] = 0.0
                dt_next = self._wind_gust_dt_min + (
                    self._wind_gust_dt_max - self._wind_gust_dt_min
                ) * torch.rand(m, device=dev)
                self._wind_t_next[idx] = dt_next

            noise = torch.randn(n, 3, device=dev)
            if self._wind_axis == "xy":
                noise[:, 2] = 0.0
            self._wind_ou = (
                self._wind_ou
                + (-self._wind_ou_theta * self._wind_ou) * dt
                + self._wind_ou_sigma * math.sqrt(max(dt, 1e-6)) * noise
            )
            if self._wind_axis == "xy":
                self._wind_ou[:, 2] = 0.0

            a_w = self._wind_mean + self._wind_gust + self._wind_ou
            if self._wind_axis == "xy":
                a_w[:, 2] = 0.0
            xy = a_w[:, :2]
            norm = torch.norm(xy, dim=-1).clamp_min(1e-6)
            scale = torch.clamp(self._wind_total_accel_max / norm, max=1.0)
            a_w[:, :2] = xy * scale.unsqueeze(-1)
            if self._wind_axis == "xy":
                a_w[:, 2] = 0.0
            # Evaluation scaling is applied after the training-range clamp.
            self._wind_acc_w = a_w * self._eval_wind_scale
        else:
            self._wind_acc_w.zero_()

        if getattr(self, "_startup_gust_enabled", False):
            self._startup_elapsed_s += dt
            profile = half_sine_profile(self._startup_elapsed_s, self._startup_duration_s)
            self._startup_acc_w = (
                self._startup_direction_w
                * self._startup_amplitude.unsqueeze(-1)
                * profile.unsqueeze(-1)
            )

        if getattr(self, "_downwash_enabled", False):
            downwash_noise = torch.randn(n, 3, device=dev)
            downwash_noise[:, 2] = 0.0
            self._downwash_ou_b = (
                self._downwash_ou_b
                + (-self._downwash_ou_theta * self._downwash_ou_b) * dt
                + self._downwash_ou_sigma * math.sqrt(max(dt, 1e-6)) * downwash_noise
            )
            total_force = self._downwash_bias_force_b + self._downwash_ou_b
            total_force[:, 2] = 0.0
            force_norm = torch.linalg.norm(total_force[:, :2], dim=-1).clamp_min(1e-6)
            force_scale = torch.clamp(self._downwash_force_clip / force_norm, max=1.0)
            self._downwash_force_b = total_force * force_scale.unsqueeze(-1)

    # ---------------- Quaternion helpers (wxyz) ----------------
    @staticmethod
    def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton product for wxyz quaternions."""
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    @staticmethod
    def _quat_from_roll_pitch(roll_pitch_rad: torch.Tensor) -> torch.Tensor:
        """Create a zero-yaw wxyz quaternion from roll/pitch calibration bias."""
        half_roll = 0.5 * roll_pitch_rad[:, 0]
        half_pitch = 0.5 * roll_pitch_rad[:, 1]
        cr, sr = torch.cos(half_roll), torch.sin(half_roll)
        cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
        return torch.stack([cr * cp, sr * cp, cr * sp, -sr * sp], dim=-1)

    @staticmethod
    def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # v' = v + 2*cross(qvec, cross(qvec,v) + w*v)
        qw = q[..., 0:1]
        qv = q[..., 1:4]
        t = 2.0 * torch.cross(qv, v, dim=-1)
        return v + qw * t + torch.cross(qv, t, dim=-1)

    @classmethod
    def _quat_rotate_inverse(cls, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return cls._quat_rotate(cls._quat_conjugate(q), v)

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
