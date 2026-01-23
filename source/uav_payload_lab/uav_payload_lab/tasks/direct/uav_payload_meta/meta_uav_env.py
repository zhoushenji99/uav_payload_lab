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

from .meta_uav_env_cfg import UavPayloadMetaEnvCfg



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

        # 2) 计算 F_max（固定：只看 UAV body 质量）
        g = float(self.cfg.sim.gravity[2]) if hasattr(self.cfg.sim, "gravity") else -9.81
        g = abs(g)

        masses0 = self._robot.root_physx_view.get_masses()[0]  # (num_bodies,)
        uav_mass = float(masses0[self._body_id[0]].item())     # 只取 body 的质量
        self._uav_mass = uav_mass  # <<< [新增] 缓存 UAV 质量，给风扰用
        self._F_max = self.cfg.thrust_to_weight * (uav_mass * g)

        # 3) 每个 env 的绳长 / 悬停推力 buffer
        # self._rope_len = torch.full((self.num_envs,), self.cfg.rope_length, device=self.device)
        self._F_hover = torch.full((self.num_envs,), uav_mass * g, device=self.device)  # 先给个初值

        # 缓存默认 mass / inertia（用于每次reset从“默认值”开始随机）
        self._default_masses_cpu = self._robot.root_physx_view.get_masses().clone()     # (num_envs, num_bodies) on CPU
        self._default_inertias_cpu = self._robot.root_physx_view.get_inertias().clone() # (num_envs, num_bodies, 9) on CPU

        # 记录每个env当前payload质量（放在env device上用于log）
        self._payload_mass = torch.zeros(self.num_envs, device=self.device)


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
        # Wind disturbance module (optional)
        self._init_wind_module()

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
        self._raw_actions = actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]
        # [新增] wind state update (OU + gust), store wind accel in world frame
        self._wind_step(self.step_dt)

    def _apply_action(self):
        # 默认行为：无风扰时保持原逻辑不变
        if not getattr(self, "_wind_enabled", False):
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
        acc_w = self._wind_acc_w  # (N,3) world
        if self._wind_apply_to_uav:
            F_uav_w = (self._uav_mass_tensor.unsqueeze(-1) * acc_w) * self._wind_scale_uav
        else:
            F_uav_w = torch.zeros_like(acc_w)

        # payload force uses per-env payload mass buffer
        if self._wind_apply_to_payload and (len(body_ids) > 1):
            F_pay_w = (self._payload_mass.unsqueeze(-1) * acc_w) * self._wind_scale_payload
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

            F_pay_b = self._quat_rotate_inverse(quat_pay_w, F_pay_w)
            forces[:, 1, :] = forces[:, 1, :] + F_pay_b

        # apply (forces/torques are in body frame, consistent with your thrust)
        self._robot.set_external_force_and_torque(forces, torques, body_ids=body_ids)

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
        L = self._rope_lengths  # 保持 (num_envs,) 维度，不要 unsqueeze

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
        obs = torch.cat(
            [
                e_load,       # 0-2
                tilt_deg,     # 3-4
                w_deg,        # 5-6
                root_quat_w,  # 7-10
                v_b,          # 11-13
                w_b,          # 14-16
            ],
            dim=-1,
        )
        # --- Oracle: append true payload mass (normalized to [0,1]) ---
         # --- 3) 全知模式拼接 (Oracle Mode) ---
        # 如果 Config 里开了 True，就把 Mass 和 Length 拼进去
        if getattr(self.cfg, "use_oracle_mass_obs", False):
            # A. 处理 Mass
            lo_m, hi_m = self.cfg.payload_mass_range
            denom_m = max(float(hi_m) - float(lo_m), 1e-6)
            m_norm = (self._payload_mass - float(lo_m)) / denom_m
            m_norm = torch.clamp(m_norm, 0.0, 1.0).unsqueeze(-1)
            
            # B. 处理 Rope Length
            lo_l, hi_l = self.cfg.rope_length_range
            denom_l = max(float(hi_l) - float(lo_l), 1e-6)
            l_norm = (self._rope_lengths - float(lo_l)) / denom_l 
            l_norm = torch.clamp(l_norm, 0.0, 1.0).unsqueeze(-1)
            # C. 【新增】处理 Wind (转机体系 + 归一化)
            # 1. 获取世界系风加速度 (N, 3)
            wind_w = self._wind_acc_w 
            # 2. 旋转到机体系 (使用当前四元数 root_quat_w)
            #    _quat_rotate_inverse 是把世界系向量转回机体系
            wind_b = self._quat_rotate_inverse(root_quat_w, wind_w)
            # 3. 归一化 (除以配置中的最大风力)
            #    防止神经网络输入过大，一般除以 wind_total_accel_max
            max_wind = getattr(self.cfg, "wind_total_accel_max", 3.0)
            wind_norm = wind_b / max(max_wind, 1e-6)
            # C. 最终拼接：17 + 1 + 1 + 3 = 22
            obs = torch.cat([obs, m_norm, l_norm, wind_norm], dim=-1)
            
        return {"policy": obs}



    def _get_rewards(self) -> torch.Tensor:
        """
        混合奖励函数：线性距离引导 + 高斯精度锁定 + 消摆惩罚
        """
        # === 1. 数据准备 ===
        p_uav_w = self._robot.data.root_pos_w                          
        p_load_w = self._robot.data.body_pos_w[:, self._payload_id, :] 
        goal_payload_w = self._desired_pos_w                           

        # 距离误差 (m)
        e_load = goal_payload_w - p_load_w                             
        dist = torch.linalg.norm(e_load, dim=1)                        

        # === 2. 摆角与角速度计算 (使用更精确的几何方法) ===
        # payload 相对 UAV 的向量
        r = p_load_w - p_uav_w                                         
        dx, dy, dz = r[:, 0], r[:, 1], r[:, 2]
        
        # [改进] 使用 atan2 计算真实的合摆角，比 sqrt(tx^2+ty^2) 更准
        # den 取 -dz 是因为 z 轴向下为负，我们要算的是偏离垂直向下的角度
        den = torch.clamp(-dz, min=1e-3)
        theta_rad = torch.atan2(torch.sqrt(dx*dx + dy*dy), den)
        theta_deg = theta_rad * (180.0 / math.pi)

        # 摆动角速度 (deg/s) - 直接使用观测中计算好的差分速度
        wx_deg = self._tilt_vel_deg[:, 0]
        wy_deg = self._tilt_vel_deg[:, 1]
        swing_deg_s = torch.sqrt(wx_deg * wx_deg + wy_deg * wy_deg)    
        # 归一化摆能量 E_hat（小角度线性摆近似，单位：rad^2/s^2）
        # E_hat = 0.5*(||theta_dot||^2 + (g/L)*||theta||^2)，用于诊断/画图，不进入 reward
        g = 9.81
        L = torch.clamp(self._rope_lengths, min=1e-3)
        theta_dot_rad_s = swing_deg_s * (math.pi / 180.0)
        E_hat = 0.5 * (theta_dot_rad_s * theta_dot_rad_s + (g / torch.clamp(L, min=1e-6)) * (theta_rad * theta_rad))        # 记录时间平均：episode_sums 累加的是 ∑ E_hat * dt，reset 时再除以 T
        E_hat_mean_dt = E_hat * self.step_dt
        # === 3. 计算各项奖励组件 ===
        
        # [A] 位置奖励 (r_pos)
        # 逻辑：基础生存分(4.0) - 距离惩罚(dist) + 终点高斯奖励(gauss)
        # 这样设计保证了：
        # 1. 只要在 4m 内，分数 > 0，防止自杀 (4.0 - dist)
        # 2. 远处有梯度 (dist 越小分越高)
        # 3. 近处有诱惑 (进入 sigma 范围后由高斯项提供高分)
        r_alive = 4.0
        r_dist_dense = -1.0 * dist
        r_dist_gauss = torch.exp(-0.5 * (dist / self.cfg.sigma_pos)**2)
        
        # 组合位置奖励
        r_pos_val = float(self.cfg.pos_weight) * (r_alive + r_dist_dense + 2.0 * r_dist_gauss)

        # [B] 摆角惩罚 (r_tilt)
        # 摆角越大扣分越多，平方项让大角度惩罚更重
        r_tilt_val = -1.0 * float(self.cfg.tilt_weight) * (theta_deg / self.cfg.sigma_tilt_deg)**2

        # [C] 摆速惩罚 (r_swing)
        # 摆动越快扣分越多 (注意权重系数我给小了一点，避免初期为了不摆动而不敢动)
        r_swing_val = -0.1 * float(self.cfg.tilt_weight) * (swing_deg_s / self.cfg.sigma_swing_deg_s)**2

        # [D] 动作平滑惩罚 (r_action) - 新增项，不算在 r_pos 里，算额外惩罚
        # 防止力矩控制时电机高频震荡
        r_action_val = -0.0 * torch.sum(torch.square(self._actions), dim=1)

        # [E] 死亡惩罚 (death_penalty)
        root_pos = self._robot.data.root_pos_w
        env_origins = self._terrain.env_origins.to(root_pos.device)
        
        # 高度判定 (0.1 ~ 6.0m)
        height_fail = torch.logical_or(root_pos[:, 2] < 0.1, root_pos[:, 2] > 6.0)
        # 水平出界判定 (±6.0m)
        rel_pos = root_pos - env_origins
        out_of_box = torch.any(torch.abs(rel_pos) > 6.0, dim=1)
        # [新增] 计算 Raw Action 越界程度 (用于 Log)
        # 即使你不把它加到 total reward 里，算出这个值也能在 TB 里看到
        raw_excess = torch.relu(torch.abs(self._raw_actions) - 1.0)
        r_action_raw_val = -1.0 * torch.sum(torch.square(raw_excess), dim=1)
        
        died = torch.logical_or(height_fail, out_of_box)
        death_penalty_vec = -1.0 * float(self.cfg.death_penalty) * died.float()
        # [新增] 自旋惩罚 (Spin Penalty)
        # self._robot.data.root_ang_vel_b[:, 2] 是机体系下的 Z 轴角速度 (Yaw Rate)
        # 我们希望它越接近 0 越好
        # 1. 解算 Yaw 角 (Rad)
        quat = self._robot.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw_angle = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        yaw_rate = self._robot.data.root_ang_vel_b[:, 2]
        r_spin_val = -1.0 * float(self.cfg.spin_weight) * (torch.square(yaw_rate) + torch.square(yaw_angle))
        # === [新增] 标准 Action L2 Penalty (Effort) ===
        # 惩罚动作幅度的平方。鼓励 Agent 在不需要大机动时回归到 0 (即悬停状态)。
        # 这就是 Omnidrones 和标准控制论文里的 "Control Cost"。
        r_action_l2 = -self.cfg.action_l2_penalty_scale * (self._raw_actions ** 2).sum(dim=1)        
        # === 4. 总奖励汇总 ===
        reward = r_pos_val + r_tilt_val + r_swing_val + r_spin_val + r_action_val + death_penalty_vec + r_action_l2
        # === 5. Logging (完全兼容你原来的结构) ===
        # 这里为了保持和你 __init__ 中的 keys 一致，我把各项归类
        rewards_dict = {
            "r_pos": r_pos_val,         # 包含生存、距离、高斯
            "r_tilt": r_tilt_val,       # 仅包含角度惩罚
            "r_spin": r_spin_val,
            "r_swing": r_swing_val,     # 仅包含角速度惩罚
            "death_penalty": death_penalty_vec,
            # [新增] 记录这两个数据
            "r_action_raw": r_action_raw_val,
            "action_raw_sum": torch.sum(torch.abs(self._raw_actions), dim=1),
            "dist": dist,               # 纯粹的物理距离用于记录
            "theta_deg": theta_deg,     # 纯粹的物理角度用于记录
            "swing_deg_s": swing_deg_s, # 纯粹的物理角速度用于记录
            "E_hat_mean": E_hat_mean_dt, # 纯粹的诊断量：E_hat 时间平均（不进 reward）
            # 原来代码里可能有 time_penalty，现在没用上，置0即可防止报错
            "time_penalty": torch.zeros_like(reward), 
            "total": reward
        }

        # 遍历累加，这和你原来的逻辑一模一样
        for key, value in rewards_dict.items():
            # 确保 key 存在于 _episode_sums 中 (r_action 这种没定义的就不记了，或者加到 total 里了)
            if key in self._episode_sums:
                self._episode_sums[key] += value
        
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

            # died = 高度越界 或 出盒子
            died = torch.logical_or(height_fail, out_of_box)

            return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        # 必须在 reset 物理状态之前记录，否则位置都被重置了，永远查不到死因
        self._log_termination_stats(env_ids)

        # 1. 重置 Robot (清除之前的速度、力等)
        self._robot.reset(env_ids)

        # 2. 获取默认状态
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        # 3. 绳长随机化 & 设置关节目标 (核心修复!)
        if hasattr(self.cfg, "rope_length_range"):
            lo_len, hi_len = self.cfg.rope_length_range
            L = torch.rand(len(env_ids), device=self.device) * (hi_len - lo_len) + lo_len
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
        # 5. 质量随机化 (你之前的代码好像漏了这一段，我帮你补上，为了Log正确)
        if hasattr(self.cfg, "payload_mass_range"):
            lo, hi = self.cfg.payload_mass_range
            env_ids_cpu = env_ids.to("cpu")
            # 假设你已经缓存了 _default_masses_cpu
            masses = self._default_masses_cpu.clone() 
            new_mass = torch.empty((len(env_ids_cpu),), device="cpu").uniform_(float(lo), float(hi))
            masses[env_ids_cpu, self._payload_id] = new_mass
            self._robot.root_physx_view.set_masses(masses, env_ids_cpu)
            # 记录到 buffer
            self._payload_mass[env_ids] = new_mass.to(self.device)
        
        # === 6.【关键修改】随机化完参数后，立刻记本局参数 ===
        # 确保记录的是新一局的真实质量和绳长
        self._log_task_config(env_ids)
        self._reset_wind(env_ids)
        # 4. 计算出生位置 (使用 Config 里的 start_pos_w)
        # 加上 env_origins，让无人机分散开，不要叠在一起
        env_origins = self._terrain.env_origins[env_ids]
        default_root_state[:, :3] = env_origins + self._start_offset
        # # 设置目标点
        self._desired_pos_w[env_ids] = env_origins + self._goal_offset

        # 5. 写入物理引擎
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 6. 重置历史 Buffer (用于摆角速度计算)
        if isinstance(self._prev_tilt_deg, torch.Tensor):
            self._prev_tilt_deg[env_ids] = 0.0
            self._tilt_vel_deg[env_ids] = 0.0
            self._has_prev_tilt[env_ids] = False

        # 8. 父类逻辑 (必须调用)
        super()._reset_idx(env_ids)
        
        # 9. 错峰 Reset (Spread out)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        self._actions[env_ids] = 0.0

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
        
        extras = dict()
        extras["Metrics/payload_mass_true_mean"] = float(m.mean().item())
        extras["Metrics/payload_mass_true_min"]  = float(m.min().item())
        extras["Metrics/payload_mass_true_max"]  = float(m.max().item())
        
        extras["Metrics/rope_length_mean"] = float(l.mean().item())
        extras["Metrics/rope_length_min"]  = float(l.min().item())
        extras["Metrics/rope_length_max"]  = float(l.max().item())

        if "log" not in self.extras: self.extras["log"] = dict()
        self.extras["log"].update(extras)


    # ---------------------------------------------------------------------
    # Wind disturbance module (optional)
    #   - OU smooth noise + piecewise constant gust
    #   - wind modeled as world-frame acceleration (m/s^2)
    #   - converted to force via F = m * a, then rotated into body frame
    # ---------------------------------------------------------------------

    def _init_wind_module(self):
        self._wind_enabled = bool(getattr(self.cfg, "enable_wind", False))

        # always create buffer to avoid attribute errors
        self._wind_acc_w = torch.zeros(self.num_envs, 3, device=self.device)

        if not self._wind_enabled:
            return

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

        # body ids for external wrench application
        self._uav_body_idx = int(self._body_id[0])
        self._ext_body_ids = [self._uav_body_idx]
        if self._wind_apply_to_payload:
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
        if not getattr(self, "_wind_enabled", False):
            return
        dev = self.device
        m = int(env_ids.shape[0])

        # episode-constant mean wind direction (XY)
        ang = 2.0 * math.pi * torch.rand(m, device=dev)
        dir_xy = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)  # (m,2)
        mag = self._wind_mean_accel_max * torch.rand(m, device=dev)
        mean_xy = dir_xy * mag.unsqueeze(-1)
        self._wind_mean[env_ids] = torch.cat([mean_xy, torch.zeros(m, 1, device=dev)], dim=-1)

        # clear gust/ou and timers
        self._wind_gust[env_ids].zero_()
        self._wind_ou[env_ids].zero_()
        self._wind_acc_w[env_ids].zero_()
        self._wind_t[env_ids].zero_()

        dt_next = self._wind_gust_dt_min + (self._wind_gust_dt_max - self._wind_gust_dt_min) * torch.rand(m, device=dev)
        self._wind_t_next[env_ids] = dt_next

    def _wind_step(self, dt: float):
        """Update wind state and write self._wind_acc_w (world frame)."""
        if not getattr(self, "_wind_enabled", False):
            return

        dev = self.device
        n = self.num_envs

        # gust: piecewise constant
        self._wind_t += dt
        mask = self._wind_t >= self._wind_t_next
        if mask.any():
            idx = torch.nonzero(mask).squeeze(-1)
            m = int(idx.shape[0])

            ang = 2.0 * math.pi * torch.rand(m, device=dev)
            dir_xy = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)

            # allow +/- gust along sampled direction
            mag = self._wind_gust_accel_max * (2.0 * torch.rand(m, device=dev) - 1.0)
            gust_xy = dir_xy * mag.unsqueeze(-1)
            self._wind_gust[idx] = torch.cat([gust_xy, torch.zeros(m, 1, device=dev)], dim=-1)

            self._wind_t[idx] = 0.0
            dt_next = self._wind_gust_dt_min + (self._wind_gust_dt_max - self._wind_gust_dt_min) * torch.rand(m, device=dev)
            self._wind_t_next[idx] = dt_next

        # OU: smooth, time-varying
        noise = torch.randn(n, 3, device=dev)
        if self._wind_axis == "xy":
            noise[:, 2] = 0.0

        self._wind_ou = self._wind_ou + (-self._wind_ou_theta * self._wind_ou) * dt + self._wind_ou_sigma * math.sqrt(max(dt, 1e-6)) * noise
        if self._wind_axis == "xy":
            self._wind_ou[:, 2] = 0.0

        # total accel
        a_w = self._wind_mean + self._wind_gust + self._wind_ou
        if self._wind_axis == "xy":
            a_w[:, 2] = 0.0

        # clamp XY magnitude to avoid blow-ups
        xy = a_w[:, :2]
        norm = torch.norm(xy, dim=-1).clamp_min(1e-6)
        scale = torch.clamp(self._wind_total_accel_max / norm, max=1.0)
        a_w[:, :2] = xy * scale.unsqueeze(-1)
        if self._wind_axis == "xy":
            a_w[:, 2] = 0.0

        self._wind_acc_w = a_w

    # ---------------- Quaternion helpers (wxyz) ----------------
    @staticmethod
    def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

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