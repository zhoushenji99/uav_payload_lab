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
                "time_penalty",  # 时间惩罚（你现在 reward 里没用到的话，可以以后删掉）
                "death_penalty", # 摔机惩罚
                "total",         # 总 reward
                "dist",          # payload 到目标的距离（m）
                "theta_deg",     # payload 合摆角（deg）
                "swing_deg_s",   # payload 合角速度（deg/s）
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
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

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
        L = self.cfg.rope_length

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
        r_action_val = -0.001 * torch.sum(torch.square(self._actions), dim=1)

        # [E] 死亡惩罚 (death_penalty)
        root_pos = self._robot.data.root_pos_w
        env_origins = self._terrain.env_origins.to(root_pos.device)
        
        # 高度判定 (0.1 ~ 6.0m)
        height_fail = torch.logical_or(root_pos[:, 2] < 0.1, root_pos[:, 2] > 6.0)
        # 水平出界判定 (±6.0m)
        rel_pos = root_pos - env_origins
        out_of_box = torch.any(torch.abs(rel_pos) > 6.0, dim=1)
        
        died = torch.logical_or(height_fail, out_of_box)
        death_penalty_vec = -1.0 * float(self.cfg.death_penalty) * died.float()

        # === 4. 总奖励汇总 ===
        reward = r_pos_val + r_tilt_val + r_swing_val + r_action_val + death_penalty_vec

        # === 5. Logging (完全兼容你原来的结构) ===
        # 这里为了保持和你 __init__ 中的 keys 一致，我把各项归类
        rewards_dict = {
            "r_pos": r_pos_val,         # 包含生存、距离、高斯
            "r_tilt": r_tilt_val,       # 仅包含角度惩罚
            "r_swing": r_swing_val,     # 仅包含角速度惩罚
            "death_penalty": death_penalty_vec,
            "dist": dist,               # 纯粹的物理距离用于记录
            "theta_deg": theta_deg,     # 纯粹的物理角度用于记录
            "swing_deg_s": swing_deg_s, # 纯粹的物理角速度用于记录
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
        # 计算本局的步数 / 时间（每个 env 自己的）
        ep_steps = self.episode_length_buf[env_ids].float().clamp(min=1.0)
        ep_time  = ep_steps * self.step_dt  # 单位：秒

        # Logging
        p_load_w = self._robot.data.body_pos_w[env_ids, self._payload_id, :]
        goal_payload_w = self._desired_pos_w[env_ids]
        final_distance_to_goal = torch.linalg.norm(goal_payload_w - p_load_w, dim=1).mean()

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
        # 时间平均距离（m）
        dist_per_sec = self._episode_sums["dist"][env_ids] / ep_time
        extras["Metrics/avg_dist"] = dist_per_sec.mean().item()
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