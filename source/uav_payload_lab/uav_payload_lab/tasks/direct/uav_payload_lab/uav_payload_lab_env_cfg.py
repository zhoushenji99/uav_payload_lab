# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class UavPayloadLabEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2      #每执行一次 RL action，物理 step 多少次
    episode_length_s = 5.0  #一局多长时间（秒），基类会用 dt 和这个算 max_episode_length
    # - spaces definition   动作维度、obs 维度，用于 sanity-check
    action_space = 1
    observation_space = 4
    state_space = 0

    # simulation物理步长 1/120 秒，每隔decimation步物理才渲染一帧
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s) usd换的地方
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene 并行 env 数  每个 env 之间的间距
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint用来在 env 里 find_joints
    cart_dof_name = "slider_to_cart"  
    pole_dof_name = "cart_to_pole"
    # - action scale你发过去的是无量纲 action，乘这个变成牛顿
    action_scale = 100.0  # [N]
    # - reward scales传入 compute_rewards 的权重
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditionsreset 时和 done 判据用
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]