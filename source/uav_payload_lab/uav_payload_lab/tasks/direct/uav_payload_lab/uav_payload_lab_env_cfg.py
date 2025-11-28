# uav_payload_lab_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets import CRAZYFLIE_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.envs.ui import BaseEnvWindow

class UavPayloadLabEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: "UavPayloadLabEnv", window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements —— 和官方 QuadcopterEnvWindow 一样
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class UavPayloadLabEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2                      # 每执行一次 RL action，物理 step 多少次
    episode_length_s = 10.0             # 一局多长时间（秒）
    action_space = 4
    observation_space = 12
    state_space = 0
    debug_vis = True

    # 这里先设为 None，实际的 window 类在 env 文件里定义好以后，
    # 会在那里做：UavPayloadLabEnvCfg.ui_window_class_type = UavPayloadLabEnvWindow
    ui_window_class_type = UavPayloadLabEnvWindow

    # simulation 物理步长 1/120 秒，每隔 decimation 步物理才渲染一帧
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # 地形：平面
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot（quadcopter）
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # 场景：并行 env 数 / 间距
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=24, #4096
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
