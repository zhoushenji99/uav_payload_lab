# uav_payload_lab_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.actuators import ImplicitActuatorCfg

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


# === Iris + payload 机器人配置 ===
IRIS_PAYLOAD_CFG = ArticulationCfg(
    # 和 CRAZYFLIE 一样，用 ENV_REGEX_NS 作为模板，下面再用 replace 改成 /World/envs/env_.*/Robot
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # TODO：如果你改过路径，这里换成你真实的 iris_payload.usd 路径
        usd_path="/home/shenji/uav_payload_lab/uav_payload_lab/source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_lab/iris_payload.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 这里我按你之前 peg 项目约定：payload 初始大概在 z=0.4，绳长 0.8 ⇒ UAV z≈1.2
        pos=(0.0, 0.0, 1.2),
        # 所有关节初始角度 = 0
        joint_pos={
            ".*": 0.0,
        },
        # 所有关节初始角速度 = 0（不像 CRAZYFLIE 那样给螺旋桨预转速）
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        # 和 CRAZYFLIE 一样，挂一个 dummy actuator，让 articulation 在 IsaacLab 里是“有执行器的”
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

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
    robot: ArticulationCfg = IRIS_PAYLOAD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9 #无量纲参数，意思是“最大推力大概是机重的多少倍”
    moment_scale = 0.01

    # CTBR 相关参数（body-rate 动作 → 力矩）
    body_rate_max = 5.0             # rad/s，把 [-1,1] 的无量纲动作映射到 [-ω_max, +ω_max] 的物理角速度范围
    rate_kp = 0.05                  # 简单 body-rate P 控制增益：τ = Kp (ω_des - ω)

    # 场景：并行 env 数 / 间距
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, #4096
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
