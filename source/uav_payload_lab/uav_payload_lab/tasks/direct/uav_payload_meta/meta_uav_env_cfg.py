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
        usd_path="/home/shenji/uav_payload_lab/uav_payload_lab/source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_meta/iris_payload_prismatic.usd",
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
        pos=(0.0, 0.0, 2.0),
        # 所有关节初始角度 = 0
        joint_pos={
            r"^(?!rope_joint).*": 0.0,
            "rope_joint": -0.8, # 这里的数值必须和usd的drive target对上，不然初始第一帧就会拉扯到爆炸【关键】给绳长关节一个合法的初值 (-1.5 ~ -0.1 之间)
        },
        # 所有关节初始角速度 = 0（不像 CRAZYFLIE 那样给螺旋桨预转速）
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        # 1. 螺旋桨等普通关节 (保持不变)
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[r"^(?!rope_joint).*"], 
            stiffness=0.0,
            damping=0.0,
        ),
        # 2. 【新增】绳长关节 Actuator
        # 加上这个后，Isaac Lab 会自动把 init_state 里的 -0.8 也应用给 Drive Target
        # 这样初始时刻 Target=-0.8, Pos=-0.8, 力=0，就不会炸了！
        "rope_winch": ImplicitActuatorCfg(
            joint_names_expr=["rope_joint"],
            stiffness=100.0,  # 在这里指定刚度，覆盖 USD
            damping=10.0,      # 在这里指定阻尼，覆盖 USD
        ),
    },
)

@configclass
class UavPayloadMetaEnvCfg(DirectRLEnvCfg):
        # env
    decimation = 2                      # 每执行一次 RL action，物理 step 多少次
    episode_length_s = 35.0             # 一局多长时间（秒）
    action_space = 4
    observation_space = 22          #oracle就是17+2ML+3wind,不是就是17
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
    moment_scale = 0.4 #$\tau = J \cdot \alpha$   $$\tau_{max} = 0.01 \, (\text{kg}\cdot\text{m}^2) \times 20 \, (\text{rad/s}^2) = \mathbf{0.2 \, \text{Nm}}$$
    
    # 绳长（m），暂时手动指定；后续可改为从 usd 读取 rope 可视长度
    rope_length_range = (0.3, 0.8)  # 绳长随机范围 (米)
    
    use_oracle_mass_obs = True
    payload_mass_range = (0.05, 0.15)   # kg，每个episode随机
    recompute_inertia = True         # 质量大改动时建议同步缩放惯量
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Wind disturbance (optional)
    # ---------------------------------------------------------------------
    enable_wind = True                 # 总开关：True 才启用风扰
    wind_apply_to_uav = True            # 是否对 UAV 本体施加风扰
    wind_apply_to_payload = True        # 是否对 payload 刚体施加风扰（你想要的默认就是 True）
    wind_axis = "xy"                    # "xy"：只水平风；"xyz"：允许垂直扰动（一般先别开）

    # 风扰用“等效加速度”建模（更稳：不同质量不会被同一牛顿力吹飞）
    wind_mean_accel_max = 0.5           # episode-constant mean wind accel 最大值 (m/s^2)
    wind_gust_accel_max = 1.5           # gust 加速度幅值 (m/s^2)，每段随机方向/正负
    wind_total_accel_max = 3.0          # 总风加速度限幅 (m/s^2)

    # gust 分段常值持续时间（秒）
    wind_gust_dt_min = 0.5
    wind_gust_dt_max = 2.0

    # OU 平滑噪声参数：dx = -theta*x*dt + sigma*sqrt(dt)*N(0,1)
    wind_ou_theta = 1.0                 # (1/s) 越大越“拉回 0”，变化更快但更平滑
    wind_ou_sigma = 1.0                 # (m/s^2 / sqrt(s)) 噪声强度

    # UAV vs payload 受风比例（同一风向，力度不同）
    wind_scale_uav = 0.4
    wind_scale_payload = 1.0
    
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------


    # 场景：并行 env 数 / 间距
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, #4096
        env_spacing=6,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # reward scales
    # === Reward 参数：payload 到点 + 消摆 ===
    sigma_pos = 0.2            # 位置高斯尺度（m）
    sigma_tilt_deg = 10.0       # 摆角高斯尺度（deg）
    sigma_swing_deg_s = 10.0   # 摆角角速度高斯尺度（deg/s）
    pos_weight = 0.3           # 位置主项权重
    tilt_weight = 0.15          # 摆角 / 摆速 shaping 权重
    time_penalty = 0.01         # 每秒时间惩罚系数（越大越鼓励快完成）
    death_penalty = 20       # 摔机一次性扣多少（可以先 10，觉得不够再加大）
    action_l2_penalty_scale = 0.001
    spin_weight = 0.15   # 【新增】自旋惩罚权重
    heading_weight = 0.5     # 【新增】航向对齐权重，用于抑制 Yaw 角度 (P控制)
    # === 任务设置（相对每个 env 的原点，ENU）===
    # UAV 起点用于reset：payload 初始在 (0.5, 1.0, 0.4)，绳长 0.8 ⇒ UAV z ≈ 1.2
    #斜飞start_pos_w = (0.5, 1.0, 1.2)
    #平飞start_pos_w = (2.0, 0.0, 1.2)
    start_pos_w = (-2.0, 0.0, 2.0)
    # payload 终点用于reward：0.5 1 0.4 → -0.5 0 1.2
    #斜飞goal_pos_w = (-0.5, 0.0, 1.2)
    #平飞goal_pos_w = (-2.0, 0.0, 1.2)
    goal_pos_w = (2.0, 0.0, 2.0)
