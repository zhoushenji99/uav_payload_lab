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
        usd_path="/home/shenji/uav_payload_lab/uav_payload_lab/source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/iris_payload_prismatic.usd",
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
    observation_space = 26          #oracle就是17+2ML+3wind,不是就是17 3.17还得加lastaction 4维
    state_space = 0
    debug_vis = True

    # ---------------------------------------------------------------------
    # First-flight Gap v3: measured, identifiable central training domain.
    # Wide/unmeasured tails belong in evaluation, not in Teacher training.
    # The 3.100 kg bare-airframe measurement excludes the gimbal. One measured
    # half of the 63.5 g two-stage gimbal is fixed to the UAV root; the other
    # half is included in payload_fixed_moving_mass_kg below.
    # They are applied through the PhysX tensor API; the source USD is kept intact.
    # ---------------------------------------------------------------------
    real_hover_gap_profile = "first_flight_gap_v3"
    enable_real_hover_gap = True
    uav_bare_mass_kg = 3.100
    uav_fixed_gimbal_mass_kg = 0.03175
    uav_mass_kg = uav_bare_mass_kg + uav_fixed_gimbal_mass_kg
    uav_com_m = (0.00389, 0.02922, 0.17422)
    # The raw measured Izz=0.160 violates Izz <= Ixx + Iyy. 0.150 is the
    # nearest conservative physical diagonal used for simulation.
    uav_inertia_diag_kg_m2 = (0.0763, 0.0762, 0.1500)
    uav_mass_scale_range = (0.99, 1.01)
    uav_com_offset_range_m = (-0.002, 0.002)
    uav_inertia_scale_range = (0.95, 1.05)

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
    thrust_to_weight = 2.61#无量纲参数，意思是“最大推力大概是机重的多少倍”
    moment_scale_xy = 1.0
    moment_scale_z = 0.25

    # PX4-native CTBR: action = [thrust_body[2], roll_rate, pitch_rate, yaw_rate].
    # Multicopter upward thrust is negative thrust_body[2] in PX4.
    action_interface = "px4_ctbr"
    ctbr_single_motor_max_thrust_n = 19.73
    ctbr_total_max_thrust_n = 78.92
    # Bench fit: T_motor = 14.584*u^2 + 5.438*u. Normalize its endpoint
    # to the independently measured 19.73 N/motor (78.92 N total).
    ctbr_thrust_model = "normalized_quadratic"
    ctbr_thrust_curve_coeffs = (14.584, 5.438, 0.0)
    ctbr_pwm_range_us = (1150.0, 1900.0)
    measured_voltage_range_v = (23.5, 25.2)
    measured_current_range_a = (0.22, 13.6)
    ctbr_body_rate_limit = (1.2, 1.2, 0.6)
    ctbr_max_delta_per_step = (0.03, 0.25, 0.25, 0.10)
    # First-order PX4 body-rate closed-loop uncertainty. The 0.935 s first-flight
    # window suggests roughly 0.10-0.16 s lag but is too short for a point
    # estimate, so training spans a deliberately wider per-axis interval.
    ctbr_rate_time_constant_range_s = ((0.08, 0.25), (0.08, 0.25), (0.12, 0.45))
    ctbr_rate_kp = (0.35, 0.35, 0.08)
    ctbr_moment_limit = (1.0, 1.0, 0.25)
    ctbr_px4_to_isaac_rate_sign = (1.0, -1.0, -1.0)

    # Scheme-1 lumped suspended assembly. Rope mass varies linearly with L;
    # the moving gimbal half and AprilTag plate are fixed at the payload link.
    rope_length_range = (0.25, 0.8)
    rope_mass_range_kg = (0.010, 0.030)
    payload_fixed_moving_mass_kg = 0.10265  # 31.75 g moving gimbal + 70.9 g tag
    payload_ballast_mass_range = (0.2, 0.8)

    use_oracle_mass_obs = True
    # Total mass seen by physics and hard-explicit z0:
    # fixed moving hardware + length-dependent rope + random ballast.
    payload_mass_range = (0.31265, 0.93265)
    # Evaluation-only fixed values.  Keep the ranges above unchanged because
    # they define the hard-explicit mass/rope normalization seen in training.
    eval_fixed_rope_length_m: float | None = None
    eval_fixed_payload_mass_kg: float | None = None
    eval_disable_wind: bool = False
    # Applied to the physical wind after the training-range clamp.  Keeping
    # wind_total_accel_max unchanged preserves the Teacher normalization used
    # during training, so values above 1.0 are genuine physical OOD tests.
    eval_wind_scale: float = 1.0
    # Deterministic evaluation waveform. "training" preserves the original
    # mean + piecewise gust + OU process exactly.
    eval_wind_mode: str = "training"
    eval_wind_amplitude_mps2: float = 1.0
    eval_wind_frequency_hz: float = 1.0
    eval_wind_start_sec: float = 3.0
    eval_wind_axis: str = "x"
    eval_wind_phase_rad: float = 0.0
    recompute_inertia = True         # 质量大改动时建议同步缩放惯量
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Wind disturbance (optional)
    # ---------------------------------------------------------------------
    enable_wind = True                 # 总开关：True 才启用风扰 False
    wind_apply_to_uav = True            # 是否对 UAV 本体施加风扰
    wind_apply_to_payload = True        # 是否对 payload 刚体施加风扰（你想要的默认就是 True）
    wind_axis = "xy"                    # "xy"：只水平风；"xyz"：允许垂直扰动（一般先别开）

    # 风扰用“等效加速度”建模（更稳：不同质量不会被同一牛顿力吹飞）
    wind_mean_accel_max = 0.1           # indoor central-domain mean acceleration (m/s^2)
    wind_gust_accel_max = 0.3           # identifiable, mild piecewise gust (m/s^2)
    wind_total_accel_max = 0.6          # total ambient-wind acceleration clamp (m/s^2)

    # gust 分段常值持续时间（秒）
    wind_gust_dt_min = 1.0
    wind_gust_dt_max = 3.0

    # OU 平滑噪声参数：dx = -theta*x*dt + sigma*sqrt(dt)*N(0,1)
    wind_ou_theta = 1.0                 # (1/s) 越大越“拉回 0”，变化更快但更平滑
    wind_ou_sigma = 0.2                 # reduce unpredictable high-frequency target motion

    # UAV vs payload 受风比例（同一风向，力度不同）
    wind_scale_uav = 0.4
    wind_scale_payload = 1.0

    # Smooth one-shot perturbation that represents Position-mode handover
    # with an already moving payload. It avoids non-physical joint teleporting.
    enable_startup_gust = True
    startup_gust_accel_range_mps2 = (0.3, 0.8)
    startup_gust_duration_range_s = (0.4, 0.8)
    startup_gust_uav_scale = 0.2
    startup_gust_payload_scale = 1.0

    # Payload-only horizontal force in the UAV body frame. This is kept
    # separate from ambient wind because rotor downwash is platform-relative.
    enable_payload_downwash = True
    downwash_bias_force_range_n = (0.0, 0.8)
    downwash_ou_sigma_n_sqrt_s = 0.03
    downwash_ou_theta = 1.0
    downwash_force_clip_n = 0.9

    # Privileged residual = ambient/startup acceleration + downwash force / m.
    residual_accel_norm_max = 4.5

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
    sigma_tilt_deg = 10.0       # 摆角高斯尺度（deg） 这里本来是10,但为了风扰饱和输出而妥协为40，风扰晃动和reward冲突，agent想拉回来payload但是reward会惩罚大角度
    sigma_swing_deg_s = 40.0   # 摆角角速度高斯尺度（deg/s）
    pos_weight = 0.3           # 位置主项权重
    tilt_weight = 0.15          # 摆角 / 摆速 shaping 权重
    time_penalty = 0.01         # 每秒时间惩罚系数（越大越鼓励快完成）
    death_penalty = 20       # 摔机一次性扣多少（可以先 10，觉得不够再加大）
    action_l2_penalty_scale = 0.03 #0.015 0.02有效抑制 0.03也ok
    action_raw_excess_penalty_scale = 1e-3
    spin_weight = 0.15   # 【新增】自旋惩罚权重
    heading_weight = 0.5     # 【新增】航向对齐权重，用于抑制 Yaw 角度 (P控制)
    action_smooth_penalty_scale = 0.03
    # === 悬停任务（相对每个 env 的原点，ENU）===
    # UAV 根节点固定从 1.5 m 出生。
    start_pos_w = (0.0, 0.0, 1.5)
    # x/y 是 payload 目标；z 是上端参考高度，reset 时再减去本局绳长 L。
    goal_pos_w = (0.0, 0.0, 1.5)
    goal_z_subtract_rope_length = True
    # ---------------- RMA Phase-1 (teacher) ----------------
    proprio_obs_dim = 21          # 可观测部分 还得加last action
    privileged_obs_dim = 5        # e_t = [m_norm, l_norm, wind_norm(3)]
    rma_z_dim = 5                 # z_t dim (这里先设为5，匹配你的“mlw5->z”)
    rma_z_exp_dim = 2             # z前2维当 z_exp（对应慢变量）
    rma_use_mu = True             # Phase1: True; Phase2部署/评估: False
    # Proposed Teacher: exact normalized [mass, rope length] identity path plus
    # an independent learned wind/residual branch.
    rma_context_mode = "split_hard"
    rma_use_physics_anchor = False
    rma_phys_anchor_coef = 0.0
    # split_soft + physics anchor ablation
    # rma_use_mu = True
    # rma_context_mode = "split_soft"
    # rma_z_exp_dim = 2
    # rma_use_physics_anchor = True
    # rma_phys_anchor_coef = 1.0

    # black-box / monolithic RMA ablation
    # rma_use_mu = True
    # rma_context_mode = "monolithic"
    # rma_z_exp_dim = 2   # retained only for logging the first/last dimensions
    # rma_use_physics_anchor = False
    # rma_phys_anchor_coef = 0.0


    # μ(e)->z 的网络结构（Phase1 联训）
    rma_mu_hidden_dims = (64, 64)
    rma_activation = "elu"

    # ---------------------------------------------------------------------
    # Observation noise (Sim2Real)
    # ---------------------------------------------------------------------
    enable_obs_noise = True

    # Conservative per-frame noise. The measured 9.6 mm / 4.61 deg frame
    # changes include real payload motion and therefore are upper bounds, not
    # Gaussian standard deviations.
    obs_noise_e_load_std_m = 0.003     # e_load (m)
    obs_noise_tilt_std_deg = 0.75      # theta_x, theta_y (deg)
    obs_theta_dot_lpf_alpha = 0.5
    obs_noise_v_b_std_mps = 0.01       # body linear velocity (m/s)
    obs_noise_w_b_std_rps = 0.005      # body angular velocity (rad/s)

    # Payload camera/pose pipeline. Train on the measured active-hover central
    # domain; P95+ outages are handled by the deployment safety state machine
    # and separate stress evaluation.
    enable_payload_sensor_gap = True
    payload_sensor_tail_probability = 0.05
    payload_sensor_nominal_hz = (8.0, 20.0)
    payload_sensor_tail_hz = (5.0, 8.0)
    payload_sensor_nominal_delay_s = (0.03, 0.20)
    payload_sensor_tail_delay_s = (0.20, 0.30)
    payload_sensor_valid_probability = (0.93, 0.99)
    payload_sensor_hold_cap_s = 0.25
    payload_position_bias_range_m = (-0.01, 0.01)
    payload_angle_bias_range_deg = (-1.5, 1.5)
    attitude_trim_bias_range_deg = (-1.0, 1.0)
    linear_velocity_bias_range_mps = (-0.015, 0.015)
    body_rate_bias_range_rps = (-0.005, 0.005)

    # Position-style CTBR token generator used only to augment Phase-II history.
    # PPO Teacher actions continue to drive the simulated dynamics.
    position_history_pos_kp = (1.2, 1.2, 1.6)
    position_history_vel_kd = (1.8, 1.8, 2.0)
    # Calibrated conservatively against the delivered Position 50H CTBR range;
    # higher gains over-produced saturated roll-rate prefixes in simulation.
    position_history_attitude_kp = (0.8, 0.8, 1.0)
    position_history_accel_limit_mps2 = (1.0, 1.0, 1.0)
    # The delivered real Position 50H defines the action-prefix center and
    # conservative caps. These values augment Student history only; they never
    # drive the simulated rigid-body dynamics.
    position_history_rate_limit_rps = (0.45, 0.35, 0.30)
    position_history_rate_bias_center_rps = (-0.0634, -0.0234, 0.1880)
    position_history_rate_bias_jitter_rps = (0.08, 0.08, 0.04)

    # ==========================================
    # [新增] Sim2Real: 延迟与动力学建模
    # ==========================================
    # Episode-level provisional PX4/actuator transport gap. The old scalar
    # values are retained as compatibility fallbacks when this profile is off.
    action_delay_steps: int = 0
    action_lpf_alpha: float = 1.0
    action_delay_steps_range = (0, 1)
    action_lpf_alpha_range = (0.75, 1.0)
    # The measured thrust curve is the nominal model. Independent efficiency
    # randomization is disabled because it is not separately identifiable from
    # payload mass in a 21-D history.
    collective_efficiency_range = (1.0, 1.0)
    moment_efficiency_range = (1.0, 1.0)

    # ==========================================
    # [修改] 动作惩罚项
    # ==========================================
    # 必须大幅提高平滑惩罚，逼迫网络输出低频平滑曲线
    action_smooth_penalty_scale: float = 0.03
    action_l2_penalty_scale: float = 0.03
    action_raw_excess_penalty_scale: float = 1e-3
