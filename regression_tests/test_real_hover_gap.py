import importlib.util
from pathlib import Path
import sys
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/real_hover_gap.py"
)
CFG_PATH = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env_cfg.py"
)
ENV_PATH = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env.py"
)
COLLECT_PATH = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/collect_z_dataset.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("real_hover_gap", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RealHoverGapHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_inertia_triangle_is_enforced(self):
        self.module.validate_inertia_diagonal((0.0763, 0.0762, 0.1500))
        with self.assertRaisesRegex(ValueError, "triangle"):
            self.module.validate_inertia_diagonal((0.073, 0.073, 0.160))

    def test_flat_inertia_has_physx_column_major_shape(self):
        inertia = self.module.diagonal_inertia_flat((1.0, 2.0, 2.5))
        torch.testing.assert_close(
            inertia,
            torch.tensor([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.5]),
        )

    def test_half_sine_is_zero_at_endpoints_and_one_at_midpoint(self):
        elapsed = torch.tensor([0.0, 0.5, 1.0, 1.5])
        duration = torch.ones(4)
        torch.testing.assert_close(
            self.module.half_sine_profile(elapsed, duration),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_select_delayed_actions_uses_each_environment_delay(self):
        queue = torch.tensor(
            [
                [[0.0], [1.0], [2.0]],
                [[10.0], [11.0], [12.0]],
                [[20.0], [21.0], [22.0]],
            ]
        )
        out = self.module.select_delayed_actions(queue, torch.tensor([0, 1, 2]))
        torch.testing.assert_close(out[:, 0], torch.tensor([2.0, 11.0, 20.0]))

    def test_select_delayed_ring_uses_per_environment_write_index(self):
        ring = torch.tensor(
            [
                [[0.0], [1.0], [2.0], [3.0]],
                [[10.0], [11.0], [12.0], [13.0]],
            ]
        )
        out = self.module.select_delayed_ring(
            ring,
            write_index=torch.tensor([0, 3]),
            delay_steps=torch.tensor([1, 2]),
        )
        torch.testing.assert_close(out[:, 0], torch.tensor([3.0, 11.0]))

    def test_delayed_relative_payload_is_composed_with_current_uav_pose(self):
        uav_pos = torch.tensor([[2.0, 3.0, 1.5]])
        yaw_90_wxyz = torch.tensor(
            [[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]]
        )
        delayed_uav_to_payload_b = torch.tensor([[1.0, 0.0, -0.7]])
        actual = self.module.compose_delayed_payload_world_position(
            uav_pos,
            yaw_90_wxyz,
            delayed_uav_to_payload_b,
        )
        torch.testing.assert_close(
            actual,
            torch.tensor([[2.0, 4.0, 0.8]]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_payload_rate_lpf_filters_only_an_initialized_update(self):
        actual = self.module.update_payload_rate_lpf(
            previous_filtered_rate=torch.tensor([[4.0, -2.0]]),
            raw_rate=torch.tensor([[10.0, 6.0]]),
            initialized=torch.tensor([True]),
            update=torch.tensor([True]),
            alpha=0.5,
        )
        torch.testing.assert_close(actual, torch.tensor([[7.0, 2.0]]))

    def test_payload_rate_lpf_first_measurement_is_zero(self):
        actual = self.module.update_payload_rate_lpf(
            previous_filtered_rate=torch.tensor([[4.0, -2.0]]),
            raw_rate=torch.tensor([[10.0, 6.0]]),
            initialized=torch.tensor([False]),
            update=torch.tensor([True]),
            alpha=0.5,
        )
        torch.testing.assert_close(actual, torch.zeros(1, 2))

    def test_payload_rate_lpf_dropout_holds_previous_value(self):
        actual = self.module.update_payload_rate_lpf(
            previous_filtered_rate=torch.tensor([[4.0, -2.0]]),
            raw_rate=torch.tensor([[10.0, 6.0]]),
            initialized=torch.tensor([True]),
            update=torch.tensor([False]),
            alpha=0.5,
        )
        torch.testing.assert_close(actual, torch.tensor([[4.0, -2.0]]))

    def test_normalized_quadratic_thrust_curve_matches_measured_endpoints(self):
        signal = torch.tensor([0.0, 0.58, 1.0])
        ratio = self.module.normalized_quadratic_thrust_ratio(
            signal,
            (14.584, 5.438, 0.0),
        )
        expected_mid = (14.584 * 0.58**2 + 5.438 * 0.58) / (14.584 + 5.438)
        torch.testing.assert_close(
            ratio,
            torch.tensor([0.0, expected_mid, 1.0]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_normalized_quadratic_thrust_curve_inverse_round_trip(self):
        signal = torch.linspace(0.0, 1.0, 21)
        ratio = self.module.normalized_quadratic_thrust_ratio(
            signal,
            (14.584, 5.438, 0.0),
        )
        recovered = self.module.inverse_normalized_quadratic_thrust_ratio(
            ratio,
            (14.584, 5.438, 0.0),
        )
        torch.testing.assert_close(recovered, signal, atol=1e-6, rtol=0.0)

    def test_lumped_payload_mass_tracks_rope_length_and_ballast(self):
        total_mass, rope_mass = self.module.compose_lumped_payload_mass(
            rope_length_m=torch.tensor([0.25, 0.525, 0.8]),
            ballast_mass_kg=torch.tensor([0.2, 0.5, 0.8]),
            rope_length_range_m=(0.25, 0.8),
            rope_mass_range_kg=(0.010, 0.030),
            fixed_moving_mass_kg=0.10265,
        )
        torch.testing.assert_close(
            rope_mass,
            torch.tensor([0.010, 0.020, 0.030]),
            atol=1e-7,
            rtol=0.0,
        )
        torch.testing.assert_close(
            total_mass,
            torch.tensor([0.31265, 0.62265, 0.93265]),
            atol=1e-7,
            rtol=0.0,
        )

    def test_physical_context_normalization_matches_v87_contract(self):
        mass = torch.tensor([0.31265, 0.62265, 0.93265])
        length = torch.tensor([0.25, 0.6004, 0.8])
        z0, z1 = self.module.normalize_physical_context(
            mass,
            length,
            payload_mass_range_kg=(0.31265, 0.93265),
            rope_length_range_m=(0.25, 0.8),
        )
        torch.testing.assert_close(z0, torch.tensor([0.0, 0.5, 1.0]))
        torch.testing.assert_close(
            z1,
            torch.tensor([0.0, 0.6370909, 1.0]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_rate_gain_is_derived_from_inertia_and_time_constant(self):
        gain = self.module.rate_gain_from_time_constant(
            torch.tensor([[0.0763, 0.0762, 0.1500]]),
            torch.tensor([[0.10, 0.20, 0.30]]),
        )
        torch.testing.assert_close(
            gain,
            torch.tensor([[0.763, 0.381, 0.500]]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_rate_gain_rejects_nonpositive_time_constant(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            self.module.rate_gain_from_time_constant(
                torch.ones(1, 3),
                torch.tensor([[0.1, 0.0, 0.2]]),
            )


class RealHoverGapStaticIntegrationTests(unittest.TestCase):
    def test_config_keeps_interface_and_encodes_measured_uav(self):
        source = CFG_PATH.read_text(encoding="utf-8")
        self.assertIn('real_hover_gap_profile = "first_flight_gap_v3"', source)
        self.assertIn("uav_bare_mass_kg = 3.100", source)
        self.assertIn("uav_fixed_gimbal_mass_kg = 0.03175", source)
        self.assertIn(
            "uav_mass_kg = uav_bare_mass_kg + uav_fixed_gimbal_mass_kg",
            source,
        )
        self.assertIn("uav_com_m = (0.00389, 0.02922, 0.17422)", source)
        self.assertIn("uav_inertia_diag_kg_m2 = (0.0763, 0.0762, 0.1500)", source)
        self.assertIn("uav_mass_scale_range = (0.99, 1.01)", source)
        self.assertIn("uav_com_offset_range_m = (-0.002, 0.002)", source)
        self.assertIn("uav_inertia_scale_range = (0.95, 1.05)", source)
        self.assertIn("proprio_obs_dim = 21", source)
        self.assertIn("privileged_obs_dim = 5", source)

    def test_env_uses_runtime_physx_overrides_without_editing_usd(self):
        source = ENV_PATH.read_text(encoding="utf-8")
        self.assertIn("def _apply_uav_physics", source)
        self.assertIn("set_masses", source)
        self.assertIn("set_coms", source)
        self.assertIn("set_inertias", source)

    def test_config_encodes_startup_gust_and_payload_downwash(self):
        source = CFG_PATH.read_text(encoding="utf-8")
        self.assertIn("startup_gust_accel_range_mps2 = (0.3, 0.8)", source)
        self.assertIn("startup_gust_duration_range_s = (0.4, 0.8)", source)
        self.assertIn("startup_gust_uav_scale = 0.2", source)
        self.assertIn("startup_gust_payload_scale = 1.0", source)
        self.assertIn("downwash_bias_force_range_n = (0.0, 0.8)", source)
        self.assertIn("downwash_ou_sigma_n_sqrt_s = 0.03", source)
        self.assertIn("downwash_force_clip_n = 0.9", source)
        self.assertIn("residual_accel_norm_max = 4.5", source)

    def test_config_uses_identifiable_indoor_wind_domain(self):
        source = CFG_PATH.read_text(encoding="utf-8")
        self.assertIn("wind_mean_accel_max = 0.1", source)
        self.assertIn("wind_gust_accel_max = 0.3", source)
        self.assertIn("wind_total_accel_max = 0.6", source)
        self.assertIn("wind_gust_dt_min = 1.0", source)
        self.assertIn("wind_gust_dt_max = 3.0", source)
        self.assertIn("wind_ou_sigma = 0.2", source)

    def test_config_encodes_payload_vision_transport_and_bias(self):
        cfg = CFG_PATH.read_text(encoding="utf-8")
        env = ENV_PATH.read_text(encoding="utf-8")
        self.assertIn("payload_sensor_tail_probability = 0.05", cfg)
        self.assertIn("payload_sensor_nominal_hz = (8.0, 30.0)", cfg)
        self.assertIn("payload_sensor_tail_hz = (5.0, 8.0)", cfg)
        self.assertIn("payload_sensor_nominal_delay_s = (0.03, 0.20)", cfg)
        self.assertIn("payload_sensor_tail_delay_s = (0.20, 0.50)", cfg)
        self.assertIn("payload_sensor_valid_probability = (0.93, 0.99)", cfg)
        self.assertIn("payload_sensor_hold_cap_s = 0.5", cfg)
        self.assertIn("payload_position_bias_range_m = (-0.01, 0.01)", cfg)
        self.assertIn("payload_angle_bias_range_deg = (-1.5, 1.5)", cfg)
        self.assertIn("linear_velocity_bias_range_mps = (-0.015, 0.015)", cfg)
        self.assertIn("body_rate_bias_range_rps = (-0.005, 0.005)", cfg)
        self.assertIn("enable_obs_noise = True", cfg)
        self.assertIn("obs_noise_e_load_std_m = 0.003", cfg)
        self.assertIn("obs_noise_tilt_std_deg = 0.75", cfg)
        self.assertIn("obs_noise_v_b_std_mps = 0.01", cfg)
        self.assertIn("obs_noise_w_b_std_rps = 0.005", cfg)
        self.assertIn("def _transport_payload_observation", env)
        self.assertIn("def _reset_payload_sensor_gap", env)

    def test_config_and_env_use_randomized_body_rate_time_constants(self):
        cfg = CFG_PATH.read_text(encoding="utf-8")
        env = ENV_PATH.read_text(encoding="utf-8")
        self.assertIn("ctbr_rate_time_constant_range_s", cfg)
        self.assertIn("((0.08, 0.25), (0.08, 0.25), (0.12, 0.45))", cfg)
        self.assertIn("def _sample_body_rate_dynamics", env)
        self.assertIn("rate_gain_from_time_constant", env)
        self.assertIn("self._ctbr_rate_kp_per_env", env)
        self.assertIn("self._ctbr_rate_time_constant_s", env)
        self.assertIn("moment_cmd = self._ctbr_rate_kp_per_env * rate_error", env)

    def test_config_encodes_per_environment_action_transport(self):
        cfg = CFG_PATH.read_text(encoding="utf-8")
        env = ENV_PATH.read_text(encoding="utf-8")
        self.assertIn("action_delay_steps_range = (0, 2)", cfg)
        self.assertIn("action_lpf_alpha_range = (0.35, 0.85)", cfg)
        self.assertIn("collective_efficiency_range = (1.0, 1.0)", cfg)
        self.assertIn("moment_efficiency_range = (0.85, 1.15)", cfg)
        self.assertIn("_action_delay_steps_per_env", env)
        self.assertIn("_action_lpf_alpha_per_env", env)
        self.assertIn("_collective_efficiency", env)
        self.assertIn("_moment_efficiency", env)

    def test_payload_sensor_gap_matches_v89_contract(self):
        cfg = CFG_PATH.read_text(encoding="utf-8")
        self.assertIn("payload_sensor_nominal_hz = (8.0, 30.0)", cfg)
        self.assertIn("payload_sensor_tail_delay_s = (0.20, 0.50)", cfg)
        self.assertIn("payload_sensor_hold_cap_s = 0.5", cfg)

    def test_policy_history_uses_last_transmitted_clamped_ctbr(self):
        env = ENV_PATH.read_text(encoding="utf-8")
        self.assertIn("self._last_transmitted_actions", env)
        self.assertIn("self._last_transmitted_actions = shape_ctbr_torch(", env)
        self.assertIn("self._action_queue[:, -1, :] = self._last_transmitted_actions", env)
        self.assertEqual(env.count("self._last_transmitted_actions, # 17-20"), 2)

    def test_gap_audit_covers_identifiability_sensitive_ranges(self):
        env = ENV_PATH.read_text(encoding="utf-8")
        for key in (
            '"enable_real_hover_gap"',
            '"uav_mass_scale_range"',
            '"rope_length_range"',
            '"payload_mass_range"',
            '"enable_wind"',
            '"wind_ou_sigma"',
            '"enable_startup_gust"',
            '"enable_payload_downwash"',
            '"enable_payload_sensor_gap"',
            '"payload_sensor_nominal_delay_s"',
            '"payload_sensor_valid_probability"',
            '"action_lpf_alpha_range"',
            '"collective_efficiency_range"',
            '"moment_efficiency_range"',
            '"ctbr_rate_time_constant_range_s"',
            '"residual_accel_norm_max"',
            '"observation_history_action_source": "last_transmitted_clamped_ctbr"',
        ):
            self.assertIn(key, env)

    def test_env_applies_and_inverts_measured_quadratic_thrust_curve(self):
        cfg = CFG_PATH.read_text(encoding="utf-8")
        env = ENV_PATH.read_text(encoding="utf-8")
        self.assertIn('ctbr_thrust_model = "normalized_quadratic"', cfg)
        self.assertIn("normalized_quadratic_thrust_ratio", env)
        self.assertIn("inverse_normalized_quadratic_thrust_ratio", env)
        self.assertIn("def _collective_signal_to_thrust_ratio", env)
        self.assertIn("def _thrust_ratio_to_collective_signal", env)

    def test_config_and_reset_use_lumped_payload_mass_components(self):
        cfg = CFG_PATH.read_text(encoding="utf-8")
        env = ENV_PATH.read_text(encoding="utf-8")
        self.assertIn("payload_ballast_mass_range = (0.2, 0.8)", cfg)
        self.assertIn("payload_fixed_moving_mass_kg = 0.10265", cfg)
        self.assertIn("rope_mass_range_kg = (0.010, 0.030)", cfg)
        self.assertIn("payload_mass_range = (0.31265, 0.93265)", cfg)
        self.assertIn("compose_lumped_payload_mass", env)
        self.assertIn("self._payload_ballast_mass", env)
        self.assertIn("self._rope_mass", env)

    def test_dataset_metadata_persists_real_hover_gap_profile(self):
        source = COLLECT_PATH.read_text(encoding="utf-8")
        for key in (
            '"real_hover_gap_profile"',
            '"uav_mass_kg"',
            '"payload_sensor_nominal_hz"',
            '"payload_sensor_tail_hz"',
            '"startup_gust_accel_range_mps2"',
            '"downwash_bias_force_range_n"',
            '"action_delay_steps_range"',
            '"real_hover_gap_audit"',
        ):
            self.assertIn(key, source)
        self.assertIn("residual_accel_norm(3)", source)

    def test_collect_supports_labeled_position_history_mixture(self):
        source = COLLECT_PATH.read_text(encoding="utf-8")
        self.assertIn('"--position_history_ratio"', source)
        self.assertIn('"--position_precondition_sec"', source)
        self.assertIn("compute_position_hold_ctbr", source)
        self.assertIn("position_precondition_steps", source)
        self.assertIn("step_count == position_precondition_steps", source)
        self.assertIn("select_control_actions", source)
        self.assertIn("active_position_mask", source)
        self.assertNotIn("feat[position_history_mask, 17:21]", source)
        self.assertIn('"history_source"', source)
        self.assertIn('"position_history_ratio_requested"', source)
        self.assertIn('"position_history_ratio_realized"', source)

    def test_collect_never_saves_partially_refilled_histories_after_reset(self):
        source = COLLECT_PATH.read_text(encoding="utf-8")
        self.assertIn("history_fill_count", source)
        self.assertIn("valid_history_mask = history_fill_count >= history_len", source)
        self.assertIn("history_fill_count[done_mask] = 0", source)
        self.assertIn("torch.cat(shard_inputs, dim=0)", source)
        self.assertNotIn("torch.stack(shard_inputs, dim=0)", source)

    def test_position_history_controller_keeps_ctbr_contract(self):
        cfg = CFG_PATH.read_text(encoding="utf-8")
        source = ENV_PATH.read_text(encoding="utf-8")
        self.assertIn("position_history_attitude_kp = (0.8, 0.8, 1.0)", cfg)
        self.assertIn("position_history_rate_limit_rps = (0.45, 0.35, 0.30)", cfg)
        self.assertIn(
            "position_history_rate_bias_center_rps = (-0.0634, -0.0234, 0.1880)",
            cfg,
        )
        self.assertIn("position_history_rate_bias_jitter_rps = (0.08, 0.08, 0.04)", cfg)
        self.assertIn("def compute_position_hold_ctbr", source)
        self.assertIn(
            "roll_des = torch.atan2(accel_cmd_w[:, 1], vertical_accel)", source
        )
        self.assertIn("self._thrust_ratio_to_collective_signal", source)
        self.assertIn("self._ctbr_rate_sign", source)
        self.assertIn("self._decode_px4_ctbr_action", source)
        self.assertIn("self._position_history_rate_bias_px4", source)


if __name__ == "__main__":
    unittest.main()
