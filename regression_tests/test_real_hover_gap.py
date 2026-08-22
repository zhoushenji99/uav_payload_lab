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


class RealHoverGapStaticIntegrationTests(unittest.TestCase):
    def test_config_keeps_interface_and_encodes_measured_uav(self):
        source = CFG_PATH.read_text(encoding="utf-8")
        self.assertIn('real_hover_gap_profile = "real_hover_gap_v1"', source)
        self.assertIn("uav_mass_kg = 3.230", source)
        self.assertIn("uav_com_m = (0.00389, 0.02922, 0.17422)", source)
        self.assertIn("uav_inertia_diag_kg_m2 = (0.0763, 0.0762, 0.1500)", source)
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
        self.assertIn("startup_gust_accel_range_mps2 = (0.5, 1.5)", source)
        self.assertIn("startup_gust_duration_range_s = (0.4, 1.0)", source)
        self.assertIn("startup_gust_uav_scale = 0.4", source)
        self.assertIn("startup_gust_payload_scale = 1.0", source)
        self.assertIn("downwash_bias_force_range_n = (0.0, 0.8)", source)
        self.assertIn("downwash_ou_sigma_n_sqrt_s = 0.15", source)
        self.assertIn("downwash_force_clip_n = 1.2", source)
        self.assertIn("residual_accel_norm_max = 5.5", source)

    def test_config_encodes_payload_vision_transport_and_bias(self):
        cfg = CFG_PATH.read_text(encoding="utf-8")
        env = ENV_PATH.read_text(encoding="utf-8")
        self.assertIn("payload_sensor_tail_probability = 0.15", cfg)
        self.assertIn("payload_sensor_nominal_hz = (12.0, 30.0)", cfg)
        self.assertIn("payload_sensor_tail_hz = (5.0, 12.0)", cfg)
        self.assertIn("payload_sensor_nominal_delay_s = (0.03, 0.15)", cfg)
        self.assertIn("payload_sensor_tail_delay_s = (0.15, 0.30)", cfg)
        self.assertIn("payload_sensor_valid_probability = (0.92, 0.98)", cfg)
        self.assertIn("payload_sensor_hold_cap_s = 0.50", cfg)
        self.assertIn("def _transport_payload_observation", env)
        self.assertIn("def _reset_payload_sensor_gap", env)


if __name__ == "__main__":
    unittest.main()
