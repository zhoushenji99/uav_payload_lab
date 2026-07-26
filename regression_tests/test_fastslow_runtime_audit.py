import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/fastslow_runtime.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("fastslow_runtime", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FastSlowRuntimeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_proposed_schedule_is_3s_then_1hz_with_60hz_fast_branch(self):
        schedule = self.module.compute_multirate_schedule(
            history_len=50,
            policy_dt=1.0 / 60.0,
            slow_warmup_sec=3.0,
            slow_update_hz=1.0,
            fast_update_hz=60.0,
        )

        self.assertEqual(schedule.slow_warmup_steps, 180)
        self.assertEqual(schedule.slow_period_steps, 60)
        self.assertEqual(schedule.fast_period_steps, 1)

    def test_filter_alpha_matches_exact_causal_time_constant(self):
        alpha = self.module.causal_ema_alpha(1.0 / 60.0, 0.25)

        self.assertAlmostEqual(alpha, 1.0 - math.exp(-1.0 / 15.0), places=12)
        self.assertAlmostEqual(alpha, 0.06449301496838222, places=12)

    def test_latency_summary_has_mean_p95_and_p99(self):
        result = self.module.summarize_latency_ms([1.0, 2.0, 3.0, 4.0, 100.0])

        self.assertAlmostEqual(result["mean_ms"], 22.0)
        self.assertAlmostEqual(result["p95_ms"], 80.8)
        self.assertAlmostEqual(result["p99_ms"], 96.16)
        self.assertEqual(result["count"], 5)

    def test_ctbr_total_variation_is_channelwise_l1_sum(self):
        actions = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [-0.1, 0.2, -0.3, 0.4],
                [-0.2, 0.1, -0.1, 0.4],
            ]
        )

        result = self.module.compute_action_total_variation(actions)

        np.testing.assert_allclose(result["per_channel"], [0.2, 0.3, 0.5, 0.4])
        self.assertAlmostEqual(result["total"], 1.4)

    def test_action_band_energy_detects_10hz_but_not_2hz(self):
        fs = 60.0
        t = np.arange(600) / fs
        actions = np.column_stack(
            [
                np.sin(2.0 * np.pi * 10.0 * t),
                np.sin(2.0 * np.pi * 2.0 * t),
                np.zeros_like(t),
                np.zeros_like(t),
            ]
        )

        result = self.module.compute_action_band_energy(actions, fs, 5.0, 30.0)

        self.assertGreater(result["per_channel"][0], 0.45)
        self.assertLess(result["per_channel"][1], 1e-8)
        self.assertGreater(result["fraction_per_channel"][0], 0.99)
        self.assertLess(result["fraction_per_channel"][1], 1e-8)

    def test_gust_response_latency_uses_pre_event_fast_context(self):
        fs = 60.0
        n = 120
        gust = np.zeros((n, 3))
        gust[30:, 0] = 0.5
        z_fast = np.zeros((n, 3))
        z_fast[33:, 0] = 0.2

        result = self.module.compute_gust_response_latency(
            gust,
            z_fast,
            policy_dt=1.0 / fs,
            gust_event_threshold=0.1,
            fast_response_threshold=0.1,
            response_window_sec=0.5,
        )

        self.assertEqual(result["event_count"], 1)
        self.assertEqual(result["responded_count"], 1)
        self.assertAlmostEqual(result["latencies_s"][0], 3.0 / fs)


if __name__ == "__main__":
    unittest.main()
