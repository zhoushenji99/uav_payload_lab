from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SIM2REAL = REPO / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real"


def read(name: str) -> str:
    return (SIM2REAL / name).read_text(encoding="utf-8")


def load_runtime_module():
    path = SIM2REAL / "fastslow_runtime.py"
    spec = importlib.util.spec_from_file_location("fastslow_runtime_eval_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FastSlowEvaluationOverrideChecks(unittest.TestCase):
    def test_cli_declares_fixed_physics_and_no_wind_switches(self):
        play = read("play_student_phase2.py")
        self.assertIn("--eval_payload_mass_kg", play)
        self.assertIn("--eval_rope_length_m", play)
        self.assertIn("--eval_disable_wind", play)

    def test_config_keeps_fixed_values_separate_from_training_ranges(self):
        cfg = read("meta_uav_env_cfg.py")
        for field in [
            "eval_fixed_payload_mass_kg",
            "eval_fixed_rope_length_m",
            "eval_disable_wind",
        ]:
            with self.subTest(field=field):
                self.assertIn(field, cfg)

        self.assertIn("payload_mass_range = (0.3, 0.8)", cfg)
        self.assertIn("rope_length_range = (0.25,0.8)", cfg)

    def test_reset_path_uses_optional_fixed_values(self):
        env = read("meta_uav_env.py")
        self.assertIn("eval_fixed_payload_mass_kg", env)
        self.assertIn("eval_fixed_rope_length_m", env)
        self.assertIn("eval_disable_wind", env)

    def test_rollout_audit_records_total_wind_and_overrides(self):
        play = read("play_student_phase2.py")
        for column in [
            "wind_acc_x_mps2",
            "wind_acc_y_mps2",
            "wind_acc_z_mps2",
        ]:
            with self.subTest(column=column):
                self.assertIn(column, play)
        self.assertIn('"evaluation_overrides"', play)

    def test_override_validation_accepts_midpoint_values(self):
        runtime = load_runtime_module()
        result = runtime.validate_evaluation_overrides(
            payload_mass_kg=0.55,
            rope_length_m=0.525,
            disable_wind=True,
            payload_mass_range=(0.3, 0.8),
            rope_length_range=(0.25, 0.8),
        )
        self.assertEqual(result["payload_mass_kg"], 0.55)
        self.assertEqual(result["rope_length_m"], 0.525)
        self.assertTrue(result["disable_wind"])

    def test_override_validation_rejects_out_of_range_values(self):
        runtime = load_runtime_module()
        with self.assertRaisesRegex(ValueError, "payload mass"):
            runtime.validate_evaluation_overrides(
                payload_mass_kg=0.81,
                rope_length_m=None,
                disable_wind=False,
                payload_mass_range=(0.3, 0.8),
                rope_length_range=(0.25, 0.8),
            )
        with self.assertRaisesRegex(ValueError, "rope length"):
            runtime.validate_evaluation_overrides(
                payload_mass_kg=None,
                rope_length_m=0.24,
                disable_wind=False,
                payload_mass_range=(0.3, 0.8),
                rope_length_range=(0.25, 0.8),
            )


if __name__ == "__main__":
    unittest.main()
