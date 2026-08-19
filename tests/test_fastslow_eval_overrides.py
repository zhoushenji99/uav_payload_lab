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
        self.assertIn("--eval_wind_scale", play)
        self.assertIn("--eval_wind_mode", play)
        self.assertIn("--eval_wind_amplitude_mps2", play)
        self.assertIn("--eval_wind_frequency_hz", play)
        self.assertIn("--eval_wind_start_sec", play)
        self.assertIn("--eval_wind_axis", play)
        self.assertIn("--eval_wind_phase_rad", play)

    def test_config_keeps_fixed_values_separate_from_training_ranges(self):
        cfg = read("meta_uav_env_cfg.py")
        for field in [
            "eval_fixed_payload_mass_kg",
            "eval_fixed_rope_length_m",
            "eval_disable_wind",
            "eval_wind_scale",
            "eval_wind_mode",
            "eval_wind_amplitude_mps2",
            "eval_wind_frequency_hz",
            "eval_wind_start_sec",
            "eval_wind_axis",
            "eval_wind_phase_rad",
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
            wind_scale=2.0,
            wind_mode="sinusoid",
            wind_amplitude_mps2=1.0,
            wind_frequency_hz=0.688,
            wind_start_sec=3.0,
            wind_axis="x",
            wind_phase_rad=0.0,
            payload_mass_range=(0.3, 0.8),
            rope_length_range=(0.25, 0.8),
        )
        self.assertEqual(result["payload_mass_kg"], 0.55)
        self.assertEqual(result["rope_length_m"], 0.525)
        self.assertTrue(result["disable_wind"])
        self.assertEqual(result["wind_scale"], 2.0)
        self.assertEqual(result["wind_mode"], "sinusoid")
        self.assertEqual(result["wind_amplitude_mps2"], 1.0)
        self.assertEqual(result["wind_frequency_hz"], 0.688)
        self.assertEqual(result["wind_start_sec"], 3.0)
        self.assertEqual(result["wind_axis"], "x")
        self.assertEqual(result["wind_phase_rad"], 0.0)

    def test_override_validation_rejects_invalid_wind_scale(self):
        runtime = load_runtime_module()
        for wind_scale in (-0.1, float("nan"), float("inf")):
            with self.subTest(wind_scale=wind_scale):
                with self.assertRaisesRegex(ValueError, "wind scale"):
                    runtime.validate_evaluation_overrides(
                        payload_mass_kg=0.55,
                        rope_length_m=0.525,
                        disable_wind=False,
                        wind_scale=wind_scale,
                        wind_mode="training",
                        wind_amplitude_mps2=1.0,
                        wind_frequency_hz=1.0,
                        wind_start_sec=3.0,
                        wind_axis="x",
                        wind_phase_rad=0.0,
                        payload_mass_range=(0.3, 0.8),
                        rope_length_range=(0.25, 0.8),
                    )

    def test_override_validation_rejects_invalid_sinusoid_parameters(self):
        runtime = load_runtime_module()
        common = dict(
            payload_mass_kg=0.55,
            rope_length_m=0.525,
            disable_wind=False,
            wind_scale=1.0,
            wind_mode="sinusoid",
            wind_amplitude_mps2=1.0,
            wind_frequency_hz=1.0,
            wind_start_sec=3.0,
            wind_axis="x",
            wind_phase_rad=0.0,
            payload_mass_range=(0.3, 0.8),
            rope_length_range=(0.25, 0.8),
        )
        invalid = [
            ("wind_mode", "square"),
            ("wind_amplitude_mps2", -0.1),
            ("wind_frequency_hz", 0.0),
            ("wind_start_sec", -0.1),
            ("wind_axis", "z"),
            ("wind_phase_rad", float("nan")),
        ]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                kwargs = dict(common)
                kwargs[key] = value
                with self.assertRaises(ValueError):
                    runtime.validate_evaluation_overrides(**kwargs)

    def test_override_validation_rejects_out_of_range_values(self):
        runtime = load_runtime_module()
        with self.assertRaisesRegex(ValueError, "payload mass"):
            runtime.validate_evaluation_overrides(
                payload_mass_kg=0.81,
                rope_length_m=None,
                disable_wind=False,
                wind_scale=1.0,
                wind_mode="training",
                wind_amplitude_mps2=1.0,
                wind_frequency_hz=1.0,
                wind_start_sec=3.0,
                wind_axis="x",
                wind_phase_rad=0.0,
                payload_mass_range=(0.3, 0.8),
                rope_length_range=(0.25, 0.8),
            )
        with self.assertRaisesRegex(ValueError, "rope length"):
            runtime.validate_evaluation_overrides(
                payload_mass_kg=None,
                rope_length_m=0.24,
                disable_wind=False,
                wind_scale=1.0,
                wind_mode="training",
                wind_amplitude_mps2=1.0,
                wind_frequency_hz=1.0,
                wind_start_sec=3.0,
                wind_axis="x",
                wind_phase_rad=0.0,
                payload_mass_range=(0.3, 0.8),
                rope_length_range=(0.25, 0.8),
            )


if __name__ == "__main__":
    unittest.main()
