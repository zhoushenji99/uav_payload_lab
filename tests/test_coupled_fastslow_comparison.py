from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real"
    / "build_coupled_fastslow_comparison.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "coupled_fastslow_comparison_test", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_rollout(*, offset: float = 0.0, samples: int = 120):
    time = np.arange(samples, dtype=float) / 60.0
    frame = pd.DataFrame(
        {
            "time_s": time,
            "payload_err_x": np.exp(-time) + offset,
            "payload_err_y": 0.1 * np.sin(time),
            "payload_err_z": 0.05 * np.cos(time),
            "theta_x_deg": 2.0 * np.sin(time) + offset,
            "theta_y_deg": 1.5 * np.cos(time),
            "rope_length_m": 0.525,
            "payload_mass_kg": 0.55,
            "zT0": 1.0,
            "zT1": 2.0,
            "zT2": 3.0,
            "zT3": 4.0,
            "zT4": 5.0,
            "zH0": 1.0 + offset,
            "zH1": 2.0 + offset,
            "zH2": 3.0 + offset,
            "zH3": 4.0 + offset,
            "zH4": 5.0 + offset,
            "z_rmse": abs(offset),
            "slow_batch_calls": np.arange(1, samples + 1),
            "fast_batch_calls": np.arange(1, samples + 1),
            "full_batch_calls": np.arange(1, samples + 1),
            "actor_inference_ms": 0.1,
            "end_to_end_inference_ms": 0.5,
            "gust_x_mps2": 0.1 * np.sin(time),
            "gust_y_mps2": 0.0,
            "gust_z_mps2": 0.0,
            "wind_acc_x_mps2": 0.2 * np.sin(time),
            "wind_acc_y_mps2": 0.0,
            "wind_acc_z_mps2": 0.0,
            "a0_clamp": -0.5 + 0.01 * np.sin(8.0 * time) + offset,
            "a1_clamp": 0.1 * np.sin(6.0 * time),
            "a2_clamp": 0.1 * np.cos(7.0 * time),
            "a3_clamp": 0.05 * np.sin(5.0 * time),
        }
    )
    return frame


class CoupledFastSlowComparisonChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_cross_method_audit_requires_exact_exogenous_inputs(self):
        frames = {
            "structured_teacher": synthetic_rollout(),
            "structured_student": synthetic_rollout(offset=0.01),
            "coupled_teacher": synthetic_rollout(),
            "coupled_student": synthetic_rollout(offset=0.02),
        }
        audit = self.module.audit_cross_method_pair(frames)
        self.assertTrue(audit["strict_pair"])

        frames["coupled_student"].loc[5, "wind_acc_x_mps2"] += 0.01
        audit = self.module.audit_cross_method_pair(frames)
        self.assertFalse(audit["strict_pair"])

        frames["coupled_student"] = synthetic_rollout(samples=100)
        audit = self.module.audit_cross_method_pair(frames)
        self.assertFalse(audit["strict_pair"])
        self.assertFalse(audit["equal_lengths"])

    def test_standardized_context_error_is_scale_aware(self):
        frame = synthetic_rollout(offset=0.1)
        result = self.module.compute_context_metrics(
            frame, z_std=[1.0, 2.0, 4.0, 5.0, 10.0]
        )
        expected = np.sqrt(
            np.mean(np.square(np.array([0.1, 0.05, 0.025, 0.02, 0.01])))
        )
        self.assertAlmostEqual(result["context_nrmse"], expected)
        self.assertEqual(len(result["context_nrmse_dim"]), 5)

    def test_deployment_gap_uses_teacher_student_closed_loop_traces(self):
        teacher = synthetic_rollout()
        student = synthetic_rollout(offset=0.05)
        gap = self.module.compute_deployment_gap(teacher, student)
        self.assertGreater(gap["position_trace_gap_rmse_m"], 0.0)
        self.assertGreater(gap["action_trace_gap_rmse"], 0.0)
        self.assertEqual(gap["common_samples"], len(teacher))

    def test_coupled_metrics_do_not_claim_physical_z_semantics(self):
        metrics = self.module.compute_method_metrics(
            synthetic_rollout(offset=0.01),
            z_std=[1.0] * 5,
            physical_context=False,
        )
        self.assertTrue(np.isnan(metrics["mass_rmse_kg"]))
        self.assertTrue(np.isnan(metrics["rope_length_rmse_m"]))
        self.assertTrue(np.isfinite(metrics["context_nrmse"]))

    def test_rollout_command_uses_monolithic_lineage_and_fixed_overrides(self):
        scenario = {
            "id": "seed46_nowind_mid",
            "seed": 46,
            "fixed_payload_mass_kg": 0.55,
            "fixed_rope_length_m": 0.525,
            "disable_wind": True,
        }
        teacher = self.module.build_rollout_command(
            mode="teacher",
            scenario=scenario,
            csv_path=Path("/tmp/teacher.csv"),
            checkpoint=Path("/tmp/model.pt"),
            encoder=Path("/tmp/student.pth"),
        )
        student = self.module.build_rollout_command(
            mode="student",
            scenario=scenario,
            csv_path=Path("/tmp/student.csv"),
            checkpoint=Path("/tmp/model.pt"),
            encoder=Path("/tmp/student.pth"),
        )
        for command in [teacher, student]:
            self.assertIn("--rma_context_mode", command)
            self.assertIn("monolithic", command)
            self.assertIn("--eval_payload_mass_kg", command)
            self.assertIn("--eval_rope_length_m", command)
            self.assertIn("--eval_disable_wind", command)
        self.assertNotIn("--encoder", teacher)
        self.assertIn("--encoder", student)

    def test_rollout_environment_replaces_noninteractive_dumb_terminal(self):
        environment = self.module.build_rollout_environment(
            {"TERM": "dumb", "KEEP_ME": "yes"}
        )
        self.assertEqual(environment["TERM"], "xterm")
        self.assertEqual(environment["KEEP_ME"], "yes")


if __name__ == "__main__":
    unittest.main()
