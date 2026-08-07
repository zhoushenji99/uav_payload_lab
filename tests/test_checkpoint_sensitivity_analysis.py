import sys
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd


MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "uav_payload_lab"
    / "uav_payload_lab"
    / "tasks"
    / "direct"
    / "uav_payload_sim2real"
)
sys.path.insert(0, str(MODULE_DIR))

from analyze_checkpoint_sensitivity import (  # noqa: E402
    audit_exact_exogenous,
    build_analysis,
    build_loss_comparison_table,
    build_requested_paper_figures,
    compute_swing_energy_evidence,
    compute_teacher_metrics,
)


class CheckpointSensitivityAnalysisTests(unittest.TestCase):
    def test_compute_swing_energy_evidence_preserves_power_decomposition(self):
        time_s = np.linspace(0.0, 1.0, 11)
        frame = pd.DataFrame(
            {
                "time_s": time_s,
                "uav_px": time_s**2,
                "uav_py": 0.5 * time_s**2,
                "uav_pz": np.full_like(time_s, 2.0),
                "theta_x_deg": np.linspace(5.0, 1.0, len(time_s)),
                "theta_y_deg": np.linspace(-2.0, 0.0, len(time_s)),
                "theta_dot_x_deg_s": np.full_like(time_s, -4.0),
                "theta_dot_y_deg_s": np.full_like(time_s, 2.0),
                "rope_length_m": np.full_like(time_s, 0.6),
            }
        )

        energy = compute_swing_energy_evidence(
            frame, smooth_window_s=0.0
        )

        self.assertEqual(len(energy), len(frame))
        np.testing.assert_allclose(
            energy["P_model"],
            energy["P_xy"] + energy["P_param"],
            rtol=0.0,
            atol=1e-12,
        )
        self.assertTrue(np.isfinite(energy["E_hat"]).all())

    def test_build_loss_comparison_table_reports_exact_relative_changes(self):
        records = [
            {
                "generation": "old",
                "method": "Coupled",
                "is_reference": True,
                "best_val": 0.01,
                "final_val": 0.02,
                "final_train": 0.04,
            },
            {
                "generation": "old",
                "method": "Decoupled",
                "is_reference": False,
                "best_val": 0.005,
                "final_val": 0.01,
                "final_train": 0.03,
            },
        ]

        table = build_loss_comparison_table(records)
        proposed = table.loc[table["method"] == "Decoupled"].iloc[0]

        self.assertAlmostEqual(
            proposed["best_val_relative_change_pct"], -50.0
        )
        self.assertAlmostEqual(
            proposed["final_val_relative_change_pct"], -50.0
        )
        self.assertAlmostEqual(
            proposed["final_train_relative_change_pct"], -25.0
        )
        self.assertEqual(proposed["reference_method"], "Coupled")

    def test_build_requested_paper_figures_writes_three_figures_to_both_packages(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fastslow = root / "fastslow" / "3steps3seeds"
            coupled = root / "coupled" / "3steps3seeds"
            time_s = np.linspace(0.0, 1.0, 11)
            template = pd.DataFrame(
                {
                    "time_s": time_s,
                    "uav_px": time_s,
                    "uav_py": np.zeros_like(time_s),
                    "uav_pz": np.full_like(time_s, 2.0),
                    "payload_px": time_s - 0.1,
                    "payload_py": np.zeros_like(time_s),
                    "payload_pz": np.full_like(time_s, 1.4),
                    "goal_px": np.full_like(time_s, 4.0),
                    "goal_py": np.zeros_like(time_s),
                    "goal_pz": np.full_like(time_s, 2.0),
                    "payload_err_x": 4.0 - time_s,
                    "payload_err_y": np.zeros_like(time_s),
                    "payload_err_z": np.full_like(time_s, 0.6),
                    "theta_x_deg": np.linspace(5.0, 1.0, len(time_s)),
                    "theta_y_deg": np.linspace(-2.0, 0.0, len(time_s)),
                    "theta_dot_x_deg_s": np.full_like(time_s, -4.0),
                    "theta_dot_y_deg_s": np.full_like(time_s, 2.0),
                    "rope_length_m": np.full_like(time_s, 0.6),
                }
            )
            for package in (fastslow, coupled):
                run = (
                    package
                    / "raw"
                    / "step_19500"
                    / "seed_42"
                    / "teacher"
                )
                run.mkdir(parents=True)
                template.to_csv(run / "rollout.csv", index=False)

            reports = {}
            for key, best in [
                ("old_coupled", 0.010),
                ("old_decoupled", 0.005),
                ("current_coupled", 0.004),
                ("current_structured", 0.0042),
            ]:
                path = root / f"{key}.json"
                path.write_text(
                    json.dumps(
                        {
                            "best_val": best,
                            "train_hist": [best * 1.2, best * 1.1],
                            "val_hist": [best * 1.1, best * 1.05],
                        }
                    ),
                    encoding="utf-8",
                )
                reports[key] = path

            outputs = build_requested_paper_figures(
                fastslow,
                coupled,
                loss_report_paths=reports,
                step=19500,
                seed=42,
            )

            self.assertEqual(len(outputs), 3)
            for package in (fastslow, coupled):
                for name in [
                    "06_step19500_seed42_payload_xyz_swingxy.png",
                    "07_step19500_seed42_swing_energy_3panel.png",
                    "08_PhaseII_loss_raw_comparison.png",
                ]:
                    self.assertGreater(
                        (package / "figures" / name).stat().st_size,
                        0,
                    )
                self.assertTrue(
                    (package / "data" / "phase2_loss_detailed.csv").is_file()
                )

    def test_compute_teacher_metrics_uses_norm_rmse_and_action_variation(self):
        frame = pd.DataFrame(
            {
                "time_s": [0.0, 1.0],
                "payload_err_x": [3.0, 0.0],
                "payload_err_y": [4.0, 0.0],
                "payload_err_z": [0.0, 0.0],
                "theta_x_deg": [3.0, 0.0],
                "theta_y_deg": [4.0, 0.0],
                "a0_raw": [0.0, 1.2],
                "a1_raw": [0.0, 0.0],
                "a2_raw": [0.0, 0.0],
                "a3_raw": [0.0, 0.0],
                "a0_clamp": [0.0, 1.0],
                "a1_clamp": [0.0, 0.0],
                "a2_clamp": [0.0, 0.0],
                "a3_clamp": [0.0, 0.0],
                "payload_mass_kg": [0.5, 0.5],
                "rope_length_m": [0.6, 0.6],
                "wind_acc_x_mps2": [0.0, 0.0],
                "wind_acc_y_mps2": [0.0, 0.0],
                "wind_acc_z_mps2": [0.0, 0.0],
                "actor_inference_ms": [0.1, 0.3],
                "end_to_end_inference_ms": [0.2, 0.4],
            }
        )

        metrics = compute_teacher_metrics(frame, sample_rate_hz=2.0)

        self.assertAlmostEqual(metrics["position_rmse_m"], np.sqrt(12.5))
        self.assertAlmostEqual(metrics["swing_rms_deg"], np.sqrt(12.5))
        self.assertAlmostEqual(metrics["ctbr_tv_total_l1"], 1.0)
        self.assertAlmostEqual(metrics["ctbr_tv_mean_per_transition"], 1.0)
        self.assertAlmostEqual(metrics["raw_action_clip_fraction"], 0.125)
        self.assertAlmostEqual(metrics["actor_mean_ms"], 0.2)

    def test_audit_exact_exogenous_detects_one_changed_wind_value(self):
        columns = {
            "time_s": [0.0, 1.0],
            "payload_mass_kg": [0.5, 0.5],
            "rope_length_m": [0.6, 0.6],
            "gust_x_mps2": [0.1, 0.2],
            "gust_y_mps2": [0.0, 0.0],
            "gust_z_mps2": [0.0, 0.0],
            "wind_acc_x_mps2": [0.1, 0.2],
            "wind_acc_y_mps2": [0.0, 0.0],
            "wind_acc_z_mps2": [0.0, 0.0],
        }
        left = pd.DataFrame(columns)
        right = left.copy()
        right.loc[1, "wind_acc_x_mps2"] = 0.25

        audit = audit_exact_exogenous(left, right)

        self.assertFalse(audit["passed"])
        self.assertFalse(audit["exact_columns"]["wind_acc_x_mps2"])
        self.assertAlmostEqual(
            audit["max_abs_difference"]["wind_acc_x_mps2"], 0.05
        )

    def test_audit_exact_exogenous_accepts_identical_frames(self):
        frame = pd.DataFrame(
            {
                "time_s": [0.0],
                "payload_mass_kg": [0.5],
                "rope_length_m": [0.6],
                "gust_x_mps2": [0.1],
                "gust_y_mps2": [0.0],
                "gust_z_mps2": [0.0],
                "wind_acc_x_mps2": [0.1],
                "wind_acc_y_mps2": [0.0],
                "wind_acc_z_mps2": [0.0],
            }
        )

        audit = audit_exact_exogenous(frame, frame.copy())

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["common_samples"], 1)

    def test_build_analysis_writes_paired_tables_and_passes_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fastslow = root / "fastslow" / "3steps3seeds"
            coupled = root / "coupled" / "3steps3seeds"
            template = pd.DataFrame(
                {
                    "time_s": [0.0, 1.0, 2.0, 3.0],
                    "payload_err_x": [0.2, 0.1, 0.05, 0.0],
                    "payload_err_y": [0.0, 0.0, 0.0, 0.0],
                    "payload_err_z": [0.0, 0.0, 0.0, 0.0],
                    "theta_x_deg": [4.0, 2.0, 1.0, 0.5],
                    "theta_y_deg": [0.0, 0.0, 0.0, 0.0],
                    "a0_raw": [0.0, 0.1, 0.0, 0.1],
                    "a1_raw": [0.0, 0.0, 0.0, 0.0],
                    "a2_raw": [0.0, 0.0, 0.0, 0.0],
                    "a3_raw": [0.0, 0.0, 0.0, 0.0],
                    "a0_clamp": [0.0, 0.1, 0.0, 0.1],
                    "a1_clamp": [0.0, 0.0, 0.0, 0.0],
                    "a2_clamp": [0.0, 0.0, 0.0, 0.0],
                    "a3_clamp": [0.0, 0.0, 0.0, 0.0],
                    "payload_mass_kg": [0.5] * 4,
                    "rope_length_m": [0.6] * 4,
                    "gust_x_mps2": [0.1] * 4,
                    "gust_y_mps2": [0.0] * 4,
                    "gust_z_mps2": [0.0] * 4,
                    "wind_acc_x_mps2": [0.1] * 4,
                    "wind_acc_y_mps2": [0.0] * 4,
                    "wind_acc_z_mps2": [0.0] * 4,
                    "actor_inference_ms": [0.1] * 4,
                    "end_to_end_inference_ms": [0.2] * 4,
                    "zT0": [0.4] * 4,
                    "zT1": [(0.6 - 0.25) / 0.55] * 4,
                }
            )
            for package in (fastslow, coupled):
                run = package / "raw" / "step_1" / "seed_2" / "teacher"
                run.mkdir(parents=True)
                template.to_csv(run / "rollout.csv", index=False)
                (run / "phase2_teacher_play_summary.json").write_text(
                    '{"steps": 4}', encoding="utf-8"
                )

            result = build_analysis(
                fastslow,
                coupled,
                steps=(1,),
                seeds=(2,),
            )

            self.assertTrue(result["audit_passed"])
            self.assertTrue(
                (fastslow / "data" / "rollout_metrics.csv").is_file()
            )
            self.assertTrue(
                (coupled / "data" / "cross_method_paired_metrics.csv").is_file()
            )
            paired = pd.read_csv(
                fastslow / "data" / "cross_method_paired_metrics.csv"
            )
            self.assertEqual(len(paired), 1)
            self.assertAlmostEqual(
                paired.loc[0, "delta_position_rmse_m"], 0.0
            )


if __name__ == "__main__":
    unittest.main()
