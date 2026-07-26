import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/analyze_phase2_csv.py"
)


def _load_analysis_module():
    spec = importlib.util.spec_from_file_location("analyze_phase2_csv", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Phase2ContextPhysicalUnitsTests(unittest.TestCase):
    def test_semantic_context_is_converted_to_kg_and_m(self):
        module = _load_analysis_module()
        df = pd.DataFrame(
            {
                "zH0": [0.0, 1.0],
                "zT0": [0.25, 0.75],
                "zH1": [0.0, 1.0],
                "zT1": [0.25, 0.75],
                "priv0": [0.1, 0.9],
                "priv1": [0.2, 0.8],
                # Ground truth must come from the physical CSV columns, not priv0/priv1.
                "payload_mass_kg": [0.42, 0.42],
                "rope_length_m": [0.63, 0.63],
            }
        )

        physical = module._semantic_context_physical(df)

        np.testing.assert_allclose(physical["student_mass_kg"], [0.3, 0.8])
        np.testing.assert_allclose(physical["teacher_mass_kg"], [0.425, 0.675])
        np.testing.assert_allclose(physical["student_rope_length_m"], [0.25, 0.8])
        np.testing.assert_allclose(physical["teacher_rope_length_m"], [0.3875, 0.6625])
        np.testing.assert_allclose(physical["true_mass_kg"], [0.42, 0.42])
        np.testing.assert_allclose(physical["true_rope_length_m"], [0.63, 0.63])

    def test_teacher_structure_label_distinguishes_hard_identity_from_soft_teacher(self):
        module = _load_analysis_module()
        hard = pd.DataFrame(
            {
                "zT0": [0.2, 0.8],
                "zT1": [0.3, 0.7],
                "priv0": [0.2, 0.8],
                "priv1": [0.3, 0.7],
            }
        )
        soft = hard.copy()
        soft["zT1"] += 0.01

        self.assertEqual(module._teacher_structure_label(hard), "hard-explicit Teacher")
        self.assertEqual(module._teacher_structure_label(soft), "learned/soft Teacher")

    def test_fig3_uses_physical_units_for_slow_context_axes(self):
        module = _load_analysis_module()
        student_df = pd.DataFrame(
            {
                "time_s": [0.0, 1.0],
                "zT0": [0.25, 0.25],
                "zT1": [0.75, 0.75],
                "zT2": [1.0, 2.0],
                "zT3": [3.0, 4.0],
                "zT4": [5.0, 6.0],
                "zH0": [0.0, 1.0],
                "zH1": [0.0, 1.0],
                "zH2": [1.1, 2.1],
                "zH3": [3.1, 4.1],
                "zH4": [5.1, 6.1],
                "payload_mass_kg": [0.42, 0.42],
                "rope_length_m": [0.63, 0.63],
                "priv0": [0.24, 0.24],
                "priv1": [0.69, 0.69],
            }
        )
        teacher_df = student_df.copy()
        origin = np.zeros(3)
        teacher = module.Series("teacher", teacher_df, origin, origin, origin)
        student = module.Series("student", student_df, origin, origin, origin)

        captured = {}

        def capture_figure(fig, path):
            captured["fig"] = fig
            captured["path"] = path

        original_save = module._save_fig
        module._save_fig = capture_figure
        try:
            module.plot_fig3_z_compare(teacher, student, Path("/tmp"))
        finally:
            module._save_fig = original_save

        axes = captured["fig"].axes
        self.assertEqual(axes[0].get_ylabel(), "Payload mass (kg)")
        self.assertEqual(axes[1].get_ylabel(), "Rope length (m)")
        np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [0.425, 0.425])
        np.testing.assert_allclose(axes[0].lines[1].get_ydata(), [0.3, 0.8])
        np.testing.assert_allclose(axes[0].lines[-1].get_ydata(), [0.42, 0.42])
        np.testing.assert_allclose(axes[1].lines[-1].get_ydata(), [0.63, 0.63])

    def test_fastslow_runtime_plot_uses_raw_target_and_actor_cache(self):
        module = _load_analysis_module()
        df = pd.DataFrame(
            {
                "time_s": [0.0, 1.0, 2.0],
                "z_slow_raw0": [0.2, 0.8, 0.8],
                "z_slow_target0": [0.2, 0.2, 0.8],
                "z_slow_cache0": [0.2, 0.2, 0.4],
                "z_slow_raw1": [0.4, 0.6, 0.6],
                "z_slow_target1": [0.4, 0.4, 0.6],
                "z_slow_cache1": [0.4, 0.4, 0.5],
                "payload_mass_kg": [0.4, 0.4, 0.4],
                "rope_length_m": [0.5, 0.5, 0.5],
                "slow_updated": [1, 0, 1],
                "actor_inference_ms": [0.2, 0.2, 0.2],
                "end_to_end_inference_ms": [1.0, 0.3, 1.0],
                "slow_inference_ms": [0.7, 0.0, 0.7],
                "fast_inference_ms": [0.2, 0.2, 0.2],
                "executed_action_delta_l1": [0.0, 0.1, 0.2],
                "context_refresh_action_l1": [0.3, 0.0, 0.4],
            }
        )
        origin = np.zeros(3)
        student = module.Series("student", df, origin, origin, origin)
        captured = {}

        def capture_figure(fig, path):
            captured["fig"] = fig
            captured["path"] = path

        original_save = module._save_fig
        module._save_fig = capture_figure
        try:
            module.plot_fig5_fastslow_runtime_audit(student, Path("/tmp"))
        finally:
            module._save_fig = original_save

        self.assertEqual(
            captured["path"].name,
            "fig5_fastslow_runtime_audit.png",
        )
        axes = captured["fig"].axes
        np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [0.4, 0.7, 0.7])
        np.testing.assert_allclose(axes[0].lines[2].get_ydata(), [0.4, 0.4, 0.5])
        np.testing.assert_allclose(
            axes[1].lines[2].get_ydata(),
            [0.47, 0.47, 0.525],
        )


if __name__ == "__main__":
    unittest.main()
