import importlib.util
from pathlib import Path
import sys
import unittest

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/phase2_shadow_handover.py"
)
PLAY_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/play_student_phase2.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("phase2_shadow_handover", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Phase2ShadowHandoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_three_second_shadow_maps_to_180_policy_steps(self):
        steps = self.module.validate_shadow_warmup(
            shadow_warmup_sec=3.0,
            policy_dt=1.0 / 60.0,
            history_len=50,
            slow_warmup_sec=3.0,
            mode="student",
        )

        self.assertEqual(steps, 180)

    def test_zero_seconds_preserves_legacy_immediate_student_control(self):
        steps = self.module.validate_shadow_warmup(
            shadow_warmup_sec=0.0,
            policy_dt=1.0 / 60.0,
            history_len=50,
            slow_warmup_sec=3.0,
            mode="student",
        )

        self.assertEqual(steps, 0)

    def test_positive_shadow_must_fill_history_and_finish_slow_startup(self):
        with self.assertRaisesRegex(ValueError, "history"):
            self.module.validate_shadow_warmup(
                shadow_warmup_sec=0.8,
                policy_dt=1.0 / 60.0,
                history_len=50,
                slow_warmup_sec=0.5,
                mode="student",
            )

        with self.assertRaisesRegex(ValueError, "slow startup"):
            self.module.validate_shadow_warmup(
                shadow_warmup_sec=2.9,
                policy_dt=1.0 / 60.0,
                history_len=50,
                slow_warmup_sec=3.0,
                mode="student",
            )

    def test_positive_shadow_is_student_only(self):
        with self.assertRaisesRegex(ValueError, "student mode"):
            self.module.validate_shadow_warmup(
                shadow_warmup_sec=3.0,
                policy_dt=1.0 / 60.0,
                history_len=50,
                slow_warmup_sec=3.0,
                mode="teacher",
            )

    def test_handover_boundary_is_teacher_before_180_and_student_at_180(self):
        episode_steps = torch.tensor([0, 179, 180, 181])

        mask = self.module.teacher_shadow_mask(episode_steps, shadow_steps=180)

        self.assertEqual(mask.tolist(), [True, True, False, False])

    def test_position_precontrol_executes_position_until_exact_boundary(self):
        steps = torch.tensor([179, 180])
        executed, mask = self.module.select_precontrol_actions(
            student=torch.tensor([[1.0], [2.0]]),
            position=torch.tensor([[10.0], [20.0]]),
            episode_steps=steps,
            precontrol_steps=180,
        )

        torch.testing.assert_close(executed, torch.tensor([[10.0], [2.0]]))
        self.assertEqual(mask.tolist(), [True, False])

    def test_action_selection_is_per_environment_and_keeps_student_candidate(self):
        student_raw = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        student_clipped = torch.tensor([[0.8, 0.9], [0.7, 0.6]])
        teacher_raw = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        teacher_clipped = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        episode_steps = torch.tensor([179, 180])

        executed_raw, executed_clipped, mask = self.module.select_shadow_actions(
            student_raw=student_raw,
            student_clipped=student_clipped,
            teacher_raw=teacher_raw,
            teacher_clipped=teacher_clipped,
            episode_steps=episode_steps,
            shadow_steps=180,
        )

        torch.testing.assert_close(
            executed_raw,
            torch.tensor([[10.0, 20.0], [3.0, 4.0]]),
        )
        torch.testing.assert_close(
            executed_clipped,
            torch.tensor([[0.1, 0.2], [0.7, 0.6]]),
        )
        self.assertEqual(mask.tolist(), [True, False])

    def test_play_script_exposes_and_audits_shadow_handover(self):
        source = PLAY_PATH.read_text(encoding="utf-8")

        self.assertIn("--student_shadow_warmup_sec", source)
        self.assertIn('"--precontrol"', source)
        self.assertIn('"--precontrol_sec"', source)
        self.assertIn("select_precontrol_actions(", source)
        self.assertIn("compute_position_hold_ctbr", source)
        self.assertIn('"control_source"', source)
        self.assertIn('"student_candidate_a0_raw"', source)
        self.assertIn('"precontrol_steps"', source)


if __name__ == "__main__":
    unittest.main()
