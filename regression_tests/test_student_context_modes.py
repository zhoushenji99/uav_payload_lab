import importlib.util
from pathlib import Path
import sys
import unittest

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/train_student_z.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("train_student_z", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StudentContextModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_split_student_has_disjoint_slow_and_fast_parameters(self):
        model = self.module.FastSlowStudentEncoder(
            input_dim=21,
            history_len=50,
            z_slow_dim=2,
            z_fast_dim=3,
        )
        slow_ids = {id(parameter) for parameter in model.slow_encoder.parameters()}
        fast_ids = {id(parameter) for parameter in model.fast_encoder.parameters()}
        x = torch.randn(2, 50, 21)

        z_slow, z_fast = model(x)

        self.assertTrue(slow_ids.isdisjoint(fast_ids))
        self.assertEqual(tuple(z_slow.shape), (2, 2))
        self.assertEqual(tuple(z_fast.shape), (2, 3))

    def test_monolithic_student_is_one_joint_five_dimensional_encoder(self):
        model = self.module.MonolithicStudentEncoder(
            input_dim=21,
            history_len=50,
            z_dim=5,
        )
        x = torch.randn(2, 50, 21)

        z = model(x)

        self.assertEqual(tuple(z.shape), (2, 5))
        self.assertFalse(hasattr(model, "slow_encoder"))
        self.assertFalse(hasattr(model, "fast_encoder"))


if __name__ == "__main__":
    unittest.main()
