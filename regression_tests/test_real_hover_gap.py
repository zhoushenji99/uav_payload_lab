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


if __name__ == "__main__":
    unittest.main()
