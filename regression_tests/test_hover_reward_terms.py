import importlib.util
import math
from pathlib import Path
import sys

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/hover_reward_terms.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("hover_reward_terms", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_uav_tilt_is_zero_for_level_and_half_pi_for_ninety_degree_roll():
    module = _load_module()
    level = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    roll_90 = torch.tensor([[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]])
    torch.testing.assert_close(
        module.uav_tilt_rad_wxyz(level),
        torch.tensor([0.0]),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        module.uav_tilt_rad_wxyz(roll_90),
        torch.tensor([math.pi / 2]),
        atol=1e-6,
        rtol=0.0,
    )


def test_normalized_ctbr_terms_use_the_shared_action_scale():
    module = _load_module()
    sent = torch.tensor([[-0.6, 0.6, -1.2, 0.3]])
    delta = torch.tensor([[0.03, 0.12, -0.24, 0.06]])
    jerk = torch.tensor([[0.01, -0.12, 0.24, -0.06]])
    scale = torch.tensor([1.0, 1.2, 1.2, 0.6])
    sent_norm, delta_norm, jerk_norm = module.normalized_ctbr_terms(
        sent, delta, jerk, scale
    )
    torch.testing.assert_close(sent_norm, torch.tensor([[-0.6, 0.5, -1.0, 0.5]]))
    torch.testing.assert_close(delta_norm, torch.tensor([[0.03, 0.1, -0.2, 0.1]]))
    torch.testing.assert_close(jerk_norm, torch.tensor([[0.01, -0.1, 0.2, -0.1]]))
