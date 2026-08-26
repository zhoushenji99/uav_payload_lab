import importlib.util
import numpy as np
from pathlib import Path
import sys
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/ctbr_command_contract.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("ctbr_command_contract", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ctbr_contract_clamps_absolute_and_per_step_delta():
    module = _load_module()
    limits = module.CtbrLimits(
        low=(-1.0, -1.2, -1.2, -0.6),
        high=(0.0, 1.2, 1.2, 0.6),
        max_delta=(0.03, 0.25, 0.25, 0.10),
    )
    previous = torch.tensor([[-0.60, 0.0, 0.0, 0.0]])
    target = torch.tensor([[-1.50, 2.50, -2.50, 1.50]])
    shaped = module.shape_ctbr_torch(target, previous, limits)
    torch.testing.assert_close(
        shaped,
        torch.tensor([[-0.63, 0.25, -0.25, 0.10]]),
    )


def test_numpy_and_torch_shapers_are_identical():
    module = _load_module()
    limits = module.CtbrLimits(
        low=(-1.0, -1.2, -1.2, -0.6),
        high=(0.0, 1.2, 1.2, 0.6),
        max_delta=(0.03, 0.25, 0.25, 0.10),
    )
    previous = np.array([[-0.62, 0.12, -0.03, 0.04]], dtype=np.float32)
    target = np.array([[-0.10, -1.50, 0.80, -0.90]], dtype=np.float32)
    expected = module.shape_ctbr_numpy(target, previous, limits)
    actual = module.shape_ctbr_torch(
        torch.from_numpy(target), torch.from_numpy(previous), limits
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
