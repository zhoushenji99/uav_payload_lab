import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/replay_real_position_history_v89.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "replay_real_position_history_v89", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fifo_appends_oldest_to_newest_without_action_feedback(tmp_path):
    module = _load_module()
    frames = np.arange(50 * 21, dtype=np.float32).reshape(50, 21)
    path = tmp_path / "real_position_prefix_50H.npy"
    np.save(path, frames)
    loaded = module.load_history_npy(path)
    assert loaded.shape == (50, 21)

    history = np.zeros((50, 21), dtype=np.float32)
    for frame in loaded:
        original = frame.copy()
        module.append_observation_to_fifo(history, frame)
        np.testing.assert_array_equal(frame, original)

    np.testing.assert_array_equal(history, frames)
    np.testing.assert_array_equal(history[-1, 17:21], frames[-1, 17:21])


def test_trace_reader_uses_only_valid_position_prefix_before_candidate_ready(tmp_path):
    module = _load_module()
    path = tmp_path / "trace.csv"
    fieldnames = [
        "observation_valid",
        "candidate_ready",
        *[f"obs_{index}" for index in range(21)],
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "observation_valid": "False",
                "candidate_ready": "False",
                **{f"obs_{index}": "nan" for index in range(21)},
            }
        )
        for value in (1.0, 2.0):
            writer.writerow(
                {
                    "observation_valid": "True",
                    "candidate_ready": "False",
                    **{f"obs_{index}": value for index in range(21)},
                }
            )
        writer.writerow(
            {
                "observation_valid": "True",
                "candidate_ready": "True",
                **{f"obs_{index}": 3.0 for index in range(21)},
            }
        )
    observations, rows = module.load_trace_position_prefix(path)
    assert observations.shape == (2, 21)
    np.testing.assert_array_equal(observations[:, 0], [1.0, 2.0])
    assert len(rows) == 2


def test_real_replay_gates_use_fixed_z1_and_ctbr_contract():
    module = _load_module()
    contract = json.loads(
        (REPO_ROOT / "configs/v89_training_acceptance_contract.json").read_text()
    )
    passing_rows = [
        {
            "z_raw": [0.5, 0.637090909090909, 0.0, 0.0, 0.0],
            "z_clamped": [0.5, 0.637090909090909, 0.0, 0.0, 0.0],
            "action_raw": [-0.6, 0.1, -0.1, 0.05],
            "action_shaped": [-0.6, 0.1, -0.1, 0.05],
            "delta": [0.0, 0.1, -0.1, 0.05],
            "finite": True,
            "absolute_saturation": False,
            "delta_contract_passed": True,
        }
    ]
    verdict = module.evaluate_replay_gates(passing_rows, contract)
    assert verdict["passed"] is True

    failing_rows = [dict(passing_rows[0], delta_contract_passed=False)]
    verdict = module.evaluate_replay_gates(failing_rows, contract)
    assert verdict["passed"] is False
    assert "candidate_must_respect_delta_contract" in verdict["failures"]
