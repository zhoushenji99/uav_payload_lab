import json
import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).parents[1]
EVALUATOR_PATH = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/evaluate_v89_policy.py"
)
SELECTOR_PATH = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/select_v89_checkpoint.py"
)


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("evaluate_v89_policy", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_selector():
    spec = importlib.util.spec_from_file_location("select_v89_checkpoint", SELECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _passing_summary(seed=42):
    return {
        "seed": seed,
        "scenario": "no_ambient_wind",
        "finite": True,
        "early_termination": False,
        "payload_position_rmse_m": 0.03,
        "uav_height_mean_abs_error_m": 0.03,
        "swing_rms_deg": 2.0,
        "uav_tilt_p95_deg": 4.0,
        "uav_tilt_absolute_max_deg": 8.0,
        "actual_body_rate_p95_rad_s": 0.4,
        "actual_body_rate_absolute_max_rad_s": 0.8,
        "command_rate_p95_abs_rad_s": [0.3, 0.3, 0.2],
        "command_saturation_fraction": 0.0,
        "ctbr_tv_mean": 0.04,
        "command_high_frequency_5_30hz_fraction": 0.1,
        "context_z0_rmse": 0.03,
        "context_z1_rmse": 0.03,
        "context_fast_rmse": 0.05,
        "context_severe_out_of_range_fraction": 0.0,
    }


def test_v89_contract_is_exact_and_internally_consistent():
    cfg = json.loads(
        (Path(__file__).parents[1] / "configs/v89_training_acceptance_contract.json")
        .read_text(encoding="utf-8")
    )
    assert cfg["base"]["commit"] == "889b6c70ec8bde576947c463644e974b7cce5591"
    assert cfg["target"]["resume_old_teacher"] is False
    assert cfg["target"]["resume_old_student"] is False
    assert cfg["immutable_observation_contract"]["shape"] == [50, 21]
    assert len(cfg["immutable_observation_contract"]["observation_order"]) == 21
    assert cfg["ctbr_execution_contract"]["absolute_low"] == [-1.0, -1.2, -1.2, -0.6]
    assert cfg["ctbr_execution_contract"]["absolute_high"] == [0.0, 1.2, 1.2, 0.6]
    assert cfg["ctbr_execution_contract"]["max_delta_per_60hz_step"] == [0.03, 0.25, 0.25, 0.1]
    assert abs(cfg["slow_context_contract"]["real_rope_length_normalized"] - 0.637090909090909) < 1e-12


def test_one_failed_seed_rejects_checkpoint_even_if_mean_passes():
    evaluator = _load_evaluator()
    contract = json.loads(
        (REPO_ROOT / "configs/v89_training_acceptance_contract.json").read_text()
    )
    failing = _passing_summary(seed=101)
    failing["uav_tilt_absolute_max_deg"] = 21.0
    verdict = evaluator.evaluate_hard_gates(
        [_passing_summary(seed=42), failing], contract
    )
    assert verdict["passed"] is False
    assert verdict["failed_seeds"] == [101]


def test_checkpoint_score_uses_fixed_v89_formula():
    evaluator = _load_evaluator()
    summary = _passing_summary()
    expected = 0.03 + 0.01 * 2.0 + 0.05 * 4.0 + 0.10 * 0.04
    assert abs(evaluator.checkpoint_score([summary]) - expected) < 1e-12


def test_selector_uses_only_fully_passing_candidate_with_lowest_score():
    selector = _load_selector()
    evaluation = {
        "checkpoints": [
            {"checkpoint": "/tmp/failed.pt", "passed": False, "score": 0.01},
            {"checkpoint": "/tmp/pass_b.pt", "passed": True, "score": 0.30},
            {"checkpoint": "/tmp/pass_a.pt", "passed": True, "score": 0.20},
        ]
    }
    selected = selector.select_passing_checkpoint(evaluation)
    assert selected["checkpoint"] == "/tmp/pass_a.pt"


def test_no_passing_candidate_exits_two_without_selected_file(tmp_path, monkeypatch):
    selector = _load_selector()
    evaluation_root = tmp_path / "evaluation"
    output = tmp_path / "selection"
    evaluation_root.mkdir()
    output.mkdir()
    (evaluation_root / "summary.json").write_text(
        json.dumps(
            {
                "policy_kind": "student",
                "checkpoints": [
                    {
                        "checkpoint": "/tmp/rejected.pth",
                        "passed": False,
                        "score": None,
                    }
                ],
            }
        )
    )
    stale = output / "selected_checkpoint.txt"
    stale.write_text("/tmp/stale.pth\n")
    contract = REPO_ROOT / "configs/v89_training_acceptance_contract.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_v89_checkpoint.py",
            "--policy-kind",
            "student",
            "--evaluation-root",
            str(evaluation_root),
            "--contract",
            str(contract),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as error:
        selector.main()
    assert error.value.code == 2
    assert not stale.exists()
    report = json.loads((output / "selection_report.json").read_text())
    assert report["selected"] is None
