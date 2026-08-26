import json
from pathlib import Path


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
