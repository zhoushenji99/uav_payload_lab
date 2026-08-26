import importlib.util
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/phase2_control_sources.py"
)
COLLECT_PATH = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/collect_z_dataset.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("phase2_control_sources", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_position_mask_changes_executed_plant_action():
    module = _load_module()
    teacher = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    position = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
    mask = torch.tensor([True, False])
    actual = module.select_control_actions(teacher, position, mask)
    torch.testing.assert_close(actual, torch.tensor([[10.0, 10.0], [2.0, 2.0]]))


def test_collector_does_not_overwrite_previous_action_feature():
    source = COLLECT_PATH.read_text(encoding="utf-8")
    assert "feat[position_history_mask, 17:21]" not in source
    assert "actions_to_step = select_control_actions(" in source
