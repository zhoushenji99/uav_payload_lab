import importlib.util
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DAGGER_PATH = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/collect_student_dagger_dataset.py"
)
RUNTIME_PATH = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/fastslow_runtime.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dagger_student_drives_plant_teacher_only_labels():
    module = _load(DAGGER_PATH, "collect_student_dagger_dataset")
    result = module.choose_dagger_step(
        position_action=torch.tensor([[10.0]]),
        student_action=torch.tensor([[20.0]]),
        teacher_z=torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        episode_step=torch.tensor([180]),
        precontrol_steps=180,
    )
    torch.testing.assert_close(result.executed_action, torch.tensor([[20.0]]))
    torch.testing.assert_close(
        result.label_z,
        torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]),
    )
    assert result.position_active.tolist() == [False]


class _FakeEncoder:
    def encode_slow(self, history):
        return history[:, -1, :2]

    def encode_fast(self, history):
        return history[:, -1, 2:5]


def _runtime_state():
    return {
        "z_slow_raw": torch.zeros(2, 2),
        "z_slow_target": torch.zeros(2, 2),
        "z_slow_cache": torch.zeros(2, 2),
        "z_fast_cache": torch.zeros(2, 3),
    }


def test_dagger_and_play_shared_runtime_are_numerically_identical():
    runtime = _load(RUNTIME_PATH, "fastslow_runtime_dagger_test")
    history = torch.arange(2 * 50 * 21, dtype=torch.float32).reshape(2, 50, 21) / 1000
    episode_steps = torch.tensor([179, 180])
    schedule = runtime.compute_multirate_schedule(50, 1 / 60, 3.0, 1.0, 60.0)
    play_state = _runtime_state()
    dagger_state = _runtime_state()

    play = runtime.update_fastslow_context(
        encoder=_FakeEncoder(),
        obs_history=history,
        episode_steps=episode_steps,
        schedule=schedule,
        slow_filter_alpha=runtime.causal_ema_alpha(1 / 60, 0.25),
        context_runtime_mode="fast_slow",
        **play_state,
    )
    dagger = runtime.update_fastslow_context(
        encoder=_FakeEncoder(),
        obs_history=history,
        episode_steps=episode_steps,
        schedule=schedule,
        slow_filter_alpha=runtime.causal_ema_alpha(1 / 60, 0.25),
        context_runtime_mode="fast_slow",
        **dagger_state,
    )
    torch.testing.assert_close(play.z_hat, dagger.z_hat, atol=1e-6, rtol=0.0)
    proprio = history[:, -1, :21]
    actor_weight = torch.arange(26 * 4, dtype=torch.float32).reshape(26, 4) / 100
    play_actor_raw = torch.cat((proprio, play.z_hat), dim=1) @ actor_weight
    dagger_actor_raw = torch.cat((proprio, dagger.z_hat), dim=1) @ actor_weight
    torch.testing.assert_close(
        play_actor_raw, dagger_actor_raw, atol=1e-6, rtol=0.0
    )


def test_dagger_script_uses_shared_runtime_and_shared_ctbr_shaper():
    source = DAGGER_PATH.read_text(encoding="utf-8")
    assert "update_fastslow_context(" in source
    assert "shape_ctbr_torch(" in source
    assert '"student_action_raw"' in source
    assert '"student_action_shaped"' in source
    assert "student_dagger_round_" in source
