import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_SOURCE = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env.py"
)


def load_reset_wind_method():
    tree = ast.parse(ENV_SOURCE.read_text(encoding="utf-8"), filename=str(ENV_SOURCE))
    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef):
            continue
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_reset_wind":
                namespace = {"math": math, "torch": torch}
                module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
                exec(compile(module, str(ENV_SOURCE), "exec"), namespace)
                return namespace["_reset_wind"]
    raise AssertionError(f"_reset_wind was not found in {ENV_SOURCE}")


def load_wind_step_method():
    tree = ast.parse(ENV_SOURCE.read_text(encoding="utf-8"), filename=str(ENV_SOURCE))
    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef):
            continue
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_wind_step":
                namespace = {"math": math, "torch": torch}
                module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
                exec(compile(module, str(ENV_SOURCE), "exec"), namespace)
                return namespace["_wind_step"]
    raise AssertionError(f"_wind_step was not found in {ENV_SOURCE}")


class TestSim2RealWindReset(unittest.TestCase):
    def test_reset_wind_clears_only_selected_environment_rows(self):
        reset_wind = load_reset_wind_method()
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda:0"))

        for device in devices:
            with self.subTest(device=str(device)):
                torch.manual_seed(42)
                num_envs = 5
                env_ids = torch.tensor([1, 3], dtype=torch.long, device=device)
                state = SimpleNamespace(
                    _wind_enabled=True,
                    device=device,
                    _wind_mean_accel_max=0.2,
                    _wind_gust_dt_min=0.0,
                    _wind_gust_dt_max=2.0,
                    _wind_mean=torch.zeros(num_envs, 3, device=device),
                    _wind_gust=torch.full((num_envs, 3), 1.0, device=device),
                    _wind_ou=torch.full((num_envs, 3), 2.0, device=device),
                    _wind_acc_w=torch.full((num_envs, 3), 3.0, device=device),
                    _wind_t=torch.full((num_envs,), 4.0, device=device),
                    _wind_t_next=torch.full((num_envs,), 5.0, device=device),
                    _eval_wind_elapsed_s=torch.full((num_envs,), 6.0, device=device),
                )
                untouched_ids = torch.tensor([0, 2, 4], dtype=torch.long, device=device)
                untouched_before = {
                    name: getattr(state, name)[untouched_ids].clone()
                    for name in (
                        "_wind_mean",
                        "_wind_gust",
                        "_wind_ou",
                        "_wind_acc_w",
                        "_wind_t",
                        "_wind_t_next",
                        "_eval_wind_elapsed_s",
                    )
                }

                reset_wind(state, env_ids)

                for name in (
                    "_wind_gust",
                    "_wind_ou",
                    "_wind_acc_w",
                    "_wind_t",
                    "_eval_wind_elapsed_s",
                ):
                    selected = getattr(state, name)[env_ids]
                    self.assertTrue(
                        torch.equal(selected, torch.zeros_like(selected)),
                        msg=f"{name} selected rows were not cleared on {device}: {selected}",
                    )
                for name, expected in untouched_before.items():
                    self.assertTrue(
                        torch.equal(getattr(state, name)[untouched_ids], expected),
                        msg=f"{name} modified unselected rows on {device}",
                    )
                self.assertTrue(torch.all(state._wind_mean[env_ids, 2] == 0.0))
                self.assertTrue(torch.all(state._wind_t_next[env_ids] >= state._wind_gust_dt_min))
                self.assertTrue(torch.all(state._wind_t_next[env_ids] <= state._wind_gust_dt_max))

    def test_wind_step_applies_eval_scale_after_training_range_clamp(self):
        wind_step = load_wind_step_method()
        state = SimpleNamespace(
            _wind_enabled=True,
            device=torch.device("cpu"),
            num_envs=1,
            _wind_t=torch.zeros(1),
            _wind_t_next=torch.full((1,), 100.0),
            _wind_gust_dt_min=1.0,
            _wind_gust_dt_max=1.0,
            _wind_gust_accel_max=0.0,
            _wind_gust=torch.zeros(1, 3),
            _wind_axis="xy",
            _wind_ou=torch.zeros(1, 3),
            _wind_ou_theta=1.0,
            _wind_ou_sigma=0.0,
            _wind_mean=torch.tensor([[2.0, 0.0, 0.0]]),
            _wind_total_accel_max=1.5,
            _eval_wind_scale=2.0,
            _eval_wind_mode="training",
            _eval_wind_elapsed_s=torch.zeros(1),
            _wind_acc_w=torch.zeros(1, 3),
        )

        wind_step(state, 1.0 / 60.0)

        self.assertTrue(
            torch.allclose(state._wind_acc_w, torch.tensor([[3.0, 0.0, 0.0]])),
            msg=f"expected baseline clamp 1.5 followed by eval scale 2.0, got {state._wind_acc_w}",
        )

    def test_wind_step_generates_requested_evaluation_sinusoid_after_start(self):
        wind_step = load_wind_step_method()
        state = SimpleNamespace(
            _wind_enabled=True,
            device=torch.device("cpu"),
            num_envs=1,
            _eval_wind_mode="sinusoid",
            _eval_wind_elapsed_s=torch.tensor([2.75]),
            _eval_wind_amplitude_mps2=1.2,
            _eval_wind_frequency_hz=1.0,
            _eval_wind_start_sec=3.0,
            _eval_wind_axis="x",
            _eval_wind_phase_rad=0.0,
            _eval_wind_scale=1.0,
            _wind_acc_w=torch.zeros(1, 3),
        )

        wind_step(state, 0.25)
        self.assertTrue(torch.allclose(state._wind_acc_w, torch.zeros(1, 3), atol=1e-6))

        wind_step(state, 0.25)
        self.assertTrue(
            torch.allclose(state._wind_acc_w, torch.tensor([[1.2, 0.0, 0.0]]), atol=1e-6),
            msg=f"expected sine peak on x axis, got {state._wind_acc_w}",
        )


if __name__ == "__main__":
    unittest.main()
