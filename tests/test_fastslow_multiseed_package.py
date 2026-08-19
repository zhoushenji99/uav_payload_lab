from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real"
    / "build_fastslow_multiseed_package.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "fastslow_multiseed_package_test", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_rollout(
    *,
    mode: str,
    runtime_mode: str,
    wind: bool,
    mass: float = 0.55,
    rope: float = 0.525,
    samples: int = 180,
) -> pd.DataFrame:
    t = np.arange(samples, dtype=float) / 60.0
    wind_x = 0.2 * np.sin(2.0 * np.pi * 1.0 * t) if wind else np.zeros(samples)
    gust_x = np.where((np.arange(samples) // 30) % 2 == 0, 0.1, -0.1) if wind else np.zeros(samples)
    scale = 0.9 if runtime_mode == "fast_slow" else 1.0
    if mode == "teacher":
        scale = 0.8

    frame = pd.DataFrame(
        {
            "time_s": t,
            "mode": mode,
            "payload_err_x": scale * np.exp(-t),
            "payload_err_y": 0.1 * scale * np.sin(t),
            "payload_err_z": 0.05 * scale * np.cos(t),
            "theta_x_deg": 3.0 * scale * np.sin(2.0 * t),
            "theta_y_deg": 2.0 * scale * np.cos(2.0 * t),
            "rope_length_m": rope,
            "payload_mass_kg": mass,
            "zT0": (mass - 0.3) / 0.5,
            "zT1": (rope - 0.25) / 0.55,
            "zT2": wind_x,
            "zT3": 0.5 * wind_x,
            "zT4": -0.25 * wind_x,
            "zH0": (mass - 0.3) / 0.5 + (0.0 if mode == "teacher" else 0.01),
            "zH1": (rope - 0.25) / 0.55 + (0.0 if mode == "teacher" else 0.02),
            "zH2": wind_x + (0.0 if mode == "teacher" else 0.01),
            "zH3": 0.5 * wind_x + (0.0 if mode == "teacher" else 0.01),
            "zH4": -0.25 * wind_x + (0.0 if mode == "teacher" else 0.01),
            "z_rmse": 0.0 if mode == "teacher" else 0.01,
            "z_slow_cache0": (mass - 0.3) / 0.5,
            "z_slow_cache1": (rope - 0.25) / 0.55,
            "slow_updated": 1 if runtime_mode == "all_60hz" else (np.arange(samples) < 180),
            "fast_updated": 1,
            "slow_batch_calls": np.arange(1, samples + 1),
            "fast_batch_calls": np.arange(1, samples + 1),
            "actor_inference_ms": 0.10,
            "end_to_end_inference_ms": 0.35 if runtime_mode == "all_60hz" else 0.25,
            "gust_x_mps2": gust_x,
            "gust_y_mps2": 0.0,
            "gust_z_mps2": 0.0,
            "wind_acc_x_mps2": wind_x,
            "wind_acc_y_mps2": 0.0,
            "wind_acc_z_mps2": 0.0,
            "a0_clamp": -0.5 + 0.01 * np.sin(2.0 * np.pi * 8.0 * t),
            "a1_clamp": 0.1 * np.sin(2.0 * np.pi * 6.0 * t),
            "a2_clamp": 0.1 * np.cos(2.0 * np.pi * 7.0 * t),
            "a3_clamp": 0.05 * np.sin(2.0 * np.pi * 5.0 * t),
        }
    )
    return frame


class FastSlowMultiSeedPackageChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_metrics_cover_control_context_action_and_compute(self):
        frame = synthetic_rollout(
            mode="student", runtime_mode="fast_slow", wind=True
        )
        metrics = self.module.compute_rollout_metrics(frame)
        for key in [
            "position_rmse_norm_m",
            "swing_rms_deg",
            "swing_peak_deg",
            "mass_rmse_kg",
            "rope_length_rmse_m",
            "fast_context_rmse",
            "ctbr_total_variation_l1",
            "ctbr_5_30hz_energy",
            "end_to_end_mean_ms",
            "end_to_end_p95_ms",
            "end_to_end_p99_ms",
            "wind_rms_mps2",
        ]:
            with self.subTest(key=key):
                self.assertTrue(math.isfinite(metrics[key]))
        self.assertGreater(metrics["swing_peak_deg"], metrics["swing_rms_deg"])
        self.assertGreater(metrics["ctbr_total_variation_l1"], 0.0)

    def test_schedule_pair_requires_identical_exogenous_inputs(self):
        all60 = synthetic_rollout(
            mode="student", runtime_mode="all_60hz", wind=True
        )
        fastslow = synthetic_rollout(
            mode="student", runtime_mode="fast_slow", wind=True
        )
        audit = self.module.audit_schedule_pair(all60, fastslow)
        self.assertTrue(audit["strict_pair"])

        fastslow.loc[10, "wind_acc_x_mps2"] += 0.01
        audit = self.module.audit_schedule_pair(all60, fastslow)
        self.assertFalse(audit["strict_pair"])

    def test_fixed_nowind_audit_checks_mass_rope_and_total_wind(self):
        frame = synthetic_rollout(
            mode="student", runtime_mode="fast_slow", wind=False
        )
        audit = self.module.audit_fixed_nowind(
            frame, expected_mass_kg=0.55, expected_rope_length_m=0.525
        )
        self.assertTrue(audit["passed"])

        frame.loc[4, "wind_acc_y_mps2"] = 0.1
        audit = self.module.audit_fixed_nowind(
            frame, expected_mass_kg=0.55, expected_rope_length_m=0.525
        )
        self.assertFalse(audit["passed"])

    def test_builder_writes_tables_audit_hashes_and_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            scenarios = [
                {
                    "id": "seed38_random",
                    "seed": 38,
                    "kind": "random_wind",
                    "fixed_payload_mass_kg": None,
                    "fixed_rope_length_m": None,
                    "disable_wind": False,
                },
                {
                    "id": "seed46_nowind_mid",
                    "seed": 46,
                    "kind": "fixed_nowind",
                    "fixed_payload_mass_kg": 0.55,
                    "fixed_rope_length_m": 0.525,
                    "disable_wind": True,
                },
            ]
            manifest = {
                "source_run": "2026-07-24_06-06-18_hardexplicit_teacher_fastslow_seed42",
                "scenarios": scenarios,
                "modes": ["teacher", "all60", "fastslow"],
            }
            (package / "experiment_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            for scenario in scenarios:
                for mode in manifest["modes"]:
                    run_dir = package / "raw" / scenario["id"] / mode
                    run_dir.mkdir(parents=True)
                    frame = synthetic_rollout(
                        mode="teacher" if mode == "teacher" else "student",
                        runtime_mode=(
                            "all_60hz" if mode == "all60" else "fast_slow"
                        ),
                        wind=not scenario["disable_wind"],
                    )
                    frame.to_csv(run_dir / "rollout.csv", index=False)
                    (run_dir / "summary.json").write_text(
                        json.dumps(
                            {
                                "steps": len(frame),
                                "seed": scenario["seed"],
                                "context_runtime_mode": (
                                    "all_60hz"
                                    if mode == "all60"
                                    else "fast_slow"
                                ),
                                "evaluation_overrides": {
                                    "payload_mass_kg": scenario[
                                        "fixed_payload_mass_kg"
                                    ],
                                    "rope_length_m": scenario[
                                        "fixed_rope_length_m"
                                    ],
                                    "disable_wind": scenario["disable_wind"],
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    (run_dir / "console.log").write_text(
                        "synthetic completed\n", encoding="utf-8"
                    )

            result = self.module.build_package(package)
            self.assertTrue(result["audit"]["passed"])

            for name in [
                "rollout_metrics.csv",
                "paired_schedule_metrics.csv",
                "aggregate_metrics.csv",
                "data_audit.json",
                "sha256_manifest.csv",
            ]:
                with self.subTest(data=name):
                    self.assertTrue((package / "data" / name).exists())

            expected_figures = [
                "01_场景物理参数与风扰.png",
                "02_位置误差多场景.png",
                "03_摆角多场景.png",
                "04_Student上下文恢复.png",
                "05_CTBR动作连续性.png",
                "06_计算开销与调用次数.png",
                "07_阵风与快速上下文响应.png",
                "08_整体性能汇总.png",
                "09_Teacher到Student部署差距.png",
                "10_无风中等参数案例.png",
            ]
            for name in expected_figures:
                with self.subTest(figure=name):
                    path = package / "figures" / name
                    self.assertTrue(path.exists())
                    self.assertGreater(path.stat().st_size, 5_000)


if __name__ == "__main__":
    unittest.main()
