"""Export and verify the fast/slow hover policy for NVIDIA Jetson deployment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator

from jetson_deployment import load_deployment_models
from jetson_onnx_runtime import (
    DEFAULT_CONTEXT_HIGH,
    DEFAULT_CONTEXT_LOW,
    DEFAULT_CONTEXT_SEVERE_HIGH,
    DEFAULT_CONTEXT_SEVERE_LOW,
)
from jetson_reference_runtime import FastSlowRuntime


REQUIRED_ARTIFACTS = [
    "models/actor.onnx",
    "models/actor.ts",
    "models/slow_encoder.onnx",
    "models/slow_encoder.ts",
    "models/fast_encoder.onnx",
    "models/fast_encoder.ts",
    "models/source_teacher_model_1500.pt",
    "models/source_student_best.pth",
    "runtime/jetson_reference_runtime.py",
    "runtime/jetson_onnx_runtime.py",
    "runtime/jetson_inference_trace.py",
    "runtime/rl_fastslow_inference_onnx_v1.py",
    "runtime/rl_ctbr_handover_core_v1.py",
    "runtime/rl_ctbr_offboard_handover_v1.py",
    "verification/verify_jetson_bundle.py",
    "verification/parity_report.json",
    "verification/parity_vectors.npz",
    "config/manifest.json",
    "config/training_config_snapshot/agent.yaml",
    "config/training_config_snapshot/env.yaml",
    "config/training_config_snapshot/context_architecture.json",
    "config/training_config_snapshot/dataset_audit.json",
    "config/training_config_snapshot/collect_report.json",
    "config/training_config_snapshot/phase2_student_play_summary.json",
    "config/training_config_snapshot/phase2_student_shadow3s_seed42.csv",
    "config/training_config_snapshot/meta_uav_env.py",
    "config/training_config_snapshot/meta_uav_env_cfg.py",
    "config/仿真预导出验收报告.md",
    "sha256sums.txt",
    "README.md",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_parity_history(
    dataset_dir: str | Path, sample_count: int = 8
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load deterministic float32 history samples from the collected dataset."""

    root = Path(dataset_dir).resolve()
    shards = sorted(root.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.pt files found in parity dataset: {root}")
    shard_path = shards[0]
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    if not isinstance(shard, dict) or "inputs" not in shard:
        raise ValueError(f"Parity shard has no inputs tensor: {shard_path}")
    inputs = shard["inputs"]
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 3 or tuple(inputs.shape[1:]) != (50, 21):
        shape = tuple(inputs.shape) if isinstance(inputs, torch.Tensor) else type(inputs)
        raise ValueError(f"Parity inputs must have shape [N, 50, 21], got {shape}")
    count = min(max(int(sample_count), 1), int(inputs.shape[0]))
    indices = torch.linspace(0, inputs.shape[0] - 1, steps=count).round().long()
    history = inputs[indices].to(dtype=torch.float32).contiguous()
    audit = {
        "dataset_dir": str(root),
        "shard_name": shard_path.name,
        "shard_sha256": sha256_file(shard_path),
        "sample_count": count,
        "source_dtype": str(inputs.dtype),
        "parity_dtype": str(history.dtype),
    }
    return history, audit


def build_manifest(
    metadata: dict[str, Any], source_hashes: dict[str, str]
) -> dict[str, Any]:
    policy_hz = 60.0
    slow_warmup_sec = 3.0
    slow_update_hz = 1.0
    fast_update_hz = 60.0
    tau_sec = 0.25
    alpha = 1.0 - np.exp(-(1.0 / policy_hz) / tau_sec)
    return {
        "schema_version": 1,
        "bundle_type": "uav_payload_fastslow_rma_jetson",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lineage": {
            "teacher_iteration": metadata["teacher_iteration"],
            "student_epoch_zero_based": metadata["student_epoch"],
            "student_epoch_human": metadata["student_epoch"] + 1,
            "student_best_val": metadata["student_best_val"],
            "seed": metadata["seed"],
            "teacher_context_mode": metadata["teacher_context_mode"],
            "student_context_mode": metadata["student_context_mode"],
        },
        "source_hashes": dict(source_hashes),
        "teacher_source": "models/source_teacher_model_1500.pt",
        "student_source": "models/source_student_best.pth",
        "models": {
            "slow_encoder": {
                "onnx": "models/slow_encoder.onnx",
                "torchscript": "models/slow_encoder.ts",
                "input_name": "history",
                "input_shape": ["B", 50, 21],
                "output_name": "z_slow",
                "output_shape": ["B", 2],
            },
            "fast_encoder": {
                "onnx": "models/fast_encoder.onnx",
                "torchscript": "models/fast_encoder.ts",
                "input_name": "history",
                "input_shape": ["B", 50, 21],
                "output_name": "z_fast",
                "output_shape": ["B", 3],
            },
            "actor": {
                "onnx": "models/actor.onnx",
                "torchscript": "models/actor.ts",
                "input_name": "actor_input",
                "input_shape": ["B", 26],
                "output_name": "ctbr_raw",
                "output_shape": ["B", 4],
                "observation_normalization_embedded": True,
                "normalizer_eps": metadata["normalizer_eps"],
            },
        },
        "observation_21": {
            "topic": "/uav_payload/observation21",
            "task_world_frame": "NWU",
            "uav_body_frame": "FLU",
            "quaternion_order": "wxyz",
            "quaternion_semantics": "UAV body orientation in task world",
            "payload_error_definition": "goal_position_NWU - payload_position_NWU",
            "order": [
                "payload_error_world_x_m",
                "payload_error_world_y_m",
                "payload_error_world_z_m",
                "swing_theta_x_deg",
                "swing_theta_y_deg",
                "swing_rate_x_deg_s",
                "swing_rate_y_deg_s",
                "uav_world_quaternion_w",
                "uav_world_quaternion_x",
                "uav_world_quaternion_y",
                "uav_world_quaternion_z",
                "uav_body_velocity_x_mps",
                "uav_body_velocity_y_mps",
                "uav_body_velocity_z_mps",
                "uav_body_rate_x_rad_s",
                "uav_body_rate_y_rad_s",
                "uav_body_rate_z_rad_s",
                "previous_executed_thrust_body_z",
                "previous_executed_roll_rate_rad_s",
                "previous_executed_pitch_rate_rad_s",
                "previous_executed_yaw_rate_rad_s",
            ],
            "history_shape": [1, 50, 21],
            "history_update": "left_shift_then_append_current_observation",
            "position_previous_action_source": "/fmu/out/vehicle_rates_setpoint converted to the fixed CTBR contract",
            "rl_previous_action_source": "/uav_payload/ctbr_action_executed",
            "previous_action_requirement": "previous clamped command actually sent for the active controller",
            "history_reset_conditions": [
                "manual runtime reset",
                "observation dimension or finite-value failure",
                "observation age above 0.1 seconds",
            ],
            "input_sequence_logged": True,
        },
        "runtime": {
            "policy_hz": policy_hz,
            "history_len": 50,
            "slow_warmup_sec": slow_warmup_sec,
            "slow_warmup_steps": 180,
            "slow_update_hz_after_warmup": slow_update_hz,
            "slow_period_steps": 60,
            "fast_update_hz": fast_update_hz,
            "fast_period_steps": 1,
            "slow_filter_tau_sec": tau_sec,
            "slow_filter_alpha": float(alpha),
            "max_observation_age_sec": 0.1,
            "slow_context_rule": "60Hz for steps 0-179; refresh at step 180 then every 60 steps",
            "fast_context_rule": "refresh every policy step",
        },
        "action": {
            "order": [
                "thrust_body_z",
                "roll_rate_rad_s",
                "pitch_rate_rad_s",
                "yaw_rate_rad_s",
            ],
            "low": [-1.0, -2.5, -2.5, -1.5],
            "high": [0.0, 2.5, 2.5, 1.5],
            "interface": "PX4_CTBR",
            "frame": "PX4_FRD",
            "candidate_topic": "/uav_payload/rl_ctbr_candidate",
            "clamp_required_after_actor": True,
        },
        "context_semantics": {
            "z0": "normalized total swinging payload mass over [0.31265, 0.93265] kg",
            "z1": "normalized rope length over [0.25, 0.8] m",
            "z2_z4": "learned fast residual context without physical units",
        },
        "context_bounds": {
            "source": "collected dataset z_stats; z0/z1 use physical normalized [0,1]",
            "low": list(DEFAULT_CONTEXT_LOW),
            "high": list(DEFAULT_CONTEXT_HIGH),
            "severe_low": list(DEFAULT_CONTEXT_SEVERE_LOW),
            "severe_high": list(DEFAULT_CONTEXT_SEVERE_HIGH),
            "moderate_rule": "clamp context before Actor and log context_out_of_range",
            "severe_rule": "do not publish candidate and log context_severe_out_of_range",
        },
        "training_ranges": {
            "payload_mass_kg": [0.31265, 0.93265],
            "rope_length_m": [0.25, 0.8],
        },
        "safety": {
            "flight_approved": False,
            "deployment_status": "shadow_only",
            "startup_guard_required_before_flight": True,
            "unverified_items": [
                "Position real CTBR prefix distribution matches Student training",
                "Teacher and Student paired closed-loop results across multiple evaluation seeds",
                "no-wind closed-loop result with eval_disable_wind actually enabled",
            ],
            "reason": (
                "The runtime requires 50 valid real observations and a 3 second Position-mode "
                "shadow window before candidate_ready. Flight authority remains in the external "
                "handover gateway and pilot failsafe."
            ),
        },
        "artifacts": list(REQUIRED_ARTIFACTS),
    }


def validate_existing_bundle_lineage(
    output_dir: str | Path, source_hashes: dict[str, str]
) -> None:
    output = Path(output_dir)
    if not output.exists() or not any(output.iterdir()):
        return
    manifest_path = output / "config" / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Refusing to overwrite non-empty directory without manifest: {output}"
        )
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing.get("source_hashes") != source_hashes:
        raise RuntimeError(
            "Refusing to overwrite a deployment bundle from a different checkpoint lineage"
        )


def _export_torchscript(module: torch.nn.Module, example: torch.Tensor, path: Path) -> None:
    traced = torch.jit.trace(module, example, check_trace=True)
    traced = torch.jit.freeze(traced.eval())
    torch.jit.save(traced, str(path))


def _export_onnx(
    module: torch.nn.Module,
    example: torch.Tensor,
    path: Path,
    *,
    input_name: str,
    output_name: str,
) -> None:
    torch.onnx.export(
        module,
        example,
        str(path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        dynamo=False,
    )
    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def _onnx_output(path: Path, input_name: str, value: torch.Tensor) -> torch.Tensor:
    evaluator = ReferenceEvaluator(str(path))
    outputs = evaluator.run(None, {input_name: value.detach().cpu().numpy()})
    return torch.from_numpy(np.asarray(outputs[0]))


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left.detach().cpu() - right.detach().cpu())).item())


def _build_parity_report(
    models: Any,
    output: Path,
    parity_history: torch.Tensor,
    parity_input_audit: dict[str, Any],
    *,
    torchscript_tolerance: float = 1.0e-6,
    onnx_tolerance: float = 1.0e-5,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    models_dir = output / "models"
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260820)
    history = parity_history
    with torch.inference_mode():
        parity_slow = models.slow_encoder(history)
        parity_fast = models.fast_encoder(history)
    actor_input = torch.cat(
        [history[:, -1, :], parity_slow, parity_fast], dim=-1
    )
    native_inputs = {
        "slow_encoder": (models.slow_encoder, history, "history"),
        "fast_encoder": (models.fast_encoder, history, "history"),
        "actor": (models.actor, actor_input, "actor_input"),
    }
    model_reports: dict[str, Any] = {}
    parity_vectors: dict[str, np.ndarray] = {
        "history": history.detach().cpu().numpy(),
        "actor_input": actor_input.detach().cpu().numpy(),
    }
    for name, (native_model, sample, input_name) in native_inputs.items():
        with torch.inference_mode():
            expected = native_model(sample)
            scripted = torch.jit.load(str(models_dir / f"{name}.ts"), map_location="cpu").eval()
            actual_ts = scripted(sample)
            actual_onnx = _onnx_output(models_dir / f"{name}.onnx", input_name, sample)
        ts_error = _max_abs(expected, actual_ts)
        onnx_error = _max_abs(expected, actual_onnx)
        model_reports[name] = {
            "pytorch_vs_torchscript_max_abs": ts_error,
            "pytorch_vs_onnx_max_abs": onnx_error,
            "torchscript_tolerance": torchscript_tolerance,
            "onnx_tolerance": onnx_tolerance,
            "torchscript_passed": ts_error <= torchscript_tolerance,
            "onnx_passed": onnx_error <= onnx_tolerance,
        }
        parity_vectors[f"{name.removesuffix('_encoder')}_expected"] = (
            expected.detach().cpu().numpy()
        )

    native_runtime = FastSlowRuntime(
        models.slow_encoder, models.fast_encoder, models.actor
    )
    scripted_runtime = FastSlowRuntime(
        torch.jit.load(str(models_dir / "slow_encoder.ts"), map_location="cpu").eval(),
        torch.jit.load(str(models_dir / "fast_encoder.ts"), map_location="cpu").eval(),
        torch.jit.load(str(models_dir / "actor.ts"), map_location="cpu").eval(),
    )
    stateful_action_error = 0.0
    stateful_context_error = 0.0
    for _ in range(185):
        proprio = 0.1 * torch.randn(21, generator=generator)
        native_step = native_runtime.step(proprio)
        scripted_step = scripted_runtime.step(proprio)
        stateful_action_error = max(
            stateful_action_error,
            _max_abs(native_step["action_clamped"], scripted_step["action_clamped"]),
        )
        stateful_context_error = max(
            stateful_context_error,
            _max_abs(native_step["context"], scripted_step["context"]),
        )
    stateful_passed = (
        stateful_action_error <= torchscript_tolerance
        and stateful_context_error <= torchscript_tolerance
    )
    all_passed = all(
        item["torchscript_passed"] and item["onnx_passed"]
        for item in model_reports.values()
    ) and stateful_passed
    report = {
        "all_passed": all_passed,
        "parity_input": parity_input_audit,
        "models": model_reports,
        "stateful_runtime": {
            "steps": 185,
            "action_max_abs": stateful_action_error,
            "context_max_abs": stateful_context_error,
            "tolerance": torchscript_tolerance,
            "passed": stateful_passed,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
        },
    }
    return report, parity_vectors


def _readme(metadata: dict[str, Any]) -> str:
    return f"""# V8.7 DeployCoreV2 model1500 StudentBest Jetson完整部署包

固定谱系：Teacher/Actor iteration {metadata['teacher_iteration']}，Student best
内部 epoch {metadata['student_epoch']}（人工计数第 {metadata['student_epoch'] + 1}
轮），seed {metadata['seed']}，best validation loss
{metadata['student_best_val']:.12g}。包内只有这个 best Student，不含 last checkpoint。

## 1. Jetson依赖与校验

```bash
cd /你的路径/V8.7_DeployCoreV2_model1500_StudentBest_Jetson完整部署包
python3 -m pip install numpy onnx onnxruntime torch
python3 verification/verify_jetson_bundle.py --bundle . --backend all --device cpu
```

只有输出 `\"passed\": true` 才能继续。ONNX Runtime 是实际运行后端；
TorchScript 仅用于一致性复核和备用。

## 2. Shadow推理

先启动现有21维观测节点，确保最后4维是真实上一帧Position/已执行CTBR，然后：

```bash
source /opt/ros/humble/setup.bash
cd /你的路径/V8.7_DeployCoreV2_model1500_StudentBest_Jetson完整部署包/runtime
python3 rl_fastslow_inference_onnx_v1.py \
  --bundle .. \
  --observation-topic /uav_payload/observation21 \
  --candidate-topic /uav_payload/rl_ctbr_candidate
```

节点收满50个连续有效观测并完成3秒shadow后才发布ready candidate。它不解锁、
不切模式，也不直接向PX4发送控制命令。

## 3. 权限网关

`runtime/rl_ctbr_offboard_handover_v1.py`继续负责人工接管、退出和实际CTBR发送；
`runtime/rl_ctbr_handover_core_v1.py`是其纯逻辑核心。推理节点和权限网关应写入
同一运行目录，分别保存上下文/候选动作与实际发送动作。

## 4. 推理日志

`rl_inference_trace.csv`每个有效tick记录21维观测、history fill、z0-z4、慢分支
raw/target/cache、raw/clamped CTBR、上一帧实际CTBR、刷新标志、输入时效和各阶段
延迟。按行序列即可严格重建任意时刻的50x21历史窗口。

## 5. 安全边界

模型格式一致性不等于自由飞行批准。首次必须Position起飞、后台shadow、飞手保留
退出权，并同时保存ULog和Jetson日志。输入过期或非有限时，推理节点会撤销ready，
且不会重复发布旧动作。
"""


def validate_best_student_source(path: str | Path) -> dict[str, Any]:
    student_path = Path(path).resolve()
    if not student_path.name.startswith("best_"):
        raise ValueError(
            f"Student deployment source must be a best checkpoint, got {student_path.name}"
        )
    checkpoint = torch.load(student_path, map_location="cpu", weights_only=False)
    val_hist = checkpoint.get("val_hist")
    best_val = float(checkpoint.get("best_val", float("nan")))
    epoch = int(checkpoint.get("epoch", -1))
    if not isinstance(val_hist, list) or not val_hist:
        raise ValueError("Best Student checkpoint has no validation history")
    if epoch != len(val_hist) - 1:
        raise ValueError(
            "Best Student checkpoint epoch does not match its validation history"
        )
    if not np.isfinite(best_val) or abs(best_val - min(val_hist)) > 1.0e-12:
        raise ValueError("Best Student checkpoint best_val is inconsistent")
    return checkpoint


def _copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required deployment source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.chmod(destination.stat().st_mode | 0o200)
    shutil.copy2(source, destination)
    destination.chmod(0o644)


def _preexport_acceptance_report(
    student_dir: Path, metadata: dict[str, Any]
) -> tuple[str, Path, Path]:
    summary_path = student_dir / "phase2_student_play_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing Student closed-loop summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    csv_path = Path(summary["csv_path"]).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing Student closed-loop CSV: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    student_rows = [row for row in rows if row.get("control_source") == "student"]
    if not student_rows:
        raise ValueError("Student closed-loop CSV contains no student-controlled rows")

    abs_z = [abs(float(row["payload_err_z"])) for row in student_rows]
    z_values = np.asarray(
        [[float(row[f"zH{index}"]) for index in range(5)] for row in student_rows],
        dtype=np.float64,
    )
    severe_low = np.asarray(DEFAULT_CONTEXT_SEVERE_LOW)
    severe_high = np.asarray(DEFAULT_CONTEXT_SEVERE_HIGH)
    severe_rows = int(
        np.count_nonzero(
            np.any((z_values < severe_low) | (z_values > severe_high), axis=1)
        )
    )
    disable_wind = bool(
        summary.get("evaluation_overrides", {}).get("disable_wind", False)
    )
    report = f"""# DeployCoreV2 仿真预导出验收报告

## 结论

**状态：SHADOW ONLY，未通过自由飞行放行。** 模型格式可导出并做Jetson旁路
推理，但现有证据不足以声称 `model_1500 + best Student` 已完成正式实飞验收。

## 已固定的模型与训练配置

- Teacher iteration：{metadata['teacher_iteration']}
- Student best：内部epoch {metadata['student_epoch']}（人工计数第 {metadata['student_epoch'] + 1}轮）
- Student best validation loss：{metadata['student_best_val']:.12g}
- seed：{metadata['seed']}
- context：Teacher `{metadata['teacher_context_mode']}`，Student `{metadata['student_context_mode']}`
- DeployCoreV2 Gap、质量、推力曲线和观测随机化：见
  `training_config_snapshot/env.yaml`、`dataset_audit.json`和`collect_report.json`。

## 当前闭环硬证据

- 已有闭环评估seed：仅42（这是评估seed，不是多训练seed）。
- Student实际接管行数：{len(student_rows)}，时段
  {student_rows[0]['time_s']}–{student_rows[-1]['time_s']} s。
- Student接管后payload z绝对误差：均值 {np.mean(abs_z):.6f} m，
  最大 {np.max(abs_z):.6f} m；该轨迹未显示之前的持续掉高。
- Student上下文z0-z4最小值：{z_values.min(axis=0).tolist()}
- Student上下文z0-z4最大值：{z_values.max(axis=0).tolist()}
- 按部署严重越界边界计数：{severe_rows}/{len(student_rows)}。
- Summary中的环境风开关：`disable_wind={str(disable_wind).lower()}`。
  因此即使CSV文件名包含`noambientwind`，本次也**不能作为严格无风证据**。

## 尚未通过的阻断项

1. 没有Teacher与Student在相同条件下的多seed配对闭环统计；
2. 没有把真实Position控制器CTBR作为50H前缀的仿真闭环验收；当前3秒前缀是
   Teacher shadow，不等同于PX4 Position前缀；
3. Student监督数据来自Teacher闭环完整历史，尚未证明Position前缀属于相同输入分布；
4. 尚无真正启用`--eval_disable_wind`的严格无风闭环结果；
5. 上下文边界保护是部署安全门，不是对Student辨识问题已解决的证明。

## 放行规则

当前压缩包只允许：校验模型、读取真机21维、后台shadow、记录z/CTBR/延迟，
不得据此直接切换RL自由飞行。补齐上述配对仿真并通过后，才能把Manifest中的
`flight_approved`从false提升。
"""
    return report, summary_path, csv_path


def export_bundle(
    teacher_checkpoint: str | Path,
    student_checkpoint: str | Path,
    output_dir: str | Path,
    parity_dataset_dir: str | Path,
    *,
    handover_core: str | Path = "/home/shenji/桌面/rl_ctbr_handover_core_v1.py",
    handover_node: str | Path = "/home/shenji/桌面/rl_ctbr_offboard_handover_v1.py",
) -> dict[str, Any]:
    teacher_path = Path(teacher_checkpoint).resolve()
    student_path = Path(student_checkpoint).resolve()
    output = Path(output_dir).resolve()
    if not teacher_path.is_file() or not student_path.is_file():
        raise FileNotFoundError(
            f"Missing source checkpoint: teacher={teacher_path}, student={student_path}"
        )

    validate_best_student_source(student_path)
    source_hashes = {
        "teacher": sha256_file(teacher_path),
        "student": sha256_file(student_path),
    }
    output.mkdir(parents=True, exist_ok=True)
    validate_existing_bundle_lineage(output, source_hashes)
    models, metadata = load_deployment_models(teacher_path, student_path)
    if metadata["teacher_iteration"] != 1500:
        raise ValueError(
            f"This V8.7 bundle requires Teacher iteration 1500, got {metadata['teacher_iteration']}"
        )

    models_dir = output / "models"
    runtime_dir = output / "runtime"
    verification_dir = output / "verification"
    config_dir = output / "config"
    snapshot_dir = config_dir / "training_config_snapshot"
    for directory in (
        models_dir,
        runtime_dir,
        verification_dir,
        snapshot_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    history_example = torch.zeros(1, 50, 21, dtype=torch.float32)
    actor_example = torch.zeros(1, 26, dtype=torch.float32)
    export_specs = {
        "slow_encoder": (models.slow_encoder, history_example, "history", "z_slow"),
        "fast_encoder": (models.fast_encoder, history_example, "history", "z_fast"),
        "actor": (models.actor, actor_example, "actor_input", "ctbr_raw"),
    }
    for name, (model, example, input_name, output_name) in export_specs.items():
        _export_torchscript(model, example, models_dir / f"{name}.ts")
        _export_onnx(
            model,
            example,
            models_dir / f"{name}.onnx",
            input_name=input_name,
            output_name=output_name,
        )

    _copy_required(teacher_path, models_dir / "source_teacher_model_1500.pt")
    _copy_required(student_path, models_dir / "source_student_best.pth")
    module_dir = Path(__file__).resolve().parent
    for name in (
        "jetson_reference_runtime.py",
        "jetson_onnx_runtime.py",
        "jetson_inference_trace.py",
        "rl_fastslow_inference_onnx_v1.py",
    ):
        _copy_required(module_dir / name, runtime_dir / name)
    _copy_required(Path(handover_core).resolve(), runtime_dir / "rl_ctbr_handover_core_v1.py")
    _copy_required(Path(handover_node).resolve(), runtime_dir / "rl_ctbr_offboard_handover_v1.py")
    _copy_required(
        module_dir / "verify_jetson_bundle.py",
        verification_dir / "verify_jetson_bundle.py",
    )

    run_root = teacher_path.parent
    dataset_root = Path(parity_dataset_dir).resolve()
    snapshot_sources = {
        "agent.yaml": run_root / "params" / "agent.yaml",
        "env.yaml": run_root / "params" / "env.yaml",
        "context_architecture.json": run_root / "context_architecture.json",
        "dataset_audit.json": dataset_root / "dataset_audit.json",
        "collect_report.json": dataset_root / "collect_report.json",
        "meta_uav_env.py": module_dir / "meta_uav_env.py",
        "meta_uav_env_cfg.py": module_dir / "meta_uav_env_cfg.py",
    }
    acceptance_report, summary_path, student_csv_path = _preexport_acceptance_report(
        student_path.parent, metadata
    )
    snapshot_sources["phase2_student_play_summary.json"] = summary_path
    snapshot_sources["phase2_student_shadow3s_seed42.csv"] = student_csv_path
    for destination_name, source in snapshot_sources.items():
        _copy_required(source, snapshot_dir / destination_name)
    (config_dir / "仿真预导出验收报告.md").write_text(
        acceptance_report, encoding="utf-8"
    )

    manifest = build_manifest(metadata, source_hashes)
    (config_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(_readme(metadata), encoding="utf-8")

    parity_history, parity_input_audit = load_parity_history(parity_dataset_dir)
    parity, parity_vectors = _build_parity_report(
        models, output, parity_history, parity_input_audit
    )
    np.savez(verification_dir / "parity_vectors.npz", **parity_vectors)
    (verification_dir / "parity_report.json").write_text(
        json.dumps(parity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not parity["all_passed"]:
        raise RuntimeError(
            f"Deployment numerical parity failed; inspect {verification_dir / 'parity_report.json'}"
        )

    missing = [name for name in REQUIRED_ARTIFACTS if name != "sha256sums.txt" and not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Deployment bundle is missing artifacts: {missing}")
    checksum_names = sorted(name for name in REQUIRED_ARTIFACTS if name != "sha256sums.txt")
    checksum_text = "".join(
        f"{sha256_file(output / name)}  {name}\n" for name in checksum_names
    )
    (output / "sha256sums.txt").write_text(checksum_text, encoding="utf-8")
    return {
        "output_dir": str(output),
        "manifest": manifest,
        "parity_report": parity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True, help="Phase-I Teacher checkpoint")
    parser.add_argument("--student", required=True, help="Best Phase-II Student checkpoint")
    parser.add_argument("--output", required=True, help="Output deployment bundle directory")
    parser.add_argument(
        "--parity_data_dir",
        required=True,
        help="Collected dataset directory providing physical history samples for parity",
    )
    args = parser.parse_args()
    result = export_bundle(
        args.teacher, args.student, args.output, args.parity_data_dir
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
