"""Export and verify the fast/slow hover policy for NVIDIA Jetson deployment."""

from __future__ import annotations

import argparse
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
from jetson_reference_runtime import FastSlowRuntime


REQUIRED_ARTIFACTS = [
    "actor.onnx",
    "actor.ts",
    "slow_encoder.onnx",
    "slow_encoder.ts",
    "fast_encoder.onnx",
    "fast_encoder.ts",
    "manifest.json",
    "parity_report.json",
    "parity_vectors.npz",
    "sha256sums.txt",
    "README.md",
    "jetson_reference_runtime.py",
    "verify_jetson_bundle.py",
    "source_teacher_model_3000.pt",
    "source_student_best_epoch473.pth",
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
            "student_epoch": metadata["student_epoch"],
            "student_best_val": metadata["student_best_val"],
            "seed": metadata["seed"],
            "teacher_context_mode": metadata["teacher_context_mode"],
            "student_context_mode": metadata["student_context_mode"],
        },
        "source_hashes": dict(source_hashes),
        "models": {
            "slow_encoder": {
                "onnx": "slow_encoder.onnx",
                "torchscript": "slow_encoder.ts",
                "input_name": "history",
                "input_shape": ["B", 50, 21],
                "output_name": "z_slow",
                "output_shape": ["B", 2],
            },
            "fast_encoder": {
                "onnx": "fast_encoder.onnx",
                "torchscript": "fast_encoder.ts",
                "input_name": "history",
                "input_shape": ["B", 50, 21],
                "output_name": "z_fast",
                "output_shape": ["B", 3],
            },
            "actor": {
                "onnx": "actor.onnx",
                "torchscript": "actor.ts",
                "input_name": "actor_input",
                "input_shape": ["B", 26],
                "output_name": "ctbr_raw",
                "output_shape": ["B", 4],
                "observation_normalization_embedded": True,
                "normalizer_eps": metadata["normalizer_eps"],
            },
        },
        "observation_21": {
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
            "initial_history": "all_zeros",
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
            "clamp_required_after_actor": True,
        },
        "training_ranges": {
            "payload_mass_kg": [0.3, 0.8],
            "rope_length_m": [0.25, 0.8],
        },
        "safety": {
            "flight_approved": False,
            "startup_guard_required_before_flight": True,
            "reason": (
                "The exact-parity runtime uses an all-zero initial history and immediately "
                "applies Student context. Validate a real-flight authority handover guard "
                "and the actual vehicle mass/thrust model before flight."
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
            scripted = torch.jit.load(str(output / f"{name}.ts"), map_location="cpu").eval()
            actual_ts = scripted(sample)
            actual_onnx = _onnx_output(output / f"{name}.onnx", input_name, sample)
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
        torch.jit.load(str(output / "slow_encoder.ts"), map_location="cpu").eval(),
        torch.jit.load(str(output / "fast_encoder.ts"), map_location="cpu").eval(),
        torch.jit.load(str(output / "actor.ts"), map_location="cpu").eval(),
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
    return f"""# Jetson Fast-Slow RMA Deployment Bundle

This bundle exports Teacher iteration {metadata['teacher_iteration']} and the
best Student epoch {metadata['student_epoch']} (seed {metadata['seed']}).

## Recommended deployment

- Convert `actor.onnx`, `slow_encoder.onnx`, and `fast_encoder.onnx` to three
  TensorRT engines on the target Jetson.
- Keep the history buffer and rate scheduler in host code. The exact reference
  implementation is `jetson_reference_runtime.py`.
- Feed exactly the 21 values and units recorded in `manifest.json`.
- Concatenate current proprioception with `[z_slow_cache, z_fast_cache]` before
  calling the Actor.
- Clamp the raw Actor output to the CTBR limits recorded in `manifest.json`.

## TorchScript smoke example

```python
import torch
from jetson_reference_runtime import load_torchscript_runtime

runtime = load_torchscript_runtime(".")
proprio_21 = torch.zeros(21, dtype=torch.float32)
result = runtime.step(proprio_21)
ctbr = result["action_clamped"][0]
```

## Target-side consistency test

```bash
python3 verify_jetson_bundle.py --bundle . --backend all --device cpu
```

## Safety

`parity_report.json` proves model-format consistency, not flight safety. This
bundle reproduces the simulator's zero-padded startup. Do not hand it CTBR flight
authority until the startup handover, real vehicle mass/thrust mapping, units,
quaternion ordering, and failsafes have been independently validated.
"""


def export_bundle(
    teacher_checkpoint: str | Path,
    student_checkpoint: str | Path,
    output_dir: str | Path,
    parity_dataset_dir: str | Path,
) -> dict[str, Any]:
    teacher_path = Path(teacher_checkpoint).resolve()
    student_path = Path(student_checkpoint).resolve()
    output = Path(output_dir).resolve()
    if not teacher_path.is_file() or not student_path.is_file():
        raise FileNotFoundError(
            f"Missing source checkpoint: teacher={teacher_path}, student={student_path}"
        )

    source_hashes = {
        "teacher": sha256_file(teacher_path),
        "student": sha256_file(student_path),
    }
    output.mkdir(parents=True, exist_ok=True)
    validate_existing_bundle_lineage(output, source_hashes)
    models, metadata = load_deployment_models(teacher_path, student_path)

    history_example = torch.zeros(1, 50, 21, dtype=torch.float32)
    actor_example = torch.zeros(1, 26, dtype=torch.float32)
    export_specs = {
        "slow_encoder": (models.slow_encoder, history_example, "history", "z_slow"),
        "fast_encoder": (models.fast_encoder, history_example, "history", "z_fast"),
        "actor": (models.actor, actor_example, "actor_input", "ctbr_raw"),
    }
    for name, (model, example, input_name, output_name) in export_specs.items():
        _export_torchscript(model, example, output / f"{name}.ts")
        _export_onnx(
            model,
            example,
            output / f"{name}.onnx",
            input_name=input_name,
            output_name=output_name,
        )

    shutil.copy2(teacher_path, output / "source_teacher_model_3000.pt")
    shutil.copy2(student_path, output / "source_student_best_epoch473.pth")
    runtime_source = Path(__file__).with_name("jetson_reference_runtime.py")
    shutil.copy2(runtime_source, output / "jetson_reference_runtime.py")
    verifier_source = Path(__file__).with_name("verify_jetson_bundle.py")
    shutil.copy2(verifier_source, output / "verify_jetson_bundle.py")

    manifest = build_manifest(metadata, source_hashes)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(_readme(metadata), encoding="utf-8")

    parity_history, parity_input_audit = load_parity_history(parity_dataset_dir)
    parity, parity_vectors = _build_parity_report(
        models, output, parity_history, parity_input_audit
    )
    np.savez(output / "parity_vectors.npz", **parity_vectors)
    (output / "parity_report.json").write_text(
        json.dumps(parity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not parity["all_passed"]:
        raise RuntimeError(
            f"Deployment numerical parity failed; inspect {output / 'parity_report.json'}"
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
