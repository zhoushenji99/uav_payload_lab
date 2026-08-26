# V8.7 DeployCoreV2 Jetson Complete Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the fixed model_1500 Actor and best-only Fast-Slow Student as a verified Jetson ROS 2 deployment archive with stateful ONNX inference, safe candidate-only publishing, and complete analysis logs.

**Architecture:** Generalize the existing exporter around a nested, manifest-driven bundle. Add a pure NumPy/ONNX state machine for 50H and dual-rate scheduling, then wrap it in a ROS 2 node that publishes candidate CTBR only. Keep PX4 authority in the existing gateway and verify numerical parity, scheduling, logging, checksums, and archive contents before delivery.

**Tech Stack:** Python 3.11, PyTorch/TorchScript, ONNX/ONNX Runtime, NumPy, ROS 2 Humble rclpy, pytest/unittest, SHA-256, ZIP.

---

### Task 1: Generalize bundle structure and lineage

**Files:**
- Modify: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/export_jetson_bundle.py`
- Modify: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/verify_jetson_bundle.py`
- Modify: `tests/test_jetson_deployment.py`

- [ ] Write failing tests that require nested `models/`, `runtime/`, `verification/`, and `config/` paths, generic source checkpoint names, current mass/rope ranges, and best-only Student lineage.
- [ ] Run `python -m pytest tests/test_jetson_deployment.py -q` and confirm the new assertions fail against the flat legacy exporter.
- [ ] Implement manifest-driven nested paths, copy only `source_student_best.pth`, preserve the exact Teacher/Student source hashes, and reject a Student checkpoint whose metadata does not identify the selected best checkpoint.
- [ ] Update the verifier to resolve nested paths safely and validate required artifacts from `config/manifest.json`.
- [ ] Re-run `python -m pytest tests/test_jetson_deployment.py -q` and confirm PASS.

### Task 2: Add stateful ONNX fast-slow runtime

**Files:**
- Create: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/jetson_onnx_runtime.py`
- Create: `tests/test_jetson_onnx_runtime.py`

- [ ] Write failing tests for 50x21 FIFO reconstruction, invalid-input reset, fast refresh every valid tick, slow refresh at steps 0-179 then 180/240, 0.25 s causal EMA, action clipping, and readiness only after 50 valid frames plus 180 valid shadow ticks.
- [ ] Run `python -m pytest tests/test_jetson_onnx_runtime.py -q` and confirm import/behavior failures.
- [ ] Implement `FastSlowOnnxRuntime` with injectable sessions for tests and ONNX Runtime sessions for Jetson.
- [ ] Return a structured tick record containing contexts, raw/cache values, update flags, raw/clamped CTBR, previous executed CTBR, fill/readiness, rejection reason, and per-stage latency.
- [ ] Re-run `python -m pytest tests/test_jetson_onnx_runtime.py -q` and confirm PASS.

### Task 3: Add reconstructable inference CSV trace

**Files:**
- Create: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/jetson_inference_trace.py`
- Extend: `tests/test_jetson_onnx_runtime.py`

- [ ] Write failing tests that require timestamp fields, `obs_0..obs_20`, history state, `z0..z4`, slow raw/target/cache, refresh flags, raw/clamped CTBR, previous executed CTBR, latency, age, readiness, and reject reason.
- [ ] Implement a buffered CSV writer with explicit `flush()` and `close()` and deterministic field ordering.
- [ ] Verify consecutive rows and history-fill/reset fields are sufficient to reconstruct every 50H window.
- [ ] Run `python -m pytest tests/test_jetson_onnx_runtime.py -q` and confirm PASS.

### Task 4: Add ROS 2 candidate-only inference node

**Files:**
- Create: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/rl_fastslow_inference_onnx_v1.py`
- Create: `tests/test_rl_fastslow_inference_contract.py`

- [ ] Write a source-contract test requiring `/uav_payload/observation_21`, `/uav_payload/rl_ctbr_candidate`, context/status topics, 60 Hz timer, manifest-based model loading, run-directory logging, and no PX4 authority message imports or publishers.
- [ ] Run `python -m pytest tests/test_rl_fastslow_inference_contract.py -q` and confirm FAIL because the node does not exist.
- [ ] Implement the ROS 2 wrapper, validating observation dimension/finiteness/age and publishing candidate CTBR only when runtime readiness is true.
- [ ] Preserve rejection/status logging and stop candidate publication immediately on invalid or stale input.
- [ ] Re-run `python -m pytest tests/test_rl_fastslow_inference_contract.py -q` and confirm PASS.

### Task 5: Export, verify, and package the fixed model

**Inputs:**
- Teacher: `logs/rsl_rl/uav_payload_sim2real_hover_deploy_core_v2/2026-08-25_01-14-54_hardexplicit_teacher_hover_deploy_core_v2_seed42/model_1500.pt`
- Student: `logs/rsl_rl/uav_payload_sim2real_hover_deploy_core_v2/2026-08-25_01-14-54_hardexplicit_teacher_hover_deploy_core_v2_seed42/StudentFastSlow_hover_deploy_core_v2_model1500_seed42_noprobe/best_fast_slow_student_encoder_z.pth`
- Dataset: `logs/rsl_rl/uav_payload_sim2real_hover_deploy_core_v2/2026-08-25_01-14-54_hardexplicit_teacher_hover_deploy_core_v2_seed42/DecoderAddedDataset_hover_deploy_core_v2_model1500_seed42_noprobe`
- Output archive: `/home/shenji/桌面/V8.7_DeployCoreV2_model1500_StudentBest_Jetson完整部署包.zip`

- [ ] Run the focused test suite and syntax checks before export.
- [ ] Export Actor, slow encoder, and fast encoder to ONNX and TorchScript, copy exact source checkpoints, current configuration snapshots, existing handover gateway scripts, runtime sources, parity vectors, and verification script.
- [ ] Generate `README.md`, `config/manifest.json`, `verification/parity_report.json`, and `sha256sums.txt` with copy-runnable Jetson verification/shadow commands.
- [ ] Run ONNX checker, PyTorch/TorchScript parity (`<=1e-6`), PyTorch/ONNX parity (`<=1e-5`), and the 185-step stateful scheduler parity test.
- [ ] Assert the bundle contains `source_student_best.pth` and contains no `last_checkpoint.pth` or alternate Student checkpoint.
- [ ] Create the ZIP, extract it into a temporary directory, rerun checksums and model verification, and confirm the README/runtime files survive unchanged.
- [ ] Run `git diff --check` and report only the intentional deployment source/test changes; do not stage unrelated user files or generated model artifacts.
