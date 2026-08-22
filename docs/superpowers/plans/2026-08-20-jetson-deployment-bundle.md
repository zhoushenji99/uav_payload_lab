# Jetson Deployment Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a complete ONNX/TorchScript deployment bundle for the validated hover Student and Actor.

**Architecture:** A pure-PyTorch exporter reconstructs the two CNN Student branches and the normalized Actor from checkpoint state dictionaries. A separate Isaac-Sim-free runtime owns the 50-frame history and fast/slow scheduling. Export verification compares native PyTorch, TorchScript, and ONNX outputs and records hashes and metadata.

**Tech Stack:** Python 3.11, PyTorch 2.7, ONNX 1.19, standard library JSON/hash utilities, unittest/pytest.

---

### Task 1: Define checkpoint loaders and deployable modules

**Files:**
- Create: `tests/test_jetson_deployment.py`
- Create: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/jetson_deployment.py`

- [x] Write failing tests that require strict Student lineage checks, an Actor wrapper that includes `(x - mean) / (std + 1e-2)`, and exact output dimensions.
- [x] Run `/home/shenji/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_jetson_deployment.py -q` and confirm import failure because the module does not exist.
- [x] Implement `CNNContextEncoder`, `SlowEncoder`, `FastEncoder`, `NormalizedActor`, and strict checkpoint-loading helpers.
- [x] Re-run the focused test and require all assertions to pass.

### Task 2: Define the stateful Jetson reference runtime

**Files:**
- Modify: `tests/test_jetson_deployment.py`
- Create: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/jetson_reference_runtime.py`

- [x] Add failing tests for zero-padded 50-frame history, 180-step startup, 60-step slow period, one-step fast period, causal EMA, and CTBR bounds.
- [x] Run the focused test and confirm failure because the runtime does not exist.
- [x] Implement `FastSlowRuntime` with explicit cache state and deterministic `step(proprio_21)` behavior.
- [x] Re-run the focused test and require all assertions to pass.

### Task 3: Export bundle and metadata

**Files:**
- Modify: `tests/test_jetson_deployment.py`
- Create: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/export_jetson_bundle.py`

- [x] Add failing tests for required manifest fields, artifact names, source hashes, and refusal to overwrite mismatched lineage.
- [x] Run the focused test and confirm failure because the exporter does not exist.
- [x] Implement ONNX/TorchScript export, manifest/README generation, checkpoint copying, and SHA256 generation.
- [x] Use ONNX opset 17 with dynamic batch axes but fixed `[50, 21]` history dimensions.
- [x] Re-run the focused test and require all assertions to pass.

### Task 4: Numerical parity and actual run export

**Files:**
- Generate: `logs/rsl_rl/uav_payload_sim2real_hover_rl/2026-08-19_21-26-54_hardexplicit_teacher_hover_seed42/StudentFastSlow_hover_model3000_seed42_noprobe/jetson_deployment_bundle/*`

- [x] Export the actual `model_3000.pt` and best epoch-473 Student checkpoint.
- [x] Validate all ONNX files with `onnx.checker`.
- [x] Compare PyTorch and TorchScript outputs at `atol=1e-6`.
- [x] Compare PyTorch and ONNX ReferenceEvaluator outputs at `atol=1e-5`.
- [x] Run the stateful runtime test over deterministic input history.
- [x] Write `parity_report.json`, `manifest.json`, `README.md`, and `sha256sums.txt`.
- [x] Run `git diff --check` and the focused regression suite before reporting completion.

The repository contained unrelated user changes. The V8.5 release commit stages
only the hover-task and Jetson-deployment files listed in this plan.
