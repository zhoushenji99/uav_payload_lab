# Phase-II Full-History Shadow Handover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in three-second Teacher-shadow warmup that fills the Student history and context caches before Student closed-loop control.

**Architecture:** Put tensor selection and duration validation in a small Isaac-independent runtime helper, call it from `play_student_phase2.py`, and expose the executed source plus Student candidate in the CSV. The legacy path remains unchanged when the new duration is zero.

**Tech Stack:** Python, PyTorch, NumPy, Isaac Lab, `unittest`.

---

### Task 1: Test the shadow handover contract

**Files:**
- Create: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/phase2_shadow_handover.py`
- Create: `regression_tests/test_phase2_shadow_handover.py`

- [ ] **Step 1: Write failing tests** for duration validation, the last Teacher-controlled step, the first Student-controlled step, and per-environment action selection.
- [ ] **Step 2: Run the focused test** with Isaac Lab Python and confirm failure because the helper module does not exist.
- [ ] **Step 3: Implement the minimal helper** `validate_shadow_warmup`, `teacher_shadow_mask`, and `select_shadow_actions`.
- [ ] **Step 4: Run the focused test** and confirm all cases pass.

### Task 2: Wire shadow control into Phase-II play

**Files:**
- Modify: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/play_student_phase2.py`
- Modify: `regression_tests/test_phase2_shadow_handover.py`

- [ ] **Step 1: Add a failing source-contract test** requiring the new CLI option, control-source CSV field, Student candidate fields, and helper call.
- [ ] **Step 2: Run the focused test** and confirm the source-contract assertions fail.
- [ ] **Step 3: Add `--student_shadow_warmup_sec`**, validate it against history and slow startup durations, run Teacher and Student inference during shadow, select Teacher actions before the boundary, and keep all Student buffers intact.
- [ ] **Step 4: Add CSV and JSON audit fields** for executed source, Student candidate, shadow duration, and shadow steps.
- [ ] **Step 5: Run the focused regression tests** and confirm they pass.

### Task 3: Verify the existing runtime contract

**Files:**
- Verify: `regression_tests/test_fastslow_runtime_audit.py`
- Verify: `regression_tests/test_phase2_shadow_handover.py`

- [ ] **Step 1: Run both focused regression modules** using Isaac Lab Python.
- [ ] **Step 2: Run Python syntax compilation** for the helper and rollout script.
- [ ] **Step 3: Inspect the final diff** to confirm no Teacher, environment, reward, collection, or Student-training files changed.
- [ ] **Step 4: Run the one-environment no-wind rollout** with `--student_shadow_warmup_sec 3.0` and compare it with the existing zero-history CSV.
