# Fast/Slow Multi-Seed Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe evaluation-only fixed-physics switches, collect fifteen paired July 24 rollouts, and build a reproducible multi-seed data package.

**Architecture:** The environment configuration stores optional fixed evaluation values while retaining the original randomization ranges for privileged-context normalization. The Phase-2 player validates and applies those overrides, records total wind and lineage metadata, and a standalone analysis script audits and summarizes the resulting directory tree.

**Tech Stack:** Python 3, Isaac Lab, PyTorch, pandas, NumPy, Matplotlib, unittest.

---

### Task 1: Add failing static tests for evaluation overrides

**Files:**
- Create: `tests/test_fastslow_eval_overrides.py`
- Inspect: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env_cfg.py`
- Inspect: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env.py`
- Inspect: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/play_student_phase2.py`

- [ ] **Step 1: Write tests for the three CLI switches**

Assert that the Phase-2 player declares `--eval_payload_mass_kg`,
`--eval_rope_length_m`, and `--eval_disable_wind`.

- [ ] **Step 2: Write tests for non-destructive normalization**

Assert that fixed reset fields exist separately from `payload_mass_range` and
`rope_length_range`, and that reset code reads the fixed fields without
overwriting the ranges.

- [ ] **Step 3: Write tests for audit output**

Assert that the CSV includes total wind acceleration and the summary includes
an `evaluation_overrides` object.

- [ ] **Step 4: Verify RED**

Run:

```bash
python3 -m unittest -v tests.test_fastslow_eval_overrides
```

Expected: failures because the switches and audit fields do not yet exist.

### Task 2: Implement safe fixed evaluation conditions

**Files:**
- Modify: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env_cfg.py`
- Modify: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env.py`
- Modify: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/play_student_phase2.py`

- [ ] **Step 1: Add optional configuration fields**

Add `eval_fixed_payload_mass_kg`, `eval_fixed_rope_length_m`, and
`eval_disable_wind` with inactive defaults.

- [ ] **Step 2: Apply fixed reset values**

Use the fixed rope length and payload mass only when the corresponding value is
not `None`; otherwise retain the existing randomization.

- [ ] **Step 3: Validate and wire CLI arguments**

Reject fixed values outside the original training ranges, assign valid values
to the environment configuration, and disable wind only when requested.

- [ ] **Step 4: Extend raw audit fields**

Append `wind_acc_x_mps2`, `wind_acc_y_mps2`, and `wind_acc_z_mps2` to the CSV
and add the exact override values to the summary JSON.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
python3 -m unittest -v tests.test_fastslow_eval_overrides
python3 -m py_compile \
  source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env_cfg.py \
  source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env.py \
  source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/play_student_phase2.py
```

Expected: all focused tests pass and all three files compile.

### Task 3: Build the multi-seed analysis package

**Files:**
- Create: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/build_fastslow_multiseed_package.py`
- Create: `tests/test_fastslow_multiseed_package.py`

- [ ] **Step 1: Write failing tests for metrics and pair auditing**

Create synthetic rollout frames and assert correct position RMSE, swing RMS,
action total variation, high-frequency energy, fixed no-wind validation, and
exact exogenous pairing.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m unittest -v tests.test_fastslow_multiseed_package
```

Expected: import failure because the package builder does not exist.

- [ ] **Step 3: Implement deterministic metrics and audits**

Read the manifest and rollout tree, compute per-rollout and paired metrics,
separate the fixed no-wind case from random-wind aggregation, and hash the raw
artifacts.

- [ ] **Step 4: Implement figures**

Produce scenario audit, position, swing, context recovery, CTBR, computation,
gust response, aggregate performance, and Student–Teacher gap figures.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
python3 -m unittest -v tests.test_fastslow_multiseed_package
```

Expected: all synthetic-data tests pass.

### Task 4: Create the experiment manifest and run fifteen rollouts

**Files:**
- Create: `logs/rsl_rl/uav_payload_sim2real_rl/2026-07-24_06-06-18_hardexplicit_teacher_fastslow_seed42/快慢结构7.27数据整理/experiment_manifest.json`
- Create: per-rollout `rollout.csv`, `summary.json`, and `console.log` files.

- [ ] **Step 1: Verify source lineage**

Check that `model_19999.pt` and
`best_fast_slow_student_encoder_z.pth` exist and record their SHA256 hashes.

- [ ] **Step 2: Execute the five Teacher rollouts**

Use seeds 38, 40, 42, 44, and fixed no-wind seed 46 with the July 24 Teacher.

- [ ] **Step 3: Execute the five All-60 Hz Student rollouts**

Use the same scenario definitions and the July 24 Student checkpoint with
`--context_runtime_mode all_60hz`.

- [ ] **Step 4: Execute the five fast/slow Student rollouts**

Use the same scenario definitions with `--slow_warmup_sec 3.0`,
`--slow_update_hz 1.0`, `--fast_update_hz 60.0`,
`--slow_filter_tau_sec 0.25`, and `--context_runtime_mode fast_slow`.

- [ ] **Step 5: Stop on any failed command**

Do not retry automatically. Preserve the console log and report the exact
failing command.

### Task 5: Analyze and verify the package

**Files:**
- Create: all `data/` and `figures/` outputs under `快慢结构7.27数据整理`
- Create: `快慢结构7.27数据整理/README.md`

- [ ] **Step 1: Run the package builder**

Run:

```bash
python3 source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/build_fastslow_multiseed_package.py \
  --package_dir "/home/shenji/uav_payload_lab/uav_payload_lab/logs/rsl_rl/uav_payload_sim2real_rl/2026-07-24_06-06-18_hardexplicit_teacher_fastslow_seed42/快慢结构7.27数据整理"
```

Expected: exit code 0 with metrics, audit, hashes, and figures written.

- [ ] **Step 2: Verify package completeness**

Confirm 15 non-empty rollout CSV files, 15 summary JSON files, 15 console logs,
all required data tables, and all required figures.

- [ ] **Step 3: Verify the fixed scenario**

Confirm recorded mass is exactly 0.55 kg, rope length exactly 0.525 m, and total
wind acceleration exactly zero for Teacher, All-60 Hz, and fast/slow.

- [ ] **Step 4: Verify strict schedule pairing**

Confirm All-60 Hz and fast/slow exogenous inputs are identical over the common
episode prefix for every scenario.

- [ ] **Step 5: Record limitations**

State that five evaluation seeds quantify rollout variability but do not replace
multiple independently trained Teacher/Student seeds.
