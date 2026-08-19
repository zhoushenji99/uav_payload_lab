# Student Training Seed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Phase-II Student training and its 500-to-1000 epoch resume reproducible under an explicit experiment seed.

**Architecture:** The Sim2Real Student trainer will require `--seed`, configure deterministic Python, NumPy, PyTorch, CUDA, and DataLoader behavior before model construction, and serialize every RNG state needed for exact continuation. Seeded training will reject legacy or mismatched resume checkpoints so an apparently comparable run cannot silently inherit uncontrolled randomness.

**Tech Stack:** Python, PyTorch, NumPy, pytest, IsaacLab Python launcher.

---

### Task 1: Reproducibility regression tests

**Files:**
- Create: `regression_tests/test_train_student_z_seed.py`

- [ ] **Step 1: Write failing unit tests**

  Test that `--seed` is mandatory, reseeding reproduces Python/NumPy/PyTorch draws, RNG capture/restore resumes the same sequence, and resume validation rejects legacy or mismatched checkpoints.

- [ ] **Step 2: Write a failing end-to-end resume test**

  Build a tiny audited five-dimensional dataset, compare a two-epoch uninterrupted run with a one-epoch plus resumed second epoch run, and require identical histories and model tensors.

- [ ] **Step 3: Run the focused test before implementation**

  Run:

  ```bash
  TERM=xterm ~/IsaacLab/isaaclab.sh -p -m pytest -q regression_tests/test_train_student_z_seed.py
  ```

  Expected: failure because the seed and RNG helper behavior does not exist yet.

### Task 2: Seeded Student trainer

**Files:**
- Modify: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/train_student_z.py`

- [ ] **Step 1: Add explicit seed configuration**

  Require `--seed`; seed Python, NumPy, CPU/CUDA PyTorch, enable deterministic algorithms, and provide a deterministic DataLoader worker initializer.

- [ ] **Step 2: Make shard shuffling reproducible**

  Give every training DataLoader a dedicated seeded `torch.Generator` so model randomness and sample-order randomness are controlled independently.

- [ ] **Step 3: Make resume exact and auditable**

  Save the experiment seed, global RNG states, and DataLoader generator state in best and last checkpoints. On resume, reject checkpoints lacking this metadata or carrying another seed, then restore the states before the next epoch.

- [ ] **Step 4: Record reproducibility metadata**

  Add the seed and deterministic setting to `report.json` and print them at startup.

### Task 3: Verification and handoff

**Files:**
- Test: `regression_tests/test_train_student_z_seed.py`

- [ ] **Step 1: Run the focused regression test**

  Run:

  ```bash
  TERM=xterm ~/IsaacLab/isaaclab.sh -p -m pytest -q regression_tests/test_train_student_z_seed.py
  ```

  Expected: all tests pass, including exact uninterrupted-versus-resumed equivalence.

- [ ] **Step 2: Compile the modified trainer**

  Run:

  ```bash
  ~/IsaacLab/isaaclab.sh -p -m py_compile source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/train_student_z.py
  ```

  Expected: exit code 0.

- [ ] **Step 3: Provide fresh paired commands**

  Use new output directories and the same explicit seed, data schedule, batch size, optimizer schedule, weighting, and auxiliary coefficient for FastSlow and Coupled. Do not resume any legacy unseeded checkpoint.
