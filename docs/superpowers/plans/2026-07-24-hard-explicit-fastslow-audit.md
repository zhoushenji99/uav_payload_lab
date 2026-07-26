# Hard-Explicit Teacher and Fast/Slow Runtime Audit Plan

> **Execution boundary:** Modify only the Sim2Real task, its RSL-RL entry points,
> focused regression tests, and the V5 paper draft. Preserve all unrelated dirty
> worktree files. The user explicitly requested implementation in the current
> workspace, so no separate worktree is used.

**Goal:** Make the proposed TS-PhySCA implementation match the claimed method:
the Teacher structural context is exactly the normalized true payload mass and
rope length; the Student uses independent slow/fast encoders; deployment uses a
3 s high-rate slow startup, 1 Hz slow refresh, 60 Hz fast refresh, and a 0.25 s
causal slow-context filter; genuine monolithic/split switches and paper-grade
runtime/data audits are available.

**Architecture:** Introduce explicit context modes in `RMAActorCritic`:
`split_hard`, `split_soft`, and `monolithic`. The proposed run uses
`split_hard`, for which
`z_s = [m_norm, l_norm]` is an identity path and only the residual wind branch
is learned. Keep all normalized ranges and the five-dimensional Actor context
unchanged. Add a pure-Python runtime/audit module so scheduling, filtering, and
metrics can be regression-tested without launching Isaac Sim. Extend Phase-II
rollout logging with raw/target/cache context, exact branch calls, CUDA-
synchronized timings, action-continuity metrics, gust-response diagnostics, and
summary percentiles.

**Tech stack:** Python 3, PyTorch, NumPy, RSL-RL, Isaac Lab, `unittest`, JSON/CSV.

---

## Task 1: Lock the required semantics with focused tests

**Files:**
- Create: `regression_tests/test_rma_context_modes.py`
- Create: `regression_tests/test_fastslow_runtime_audit.py`
- Modify: `regression_tests/test_phase2_context_physical_units.py`

- [ ] Test that `split_hard.mu(priv)[..., :2]` is bitwise equal to
      `priv[..., :2]` and that the hard mode has no learned slow encoder.
- [ ] Test that `split_soft` has independent `mu_exp` and `mu_imp` networks.
- [ ] Test that `monolithic` has one five-to-five context network and no split
      encoder parameters.
- [ ] Test that incompatible checkpoint context structures are rejected rather
      than partially loaded.
- [ ] Test the 60 Hz schedule: 3 s startup gives 180 steps, 1 Hz gives a 60-step
      slow period, and the fast period is one step.
- [ ] Test the causal filter
      `alpha = 1 - exp(-dt/tau)` with `dt=1/60`, `tau=0.25`.
- [ ] Test latency percentiles, CTBR total variation, 5--30 Hz action-band
      energy, and gust-to-fast-context response latency on synthetic signals.

## Task 2: Implement exact Teacher context modes

**Files:**
- Modify:
  `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/rma_actor_critic.py`
- Modify:
  `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env_cfg.py`
- Modify: `scripts/rsl_rl/train.py`

- [ ] Add `context_mode` with valid values `split_hard`, `split_soft`, and
      `monolithic`.
- [ ] In `split_hard`, return
      `concat(priv[..., :2], mu_imp(priv[..., 2:]))`; do not instantiate
      `mu_exp`.
- [ ] In `split_soft`, retain the current independent learned `mu_exp` and
      `mu_imp` branches and optional soft physics loss.
- [ ] In `monolithic`, instantiate one learned MLP from all five privileged
      inputs to all five latent outputs.
- [ ] Make checkpoint loading strict by architecture. A checkpoint from another
      mode must fail with an actionable error.
- [ ] Set the task default to `split_hard`, physics-anchor disabled and
      coefficient zero. The hard identity path makes the auxiliary physics loss
      unnecessary.
- [ ] Add `--rma_context_mode`. Keep `--black_box_rma` only as a deprecated
      alias for `monolithic`; reject contradictory arguments.
- [ ] Write `context_architecture.json` into each training run so later
      checkpoints are auditable even after conversation context is lost.

## Task 3: Make collection hard-label-safe and auditable

**Files:**
- Modify:
  `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/collect_z_dataset.py`
- Create:
  `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/audit_z_dataset.py`

- [ ] Build the RSL-RL policy with the selected `rma_context_mode` before
      checkpoint loading.
- [ ] Load checkpoints strictly and reject an architecture mismatch.
- [ ] Continue storing `labels = z_teacher` and
      `labels_ml = priv[:, :2]`.
- [ ] For `split_hard`, assert during collection that
      `labels[:, :2] == labels_ml` exactly.
- [ ] Store Teacher mode, normalized/physical ranges, shapes, dtypes, sample
      count, finite counts, min/max/mean/std, and maximum slow-label identity
      error in metadata and `dataset_audit.json`.
- [ ] Add a standalone shard auditor that checks every shard for consistent
      shapes, finite values, sample counts, normalized coverage, and hard-label
      identity.

## Task 4: Implement the proposed Student runtime

**Files:**
- Create:
  `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/fastslow_runtime.py`
- Modify:
  `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/train_student_z.py`
- Modify:
  `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/play_student_phase2.py`

- [ ] Retain independent `slow_encoder` and `fast_encoder` networks for the
      proposed `split` Student.
- [ ] Add a real `monolithic` Student option that uses one history encoder for
      all five context dimensions. Save the architecture in every checkpoint.
- [ ] For the proposed split runtime, update both branches at 60 Hz for the
      first 3 s.
- [ ] After 3 s, update the slow network at 1 Hz and the fast network at 60 Hz.
- [ ] On a post-startup slow refresh, set `z_slow_target` to the new raw CNN
      output. At every policy step update the Actor-visible cache by
      `cache += alpha * (target - cache)`, where
      `alpha = 1-exp(-dt/0.25)`.
- [ ] During the first 3 s, use the current slow raw output directly, without
      filtering, so early payload identification is not delayed.
- [ ] Reset raw/target/cache/history independently for completed environments.
- [ ] Add an explicit runtime switch between `fast_slow` and `all_60hz`; this
      switch is for later scheduling ablation and is not used in the proposed
      Teacher run.

## Task 5: Add full closed-loop runtime and control audit

**Files:**
- Modify:
  `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/play_student_phase2.py`
- Modify:
  `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/analyze_phase2_csv.py`

- [ ] Add per-step CSV columns:
      `z_slow_raw[0:2]`, `z_slow_target[0:2]`,
      `z_slow_cache[0:2]`, slow/fast/full branch called flags and cumulative
      calls, slow/fast/full/Actor/end-to-end inference milliseconds, physical
      gust vector/event flag, consecutive CTBR action change, and
      counterfactual context-refresh action change.
- [ ] Count both batched forward invocations and processed environment samples
      for every context branch.
- [ ] Use CUDA synchronization around profiled regions; describe timings as
      evaluation/profiling timings, not training throughput.
- [ ] Save mean, P95, and P99 latency for each component in the summary JSON.
- [ ] Compute CTBR action total variation per channel and in total.
- [ ] Compute absolute and fractional one-sided action energy in 5--30 Hz.
- [ ] Define gust response latency operationally as time from a logged
      piecewise-gust change above the configured threshold to the first
      fast-context departure from its pre-event value above a configured
      threshold; report event/responded counts and latency distribution.
- [ ] At every slow refresh, evaluate the Actor counterfactually with the old
      cache and the new unfiltered target under the same observation. Log the
      resulting CTBR difference separately from the actually executed filtered
      action.

## Task 6: Update the V5 method and experiment record

**Files:**
- Modify:
  `/home/shenji/文档/WPS Cloud Files/308564339/zsj的第一篇文章payload Meta-learning轨迹追踪优化/论文撰写/uav-payload-paper-main/uav-payload-paper-main/intro+problem+method.V5_快慢结构.md`

- [ ] Insert a detailed implementation/experiment record before the current
      Introduction.
- [ ] Record exact environment ranges, observation/action dimensions, policy
      frequency, Teacher/Student architectures, startup/update/filter equations,
      required CSV fields, summary metrics, random seeds, checkpoint matching,
      paired-comparison rules, and acceptance gates.
- [ ] Replace the obsolete soft-anchored proposed Teacher description with the
      hard identity structural path. Keep `split_soft` and `monolithic` only as
      named future ablations.
- [ ] State evidence boundaries: call-count reduction is structural; latency,
      smoothness, and gust-response benefits require paired measurements.

## Task 7: Verification and handoff

**Files:**
- Test all files above.

- [ ] Run focused unit tests with the Isaac Lab Python launcher where required.
- [ ] Run `py_compile` for every modified Python file.
- [ ] Run CLI `--help`/import checks that do not require a full simulation.
- [ ] Verify the final diff does not alter unrelated user files.
- [ ] Give the user one absolute, copy-runnable Teacher smoke command and one
      20,000-iteration Teacher command using `--rma_context_mode split_hard`.
- [ ] Do not start the expensive Teacher training automatically.
