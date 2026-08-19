# Fast/Slow Student Performance Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify five publication-ready figures for Teacher-reference, All-60 Hz Student, and Fast/slow Student control, sway, gust-response, and compute comparisons.

**Architecture:** Add one focused plotting module beside the existing PPT package builder. The module selects clean constant-physics episodes, audits the strict Student pair, computes metrics from the selected data, and writes figures plus machine-readable evidence. Existing figures remain unchanged.

**Tech Stack:** Python 3, pandas, NumPy, matplotlib, pytest.

---

### Task 1: Clean-episode selection and strict-pair audit

**Files:**
- Create: `tests/test_fastslow_performance_figures.py`
- Create: `logs/rsl_rl/uav_payload_sim2real_rl/2026-07-24_06-06-18_hardexplicit_teacher_fastslow_seed42/PPT_快慢调度消融_整理/scripts/build_control_performance_figures.py`

- [ ] **Step 1: Write the failing episode-selection test**

```python
def test_select_longest_constant_physics_segment_uses_main_student_episode():
    module = _load_plot_module()
    frame = pd.read_csv(module.ALL60_CSV)
    selected, bounds = module.select_longest_constant_physics_segment(frame)
    assert bounds == (26, 2125)
    assert len(selected) == 2099
    assert selected["payload_mass_kg"].nunique() == 1
    assert selected["rope_length_m"].nunique() == 1
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 -m pytest -q tests/test_fastslow_performance_figures.py
```

Expected: FAIL because `build_control_performance_figures.py` does not exist.

- [ ] **Step 3: Implement segment selection and pair audit**

Implement:

```python
def select_longest_constant_physics_segment(frame):
    changed = (
        frame[["payload_mass_kg", "rope_length_m"]]
        .diff()
        .abs()
        .fillna(0.0)
        .gt(1e-12)
        .any(axis=1)
    )
    starts = [0, *np.flatnonzero(changed.to_numpy()).tolist()]
    stops = [*starts[1:], len(frame)]
    start, stop = max(zip(starts, stops), key=lambda pair: pair[1] - pair[0])
    selected = frame.iloc[start:stop].reset_index(drop=True).copy()
    selected["episode_time_s"] = np.arange(len(selected), dtype=float) / 60.0
    return selected, (start, stop)
```

Audit exact equality of goal, mass, rope length, and gust columns across the two
selected Student frames.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
python3 -m pytest -q tests/test_fastslow_performance_figures.py
```

Expected: PASS.

### Task 2: Metric computation

**Files:**
- Modify: `tests/test_fastslow_performance_figures.py`
- Modify: `logs/rsl_rl/uav_payload_sim2real_rl/2026-07-24_06-06-18_hardexplicit_teacher_fastslow_seed42/PPT_快慢调度消融_整理/scripts/build_control_performance_figures.py`

- [ ] **Step 1: Add failing metric tests**

Test that the computed record contains finite tracking RMSE, swing RMS/peak,
cumulative swing exposure, settling time, CTBR total variation, and 5–30 Hz
energy, and that both paired Student records use exactly 2099 samples.

- [ ] **Step 2: Run tests and verify RED**

Run the same focused pytest command. Expected: FAIL because
`compute_performance_metrics` is missing.

- [ ] **Step 3: Implement metric functions**

Compute:

```python
error_xyz = frame[["payload_err_x", "payload_err_y", "payload_err_z"]].to_numpy()
swing_xy = frame[["theta_x_deg", "theta_y_deg"]].to_numpy()
swing_mag = np.linalg.norm(swing_xy, axis=1)
```

Use RMS for position/swing, `max` for peak swing, trapezoidal integration for
cumulative absolute swing exposure, one continuous second below five degrees
for settling, adjacent L1 action changes for CTBR total variation, and an
FFT-based 5–30 Hz energy calculation.

- [ ] **Step 4: Run tests and verify GREEN**

Run the focused pytest command. Expected: PASS.

### Task 3: Generate task-performance and anti-sway figures

**Files:**
- Modify: `tests/test_fastslow_performance_figures.py`
- Modify: `logs/rsl_rl/uav_payload_sim2real_rl/2026-07-24_06-06-18_hardexplicit_teacher_fastslow_seed42/PPT_快慢调度消融_整理/scripts/build_control_performance_figures.py`

- [ ] **Step 1: Add failing figure-output test**

Invoke the builder in a temporary output directory and assert that figures
08–10 exist, are non-empty, and the metric CSV has Teacher, All-60 Hz Student,
and Fast/slow Student rows.

- [ ] **Step 2: Run tests and verify RED**

Expected: FAIL because the plotting functions are missing.

- [ ] **Step 3: Implement figures 08–10**

Use goal/zero as the reference curve, dashed black for the unpaired Teacher,
blue for All-60 Hz, and orange for Fast/slow. Plot all raw curves without
smoothing that would hide oscillation. Add the evidence-boundary note in each
figure footer.

- [ ] **Step 4: Run tests and verify GREEN**

Run the focused pytest command. Expected: PASS.

### Task 4: Generate event-aligned gust-response figure

**Files:**
- Modify: `tests/test_fastslow_performance_figures.py`
- Modify: `logs/rsl_rl/uav_payload_sim2real_rl/2026-07-24_06-06-18_hardexplicit_teacher_fastslow_seed42/PPT_快慢调度消融_整理/scripts/build_control_performance_figures.py`

- [ ] **Step 1: Add failing gust-event test**

Assert that post-warm-up events are detected from the identical paired gust
trace and that both Student matrices contain the same number of complete event
windows.

- [ ] **Step 2: Run tests and verify RED**

Expected: FAIL because event alignment is missing.

- [ ] **Step 3: Implement event alignment and figure 11**

Detect a gust event when adjacent gust-vector L2 change exceeds `0.05 m/s^2`.
Keep events after 3.0 s with complete `[-0.25, +0.50] s` windows. Plot median
and IQR for gust change, standardized fast-context change, CTBR L1 change, and
swing-magnitude change.

- [ ] **Step 4: Run tests and verify GREEN**

Run the focused pytest command. Expected: PASS.

### Task 5: Generate compute figure and evidence audit

**Files:**
- Modify: `tests/test_fastslow_performance_figures.py`
- Modify: `logs/rsl_rl/uav_payload_sim2real_rl/2026-07-24_06-06-18_hardexplicit_teacher_fastslow_seed42/PPT_快慢调度消融_整理/scripts/build_control_performance_figures.py`

- [ ] **Step 1: Add failing final-output test**

Assert that figure 12 and the audit JSON exist, that strict pairing is true,
that Teacher is marked unpaired, and that the event count is positive.

- [ ] **Step 2: Run tests and verify RED**

Expected: FAIL because compute plotting and audit writing are missing.

- [ ] **Step 3: Implement compute plotting and JSON audit**

Read the one-environment latency summaries, plot slow calls and mean/P95/P99
end-to-end latency in separate panels, and serialize episode bounds, pairing
checks, event count, and source paths.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
python3 -m pytest -q tests/test_fastslow_performance_figures.py tests/test_student_context_recovery_plot.py
```

Expected: all focused tests PASS.

### Task 6: Build and visually verify

**Files:**
- Generate: `PPT_快慢调度消融_整理/figures/08_位置XYZ整体性能.png`
- Generate: `PPT_快慢调度消融_整理/figures/09_消摆整体性能.png`
- Generate: `PPT_快慢调度消融_整理/figures/10_闭环性能指标汇总.png`
- Generate: `PPT_快慢调度消融_整理/figures/11_阵风快速响应严格对比.png`
- Generate: `PPT_快慢调度消融_整理/figures/12_计算开销严格对比.png`

- [ ] **Step 1: Run the builder**

```bash
python3 logs/rsl_rl/uav_payload_sim2real_rl/2026-07-24_06-06-18_hardexplicit_teacher_fastslow_seed42/PPT_快慢调度消融_整理/scripts/build_control_performance_figures.py
```

- [ ] **Step 2: Inspect all five rendered images**

Check readable labels, correct units, no duplicate Teacher, no clipped legends,
and visible disclosure that Teacher is unpaired.

- [ ] **Step 3: Run final verification**

```bash
python3 -m pytest -q tests/test_fastslow_performance_figures.py tests/test_student_context_recovery_plot.py
```

Expected: all focused tests PASS with no failures.
