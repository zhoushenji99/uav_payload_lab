# Real Hover Gap v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the measured UAV rigid-body properties, startup gust, payload-only downwash, real payload-vision timing, observation bias, and conservative actuator gaps without changing the existing 21D/5D RMA interface or editing the USD.

**Architecture:** Keep `meta_uav_env.py` as the Isaac Lab integration point and add a small pure-Torch module for deterministic, CPU-testable gap calculations. Episode parameters are sampled into per-environment buffers and reset only through indexed assignment. Clean signals continue to drive rewards and privileged labels; the policy receives held, delayed, biased observations.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab/PhysX tensor API, `unittest`, existing RSL-RL scripts.

---

## File map

- Create `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/real_hover_gap.py`: pure tensor validation, startup profile, inertia construction, and delay selection helpers.
- Modify `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env_cfg.py`: all `real_hover_gap_v1` constants.
- Modify `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env.py`: PhysX property application, per-environment gap buffers, reset logic, force application, observation transport, and action transport.
- Modify `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/collect_z_dataset.py`: persist the gap profile and realized audit statistics without changing tensor schemas.
- Create `regression_tests/test_real_hover_gap.py`: pure CPU regression tests.
- Modify `regression_tests/test_sim2real_wind_reset.py`: startup/downwash reset-isolation coverage.

### Task 1: Add pure gap helpers and red-green CPU tests

**Files:**
- Create: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/real_hover_gap.py`
- Create: `regression_tests/test_real_hover_gap.py`

- [ ] **Step 1: Write failing tests for inertia validation, startup profile, and per-row delayed action selection**

```python
class TestRealHoverGapHelpers(unittest.TestCase):
    def test_inertia_triangle_is_enforced(self):
        validate_inertia_diagonal((0.0763, 0.0762, 0.1500))
        with self.assertRaisesRegex(ValueError, "triangle"):
            validate_inertia_diagonal((0.073, 0.073, 0.160))

    def test_half_sine_is_zero_at_endpoints_and_one_at_midpoint(self):
        elapsed = torch.tensor([0.0, 0.5, 1.0, 1.5])
        duration = torch.ones(4)
        torch.testing.assert_close(
            half_sine_profile(elapsed, duration),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_select_delayed_actions_uses_each_environment_delay(self):
        queue = torch.tensor([
            [[0.0], [1.0], [2.0]],
            [[10.0], [11.0], [12.0]],
            [[20.0], [21.0], [22.0]],
        ])
        out = select_delayed_actions(queue, torch.tensor([0, 1, 2]))
        torch.testing.assert_close(out[:, 0], torch.tensor([2.0, 11.0, 20.0]))
```

- [ ] **Step 2: Run the focused test and verify the import/function failures**

Run:

```bash
python3 -m unittest regression_tests.test_real_hover_gap -v
```

Expected: FAIL because `real_hover_gap.py` and its functions do not exist.

- [ ] **Step 3: Implement the pure helpers**

```python
def validate_inertia_diagonal(values):
    diag = torch.as_tensor(values, dtype=torch.float64)
    if diag.shape != (3,) or not torch.isfinite(diag).all() or torch.any(diag <= 0):
        raise ValueError("inertia diagonal must contain three finite positive values")
    if diag[0] + diag[1] < diag[2] or diag[0] + diag[2] < diag[1] or diag[1] + diag[2] < diag[0]:
        raise ValueError("inertia diagonal violates rigid-body triangle inequality")
    return diag


def diagonal_inertia_flat(values, *, device="cpu"):
    diag = validate_inertia_diagonal(values).to(device=device, dtype=torch.float32)
    matrix = torch.diag(diag)
    return matrix.transpose(0, 1).reshape(9)


def half_sine_profile(elapsed_s, duration_s):
    safe_duration = duration_s.clamp_min(1e-6)
    phase = math.pi * elapsed_s / safe_duration
    active = (elapsed_s >= 0.0) & (elapsed_s < duration_s)
    return torch.where(active, torch.sin(phase), torch.zeros_like(elapsed_s))


def select_delayed_actions(queue, delay_steps):
    max_delay = queue.shape[1] - 1
    indices = (max_delay - delay_steps).clamp(0, max_delay)
    rows = torch.arange(queue.shape[0], device=queue.device)
    return queue[rows, indices]
```

- [ ] **Step 4: Run tests and verify PASS**

Run the same command. Expected: all helper tests PASS.

- [ ] **Step 5: Commit the helper boundary**

```bash
git add regression_tests/test_real_hover_gap.py source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/real_hover_gap.py
git commit -m "test: add real hover gap helpers"
```

### Task 2: Add configuration and runtime rigid-body physics

**Files:**
- Modify: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env_cfg.py:130-290`
- Modify: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env.py:63-120,639-712`
- Test: `regression_tests/test_real_hover_gap.py`

- [ ] **Step 1: Add failing static-source assertions for the agreed constants and runtime setters**

```python
def test_config_keeps_interface_and_encodes_measured_uav(self):
    cfg = CFG_PATH.read_text(encoding="utf-8")
    self.assertIn('real_hover_gap_profile = "real_hover_gap_v1"', cfg)
    self.assertIn("uav_mass_kg = 3.230", cfg)
    self.assertIn("uav_com_m = (0.00389, 0.02922, 0.17422)", cfg)
    self.assertIn("uav_inertia_diag_kg_m2 = (0.0763, 0.0762, 0.1500)", cfg)
    self.assertIn("proprio_obs_dim = 21", cfg)
    self.assertIn("privileged_obs_dim = 5", cfg)


def test_env_uses_runtime_physx_overrides_without_editing_usd(self):
    source = ENV_PATH.read_text(encoding="utf-8")
    self.assertIn("set_masses", source)
    self.assertIn("set_coms", source)
    self.assertIn("set_inertias", source)
    self.assertIn("_apply_uav_physics", source)
```

- [ ] **Step 2: Run tests and verify FAIL on missing constants/method**

Run:

```bash
python3 -m unittest regression_tests.test_real_hover_gap -v
```

- [ ] **Step 3: Add grouped configuration values**

Add the exact nominal and randomization values from the approved spec:

```python
real_hover_gap_profile = "real_hover_gap_v1"
enable_real_hover_gap = True
uav_mass_kg = 3.230
uav_com_m = (0.00389, 0.02922, 0.17422)
uav_inertia_diag_kg_m2 = (0.0763, 0.0762, 0.1500)
uav_mass_scale_range = (0.975, 1.025)
uav_com_offset_range_m = (-0.005, 0.005)
uav_inertia_scale_range = (0.90, 1.10)
```

- [ ] **Step 4: Apply nominal physics before caching defaults**

Add `_apply_uav_physics(env_ids, randomize)` that clones complete PhysX arrays, modifies only `self._body_id[0]`, keeps the COM principal-axis quaternion unchanged, writes column-major diagonal inertias, then calls:

```python
self._robot.root_physx_view.set_masses(masses, env_ids_cpu)
self._robot.root_physx_view.set_coms(coms, env_ids_cpu)
self._robot.root_physx_view.set_inertias(inertias, env_ids_cpu)
```

Call it for all environments before `_default_masses_cpu`, `_default_inertias_cpu`, and the new `_default_coms_cpu` are cached. At episode reset, resample only selected rows from nominal values, not from the previous randomized values.

- [ ] **Step 5: Scale payload inertia with payload mass**

After setting `new_mass`, compute:

```python
ratio = new_mass / self._default_payload_mass_cpu.clamp_min(1e-6)
inertias[env_ids_cpu, self._payload_id] = (
    self._default_inertias_cpu[env_ids_cpu, self._payload_id] * ratio[:, None]
)
self._robot.root_physx_view.set_inertias(inertias, env_ids_cpu)
```

Read back selected mass/COM/inertia rows and raise a `RuntimeError` if values are not finite or the UAV diagonal fails validation.

- [ ] **Step 6: Run tests and a one-environment readback smoke**

CPU tests must PASS. Then run one headless environment for two steps and require the startup log to print mass `3.230` before random scaling and a valid inertia.

- [ ] **Step 7: Commit runtime physics**

```bash
git add regression_tests/test_real_hover_gap.py source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env_cfg.py source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env.py
git commit -m "feat: apply measured hover rigid-body physics"
```

### Task 3: Add startup gust and payload-only downwash

**Files:**
- Modify: `meta_uav_env_cfg.py:177-200`
- Modify: `meta_uav_env.py:232-284,827-1026`
- Modify: `regression_tests/test_sim2real_wind_reset.py`
- Test: `regression_tests/test_real_hover_gap.py`

- [ ] **Step 1: Add failing tests for profile endpoints, selected-row reset isolation, and residual acceleration**

Tests must assert:

```python
self.assertEqual(cfg.startup_gust_accel_range_mps2, (0.5, 1.5))
self.assertEqual(cfg.startup_gust_duration_range_s, (0.4, 1.0))
self.assertEqual(cfg.downwash_bias_force_range_n, (0.0, 0.8))

# reset selected envs only
torch.testing.assert_close(state._startup_elapsed_s[env_ids], torch.zeros(len(env_ids)))
torch.testing.assert_close(state._downwash_ou_b[env_ids], torch.zeros(len(env_ids), 3))
torch.testing.assert_close(state._startup_elapsed_s[untouched], elapsed_before)

# force-to-acceleration conversion
residual = ambient + rotate(downwash_b) / payload_mass[:, None]
```

- [ ] **Step 2: Run focused tests and verify FAIL**

```bash
python3 -m unittest regression_tests.test_real_hover_gap regression_tests.test_sim2real_wind_reset -v
```

- [ ] **Step 3: Add exact startup/downwash configuration**

```python
enable_startup_gust = True
startup_gust_accel_range_mps2 = (0.5, 1.5)
startup_gust_duration_range_s = (0.4, 1.0)
startup_gust_uav_scale = 0.4
startup_gust_payload_scale = 1.0
enable_payload_downwash = True
downwash_bias_force_range_n = (0.0, 0.8)
downwash_ou_sigma_n_sqrt_s = 0.15
downwash_ou_theta = 1.0
downwash_force_clip_n = 1.2
```

- [ ] **Step 4: Allocate and reset independent buffers**

Allocate direction, amplitude, duration, elapsed, acceleration, bias-force, OU-force, and total-force buffers. `_reset_wind(env_ids)` must sample startup and downwash episode parameters while clearing only selected rows with direct indexed assignment.

- [ ] **Step 5: Update forces without changing the old wind model**

In `_wind_step(dt)`, compute startup acceleration from `half_sine_profile` and add it before the existing wind clamp. In `_apply_action()`, rotate UAV-body downwash to world and then into the payload body before adding it only to the payload force slot. Do not apply downwash to the UAV body.

For privileged context, replace raw ambient wind with:

```python
downwash_w = self._quat_rotate(root_quat_w, self._downwash_force_b)
residual_w = self._wind_acc_w + downwash_w / self._payload_mass.clamp_min(1e-6).unsqueeze(-1)
residual_b = self._quat_rotate_inverse(root_quat_w, residual_w)
wind_norm = residual_b / max(float(self.cfg.residual_accel_norm_max), 1e-6)
```

Set `residual_accel_norm_max = 5.5` so the worst configured `1.2 N / 0.3 kg + 1.5 m/s^2` remains within the nominal normalized range.

- [ ] **Step 6: Run focused tests and verify PASS**

Run the Step 2 command. Expected: all tests PASS on CPU and CUDA when CUDA is available.

- [ ] **Step 7: Commit disturbance model**

```bash
git add regression_tests/test_real_hover_gap.py regression_tests/test_sim2real_wind_reset.py source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env_cfg.py source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env.py
git commit -m "feat: model startup gust and payload downwash"
```

### Task 4: Add event-driven payload observations and per-episode bias

**Files:**
- Modify: `meta_uav_env_cfg.py:264-283`
- Modify: `meta_uav_env.py:120-133,286-466,731-743`
- Test: `regression_tests/test_real_hover_gap.py`

- [ ] **Step 1: Write failing pure-state tests for update, hold, dropout, and timestamped angle rate**

Use a three-environment fixture where one row updates, one drops, and one is not due. Assert:

```python
torch.testing.assert_close(held[0], delayed_clean[0] + bias[0])
torch.testing.assert_close(held[1], held_before[1])
torch.testing.assert_close(held[2], held_before[2])
torch.testing.assert_close(theta_dot[0], (new_theta[0] - old_theta[0]) / source_dt[0])
torch.testing.assert_close(theta_dot[1:], theta_dot_before[1:])
```

- [ ] **Step 2: Run the test and verify FAIL**

```bash
python3 -m unittest regression_tests.test_real_hover_gap -v
```

- [ ] **Step 3: Add the approved sensor mixture and bias values**

```python
enable_payload_sensor_gap = True
payload_sensor_tail_probability = 0.15
payload_sensor_nominal_hz = (12.0, 30.0)
payload_sensor_tail_hz = (5.0, 12.0)
payload_sensor_nominal_delay_s = (0.03, 0.15)
payload_sensor_tail_delay_s = (0.15, 0.30)
payload_sensor_valid_probability = (0.92, 0.98)
payload_sensor_hold_cap_s = 0.50
payload_position_bias_range_m = (-0.02, 0.02)
payload_angle_bias_range_deg = (-3.0, 3.0)
attitude_trim_bias_range_deg = (-1.0, 1.0)
linear_velocity_bias_range_mps = (-0.03, 0.03)
body_rate_bias_range_rps = (-0.01, 0.01)
```

- [ ] **Step 4: Allocate a bounded clean payload ring and held-output state**

Use `ceil(0.30 / step_dt) + 2` ring entries for `[e_load(3), tilt(2)]`. Maintain per environment: write index, source step, update period, delay steps, next update time, validity probability, held sample, held angle rate, last source time, age, valid-update count, dropout count, and constant episode bias.

- [ ] **Step 5: Replace only policy payload channels with transported values**

Reward and critic remain clean. On a due valid update, read the delayed clean ring entry, add episode bias and per-update white noise, compute angle rate from source timestamps, and update the held sample. On dropout or between events, hold channels 0-6. UAV quaternion/velocity/body-rate retain 60 Hz updates with their episode bias and existing white noise.

Capture a task-yaw reference at reset and output `q_task_body = conjugate(q_task_yaw) * q_world_body`. Add quaternion multiplication and yaw-only reference helpers using wxyz internally; convert no existing tensor ordering.

- [ ] **Step 6: Reset all selected sensor rows and prove no state leakage**

Reset held values, timers, ring contents, timestamps, counters, and biases with indexed assignment. Extend tests to preserve untouched rows.

- [ ] **Step 7: Run tests and verify 21D/5D unchanged**

Run focused tests and `regression_tests.test_z_dataset_audit`. Expected: all PASS; observation sizes remain 21 and 26 with privilege.

- [ ] **Step 8: Commit observation transport**

```bash
git add regression_tests/test_real_hover_gap.py source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env_cfg.py source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env.py
git commit -m "feat: model payload vision timing and bias"
```

### Task 5: Add per-environment provisional action transport

**Files:**
- Modify: `meta_uav_env_cfg.py:276-283`
- Modify: `meta_uav_env.py:135-162,193-230,755-775`
- Test: `regression_tests/test_real_hover_gap.py`

- [ ] **Step 1: Write failing tests for delay, alpha, efficiency, and selected reset**

Tests must cover delay steps 0/1/2 in the same batch and assert that reset fills all queue slots with that environment's hover command. Also assert alpha and efficiency stay within configured ranges.

- [ ] **Step 2: Run tests and verify FAIL**

- [ ] **Step 3: Add exact actuator ranges**

```python
action_delay_steps_range = (0, 2)
action_lpf_alpha_range = (0.35, 1.0)
collective_efficiency_range = (0.85, 1.05)
moment_efficiency_range = (0.90, 1.10)
```

- [ ] **Step 4: Replace the scalar queue with a max-delay queue and per-row gather**

Allocate `max_delay + 1` slots. Each step roll, append current raw action, and call `select_delayed_actions(self._action_queue, self._action_delay_steps_per_env)`. Apply per-row alpha with shape `(N,1)`.

Apply collective efficiency only to the physical thrust after the action remains clipped to PX4 bounds. Apply moment efficiency to the rate-loop moment before final moment clipping. Do not change reward inputs or policy raw-action logging.

- [ ] **Step 5: Reset selected transport state**

Sample delay/alpha/efficiencies once per episode and fill the complete selected queue with the selected hover action. Preserve untouched rows.

- [ ] **Step 6: Run focused tests and verify PASS**

- [ ] **Step 7: Commit actuator transport**

```bash
git add regression_tests/test_real_hover_gap.py source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env_cfg.py source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env.py
git commit -m "feat: randomize hover action transport"
```

### Task 6: Persist audit metadata and run integration gates

**Files:**
- Modify: `collect_z_dataset.py:463-533`
- Modify: `meta_uav_env.py:809-824`
- Test: `regression_tests/test_real_hover_gap.py`

- [ ] **Step 1: Add failing metadata assertions**

Require `meta.pt` and `collect_report.json` construction to include:

```python
"real_hover_gap_profile"
"uav_mass_kg"
"payload_sensor_nominal_hz"
"payload_sensor_tail_hz"
"startup_gust_accel_range_mps2"
"downwash_bias_force_range_n"
"action_delay_steps_range"
```

- [ ] **Step 2: Run tests and verify FAIL**

- [ ] **Step 3: Add a serializable config snapshot and realized counters**

Expose `env.unwrapped.get_real_hover_gap_audit()` returning only Python scalars/lists. Save the configured ranges plus mean realized sensor Hz, delay, update/dropout counts, startup amplitude/duration, downwash magnitude, action delay, alpha, and efficiencies. Change the privileged layout description from `wind_norm(3)` to `residual_accel_norm(3)`.

- [ ] **Step 4: Run the focused regression suite**

```bash
python3 -m unittest \
  regression_tests.test_real_hover_gap \
  regression_tests.test_sim2real_wind_reset \
  regression_tests.test_z_dataset_audit \
  regression_tests.test_rma_context_modes \
  regression_tests.test_student_context_modes -v
```

Expected: PASS with no unrelated legacy test discovery.

- [ ] **Step 5: Run syntax/static checks**

```bash
python3 -m py_compile \
  source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/real_hover_gap.py \
  source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env_cfg.py \
  source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env.py \
  source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/collect_z_dataset.py
git diff --check
```

Expected: no output except successful test summaries.

- [ ] **Step 6: Run one-environment simulation smoke**

Run a short headless Teacher play or training smoke with one environment and assert:

- runtime mass/COM/inertia readback is valid;
- startup profile starts and returns to zero;
- downwash affects only the payload;
- observations and actions contain no NaN/Inf;
- reward and episode state stay finite.

- [ ] **Step 7: Run the three-iteration 4096-environment Teacher smoke**

```bash
cd /home/shenji/uav_payload_lab/uav_payload_lab

~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task=Isaac-Uav-Sim2Real-v0 \
  --seed 42 \
  --num_envs 4096 \
  --max_iterations 3 \
  --headless \
  agent.experiment_name=uav_payload_sim2real_hover_real_gap_v1 \
  agent.run_name=real_hover_gap_v1_teacher_smoke_seed42
```

Expected: three iterations complete, finite reward/context/action diagnostics, and checkpoints are written. Do not start the long run automatically.

- [ ] **Step 8: Commit final metadata and verification support**

```bash
git add regression_tests/test_real_hover_gap.py source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/collect_z_dataset.py source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env.py
git commit -m "feat: audit real hover gap profile"
```

