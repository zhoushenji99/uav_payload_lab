# V8.9 Payload Observation Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align simulated Payload timing/rate observations with the deployed Jetson 21-D builder without changing mass, CTBR, reward, or battery settings.

**Architecture:** Add two pure tensor helpers in `real_hover_gap.py`: one composes a delayed body-relative Payload measurement with the current UAV pose, and one applies measurement-triggered angular-rate LPF. The environment stores body-relative Payload vectors in its latency ring, reconstructs policy-only world observations with the current UAV pose, and keeps critic/reward clean-state paths unchanged.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab DirectRLEnv, unittest, Git.

---

### Task 1: Freeze Payload composition and LPF semantics with failing tests

**Files:**
- Modify: `regression_tests/test_real_hover_gap.py`
- Test: `regression_tests/test_real_hover_gap.py`

- [ ] **Step 1: Write the failing composition tests**

Add tests that import `compose_delayed_payload_world_position` and verify current UAV translation and current UAV yaw both affect the reconstructed world Payload position:

```python
def test_delayed_relative_payload_is_composed_with_current_uav_pose(self):
    uav_pos = torch.tensor([[2.0, 3.0, 1.5]])
    yaw_90_wxyz = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]])
    delayed_relative_b = torch.tensor([[1.0, 0.0, -0.7]])
    actual = self.module.compose_delayed_payload_world_position(
        uav_pos, yaw_90_wxyz, delayed_relative_b
    )
    torch.testing.assert_close(actual, torch.tensor([[2.0, 4.0, 0.8]]), atol=1e-6, rtol=0.0)
```

- [ ] **Step 2: Write the failing LPF tests**

Add separate tests for first measurement, second valid measurement, and hold/dropout:

```python
actual = self.module.update_payload_rate_lpf(
    previous_filtered_rate=torch.tensor([[4.0, -2.0]]),
    raw_rate=torch.tensor([[10.0, 6.0]]),
    initialized=torch.tensor([True]),
    update=torch.tensor([True]),
    alpha=0.5,
)
torch.testing.assert_close(actual, torch.tensor([[7.0, 2.0]]))
```

First measurement must return zero; `update=False` must return the previous filtered value.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
~/IsaacLab/isaaclab.sh -p -m unittest \
  regression_tests.test_real_hover_gap.RealHoverGapHelperTests -v
```

Expected: new tests fail because the two helper functions do not exist.

### Task 2: Implement pure Payload observation helpers

**Files:**
- Modify: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/real_hover_gap.py`
- Test: `regression_tests/test_real_hover_gap.py`

- [ ] **Step 1: Implement current-pose composition**

Add a validated wxyz quaternion rotation and composition helper:

```python
def compose_delayed_payload_world_position(
    current_uav_pos_w: torch.Tensor,
    current_uav_quat_w: torch.Tensor,
    delayed_uav_to_payload_b: torch.Tensor,
) -> torch.Tensor:
    qw = current_uav_quat_w[..., 0:1]
    qv = current_uav_quat_w[..., 1:4]
    t = 2.0 * torch.cross(qv, delayed_uav_to_payload_b, dim=-1)
    relative_w = delayed_uav_to_payload_b + qw * t + torch.cross(qv, t, dim=-1)
    return current_uav_pos_w + relative_w
```

Validate matching `(..., 3)`, `(..., 4)`, `(..., 3)` shapes and finite values.

- [ ] **Step 2: Implement measurement-triggered LPF**

```python
def update_payload_rate_lpf(
    previous_filtered_rate: torch.Tensor,
    raw_rate: torch.Tensor,
    initialized: torch.Tensor,
    update: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    candidate = torch.where(
        initialized.unsqueeze(-1),
        alpha * previous_filtered_rate + (1.0 - alpha) * raw_rate,
        torch.zeros_like(raw_rate),
    )
    return torch.where(update.unsqueeze(-1), candidate, previous_filtered_rate)
```

Reject non-finite alpha or alpha outside `[0, 1]`, and reject mismatched tensor shapes.

- [ ] **Step 3: Run helper tests and verify GREEN**

Run the Task 1 command. Expected: all `RealHoverGapHelperTests` pass.

- [ ] **Step 4: Commit helper behavior**

```bash
git add regression_tests/test_real_hover_gap.py \
  source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/real_hover_gap.py
git commit -m "fix: align payload measurement helpers with Jetson"
```

### Task 3: Integrate delayed relative geometry and LPF into the environment

**Files:**
- Modify: `source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env.py`
- Modify: `regression_tests/test_real_hover_gap.py`

- [ ] **Step 1: Write failing static integration assertions**

Require the environment to import and use both helpers, allocate a three-dimensional relative-geometry ring, and pass `obs_theta_dot_lpf_alpha` to the LPF helper.

- [ ] **Step 2: Run the new static assertions and verify RED**

Run:

```bash
~/IsaacLab/isaaclab.sh -p -m unittest \
  regression_tests.test_real_hover_gap.RealHoverGapStaticIntegrationTests -v
```

Expected: new assertions fail on the old five-dimensional world-observation ring.

- [ ] **Step 3: Change the sensor ring to relative geometry**

Allocate `_payload_clean_ring` with sample dimension 3. Change `_transport_payload_observation` to accept current UAV position/quaternion, current UAV-to-Payload body vector, desired position, and rope length. Store the relative vector; after delayed selection, call `compose_delayed_payload_world_position`, then calculate measured `e_load` and measured tilt.

- [ ] **Step 4: Apply LPF only on valid visual updates**

Compute raw angular rate from consecutive held noisy angles and source timestamps, then call `update_payload_rate_lpf`. Keep first rate zero and preserve the previous rate when `update=False`.

- [ ] **Step 5: Keep clean and policy paths separated**

Move access to current root quaternion before sensor transport. Build the delayed relative body vector with `_quat_rotate_inverse(root_quat_w, p_load_w - p_uav_w)`. Continue using clean `e_load`, `tilt_deg`, and `w_deg` in `obs_critic` and reward terms.

- [ ] **Step 6: Run focused integration tests and verify GREEN**

Run the Task 2 command. Expected: all tests pass.

- [ ] **Step 7: Commit environment integration**

```bash
git add regression_tests/test_real_hover_gap.py \
  source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/meta_uav_env.py
git commit -m "fix: align simulated payload latency with real builder"
```

### Task 4: Correct operator documentation without changing experiment scope

**Files:**
- Modify: `/home/shenji/桌面/V8.9_代码与文档真机端审核包_2026-08-26/original_rectification_package/03_V8.9训练评估与导出命令.md`
- Modify: `/home/shenji/桌面/2026-08-26_V8.9仿真源码真机端审核.md`

- [ ] **Step 1: Correct Python launchers**

Keep lightweight `python3 -m json.tool` commands, but replace system-Python launches of Torch-dependent project scripts with `~/IsaacLab/isaaclab.sh -p`.

- [ ] **Step 2: Correct evaluation accounting and boundaries**

Write `5 seeds × 8 simulation scenarios + 1 fixed real replay`, state that `training_wind` is the command alias for contract scenario `training_wind_and_downwash`, document that 3.100 kg excludes the fixed gimbal half, and state battery efficiency is intentionally outside this one-minute-flight training revision.

- [ ] **Step 3: Verify documentation paths and commands**

Search for stale `5 seed × 9` wording and direct system-Python calls to Torch project scripts. Expected: none remain in the updated command document.

### Task 5: Full verification and audited delivery ZIP

**Files:**
- Create: `/home/shenji/桌面/V8.9_Payload观测语义对齐_代码与文档审核包_2026-08-26/`
- Create: `/home/shenji/桌面/V8.9_Payload观测语义对齐_代码与文档审核包_2026-08-26.zip`

- [ ] **Step 1: Run focused and full V8.9 regression gates**

Run the focused helper tests, the V8.9 test selection used by the prior package, syntax compilation for changed Python files, and `git diff --check`.

- [ ] **Step 2: Inspect the exact diff**

Confirm no changes to mass values, collective-efficiency range, CTBR limits, rewards, or unrelated dirty user files.

- [ ] **Step 3: Commit code/document changes**

Commit only files intentionally modified by this task. Do not add existing images, Paper files, or unrelated working-tree content.

- [ ] **Step 4: Build a self-auditing delivery directory**

Include README, design, plan, updated review, updated commands, changed source files, changed tests, commit/diff manifests, fresh test output, and per-file SHA256 checksums.

- [ ] **Step 5: Create and verify ZIP**

Create the ZIP, run `unzip -t`, independently verify package checksums, and report the absolute path and ZIP SHA256.
