# Real Hover Gap v1 Design

## 1. Scope and fixed decisions

This change prepares the existing hover Teacher/Student pipeline for a first real-flight deployment attempt while preserving the existing task and policy interface.

The following remain unchanged:

- task geometry and hover target;
- reward function;
- 21-dimensional policy observation layout;
- 5-dimensional privileged context layout;
- hard-explicit Teacher, dual Student, and fast/slow scheduling;
- PX4 CTBR action bounds;
- payload mass range `0.3-0.8 kg` and rope-length range `0.25-0.8 m`.

The loaded asset remains:

`source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/iris_payload_prismatic.usd`

The USD is not edited. Measured UAV properties are applied and audited at runtime.

## 2. Measured UAV properties

Assumption: `3.230 kg` is the UAV rigid-body mass including onboard equipment, but excluding the separately modelled rod and payload.

The old rigid body was `3.085 kg` with center of mass `[0.00407, 0.03059, 0.181] m`. Treating the added `0.145 kg` as a point mass at `[0, 0, 0.03] m` gives:

```python
uav_mass_kg = 3.230
uav_com_m = (0.00389, 0.02922, 0.17422)
uav_inertia_diag_kg_m2 = (0.0763, 0.0762, 0.1500)
```

`Izz=0.1500` is used instead of the raw `0.1600` estimate because the raw diagonal violates the rigid-body inertia triangle inequality. It remains within the measured double-pendulum interval.

At environment initialization, `meta_uav_env.py` will apply mass, COM, and inertia to the UAV body using PhysX setters, read them back, and fail fast on non-finite values, non-positive eigenvalues, or an inertia-triangle violation. Payload mass and inertia remain separate.

Training randomization around the nominal rigid-body values:

```text
UAV mass scale:      0.975-1.025
UAV COM offset:      x/y/z each within +/- 0.005 m
UAV inertia scale:   0.90-1.10
```

Payload inertia is scaled consistently when payload mass is randomized:

`I_payload,new = I_payload,default * m_new / m_default`.

## 3. Startup disturbance instead of direct state reset

No UAV root pose/velocity or payload swing-joint pose/velocity is directly randomized.

At the beginning of each episode, the existing wind path generates a smooth startup gust:

```text
profile:              half sine
acceleration:         0.5-1.5 m/s^2
duration:             0.4-1.0 s
UAV scale:            0.4
payload scale:        1.0
direction:            random horizontal direction
```

The startup gust is separate from the persistent mean, segmented gust, and OU wind states. All startup buffers are reset with indexed assignment so that no state crosses episode boundaries.

The previously proposed UAV and payload initial-state ranges become audit targets rather than reset distributions. The smoke run records the states reached at the end of the startup gust. Amplitude or duration is adjusted only if the resulting distribution misses or greatly exceeds the measured handover envelope.

## 4. Payload observation timing and bias

The 21-dimensional observation order and units do not change. Only payload-derived channels are passed through an event-driven sample-and-hold model.

Episode mixture:

```text
85% nominal sensor episodes: 12-30 Hz, delay 0.03-0.15 s
15% tail sensor episodes:     5-12 Hz, delay 0.15-0.30 s
valid update probability:     0.92-0.98
training hold duration cap:   0.50 s
evaluation-only stress hold:  0.75-1.75 s
```

Biases are sampled once per episode and remain constant:

```text
payload position bias: x/y/z within +/- 0.02 m
payload angle bias:    theta_x/theta_y within +/- 3 deg
attitude trim error:   roll/pitch within +/- 1 deg
linear velocity bias:  each axis within +/- 0.03 m/s
body-rate bias:        each axis within +/- 0.01 rad/s
```

Payload angle rate is recomputed from timestamped, held angle observations and then low-pass filtered. It is not independently randomized. During a dropped update, the last valid payload measurement is held and its age increases. No validity or age fields are appended to the policy input.

The UAV quaternion is expressed relative to the task yaw captured at handover, not as an arbitrary global-VIO yaw. Deployment code must use the same convention.

## 5. Persistent downwash

Ambient wind and rotor downwash are modelled separately:

- ambient wind remains a world-frame equivalent acceleration applied to the UAV and payload;
- downwash is a UAV-body-frame horizontal force applied only to the payload.

The first downwash model is:

`F_downwash_b(t) = F_bias_b + F_ou_b(t)`.

Configuration:

```text
episode bias magnitude: 0.0-0.8 N
OU sigma:               0.15 N/sqrt(s)
OU theta:               1.0 1/s
total force clip:       1.2 N
direction:              random in the UAV horizontal plane
```

The constant component represents the non-zero equilibrium angle produced by persistent rotor wake; the OU component represents correlated wake variation. No `0.626 Hz` sinusoid is injected. The `0.6004 m` suspension should generate its natural swing mode.

The Teacher's final three privileged values remain three-dimensional but represent the combined effective payload residual acceleration:

`a_residual_w = a_ambient_w + R_b_to_w F_downwash_b / m_payload`.

## 6. Provisional actuator gap

The existing CTBR mapping remains nominal. Until a fixed-rig or tethered CTBR identification is available, use conservative per-episode randomization:

```text
action delay:          0-2 policy steps
action LPF alpha:      0.35-1.0
collective efficiency: 0.85-1.05
moment efficiency:     0.90-1.10
```

No PWM polynomial or per-motor mixer is added in this revision because normalized PX4 CTBR thrust has not yet been mapped to executed per-motor PWM under load.

## 7. Files and data flow

### `meta_uav_env_cfg.py`

Add measured rigid-body properties and grouped `real_hover_gap_v1` configuration values.

### `meta_uav_env.py`

- apply and verify the measured UAV physics at runtime;
- recompute randomized payload inertia;
- generate/reset the startup gust;
- generate payload-only downwash;
- implement event-driven payload observation timing, hold, dropout, and bias;
- apply per-environment action delay, filtering, and effectiveness;
- expose audit statistics without changing policy observation dimensions.

### `collect_z_dataset.py`

Keep the tensor schema unchanged. Save the gap-profile name, effective sensor update statistics, startup-gust statistics, and domain configuration in collection metadata.

### `regression_tests/`

Add focused tests for:

- runtime mass/COM/inertia validity and payload separation;
- selected-environment reset isolation;
- smooth startup-gust start/end behavior;
- payload-only downwash force and residual-context conversion;
- sample-and-hold, dropout, timestamped angle-rate calculation, and recovery;
- per-environment delay/LPF state reset;
- unchanged 21-dimensional and 5-dimensional layouts.

## 8. Verification gates

1. Static and CPU regression tests pass.
2. One-environment visual smoke shows a smooth startup disturbance and no reset explosion.
3. A 4096-environment, three-iteration Teacher smoke has no NaN/Inf and saves checkpoints.
4. Audit output confirms the realized startup-state, sensor-rate, delay, dropout, and downwash distributions.
5. Only after the gates pass, start a fresh Teacher run; do not resume a pre-gap checkpoint.

