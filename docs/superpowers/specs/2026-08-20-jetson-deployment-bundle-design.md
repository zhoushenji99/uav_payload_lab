# Jetson Deployment Bundle Design

## Goal

Export the validated hover Student and Actor into a portable, Isaac-Sim-free
deployment bundle for NVIDIA Jetson, with numerical parity evidence against the
original PyTorch checkpoints.

## Inputs

- Teacher policy checkpoint: `model_3000.pt`.
- Student checkpoint: `best_fast_slow_student_encoder_z.pth` (best epoch 473).
- Policy observation: 21 proprioceptive values plus 5 Student context values.
- Student history: `[batch, 50, 21]`.

## Bundle architecture

The bundle contains three independent inference graphs:

1. `slow_encoder`: `[B, 50, 21] -> [B, 2]`.
2. `fast_encoder`: `[B, 50, 21] -> [B, 3]`.
3. `actor`: `[B, 26] -> [B, 4]`, including the trained empirical observation
   normalization before the Actor MLP.

Each graph is exported as ONNX for TensorRT and TorchScript for reference and
fallback execution. The online history buffer, 3 s startup schedule, 1 Hz slow
updates, 60 Hz fast updates, 0.25 s slow-context filter, concatenation, and CTBR
clamp remain explicit host-side logic because they are stateful and run at
different rates.

## Runtime contract

The 21-dimensional input order is fixed:

1. payload position error in world coordinates, metres: 3 values;
2. payload swing angle, degrees: 2 values;
3. payload swing angular rate, degrees per second: 2 values;
4. UAV world quaternion in Isaac ordering: 4 values;
5. UAV body-frame linear velocity, metres per second: 3 values;
6. UAV body-frame angular velocity, radians per second: 3 values;
7. previous executed CTBR command: 4 values.

The raw Actor output order is PX4 CTBR
`[thrust_body_z, roll_rate, pitch_rate, yaw_rate]`. Deployment clamps it to
`[-1, 0]`, `[-2.5, 2.5]`, `[-2.5, 2.5]`, and `[-1.5, 1.5]` respectively.

## Safety boundary

The exported reference runtime reproduces the simulator's zero-padded history
and fast/slow schedule exactly. It does not claim real-flight safety. The
manifest explicitly records the first-50-frame startup limitation and the need
for a separately validated flight startup guard before CTBR authority is handed
to the network.

## Verification

- Validate checkpoint lineage, dimensions, context mode, and seed.
- Run deterministic, physically valid history samples from the collected hover
  dataset through PyTorch, TorchScript, and ONNX.
- Require maximum absolute differences within declared tolerances.
- Run a stateful reference-runtime parity test for history update, scheduling,
  filtering, concatenation, normalization, and action clamping.
- Validate each ONNX model with `onnx.checker`.
- Write `parity_report.json` and SHA256 hashes for every deployment artifact.

## Outputs

The generated run-specific directory is
`StudentFastSlow_hover_model3000_seed42_noprobe/jetson_deployment_bundle/` and
contains model files, manifest, parity report, checksums, source checkpoints,
and an Isaac-Sim-free reference runtime.
