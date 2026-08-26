# Phase-II Full-History Shadow Handover Design

## Goal

Evaluate an existing Student only after its `50 x 21` history and three-second fast/slow startup cache have been populated by a valid closed-loop trajectory.

## Scope

- Modify only the Phase-II rollout path and its focused regression tests.
- Keep the Teacher checkpoint, Student checkpoint, environment, rewards, dataset, and training code unchanged.
- Preserve legacy Student rollout behavior unless the new CLI option is explicitly enabled.

## Runtime behavior

When `--student_shadow_warmup_sec 3.0` is enabled in Student mode:

1. The Student encoder and Actor run from the first simulator step so history and fast/slow caches evolve normally.
2. During the first three seconds, the Teacher action is the action actually sent to the environment; the Student action remains a shadow candidate.
3. At the first step after the shadow interval, the environment receives the Student action.
4. History, slow context, fast context, filters, and episode counters are not reset at handover.
5. CSV output records the executed controller source and the shadow Student candidate so the handover discontinuity is auditable.

The shadow duration must be zero, or at least both the 50-frame history duration and the configured slow-context startup duration. This prevents a command that claims to test full-history handover while switching too early.

## Evidence boundary

Teacher shadow control is an evaluation proxy for a valid pre-handover controller. It verifies whether full-history startup fixes the current Student simulation failure; it does not prove that PX4 Position histories have the same distribution. Real deployment still requires a valid Position-mode `50 x 21` history with actual previous CTBR commands.

## Verification

- Unit tests cover shadow-mask boundaries, action selection, and invalid duration handling.
- Existing focused fast/slow regression tests must remain green.
- Python syntax compilation must pass.
- A one-environment, no-wind rollout will provide the behavioral A/B evidence after implementation.
