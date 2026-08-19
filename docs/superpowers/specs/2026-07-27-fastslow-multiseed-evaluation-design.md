# Fast/Slow Multi-Seed Evaluation Design

**Goal:** Preserve and evaluate only the July 24 hard-explicit fast/slow run using five paired evaluation scenarios that can later be replayed by the Coupled baseline.

## Source boundary

- Teacher checkpoint:
  `logs/rsl_rl/uav_payload_sim2real_rl/2026-07-24_06-06-18_hardexplicit_teacher_fastslow_seed42/model_19999.pt`
- Student checkpoint:
  `logs/rsl_rl/uav_payload_sim2real_rl/2026-07-24_06-06-18_hardexplicit_teacher_fastslow_seed42/StudentFastSlow_sim2real_hardexplicit_seed42_noprobe/best_fast_slow_student_encoder_z.pth`
- Output:
  `logs/rsl_rl/uav_payload_sim2real_rl/2026-07-24_06-06-18_hardexplicit_teacher_fastslow_seed42/快慢结构7.27数据整理`
- No WPS data, old Decoupled checkpoints, or other Teacher runs may enter this package.

## Evaluation-only overrides

`play_student_phase2.py` exposes:

- `--eval_payload_mass_kg`: fixes the payload mass used at reset.
- `--eval_rope_length_m`: fixes the rope length used at reset.
- `--eval_disable_wind`: disables all mean, gust, and OU wind during evaluation.

The fixed physical values are independent of `payload_mass_range` and
`rope_length_range`. Those training ranges remain unchanged so hard-explicit
normalization retains its original meaning.

All overrides are validated before environment creation. A fixed mass or rope
length outside its training range fails with a clear error.

## Scenario matrix

| Scenario | Seed | Wind | Payload mass | Rope length |
|---|---:|---|---|---|
| seed38_random | 38 | configured stochastic wind | randomized | randomized |
| seed40_random | 40 | configured stochastic wind | randomized | randomized |
| seed42_random | 42 | configured stochastic wind | randomized | randomized |
| seed44_random | 44 | configured stochastic wind | randomized | randomized |
| seed46_nowind_mid | 46 | disabled | 0.55 kg | 0.525 m |

Each scenario runs the July 24 Hard-explicit Teacher, the July 24 split Student
with both branches at 60 Hz, and the same Student with 3 s startup, 1 Hz slow
updates, 60 Hz fast updates, and a 0.25 s causal slow-context filter.

## Data contract

Each rollout has its own directory and contains:

- `rollout.csv`: aligned state, physical parameter, context, action, wind, update
  count, and inference timing samples.
- `summary.json`: checkpoint lineage, scenario overrides, update counts, latency
  summaries, action total variation, action high-frequency energy, and gust
  response.
- `console.log`: complete command output.

The package additionally contains:

- `experiment_manifest.json`: exact source paths, scenario definitions, and
  commands.
- `data/rollout_metrics.csv`: metrics for every rollout.
- `data/paired_schedule_metrics.csv`: All-60 Hz versus fast/slow differences.
- `data/aggregate_metrics.csv`: mean and standard deviation across the four
  random-wind seeds, with the fixed no-wind case reported separately.
- `data/data_audit.json`: lineage, required-column, fixed-scenario, and paired
  exogenous-input checks.
- `data/sha256_manifest.csv`: hashes of all raw and summary files.
- `figures/`: scenario audit, position, swing, context recovery, CTBR,
  computation, gust response, and aggregate performance figures.

## Pairing rule

Within each scenario, the three modes use the same seed, number of environments,
maximum steps, physics ranges, and wind configuration. All-60 Hz and fast/slow
must have identical exogenous columns over their common episode prefix. The
analysis marks a pair invalid rather than silently comparing it when this check
fails.

## Safety and reproducibility

- Evaluation overrides never change the default training configuration.
- No experiment is automatically retried after a crash.
- Existing files in the July 24 run are not overwritten.
- The no-wind audit requires the recorded total wind acceleration to be exactly
  zero and the recorded mass/rope length to equal 0.55 kg/0.525 m.
