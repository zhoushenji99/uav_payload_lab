# Fast/Slow Student Performance Figure Design

## Goal

Generate a reproducible figure package that compares the common Hard-explicit
Teacher reference, the split Student evaluated at 60 Hz on both branches, and
the same split Student evaluated with the deployed fast/slow schedule.

## Evidence boundary

- `All-60 Hz Student` and `Fast/slow Student` are the strict runtime ablation:
  they share checkpoint, Student weights, seed, goals, physical parameters, and
  gust sequence.
- The available Teacher rollout has a different physical/gust realization. It
  is shown only as an unpaired performance reference and is excluded from
  causal percentage comparisons.
- The schedule ablation can show compute reduction with preserved control and
  fast response. It cannot show lower Student training loss because both
  runtime modes use the same trained Student.

## Inputs

- `teacher_audit/phase2_teacher_hardexplicit_seed42_longest_episode.csv`
- `StudentFastSlow_sim2real_hardexplicit_seed42_noprobe/phase2_student_all60hz_seed42.csv`
- `StudentFastSlow_sim2real_hardexplicit_seed42_noprobe/phase2_student_fastslow_tau025_seed42.csv`
- The corresponding one-environment latency summary JSON files.

## Episode handling

The Student trace contains a short initial fragment, one long clean episode,
and a final fragment. Select the longest contiguous interval over which payload
mass and rope length are constant. Apply the same selected row interval to both
strictly paired Student traces and verify that goals, physical parameters, and
gust inputs are exactly equal.

## Figures

1. `08_位置XYZ整体性能.png`: payload x/y/z with goal, unpaired Teacher
   reference, All-60 Hz Student, and Fast/slow Student.
2. `09_消摆整体性能.png`: theta-x, theta-y, swing magnitude, and cumulative
   absolute swing exposure. The zero-swing reference is the fourth curve.
3. `10_闭环性能指标汇总.png`: grouped summaries of tracking, swing, settling,
   and CTBR continuity metrics. Only the two Student bars receive paired
   percentage annotations.
4. `11_阵风快速响应严格对比.png`: event-aligned gust, fast-context, CTBR, and
   swing responses for the two strictly paired Students, using median and IQR
   over detected post-warm-up gust events.
5. `12_计算开销严格对比.png`: slow-encoder calls and mean/P95/P99 end-to-end
   latency from the one-environment profiling records.

## Outputs

- Five PNG figures at 300 DPI under `PPT_快慢调度消融_整理/figures/`.
- `data/07_整体性能指标对比.csv` with the exact plotted scalar metrics.
- `data/08_整体性能数据审计.json` recording episode bounds, pairing checks,
  event count, source paths, and the Teacher reference limitation.
- A standalone plotting script under `PPT_快慢调度消融_整理/scripts/`.

## Visual rules

- Use colorblind-safe colors and line styles.
- Every axis includes units.
- Do not use a duplicated Teacher curve to manufacture four methods.
- State directly in titles/captions that Teacher is an unpaired reference.
- Preserve raw findings even if Fast/slow and All-60 Hz overlap almost exactly.
