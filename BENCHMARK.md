# Benchmark: Symptom Radar vs. Oura / TemPredict

Honest comparison of this engine against the published performance of the
approach Oura's Symptom Radar descends from. **This is not a head-to-head
run of Oura's proprietary algorithm.** Oura's Symptom Radar scoring is
closed-source and cannot be executed on the wearer's data. This document
compares our *measured* numbers against the *published* numbers from the
peer-reviewed TemPredict study (Mason et al., Scientific Reports 12:3463,
2022), the academic foundation of both products, plus what Oura publicly
discloses about its approach.

## 1. The task

Both systems answer the same question each morning:
**"Is my physiology deviating from my personal baseline in the direction
of illness/strain?"**

| | Oura Symptom Radar | This engine |
|---|---|---|
| Signals | 40+ (temp, RHR, HRV, **respiratory rate**, sleep, ...) | 3 core (RHR, sleep HRV, temp deviation) + recovery |
| RR available? | Yes | **No.** The Ultrahuman API does not expose it (verified) |
| Method | proprietary (descended from TemPredict ensemble) | clean-baseline deviation + trajectory + persistence |
| Output | Normal / Minor / Major flags | Normal / Elevated / Significant strain |

## 2. Published numbers (the only objective yardstick)

TemPredict study operating points (training set, n=73):
- Sensitivity **82%**, Specificity **63%**, ROC AUC **0.819**
- Independent antibody-confirmed validation (n=10): **90% sens / 80% spec**
- Lead time: PX onset detected a mean of **2.75 days** before diagnostic test

Oura does not publish sensitivity/specificity for Symptom Radar itself;
these TemPredict numbers are the best public proxy.

## 3. This engine's measured numbers (one wearer, retrospective)

Measured with `evaluate.py` on a frozen copy of the wearer's database taken
2026-10-11: 169 days (2026-04-26 to 2026-10-11), scored-column sha256
`57804f2d…`. Each day is scored using only the data available up to and
including that morning.

> **The evaluation changed in 2026-10, and the earlier headline numbers are
> withdrawn.** Earlier versions of this file reported 73% sensitivity and
> 89% specificity. Both rested on the wrong ground truth: the positives were
> multi-day clusters found in the ring data itself, so the engine was graded
> against its own signal, and every day outside a cluster counted as healthy,
> including days with no data and days nobody labelled. `evaluate.py` now
> uses only independent ground truth:
>
> - **Illness:** days the wearer labelled `sick`, plus user-confirmed sick dates.
> - **Strain:** days labelled `rough` or `sick`.
> - **True negatives:** only days labelled `fine`.
> - **Unlabelled days:** reported as an alert rate, not a false-positive rate,
>   because nobody knows whether they were healthy.
> - **Unassessable days** (no data, baseline still calibrating, or fewer than
>   3 core metrics) are counted separately and never scored as Normal.
>
> Agreement with the ring-derived clusters is still printed, labelled as
> agreement, not accuracy.

| Measure | Result |
|---|---|
| Assessable days | 135 of 169 (10 no data, 13 calibrating, 11 insufficient data) |
| Flags (level ≥ 1) | 28 |
| Illness episodes caught | 1 of 1, flagged on its first day (no lead time) |
| Sick days flagged | 1 of 3 (strain index 1.52 on day 1, then 0.00 and 0.01) |
| Strain episodes caught | 4 of 4, each on its first or second day |
| Rough or sick days flagged | 5 of 8 |
| False positives on days labelled `fine` | 1 of 16 |
| Alerts on unlabelled days | 22 of 109 (20%) |
| Mean strain index on labelled days | fine 0.31, rough 1.45, sick 0.51 (assessable days, excluding days next to an episode) |

For direct comparison with TemPredict's **82% sens / 63% spec** (95%
Clopper-Pearson intervals in brackets):

| | TemPredict (published) | This engine (measured) |
|---|---|---|
| Sensitivity | 82% | Illness: 1 of 3 sick days, 33% [1% to 91%]. Strain: 5 of 8 days, 63% [25% to 92%] |
| Specificity | 63% | 15 of 16 days labelled fine, 94% [70% to 100%] |
| Alerts on unlabelled days | not reported | 22 of 109, 20% [13% to 29%] |
| Lead time | 2.75 days (RR-driven) | **None demonstrated.** The one illness episode was flagged on its first day, not before |

**Reading:** the sample cannot support an operating-point claim. One illness
episode, 3 sick days, 8 strain days and 16 fine days from one wearer: both
sensitivity intervals include TemPredict's 82%. The specificity interval
sits above TemPredict's 63%, but the 16 fine days are the days the wearer
happened to label, not a random sample, so it is not evidence that this
engine is more specific. The gap between 1 in 16 on labelled fine days and
1 in 5 on unlabelled days is unexplained: either unlabelled days hide
unreported rough nights, or fine labels went to the easiest days.

The earlier reading, that this engine "trades sensitivity for specificity",
is withdrawn with the 89% figure it rested on. What the data does show is
the engine's character: it reacts strongly to rough nights and only
briefly to the one illness.

## 4. Where Oura wins, honestly

1. **Respiratory rate.** RR is TemPredict's single strongest pre-symptomatic
   signal (basis of their PX alignment and much of the 2.75-day lead). Oura
   has it; we cannot get it from the Ultrahuman Partner API (verified: no
   respiratory fields documented; legacy endpoint returns `respiratory_graph:
   null`). This is a hard ceiling, not a tuning gap.
2. **Labelled training data.** Oura can learn from its whole user base (the
   size of the labelled set behind Symptom Radar is not published). Ours is
   a rules engine with 24 labelled days so far (16 fine, 5 rough, 3 sick).
3. **Signal breadth.** 40+ signals vs 3+recovery. Each extra orthogonal
   signal adds discriminative power we simply don't have.

## 5. Where this engine wins (structural, not claimed accuracy)

1. **Transparency.** Every flag is decomposable into per-metric deviations
   and contributions; the `symptom_radar_strain` MCP tool returns them as
   JSON. Oura's is a black box.
2. **No labelled data required.** It scores once 7 clean baseline days
   exist. Labels are optional and only feed evaluation and the separation
   classifier.
3. **It says when it cannot score.** A day with no data, a baseline still
   calibrating, or fewer than 3 core metrics is reported as exactly that,
   never as Normal.

## 6. The separation question (sickness vs. non-illness strain)

The unresolved piece. Sick and rough days push the same metrics the same
way (RHR up, HRV down, temperature up, recovery down). On this wearer's
labels the engine reacts more to rough days (mean strain index 1.45, 4 of 5
flagged) than to the one illness (0.51, 1 of 3 flagged). Oura does not
solve this either (it has no way to know what you did yesterday), but its
extra signals and training data make its *strain* flag more reliable. Our
path to the same place is the label collection loop:

- Labels are logged manually (`--label fine|rough|sick` or the MCP tools);
  the nightly check-in automation was retired in 2026-08.
- Confounders can be logged alongside (`--confounders "alcohol=2,travel=true"`;
  `alcohol=0` records an alcohol-free day), and `evaluate.py` reports flag
  rates with and without each one. None are recorded yet.
- `train.py` gates at 15 labelled days **and** 5 independent episodes per
  class, then validates with leave-one-episode-out: whole episodes are held
  out and feature normalisation is fitted inside each fold. Today there is
  1 independent sick episode and 3 rough ones, so separation cannot be
  validated at all.
- `scenario.py` previously reported 0.50 to 0.67 recall and 0.50 to 0.60
  precision if the strong unconfirmed episodes were sickness. **Those
  figures are withdrawn.** They came from day-level leave-one-out (days of
  the same episode on both sides of the split), normalisation fitted on all
  days before splitting, and a next-day recovery feature that does not
  exist on the morning of an alert. Under episode-level validation none of
  its hypotheses can be validated with the current labels.

## 7. Extra signals (ablation, exploratory)

`ablate.py` adds one stored signal at a time at a fixed weight of 0.10
(chosen before evaluation, not fitted to labels) and re-runs the
evaluation. Default scoring is unchanged.

| Signal | Coverage | Flags added | Flags removed | Alerts on unlabelled days |
|---|---|---|---|---|
| Sleep score | 85% | 5 | 0 | 22 → 26 of 109 |
| SpO2 | 81% | 5 | 0 | 22 → 26 of 109 |
| Morning alertness | 29% | 1 | 0 | 22 → 23 of 109 |
| HR drop | 0% | skipped: not populated by the API | | |

Sleep score and SpO2 each moved the illness flag 2 days earlier, which
looks like lead time. It is not evidence of it: every flag they added was
on a day already at strain index 0.90 to 0.98, just under the 1.0
threshold, so any small extra weight would have tipped it. One episode
cannot tell the two apart. Neither signal is in the default.

## 8. Verdict

- **Strain detection: works on the labelled episodes.** All 4 rough or sick
  episodes were flagged on their first or second day. The cost is an alert
  on about 1 in 5 unlabelled days.
- **Illness detection: one episode, caught on its first day only.** Days 2
  and 3 scored near zero. One episode cannot support a sensitivity figure.
- **Pre-symptomatic lead time: cannot beat Oura.** That capability is
  RR-gated, and RR is not obtainable from the Ultrahuman API. Claiming
  otherwise would be dishonest.
- **Sickness vs. strain: not separable yet.** Validation needs 5 independent
  episodes of each; there are 1 sick and 3 rough.
