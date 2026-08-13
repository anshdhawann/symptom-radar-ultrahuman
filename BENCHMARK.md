# Benchmark: Symptom Radar vs. Oura / TemPredict

Honest comparison of this engine against the published performance of the
approach Oura's Symptom Radar descends from. **This is not a head-to-head
run of Oura's proprietary algorithm** — Oura's Symptom Radar scoring is
closed-source and cannot be executed on the wearer's data. This document
compares our *measured* numbers against the *published* numbers from the
peer-reviewed TemPredict study (Mason et al., Scientific Reports 12:3463,
2022) — the academic foundation of both products — plus what Oura publicly
discloses about its approach.

## 1. The task

Both systems answer the same question each morning:
**"Is my physiology deviating from my personal baseline in the direction
of illness/strain?"**

| | Oura Symptom Radar | This engine |
|---|---|---|
| Signals | 40+ (temp, RHR, HRV, **respiratory rate**, sleep, ...) | 3 core (RHR, sleep HRV, temp deviation) + recovery |
| RR available? | Yes | **No — Ultrahuman API doesn't expose it** (verified) |
| Method | proprietary (descended from TemPredict ensemble) | clean-baseline z + trajectory + persistence |
| Output | Normal / Minor / Major flags | Normal / Elevated / Significant strain |

## 2. Published numbers (the only objective yardstick)

TemPredict study operating points (training set, n=73):
- Sensitivity **82%**, Specificity **63%**, ROC AUC **0.819**
- Independent antibody-confirmed validation (n=10): **90% sens / 80% spec**
- Lead time: PX onset detected a mean of **2.75 days** before diagnostic test

Oura does not publish sensitivity/specificity for Symptom Radar itself;
these TemPredict numbers are the best public proxy.

## 3. This engine's measured numbers (retrospective evaluation dataset)

From `evaluate.py` (no-lookahead scoring, corrected episode labels):

| Metric | Value |
|---|---|
| Flags (level ≥ 1) | 14 |
| True positives (episode days) | 11 |
| False positives | 3 (4% FPR over 78 healthy days) |
| Episode-day recall | 11/16 = 0.69 |
| Confirmed sick window | day 1 caught ✓; warning builds the day before (rec -11/day); day 3 honest miss |

For direct comparison with TemPredict's **82% sens / 63% spec**:

| | TemPredict | This engine |
|---|---|---|
| Sensitivity | 82% | **69%** (episode days) |
| Specificity | 63% | **96%** (3 FP / 78 healthy) |
| Lead time | 2.75 days (RR-driven) | **1-2 days** (recovery-slope, on episodes where it fired) |

**Reading:** this engine trades sensitivity for specificity — it flags less
often but is far more accurate when it does. That is a *deliberate design
choice* (persistence gate) aligned with the target use case: "should I take
it easy today" rather than "should I get tested". TemPredict explicitly
biased toward sensitivity because its use case was screening.

## 4. Where Oura wins, honestly

1. **Respiratory rate.** RR is TemPredict's single strongest pre-symptomatic
   signal (basis of their PX alignment and much of the 2.75-day lead). Oura
   has it; we cannot get it from the Ultrahuman Partner API (verified: no
   respiratory fields documented; legacy endpoint returns `respiratory_graph:
   null`). This is a hard ceiling, not a tuning gap.
2. **Labeled training data.** Oura's model is trained on millions of
   annotated nights. Ours is a rules engine with a handful of labeled days
   so far.
3. **Signal breadth.** 40+ signals vs 3+recovery. Each extra orthogonal
   signal adds discriminative power we simply don't have.

## 5. Where this engine wins (measured, not claimed)

1. **Specificity: 96% vs 63%.** At the current operating point, this engine
   fires on 4% of healthy days; TemPredict's published specificity accepts
   37% false flags. On a retrospective dataset, that difference is the
   difference between "the tool is useful" and "the tool is noise."
2. **Transparency.** Every flag is decomposable into per-metric z-scores and
   contributions. Oura's is a black box.
3. **Zero training data required.** Works from day 1; improves with labels.

## 6. The separation question (sickness vs. non-illness strain)

The unresolved piece. Sick and rough days produce the same RHR↑/HRV↓/Temp↑/
Recovery↓ signature; on the labeled strain days they fully interleave.
Oura does not solve this either (it has no way to know what you did
yesterday), but its extra signals and training data make its *strain* flag
more reliable. Our path to the same place is the label collection loop:

- Nightly cron collects fine/rough/sick labels
- `train.py` gates at 15 sick + 15 rough, then trains + verifies the
  separation classifier (leave-one-out, honest verdict)
- `scenario.py` shows the stakes: if the strong unconfirmed episodes were
  real sickness, the classifier reaches **0.50-0.67 recall / 0.50-0.60
  precision today** — the labels are the single highest-leverage input
  available.

## 7. Verdict

- **On strain detection: competitive.** 69% recall / 96% specificity on the
  retrospective dataset vs. TemPredict's published 82% / 63% — a different,
  more conservative operating point.
- **On pre-symptomatic lead time: cannot beat Oura.** That capability is
  RR-gated, and RR is not obtainable from the Ultrahuman API. Claiming
  otherwise would be dishonest.
- **On sickness-vs-strain: not yet separable.** Requires the label loop
  to complete; the machinery is built, live, and tested.
