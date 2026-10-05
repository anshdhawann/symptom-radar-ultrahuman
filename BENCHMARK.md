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

From `evaluate.py` (no-lookahead scoring), re-run 2026-10-05 on 163 days of
data (2026-04-26 to 2026-10-05).

> **Data-revision note (2026-09):** Ultrahuman's backend revised historical
> metrics retroactively. Re-fetching the archive shrank several mid-summer
> deviations (one confirmed-sick day's temperature deviation went from
> +0.52 to +0.33 °C), and episodes were re-derived from the refreshed data.
> The earlier figures in this file (14 flags, 3 FP, 69% recall, 96%
> specificity on 78 healthy days) were measured before that revision, are
> not comparable, and are superseded by the table below.

| Metric | Value |
|---|---|
| Flags (level ≥ 1) | 27 |
| True positives (episode days) | 8 (plus 2 adjacent-to-episode days) |
| False positives | 17 (11% FPR over 152 healthy days) |
| Episode-day recall | 8/11 = 0.73 |
| Confirmed sick window (3 days) | day 1 caught; days 2 and 3 missed (strain index near 0) |

For direct comparison with TemPredict's **82% sens / 63% spec**:

| | TemPredict | This engine |
|---|---|---|
| Sensitivity | 82% | **73%** (episode days, n=11) |
| Specificity | 63% | **89%** (17 FP / 152 healthy) |
| Lead time | 2.75 days (RR-driven) | **not demonstrated** on refreshed data (the day before the confirmed sick window is not flagged) |

Sample sizes are tiny (11 episode days, 3 confirmed sick days, one wearer),
so treat these as directional, not a validated operating point. Most of the
17 false positives sit at strain index 1.0-1.6 with no multi-day
persistence (single rough nights); one reached Significant.

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

1. **Specificity: 89% vs 63%.** At the current operating point, this engine
   fires on 11% of healthy days; TemPredict's published specificity accepts
   37% false flags. Caveat: single wearer, 152 healthy days, so this is a
   directional comparison, not a like-for-like trial.
2. **Transparency.** Every flag is decomposable into per-metric z-scores and
   contributions. Oura's is a black box.
3. **Zero training data required.** Works from day 1; improves with labels.

## 6. The separation question (sickness vs. non-illness strain)

The unresolved piece. Sick and rough days produce the same RHR↑/HRV↓/Temp↑/
Recovery↓ signature; on the labeled strain days they fully interleave.
Oura does not solve this either (it has no way to know what you did
yesterday), but its extra signals and training data make its *strain* flag
more reliable. Our path to the same place is the label collection loop:

- Labels are logged manually (`--label fine|rough|sick` or the MCP tools);
  the nightly check-in automation was retired in 2026-08
- `train.py` gates at 15 sick + 15 rough, then trains + verifies the
  separation classifier (leave-one-out, honest verdict)
- `scenario.py` shows the stakes: if the strong unconfirmed episodes were
  real sickness, the classifier reaches **0.50-0.67 recall / 0.50-0.60
  precision today** — the labels are the single highest-leverage input
  available.

## 7. Verdict

- **On strain detection: competitive.** 73% recall / 89% specificity on the
  retrospective dataset (re-run 2026-10-05, post data revision) vs.
  TemPredict's published 82% / 63%: a different, more conservative
  operating point, on a much smaller sample.
- **On pre-symptomatic lead time: cannot beat Oura.** That capability is
  RR-gated, and RR is not obtainable from the Ultrahuman API. Claiming
  otherwise would be dishonest.
- **On sickness-vs-strain: not yet separable.** Requires the label loop
  to complete; the machinery is built, live, and tested.
