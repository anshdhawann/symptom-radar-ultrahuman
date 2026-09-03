# Symptom Radar — Project Status

Handoff artifact. Everything below is measured or verified; nothing is claimed
beyond the evidence.

## What this project is

A TemPredict-inspired strain detector for the Ultrahuman Ring. Goal: improve
accuracy in predicting whether the wearer is getting sick, as a local,
privacy-preserving alternative to commercial symptom-radar products.

## What is DONE and verified

| Item | Evidence |
|---|---|
| Strain engine (clean baseline, trajectory, persistence, noise scaling) | `symptom_radar.py` — 40/40 tests pass |
| Accuracy fixes: one-sided HRV, median-gated mean-reversion, recovery-slope leading indicator | commit `399f349` |
| Retrospective evaluation on months of real data | `evaluate.py`: 23 flags, 8 true positives, 13 false positives on refreshed data (see note below) |
| Oura comparison (honest, vs published TemPredict numbers) | `BENCHMARK.md` |
| Labeling infrastructure (fine/rough/sick) | `--label` CLI, MCP tools, `labels.py` |
| Label collection | manual only (`--label` CLI / MCP tools) — the nightly check-in automation was retired |
| Classification + verification | `train.py` (pure-stdlib logistic, leave-one-out, honest verdicts) |
| Sensitivity analysis | `scenario.py` (hypotheses A/B/C for unconfirmed episodes) |
| End-to-end pipeline proof | dry-run on scratch DB: from-memory → status → train → evaluate all work |

## The measured verdict (BENCHMARK.md, in full)

> **Data-revision note (2026-09):** Ultrahuman's backend revised historical
> metrics retroactively. Re-fetching the full archive changed mid-summer
> values enough that most multi-day deviation clusters the original episode
> list was built from no longer exist in the data (one confirmed-sick day's
> temperature deviation shrank from +0.52 to +0.33 °C, for example). The
> episodes were re-derived from the refreshed data, and `evaluate.py` now
> also counts self-reported rough/sick labels as ground truth. Numbers
> before/after the refresh are not comparable.

1. **Strain detection: catches every labeled strain day.** On refreshed
   data: 8/11 episode-or-labeled days flagged; all self-reported rough days
   with ring data caught. The cost is a band of marginal level-1 flags on
   unlabeled days (13, ~11% of healthy days) — most sit at index 1.0-1.4
   with no multi-day persistence, i.e. single rough-sleep nights, not
   illness-shaped signatures.
2. **Pre-symptomatic lead time: cannot beat Oura.** That capability is
   RR-gated, and RR is not obtainable from the Ultrahuman Partner API
   (verified: no respiratory fields documented; legacy endpoint returns
   `respiratory_graph: null`).
3. **Sickness vs non-illness strain: NOT yet separable.** `evaluate.py --labels`
   shows sick/rough fully interleaved. `train.py --force` shows sick recall
   0/2 — the classifier catches no real sick days with current labels.

## What is BLOCKED and why

The sick-vs-strain separation requires **labeled days**. `train.py` gates at
15 sick + 15 rough. **Labels must not be fabricated** — that would corrupt the
evaluation worse than missing data.

The retrospective route is **closed**: the wearer no longer remembers the
strongest unconfirmed episodes, and the nightly check-in no longer asks about
them. `scenario.py` documented the stakes at the time: if those episodes were
**sick** (hypothesis A), the classifier reached 0.50-0.67 recall / 0.50-0.60
precision; if **rough**, it stayed at 0 until real illness added sick labels.
Those hypotheses are no longer testable — collection is forward-only.

## The path forward (forward-only collection)

Realistic pace from the wearer's history (3 sick days in ~3 months, most in
one episode): reaching 15 sick labels by manual logging alone is likely a
**12+ month** wait. Rough labels arrive faster (~1.7/month → 10 more ≈ 6
months). The honest verdict: measured sick-vs-alcohol separation will not
exist until real sick episodes accumulate AND are labeled as they happen.

Until then, the engine still ships the practical value: every labeled
strain day is flagged, plus `morning_alertness` (sleep-inertia minutes,
captured since 2026-09) is stored for the label-correlation work ahead,
and BENCHMARK.md's honest Oura comparison stands.

## If the wearer ever remembers a past episode

```bash
# One sentence, natural language — any remembered episode:
python3 labels.py --from-memory "Jan 15 sick, Feb 18 sick, Mar 26-27 rough"
# (or per-day: python3 symptom_radar.py --label sick --label-date YYYY-MM-DD)
# 'don't remember' is also a valid, honest answer.

# Then — ONE command runs the whole measurement chain
# (status → scenario → train --force → evaluate --labels):
python3 labels.py --verify
```

If the gate (15 sick + 15 rough) is met, drop `--force` (or edit run_verify)
and `train.py` gives the validated verdict.

## Collection (manual, forward-only)

- The nightly check-in automation was retired (2026-08) — collection is
  manual: `python3 symptom_radar.py --label fine|rough|sick` or the MCP
  tools whenever the wearer chooses to log.
- Retrospective probing is retired (the wearer doesn't recall the old
  episodes); collection is forward-only.
- Progress checkable anytime: `python3 labels.py --status`.
