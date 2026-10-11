# Symptom Radar: Project Status

Handoff artifact. Everything below is measured or verified; nothing is claimed
beyond the evidence. Last measured 2026-10-11 on a frozen 169-day snapshot.

## What this project is

A TemPredict-inspired strain detector for the Ultrahuman Ring. Goal: improve
accuracy in predicting whether the wearer is getting sick, as a local,
privacy-preserving alternative to commercial symptom-radar products.

## What is DONE and verified

| Item | Evidence |
|---|---|
| Strain engine: calendar-aligned clean baseline (the 21 days ending 3 days before today, median/MAD reference), illness-grade days kept out of their own baseline, trajectory boost, persistence, per-metric noise floors | `symptomradar/engine.py` and its tests |
| Honest non-scores: "No data", "Calibrating" and "Insufficient data" instead of a silent Normal | daily report and MCP JSON |
| Re-fetches merge into stored days instead of blanking them; every real change is audited | `metric_revisions` table |
| One definition of "today" (`SYMPTOM_RADAR_TZ`, default: the host's timezone) | `symptomradar/config.py` and its tests |
| Evaluation against independent labels on a frozen, read-only input | `evaluate.py` (numbers below) |
| Separation classifier without lookahead, validated by leave-one-episode-out | `train.py` |
| Extra-signal ablation, defaults unchanged | `ablate.py` (BENCHMARK.md section 7) |
| Confounder logging (alcohol, exercise, travel, sleep loss) | `--confounders`, MCP tools, `evaluate.py` |
| Labelling | `--label` CLI, MCP tools, `labels.py`; manual only, the nightly check-in was retired in 2026-08 |
| Sensitivity analysis | `scenario.py`; now reports that no hypothesis can be validated on current labels |
| Package, console command, offline commands without a token | `symptomradar/` package, `pyproject.toml` (`symptom-radar` command); the root `symptom_radar.py` is a thin entry script |
| Tests and CI | 172 tests, no network and no token; GitHub Actions on Python 3.10 to 3.14 |
| MCP server error handling | standard JSON-RPC codes for bad input, server keeps serving; `test_mcp.py` |

## The measured verdict (details in BENCHMARK.md)

Frozen snapshot, 169 days, 135 assessable:

1. **Strain: every labelled episode flagged.** 4 of 4 rough or sick episodes,
   each on its first or second day; 5 of 8 labelled days. False positives on
   days labelled fine: 1 of 16. Alerts on unlabelled days: 22 of 109 (20%).
2. **Illness: one episode, flagged on its first day, not before.** Days 2
   and 3 scored 0.00 and 0.01. No lead time demonstrated.
3. **Pre-symptomatic lead time: cannot beat Oura.** That capability is
   RR-gated, and RR is not obtainable from the Ultrahuman Partner API
   (verified: no respiratory fields documented; legacy endpoint returns
   `respiratory_graph: null`).
4. **Sickness vs non-illness strain: NOT yet separable.** `train.py --force`
   prints "Cannot validate sick-vs-rough separation with 1 independent sick
   episode(s) and 3 independent rough episode(s)."

The earlier 73% recall / 89% specificity figures, and `scenario.py`'s 0.50 to
0.67 recall, are withdrawn. BENCHMARK.md sections 3 and 6 explain why.

> **Data-revision note (2026-09):** Ultrahuman's backend revised historical
> metrics retroactively (one confirmed-sick day's temperature deviation
> shrank from +0.52 to +0.33 °C, for example). Numbers measured before that
> refresh are not comparable with the ones above.

## What is BLOCKED and why

The sick-vs-strain separation needs **labelled episodes**. `train.py` gates at
15 labelled days **and** 5 independent episodes per class. Current: sick 3
days in 1 episode, rough 5 days in 3 episodes. **Labels must not be
fabricated**; that would corrupt the evaluation worse than missing data.

The retrospective route is **closed**: the wearer no longer remembers the
strongest unconfirmed episodes, so collection is forward-only.

## The path forward (forward-only collection)

- **Sick:** 4 more independent episodes and 12 more days needed. One episode
  occurred in 5.5 months of data; at that pace this is roughly 2 years away
  (ESTIMATED from a single episode).
- **Rough:** 2 more independent episodes and 10 more days needed.
- Log confounders with each label so `evaluate.py` can show which flags they
  explain.

Until then, the engine still ships the practical value: every labelled
strain episode is flagged, and BENCHMARK.md's honest Oura comparison stands.

## If the wearer ever remembers a past episode

```bash
# One sentence, natural language, any remembered episode:
python3 labels.py --from-memory "Jan 15 sick, Feb 18 sick, Mar 26-27 rough"
# (or per day: python3 symptom_radar.py --label sick --label-date YYYY-MM-DD)
# "don't remember" is also a valid, honest answer.

# Then ONE command runs the whole measurement chain
# (status, scenario, train --force, evaluate --labels):
python3 labels.py --verify
```

Once the gate (15 days and 5 independent episodes per class) is met, run
`python3 train.py` without `--force` for the validated verdict.

## Collection (manual, forward-only)

- `python3 symptom_radar.py --label fine|rough|sick`, optionally with
  `--confounders "alcohol=2,exercise=true"`, or the MCP tools, whenever the
  wearer chooses to log.
- Retrospective probing is retired (the wearer doesn't recall the old
  episodes).
- Progress is checkable anytime: `python3 labels.py --status`.
