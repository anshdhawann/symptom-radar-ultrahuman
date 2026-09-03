#!/usr/bin/env python3
"""
Retrospective evaluation harness for the Symptom Radar engine.

Scores every day in ultrahuman.db using only history BEFORE that day
(no lookahead), then compares against labeled ground truth to measure
recall (did we catch strain episodes?) and false-positive rate.

Ground truth:
  - CONFIRMED_SICK: days the user confirmed feeling ill.
  - EPISODES: multi-signal physiological clusters (RHR↑ + HRV↓ + Temp↑ + Rec↓
    together for 2+ days) — the same physiological pattern as a confirmed
    sick window. These are treated as true positives.
  - Everything else = healthy baseline. Days adjacent to episodes (1 day
    before/after) are excluded from both counts.

The dates below are SYNTHETIC EXAMPLE DATA so the harness runs out of the
box. Replace them with dates from your own data before trusting the
numbers (see labels.py --from-memory / --review to collect yours).

Usage:
    python3 evaluate.py              # summary vs corrected labels
    python3 evaluate.py --labels     # label-aware separation analysis:
                                     # do sick/rough/fine days cluster apart?
"""
import argparse
import datetime
import json
import os
import sqlite3

import symptom_radar as sr

# ── Ground truth ──────────────────────────────────────────────────────────────
# Real episode dates are PERSONAL HEALTH DATA and live in episodes.json
# (gitignored), NOT in this file. Shape:
#   {"confirmed_sick": ["YYYY-MM-DD", ...],
#    "episodes":        ["YYYY-MM-DD", ...]}   # multi-signal clusters
# Additionally, self-reported rough/sick labels (daily_labels table, local DB)
# are independent user ground truth and count as episode days.
# If episodes.json is absent, synthetic example dates are used and a warning
# is printed — the printed metrics are then meaningless.
_EPISODES_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "episodes.json")

if os.path.exists(_EPISODES_FILE):
    with open(_EPISODES_FILE) as f:
        _gt = json.load(f)
    CONFIRMED_SICK = set(_gt.get("confirmed_sick", []))
    EPISODES = CONFIRMED_SICK | set(_gt.get("episodes", []))
else:
    print("⚠️  episodes.json not found — using SYNTHETIC example ground truth.")
    print("    All numbers below are meaningless until you add your own "
          "episode dates (see README / labels.py).")
    # Synthetic example data — replace with your own via episodes.json.
    CONFIRMED_SICK = {"2024-03-10", "2024-03-11", "2024-03-12"}
    EPISODES = CONFIRMED_SICK | {
        "2024-02-14", "2024-02-15", "2024-02-16",
        "2024-02-28", "2024-02-29",
        "2024-03-01",
    }

# User self-reported rough/sick days are ground truth in their own right:
# the engine should flag strain on days the wearer actually felt rough.
try:
    _labeled = {d for d, lab, _ in sr.get_labels() if lab in ("rough", "sick")}
except Exception:
    _labeled = set()
EPISODES |= _labeled

# 1-day buffer around episodes: pre-onset ramp + recovery tail.
BUFFER = set()
for e in EPISODES:
    d = datetime.date.fromisoformat(e)
    BUFFER |= {str(d - datetime.timedelta(days=i)) for i in range(-1, 2)}


def load_history():
    conn = sqlite3.connect(sr.DB_PATH)
    cur = conn.execute("SELECT * FROM daily_snapshots ORDER BY date ASC")
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


def score_all(history):
    results = []
    for i in range(len(history)):
        level, detail = sr.assess_strain(history[:i + 1])
        results.append((history[i]["date"], level, detail))
    return results


def analyze_separation(history, results):
    """Label-aware analysis: do sick / rough / fine days cluster apart on the
    engine's features? This is the honest test of whether we can separate
    real illness from non-illness strain using the self-report labels.

    Features per day (all computed from data BEFORE that day — no lookahead):
      strain_idx  — the engine's strain index (weighted deviation)
      rec_slope   — recovery 3-day slope (positive = falling = warning)
      elevated_3d — how many of last 3 days were elevated
    """
    labels = {d: lbl for d, lbl, _ in sr.get_labels()}
    import re

    rows = []
    for i in range(len(history)):
        date = history[i]["date"]
        if date not in labels:
            continue
        level, detail = results[i][1], results[i][2]
        m_idx = re.search(r"Strain index: ([\d.]+)", detail)
        m_el = re.search(r"Elevated in (\d)/3", detail)
        strain = float(m_idx.group(1)) if m_idx else 0.0
        elev = int(m_el.group(1)) if m_el else 0
        has_rec_down = "Recovery↓" in detail
        rows.append({
            "date": date, "label": labels[date],
            "strain": strain, "elev": elev, "rec_down": has_rec_down,
            "level": level,
        })

    by_label = {"fine": [], "rough": [], "sick": []}
    for r in rows:
        by_label[r["label"]].append(r)

    print("=" * 78)
    print("LABEL SEPARATION ANALYSIS (self-reported fine/rough/sick)")
    print("=" * 78)
    print(f"{'Label':<8} {'n':>3} {'strain mean':>11} {'strain range':>20} "
          f"{'flag%':>6} {'rec↓%':>6}")
    for lbl in ("fine", "rough", "sick"):
        rs = by_label[lbl]
        if not rs:
            print(f"{lbl:<8} {0:>3}  (no labels)")
            continue
        strains = [r["strain"] for r in rs]
        flag_pct = 100 * sum(1 for r in rs if r["level"] >= 1) / len(rs)
        rec_pct = 100 * sum(1 for r in rs if r["rec_down"]) / len(rs)
        print(f"{lbl:<8} {len(rs):>3} {sum(strains)/len(strains):>11.2f} "
              f"[{min(strains):.2f}–{max(strains):.2f}]".ljust(41)
              + f"{flag_pct:>6.0f}% {rec_pct:>6.0f}%")

    print()
    print("Per-day detail:")
    for r in sorted(rows, key=lambda x: -x["strain"]):
        mark = " ✓" if r["level"] >= 1 else ""
        print(f"  {r['date']}  {r['label']:<6} strain={r['strain']:.2f} "
              f"elev={r['elev']}/3 rec↓={int(r['rec_down'])} lvl={r['level']}{mark}")

    # Separation score: how well does strain index separate sick from rough?
    sick = [r["strain"] for r in by_label["sick"]]
    rough = [r["strain"] for r in by_label["rough"]]
    fine = [r["strain"] for r in by_label["fine"]]
    print()
    if sick and rough and fine:
        # Simple overlap measure: fraction of sick days with strain above the
        # max fine day's strain (a usable threshold).
        threshold = max(fine)
        above = sum(1 for s in sick if s > threshold)
        rough_above = sum(1 for s in rough if s > threshold)
        print(f"Separation: strain threshold = {threshold:.2f} "
              f"(max of {len(fine)} fine days)")
        print(f"  sick days above threshold: {above}/{len(sick)}")
        print(f"  rough days above threshold: {rough_above}/{len(rough)}")
        print(f"  → sick-vs-fine separation: {above}/{len(sick)} "
              f"({100*above/len(sick):.0f}%)")
        if rough:
            print(f"  → rough-vs-fine separation: {rough_above}/{len(rough)} "
                  f"({100*rough_above/len(rough):.0f}%)")
    print()
    print("Interpretation: if 'sick' and 'rough' rows interleave, the current")
    print("features CANNOT separate illness from non-illness strain — that is")
    print("the limit the labels exist to overcome (need more labeled days).")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="store_true",
                    help="run label-aware separation analysis")
    args = ap.parse_args()

    history = load_history()
    results = score_all(history)

    if args.labels:
        analyze_separation(history, results)
        return

    flagged = [(d, lvl) for d, lvl, _ in results if lvl >= 1]
    tp = [d for d, _ in flagged if d in EPISODES]
    fp = [d for d, _ in flagged if d not in EPISODES and d not in BUFFER]

    episode_days = [d for d in EPISODES
                    if d in {h["date"] for h in history}]
    episodes_hit = set(tp)

    print(f"Loaded {len(history)} days: {history[0]['date']} → {history[-1]['date']}")
    print(f"Confirmed sick (user report): {sorted(CONFIRMED_SICK)}")
    print(f"Episodes (multi-signal clusters): {len(episode_days)} days")
    print()
    print(f"Flags: {len(flagged)}")
    print(f"  True positives (episode days): {len(tp)}")
    print(f"  False positives: {len(fp)}  → {sorted(fp)}")
    print(f"  Adjacent-to-episode (ok): {len([d for d, _ in flagged if d not in EPISODES and d in BUFFER])}")
    print()
    print(f"Recall (episode days caught): {len(episodes_hit)}/{len(episode_days)} = {len(episodes_hit)/len(episode_days):.2f}")
    healthy_n = len([d for d in history if d['date'] not in EPISODES])
    print(f"False-positive rate: {len(fp)}/{healthy_n} healthy days = {len(fp)/healthy_n:.2f}")
    print()
    print(f"Confirmed sick window: day1 caught={any(d==sorted(CONFIRMED_SICK)[0] for d in tp)}  "
          f"day3 caught={any(d==sorted(CONFIRMED_SICK)[-1] for d in tp)}")
    print()
    print("All flagged days:")
    for d, lvl, detail in results:
        if lvl >= 1:
            tag = "EPISODE✓" if d in EPISODES else ("buffer" if d in BUFFER else "FP")
            idx_line = [x for x in detail.split("\n") if "Strain index" in x]
            print(f"  {d}  {sr.STRAIN_ICONS[lvl]:<24} {tag:<10} {idx_line[0] if idx_line else ''}")


if __name__ == "__main__":
    main()
