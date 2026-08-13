#!/usr/bin/env python3
"""
Sensitivity analysis: what would the sickness-vs-strain classifier achieve
IF the strong unconfirmed episodes were labeled one way or the other?

The user has not yet confirmed whether the strong episodes were real
sickness or non-illness strain. Rather than fabricate ground truth, this
script runs BOTH hypotheses explicitly and reports what the classifier
would score under each — so the user can see whether their memory is
worth digging up.

Dates below are SYNTHETIC EXAMPLE DATA — replace with dates from your own
data before running (see labels.py --from-memory / --review).

Hypotheses:
  A (sick-heavy):  episodes = sick (plausible illness pattern)
  B (rough-heavy): episodes = rough (non-illness strain)
  C (mixed):       strongest episode = sick, others = rough (null
                   hypothesis: strong episodes are indistinguishable strain)

Usage:
    python3 scenario.py
"""
import sqlite3

import symptom_radar as sr
import train as tr

# Candidate strong episodes whose true nature only the user knows
# (synthetic example data — replace)
UNCONFIRMED = {
    "2024-02-15": "example day 1 (Significant, HRV depressed, strain 2.82)",
    "2024-03-01": "example day 2 (temp +0.70, strain 3.51 — strongest in dataset)",
    "2024-03-10": "example day 3 (recovery 33, strain 2.03)",
    "2024-03-11": "example day 4 (recovery 29, strain 2.10)",
}

HYPOTHESES = {
    "A sick-heavy": {"2024-02-15": "sick", "2024-03-01": "sick",
                     "2024-03-10": "sick", "2024-03-11": "sick"},
    "B rough-heavy": {"2024-02-15": "rough", "2024-03-01": "rough",
                      "2024-03-10": "rough", "2024-03-11": "rough"},
    "C mixed (day2 sick)": {"2024-02-15": "rough", "2024-03-01": "sick",
                            "2024-03-10": "rough", "2024-03-11": "rough"},
}


def run_hypothesis(history, override, include_extra=True):
    """Train + LOO-verify with the hypothesis's labels overlaid on real labels.

    include_extra=False uses ONLY the core engine features (strain_idx,
    elevated_3d, rec_decline, rhr_z, hrv_z, temp_z) — the comparison set.
    include_extra=True adds sleep_z + rec_rebound (the hypothesized
    discriminators). This shows whether the new features actually help.
    """
    labels = {d: lbl for d, lbl, _ in sr.get_labels()}
    for date, lbl in override.items():
        labels[date] = lbl  # override, not write to DB

    order = list(tr.FEATURE_ORDER)
    if not include_extra:
        order = [k for k in order if k not in ("sleep_z", "rec_rebound")]

    X, y, meta = [], [], []
    by_date = {h["date"]: i for i, h in enumerate(history)}
    for date, lbl in labels.items():
        if lbl not in ("sick", "rough"):
            continue
        idx = by_date.get(date)
        if idx is None:
            continue
        feats = tr.extract_features(history, idx)
        if feats is None:
            continue
        X.append([feats[k] for k in order])
        y.append(1 if lbl == "sick" else 0)
        meta.append((date, lbl))

    if len(X) < 6 or sum(y) < 2 or (len(y) - sum(y)) < 2:
        return None

    Xn, _ = tr.zscore_cols(X)
    acc, preds = tr.loo_accuracy(Xn, y)
    n_sick = sum(y)
    tp = sum(1 for p, t in preds if p == 1 and t == 1)
    fp = sum(1 for p, t in preds if p == 1 and t == 0)
    return {
        "n": len(X), "n_sick": n_sick, "acc": acc,
        "sick_recall": tp / n_sick if n_sick else 0.0,
        "sick_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "tp": tp, "fp": fp,
    }


def main():
    conn = sqlite3.connect(sr.DB_PATH)
    cur = conn.execute("SELECT * FROM daily_snapshots ORDER BY date ASC")
    cols = [d[0] for d in cur.description]
    history = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()

    print("=" * 78)
    print("SENSITIVITY ANALYSIS — what the classifier achieves IF the")
    print("unconfirmed strong episodes were labeled one way or the other")
    print("=" * 78)
    print()
    print("Unconfirmed episodes (only the user knows the truth):")
    for d, desc in UNCONFIRMED.items():
        print(f"  {d}  {desc}")
    print()

    for label, extra in [("core features", False), ("core+sleep+rebound", True)]:
        print(f"\n--- Feature set: {label} ---")
        print(f"{'Hypothesis':<24} {'n':>3} {'sick':>4} {'acc':>6} {'recall':>7} {'prec':>6}")
        print("-" * 60)
        for name, override in HYPOTHESES.items():
            r = run_hypothesis(history, override, include_extra=extra)
            if r is None:
                print(f"{name:<24} (insufficient labeled classes)")
                continue
            print(f"{name:<24} {r['n']:>3} {r['n_sick']:>4} {r['acc']:>6.2f} "
                  f"{r['sick_recall']:>7.2f} {r['sick_precision']:>6.2f}")

    print()
    print("Reading: recall = fraction of real sick days the classifier catches;")
    print("precision = of days it calls sick, how many really were.")
    print("Compare the two feature sets: if sleep_z/rec_rebound help under the")
    print("same hypothesis, the new features earn their place; if not, they")
    print("don't (tested, not assumed).")
    print()
    print("Current REAL labels (no hypothesis applied): see labels.py --status.")
    print("The nightly cron keeps collecting; labels.py --status tracks progress.")


if __name__ == "__main__":
    main()
