#!/usr/bin/env python3
"""
Train and verify the sickness-vs-alcohol separation classifier.

Once enough self-report labels exist (target: ~15 sick and ~15 rough days),
this trains a small logistic classifier on the engine's features and
verifies whether it can separate REAL ILLNESS (sick) from ALCOHOL STRAIN
(rough) — the distinction the biometrics alone cannot make.

Features per labeled day (all computed from data BEFORE that day, no lookahead):
  strain_idx  — engine strain index (weighted deviation from clean baseline)
  elevated_3d — how many of the last 3 days were elevated
  rec_decline — 1 if the recovery-slope leading indicator fired (Recovery↓)
  rhr_z       — RHR level z vs baseline (strain direction)
  hrv_z       — HRV level z vs baseline (strain direction)
  temp_z      — Temp level z vs baseline (strain direction)

Method:
  - Pure-stdlib logistic regression (batch gradient descent, L2) — no numpy,
    no sklearn, consistent with the project's requests-only philosophy.
  - Features are z-normalized per column before training.
  - Verification: leave-one-out cross-validation (honest for small n), plus
    per-class strain statistics.

Usage:
    python3 train.py              # train if enough labels, else show what's needed
    python3 train.py --force      # train anyway (e.g. to see where it stands)
    python3 train.py --status     # just show collection progress
"""
import argparse
import math
import random
import sqlite3
import sys

import symptom_radar as sr

# Minimum labels per class before training is meaningful.
MIN_SICK = 15
MIN_ROUGH = 15


# ─── Feature extraction ───────────────────────────────────────────────────────
def _baseline_stats_series(pool, extract):
    base = [extract(d) for d in pool]
    return sr._baseline_stats(base)

def extract_features(history, today_idx):
    """Return the feature vector for the day at today_idx (no lookahead)."""
    today = history[today_idx]
    baseline_pool = history[max(0, today_idx - sr.CLEAN_GUARD - sr.BASELINE_WINDOW)
                            :today_idx - sr.CLEAN_GUARD] or []
    if len(baseline_pool) < 7:
        return None

    recent = history[max(0, today_idx - sr.TRAJ_WINDOW + 1):today_idx + 1]

    feats = {}
    for key, extract, _w, inverted, _lbl in sr.STRAIN_METRICS:
        mean, noise = _baseline_stats_series(baseline_pool, extract)
        val = extract(today)
        if mean is None or noise is None or noise == 0 or val is None:
            feats[key + "_z"] = 0.0
            continue
        z = (val - mean) / noise
        feats[key + "_z"] = (-z if inverted else z)  # strain direction, + = strain

    # Strain index + persistence from the engine's own scoring
    level, detail = sr.assess_strain(history[:today_idx + 1])
    import re
    m_idx = re.search(r"Strain index: ([\d.]+)", detail)
    m_el = re.search(r"Elevated in (\d)/3", detail)
    feats["strain_idx"] = float(m_idx.group(1)) if m_idx else 0.0
    feats["elevated_3d"] = int(m_el.group(1)) if m_el else 0
    feats["rec_decline"] = 1.0 if "Recovery↓" in detail else 0.0

    # Sleep score as an additional discriminator (poor sleep accompanies
    # illness; recovered days tend to have higher sleep scores). Zero-padded
    # when missing so the classifier still trains.
    feats["sleep_z"] = today.get("sleep_score") or 0.0

    # Recovery-rebound shape: change in recovery 24h AFTER today (sick days
    # tend to stay depressed, rough days bounce back). Only computable
    # retroactively — used by scenario.py's diagnostic analysis, NOT for
    # live prediction. Zero when the next day is missing.
    if today_idx + 1 < len(history):
        rec_today = today.get("recovery_index")
        rec_next = history[today_idx + 1].get("recovery_index")
        if rec_today is not None and rec_next is not None:
            feats["rec_rebound"] = (rec_next - rec_today) / 10.0  # scaled to ~unit
        else:
            feats["rec_rebound"] = 0.0
    else:
        feats["rec_rebound"] = 0.0
    return feats


# Feature order (must stay stable across calls)
FEATURE_ORDER = ["strain_idx", "elevated_3d", "rec_decline",
                 "rhr_z", "hrv_z", "temp_z", "sleep_z", "rec_rebound"]


# ─── Pure-stdlib logistic regression ──────────────────────────────────────────
def logistic_predict(w, b, x):
    z = b + sum(wi * xi for wi, xi in zip(w, x))
    # numerically stable sigmoid
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)

def train_logistic(X, y, l2=0.01, lr=0.1, epochs=2000, seed=42):
    """Batch gradient descent logistic regression. X: list of feature vectors,
    y: list of 0/1 labels. Returns (w, b)."""
    rng = random.Random(seed)
    n = len(X)
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            p = logistic_predict(w, b, xi)
            err = p - yi
            for j in range(d):
                gw[j] += err * xi[j]
            gb += err
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * gb / n
    return w, b

def zscore_cols(X):
    """Column-wise z-normalization. Returns (X_norm, (means, stds))."""
    d = len(X[0])
    means, stds = [], []
    for j in range(d):
        col = [x[j] for x in X]
        m = sum(col) / len(col)
        s = math.sqrt(sum((v - m) ** 2 for v in col) / len(col)) or 1.0
        means.append(m)
        stds.append(s)
    Xn = [[(x[j] - means[j]) / stds[j] for j in range(d)] for x in X]
    return Xn, (means, stds)

def loo_accuracy(X, y, l2=0.01):
    """Leave-one-out cross-validation accuracy."""
    n = len(X)
    correct = 0
    preds = []
    for i in range(n):
        Xtr = [x for j, x in enumerate(X) if j != i]
        ytr = [v for j, v in enumerate(y) if j != i]
        w, b = train_logistic(Xtr, ytr, l2=l2)
        p = logistic_predict(w, b, X[i])
        pred = 1 if p >= 0.5 else 0
        preds.append((pred, y[i]))
        if pred == y[i]:
            correct += 1
    return correct / n, preds


# ─── Main ─────────────────────────────────────────────────────────────────────
def collect_labeled(history):
    """Build (features, y, meta) for every labeled day, no lookahead."""
    labels = {d: lbl for d, lbl, _ in sr.get_labels()}
    by_date = {h["date"]: i for i, h in enumerate(history)}
    X, y, meta = [], [], []
    for date, lbl in labels.items():
        if lbl not in ("sick", "rough"):
            continue
        idx = by_date.get(date)
        if idx is None:
            continue
        feats = extract_features(history, idx)
        if feats is None:
            continue
        X.append([feats[k] for k in FEATURE_ORDER])
        y.append(1 if lbl == "sick" else 0)
        meta.append((date, lbl, feats))
    return X, y, meta


def status_report(history):
    labels = sr.get_labels()
    counts = {"sick": 0, "rough": 0, "fine": 0}
    for _, lbl, _ in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    print("Label collection status:")
    print(f"  fine : {counts['fine']}")
    print(f"  rough: {counts['rough']}  (need {MIN_ROUGH} → "
          f"{max(0, MIN_ROUGH - counts['rough'])} more)")
    print(f"  sick : {counts['sick']}  (need {MIN_SICK} → "
          f"{max(0, MIN_SICK - counts['sick'])} more)")
    print()
    ready = counts["sick"] >= MIN_SICK and counts["rough"] >= MIN_ROUGH
    if not ready:
        print(f"Not ready to train: need {max(0, MIN_SICK - counts['sick'])} more "
              f"sick + {max(0, MIN_ROUGH - counts['rough'])} more rough days.")
        print("The nightly cron (21:00) is collecting these automatically.")
        print(f"Estimated: {max(MIN_SICK - counts['sick'], MIN_ROUGH - counts['rough'], 0)} "
              f"more labeled days minimum.")
    return ready, counts


def main():
    ap = argparse.ArgumentParser(description="Sickness-vs-alcohol separation classifier")
    ap.add_argument("--force", action="store_true",
                    help="train even if below minimum label counts")
    ap.add_argument("--status", action="store_true",
                    help="show collection progress only")
    args = ap.parse_args()

    history = []
    conn = sqlite3.connect(sr.DB_PATH)
    cur = conn.execute("SELECT * FROM daily_snapshots ORDER BY date ASC")
    cols = [d[0] for d in cur.description]
    history = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()

    ready, counts = status_report(history)
    if args.status:
        return
    if not ready and not args.force:
        return
    if not ready and args.force:
        print("\n[--force] Training anyway on the current (insufficient) labels — "
              "results are a diagnostic peek, not a validated classifier.")

    X, y, meta = collect_labeled(history)
    print(f"\nTraining on {len(X)} labeled days "
          f"({sum(1 for v in y if v == 1)} sick, {sum(1 for v in y if v == 0)} rough)")
    if len(X) < 4 or (not args.force and (counts["sick"] < MIN_SICK or counts["rough"] < MIN_ROUGH)):
        print("Too few labels for a meaningful classifier. "
              "Keep the nightly labels coming.")
        return

    Xn, _ = zscore_cols(X)
    acc, preds = loo_accuracy(Xn, y)

    print(f"\n=== Leave-one-out verification ===")
    print(f"Accuracy: {acc:.2f}  ({len(X)} labeled days)")
    tp = sum(1 for p, t in preds if p == 1 and t == 1)
    tn = sum(1 for p, t in preds if p == 0 and t == 0)
    fp = sum(1 for p, t in preds if p == 1 and t == 0)
    fn = sum(1 for p, t in preds if p == 0 and t == 1)
    print(f"  sick correctly classified (TP): {tp}/{sum(1 for t in y if t == 1)}")
    print(f"  rough correctly classified (TN): {tn}/{sum(1 for t in y if t == 0)}")
    print(f"  sick misclassified as rough (FN): {fn}")
    print(f"  rough misclassified as sick (FP): {fp}")
    print()
    print("Confusion matrix (rows=predicted, cols=actual):")
    print(f"          sick  rough")
    print(f"  sick    {tp:>3}   {fp:>3}")
    print(f"  rough   {fn:>3}   {tn:>3}")
    # The honest verdict: accuracy alone is misleading on imbalanced data
    # (predicting "rough" for everything scores well when most days are rough).
    # What matters is sick-recall (TP rate) and sick-precision.
    n_sick = sum(1 for t in y if t == 1)
    n_rough = sum(1 for t in y if t == 0)
    sick_recall = tp / n_sick if n_sick else 0.0
    sick_precision = tp / (tp + fp) if (tp + fp) else 0.0
    print()
    print(f"Sick recall: {tp}/{n_sick} = {sick_recall:.2f}  "
          f"(fraction of real sick days correctly caught)")
    print(f"Sick precision: {sick_precision:.2f}  "
          f"(of days called sick, how many were really sick)")
    print()
    if n_sick < 5:
        print(f"⚠️ Too few sick labels ({n_sick}) for a trustworthy verdict —")
        print("   this is a diagnostic peek, not a validated classifier.")
        verdict = "peek"
    elif sick_recall >= 0.7 and sick_precision >= 0.5 and acc >= 0.7:
        print("✅ Separable — the classifier catches ≥70% of real sick days")
        print("   with ≥50% precision on its sick calls. The labels have paid off.")
        verdict = "separable"
    elif sick_recall >= 0.5 and sick_precision >= 0.3:
        print("⚠️ Weak separation — better than chance, catches half the sick")
        print("   days, but with false alarms. More labels will sharpen it.")
        verdict = "weak"
    else:
        print("❌ Not separable yet — sick days are still being missed or"
              " confounded.")
        print("   Keep collecting. The recovery-curve shape (hangovers rebound")
        print("   in 1 day, illness persists 2-3) may need a dedicated feature.")
        verdict = "not_separable"
    return verdict
    print()
    print("Per-day predictions (highest strain first):")
    for (date, lbl, feats), (pred, true) in sorted(
            zip(meta, preds), key=lambda x: -x[0][2]["strain_idx"]):
        mark = "✓" if pred == true else "✗"
        print(f"  {date}  actual={lbl:<6} pred={'sick' if pred else 'rough':<6} "
              f"{mark}  strain={feats['strain_idx']:.2f} "
              f"elev={feats['elevated_3d']}/3 rec↓={int(feats['rec_decline'])}")


if __name__ == "__main__":
    main()
