#!/usr/bin/env python3
"""Train a live-feature sick-vs-rough classifier from real self-report labels.

Training requires 15 assessable days and five independent episodes per class.
Validation holds out whole same-label episodes, allowing one unlabeled gap
day, and fits column normalization on each training fold only.

Usage:
    python3 train.py              # show progress unless collection gate is met
    python3 train.py --force      # diagnostic analysis below the collection gate
    python3 train.py --status     # collection progress only
    python3 train.py --offline    # next-day rebound, not deployable
"""
import argparse
import math
import random
import sqlite3
from datetime import date as calendar_date

from symptomradar import engine, storage

# Minimum labels per class before training is meaningful.
MIN_SICK = 15
MIN_ROUGH = 15
# Five episodes per class leave four independent examples in every held-out fold.
MIN_EPISODES = 5


# ─── Feature extraction ───────────────────────────────────────────────────────
def extract_features(history, today_idx, offline=False):
    """Return morning features, with next-day rebound only when offline=True."""
    today = history[today_idx]
    result = engine.assess_strain_result(history[:today_idx + 1])
    if result["status"] != "ok":
        return None

    feats = {}
    for key, _extract, _w, inverted, _lbl in engine.STRAIN_METRICS:
        z = result["metrics"][key]["z"]
        feats[key + "_z"] = 0.0 if z is None else -z if inverted else z
    feats["strain_idx"] = result["strain_index"]
    feats["elevated_3d"] = result["elevated_recent"]
    recovery_slope = result["metrics"].get("recovery", {}).get("z_slope")
    feats["rec_decline"] = float(recovery_slope is not None and recovery_slope > 1.0)

    # Sleep score as an additional discriminator (poor sleep accompanies
    # illness; recovered days tend to have higher sleep scores). Zero-padded
    # when missing so the classifier still trains.
    feats["sleep_z"] = today.get("sleep_score") or 0.0

    if not offline:
        return feats

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


# Morning features include the scored night, but never the following day.
FEATURE_ORDER = ["strain_idx", "elevated_3d", "rec_decline",
                 "rhr_z", "hrv_z", "temp_z", "sleep_z"]
OFFLINE_FEATURE_ORDER = FEATURE_ORDER + ["rec_rebound"]


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

def normalize_fold(X, train_indices, test_indices):
    """Fit on training rows and apply those statistics to held-out rows."""
    Xtr, stats = zscore_cols([X[i] for i in train_indices])
    means, stds = stats
    Xte = [[(X[i][j] - means[j]) / stds[j] for j in range(len(means))]
           for i in test_indices]
    return Xtr, Xte, stats


def label_episodes(labels):
    """Map dates to same-label episode starts, allowing one unlabeled gap day."""
    groups = {}
    previous_date = previous_label = start = None
    for day, label, _ in sorted(labels):
        current = calendar_date.fromisoformat(day)
        if (previous_date is None or label != previous_label
                or (current - previous_date).days > 2):
            start = day
        groups[day] = start
        previous_date, previous_label = current, label
    return groups


def episode_counts(y, groups):
    return {label: len({g for value, g in zip(y, groups) if value == target})
            for label, target in (("sick", 1), ("rough", 0))}


def loo_accuracy(X, y, l2=0.01, groups=None):
    """Cross-validation on raw features with fold-local normalization.

    Without groups this is day LOO, retained for numerical unit tests only.
    User-facing analyses always supply episode IDs.
    """
    groups = list(range(len(X))) if groups is None else groups
    preds = [(0, value) for value in y]
    for group in dict.fromkeys(groups):
        train_indices = [i for i, g in enumerate(groups) if g != group]
        test_indices = [i for i, g in enumerate(groups) if g == group]
        if not train_indices:
            return None, []
        Xtr, Xte, _ = normalize_fold(X, train_indices, test_indices)
        ytr = [y[i] for i in train_indices]
        w, b = train_logistic(Xtr, ytr, l2=l2)
        for i, x in zip(test_indices, Xte):
            pred = int(logistic_predict(w, b, x) >= 0.5)
            preds[i] = (pred, y[i])
    if not preds:
        return None, []
    return sum(p == t for p, t in preds) / len(preds), preds


def episode_accuracy(X, y, groups, l2=0.01):
    """Leave one whole episode out, requiring both classes in every fold."""
    if min(episode_counts(y, groups).values()) < 2:
        return None, []
    return loo_accuracy(X, y, l2=l2, groups=groups)


# ─── Main ─────────────────────────────────────────────────────────────────────
def collect_labeled(history, offline=False):
    """Build raw (features, y, meta) for assessable labeled days."""
    labels = {d: lbl for d, lbl, _ in storage.get_labels()}
    by_date = {h["date"]: i for i, h in enumerate(history)}
    X, y, meta = [], [], []
    for date, lbl in sorted(labels.items()):
        if lbl not in ("sick", "rough"):
            continue
        idx = by_date.get(date)
        if idx is None:
            continue
        feats = extract_features(history, idx, offline=offline)
        if feats is None:
            continue
        order = OFFLINE_FEATURE_ORDER if offline else FEATURE_ORDER
        X.append([feats[k] for k in order])
        y.append(1 if lbl == "sick" else 0)
        meta.append((date, lbl, feats))
    return X, y, meta


def status_report(history):
    labels = storage.get_labels()
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
    _, eligible_y, meta = collect_labeled(history)
    episodes = label_episodes(labels)
    ec = episode_counts(eligible_y, [episodes[d] for d, _, _ in meta])
    print(f"Assessable labeled days: sick={eligible_y.count(1)}, rough={eligible_y.count(0)}")
    print(f"Independent assessable episodes: sick={ec['sick']}, rough={ec['rough']}")
    print(f"Need {MIN_EPISODES} episodes per class as well as 15 days per class.")
    ready = (eligible_y.count(1) >= MIN_SICK and eligible_y.count(0) >= MIN_ROUGH
             and min(ec.values()) >= MIN_EPISODES)
    if ready:
        print("Ready to train: collection gate met. Run python3 train.py.")
    else:
        print(f"Need {max(0, MIN_EPISODES - ec['sick'])} more sick episodes + "
              f"{max(0, MIN_EPISODES - ec['rough'])} more rough episodes.")
        print(f"Not ready to train: need {max(0, MIN_SICK - eligible_y.count(1))} more "
              f"assessable sick + {max(0, MIN_ROUGH - eligible_y.count(0))} more assessable rough days.")
        print("Collect labels manually: python3 symptom_radar.py --label fine|rough|sick")
    return ready, counts


def main():
    ap = argparse.ArgumentParser(description="Sickness-vs-alcohol separation classifier")
    ap.add_argument("--force", action="store_true",
                    help="train even if below minimum label counts")
    ap.add_argument("--status", action="store_true",
                    help="show collection progress only")
    ap.add_argument("--offline", action="store_true",
                    help="include lookahead features for retrospective analysis only")
    args = ap.parse_args()
    if args.offline:
        print("WARNING: --offline uses next-day recovery; results are not deployable.")
    print("Features: " + ", ".join(OFFLINE_FEATURE_ORDER if args.offline else FEATURE_ORDER))

    history = []
    conn = sqlite3.connect(storage.DB_PATH)
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

    X, y, meta = collect_labeled(history, offline=args.offline)
    print(f"\nTraining on {len(X)} labeled days "
          f"({sum(1 for v in y if v == 1)} sick, {sum(1 for v in y if v == 0)} rough)")
    if len(X) < 4 or (not args.force and (counts["sick"] < MIN_SICK or counts["rough"] < MIN_ROUGH)):
        print("Too few labels for a meaningful classifier. "
              "Keep collecting manual labels.")
        return

    episodes = label_episodes(storage.get_labels())
    groups = [episodes[d] for d, _, _ in meta]
    ec = episode_counts(y, groups)
    acc, preds = episode_accuracy(X, y, groups)
    print("\n=== Leave-one-episode-out verification ===")
    if acc is None:
        print(f"Cannot validate sick-vs-rough separation with {ec['sick']} independent sick episode(s) "
              f"and {ec['rough']} independent rough episode(s).")
        print("Every held-out fold needs both classes in training. Keep collecting independent episodes.")
        return "unvalidated"
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
    if not ready or args.offline:
        print("Insufficient independent episodes/days or offline features for a deployable verdict.")
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
        print("   Keep collecting independent episodes.")
        verdict = "not_separable"
    print()
    print("Per-day predictions (highest strain first):")
    for (date, lbl, feats), (pred, true) in sorted(
            zip(meta, preds), key=lambda x: -x[0][2]["strain_idx"]):
        mark = "✓" if pred == true else "✗"
        print(f"  {date}  actual={lbl:<6} pred={'sick' if pred else 'rough':<6} "
              f"{mark}  strain={feats['strain_idx']:.2f} "
              f"elev={feats['elevated_3d']}/3 rec↓={int(feats['rec_decline'])}")
    return verdict


if __name__ == "__main__":
    main()
