#!/usr/bin/env python3
"""Chronological, read-only evaluation against independent self-report labels.

Illness uses sick labels plus confirmed_sick dates. Strain uses rough/sick
labels. Ring-derived clusters measure agreement only. Only fine labels are
true negatives; unknown days have their own alert rate. No synthetic labels
are substituted when episodes.json is absent.
"""
import argparse
from collections import Counter
from contextlib import closing
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import statistics

SCORED_COLUMNS = ("date", "night_rhr", "sleep_rhr", "avg_sleep_hrv",
                  "temp_deviation", "recovery_index")
EPISODES_FILE = Path(__file__).with_name("episodes.json")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_inputs(db_path, episodes_path=None, include_confounders=False):
    """Read one SQLite transaction without creating tables or migrating."""
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("BEGIN")
        conn.row_factory = sqlite3.Row
        history = [dict(row) for row in conn.execute("SELECT * FROM daily_snapshots ORDER BY date")]
        has_labels = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='daily_labels'").fetchone()
        labels = {row["date"]: row["label"] for row in conn.execute("SELECT date, label FROM daily_labels")} if has_labels else {}
        has_confounders = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='daily_confounders'").fetchone()
        confounders = {row["date"]: {key: row[key] for key in ("alcohol", "exercise", "travel", "sleep_loss")}
                       for row in conn.execute("SELECT * FROM daily_confounders")} if has_confounders else {}
    annotations = {}
    if episodes_path is not None and Path(episodes_path).exists():
        annotations = json.loads(Path(episodes_path).read_text())
    confirmed = set(annotations.get("confirmed_sick", []))
    clusters = set(annotations.get("episodes", []))
    for day in {row["date"] for row in history} | set(labels) | confirmed | clusters:
        date.fromisoformat(day)
    if set(labels.values()) - {"fine", "rough", "sick"}:
        raise ValueError("Labels must be fine, rough, or sick")
    fingerprint = {
        "row_count": len(history),
        "date_range": [history[0]["date"], history[-1]["date"]] if history else [None, None],
        "scored_columns": list(SCORED_COLUMNS),
        "sha256": digest([[row.get(key) for key in SCORED_COLUMNS] for row in history]),
        "labels_sha256": digest(labels),
        "confounders_sha256": digest(confounders),
        "annotations_sha256": digest({"confirmed_sick": sorted(confirmed), "episodes": sorted(clusters)}),
        "annotations_present": bool(episodes_path is not None and Path(episodes_path).exists()),
    }
    inputs = (history, labels, confirmed, clusters, fingerprint)
    return (*inputs, confounders) if include_confounders else inputs


def score_all(history, scorer=None):
    if scorer is None:
        from symptomradar import engine
        scorer = engine.assess_strain_result
    return [{"date": row["date"], **scorer(history[:i + 1])}
            for i, row in enumerate(history)]


def rate(numerator, n):
    return {"numerator": numerator, "n": n, "value": numerator / n if n else None}


def group_episodes(positives):
    """Group calendar positives separated by at most one missing day."""
    groups = []
    for day in sorted(positives):
        if not groups or (date.fromisoformat(day) - date.fromisoformat(groups[-1][-1])).days > 2:
            groups.append([])
        groups[-1].append(day)
    return groups


def section_metrics(scores, labels, positives, interpretation):
    observed = {row["date"] for row in scores}
    assessable = {row["date"] for row in scores if row["status"] == "ok"}
    flagged = {row["date"] for row in scores if row["status"] == "ok" and row["level"] >= 1}
    buffer = {str(date.fromisoformat(day) + timedelta(days=offset))
              for day in positives for offset in (-1, 1)} - positives
    eligible = assessable - buffer
    positive_days = eligible & positives
    fine = eligible & {d for d, label in labels.items() if label == "fine"} - positives
    unknown = eligible - set(labels) - positives
    episodes = []
    unassessable_episodes = 0
    outside_episodes = 0
    for group in group_episodes(positives):
        if not set(group) & observed:
            outside_episodes += 1
            continue
        if not set(group) & assessable:
            unassessable_episodes += 1
            continue
        start, end = date.fromisoformat(group[0]), date.fromisoformat(group[-1])
        # Pre-onset buffers are excluded from day rates, not warning windows.
        hits = sorted(d for d in flagged if start - timedelta(days=2) <= date.fromisoformat(d) <= end)
        episodes.append({"start": str(start), "end": str(end),
                         "positive_days": len(set(group) & eligible),
                         "caught": bool(hits), "first_flagged_date": hits[0] if hits else None,
                         "onset_delay_days": (date.fromisoformat(hits[0]) - start).days if hits else None})
    delays = [e["onset_delay_days"] for e in episodes if e["caught"]]
    return {
        "interpretation": interpretation,
        "positive_days_observed": len(positives & observed),
        "positive_days_outside_input": len(positives - observed),
        "unassessable_days": len(observed - assessable),
        "buffer_days": len(buffer & assessable),
        "eligible_days": len(eligible),
        "other_labeled_days": len(eligible - positive_days - fine - unknown),
        "day_recall": rate(len(positive_days & flagged), len(positive_days)),
        "false_positive_rate": rate(len(fine & flagged), len(fine)),
        "specificity": rate(len(fine - flagged), len(fine)),
        "unknown_alert_rate": rate(len(unknown & flagged), len(unknown)),
        "episode_recall": rate(sum(e["caught"] for e in episodes), len(episodes)),
        "unassessable_episodes": unassessable_episodes,
        "episodes_outside_input": outside_episodes,
        "episodes": episodes,
        "onset_delay": {"n": len(delays),
                        "mean_days": statistics.mean(delays) if delays else None,
                        "median_days": statistics.median(delays) if delays else None,
                        "min_days": min(delays) if delays else None,
                        "max_days": max(delays) if delays else None},
    }


def analyze_separation(scores, labels, buffer):
    groups = {}
    for label in ("fine", "rough", "sick"):
        rows = [r for r in scores if labels.get(r["date"]) == label and
                r["status"] == "ok" and r["date"] not in buffer]
        indices = [r["strain_index"] for r in rows if r["strain_index"] is not None]
        groups[label] = {"n": len(rows), "flag_rate": rate(sum(r["level"] >= 1 for r in rows), len(rows)),
                         "strain_index_n": len(indices),
                         "strain_index_mean": statistics.mean(indices) if indices else None,
                         "strain_index_range": [min(indices), max(indices)] if indices else [None, None]}
    return groups


def confounder_metrics(scores, confounders):
    """Compare explicitly recorded exposure and absence, without causal claims."""
    result = {}
    for key in ("alcohol", "exercise", "travel", "sleep_loss"):
        groups = {"with": [], "without": []}
        unknown = unassessable = 0
        for row in scores:
            value = confounders.get(row["date"], {}).get(key)
            if row["status"] != "ok":
                unassessable += value is not None
            elif value is None:
                unknown += 1
            else:
                groups["with" if value > 0 else "without"].append(row)
        result[key] = {name: rate(sum(row["level"] >= 1 for row in rows), len(rows))
                       for name, rows in groups.items()}
        result[key].update(unrecorded_assessable_days=unknown,
                           unassessable_recorded_days=unassessable)
    return result


def evaluate_database(db_path, episodes_path=None, scorer=None):
    history, labels, confirmed, clusters, fingerprint, confounders = load_inputs(db_path, episodes_path, include_confounders=True)
    scores = score_all(history, scorer)
    illness = confirmed | {d for d, label in labels.items() if label == "sick"}
    strain = {d for d, label in labels.items() if label in ("rough", "sick")}
    definitions = {
        "illness": (illness, "independent ground truth: sick labels plus confirmed_sick dates"),
        "strain": (strain, "independent ground truth: self-reported rough or sick labels"),
        "cluster_agreement": (clusters, "agreement with cluster annotations, not accuracy"),
    }
    buffer = {str(date.fromisoformat(d) + timedelta(days=offset))
              for d in strain for offset in (-1, 1)} - strain
    return {
        "input": fingerprint,
        "policy": {"flag_level": 1, "episode_gap_days": 1, "episode_lead_days": 2,
                   "day_buffer_days": 1, "true_negatives": "fine labels only",
                   "episode_eligibility": "at least one assessable positive day",
                   "onset_delay": "first flag minus onset; negative is lead time; caught episodes only",
                   "warning_window": "includes pre-onset buffer days; flags must have status ok",
                   "scoring": "each day uses only its chronological prefix, including that day"},
        "label_counts": dict(sorted(Counter(labels.values()).items())),
        "label_days_outside_input": len(set(labels) - {row["date"] for row in history}),
        "status_counts": dict(Counter(row["status"] for row in scores)),
        "unassessable_days": sum(row["status"] != "ok" for row in scores),
        "flagged_dates": [r["date"] for r in scores if r["status"] == "ok" and r["level"] >= 1],
        "sections": {key: section_metrics(scores, labels, positive, meaning)
                     for key, (positive, meaning) in definitions.items()},
        "confounders": confounder_metrics(scores, confounders),
        "label_separation": analyze_separation(scores, labels, buffer),
        "scores": scores,
    }


def format_rate(measure):
    value = f"{measure['value']:.3f}" if measure["value"] is not None else "unavailable"
    return f"{measure['numerator']}/{measure['n']} = {value}"


def print_summary(result, labels_only=False):
    fingerprint = result["input"]
    print(f"Input: {fingerprint['row_count']} rows; date range {fingerprint['date_range']}")
    print(f"Scored-column sha256: {fingerprint['sha256']}")
    print(f"Labels sha256: {fingerprint['labels_sha256']}")
    print(f"Annotations sha256: {fingerprint['annotations_sha256']}")
    if not fingerprint["annotations_present"]:
        print("No episodes.json supplied or found; using database labels only.")
    print(f"Labels: {result['label_counts']}; statuses: {result['status_counts']}")
    print(f"Unassessable days: {result['unassessable_days']}; flags: {len(result['flagged_dates'])}")
    if labels_only:
        print("LABEL SEPARATION ANALYSIS (assessable, non-buffer self-report days)")
        for label, group in result["label_separation"].items():
            print(f"{label}: n={group['n']}; flag rate {format_rate(group['flag_rate'])}; "
                  f"strain mean={group['strain_index_mean']}; range={group['strain_index_range']}")
        return
    print("\nCONFOUNDERS (descriptive, assessable days, explicit absence only; not causal)")
    for key, group in result["confounders"].items():
        print(f"{key}: with {format_rate(group['with'])}; without {format_rate(group['without'])}; "
              f"unrecorded={group['unrecorded_assessable_days']}; unassessable recorded={group['unassessable_recorded_days']}")
    for key, section in result["sections"].items():
        print(f"\n{key.upper()}: {section['interpretation']}")
        print(f"Eligible days: {section['eligible_days']}; buffer exclusions: {section['buffer_days']}")
        for metric in ("day_recall", "false_positive_rate", "specificity", "unknown_alert_rate", "episode_recall"):
            name = "cluster-day agreement" if key == "cluster_agreement" and metric == "day_recall" else metric
            print(f"  {name}: {format_rate(section[metric])}")
        print(f"  onset_delay: {section['onset_delay']}")
        print(f"  unassessable episodes: {section['unassessable_episodes']}")
        for episode in section["episodes"]:
            print(f"  episode {episode['start']} to {episode['end']}: "
                  f"caught={episode['caught']}; onset delay={episode['onset_delay_days']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.environ.get("SYMPTOM_RADAR_DB", str(Path(__file__).with_name("ultrahuman.db"))),
                        help="input database, always opened read-only")
    parser.add_argument("--episodes", type=Path, default=EPISODES_FILE, help="personal annotations JSON")
    parser.add_argument("--labels", action="store_true", help="show label separation analysis")
    parser.add_argument("--json", action="store_true", help="print full results as one JSON object")
    args = parser.parse_args()
    result = evaluate_database(args.db, args.episodes)
    if args.json:
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    else:
        print_summary(result, args.labels)


if __name__ == "__main__":
    main()
