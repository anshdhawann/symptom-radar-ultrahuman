#!/usr/bin/env python3
"""Read-only, opt-in stored-signal ablations. No default weights are changed.

All candidates use low values as strain, including morning_alertness (lower
alertness means poorer overnight recovery). Weight 0.10 and noise floor 1.0
are exploratory, fixed before evaluation, not fitted to labels. Missing
observations contribute nothing and cannot make a core-incomplete day normal.
"""
import argparse
import json
import os
from pathlib import Path

import evaluate
from symptomradar import engine, storage

CANDIDATES = {
    key: (key, lambda row, key=key: row.get(key) if storage._valid_metric_value(row.get(key)) else None, 0.10, True, key)
    for key in ("sleep_score", "spo2", "morning_alertness", "hr_drop")
}


def ablate_database(db_path, episodes_path=None):
    history, labels, confirmed, clusters, fingerprint = evaluate.load_inputs(db_path, episodes_path)
    columns = [*evaluate.SCORED_COLUMNS, *CANDIDATES]
    input_sha256 = evaluate.digest({
        "columns": columns,
        "rows": [[row.get(key) for key in columns] for row in history],
        "ground_truth": fingerprint,
    })
    default = evaluate.evaluate_database(db_path, episodes_path)
    results = {}
    for key, metric in CANDIDATES.items():
        observed = {row["date"] for row in history if storage._valid_metric_value(row.get(key))}
        entry = {"coverage": evaluate.rate(len(observed), len(history)),
                 "weight": metric[2], "direction": "low is strain", "noise_floor": 1.0}
        results[key] = entry
        if not observed:
            entry["skipped"] = "No stored observations; cannot estimate a baseline or contribution"
            continue
        def scorer(rows, metric=metric):
            return engine.assess_strain_result(rows, metrics=[*engine.STRAIN_METRICS, metric])
        result = evaluate.evaluate_database(db_path, episodes_path, scorer=scorer)
        if result["input"] != default["input"]:
            raise ValueError("Input changed during ablation; repeat on a frozen snapshot")
        usable = {row["date"] for row in result["scores"]
                  if row["status"] == "ok" and row["metrics"].get(key, {}).get("contribution") is not None}
        entry.update(evaluation=result, usable_days=len(usable),
                     added_flags=sorted(set(result["flagged_dates"]) - set(default["flagged_dates"])),
                     removed_flags=sorted(set(default["flagged_dates"]) - set(result["flagged_dates"])))
        positives = {"illness": confirmed | {d for d, v in labels.items() if v == "sick"},
                     "strain": {d for d, v in labels.items() if v in ("rough", "sick")},
                     "cluster_agreement": clusters}
        entry["paired_usable_sections"] = {
            section: {name: evaluate.section_metrics([r for r in evaluation["scores"] if r["date"] in usable],
                                                     labels, positive, "paired usable days only")
                      for name, evaluation in (("default", default), ("added", result))}
            for section, positive in positives.items()
        }
    final_history, _, _, _, final_fingerprint = evaluate.load_inputs(db_path, episodes_path)
    final_sha256 = evaluate.digest({"columns": columns,
                                   "rows": [[row.get(key) for key in columns] for row in final_history],
                                   "ground_truth": final_fingerprint})
    if final_sha256 != input_sha256:
        raise ValueError("Input changed during ablation; repeat on a frozen snapshot")
    return {"input_sha256": input_sha256, "scored_columns": columns, "default": default,
            "candidates": results,
            "caution": "Exploratory only. Sparse independent episodes cannot establish accuracy gain."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.environ.get("SYMPTOM_RADAR_DB", str(Path(__file__).with_name("ultrahuman.db"))))
    parser.add_argument("--episodes", type=Path, default=evaluate.EPISODES_FILE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = ablate_database(args.db, args.episodes)
    if args.json:
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return
    print(f"Same-input fingerprint (all candidate columns): {result['input_sha256']}")
    print(result["caution"])
    print("DEFAULT")
    evaluate.print_summary(result["default"])
    for key, entry in result["candidates"].items():
        print(f"\n{key}: coverage {evaluate.format_rate(entry['coverage'])}; "
              f"{entry['direction']}; weight={entry['weight']}; noise floor={entry['noise_floor']}")
        if "skipped" in entry:
            print(f"Skipped: {entry['skipped']}")
            continue
        print(f"Usable days: {entry['usable_days']}; added flags={len(entry['added_flags'])}; removed flags={len(entry['removed_flags'])}")
        evaluate.print_summary(entry["evaluation"])
        print("PAIRED USABLE DAYS (same candidate-usable dates on both sides)")
        for section, pair in entry["paired_usable_sections"].items():
            for name, metrics in pair.items():
                print(section, name, {key: metrics[key] for key in
                      ("day_recall", "false_positive_rate", "unknown_alert_rate", "episode_recall", "onset_delay")})


if __name__ == "__main__":
    main()
