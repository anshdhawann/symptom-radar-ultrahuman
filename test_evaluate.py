"""Synthetic offline evaluation contracts."""
import datetime as dt
import hashlib
import json
import os
import sqlite3
import subprocess
import sys

os.environ.setdefault("ULTRAHUMAN_TOKEN", "dummy")
import evaluate


def database(tmp_path, days, labels=None):
    path = tmp_path / "input.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE daily_snapshots (date TEXT PRIMARY KEY, night_rhr REAL, sleep_rhr REAL, avg_sleep_hrv REAL, temp_deviation REAL, recovery_index REAL)")
        conn.executemany("INSERT INTO daily_snapshots VALUES (?, ?, NULL, 50, 0, 80)",
                         [(date, level) for date, level in days])
        conn.execute("CREATE TABLE daily_labels (date TEXT PRIMARY KEY, label TEXT)")
        conn.executemany("INSERT INTO daily_labels VALUES (?, ?)", (labels or {}).items())
    return path


def score(history):
    value = history[-1]["night_rhr"]
    status = {None: "no_data", -1: "calibrating", -2: "insufficient_data"}.get(value, "ok")
    return {"status": status, "level": int(value) if status == "ok" else None,
            "strain_index": value if status == "ok" else None,
            "elevated_recent": 0, "coverage": [], "metrics": {},
            "baseline_days": 7, "detail": "Unparseable display text"}


def run(tmp_path, days, labels=None, annotations=None, scorer=score):
    path = database(tmp_path, days, labels)
    episodes = tmp_path / "episodes.json"
    episodes.write_text(json.dumps(annotations or {}))
    return evaluate.evaluate_database(path, episodes, scorer=scorer)


def test_unknown_is_not_a_true_negative_and_ground_truth_is_separate(tmp_path):
    result = run(tmp_path, [("2024-01-01", 1), ("2024-01-05", 0),
                            ("2024-01-09", 1), ("2024-01-13", 1),
                            ("2024-01-17", 0), ("2024-01-21", 1)],
                 {"2024-01-01": "fine", "2024-01-05": "fine", "2024-01-09": "rough",
                  "2024-01-13": "sick"},
                 {"confirmed_sick": ["2024-01-17"], "episodes": ["2024-01-21"]})
    illness = result["sections"]["illness"]
    strain = result["sections"]["strain"]
    assert illness["false_positive_rate"] == {"numerator": 1, "n": 2, "value": 0.5}
    assert illness["specificity"] == {"numerator": 1, "n": 2, "value": 0.5}
    assert illness["day_recall"] == {"numerator": 1, "n": 2, "value": 0.5}
    assert strain["day_recall"] == {"numerator": 2, "n": 2, "value": 1.0}
    assert illness["unknown_alert_rate"] == {"numerator": 1, "n": 1, "value": 1.0}
    assert result["sections"]["cluster_agreement"]["interpretation"] == "agreement with cluster annotations, not accuracy"


def test_buffer_exclusion_is_symmetric(tmp_path):
    result = run(tmp_path, [("2024-01-01", 1), ("2024-01-02", 1),
                            ("2024-01-03", 0), ("2024-01-06", 0)],
                 {"2024-01-01": "fine", "2024-01-02": "sick",
                  "2024-01-03": "fine", "2024-01-06": "fine"})
    section = result["sections"]["illness"]
    assert section["buffer_days"] == 2
    assert section["false_positive_rate"] == {"numerator": 0, "n": 1, "value": 0.0}
    assert section["day_recall"] == {"numerator": 1, "n": 1, "value": 1.0}


def test_unassessable_days_are_excluded_from_every_rate(tmp_path):
    result = run(tmp_path, [("2024-01-01", -1), ("2024-01-05", None),
                            ("2024-01-09", -2), ("2024-01-13", 1),
                            ("2024-01-17", 0), ("2024-01-21", 1)],
                 {"2024-01-01": "sick", "2024-01-05": "fine",
                  "2024-01-13": "sick", "2024-01-17": "fine"})
    assert result["unassessable_days"] == 3
    assert result["status_counts"] == {"calibrating": 1, "no_data": 1, "insufficient_data": 1, "ok": 3}
    section = result["sections"]["illness"]
    assert section["day_recall"] == {"numerator": 1, "n": 1, "value": 1.0}
    assert section["false_positive_rate"] == {"numerator": 0, "n": 1, "value": 0.0}
    assert section["unknown_alert_rate"] == {"numerator": 1, "n": 1, "value": 1.0}
    assert section["episode_recall"] == {"numerator": 1, "n": 1, "value": 1.0}
    assert section["unassessable_episodes"] == 1


def test_episode_window_one_day_gap_and_signed_onset_delay(tmp_path):
    dates = [str(dt.date(2024, 1, 1) + dt.timedelta(days=i)) for i in range(20)]
    flags = {"2024-01-02", "2024-01-12", "2024-01-19"}
    positives = {"2024-01-04", "2024-01-06", "2024-01-11", "2024-01-12", "2024-01-18"}
    result = run(tmp_path, [(d, int(d in flags)) for d in dates],
                 {d: "sick" for d in positives})
    section = result["sections"]["illness"]
    assert section["episode_recall"] == {"numerator": 2, "n": 3, "value": 2 / 3}
    assert [e["onset_delay_days"] for e in section["episodes"]] == [-2, 1, None]
    assert section["onset_delay"] == {"n": 2, "mean_days": -0.5, "median_days": -0.5, "min_days": -2, "max_days": 1}


def test_prefix_scoring_has_no_lookahead(tmp_path):
    def prefix_score(history):
        result = score(history)
        result["level"] = int(len(history) == 2 and history[-1]["date"] == "2024-01-05")
        return result
    result = run(tmp_path, [("2024-01-01", 0), ("2024-01-05", 0), ("2024-01-09", 0)],
                 {"2024-01-05": "sick"}, scorer=prefix_score)
    assert result["sections"]["illness"]["day_recall"] == {"numerator": 1, "n": 1, "value": 1.0}
    assert result["flagged_dates"] == ["2024-01-05"]


def test_cli_read_only_json_and_labels_never_migrates(tmp_path):
    path = tmp_path / "minimal.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE daily_snapshots (date TEXT PRIMARY KEY, night_rhr REAL)")
        conn.execute("INSERT INTO daily_snapshots VALUES ('2024-01-01', 60)")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    for extra in ([], ["--labels"]):
        proc = subprocess.run([sys.executable, evaluate.__file__, "--db", str(path), "--json", *extra],
                              text=True, capture_output=True)
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout)
        assert result["input"]["row_count"] == 1
        assert result["unassessable_days"] == 1
        assert result["sections"]["illness"]["false_positive_rate"] == {"numerator": 0, "n": 0, "value": None}
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("daily_snapshots",)]
    missing = tmp_path / "absent.db"
    proc = subprocess.run([sys.executable, evaluate.__file__, "--db", str(missing), "--json"], capture_output=True)
    assert proc.returncode != 0
    assert not missing.exists()


def test_fingerprint_covers_scored_columns_and_not_unscored_columns(tmp_path):
    path = database(tmp_path, [("2024-01-01", 0)])
    first = evaluate.evaluate_database(path, scorer=score)["input"]
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE daily_snapshots ADD COLUMN ignored TEXT")
        conn.execute("UPDATE daily_snapshots SET ignored='unscored'")
    assert evaluate.evaluate_database(path, scorer=score)["input"] == first
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE daily_snapshots SET recovery_index=79")
    assert evaluate.evaluate_database(path, scorer=score)["input"]["sha256"] != first["sha256"]
    assert first["date_range"] == ["2024-01-01", "2024-01-01"]
