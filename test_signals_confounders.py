"""Offline public-surface contracts for experiments and exposure logging."""
import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys

import pytest

os.environ.setdefault("ULTRAHUMAN_TOKEN", "dummy")
from symptomradar import config, engine, mcp, storage
import evaluate


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "input.db"
    monkeypatch.setattr(storage, "DB_PATH", str(path))
    return path


def cli(db, *args, script="symptom_radar.py"):
    return subprocess.run([sys.executable, script, *args], capture_output=True, text=True,
                          env={**os.environ, "ULTRAHUMAN_TOKEN": "dummy", "SYMPTOM_RADAR_DB": str(db)})


def test_base_schema_migration_and_independent_records(db):
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE daily_snapshots (
            date TEXT PRIMARY KEY, sleep_score REAL, total_sleep_min REAL,
            sleep_efficiency REAL, deep_sleep_min REAL, light_sleep_min REAL,
            rem_sleep_min REAL, temp_deviation REAL, avg_body_temp REAL,
            night_rhr REAL, sleep_rhr REAL, avg_sleep_hrv REAL, recovery_index REAL,
            movement_index REAL, active_minutes REAL, inactive_time REAL,
            total_steps REAL, vo2_max REAL, spo2 REAL, tosses_and_turns REAL,
            full_sleep_cycles REAL, restorative_sleep REAL, hr_drop REAL,
            morning_alertness REAL, created_at TEXT DEFAULT (datetime('now'))
        )""")
        conn.execute("INSERT INTO daily_snapshots (date, night_rhr) VALUES ('2024-01-01',60)")
        conn.execute("CREATE TABLE daily_labels (date TEXT PRIMARY KEY, label TEXT NOT NULL CHECK(label IN ('fine','rough','sick')), note TEXT, created_at TEXT)")
        conn.execute("INSERT INTO daily_labels VALUES ('2024-01-01','fine','original','timestamp')")
        original_schema = conn.execute("SELECT sql FROM sqlite_master WHERE name='daily_labels'").fetchone()[0]
    for _ in range(2):
        storage.init_db().close()
    storage.log_label("2024-01-02", None, confounders="alcohol=3,exercise,travel,sleep_loss")
    storage.log_label("2024-01-02", None, confounders={"alcohol": 0})
    assert storage.get_labels() == [("2024-01-01", "fine", "original")]
    assert storage.get_confounders() == {"2024-01-02": {"alcohol": 0, "exercise": True, "travel": True, "sleep_loss": True}}
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT created_at FROM daily_labels").fetchone() == ("timestamp",)
        assert conn.execute("SELECT date,night_rhr FROM daily_snapshots").fetchall() == [("2024-01-01", 60.0)]
        assert conn.execute("SELECT sql FROM sqlite_master WHERE name='daily_labels'").fetchone()[0] == original_schema


@pytest.mark.parametrize("value", ["unknown", "alcohol=-1", "alcohol=1.5", "alcohol=x", "alcohol", "exercise=maybe", {"alcohol": True}, {"alcohol": 1.0}, {"travel": 1}, {"unknown": True}])
def test_invalid_confounders_rejected_before_label_write(db, value):
    with pytest.raises(ValueError, match="confounder|alcohol|exercise|travel"):
        storage.log_label("2024-01-01", "fine", confounders=value)
    assert not db.exists()


def test_cli_combined_and_confounder_only_and_listing(db):
    result = cli(db, "--label", "rough", "--label-date", "2024-01-01", "--confounders", "alcohol=3,exercise")
    assert result.returncode == 0, result.stderr
    result = cli(db, "--label-date", "2024-01-02", "--confounders", "travel,sleep_loss")
    assert result.returncode == 0, result.stderr
    listing = cli(db, script="labels.py")
    assert listing.returncode == 0
    assert "alcohol=3" in listing.stdout and "exercise=true" in listing.stdout
    assert "2024-01-02" in listing.stdout and "unlabeled" in listing.stdout
    assert "travel=true" in cli(db, "--labels").stdout
    bad = cli(db, "--confounders", "alcohol=-1")
    assert bad.returncode == 2 and "alcohol" in bad.stderr


def test_mcp_old_and_new_fields(db):
    old, error = mcp._mcp_handle("tools/call", {"name": "symptom_radar_label", "arguments": {"date": "2024-01-01", "label": "fine"}})
    assert error is None and json.loads(old["content"][0]["text"])["label"] == "fine"
    result, error = mcp._mcp_handle("tools/call", {"name": "symptom_radar_label", "arguments": {"date": "2024-01-02", "alcohol": 2, "exercise": True, "travel": False, "sleep_loss": True}})
    assert error is None and json.loads(result["content"][0]["text"])["confounders"]["alcohol"] == 2
    result, error = mcp._mcp_handle("tools/call", {"name": "symptom_radar_label", "arguments": {"confounders": "bad"}})
    assert result is None and "confounder" in error
    result, error = mcp._mcp_handle("tools/call", {"name": "symptom_radar_label", "arguments": {"label": "fine", "alcohol": -2}})
    assert result is None and "alcohol" in error


def history():
    return [{"date": str(dt.date(2024, 1, 1) + dt.timedelta(days=i)), "night_rhr": 60,
             "avg_sleep_hrv": 50, "temp_deviation": 0, "sleep_score": 80, "spo2": 98,
             "morning_alertness": 90} for i in range(30)]


@pytest.mark.parametrize("key", ["sleep_score", "spo2", "morning_alertness"])
def test_extra_metric_low_direction_missing_and_default_unchanged(key):
    import ablate
    rows = history()
    rows[-1][key] -= 20
    default = engine.assess_strain_result(rows)
    assert default["strain_index"] == 0.0 and default["level"] == 0
    added = engine.assess_strain_result(rows, metrics=engine.STRAIN_METRICS + [ablate.CANDIDATES[key]])
    assert added["metrics"][key]["contribution"] == 2.0
    assert added["level"] == 1
    rows[-1][key] = None
    assert engine.assess_strain_result(rows, metrics=engine.STRAIN_METRICS + [ablate.CANDIDATES[key]])["strain_index"] == 0.0
    rows[-1][key] = 100
    assert engine.assess_strain_result(rows, metrics=engine.STRAIN_METRICS + [ablate.CANDIDATES[key]])["strain_index"] == 0.0


def test_ablation_coverage_fingerprint_and_unassessable(db):
    import ablate
    conn = storage.init_db()
    for row in history():
        conn.execute("INSERT INTO daily_snapshots (date,night_rhr,avg_sleep_hrv,temp_deviation,sleep_score) VALUES (?,?,?,?,?)", (row["date"],60,50,0,80))
    conn.commit()
    conn.close()
    result = ablate.ablate_database(db)
    assert result["candidates"]["sleep_score"]["coverage"] == {"numerator": 30, "n": 30, "value": 1.0}
    assert result["candidates"]["hr_drop"]["skipped"] == "No stored observations; cannot estimate a baseline or contribution"
    assert result["candidates"]["sleep_score"]["usable_days"] == 21
    assert result["candidates"]["sleep_score"]["evaluation"]["input"] == result["default"]["input"]
    fingerprint = result["input_sha256"]
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE daily_snapshots SET sleep_score=79 WHERE date='2024-01-30'")
    assert ablate.ablate_database(db)["input_sha256"] != fingerprint


def test_confounder_rates_exclude_unrecorded_and_unassessable(db):
    conn = storage.init_db()
    conn.executemany("INSERT INTO daily_snapshots (date,night_rhr) VALUES (?,?)", [("2024-01-01",1), ("2024-01-02",0), ("2024-01-03",1), ("2024-01-04",None)])
    conn.commit()
    conn.close()
    storage.log_label("2024-01-01", None, confounders="alcohol=3,exercise")
    storage.log_label("2024-01-02", None, confounders="alcohol=0,exercise=false")
    storage.log_label("2024-01-04", None, confounders="alcohol=3")
    def scorer(rows):
        val = rows[-1]["night_rhr"]
        return {"status": "ok" if val is not None else "no_data", "level": val, "strain_index": val}
    result = evaluate.evaluate_database(db, scorer=scorer)
    alcohol = result["confounders"]["alcohol"]
    assert alcohol["with"] == {"numerator": 1, "n": 1, "value": 1.0}
    assert alcohol["without"] == {"numerator": 0, "n": 1, "value": 0.0}
    assert alcohol["unrecorded_assessable_days"] == 1
    assert alcohol["unassessable_recorded_days"] == 1
    assert result["confounders"]["travel"]["without"]["n"] == 0


def test_exposures_preserve_label_and_partial_updates(db):
    storage.log_label("2024-01-01", "sick", "fever", confounders="travel")
    storage.log_label("2024-01-01", None, confounders="alcohol=0,exercise=false,sleep_loss=false")
    storage.log_label("2024-01-01", None, confounders="alcohol=0,exercise=false,sleep_loss=false")
    assert storage.get_labels() == [("2024-01-01", "sick", "fever")]
    assert storage.get_confounders() == {"2024-01-01": {"alcohol": 0, "exercise": False, "travel": True, "sleep_loss": False}}


@pytest.mark.parametrize("arguments", [{"confounders": "exercise,exercise"}, {"confounders": "alcohol=2", "alcohol": 3}, {"unknown": True}, {}])
def test_mcp_boundary_rejects_duplicates_unknown_and_empty(db, arguments):
    result, error = mcp._mcp_handle("tools/call", {"name": "symptom_radar_label", "arguments": arguments})
    assert result is None
    assert error in ("Duplicate confounder: exercise", "Duplicate confounder fields", "Unknown label or confounder fields: ['unknown']", "Provide a label or confounders")
    assert not db.exists()


def test_extra_metrics_do_not_replace_core_coverage_or_baseline():
    import ablate
    rows = history()
    for row in rows:
        row["night_rhr"] = row["avg_sleep_hrv"] = row["temp_deviation"] = None
    result = engine.assess_strain_result(rows, metrics=[*engine.STRAIN_METRICS, ablate.CANDIDATES["sleep_score"]])
    assert result["status"] == "no_data" and result["level"] is None
    rows = history()
    for row in rows[:-1]:
        row["sleep_score"] = None
    rows[-1]["sleep_score"] = 10
    result = engine.assess_strain_result(rows, metrics=[*engine.STRAIN_METRICS, ablate.CANDIDATES["sleep_score"]])
    assert result["level"] == 0 and result["metrics"]["sleep_score"]["contribution"] is None


def test_ablation_and_evaluation_cli_are_read_only(db):
    import hashlib
    conn = storage.init_db()
    for row in history():
        conn.execute("INSERT INTO daily_snapshots (date,night_rhr,avg_sleep_hrv,temp_deviation,sleep_score) VALUES (?,?,?,?,?)", (row["date"],60,50,0,80))
    conn.commit()
    conn.close()
    storage.log_label("2024-01-30", "fine", confounders="exercise")
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    result = cli(db, "--db", str(db), script="ablate.py")
    assert result.returncode == 0, result.stderr
    assert "sleep_score: coverage 30/30 = 1.000" in result.stdout
    assert "hr_drop: coverage 0/30 = 0.000" in result.stdout
    assert "Skipped: No stored observations" in result.stdout
    result = cli(db, "--db", str(db), "--json", script="ablate.py")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["candidates"]["sleep_score"]["paired_usable_sections"]["illness"]["default"]["false_positive_rate"]["n"] == 1
    result = cli(db, "--db", str(db), script="evaluate.py")
    assert result.returncode == 0, result.stderr
    assert "exercise: with 0/1 = 0.000; without 0/0 = unavailable" in result.stdout
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("rhr, expected", [(60, "✅ Normal"), (65, "🟡 Elevated"), (80, "🔴 Significant strain")])
def test_no_argument_report_markers_with_mocked_fetch(db, monkeypatch, capsys, rhr, expected):
    import runpy
    from unittest.mock import patch
    today = config.wearer_now().date()
    conn = storage.init_db()
    for i in range(30):
        day = str(today - dt.timedelta(days=29-i))
        conn.execute("INSERT INTO daily_snapshots (date,night_rhr,avg_sleep_hrv,temp_deviation) VALUES (?,?,50,0)", (day, rhr if i >= 27 else 60))
    conn.commit()
    conn.close()
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"data": {"metrics": {str(today): [
                {"type": "night_rhr", "object": {"avg": rhr}},
                {"type": "avg_sleep_hrv", "object": {"value": 50}},
            ]}}}
    monkeypatch.setenv("SYMPTOM_RADAR_DB", str(db))
    monkeypatch.setattr(sys, "argv", ["symptom_radar.py"])
    with patch("requests.get", return_value=Response()):
        runpy.run_path(str(__import__("pathlib").Path(__file__).with_name("symptom_radar.py")), run_name="__main__")
    output = capsys.readouterr().out
    assert expected in output
    print(next(line for line in output.splitlines() if expected in line))


def test_stdio_mcp_confounder_logging_and_schema(db):
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "symptom_radar_label", "arguments": {"date": "2024-01-01", "confounders": "alcohol=2,travel"}}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "symptom_radar_label", "arguments": {"label": "fine", "alcohol": 1.5}}},
    ]
    result = subprocess.run([sys.executable, "symptom_radar.py", "--mcp"],
                            input="\n".join(json.dumps(req) for req in requests) + "\n",
                            text=True, capture_output=True,
                            env={**os.environ, "ULTRAHUMAN_TOKEN": "dummy", "SYMPTOM_RADAR_DB": str(db)})
    assert result.returncode == 0, result.stderr
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    tools = {tool["name"]: tool for tool in responses[0]["result"]["tools"]}
    assert {"symptom_radar_report", "symptom_radar_history", "symptom_radar_strain", "symptom_radar_label", "symptom_radar_labels"} <= tools.keys()
    assert {"required": ["label"]} in tools["symptom_radar_label"]["inputSchema"]["anyOf"]
    assert json.loads(responses[1]["result"]["content"][0]["text"])["confounders"] == {"alcohol": 2, "travel": True}
    assert responses[2]["error"]["message"] == "alcohol must be a non-negative integer drink count"
    assert storage.get_confounders() == {"2024-01-01": {"alcohol": 2, "exercise": None, "travel": True, "sleep_loss": None}}
