import os
import sqlite3
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("ULTRAHUMAN_TOKEN", "dummy")
import symptom_radar as sr
import train as tr
import labels


def history(n=80):
    return [dict(date=(date(2020, 1, 1) + timedelta(days=i)).isoformat(),
                 night_rhr=60 + i % 3, avg_sleep_hrv=50 + i % 3,
                 temp_deviation=0.1 * (i % 3), recovery_index=80,
                 sleep_score=90) for i in range(n)]


def test_day_zero_has_no_baseline():
    assert tr.extract_features(history(), 0) is None


def test_future_marker_cannot_supply_early_baseline():
    h = history()
    for row in h[1:]:
        row['night_rhr'] = 99999
    assert tr.extract_features(h, 1) is None
    h[30]['night_rhr'] = 60
    assert tr.extract_features(h, 30)['rhr_z'] < -1000


def test_live_features_ignore_future_recovery():
    h = history()
    feats = tr.extract_features(h, 30)
    assert [k for k in tr.FEATURE_ORDER if k in feats] == [
        'strain_idx', 'elevated_3d', 'rec_decline', 'rhr_z', 'hrv_z', 'temp_z', 'sleep_z']
    assert 'rec_rebound' not in feats
    h[31]['recovery_index'] = 99999
    assert tr.extract_features(h, 30) == feats
    assert tr.extract_features(h, 30, offline=True)['rec_rebound'] == 9991.9


def test_fold_normalization_excludes_held_out_extreme():
    xtr, xte, stats = tr.normalize_fold([[1.0], [3.0], [100000.0]], [0, 1], [2])
    assert stats == ([2.0], [1.0])
    assert xtr == [[-1.0], [1.0]]
    assert xte == [[99998.0]]


def test_episodes_allow_gap_but_not_other_label():
    records = [('2020-01-01', 'sick', ''), ('2020-01-03', 'sick', ''),
               ('2020-01-04', 'fine', ''), ('2020-01-05', 'sick', ''),
               ('2020-01-08', 'rough', '')]
    assert tr.label_episodes(records) == {
        '2020-01-01': '2020-01-01', '2020-01-03': '2020-01-01',
        '2020-01-04': '2020-01-04', '2020-01-05': '2020-01-05',
        '2020-01-08': '2020-01-08'}


def test_episode_validation_does_not_hold_out_single_days():
    acc, preds = tr.episode_accuracy([[0], [0], [0], [0], [0], [0]],
                                    [1, 1, 1, 0, 0, 0], ['a', 'a', 'b', 'c', 'c', 'd'])
    assert acc == 0.0
    assert preds == [(0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0)]


def test_one_sick_episode_is_not_validatable():
    assert tr.episode_accuracy([[0], [1], [2]], [1, 0, 0], ['a', 'b', 'c']) == (None, [])


def cli(db, *args):
    return subprocess.run([sys.executable, str(Path(__file__).with_name('labels.py')), *args],
                          env={**os.environ, 'SYMPTOM_RADAR_DB': str(db), 'ULTRAHUMAN_TOKEN': 'dummy'},
                          capture_output=True, text=True)


def setup_db(tmp_path, monkeypatch):
    db = tmp_path / 'labels.db'
    monkeypatch.setattr(sr, 'DB_PATH', str(db))
    sr.init_db().close()
    return db


def test_cli_lists_real_labels_not_seeds(tmp_path, monkeypatch):
    db = setup_db(tmp_path, monkeypatch)
    sr.log_label('2020-02-02', 'sick', 'actual second note')
    sr.log_label('2020-02-01', 'rough', 'actual first note')
    r = cli(db)
    assert r.returncode == 0
    assert 'actual first note' in r.stdout and 'actual second note' in r.stdout
    assert r.stdout.index('2020-02-01') < r.stdout.index('2020-02-02')
    assert '2024-03-10' not in r.stdout


def test_seed_removed_without_writing_labels(tmp_path, monkeypatch):
    db = setup_db(tmp_path, monkeypatch)
    r = cli(db, '--seed')
    assert r.returncode == 2
    assert sr.get_labels() == []
    assert 'unrecognized arguments: --seed' in r.stderr


def test_memory_rejects_invalid_date_and_month_without_partial_write(tmp_path, monkeypatch):
    setup_db(tmp_path, monkeypatch)
    assert labels.bulk_from_memory('2020-02-01 sick, 2020-02-31 rough') == []
    assert labels.bulk_from_memory('xyz 22 sick') == []
    assert sr.get_labels() == []


def test_episode_gate_rejects_many_days_in_one_episode(tmp_path, monkeypatch, capsys):
    setup_db(tmp_path, monkeypatch)
    h = history(120)
    for i in range(30, 45):
        sr.log_label(h[i]['date'], 'sick', 'synthetic sick fixture')
    for i in range(60, 75):
        sr.log_label(h[i]['date'], 'rough', 'synthetic rough fixture')
    ready, counts = tr.status_report(h)
    assert ready is False
    assert counts == {'sick': 15, 'rough': 15, 'fine': 0}
    out = capsys.readouterr().out
    assert 'Independent assessable episodes: sick=1, rough=1' in out
    assert 'Need 4 more sick episodes + 4 more rough episodes.' in out


def test_episode_gate_accepts_five_episodes_per_class(tmp_path, monkeypatch, capsys):
    setup_db(tmp_path, monkeypatch)
    h = history(120)
    for start in range(20, 70, 10):
        for i in range(start, start + 3):
            sr.log_label(h[i]['date'], 'sick', 'synthetic sick fixture')
        for i in range(start + 5, start + 8):
            sr.log_label(h[i]['date'], 'rough', 'synthetic rough fixture')
    ready, counts = tr.status_report(h)
    assert ready is True
    assert counts == {'sick': 15, 'rough': 15, 'fine': 0}
    out = capsys.readouterr().out
    assert 'Independent assessable episodes: sick=5, rough=5' in out
    assert 'Ready to train: collection gate met.' in out


def test_memory_valid_range_writes_actual_labels(tmp_path, monkeypatch):
    setup_db(tmp_path, monkeypatch)
    year = labels.datetime.now().year
    assert labels.bulk_from_memory('Jan 3-4 sick') == [
        (f'{year}-01-03', 'sick'), (f'{year}-01-04', 'sick')]
    assert sr.get_labels() == [
        (f'{year}-01-03', 'sick', 'remembered retrospectively'),
        (f'{year}-01-04', 'sick', 'remembered retrospectively')]


def test_offline_cli_is_explicit_and_warns(tmp_path, monkeypatch):
    db = setup_db(tmp_path, monkeypatch)
    for script in ('train.py', 'scenario.py'):
        path = str(Path(__file__).with_name(script))
        env = {**os.environ, 'SYMPTOM_RADAR_DB': str(db), 'ULTRAHUMAN_TOKEN': 'dummy'}
        live = subprocess.run([sys.executable, path], capture_output=True, text=True, env=env)
        offline = subprocess.run([sys.executable, path, '--offline'], capture_output=True, text=True, env=env)
        assert live.returncode == offline.returncode == 0
        assert 'WARNING: --offline uses next-day recovery; results are not deployable.' in offline.stdout
        assert 'WARNING: --offline' not in live.stdout


def test_review_excludes_actual_labels(tmp_path, monkeypatch, capsys):
    db = setup_db(tmp_path, monkeypatch)
    with sqlite3.connect(db) as conn:
        for row in history():
            if row['date'] >= '2020-02-01':
                row['night_rhr'] = 100
            keys = list(row)
            conn.execute(f"INSERT INTO daily_snapshots ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})", list(row.values()))
    sr.log_label('2020-02-01', 'rough', 'real label')
    labels.review_candidates()
    out = capsys.readouterr().out
    assert '2020-02-01' not in out
    assert '2020-02-02' in out
