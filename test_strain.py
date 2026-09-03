#!/usr/bin/env python3
"""
Unit tests for the Symptom Radar strain math.

Run:  python3 -m pytest test_strain.py -v
  or: python3 test_strain.py            (falls back to a tiny hand-rolled runner)
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import symptom_radar as sr


# ─── compute_zscore ──────────────────────────────────────────────────────────
def test_zscore_basic():
    assert sr.compute_zscore(5, 0, 1) == 5.0

def test_zscore_zero_std_returns_none():
    assert sr.compute_zscore(5, 5, 0) is None

def test_zscore_none_val_returns_none():
    assert sr.compute_zscore(None, 5, 2) is None


# ─── compute_rolling_avg ────────────────────────────────────────────────────
def test_rolling_avg_simple():
    assert sr.compute_rolling_avg([1, 2, 3, 4, 5], window=3) == 4.0

def test_rolling_avg_ignores_none():
    assert sr.compute_rolling_avg([1, None, 3, None, 5], window=2) == 4.0

def test_rolling_avg_insufficient_data_returns_none():
    """Regression: previously fell back to the series mean, masking shifts."""
    assert sr.compute_rolling_avg([10, 11], window=3) is None
    assert sr.compute_rolling_avg([], window=3) is None


# ─── rolling_stats_weighted ─────────────────────────────────────────────────
def test_weighted_stats_minimum_sample():
    assert sr.rolling_stats_weighted([1, 2, 3]) == (None, None)

def test_weighted_stats_basic():
    series = [60.0] * 10
    mean, std = sr.rolling_stats_weighted(series)
    assert mean == 60.0
    assert std is not None and std < 1e-6   # zero variance

def test_weighted_stats_recency_bias():
    """Recent values should pull the weighted mean toward them, away from the
    unweighted mean."""
    old = [50.0] * 10
    recent = [70.0] * 5
    series = old + recent
    unweighted = sum(series) / len(series)   # 56.67
    mean, _ = sr.rolling_stats_weighted(series)
    # With decay=0.9 toward recent (70), the weighted mean should sit clearly
    # above the unweighted mean.
    assert mean > unweighted + 2.0, f"recency bias too weak: weighted={mean}, unweighted={unweighted}"

def test_weighted_stats_bessel_correction_inflates_std():
    """Kish-corrected std should be larger than the naive biased estimate
    for a small sample with nonzero variance."""
    series = [58.0, 60.0, 59.0, 61.0, 60.0, 62.0, 59.0, 61.0]
    mean, std = sr.rolling_stats_weighted(series)
    # Naive population std (uncorrected) on this series is ~1.17; corrected > that.
    naive_var = sum((x - mean) ** 2 for x in series) / len(series)
    naive_std = math.sqrt(naive_var)
    assert std > naive_std, f"corrected std {std} should exceed naive {naive_std}"


# ─── assess_strain: HRV regression ──────────────────────────────────────────
def _flat_history(**today_overrides):
    """21 stable baseline days + a 'today' row, all with normal vitals.

    Baseline: RHR 60, HRV 50, temp_deviation 0.0, recovery 70.
    """
    base = []
    for _ in range(21):
        base.append({
            "night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50,
            "temp_deviation": 0.0, "recovery_index": 70,
        })
    today = {
        "night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50,
        "temp_deviation": 0.0, "recovery_index": 70,
        **today_overrides,
    }
    return base + [today]

def test_no_strain_on_flat_data():
    level, detail = sr.assess_strain(_flat_history())
    assert level == 0

def test_hrv_isolated_drop_is_flagged():
    """REGRESSION: an isolated 3σ HRV crash must register, not be diluted to ~0.5σ.

    Before the fix, max() was taken on raw (negative-when-abnormal) z-scores
    and picked the least-abnormal of single-day vs trend, so an isolated drop
    scored near zero. After the fix it should score at least Minor (>= 1.5).
    """
    # HRV baseline ~50 with some variance; crash today to a low value.
    hist = _flat_history(avg_sleep_hrv_today=20.0) if False else None
    # build a history with real HRV variance so std is nonzero
    base = []
    hrv_vals = [48, 50, 52, 49, 51, 50, 52, 48, 51, 50,
                49, 52, 50, 48, 51, 50, 52, 49, 51, 50, 49]
    for h in hrv_vals:
        base.append({
            "night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": h,
            "temp_deviation": 0.0, "recovery_index": 70,
        })
    today = {
        "night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 20.0,  # big drop
        "temp_deviation": 0.0, "recovery_index": 70,
    }
    level, detail = sr.assess_strain(base + [today])
    assert level >= 1, f"isolated HRV crash should flag >= Minor; got level={level}\n{detail}"
    # And the HRV contribution should be the dominant term, not diluted.
    assert "HRV" in detail

def test_rhr_elevated_is_flagged():
    base = []
    rhr_vals = [58, 60, 59, 61, 60, 59, 60, 61, 59, 60,
                58, 61, 60, 59, 60, 61, 58, 60, 59, 61, 60]
    for r in rhr_vals:
        base.append({
            "night_rhr": r, "sleep_rhr": r, "avg_sleep_hrv": 50,
            "temp_deviation": 0.0, "recovery_index": 70,
        })
    today = {
        "night_rhr": 80, "sleep_rhr": 80, "avg_sleep_hrv": 50,
        "temp_deviation": 0.0, "recovery_index": 70,
    }
    level, detail = sr.assess_strain(base + [today])
    assert level >= 1, f"elevated RHR should flag; got level={level}\n{detail}"

def test_temp_elevated_is_flagged():
    base = []
    for _ in range(21):
        base.append({
            "night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50,
            "temp_deviation": 0.0, "recovery_index": 70,
        })
    today = {
        "night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50,
        "temp_deviation": 1.2,  # +1.2°C
        "recovery_index": 70,
    }
    # A perfectly flat baseline (std=0, MAD=0 — e.g. a stuck sensor) used to
    # SILENCE the metric: z undefined → metric skipped, and a fever-grade
    # +1.2°C spike went completely unflagged. MIN_NOISE floors σ (temp
    # 0.05°C) so the metric stays alive and the spike IS caught. Single-day
    # spike → level 1 (Significant needs 2+ elevated days, by design).
    level, detail = sr.assess_strain(base + [today])
    assert level == 1
    assert "Temp" in detail

def test_insufficient_history():
    level, detail = sr.assess_strain([{"night_rhr": 60}] * 5)
    assert level == 0
    assert "baseline" in detail.lower() or "insufficient" in detail.lower()


# ─── linear_slope ─────────────────────────────────────────────────────────────
def test_slope_flat_is_near_zero():
    assert abs(sr.linear_slope([60, 60, 60, 60, 60])) < 1e-9

def test_slope_rising_is_positive():
    s = sr.linear_slope([58, 59, 60, 61, 62])
    assert s > 0
    assert abs(s - 1.0) < 1e-6   # exactly 1 bpm/day

def test_slope_falling_is_negative():
    assert sr.linear_slope([62, 61, 60, 59, 58]) < 0

def test_slope_too_few_points_returns_none():
    assert sr.linear_slope([1, 2]) is None
    assert sr.linear_slope([]) is None

def test_slope_ignores_none():
    # Gaps shouldn't distort the fitted rate.
    s = sr.linear_slope([58, None, 60, None, 62])
    assert s > 0 and abs(s - 1.0) < 1e-6

def test_slope_dates_prevent_gap_compression():
    # Values rise 0.1 per CALENDAR day, but Jul 26 has no row at all.
    # Index-based x compresses the 3-day span into 2 steps → slope 0.15;
    # calendar-dated x preserves the true rate of 0.10/day.
    vals = [0.1, 0.2, 0.4]
    dates = ["2026-07-24", "2026-07-25", "2026-07-27"]
    assert abs(sr.linear_slope(vals) - 0.15) < 1e-9
    assert abs(sr.linear_slope(vals, dates) - 0.10) < 1e-9
    # Unparseable dates fall back to index spacing without raising.
    assert abs(sr.linear_slope(vals, ["bad", None, "2026-07-27"]) - 0.15) < 1e-9


# ─── MAD / robust noise ───────────────────────────────────────────────────────
def test_median_abs_deviation_basic():
    # values 1..5, median 3, abs devs 2,1,0,1,2 → median 1
    assert sr.median_abs_deviation([1, 2, 3, 4, 5]) == 1

def test_median_abs_deviation_too_few_returns_none():
    assert sr.median_abs_deviation([1, 2, 3]) is None

def test_robust_noise_picks_larger_of_std_and_mad():
    """When a series has an outlier, MAD-scaled should exceed std and win."""
    # Mostly stable with one big outlier → std underestimates typical spread,
    # MAD captures the day-to-day noise better.
    series = [60, 60, 61, 59, 60, 60, 95, 60, 61, 59]
    _, std = sr.rolling_stats_weighted(series)
    mad = sr.median_abs_deviation(series)
    noise = sr._robust_noise(std, mad)
    assert noise is not None
    # MAD-scaled (1.4826 * MAD) should be the floor here
    assert noise >= mad * 1.4826 - 1e-9


# ─── Clean baseline (two-pass exclusion) ──────────────────────────────────────
def test_clean_baseline_excludes_recent_sick_days():
    """A 3-day ramp into illness should NOT contaminate the baseline.

    If the baseline included the rising-sick days, today's score would be
    suppressed (the mean/std would already be inflated). The clean-baseline
    guard (CLEAN_GUARD=3) excludes them.
    """
    base = []
    # 15 stable baseline days with mild variance so std is nonzero
    import random
    random.seed(42)
    for _ in range(15):
        base.append({
            "night_rhr": 60 + random.choice([-1, 0, 1]),
            "sleep_rhr": 60, "avg_sleep_hrv": 50 + random.choice([-2, 0, 2]),
            "temp_deviation": 0.0 + random.choice([-0.1, 0, 0.1]),
            "recovery_index": 70,
        })
    # 3 ramping-up sick days (these should be EXCLUDED from baseline)
    base += [
        {"night_rhr": 64, "sleep_rhr": 64, "avg_sleep_hrv": 45, "temp_deviation": 0.3, "recovery_index": 60},
        {"night_rhr": 68, "sleep_rhr": 68, "avg_sleep_hrv": 40, "temp_deviation": 0.5, "recovery_index": 50},
        {"night_rhr": 72, "sleep_rhr": 72, "avg_sleep_hrv": 35, "temp_deviation": 0.7, "recovery_index": 40},
    ]
    today = {"night_rhr": 75, "sleep_rhr": 75, "avg_sleep_hrv": 32,
             "temp_deviation": 0.9, "recovery_index": 35}
    level, detail = sr.assess_strain(base + [today])
    # Should be flagged strongly — all three metrics rising, sustained.
    assert level == 2, f"sustained illness ramp should hit Significant; got {level}\n{detail}"


# ─── Persistence requirement ─────────────────────────────────────────────────
def test_single_bad_day_does_not_hit_significant():
    """One isolated bad day → at most Elevated, never Significant.

    This is the false-positive guard: a hangover, one rough night of sleep,
    or a hard workout shouldn't fire the highest level.
    """
    base = []
    import random
    random.seed(1)
    for _ in range(18):
        base.append({
            "night_rhr": 60 + random.choice([-1, 0, 1]),
            "sleep_rhr": 60, "avg_sleep_hrv": 50 + random.choice([-2, 0, 2]),
            "temp_deviation": 0.0, "recovery_index": 70,
        })
    # Two normal recent days, then one bad today
    base += [
        {"night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50, "temp_deviation": 0.0, "recovery_index": 70},
        {"night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50, "temp_deviation": 0.0, "recovery_index": 70},
    ]
    today = {"night_rhr": 78, "sleep_rhr": 78, "avg_sleep_hrv": 32,
             "temp_deviation": 0.8, "recovery_index": 35}
    level, detail = sr.assess_strain(base + [today])
    assert level < 2, f"isolated bad day should not hit Significant; got {level}\n{detail}"

def test_sustained_elevation_hits_significant():
    """Three consecutive bad days across multiple metrics → Significant."""
    base = []
    import random
    random.seed(7)
    for _ in range(15):
        base.append({
            "night_rhr": 60 + random.choice([-1, 0, 1]),
            "sleep_rhr": 60, "avg_sleep_hrv": 50 + random.choice([-2, 0, 2]),
            "temp_deviation": 0.0 + random.choice([-0.1, 0, 0.1]),
            "recovery_index": 70,
        })
    # Three sustained bad days
    base += [
        {"night_rhr": 70, "sleep_rhr": 70, "avg_sleep_hrv": 38, "temp_deviation": 0.6, "recovery_index": 45},
        {"night_rhr": 71, "sleep_rhr": 71, "avg_sleep_hrv": 37, "temp_deviation": 0.7, "recovery_index": 42},
        {"night_rhr": 72, "sleep_rhr": 72, "avg_sleep_hrv": 36, "temp_deviation": 0.8, "recovery_index": 40},
    ]
    level, detail = sr.assess_strain(base + [base[-1]])  # today = continuation
    assert level == 2, f"sustained elevation should hit Significant; got {level}\n{detail}"


# ─── hr_drop integration ──────────────────────────────────────────────────────
def test_hr_drop_column_migrates_idempotently(tmp_path=None):
    """init_db() should add hr_drop to an old DB without losing rows, and
    be safe to call repeatedly."""
    import sqlite3, tempfile, os
    with tempfile.TemporaryDirectory() as td:
        dbp = os.path.join(td, "t.db")
        # Build an old-schema table (no hr_drop)
        c = sqlite3.connect(dbp)
        c.execute("""CREATE TABLE daily_snapshots (
            date TEXT PRIMARY KEY, sleep_score REAL, night_rhr REAL,
            created_at TEXT DEFAULT (datetime('now')))""")
        c.execute("INSERT INTO daily_snapshots (date, sleep_score, night_rhr) VALUES ('2024-01-01', 80, 60)")
        c.commit(); c.close()
        # Point the module at it and migrate
        old = sr.DB_PATH
        try:
            sr.DB_PATH = dbp
            conn = sr.init_db()
            cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_snapshots)")}
            assert "hr_drop" in cols
            n = conn.execute("SELECT COUNT(*) FROM daily_snapshots").fetchone()[0]
            assert n == 1   # row preserved
            conn.close()
            # Idempotent: second migration is a no-op
            conn2 = sr.init_db()
            conn2.close()
        finally:
            sr.DB_PATH = old


# ─── One-sided strain direction (HRV high = good, not strain) ────────────────
def test_high_hrv_is_not_strain():
    """REGRESSION: a HIGH-HRV day (60-86ms when baseline ~43) is rest and
    recovery — the opposite of illness — and must NOT contribute strain.

    Before the fix, the inverted logic treated a +4σ HRV spike symmetrically
    as -4σ strain, flagging the healthiest days (e.g. 2024-01-15, HRV 60)."""
    base = []
    hrv_vals = [42, 44, 43, 41, 45, 43, 42, 44, 43, 41,
                45, 43, 42, 44, 43, 41, 45, 43, 42, 44, 43]
    for h in hrv_vals:
        base.append({
            "night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": h,
            "temp_deviation": 0.0, "recovery_index": 70,
        })
    today = {
        "night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 75.0,  # very HIGH
        "temp_deviation": 0.0, "recovery_index": 70,
    }
    level, detail = sr.assess_strain(base + [today])
    assert level == 0, f"high HRV must not flag; got level={level}\n{detail}"
    assert "HRV" not in detail or "↑" not in detail, f"HRV should not contribute: {detail}"


# ─── Mean-reversion gate (recovering temp must not fire) ─────────────────────
def test_temp_recovering_from_low_excursion_does_not_flag():
    """REGRESSION: temp going -0.95 → +0.03 (recovering from a low excursion,
    i.e. mean reversion) must NOT fire as 'rising = strain'. This was the
    exact mechanism that falsely flagged 2024-01-15 on a completely normal
    +0.03°C temperature."""
    base = []
    for _ in range(15):
        base.append({
            "night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50,
            "temp_deviation": 0.0, "recovery_index": 70,
        })
    # Low excursion then recovery to normal
    base += [
        {"night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50, "temp_deviation": -0.95, "recovery_index": 70},
        {"night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50, "temp_deviation": -0.40, "recovery_index": 70},
        {"night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50, "temp_deviation": 0.03,  "recovery_index": 70},
    ]
    today = {"night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50,
             "temp_deviation": 0.03, "recovery_index": 70}
    level, detail = sr.assess_strain(base + [today])
    assert level == 0, f"mean-reversion temp must not flag; got level={level}\n{detail}"


# ─── Recovery-slope leading indicator ────────────────────────────────────────
def test_recovery_decline_warns_before_crash():
    """The recovery-slope warning (Recovery↓) should fire when recovery is
    falling steeply over 3 days — the leading indicator observed 1-2 days
    before every strain episode."""
    base = []
    for _ in range(18):
        base.append({
            "night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50,
            "temp_deviation": 0.0, "recovery_index": 70,
        })
    # Recovery declining steeply: 70 → 60 → 50 → 40
    base += [
        {"night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50, "temp_deviation": 0.0, "recovery_index": 60},
        {"night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50, "temp_deviation": 0.0, "recovery_index": 50},
        {"night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50, "temp_deviation": 0.0, "recovery_index": 40},
    ]
    today = {"night_rhr": 60, "sleep_rhr": 60, "avg_sleep_hrv": 50,
             "temp_deviation": 0.0, "recovery_index": 40}
    level, detail = sr.assess_strain(base + [today])
    # Even without other metrics moving, the recovery decline should appear
    # in the detail as a leading signal (contributes to the index).
    assert "Recovery↓" in detail, f"expected Recovery↓ leading signal: {detail}"


# ─── Daily self-report labels ────────────────────────────────────────────────
def test_label_roundtrip():
    """log_label → get_labels round-trips, and re-logging overwrites."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        old = sr.DB_PATH
        try:
            sr.DB_PATH = os.path.join(td, "t.db")
            sr.log_label("2024-08-01", "fine")
            sr.log_label("2024-08-02", "rough", "hangover")
            sr.log_label("2024-08-03", "sick", "fever")
            labels = sr.get_labels()
            assert [l[0] for l in labels] == ["2024-08-01", "2024-08-02", "2024-08-03"]
            assert [l[1] for l in labels] == ["fine", "rough", "sick"]
            # overwrite
            sr.log_label("2024-08-02", "fine")
            labels = sr.get_labels()
            assert [l[1] for l in labels] == ["fine", "fine", "sick"]
        finally:
            sr.DB_PATH = old

def test_label_invalid_rejected():
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        old = sr.DB_PATH
        try:
            sr.DB_PATH = os.path.join(td, "t.db")
            try:
                sr.log_label("2024-08-01", "mildly_off")
                assert False, "invalid label should raise"
            except ValueError:
                pass
        finally:
            sr.DB_PATH = old


# ─── Classifier (train.py) — synthetic data checks ──────────────────────────
def test_logistic_separates_synthetic():
    """The pure-stdlib logistic regression must separate cleanly separable
    synthetic data (2 sick-like points vs 8 rough-like points, separable by
    one feature). This validates the training tooling itself, independent of
    the (still-too-small) real label set."""
    import train as tr
    # sick: high strain feature (5.0); rough: low strain feature (1.0)
    X = [[5.0, 1.0], [5.5, 1.0]] + [[1.0, 1.0]] * 8
    y = [1, 1] + [0] * 8
    Xn, _ = tr.zscore_cols(X)
    acc, preds = tr.loo_accuracy(Xn, y)
    tp = sum(1 for p, t in preds if p == 1 and t == 1)
    assert tp == 2, f"separator should catch both sick points; got TP={tp}"
    assert acc >= 0.9, f"separator should be near-perfect; got acc={acc}"

def test_logistic_truthful_on_non_separable():
    """On data where sick/rough genuinely overlap, the classifier must NOT
    claim separation — accuracy should be near chance (~50%)."""
    import train as tr
    import random
    rng = random.Random(7)
    # both classes drawn from the SAME distribution → no feature separates them
    X = [[rng.uniform(1.0, 2.0)] for _ in range(20)]
    y = [rng.choice([0, 1]) for _ in range(20)]
    # ensure both classes are present
    if sum(y) == 0 or sum(y) == 20:
        y[0], y[1] = 1, 0
    Xn, _ = tr.zscore_cols(X)
    acc, _ = tr.loo_accuracy(Xn, y)
    # chance-ish accuracy on same-distribution data
    assert acc <= 0.75, f"same-distribution data should score near chance; got {acc}"


# ─── Retrospective label parser (labels.py --from-memory) ───────────────────
def test_bulk_from_memory_parses_and_dedupes():
    """'Jan 15 sick, Feb 18 sick, Mar 26-27 rough' must produce exactly 4
    labels with no duplicates (Mar 26 expanded from both 'Mar 26' and the
    '26-27' range previously). Month-day tokens resolve to the CURRENT
    year (dynamic since the 2027 rollover fix)."""
    import labels, tempfile, os
    from datetime import datetime
    yr = datetime.now().year
    with tempfile.TemporaryDirectory() as td:
        old = sr.DB_PATH
        try:
            sr.DB_PATH = os.path.join(td, "t.db")
            out = labels.bulk_from_memory("Jan 15 sick, Feb 18 sick, Mar 26-27 rough")
            dates = [d for d, _ in out]
            assert dates == [f"{yr}-01-15", f"{yr}-02-18", f"{yr}-03-26", f"{yr}-03-27"], dates
            assert len(out) == 4, f"expected 4, got {len(out)}: {out}"
        finally:
            sr.DB_PATH = old

def test_bulk_from_memory_dont_remember_is_noop():
    import labels, tempfile, os
    with tempfile.TemporaryDirectory() as td:
        old = sr.DB_PATH
        try:
            sr.DB_PATH = os.path.join(td, "t.db")
            out = labels.bulk_from_memory("don't remember")
            assert out == []
        finally:
            sr.DB_PATH = old


# ─── morning_alertness capture ────────────────────────────────────────────────
def test_morning_alertness_extracted():
    metrics = [{"type": "morning_alertness",
                "object": {"value": 48, "unit": "minutes",
                           "status": "calculated"}}]
    assert sr.extract_metric(metrics, "morning_alertness") == {"value": 48}

def test_init_db_migrates_morning_alertness():
    """A database created before the column existed must gain it on init_db()
    without dropping rows."""
    import sqlite3, tempfile
    old = sr.DB_PATH
    try:
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "t.db")
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE daily_snapshots ("
                         "date TEXT PRIMARY KEY, sleep_score REAL)")
            conn.execute("INSERT INTO daily_snapshots VALUES ('2026-01-01', 80)")
            conn.commit(); conn.close()

            sr.DB_PATH = db
            conn = sr.init_db()
            cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_snapshots)")}
            assert "morning_alertness" in cols and "hr_drop" in cols
            assert conn.execute("SELECT sleep_score FROM daily_snapshots").fetchone() == (80,)
            conn.close()
    finally:
        sr.DB_PATH = old

def test_morning_alertness_roundtrip():
    import tempfile
    old = sr.DB_PATH
    try:
        with tempfile.TemporaryDirectory() as td:
            sr.DB_PATH = os.path.join(td, "t.db")
            conn = sr.init_db()
            sr.store_snapshot(conn, "2026-01-01",
                              {"sleep_score": 80, "morning_alertness": 35})
            row = conn.execute("SELECT sleep_score, morning_alertness "
                               "FROM daily_snapshots WHERE date='2026-01-01'").fetchone()
            assert row == (80.0, 35.0)
            conn.close()
    finally:
        sr.DB_PATH = old


# ─── Tiny fallback runner (if pytest isn't installed) ───────────────────────
def _collect():
    fns = [(n, getattr(sys.modules[__name__], n)) for n in dir(sys.modules[__name__])
           if n.startswith("test_") and callable(getattr(sys.modules[__name__], n))]
    return fns

if __name__ == "__main__":
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        fns = _collect()
        passed, failed = 0, 0
        for name, fn in fns:
            try:
                fn()
                print(f"✓ {name}")
                passed += 1
            except Exception as e:
                print(f"✗ {name}: {type(e).__name__}: {e}")
                failed += 1
        print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
        sys.exit(1 if failed else 0)
