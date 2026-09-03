#!/usr/bin/env python3
"""
Symptom Radar for Ultrahuman Ring

A TemPredict-inspired anomaly detection system that monitors your biometric
data (RHR, HRV, skin temperature) and flags early signs of physiological strain.

Uses a 21-day rolling z-score baseline — same approach as Oura's Symptom Radar.

Requires: ULTRAHUMAN_TOKEN environment variable
Optional: SYMPTOM_RADAR_DB path (defaults to ./ultrahuman.db)

Usage:
    export ULTRAHUMAN_TOKEN="your-api-token"
    python3 symptom_radar.py          # Daily report + store snapshot
    python3 symptom_radar.py --backfill  # Seed database with ~35 days of history
"""

import json, os, sys, sqlite3, math, time, argparse
from datetime import datetime, timedelta, timezone
import requests

# ─── Configuration ────────────────────────────────────────────────────────────
# Load .env file from the same directory if it exists
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip("\"'")
                if k == "ULTRAHUMAN_TOKEN":
                    TOKEN = v
                    break
else:
    TOKEN = os.environ.get("ULTRAHUMAN_TOKEN")

if not TOKEN:
    print("❌ ULTRAHUMAN_TOKEN environment variable not set.", file=sys.stderr)
    print("   Get your token at https://vision.ultrahuman.com/developer-docs", file=sys.stderr)
    sys.exit(1)

BASE_URL = "https://partner.ultrahuman.com/api/v1/partner/daily_metrics"
DB_PATH = os.environ.get("SYMPTOM_RADAR_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ultrahuman.db"))

# ─── Database ─────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_snapshots (
            date TEXT PRIMARY KEY,
            sleep_score REAL,
            total_sleep_min REAL,
            sleep_efficiency REAL,
            deep_sleep_min REAL,
            light_sleep_min REAL,
            rem_sleep_min REAL,
            temp_deviation REAL,
            avg_body_temp REAL,
            night_rhr REAL,
            sleep_rhr REAL,
            avg_sleep_hrv REAL,
            recovery_index REAL,
            movement_index REAL,
            active_minutes REAL,
            inactive_time REAL,
            total_steps REAL,
            vo2_max REAL,
            spo2 REAL,
            tosses_and_turns REAL,
            full_sleep_cycles REAL,
            restorative_sleep REAL,
            hr_drop REAL,
            morning_alertness REAL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    # Idempotent migrations: add newer columns to pre-existing databases
    # without dropping data or forcing a re-backfill.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_snapshots)")}
    if "hr_drop" not in cols:
        conn.execute("ALTER TABLE daily_snapshots ADD COLUMN hr_drop REAL")
    if "morning_alertness" not in cols:
        conn.execute("ALTER TABLE daily_snapshots ADD COLUMN morning_alertness REAL")

    # Daily self-report labels (the sickness-vs-strain training signal).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_labels (
            date TEXT PRIMARY KEY,
            label TEXT NOT NULL CHECK (label IN ('fine', 'rough', 'sick')),
            note TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn

# ─── Daily Self-Report Labels ─────────────────────────────────────────────────
# The single most valuable training signal available: the user's own
# assessment of how they felt. A few months of daily fine/rough/sick labels
# lets the engine learn to distinguish true illness from alcohol strain /
# poor sleep — which the biometrics alone cannot (both produce the same
# RHR↑/HRV↓/Temp↑/Recovery↓ signature).
LABELS = ("fine", "rough", "sick")

def log_label(date_str, label, note=None):
    """Record a daily self-report: 'fine' (normal), 'rough' (hangover/tired/
    stressed), or 'sick' (actually felt ill). Idempotent per date — re-logging
    a day overwrites the previous label."""
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")
    conn = init_db()
    conn.execute(
        "INSERT OR REPLACE INTO daily_labels (date, label, note) VALUES (?,?,?)",
        (date_str, label, note),
    )
    conn.commit()
    conn.close()
    return True

def get_labels():
    """Return [(date, label, note), ...] ordered by date."""
    conn = init_db()
    rows = conn.execute(
        "SELECT date, label, note FROM daily_labels ORDER BY date").fetchall()
    conn.close()
    return rows

# ─── API ──────────────────────────────────────────────────────────────────────
def fetch_day(date_str):
    resp = requests.get(
        BASE_URL,
        params={"date": date_str},
        headers={"Authorization": TOKEN},
        timeout=15
    )
    resp.raise_for_status()
    return resp.json()

def fetch_range(start_epoch, end_epoch):
    resp = requests.get(
        BASE_URL,
        params={"start_epoch": int(start_epoch), "end_epoch": int(end_epoch)},
        headers={"Authorization": TOKEN},
        timeout=15
    )
    resp.raise_for_status()
    return resp.json()

# ─── Metric Extraction ────────────────────────────────────────────────────────
def extract_metric(metrics, mtype):
    for m in metrics:
        if m.get("type") == mtype:
            obj = m.get("object", {})
            if mtype in ("hr", "hrv", "steps", "temp", "spo2"):
                vals = [v.get("value") for v in obj.get("values", [])
                        if isinstance(v.get("value"), (int, float))]
                if vals:
                    return {"avg": round(sum(vals)/len(vals), 1),
                            "min": min(vals), "max": max(vals)}
            if mtype in ("recovery_index", "movement_index", "active_minutes",
                         "inactive_time", "weekly_active_minutes", "movements",
                         "vo2_max", "hr_drop", "morning_alertness"):
                return {"value": obj.get("value")}
            if mtype == "night_rhr":
                return {"avg": obj.get("avg")}
            if mtype == "avg_sleep_hrv":
                return {"value": obj.get("value")}
            if mtype == "sleep_rhr":
                return {"value": obj.get("value")}
    return None

def extract_sleep_summary(obj):
    return {
        "sleep_score": (obj.get("sleep_score") or {}).get("score"),
        "total_sleep_min": (obj.get("total_sleep") or {}).get("minutes"),
        "sleep_efficiency": (obj.get("sleep_efficiency") or {}).get("percentage"),
        "deep_sleep_min": (obj.get("deep_sleep") or {}).get("minutes"),
        "light_sleep_min": (obj.get("light_sleep") or {}).get("minutes"),
        "rem_sleep_min": (obj.get("rem_sleep") or {}).get("minutes"),
        "temp_deviation": (obj.get("temperature_deviation") or {}).get("celsius"),
        "avg_body_temp": (obj.get("average_body_temperature") or {}).get("celsius"),
        "spo2": (obj.get("spo2") or {}).get("value"),
        "tosses_and_turns": (obj.get("tosses_and_turns") or {}).get("count"),
        "full_sleep_cycles": (obj.get("full_sleep_cycles") or {}).get("cycles"),
        "restorative_sleep": (obj.get("restorative_sleep") or {}).get("percentage"),
    }

def extract_steps_total(metrics):
    for m in metrics:
        if m.get("type") == "steps":
            vals = m.get("object", {}).get("values", [])
            total = sum(v.get("value", 0) for v in vals
                        if isinstance(v.get("value"), (int, float)))
            return total
    return None

# ─── Storage ──────────────────────────────────────────────────────────────────
def store_snapshot(conn, date_str, data):
    conn.execute("""
        INSERT OR REPLACE INTO daily_snapshots
        (date, sleep_score, total_sleep_min, sleep_efficiency,
         deep_sleep_min, light_sleep_min, rem_sleep_min,
         temp_deviation, avg_body_temp, night_rhr, sleep_rhr,
         avg_sleep_hrv, recovery_index, movement_index,
         active_minutes, inactive_time, total_steps, vo2_max,
         spo2, tosses_and_turns, full_sleep_cycles, restorative_sleep,
         hr_drop, morning_alertness)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        date_str,
        data.get("sleep_score"),
        data.get("total_sleep_min"),
        data.get("sleep_efficiency"),
        data.get("deep_sleep_min"),
        data.get("light_sleep_min"),
        data.get("rem_sleep_min"),
        data.get("temp_deviation"),
        data.get("avg_body_temp"),
        data.get("night_rhr"),
        data.get("sleep_rhr"),
        data.get("avg_sleep_hrv"),
        data.get("recovery_index"),
        data.get("movement_index"),
        data.get("active_minutes"),
        data.get("inactive_time"),
        data.get("total_steps"),
        data.get("vo2_max"),
        data.get("spo2"),
        data.get("tosses_and_turns"),
        data.get("full_sleep_cycles"),
        data.get("restorative_sleep"),
        data.get("hr_drop"),
        data.get("morning_alertness"),
    ))
    conn.commit()

def get_recent(conn, days=30):
    cur = conn.execute("""
        SELECT * FROM daily_snapshots
        ORDER BY date DESC LIMIT ?
    """, (days,))
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    rows.reverse()
    return [dict(zip(cols, r)) for r in rows]

# ─── Strain Detection ─────────────────────────────────────────────────────────
# Clean-baseline constants. The last CLEAN_GUARD days are treated as
# candidate-sick and excluded from the baseline so an ongoing illness doesn't
# inflate the mean/std and suppress today's score. TemPredict excluded febrile
# baseline days; we exclude the recent window proactively.
CLEAN_GUARD = 3
BASELINE_WINDOW = 21
TRAJ_WINDOW = 5      # days for slope / trajectory (includes today)
MIN_HISTORY = 11     # CLEAN_GUARD + 7 (min for rolling_stats_weighted) + 1
MIN_SLOPE_LEVEL = 0.5  # slope trend only boosts when level is already ≥0.5σ

def compute_zscore(val, mean, std):
    if std is None or std == 0 or val is None:
        return None
    return (val - mean) / std

def rolling_stats_weighted(series, decay=0.90):
    """
    Compute weighted mean and std with exponential decay.
    More recent values get higher weight.
    Decay=0.90: days 8–21 still contribute meaningfully, day 1 ≈ 12% of recent weight.
    Returns (None, None) if < 7 valid values.

    Std uses a Kish effective-sample-size correction (1 / (1 - sum(w^2)/W^2))
    so that weighting down older days doesn't systematically underestimate σ.
    """
    valid = [s for s in series if s is not None]
    if len(valid) < 7:
        return None, None

    n = len(valid)
    weights = [decay ** (n - 1 - i) for i in range(n)]
    total_w = sum(weights)
    norm_w = [w / total_w for w in weights]

    m = sum(w * v for w, v in zip(norm_w, valid))
    # Kish effective sample size for the bias correction
    eff_n = 1.0 / sum(w * w for w in norm_w)          # = (sum w)^2 / sum(w^2)
    variance = sum(w * (v - m) ** 2 for w, v in zip(norm_w, valid)) / (1 - 1.0 / eff_n)
    return m, math.sqrt(variance)

def compute_rolling_avg(series, window=3):
    """Compute rolling average of last N values, ignoring None.

    Returns None when fewer than `window` valid values exist, so callers can
    skip the trend term rather than having it silently regress toward the
    baseline mean (which would mask the very shift it's meant to catch).
    """
    valid = [s for s in series if s is not None]
    if len(valid) < window:
        return None
    return sum(valid[-window:]) / window

def _median(values):
    """Median ignoring None values."""
    valid = sorted(v for v in values if v is not None)
    n = len(valid)
    if n == 0:
        return None
    mid = n // 2
    return valid[mid] if n % 2 else (valid[mid - 1] + valid[mid]) / 2

def median_abs_deviation(series):
    """Median Absolute Deviation — a robust noise estimate that's insensitive
    to outliers (unlike std). Returns None if < 4 valid values (MAD is
    unreliable on tiny samples)."""
    med = _median(series)
    if med is None:
        return None
    devs = [abs(v - med) for v in series if v is not None]
    if len(devs) < 4:
        return None
    return _median(devs)

def linear_slope(series, dates=None):
    """Least-squares slope of a series over time (units per day).

    None values are dropped; the slope is fit on whatever points remain.
    If `dates` (a parallel list of 'YYYY-MM-DD' strings) is provided, x-values
    are real calendar-day offsets, so days missing entirely from the DB don't
    compress the time axis and steepen the apparent rate of rise. Without
    dates, x is array position.
    Returns None if < 3 valid points (can't fit a meaningful line).
    """
    # Calendar x-axis when dates are available; array index otherwise.
    # Any missing/unparseable date → fall back to index for the WHOLE series
    # (mixing calendar and index x-values in one fit would be incoherent).
    xs = None
    if dates and len(dates) == len(series):
        parsed, base = [], None
        for ds in dates:
            if not ds:
                parsed.append(None)
                continue
            try:
                d0 = datetime.strptime(ds, "%Y-%m-%d").date()
            except ValueError:
                parsed.append(None)
                continue
            if base is None:
                base = d0
            parsed.append((d0 - base).days)
        if all(x is not None for x in parsed):
            xs = parsed
    pts = []
    for i, v in enumerate(series):
        if v is None:
            continue
        pts.append((xs[i], v) if xs else (i, v))
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(i for i, _ in pts) / n
    my = sum(v for _, v in pts) / n
    num = sum((i - mx) * (v - my) for i, v in pts)
    den = sum((i - mx) ** 2 for i, _ in pts)
    return num / den if den else 0.0

def _robust_noise(std, mad):
    """Pick the LARGER of std and MAD-scaled (1.4826 = Φ⁻¹(0.75)).

    This ensures a metric with erratic day-to-day variation self-dampens:
    if HRV naturally swings ±15ms, its MAD will be large, its noise estimate
    will be large, and a given deviation will score lower than for a metric
    that's normally metronomically stable. This is per-metric personalization.
    """
    candidates = [c for c in [std, mad * 1.4826 if mad is not None else None] if c is not None]
    return max(candidates) if candidates else None

def _baseline_stats(baseline_series):
    """Compute mean and robust noise estimate for one metric's clean baseline."""
    mean, std = rolling_stats_weighted(baseline_series)
    mad = median_abs_deviation(baseline_series)
    noise = _robust_noise(std, mad)
    return mean, noise

# Metric definitions: (key, extractor, weight, inverted, display_label)
# inverted=True means LOW values = strain (HRV).
# Note: hr_drop is documented by Ultrahuman but not actually populated via the
# Partner API (verified across 7 days of live data — see README "Respiratory
# rate — the gap"). The extraction code + DB column remain so it's picked up
# automatically if Ultrahuman starts returning it; until then it would only
# contribute 0, so it's kept out of the active weight distribution.
STRAIN_METRICS = [
    ("rhr",  lambda d: d.get("night_rhr") or d.get("sleep_rhr"), 0.25, False, "RHR"),
    ("hrv",  lambda d: d.get("avg_sleep_hrv"),                    0.30, True,  "HRV"),
    ("temp", lambda d: d.get("temp_deviation"),                   0.45, False, "Temp Δ"),
]

# Minimum noise floor (σ) per metric. A perfectly flat baseline (std=0,
# MAD=0 — e.g. a sensor stuck on one value) would make the z undefined and
# the metric silently skipped (noise==0 → continue) exactly when a real
# spike arrives. Floors keep the metric alive; they sit below real
# physiological variability so they never change live scoring.
MIN_NOISE = {"rhr": 1.0, "hrv": 1.0, "temp": 0.05}

def _day_level_score(day, metric_baselines):
    """Quick level-only strain score for a single day (used for the
    persistence check). Uses the same noise-normalized z as the main scorer
    but skips trajectory — just 'how abnormal was this day vs baseline?'.

    metric_baselines: {key: {"mean", "noise", "weight", "inverted", "extract"}}
    """
    total = 0.0
    for key, mb in metric_baselines.items():
        mean, noise = mb["mean"], mb["noise"]
        if mean is None or noise is None or noise == 0:
            continue
        val = mb["extract"](day)
        if val is None:
            continue
        z = (val - mean) / noise
        z_s = -z if mb["inverted"] else z
        if z_s > 0:
            total += z_s * mb["weight"]
    return total

def assess_strain(history):
    """
    TemPredict-inspired strain detection (RR-free adaptation).

    This is NOT the original TemPredict ensemble (which used Random Forests on
    5 signal streams including respiratory rate). It adapts the parts that
    don't depend on RR:

    1. CLEAN BASELINE — excludes the last 3 days (candidate-sick window) so
       an ongoing illness doesn't contaminate the mean/std. This is the #1
       reason the old detector reacted late: the baseline included the
       sickness itself. TemPredict excluded febrile (>38°C) baseline days;
       we exclude the recent guard proactively.

    2. TRAJECTORY — uses 5-day least-squares slope (rate of rise), not just
       today's level. Rate-of-rise is the real illness signature; a flat
       shift to a new normal is more likely lifestyle.

    3. NOISE SCALING — each metric's z is divided by its own robust noise
       estimate (max of std and MAD-scaled), so an erratically-variable HRV
       doesn't dominate a stable temperature. Thresholds become personal.

    4. PERSISTENCE — the Major level requires sustained elevation across
       multiple recent days. A single noisy night won't trigger it. This is
       the single biggest false-positive reducer (filters hangover, exercise,
       one bad night of sleep).

    Returns:
        level (int): 0 = Normal, 1 = Elevated, 2 = Significant strain
        detail (str): Human-readable breakdown
    """
    if len(history) < MIN_HISTORY:
        return 0, f"Need {MIN_HISTORY}+ days of data for a clean baseline"

    # ── Clean baseline: exclude last CLEAN_GUARD days ──
    baseline_pool = history[:-CLEAN_GUARD] if len(history) > CLEAN_GUARD else []
    if len(baseline_pool) > BASELINE_WINDOW:
        baseline_pool = baseline_pool[-BASELINE_WINDOW:]
    if len(baseline_pool) < 7:
        return 0, "Insufficient clean baseline days"

    today = history[-1]
    recent = history[-TRAJ_WINDOW:]

    # ── Precompute baseline stats per metric (done once, reused for persistence) ──
    metric_baselines = {}
    for key, extract, weight, inverted, _label in STRAIN_METRICS:
        base_series = [extract(d) for d in baseline_pool]
        mean, noise = _baseline_stats(base_series)
        if noise is not None:
            noise = max(noise, MIN_NOISE.get(key, 0.0))
        metric_baselines[key] = {
            "mean": mean, "noise": noise, "weight": weight,
            "inverted": inverted, "extract": extract,
        }

    scores = {}
    contributions = []

    for key, _extract, _weight, inverted, label in STRAIN_METRICS:
        mb = metric_baselines[key]
        mean, noise = mb["mean"], mb["noise"]
        recent_series = [mb["extract"](d) for d in recent]
        today_val = mb["extract"](today)

        if mean is None or noise is None or noise == 0 or today_val is None:
            continue

        # Level z (today vs clean baseline, noise-normalized → strain direction)
        # Strain MAGNITUDE is always positive: for inverted metrics (HRV), the
        # strain direction is DEPRESSION, so we take the magnitude of the
        # negative z only. A high-HRV day (60-86 ms vs baseline ~43) is RESTED
        # and recovered — the opposite of illness — and must contribute 0.
        z_level = (today_val - mean) / noise
        if inverted:
            z_level_s = max(0.0, -z_level)   # only depression = strain
        else:
            z_level_s = max(0.0, z_level)    # only elevation = strain

        # Trend z (5-day slope → cumulative shift in σ over the window).
        # GATED against mean reversion: the slope only ADDS to the score when
        # today's level is already meaningfully strain-abnormal vs a ROBUST
        # reference (the baseline MEDIAN, not the mean). Without a robust
        # level gate, a metric recovering from a low excursion (e.g. temp
        # -0.95 → +0.03) produces a positive "rise" that fires a false flag —
        # the mechanism that flagged 07-24 on a completely normal temp. The
        # median reference is essential: a single outlier (e.g. the -1.74°C
        # temp on 07-21) drags the mean to -0.30, making a normal +0.03 day
        # look +0.5σ elevated and letting the reversion slope through.
        slope = linear_slope(recent_series, [d.get("date") for d in recent])
        if slope is not None:
            gate_ref = _median([mb["extract"](d) for d in baseline_pool]) or mean
            z_level_gate = (today_val - gate_ref) / noise
            z_level_gate_s = -z_level_gate if inverted else z_level_gate
            if z_level_gate_s > MIN_SLOPE_LEVEL:
                z_slope = (slope * TRAJ_WINDOW) / noise
                z_slope_s = abs(z_slope) if inverted else max(0.0, z_slope)
            else:
                z_slope_s = 0.0  # rising back toward typical — not strain
        else:
            z_slope_s = 0.0  # slope is only a boost for a real level deviation

        # Blend: max of level and trend (trend as boost, not dilution)
        z_blend = max(z_level_s, z_slope_s)

        if z_blend > 0:
            scores[key] = z_blend * mb["weight"]
            arrow = "↓" if inverted else "↑"
            contributions.append(f"{label} {arrow}{today_val}")

    # Recovery: level deficit + SLOPE (leading indicator)
    # A sharp multi-day recovery decline precedes the RHR/HRV/temp crash by
    # 1-2 days (observed in the user's data before every strain episode), so
    # it's the closest thing we have to an early-warning signal without RR.
    rec_series = [d.get("recovery_index") for d in baseline_pool]
    rec_mean, rec_noise = _baseline_stats(rec_series)
    today_rec = today.get("recovery_index")
    if rec_mean is not None and rec_noise is not None and rec_noise > 0 and today_rec is not None:
        z_rec = (today_rec - rec_mean) / rec_noise
        # Level deficit: recovery already well below baseline
        if z_rec < -1.0:
            scores["recovery_mod"] = min(abs(z_rec) * 0.10, 0.30)
            contributions.append(f"Recovery {today_rec}")
        # Slope warning: recovery falling fast over last 3 days — the leading
        # indicator observed 1-2 days BEFORE every strain episode in the user's
        # data (e.g. a -11/day recovery slope). Fires
        # whenever the decline is strong (>1σ cumulative over 3 days); the
        # 0.08 weight cap keeps it from dominating even when the level has
        # already crashed (it ADDS to the level signal rather than replacing).
        recent_rec = [d.get("recovery_index") for d in history[-3:]]
        rec_slope = linear_slope(recent_rec, [d.get("date") for d in history[-3:]])
        if rec_slope is not None and rec_slope < 0:
            z_rec_slope_s = -(rec_slope * 3) / rec_noise   # 3-day drop in σ, + = strain
            if z_rec_slope_s > 1.0:
                scores["recovery_trend"] = min(z_rec_slope_s * 0.08, 0.25)
                contributions.append(f"Recovery↓ {today_rec} ({rec_slope:+.0f}/day)")

    if not scores:
        return 0, "Biometrics within normal range"

    strain_today = sum(scores.values())

    # ── Persistence: how many of the last 3 CALENDAR days were elevated? ──
    # Calendar-aligned, not row-aligned: days missing entirely from the DB
    # would make the last 3 rows span more than 3 calendar days and distort
    # the persistence count.
    try:
        t_date = datetime.strptime(today.get("date") or "", "%Y-%m-%d").date()
        cal_window = {(t_date - timedelta(days=i)).isoformat() for i in range(3)}
        recent_rows = [d for d in history[-5:] if d.get("date") in cal_window]
    except ValueError:
        recent_rows = history[-3:]
    elevated_days = sum(
        1 for d in recent_rows
        if _day_level_score(d, metric_baselines) >= 1.0
    )

    detail = " | ".join(contributions)
    detail += f"\nStrain index: {strain_today:.2f} | Elevated in {elevated_days}/3 recent days"

    # Level decision:
    # 2 = Significant: strong (>= 2.0) AND sustained (>= 2 of last 3 days elevated)
    # 1 = Elevated: notable single-day deviation (worth watching — the "Oura flag")
    # 0 = Normal
    if strain_today >= 2.0 and elevated_days >= 2:
        return 2, detail
    elif strain_today >= 1.0:
        return 1, detail
    else:
        return 0, detail

# ─── Report ───────────────────────────────────────────────────────────────────
STRAIN_ICONS = {0: "✅ Normal", 1: "🟡 Elevated", 2: "🔴 Significant strain"}

def format_display(val, suffix=""):
    return "—" if val is None else f"{val}{suffix}"

def build_report():
    """Fetch, store, assess, and return the daily report string."""
    conn = init_db()
    # LOCAL calendar date, not UTC: the nightly check-in runs at 21:00 local,
    # which is already the next UTC day in North America — a UTC "today" would
    # fetch an unpopulated date and file data under tomorrow.
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    y_str = yesterday.strftime("%Y-%m-%d")
    t_str = today.strftime("%Y-%m-%d")

    try:
        y_data = fetch_day(y_str)
        t_data = fetch_day(t_str)
    except Exception as e:
        conn.close()
        return f"❌ API error: {e}"

    y_metrics = y_data.get("data", {}).get("metrics", {}).get(y_str, [])
    t_metrics = t_data.get("data", {}).get("metrics", {}).get(t_str, [])

    # Extract today's sleep + vitals
    sleep_raw = None
    for m in t_metrics:
        if m.get("type") == "sleep":
            sleep_raw = extract_sleep_summary(m.get("object", {}))
            break

    t_night_rhr = extract_metric(t_metrics, "night_rhr")
    t_sleep_rhr = extract_metric(t_metrics, "sleep_rhr")
    t_sleep_hrv = extract_metric(t_metrics, "avg_sleep_hrv")
    t_recovery = extract_metric(t_metrics, "recovery_index")
    t_movement = extract_metric(t_metrics, "movement_index")
    t_active = extract_metric(t_metrics, "active_minutes")
    t_inactive = extract_metric(t_metrics, "inactive_time")
    t_vo2 = extract_metric(t_metrics, "vo2_max")
    t_hr_drop = extract_metric(t_metrics, "hr_drop")
    t_alertness = extract_metric(t_metrics, "morning_alertness")
    t_hr = extract_metric(t_metrics, "hr")
    t_hrv = extract_metric(t_metrics, "hrv")
    t_temp = extract_metric(t_metrics, "temp")
    y_steps = extract_steps_total(y_metrics)

    rhr_val = (t_sleep_rhr or {}).get("value") or (t_night_rhr or {}).get("avg")
    hrv_val = (t_sleep_hrv or {}).get("value")

    # Store snapshot
    snapshot = {
        "sleep_score": (sleep_raw or {}).get("sleep_score"),
        "total_sleep_min": (sleep_raw or {}).get("total_sleep_min"),
        "sleep_efficiency": (sleep_raw or {}).get("sleep_efficiency"),
        "deep_sleep_min": (sleep_raw or {}).get("deep_sleep_min"),
        "light_sleep_min": (sleep_raw or {}).get("light_sleep_min"),
        "rem_sleep_min": (sleep_raw or {}).get("rem_sleep_min"),
        "temp_deviation": (sleep_raw or {}).get("temp_deviation"),
        "avg_body_temp": (sleep_raw or {}).get("avg_body_temp"),
        "night_rhr": (t_night_rhr or {}).get("avg") if t_night_rhr else None,
        "sleep_rhr": (t_sleep_rhr or {}).get("value") if t_sleep_rhr else None,
        "avg_sleep_hrv": hrv_val,
        "recovery_index": (t_recovery or {}).get("value"),
        "movement_index": (t_movement or {}).get("value"),
        "active_minutes": (t_active or {}).get("value"),
        "inactive_time": (t_inactive or {}).get("value"),
        "total_steps": y_steps,
        "vo2_max": (t_vo2 or {}).get("value"),
        "hr_drop": (t_hr_drop or {}).get("value"),
        "morning_alertness": (t_alertness or {}).get("value"),
        "spo2": (sleep_raw or {}).get("spo2"),
        "tosses_and_turns": (sleep_raw or {}).get("tosses_and_turns"),
        "full_sleep_cycles": (sleep_raw or {}).get("full_sleep_cycles"),
        "restorative_sleep": (sleep_raw or {}).get("restorative_sleep"),
    }
    store_snapshot(conn, t_str, snapshot)

    # Strain assessment
    history = get_recent(conn, 30)
    strain_level, strain_detail = assess_strain(history)

    # Build report
    parts = ["## 🩸 Ultrahuman Daily"]

    # Symptom Radar (top)
    parts.append(f"\n**🦠 Symptom Radar**")
    parts.append(f"**{STRAIN_ICONS[strain_level]}**")
    if strain_level > 0:
        parts.append(f"`{strain_detail}`")
    if strain_level == 1:
        parts.append("🟡 *Elevated deviations — worth watching today*")
    elif strain_level == 2:
        parts.append("🔴 *Significant strain detected — prioritize rest and recovery*")
    elif strain_level == 0 and strain_detail != "Insufficient data for strain assessment":
        parts.append("🟢 *Biometrics within normal range*")
    parts.append("")

    # Sleep
    if sleep_raw:
        s = sleep_raw
        score = format_display(s.get("sleep_score"))
        total = format_display(s.get("total_sleep_min"), " min")
        eff = format_display(s.get("sleep_efficiency"), "%")
        deep = format_display(s.get("deep_sleep_min"), " min")
        light = format_display(s.get("light_sleep_min"), " min")
        rem = format_display(s.get("rem_sleep_min"), " min")
        temp_dev = s.get("temp_deviation")
        temp_str = f"{temp_dev:+.1f}°C" if temp_dev is not None else "—"
        avg_temp = format_display(s.get("avg_body_temp"), "°C")
        spo2 = format_display(s.get("spo2"), "%")
        tosses = format_display(s.get("tosses_and_turns"))
        cycles = format_display(s.get("full_sleep_cycles"))
        restor = format_display(s.get("restorative_sleep"), "%")
        rhr_display = format_display(rhr_val, " bpm")

        parts.append("\n**😴 Sleep**")
        parts.append(f"Score: **{score}/100** | Total: **{total}** | Eff: **{eff}**")
        parts.append(f"Deep: **{deep}** | Light: **{light}** | REM: **{rem}**")
        parts.append(f"Cycles: **{cycles}** | Restorative: **{restor}**")
        parts.append(f"Sleep HRV: **{hrv_val}** | RHR: **{rhr_display}**")
        parts.append(f"Body Temp: **{avg_temp}** (Δ{temp_str})")
        parts.append(f"SPO2: **{spo2}** | Tosses: {tosses}")

    # Recovery & Activity
    parts.append("\n**💪 Recovery & Activity**")
    rec = format_display((t_recovery or {}).get("value"))
    mov = format_display((t_movement or {}).get("value"))
    act = format_display((t_active or {}).get("value"))
    ict = format_display((t_inactive or {}).get("value"))
    parts.append(f"Recovery: **{rec}/100** | Movement: **{mov}/100**")
    parts.append(f"Active: **{act} min** | Inactive: **{ict} min**")
    alertness = (t_alertness or {}).get("value")
    if alertness is not None:
        # Sleep-inertia minutes: how long after waking before the nervous
        # system is fully alert. Higher = rougher wakeup.
        parts.append(f"Morning Alertness: **{int(alertness)} min**")
    if y_steps:
        parts.append(f"Total Steps: **{int(y_steps)}**")
    vo2 = format_display((t_vo2 or {}).get("value"))
    if (t_vo2 or {}).get("value"):
        parts.append(f"VO2 Max: **{vo2}**")

    # Vitals
    parts.append("\n**❤️ Vitals**")
    if t_hr:
        parts.append(f"HR: avg **{t_hr['avg']}** bpm ({t_hr['min']}–{t_hr['max']})")
    if t_hrv:
        parts.append(f"HRV: avg **{t_hrv['avg']}** ms ({t_hrv['min']}–{t_hrv['max']})")
    if t_temp:
        parts.append(f"Skin Temp: avg **{t_temp['avg']}**°C ({t_temp['min']}–{t_temp['max']})")

    history_count = len([d for d in history if d.get("sleep_score") is not None])
    parts.append(f"\n📊 *Baseline: {history_count} days of data*")

    conn.close()
    return "\n".join([p for p in parts if p])

# ─── Backfill ─────────────────────────────────────────────────────────────────
def backfill(days=35):
    """Fetch historical data to seed the baseline database."""
    conn = init_db()
    today = datetime.now(timezone.utc)
    start = today - timedelta(days=days)
    end = today - timedelta(days=1)
    current = start
    total = 0

    print(f"Backfilling {days} days ({start.date()} to {end.date()})...")

    while current <= end:
        chunk_end = min(current + timedelta(days=6), end)
        s_epoch = int(current.replace(tzinfo=timezone.utc).timestamp())
        e_epoch = int((chunk_end + timedelta(days=1)).replace(tzinfo=timezone.utc).timestamp())

        try:
            data = fetch_range(s_epoch, e_epoch)
        except Exception as e:
            print(f"  Error fetching {current.date()}–{chunk_end.date()}: {e}")
            current = chunk_end + timedelta(days=1)
            time.sleep(2)
            continue

        metrics_by_date = data.get("data", {}).get("metrics", {})
        day = current

        while day <= chunk_end:
            d_str = day.strftime("%Y-%m-%d")
            metrics = metrics_by_date.get(d_str, [])

            if metrics:
                sleep_raw = None
                for m in metrics:
                    if m.get("type") == "sleep":
                        sleep_raw = extract_sleep_summary(m.get("object", {}))
                        break

                snapshot = {
                    "sleep_score": (sleep_raw or {}).get("sleep_score"),
                    "total_sleep_min": (sleep_raw or {}).get("total_sleep_min"),
                    "sleep_efficiency": (sleep_raw or {}).get("sleep_efficiency"),
                    "deep_sleep_min": (sleep_raw or {}).get("deep_sleep_min"),
                    "light_sleep_min": (sleep_raw or {}).get("light_sleep_min"),
                    "rem_sleep_min": (sleep_raw or {}).get("rem_sleep_min"),
                    "temp_deviation": (sleep_raw or {}).get("temp_deviation"),
                    "avg_body_temp": (sleep_raw or {}).get("avg_body_temp"),
                    "night_rhr": (extract_metric(metrics, "night_rhr") or {}).get("avg"),
                    "sleep_rhr": (extract_metric(metrics, "sleep_rhr") or {}).get("value"),
                    "avg_sleep_hrv": (extract_metric(metrics, "avg_sleep_hrv") or {}).get("value"),
                    "recovery_index": (extract_metric(metrics, "recovery_index") or {}).get("value"),
                    "movement_index": (extract_metric(metrics, "movement_index") or {}).get("value"),
                    "active_minutes": (extract_metric(metrics, "active_minutes") or {}).get("value"),
                    "inactive_time": (extract_metric(metrics, "inactive_time") or {}).get("value"),
                    "total_steps": extract_steps_total(metrics),
                    "vo2_max": (extract_metric(metrics, "vo2_max") or {}).get("value"),
                    "hr_drop": (extract_metric(metrics, "hr_drop") or {}).get("value"),
                    "morning_alertness": (extract_metric(metrics, "morning_alertness") or {}).get("value"),
                    "spo2": (sleep_raw or {}).get("spo2"),
                    "tosses_and_turns": (sleep_raw or {}).get("tosses_and_turns"),
                    "full_sleep_cycles": (sleep_raw or {}).get("full_sleep_cycles"),
                    "restorative_sleep": (sleep_raw or {}).get("restorative_sleep"),
                }
                store_snapshot(conn, d_str, snapshot)
                total += 1

            day += timedelta(days=1)

        current = chunk_end + timedelta(days=1)
        time.sleep(0.5)

    conn.close()
    print(f"\nDone. Stored {total} days.")
    return total

# ─── MCP Server (stdio) ───────────────────────────────────────────────────────
# Minimal JSON-RPC 2.0 server over stdio. No third-party SDK required — speaks
# the line-delimited protocol that Claude / Hermes / Cursor expect.
#
# Exposed tools:
#   symptom_radar_report  → today's daily report (markdown)
#   symptom_radar_history → query raw metric series (days, metric)
#   symptom_radar_strain  → strain level + breakdown for today
def _mcp_handle(method, params):
    """Dispatch one MCP method call. Returns (result_dict, error_str_or_None)."""
    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "symptom-radar",
                "version": "1.0.0",
            },
        }, None

    if method == "tools/list":
        return {"tools": [
            {
                "name": "symptom_radar_report",
                "description": "Fetch today's Ultrahuman data, store a snapshot, and return the daily report (markdown) including the Symptom Radar strain assessment.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "symptom_radar_history",
                "description": "Query the local biometric database. Returns the most recent N days of one or more metrics, oldest-first. Metrics: sleep_score, total_sleep_min, night_rhr, sleep_rhr, avg_sleep_hrv, temp_deviation, recovery_index, movement_index, total_steps, vo2_max, hr_drop, morning_alertness, spo2.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "days": {"type": "integer", "default": 14, "minimum": 1, "maximum": 90},
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to return. Omit for a default set (date, sleep_score, night_rhr, sleep_rhr, avg_sleep_hrv, temp_deviation).",
                        },
                    },
                },
            },
            {
                "name": "symptom_radar_strain",
                "description": "Return today's Symptom Radar strain level (0/1/2), the strain index, and the per-metric breakdown without re-fetching from the API.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "symptom_radar_label",
                "description": "Log a daily self-report label — the training signal that lets the engine distinguish real illness from alcohol strain / poor sleep (both look identical in the biometrics). Call this daily with how the user felt.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": ["fine", "rough", "sick"],
                                  "description": "fine=normal, rough=hangover/tired/stressed, sick=actually felt ill"},
                        "date": {"type": "string", "description": "YYYY-MM-DD (defaults to today UTC)"},
                        "note": {"type": "string", "description": "Optional context, e.g. 'hangover' or 'fever'"},
                    },
                    "required": ["label"],
                },
            },
            {
                "name": "symptom_radar_labels",
                "description": "Return all logged self-report labels as [[date, label, note], ...].",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]}, None

    if method == "tools/call":
        name = (params or {}).get("name")
        args = (params or {}).get("arguments", {}) or {}
        if name == "symptom_radar_report":
            return {"content": [{"type": "text", "text": build_report()}]}, None
        if name == "symptom_radar_history":
            days = int(args.get("days", 14))
            metrics = args.get("metrics") or [
                "date", "sleep_score", "night_rhr", "sleep_rhr",
                "avg_sleep_hrv", "temp_deviation",
            ]
            # Allow only known columns to prevent SQL injection.
            valid_cols = {
                "date", "sleep_score", "total_sleep_min", "sleep_efficiency",
                "deep_sleep_min", "light_sleep_min", "rem_sleep_min",
                "temp_deviation", "avg_body_temp", "night_rhr", "sleep_rhr",
                "avg_sleep_hrv", "recovery_index", "movement_index",
                "active_minutes", "inactive_time", "total_steps", "vo2_max",
                "spo2", "tosses_and_turns", "full_sleep_cycles", "hr_drop",
                "restorative_sleep", "morning_alertness", "created_at",
            }
            safe = [m for m in metrics if m in valid_cols] or ["date", "sleep_score"]
            conn = init_db()
            cur = conn.execute(
                f"SELECT {', '.join(safe)} FROM daily_snapshots ORDER BY date DESC LIMIT ?",
                (days,),
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            rows.reverse()
            conn.close()
            return {"content": [{"type": "text", "text": json.dumps(rows, indent=2)}]}, None
        if name == "symptom_radar_strain":
            conn = init_db()
            history = get_recent(conn, 30)
            conn.close()
            level, detail = assess_strain(history)
            return {"content": [{"type": "text", "text": json.dumps({
                "level": level,
                "level_label": STRAIN_ICONS.get(level),
                "detail": detail,
            }, indent=2)}]}, None
        if name == "symptom_radar_label":
            date_str = str(args.get("date") or
                          datetime.now().strftime("%Y-%m-%d"))
            label = str(args.get("label") or "")
            if label not in LABELS:
                return None, f"label must be one of {LABELS}"
            note = args.get("note")
            log_label(date_str, label, note)
            return {"content": [{"type": "text", "text": json.dumps({
                "logged": date_str, "label": label, "note": note,
            })}]}, None
        if name == "symptom_radar_labels":
            return {"content": [{"type": "text",
                                 "text": json.dumps(get_labels(), indent=2)}]}, None
        return None, f"Unknown tool: {name}"

    return None, f"Unknown method: {method}"

def run_mcp_server():
    """Read JSON-RPC requests from stdin, write responses to stdout.

    Honors the initialized notification and the standard initialize handshake.
    Responds to ping. Closes on EOF or shutdown request.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps({
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }) + "\n")
            sys.stdout.flush()
            continue

        req_id = req.get("id")
        method = req.get("method")

        if method == "shutdown":
            if req_id is not None:
                sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {}}) + "\n")
                sys.stdout.flush()
            break

        # Notifications (no id) — initialize, notifications/initialized — just ack silently
        if req_id is None:
            continue

        if method == "ping":
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {}}) + "\n")
            sys.stdout.flush()
            continue

        result, err = _mcp_handle(method, req.get("params"))
        if err is not None:
            sys.stdout.write(json.dumps({
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": err},
            }) + "\n")
        else:
            sys.stdout.write(json.dumps({
                "jsonrpc": "2.0", "id": req_id, "result": result,
            }) + "\n")
        sys.stdout.flush()

# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symptom Radar for Ultrahuman Ring")
    parser.add_argument("--backfill", type=int, nargs="?",
                        const=35, metavar="DAYS",
                        help="Backfill historical data (default: 35 days)")
    parser.add_argument("--mcp", action="store_true",
                        help="Run as a stdio MCP server (for Claude / Hermes / Cursor)")
    parser.add_argument("--label", metavar="STATUS",
                        choices=LABELS,
                        help="Log today's self-report: fine / rough / sick "
                             "(the training signal that lets the engine learn "
                             "your illness-vs-alcohol distinction)")
    parser.add_argument("--label-date", metavar="YYYY-MM-DD", default=None,
                        help="Date for --label (defaults to today, UTC)")
    parser.add_argument("--label-note", metavar="TEXT", default=None,
                        help="Optional note for --label (e.g. 'hangover', 'fever')")
    parser.add_argument("--labels", action="store_true",
                        help="Print all logged self-report labels")
    args = parser.parse_args()

    if args.mcp:
        run_mcp_server()
    elif args.backfill:
        backfill(args.backfill)
    elif args.label:
        date_str = args.label_date or datetime.now().strftime("%Y-%m-%d")
        log_label(date_str, args.label, args.label_note)
        print(f"✅ Logged {date_str}: {args.label}"
              + (f" ({args.label_note})" if args.label_note else ""))
        print("   Tip: run daily — 'fine', 'rough', or 'sick'. A few months of "
              "labels lets the engine separate real illness from alcohol strain.")
    elif args.labels:
        labels = get_labels()
        if not labels:
            print("No labels logged yet. Use --label fine|rough|sick (e.g. "
                  "python3 symptom_radar.py --label rough).")
        else:
            print(f"{'Date':<12} {'Label':<8} Note")
            for d, lbl, note in labels:
                print(f"{d:<12} {lbl:<8} {note or ''}")
    else:
        report = build_report()
        print(report)
