import math
from datetime import datetime, timedelta

# The baseline ends CLEAN_GUARD calendar days before today: today and the two
# days before it are the candidate-sick window and never enter the reference.
CLEAN_GUARD = 3
BASELINE_WINDOW = 21
TRAJ_WINDOW = 5      # days for slope / trajectory (includes today)
MIN_SLOPE_LEVEL = 0.5  # slope trend only boosts when level is already ≥0.5σ

def compute_zscore(val, mean, std):
    if std is None or std == 0 or val is None:
        return None
    return (val - mean) / std

def rolling_stats_weighted(series, decay=0.90, dates=None, today=None):
    """
    Compute weighted mean and std with exponential decay.
    More recent values get higher weight.
    Decay=0.90: days 8–21 still contribute meaningfully, day 1 ≈ 12% of recent weight.
    Returns (None, None) if < 7 valid values.

    Std uses a Kish effective-sample-size correction (1 / (1 - sum(w^2)/W^2))
    so that weighting down older days doesn't systematically underestimate σ.
    """
    points = [(i, value) for i, value in enumerate(series) if value is not None]
    if len(points) < 7:
        return None, None
    valid = [value for _, value in points]
    if dates is not None:
        anchor = datetime.strptime(today or dates[-1], "%Y-%m-%d").date()
        ages = [(anchor - datetime.strptime(dates[i], "%Y-%m-%d").date()).days
                for i, _ in points]
    else:
        ages = [len(series) - 1 - i for i, _ in points]
    weights = [decay ** age for age in ages]
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

def _baseline_stats(baseline_series, dates=None, today=None):
    """Compute mean and robust noise estimate for one metric's clean baseline."""
    mean, std = rolling_stats_weighted(baseline_series, dates=dates, today=today)
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
    ("rhr",  lambda d: d.get("night_rhr") if d.get("night_rhr") is not None else d.get("sleep_rhr"), 0.25, False, "RHR"),
    ("hrv",  lambda d: d.get("avg_sleep_hrv"),                    0.30, True,  "HRV"),
    ("temp", lambda d: d.get("temp_deviation"),                   0.45, False, "Temp Δ"),
]

# Minimum noise floor (σ) per metric. A perfectly flat baseline (std=0,
# MAD=0 — e.g. a sensor stuck on one value) would make the z undefined and
# the metric silently skipped (noise==0 → continue) exactly when a real
# spike arrives. Floors keep the metric alive; they sit below real
# physiological variability so they never change live scoring.
MIN_NOISE = {"rhr": 1.0, "hrv": 1.0, "temp": 0.10}

# A baseline day whose weighted strain-direction deviation from the window's
# robust reference reaches this score (Significant-grade on the strain index
# scale) is treated as illness and kept out of the reference.
ILLNESS_EXCLUSION_SCORE = 3.0

def _calendar_rows(history):
    """Give legacy undated numerical inputs daily spacing."""
    if history and not any(day.get("date") for day in history):
        start = datetime(2000, 1, 1).date()
        return [{**day, "date": (start + timedelta(days=i)).isoformat()}
                for i, day in enumerate(history)]
    return history


def _calendar_window(history, first_age, last_age):
    if not history:
        return []
    today = datetime.strptime(history[-1]["date"], "%Y-%m-%d").date()
    start = (today - timedelta(days=last_age)).isoformat()
    end = (today - timedelta(days=first_age)).isoformat()
    return [day for day in history if start <= day["date"] <= end]


def build_baseline(history, metrics=None):
    """Label-blind, non-recursive clean baseline for the last dated snapshot.

    The window is the BASELINE_WINDOW calendar days ending CLEAN_GUARD days
    before today (D-23 to D-3 with the defaults). A day is dropped when its
    weighted strain-direction deviation from the window's median/MAD reference
    reaches ILLNESS_EXCLUSION_SCORE, the same scale as the strain index. That
    keeps a prolonged illness out of its own reference (TemPredict dropped
    febrile baseline days) until it fills about half the window, after which
    a sustained shift becomes the new normal. Ordinary variability, and
    excursions in the healthy direction, stay in so the noise estimate is not
    trimmed.
    """
    metrics = STRAIN_METRICS if metrics is None else metrics
    history = _calendar_rows(history)
    pool = _calendar_window(history, CLEAN_GUARD, CLEAN_GUARD + BASELINE_WINDOW - 1)
    references = []
    for _key, extract, weight, inverted, _label in metrics:
        series = [extract(day) for day in pool]
        if sum(value is not None for value in series) >= 7:
            noise = max(1.4826 * median_abs_deviation(series), MIN_NOISE.get(_key, 1.0))
            references.append((_median(series), noise, extract, weight, inverted))

    def illness_grade(day):
        score = 0.0
        for median, noise, extract, weight, inverted in references:
            value = extract(day)
            if value is not None:
                deviation = (value - median) / noise
                score += weight * max(0.0, -deviation if inverted else deviation)
        return score >= ILLNESS_EXCLUSION_SCORE

    pool = [day for day in pool if not illness_grade(day)]
    dates = [day["date"] for day in pool]
    today = history[-1]["date"] if history else None
    metric_baselines = {}
    for key, extract, weight, inverted, _label in metrics:
        series = [extract(day) for day in pool]
        mean, noise = _baseline_stats(series, dates, today)
        if noise is not None:
            noise = max(noise, MIN_NOISE.get(key, 1.0))
        metric_baselines[key] = {"mean": mean, "noise": noise, "weight": weight,
                        "inverted": inverted, "extract": extract}
    baseline_days = sum(
        sum(extract(day) is not None for _, extract, *_ in STRAIN_METRICS) >= 2
        for day in pool)
    return {"rows": pool, "metrics": metric_baselines, "baseline_days": baseline_days}


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
    """Legacy tuple adapter for retrospective callers. Unknown is unflagged.

    Use assess_strain_result for status-aware decisions.
    """
    result = assess_strain_result(history)
    return result["level"] if result["level"] is not None else 0, result["detail"]


def assess_strain_result(history, metrics=None):
    """
    TemPredict-inspired strain detection (RR-free adaptation).

    This is NOT the original TemPredict ensemble (which used Random Forests on
    5 signal streams including respiratory rate). It adapts the parts that
    don't depend on RR:

    1. CLEAN BASELINE: calendar days D-23 through D-3, minus illness-grade
       days, before weighted statistics. No labels or recursive scoring.

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

    Returns a structured dictionary with status, optional level (0/1/2),
    strain_index, elevated_recent, metrics, coverage, baseline_days, and detail.
    Status is one of ok, no_data, calibrating, or insufficient_data.
    Normal requires two assessable core metrics. Partial elevation is retained.
    """
    history = _calendar_rows(history)
    metrics = STRAIN_METRICS if metrics is None else metrics
    today = history[-1] if history else {}
    coverage = [key for key, extract, *_ in STRAIN_METRICS if extract(today) is not None]
    baseline = build_baseline(history, metrics=metrics)
    baseline_pool = baseline["rows"]
    baseline_days = baseline["baseline_days"]
    result = {"status": "no_data", "level": None, "strain_index": None,
              "elevated_recent": None,
              "metrics": {key: {"z": None, "z_slope": None, "contribution": None}
                          for key, *_ in metrics}, "coverage": coverage,
              "baseline_days": baseline_days, "detail": "No core metrics for today"}
    if not coverage:
        return result
    if baseline_days < 7:
        result.update(status="calibrating",
                      detail=f"{baseline_days} of 7 clean baseline days")
        return result
    recent = _calendar_window(history, 0, TRAJ_WINDOW - 1)
    metric_baselines = baseline["metrics"]

    scores = {}
    contributions = []
    metric_details = {}
    assessable = []

    for key, _extract, _weight, inverted, label in metrics:
        mb = metric_baselines[key]
        mean, noise = mb["mean"], mb["noise"]
        recent_series = [mb["extract"](d) for d in recent]
        today_val = mb["extract"](today)

        metric_details[key] = {"z": None, "z_slope": None, "contribution": None}
        if mean is None or noise is None or noise == 0 or today_val is None:
            continue
        if key in {metric[0] for metric in STRAIN_METRICS}:
            assessable.append(key)

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
        gate_ref = _median([mb["extract"](d) for d in baseline_pool])
        if gate_ref is None:
            gate_ref = mean
        slope = linear_slope(recent_series, [d["date"] for d in recent])
        if slope is not None:
            z_level_gate = (today_val - gate_ref) / noise
            z_level_gate_s = -z_level_gate if inverted else z_level_gate
            if z_level_gate_s > MIN_SLOPE_LEVEL:
                z_slope = (slope * TRAJ_WINDOW) / noise
                z_slope_s = max(0.0, -z_slope if inverted else z_slope)
            else:
                z_slope_s = 0.0  # rising back toward typical — not strain
        else:
            z_slope_s = 0.0  # slope is only a boost for a real level deviation

        # Blend: max of level and trend (trend as boost, not dilution)
        z_blend = max(z_level_s, z_slope_s)

        metric_details[key] = {"z": z_level, "z_slope": z_slope_s,
                               "contribution": z_blend * mb["weight"]}
        if z_blend > 0:
            scores[key] = z_blend * mb["weight"]
            arrow = "↓" if inverted else "↑"
            # Raw API floats (e.g. temp_deviation 0.02447553752388387)
            # leak ugly into the digest; round for display.
            contributions.append(f"{label} {arrow}{round(today_val, 2)}")

    # Recovery: level deficit + SLOPE (leading indicator)
    # A sharp multi-day recovery decline precedes the RHR/HRV/temp crash by
    # 1-2 days (observed in the user's data before every strain episode), so
    # it's the closest thing we have to an early-warning signal without RR.
    rec_series = [d.get("recovery_index") for d in baseline_pool]
    rec_mean, rec_noise = _baseline_stats(
        rec_series, [d["date"] for d in baseline_pool], today["date"])
    if rec_noise is not None:
        rec_noise = max(rec_noise, 1.0)
    today_rec = today.get("recovery_index")
    metric_details["recovery"] = {"z": None, "z_slope": None, "contribution": None}
    if rec_mean is not None and rec_noise is not None and rec_noise > 0 and today_rec is not None:
        z_rec = (today_rec - rec_mean) / rec_noise
        metric_details["recovery"] = {"z": z_rec, "z_slope": 0.0, "contribution": 0.0}
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
        recovery_rows = _calendar_window(history, 0, 2)
        recent_rec = [d.get("recovery_index") for d in recovery_rows]
        rec_slope = linear_slope(recent_rec, [d["date"] for d in recovery_rows])
        if rec_slope is not None and rec_slope < 0:
            z_rec_slope_s = -(rec_slope * 3) / rec_noise   # 3-day drop in σ, + = strain
            metric_details["recovery"]["z_slope"] = z_rec_slope_s
            if z_rec_slope_s > 1.0:
                scores["recovery_trend"] = min(z_rec_slope_s * 0.08, 0.25)
                contributions.append(f"Recovery↓ {today_rec} ({rec_slope:+.0f}/day)")

        metric_details["recovery"]["contribution"] = (
            scores.get("recovery_mod", 0.0) + scores.get("recovery_trend", 0.0))

    strain_today = sum(scores.values())

    # ── Persistence: how many of the last 3 CALENDAR days were elevated? ──
    # Calendar-aligned, not row-aligned: days missing entirely from the DB
    # would make the last 3 rows span more than 3 calendar days and distort
    # the persistence count.
    recent_rows = _calendar_window(history, 0, 2)
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
    level = 2 if strain_today >= 2.0 and elevated_days >= 2 else 1 if strain_today >= 1.0 else 0
    status = "ok"
    if level == 0 and len(assessable) < 2:
        status = "calibrating" if len(coverage) >= 2 else "insufficient_data"
        level = None
        if status == "calibrating":
            detail = (f"baseline available for {len(assessable)} of {len(coverage)} "
                      "present core metrics; need 2")
    result.update(status=status, level=level, strain_index=strain_today,
                  elevated_recent=elevated_days, metrics=metric_details, detail=detail)
    return result
