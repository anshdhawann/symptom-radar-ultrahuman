from . import client, config, engine, storage
import time
from datetime import timedelta

STRAIN_ICONS = {0: "✅ Normal", 1: "🟡 Elevated", 2: "🔴 Significant strain"}

def format_display(val, suffix=""):
    return "—" if val is None else f"{val}{suffix}"

def fmt_minutes(val):
    """Whole minutes for the digest; raw API floats (inactive_time arrives as
    40.81666666666667) round to the nearest minute."""
    if isinstance(val, (int, float)):
        return f"{round(val)} min"
    return "—" if val is None else f"{val} min"

def build_report():
    """Fetch, store, assess, and return the daily report string."""
    conn = storage.init_db()
    today = config.wearer_now()
    yesterday = today - timedelta(days=1)
    y_str = yesterday.strftime("%Y-%m-%d")
    t_str = today.strftime("%Y-%m-%d")

    try:
        y_data = client.fetch_day(y_str)
        t_data = client.fetch_day(t_str)
    except Exception as e:
        conn.close()
        return f"❌ API error: {e}"

    y_metrics = y_data.get("data", {}).get("metrics", {}).get(y_str, [])
    t_metrics = t_data.get("data", {}).get("metrics", {}).get(t_str, [])

    # Extract today's sleep + vitals
    sleep_raw = None
    for m in t_metrics:
        if m.get("type") == "sleep":
            sleep_raw = client.extract_sleep_summary(m.get("object", {}))
            break

    if sleep_raw is not None:
        sleep_raw = {key: value if storage._valid_metric_value(value) else None
                     for key, value in sleep_raw.items()}

    t_night_rhr = client.extract_metric(t_metrics, "night_rhr")
    t_sleep_rhr = client.extract_metric(t_metrics, "sleep_rhr")
    t_sleep_hrv = client.extract_metric(t_metrics, "avg_sleep_hrv")
    t_recovery = client.extract_metric(t_metrics, "recovery_index")
    t_movement = client.extract_metric(t_metrics, "movement_index")
    t_active = client.extract_metric(t_metrics, "active_minutes")
    t_inactive = client.extract_metric(t_metrics, "inactive_time")
    t_vo2 = client.extract_metric(t_metrics, "vo2_max")
    t_hr_drop = client.extract_metric(t_metrics, "hr_drop")
    t_alertness = client.extract_metric(t_metrics, "morning_alertness")
    t_hr = client.extract_metric(t_metrics, "hr")
    t_hrv = client.extract_metric(t_metrics, "hrv")
    t_temp = client.extract_metric(t_metrics, "temp")
    y_steps = client.extract_steps_total(y_metrics)

    rhr_val = (t_sleep_rhr or {}).get("value")
    if rhr_val is None:
        rhr_val = (t_night_rhr or {}).get("avg")
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
    snapshot = {key: value if storage._valid_metric_value(value) else None
                for key, value in snapshot.items()}
    if any(value is not None for key, value in snapshot.items() if key != "total_steps"):
        storage.store_snapshot(conn, t_str, snapshot)

    # Strain assessment
    history = [day for day in storage.get_recent(conn, 30) if day["date"] < t_str]
    assessment = engine.assess_strain_result(history + [{"date": t_str, **snapshot}])
    strain_level, strain_detail = assessment["level"], assessment["detail"]

    # Build report
    parts = ["## 🩸 Ultrahuman Daily"]

    # Symptom Radar (top)
    parts.append(f"\n**🦠 Symptom Radar**")
    if assessment["status"] == "ok":
        verdict = STRAIN_ICONS[strain_level]
    elif assessment["status"] == "no_data":
        verdict = "⚪ No data: ring not synced"
    elif assessment["status"] == "calibrating":
        verdict = f"⏳ Calibrating: {assessment['detail']}"
    else:
        verdict = f"⚪ Insufficient data: {len(assessment['coverage'])} of 3 core metrics"
    parts.append(f"**{verdict}**")
    if strain_level is not None and strain_level > 0:
        parts.append(f"`{strain_detail}`")
    if strain_level == 1:
        parts.append("🟡 *Elevated deviations — worth watching today*")
    elif strain_level == 2:
        parts.append("🔴 *Significant strain detected — prioritize rest and recovery*")
    elif strain_level == 0 and strain_detail != "Insufficient data for strain assessment":
        parts.append("🟢 *Biometrics within normal range*")
    parts.append("")

    # Sleep. A sleep metric object can exist while carrying no data yet
    # ({"index_valid": false} — ring hasn't processed/synced the night), which
    # used to render a block of em-dashes that reads like a parsing failure.
    sleep_has_data = bool(sleep_raw) and any(
        v is not None for v in sleep_raw.values())
    if sleep_has_data:
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
        parts.append(f"Sleep HRV: **{format_display(hrv_val)}** | RHR: **{rhr_display}**")
        parts.append(f"Body Temp: **{avg_temp}** (Δ{temp_str})")
        parts.append(f"SPO2: **{spo2}** | Tosses: {tosses}")
    else:
        parts.append("\n**😴 Sleep**")
        parts.append("⏳ *No sleep record for today yet — ring has not synced last night*")
        if rhr_val is not None:
            parts.append(f"RHR: **{format_display(rhr_val, ' bpm')}**")

    # Recovery & Activity
    parts.append("\n**💪 Recovery & Activity**")
    rec = format_display((t_recovery or {}).get("value"))
    mov = format_display((t_movement or {}).get("value"))
    act = fmt_minutes((t_active or {}).get("value"))
    ict = fmt_minutes((t_inactive or {}).get("value"))
    parts.append(f"Recovery: **{rec}/100** | Movement: **{mov}/100**")
    parts.append(f"Active: **{act}** | Inactive: **{ict}**")
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
    conn = storage.init_db()
    today = config.wearer_now().date()
    start = today - timedelta(days=days)
    end = today - timedelta(days=1)
    current = start
    total = 0

    print(f"Backfilling {days} days ({start} to {end})...")

    while current <= end:
        chunk_end = min(current + timedelta(days=6), end)
        s_epoch = int(config.wearer_midnight(current).timestamp())
        e_epoch = int(config.wearer_midnight(chunk_end + timedelta(days=1)).timestamp())

        try:
            data = client.fetch_range(s_epoch, e_epoch)
        except Exception as e:
            print(f"  Error fetching {current} to {chunk_end}: {e}")
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
                        sleep_raw = client.extract_sleep_summary(m.get("object", {}))
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
                    "night_rhr": (client.extract_metric(metrics, "night_rhr") or {}).get("avg"),
                    "sleep_rhr": (client.extract_metric(metrics, "sleep_rhr") or {}).get("value"),
                    "avg_sleep_hrv": (client.extract_metric(metrics, "avg_sleep_hrv") or {}).get("value"),
                    "recovery_index": (client.extract_metric(metrics, "recovery_index") or {}).get("value"),
                    "movement_index": (client.extract_metric(metrics, "movement_index") or {}).get("value"),
                    "active_minutes": (client.extract_metric(metrics, "active_minutes") or {}).get("value"),
                    "inactive_time": (client.extract_metric(metrics, "inactive_time") or {}).get("value"),
                    "total_steps": client.extract_steps_total(metrics),
                    "vo2_max": (client.extract_metric(metrics, "vo2_max") or {}).get("value"),
                    "hr_drop": (client.extract_metric(metrics, "hr_drop") or {}).get("value"),
                    "morning_alertness": (client.extract_metric(metrics, "morning_alertness") or {}).get("value"),
                    "spo2": (sleep_raw or {}).get("spo2"),
                    "tosses_and_turns": (sleep_raw or {}).get("tosses_and_turns"),
                    "full_sleep_cycles": (sleep_raw or {}).get("full_sleep_cycles"),
                    "restorative_sleep": (sleep_raw or {}).get("restorative_sleep"),
                }
                storage.store_snapshot(conn, d_str, snapshot)
                total += 1

            day += timedelta(days=1)

        current = chunk_end + timedelta(days=1)
        time.sleep(0.5)

    conn.close()
    print(f"\nDone. Stored {total} days.")
    return total
