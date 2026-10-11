from . import config
import requests

def fetch_day(date_str):
    resp = requests.get(
        config.BASE_URL,
        params={"date": date_str},
        headers={"Authorization": config.require_token()},
        timeout=15
    )
    resp.raise_for_status()
    return resp.json()

def fetch_range(start_epoch, end_epoch):
    resp = requests.get(
        config.BASE_URL,
        params={"start_epoch": int(start_epoch), "end_epoch": int(end_epoch)},
        headers={"Authorization": config.require_token()},
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
                    # Raw per-sample floats (e.g. 31.44000053405762) leak ugly
                    # into the digest; round for display. Ints pass through as
                    # ints (round(53, 1) == 53).
                    return {"avg": round(sum(vals)/len(vals), 1),
                            "min": round(min(vals), 1), "max": round(max(vals), 1)}
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
