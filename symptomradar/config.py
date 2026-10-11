import os, sys
from datetime import datetime
from zoneinfo import ZoneInfo

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
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
DB_PATH = os.environ.get("SYMPTOM_RADAR_DB", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ultrahuman.db"))

def wearer_now():
    """Current time in SYMPTOM_RADAR_TZ, or the host local timezone."""
    name = os.environ.get("SYMPTOM_RADAR_TZ")
    return datetime.now(ZoneInfo(name)) if name else datetime.now().astimezone()


def wearer_midnight(day):
    """Convert a wearer calendar date to midnight, preserving DST boundaries."""
    midnight = datetime.combine(day, datetime.min.time())
    name = os.environ.get("SYMPTOM_RADAR_TZ")
    return midnight.replace(tzinfo=ZoneInfo(name)) if name else midnight.astimezone()
