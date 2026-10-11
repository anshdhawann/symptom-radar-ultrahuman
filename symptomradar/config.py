"""Local paths, wearer time, and credentials resolved only for API calls."""
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
BASE_URL = "https://partner.ultrahuman.com/api/v1/partner/daily_metrics"
DB_PATH = os.environ.get("SYMPTOM_RADAR_DB", str(ROOT / "ultrahuman.db"))


class MissingTokenError(RuntimeError):
    pass


def require_token(env_path=None):
    """Prefer the process environment, then the checkout's optional env file."""
    token = os.environ.get("ULTRAHUMAN_TOKEN")
    if not token:
        path = Path(env_path) if env_path is not None else ENV_PATH
        if path.exists():
            with path.open() as stream:
                for line in stream:
                    key, sep, value = line.strip().partition("=")
                    if sep and key == "ULTRAHUMAN_TOKEN":
                        token = value.strip("\"'")
                        break
    if not token:
        raise MissingTokenError("ULTRAHUMAN_TOKEN is not set; API commands require a token.")
    return token


def wearer_now():
    """Current time in SYMPTOM_RADAR_TZ, or the host local timezone."""
    name = os.environ.get("SYMPTOM_RADAR_TZ")
    return datetime.now(ZoneInfo(name)) if name else datetime.now().astimezone()


def wearer_midnight(day):
    """Convert a wearer calendar date to midnight, preserving DST boundaries."""
    midnight = datetime.combine(day, datetime.min.time())
    name = os.environ.get("SYMPTOM_RADAR_TZ")
    return midnight.replace(tzinfo=ZoneInfo(name)) if name else midnight.astimezone()
