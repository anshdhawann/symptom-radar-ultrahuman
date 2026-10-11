from . import config
import sqlite3, math
DB_PATH = config.DB_PATH
from datetime import datetime, timezone

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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metric_revisions (
            id INTEGER PRIMARY KEY,
            date TEXT NOT NULL,
            field TEXT NOT NULL,
            old_value REAL NOT NULL,
            new_value REAL NOT NULL,
            fetched_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_confounders (
            date TEXT PRIMARY KEY,
            alcohol INTEGER CHECK (alcohol IS NULL OR (typeof(alcohol) = 'integer' AND alcohol >= 0)),
            exercise INTEGER CHECK (exercise IS NULL OR exercise IN (0, 1)),
            travel INTEGER CHECK (travel IS NULL OR travel IN (0, 1)),
            sleep_loss INTEGER CHECK (sleep_loss IS NULL OR sleep_loss IN (0, 1))
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

CONFOUNDERS = ("alcohol", "exercise", "travel", "sleep_loss")


def parse_confounders(value):
    """Parse explicit exposure fields. Omitted fields remain unrecorded."""
    if value is None:
        return {}
    if isinstance(value, str):
        fields = {}
        for item in value.split(","):
            key, sep, raw = item.strip().partition("=")
            if key in fields:
                raise ValueError(f"Duplicate confounder: {key}")
            if key == "alcohol":
                if not sep or not raw.isascii() or not raw.isdigit():
                    raise ValueError("alcohol must be a non-negative integer drink count")
                fields[key] = int(raw)
            elif key in CONFOUNDERS:
                if sep and raw not in ("true", "false"):
                    raise ValueError(f"{key} must be true or false")
                fields[key] = raw != "false"
            else:
                raise ValueError(f"Unknown confounder: {key}")
        value = fields
    if not isinstance(value, dict):
        raise ValueError("confounders must be a string or object")
    for key, val in value.items():
        if key not in CONFOUNDERS:
            raise ValueError(f"Unknown confounder: {key}")
        if key == "alcohol":
            if type(val) is not int or val < 0:
                raise ValueError("alcohol must be a non-negative integer drink count")
        elif type(val) is not bool:
            raise ValueError(f"{key} must be true or false")
    return dict(value)


def log_label(date_str, label, note=None, confounders=None):
    """Atomically log a feeling and/or exposures. Supplied fields overwrite."""
    datetime.strptime(date_str, "%Y-%m-%d")
    if label is not None and label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")
    fields = parse_confounders(confounders)
    if label is None and not fields:
        raise ValueError("Provide a label or confounders")
    conn = init_db()
    try:
        with conn:
            if label is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO daily_labels (date, label, note) VALUES (?,?,?)",
                    (date_str, label, note))
            if fields:
                keys = list(fields)
                conn.execute(
                    f"INSERT INTO daily_confounders (date, {', '.join(keys)}) "
                    f"VALUES ({', '.join('?' for _ in range(len(keys) + 1))}) "
                    f"ON CONFLICT(date) DO UPDATE SET {', '.join(f'{k}=excluded.{k}' for k in keys)}",
                    [date_str, *fields.values()])
    finally:
        conn.close()
    return True


def get_confounders():
    conn = init_db()
    try:
        return {row[0]: {key: (val if key == "alcohol" or val is None else bool(val))
                         for key, val in zip(CONFOUNDERS, row[1:])}
                for row in conn.execute("SELECT date, alcohol, exercise, travel, sleep_loss FROM daily_confounders ORDER BY date")}
    finally:
        conn.close()


def print_labels():
    labels = {day: (label, note) for day, label, note in get_labels()}
    confounders = get_confounders()
    if not labels and not confounders:
        print("No labels or confounders logged yet.")
        return
    print(f"{'Date':<12} {'Label':<10} Note | Confounders")
    for day in sorted(labels.keys() | confounders.keys()):
        label, note = labels.get(day, ("unlabeled", None))
        fields = ",".join(f"{key}={str(val).lower()}" for key, val in confounders.get(day, {}).items() if val is not None)
        print(f"{day:<12} {label:<10} {note or ''} | {fields}")

def get_labels():
    """Return [(date, label, note), ...] ordered by date."""
    conn = init_db()
    rows = conn.execute(
        "SELECT date, label, note FROM daily_labels ORDER BY date").fetchall()
    conn.close()
    return rows

def _valid_metric_value(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def store_snapshot(conn, date_str, data):
    columns = [r[1] for r in conn.execute("PRAGMA table_info(daily_snapshots)")
               if r[1] not in ("date", "created_at")]
    values = {key: data[key] for key in columns
              if _valid_metric_value(data.get(key))}
    if not values:
        return False
    with conn:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute("SELECT * FROM daily_snapshots WHERE date=?", (date_str,))
        row = cur.fetchone()
        previous = dict(zip([d[0] for d in cur.description], row)) if row else {}
        fetched_at = datetime.now(timezone.utc).isoformat()
        for field, value in values.items():
            old_value = previous.get(field)
            if old_value is not None and old_value != value:
                conn.execute(
                    "INSERT INTO metric_revisions "
                    "(date, field, old_value, new_value, fetched_at) VALUES (?,?,?,?,?)",
                    (date_str, field, old_value, value, fetched_at))
        fields = list(values)
        updates = ", ".join(f"{field}=excluded.{field}" for field in fields)
        placeholders = ", ".join("?" for _ in range(len(fields) + 1))
        conn.execute(
            f"INSERT INTO daily_snapshots (date, {', '.join(fields)}) "
            f"VALUES ({placeholders}) ON CONFLICT(date) DO UPDATE SET {updates}",
            [date_str] + [values[field] for field in fields])
    return True

def get_recent(conn, days=30):
    cur = conn.execute("""
        SELECT * FROM daily_snapshots
        ORDER BY date DESC LIMIT ?
    """, (days,))
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    rows.reverse()
    return [dict(zip(cols, r)) for r in rows]
