# Symptom Radar for Ultrahuman Ring 🦠

A **TemPredict-study-inspired** anomaly detection system that monitors your Ultrahuman Ring biometrics and flags physiological strain — using a clean-baseline, trajectory-aware statistical engine applied to your Ultrahuman data.

It compares each morning's vitals against your **personal 21-day clean baseline** (excluding the last 3 days so an ongoing illness doesn't contaminate its own reference), tracks the **rate of rise** across your core metrics, and requires **multi-day persistence** before elevating the alert level — so a single rough night doesn't fire a false alarm.

> **Honest scope.** This is an independent adaptation of the parts of TemPredict (Mason et al., *Nature Scientific Reports* 2022) that don't depend on respiratory rate. The original study reported ~2.75 days of pre-symptomatic lead time, and that lead time came largely from **respiratory rate (RR)** — which the Ultrahuman Partner API does not expose (verified; see [Respiratory rate — the gap](#respiratory-rate--the-gap)). This tool surfaces deviations as they emerge and sustain; it does **not** claim to predict illness days in advance. Built for the Ultrahuman Ring.

## How It Works

Four ideas, adapted from TemPredict's methodology:

1. **Clean baseline.** Your 21-day baseline excludes the most recent 3 days (candidate-sick window). TemPredict excluded febrile baseline days; we exclude the recent guard proactively, so an illness that's been brewing for 2 days doesn't inflate the mean/std and suppress today's score.

2. **Trajectory, not just level.** For each metric, a 5-day least-squares slope captures the *rate of rise* — the real illness signature. A flat shift to a new normal (lifestyle) scores lower than an accelerating climb (illness-shaped).

3. **Per-metric noise scaling.** Each metric's deviation is measured against its own robust noise floor (the larger of weighted-std and MAD-scaled), so an erratically-variable HRV doesn't dominate a stable temperature. Thresholds become personal.

4. **Persistence.** The highest level requires sustained elevation across ≥2 of the last 3 days. One noisy night stays at Elevated; a multi-day pattern across multiple metrics is what triggers Significant. This is the single biggest false-positive reducer (filters hangover, exercise, one bad night of sleep).

| Metric | Weight | Direction |
|---|---|---|
| Resting Heart Rate | 25% | Elevated = strain signal |
| Sleep HRV | 30% | Depressed = recovery impairment |
| Skin Temp Deviation | 45% | Elevated = fever/inflammation signal |
| Recovery Index | up to +30% (modifier) | Depressed beyond −1σ boosts the score |

> **Why these three?** TemPredict found skin temperature gave the largest single accuracy lift and HRV was the most indispensable once all signals were in. RHR is the classic strain marker. The Ultrahuman API exposes several other potentially-useful signals (`hr_drop`, SpO2) but `hr_drop` is documented-but-not-populated (see [the gap](#respiratory-rate--the-gap)) and SpO2 is too narrowly ranged to drive early detection.

Weights are multipliers on each metric's noise-normalized, strain-direction deviation. The weighted aggregate is the **strain index** (not a raw σ). Levels:

- ✅ **Normal** (strain index < 1.0) — Biometrics within your personal range
- 🟡 **Elevated** (≥ 1.0) — Notable deviation, worth watching today (the "Oura flag")
- 🔴 **Significant strain** (≥ 2.0 *and* sustained ≥2 of last 3 days) — Multi-day elevation across metrics, prioritize rest

## Respiratory rate — the gap

TemPredict's headline result (~2.75 days of pre-symptomatic detection) depended heavily on **respiratory rate (RR)**, which was one of its five input streams and the basis of its novel physiological-onset alignment. RR is among the strongest early illness signals in the wearable literature.

**The Ultrahuman Partner API does not expose respiratory rate.** This was verified directly:

- The documented endpoint (`/api/v1/partner/daily_metrics`) contains no respiratory, breathing, or breath-related fields across any of its ~30 metric types.
- An older undocumented endpoint (`/api/v1/metrics`, referenced in community dashboards) does carry a `respiratory_graph` schema slot, but it returns `null` for live accounts — Ultrahuman isn't populating it via this API.
- Ultrahuman's "Respiratory Health" features (snoring, coughing, breathing disturbances) are a consumer add-on built with Sleep Cycle using **on-device smartphone audio**, not ring sensors, and are not surfaced through the Partner API.

This is why this tool does not claim pre-symptomatic lead time. It adapts the parts of TemPredict that work without RR: clean baselining, trajectory detection, per-person noise normalization, and multi-day persistence. If Ultrahuman exposes RR in the future, adding it as a weighted metric (and eventually a PX-style alignment) would be the single biggest quality upgrade available.

## Requirements

- An **Ultrahuman Ring** (Ring Air, Ring Pro, etc.)
- An **Ultrahuman API token** — generate one at [vision.ultrahuman.com](https://vision.ultrahuman.com/developer-docs)
- Python 3.10+

## Setup

```bash
# Clone the repo
git clone https://github.com/anshdhawann/symptom-radar-ultrahuman.git
cd symptom-radar-ultrahuman

# Install dependencies (requests only — everything else is stdlib)
pip install -r requirements.txt

# Set your API token (or create a .env file)
export ULTRAHUMAN_TOKEN="your..."
```

Alternatively, create a `.env` file in the repo directory (gitignored by default):
```env
ULTRAHUMAN_TOKEN="your...
```

## Usage

### Daily check (stores data + prints report)

```bash
python3 symptom_radar.py
```

Output:
```
## 🩸 Ultrahuman Daily

**🦠 Symptom Radar**
**✅ Normal**
🟢 *Biometrics within normal range*

**😴 Sleep**
Score: 78/100 | Total: 312 min | Eff: 93%
Deep: 90 min | Light: 147 min | REM: 75 min
Cycles: 3 | Restorative: 49%
Sleep HRV: 36 | RHR: 62 bpm
Body Temp: 35.91°C (Δ+0.2°C)
SPO2: 98% | Tosses: 3

**💪 Recovery & Activity**
Recovery: 69/100 | Movement: 34/100
Active: 0 min | Inactive: 363 min
Total Steps: 2041

**❤️ Vitals**
HR: avg 70.7 bpm (57–100)
HRV: avg 59.8 ms (15–245)
Skin Temp: avg 34.0°C (27.5–36.3)

📊 Baseline: 22 days of data
```

### Seed the database (first run only)

```bash
python3 symptom_radar.py --backfill
```

Fetches ~35 days of historical data from the Ultrahuman API to build your baseline immediately.

### Automation (cron)

```bash
# Run daily at noon
0 12 * * * cd /path/to/symptom-radar-ultrahuman && python3 symptom_radar.py
```

## Using with AI Agents

This tool works great as a tool for AI coding agents like **Claude Code**, **Codex CLI**, **OpenCode**, and **Hermes Agent**. Here's how to integrate it:

### As a shell command (any agent)

```bash
# Agent fetches and reads your daily report
python3 /path/to/symptom_radar.py

# Agent asks questions about your data
python3 -c "
import sqlite3
conn = sqlite3.connect('/path/to/ultrahuman.db')
# Agent queries your history
cur = conn.execute('SELECT date, sleep_score, night_rhr, avg_sleep_hrv, temp_deviation FROM daily_snapshots ORDER BY date DESC LIMIT 14')
for row in cur.fetchall():
    print(row)
"
```

### As an MCP tool (Claude, Hermes, Cursor)

The script includes a built-in stdio MCP server (no separate package needed). Add it to your client's MCP config — e.g. for Claude Code:

```json
{
  "mcpServers": {
    "symptom-radar": {
      "command": "python3",
      "args": ["/absolute/path/to/symptom_radar.py", "--mcp"]
    }
  }
}
```

It exposes three tools:
- `symptom_radar_report` — fetch today's data and return the daily report (markdown)
- `symptom_radar_history` — query the local DB (`days`, `metrics`)
- `symptom_radar_strain` — today's strain level, index, and per-metric breakdown

The agent can then answer correlation questions ("does my HRV drop when I sleep less?") and surface your strain history conversationally.

### As a cron + Telegram delivery (Hermes Agent)

If you use Hermes Agent, set a cron job that runs the script daily and delivers the report to Telegram:

```
Schedule: 0 12 * * *
Prompt: Run symptom_radar.py and deliver the report
```

## Data Storage

All data is stored locally in `ultrahuman.db` (SQLite, gitignored). Set `SYMPTOM_RADAR_DB` to change the path:

```bash
export SYMPTOM_RADAR_DB="/path/to/custom.db"
```

## Project Structure

```
symptom-radar-ultrahuman/
├── symptom_radar.py   # Main script: fetch, store, assess, report
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT
├── .gitignore         # .env, *.db, __pycache__
└── .env               # Your API token (gitignored, you create this)
```

## Legal & Attribution

### Trademark Notice

This project is **not affiliated with, endorsed by, or connected to Oura Health Oy** or Ultrahuman. "Oura" and "Symptom Radar" are trademarks of Oura Health Oy. "Ultrahuman" is a trademark of Ultrahuman Healthcare Pvt. Ltd. This project uses the Ultrahuman API under its standard developer terms — it is not an official Ultrahuman product.

### Research Attribution

The strain detection approach is based on the **TemPredict study** (Mason et al., *Detection of COVID-19 using multimodal data from a wearable device*, Scientific Reports 12, 3463, 2022), an open-access publication that demonstrated pre-symptomatic illness detection using wearable biometric data. This work was conducted by the University of California San Francisco and MIT Lincoln Laboratory. Read the paper: [nature.com/articles/s41598-022-07314-0](https://www.nature.com/articles/s41598-022-07314-0)

### No IP Infringement

- **No code, models, or data** from Oura's Symptom Radar feature are used in this project.
- The z-score anomaly detection method is a standard statistical technique (in the public domain since the 1920s) applied to biometric data — not a proprietary algorithm.
- This project reads data exclusively from the **Ultrahuman Partner API** under terms provided by Ultrahuman.
- All analysis is performed using published academic methodology (TemPredict).

### Medical Disclaimer

This tool is **not a medical device**. It does not diagnose, cure, mitigate, treat, or prevent any disease. The strain assessment is a statistical deviation score — it does not replace professional medical advice. Always consult a healthcare provider about health concerns.

## License

MIT — free to use, modify, and share. No warranty, express or implied.
