# Symptom Radar for Ultrahuman Ring 🦠

[![CI](https://github.com/anshdhawann/symptom-radar-ultrahuman/actions/workflows/ci.yml/badge.svg)](https://github.com/anshdhawann/symptom-radar-ultrahuman/actions/workflows/ci.yml)

A **TemPredict-study-inspired** anomaly detection system that monitors your Ultrahuman Ring biometrics and flags physiological strain, using a clean-baseline, trajectory-aware statistical engine applied to your Ultrahuman data.

It compares each morning's vitals against your **personal 21-day clean baseline** (ending 3 days before today, so an ongoing illness doesn't contaminate its own reference), tracks the **rate of rise** across your core metrics, and requires **multi-day persistence** before raising the top alert level, so a single rough night doesn't fire a false alarm.

> **Honest scope.** This is an independent adaptation of the parts of TemPredict (Mason et al., *Nature Scientific Reports* 2022) that don't depend on respiratory rate. The original study reported ~2.75 days of pre-symptomatic lead time, and that lead time came largely from **respiratory rate (RR)**, which the Ultrahuman Partner API does not expose (verified; see [Respiratory rate: the gap](#respiratory-rate-the-gap)). This tool surfaces deviations as they emerge and sustain; it does **not** claim to predict illness days in advance. Measured results, including what it misses, are in [BENCHMARK.md](BENCHMARK.md).

## How It Works

Four ideas, adapted from TemPredict's methodology:

1. **Clean baseline.** Your baseline is the 21 calendar days ending 3 days before today, so today and the 2 days before it (the candidate-sick window) never enter the reference. Days in that window that already look illness-grade (a weighted strain-direction deviation of 3.0 or more against the window's median and MAD) are dropped too, so a long illness cannot become its own reference until it fills about half the window. TemPredict excluded febrile baseline days; this applies the same idea without labels.

2. **Trajectory, not just level.** For each metric, a 5-day least-squares slope over calendar days captures the *rate of rise*. It only boosts a metric that is already more than 0.5σ off its median in the strain direction, so recovering toward normal never counts as strain, and a flat shift to a new normal scores lower than an accelerating climb.

3. **Per-metric noise scaling.** Each metric's deviation is measured against its own robust noise estimate (the larger of the weighted std and 1.4826 × MAD, with a floor per metric), so an erratically variable HRV doesn't dominate a stable temperature. Thresholds become personal.

4. **Persistence.** The highest level requires elevation on at least 2 of the last 3 calendar days. One noisy night stays at Elevated; a multi-day pattern is what triggers Significant. This is the single biggest false-positive reducer (it filters a hangover, a hard workout, one bad night of sleep).

| Metric | Weight | Direction |
|---|---|---|
| Resting Heart Rate | 25% | Elevated = strain signal |
| Sleep HRV | 30% | Depressed = recovery impairment |
| Skin Temp Deviation | 45% | Elevated = fever/inflammation signal |
| Recovery Index | up to +30% (modifier) | Depressed beyond −1σ boosts the score |

> **Why these three?** TemPredict found skin temperature gave the largest single accuracy lift and HRV was the most indispensable once all signals were in. RHR is the classic strain marker. The Ultrahuman API exposes other potentially useful signals, but `hr_drop` is documented and not populated (see [the gap](#respiratory-rate-the-gap)), and adding sleep score or SpO2 only tipped days that were already just under the threshold (see the ablation in [BENCHMARK.md](BENCHMARK.md#7-extra-signals-ablation-exploratory)).

Weights are multipliers on each metric's noise-normalized, strain-direction deviation. The weighted aggregate is the **strain index** (not a raw σ). Levels:

- ✅ **Normal** (strain index < 1.0): biometrics within your personal range
- 🟡 **Elevated** (≥ 1.0): notable deviation, worth watching today (the "Oura flag")
- 🔴 **Significant strain** (≥ 2.0 *and* elevated on 2 of the last 3 days): multi-day elevation, prioritize rest

When a day cannot be scored, the report says so instead of showing Normal:

- ⚪ **No data**: the ring has not synced today
- ⏳ **Calibrating**: fewer than 7 clean baseline days so far, or a baseline for only one of today's metrics
- ⚪ **Insufficient data**: too few core metrics today to call it Normal (a strong deviation on the metrics that are present still raises a flag)

## Respiratory rate: the gap

TemPredict's headline result (~2.75 days of pre-symptomatic detection) depended heavily on **respiratory rate (RR)**, which was one of its five input streams and the basis of its novel physiological-onset alignment. RR is among the strongest early illness signals in the wearable literature.

**The Ultrahuman Partner API does not expose respiratory rate.** This was verified directly:

- The documented endpoint (`/api/v1/partner/daily_metrics`) contains no respiratory, breathing, or breath-related fields across any of its ~30 metric types.
- An older undocumented endpoint (`/api/v1/metrics`, referenced in community dashboards) does carry a `respiratory_graph` schema slot, but it returns `null` for live accounts: Ultrahuman isn't populating it via this API.
- Ultrahuman's "Respiratory Health" features (snoring, coughing, breathing disturbances) are a consumer add-on built with Sleep Cycle using **on-device smartphone audio**, not ring sensors, and are not surfaced through the Partner API.

This is why this tool does not claim pre-symptomatic lead time. It adapts the parts of TemPredict that work without RR: clean baselining, trajectory detection, per-person noise normalization, and multi-day persistence. If Ultrahuman exposes RR in the future, adding it as a weighted metric (and eventually a PX-style alignment) would be the single biggest quality upgrade available.

## Requirements

- An **Ultrahuman Ring** (Ring Air, Ring Pro, etc.)
- An **Ultrahuman API token**: generate one at [vision.ultrahuman.com](https://vision.ultrahuman.com/developer-docs)
- Python 3.10+

## Setup

```bash
# Clone the repo
git clone https://github.com/anshdhawann/symptom-radar-ultrahuman.git
cd symptom-radar-ultrahuman

# Either run from the checkout (requests is the only dependency)...
pip install -r requirements.txt

# ...or install it, which also adds a `symptom-radar` command
pip install .

# Set your API token (or create a .env file)
export ULTRAHUMAN_TOKEN="your-api-token"
```

Alternatively, create a `.env` file in the repo directory (gitignored by default). If both are set, the `.env` file wins:
```env
ULTRAHUMAN_TOKEN="your-api-token"
```

Only the commands that call the Ultrahuman API (the daily check, `--backfill` and the MCP `symptom_radar_report` tool) need the token. Labelling, evaluation, `--help` and the other MCP tools work offline without it.

## Usage

`python3 symptom_radar.py` and the installed `symptom-radar` command take the same flags.

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

"Today" follows the host's timezone. Set `SYMPTOM_RADAR_TZ` (for example `America/Edmonton`) if the machine running the check is not in yours.

### Seed the database (first run only)

```bash
python3 symptom_radar.py --backfill
```

Fetches ~35 days of historical data from the Ultrahuman API to build your baseline immediately.

### Label days and measure accuracy (optional)

```bash
# Log how you actually felt (fine | rough | sick); defaults to today
python3 symptom_radar.py --label rough --label-note "late night, 3 drinks"
python3 symptom_radar.py --label sick --label-date 2026-03-14
python3 symptom_radar.py --labels          # list logged labels

# Optionally log what else happened, with or without a label
python3 symptom_radar.py --label rough --confounders "alcohol=3,travel=true"
python3 symptom_radar.py --confounders "alcohol=0"   # records an alcohol-free day

python3 labels.py --status                 # progress toward the training gate
python3 evaluate.py                        # chronological evaluation against your labels (read-only)
python3 evaluate.py --labels               # do flags separate sick vs rough vs fine?
python3 evaluate.py --db snapshot.db --json  # evaluate a frozen copy, machine-readable
python3 train.py                           # sick-vs-rough classifier (gated: 15 days and 5 episodes per class)
python3 labels.py --verify                 # runs the whole chain: status, scenario, train, evaluate
python3 ablate.py                          # exploratory: one extra signal at a time, defaults unchanged
```

`evaluate.py` treats only your own labels as ground truth: `sick` labels (plus any user-confirmed sick dates in a local, gitignored `episodes.json`) for illness, `rough` or `sick` for strain, and `fine` days as the only true negatives. Unlabelled days get an alert rate, not a false-positive rate. Personal health data never enters the repo. See [BENCHMARK.md](BENCHMARK.md) for measured results and [STATUS.md](STATUS.md) for what is done and what is blocked.

### Automation (cron)

```bash
# Run daily at noon
0 12 * * * python3 /path/to/symptom-radar-ultrahuman/symptom_radar.py
```

## Using with AI Agents

This tool works well with AI coding agents like **Claude Code**, **Codex CLI**, **OpenCode**, and **Hermes Agent**. Here's how to integrate it:

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

The script includes a built-in stdio MCP server (no separate package needed). Add it to your client's MCP config, e.g. for Claude Code:

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

It exposes five tools:
- `symptom_radar_report`: fetch today's data and return the daily report (markdown)
- `symptom_radar_history`: query the local DB (`days`, capped at 366, and `metrics`)
- `symptom_radar_strain`: today's strain level, index, and per-metric breakdown as JSON, without fetching
- `symptom_radar_label`: log a fine / rough / sick label and any confounders
- `symptom_radar_labels`: list the logged labels

Malformed requests get standard JSON-RPC errors (parse error, invalid request, unknown method, invalid params, internal error) and the server keeps serving. The agent can then answer correlation questions ("does my HRV drop when I sleep less?") and surface your strain history conversationally.

### As a cron + Telegram delivery (Hermes Agent)

If you use Hermes Agent, set a cron job that runs the script daily and delivers the report to Telegram:

```
Schedule: 0 12 * * *
Prompt: Run symptom_radar.py and deliver the report
```

## Data Storage

All data is stored locally in `ultrahuman.db` (SQLite, gitignored) next to the script: daily snapshots, labels, confounders, and an audit of every metric value Ultrahuman revises later (`metric_revisions`). Re-fetching a day never blanks a value that is already stored. Set `SYMPTOM_RADAR_DB` to change the path:

```bash
export SYMPTOM_RADAR_DB="/path/to/custom.db"
```

## Project Structure

```
symptom-radar-ultrahuman/
├── symptom_radar.py       # Entry script: run from a checkout without installing
├── symptomradar/          # The package
│   ├── cli.py             # Command-line flags (also the `symptom-radar` command)
│   ├── config.py          # Paths, wearer timezone, token lookup for API calls
│   ├── client.py          # Ultrahuman Partner API calls and payload extraction
│   ├── storage.py         # SQLite schema, snapshots, labels, confounders, revision audit
│   ├── engine.py          # Pure strain scoring: baseline, trajectory, persistence
│   ├── report.py          # Daily digest and backfill
│   └── mcp.py             # Stdio MCP server
├── evaluate.py            # Chronological evaluation against your labels (read-only)
├── ablate.py              # Exploratory extra-signal ablation
├── labels.py              # Label status, natural-language backfill, one-command verify chain
├── train.py               # Sick-vs-rough classifier, leave-one-episode-out, gated
├── scenario.py            # Sensitivity analysis for unconfirmed episodes
├── test_*.py              # Tests (pytest; no network, no token)
├── pyproject.toml         # Package metadata and the console command
├── requirements.txt       # Dependencies for running from a checkout
├── .github/workflows/     # CI: tests on Python 3.10 to 3.14
├── BENCHMARK.md           # Measured results vs published TemPredict numbers
├── STATUS.md              # What is done, what is blocked, why
├── LICENSE                # MIT
├── .gitignore             # .env, *.db, episodes.json, build output
└── .env                   # Your API token (gitignored, you create this)
```

Run the tests with `python3 -m pytest` (no token, no network calls). CI runs them on Python 3.10 to 3.14.

## Legal & Attribution

### Trademark Notice

This project is **not affiliated with, endorsed by, or connected to Oura Health Oy** or Ultrahuman. "Oura" and "Symptom Radar" are trademarks of Oura Health Oy. "Ultrahuman" is a trademark of Ultrahuman Healthcare Pvt. Ltd. This project uses the Ultrahuman API under its standard developer terms; it is not an official Ultrahuman product.

### Research Attribution

The strain detection approach is based on the **TemPredict study** (Mason et al., *Detection of COVID-19 using multimodal data from a wearable device*, Scientific Reports 12, 3463, 2022), an open-access publication that demonstrated pre-symptomatic illness detection using wearable biometric data. This work was conducted by the University of California San Francisco and MIT Lincoln Laboratory. Read the paper: [nature.com/articles/s41598-022-07314-0](https://www.nature.com/articles/s41598-022-07314-0)

### No IP Infringement

- **No code, models, or data** from Oura's Symptom Radar feature are used in this project.
- The z-score anomaly detection method is a standard statistical technique (in the public domain since the 1920s) applied to biometric data, not a proprietary algorithm.
- This project reads data exclusively from the **Ultrahuman Partner API** under terms provided by Ultrahuman.
- All analysis is performed using published academic methodology (TemPredict).

### Medical Disclaimer

This tool is **not a medical device**. It does not diagnose, cure, mitigate, treat, or prevent any disease. The strain assessment is a statistical deviation score; it does not replace professional medical advice. Always consult a healthcare provider about health concerns.

## License

MIT: free to use, modify, and share. No warranty, express or implied.
