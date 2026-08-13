#!/usr/bin/env python3
"""
Historical self-report labels for the Symptom Radar engine.

These seed the daily_labels table with known history so evaluate.py can
measure whether the model separates real sickness from strain.

Label meaning (used by the engine):
  fine  — felt normal
  rough — tired / stressed / poor sleep (strain WITHOUT illness)
  sick  — actually felt ill

SEED_LABELS below are SYNTHETIC EXAMPLE DATA, provided so the file runs
out of the box. Replace them with your own retrospective labels before
using --seed on real data (see --from-memory / --review for collecting
yours).

Usage:
    python3 labels.py --seed        # insert the seeds (idempotent)
    python3 labels.py               # print current labels
"""
import argparse
import sqlite3
import sys

import symptom_radar as sr

# (date, label, note, confidence)  — SYNTHETIC EXAMPLE DATA, replace freely
SEED_LABELS = [
    # ── Example confirmed sick window ───────────────────────────────────
    ("2024-03-10", "sick",  "example: felt ill, low energy", "example"),
    ("2024-03-11", "sick",  "example: still unwell (no ring data)", "example"),
    ("2024-03-12", "sick",  "example: recovering, felt ill", "example"),
    # ── Example strain (NOT illness) days ───────────────────────────────
    ("2024-02-15", "rough", "example: poor sleep, tired (RHR elevated)", "example"),
    ("2024-02-16", "rough", "example: rough day, low recovery", "example"),
    ("2024-03-01", "rough", "example: stressed / poor sleep", "example"),
    # ── Example fine days (healthy reference class) ─────────────────────
    ("2024-02-01", "fine", "example: all vitals at personal mean", "example"),
    ("2024-02-05", "fine", "example: all vitals at personal mean", "example"),
    ("2024-02-08", "fine", "example: all vitals at personal mean", "example"),
    ("2024-02-20", "fine", "example: all vitals at personal mean", "example"),
    ("2024-02-27", "fine", "example: all vitals at personal mean", "example"),
    ("2024-03-05", "fine", "example: all vitals at personal mean", "example"),
]


def seed(verbose=True):
    """Insert SEED_LABELS idempotently. Returns count of new labels written."""
    existing = {d for d, _, _ in sr.get_labels()}
    added = 0
    for date, label, note, _conf in SEED_LABELS:
        if date in existing:
            continue
        sr.log_label(date, label, note)
        added += 1
        if verbose:
            print(f"  + {date}: {label}  ({note})")
    return added


MIN_SICK = 15
MIN_ROUGH = 15


def review_candidates():
    """List candidate days worth labeling from memory.

    Every day with a flagged strain level or notable strain index is shown
    with its key biometrics, so the user can quickly assign fine/rough/sick
    from memory. This is the fast path to 15+ sick / 15+ rough labels —
    retrospective labels from months of data beat 6-12 weeks of nightly
    prompts.
    """
    import datetime as _dt
    conn = sqlite3.connect(sr.DB_PATH)
    cur = conn.execute("SELECT * FROM daily_snapshots ORDER BY date ASC")
    cols = [d[0] for d in cur.description]
    all_days = [dict(zip(cols, r)) for r in cur.fetchall()]
    already = {d for d, _, _ in sr.get_labels()}
    conn.close()

    print("Candidate days to label from memory (already-labeled excluded):")
    print(f"{'Date':<12} {'Day':<4} {'RHR':>5} {'HRV':>5} {'Temp':>6} {'Rec':>5} "
          f"{'Slp':>5} {'strain':>6}  hint")
    print("-" * 92)

    import re
    shown = 0
    for i in range(len(all_days)):
        d = all_days[i]
        date = d["date"]
        if date in already:
            continue
        lvl, detail = sr.assess_strain(all_days[:i + 1])
        m_idx = re.search(r"Strain index: ([\d.]+)", detail)
        strain = float(m_idx.group(1)) if m_idx else 0.0
        # Show: flagged days (lvl>=1) + strain >= 0.5 (borderline-worthy)
        if lvl < 1 and strain < 0.5:
            continue
        wd = _dt.date.fromisoformat(date).strftime("%a")
        rhr = d.get("night_rhr") or d.get("sleep_rhr")
        temp = d.get("temp_deviation")
        temp_s = ("%+.2f" % temp) if temp is not None else "-"
        hint = sr.STRAIN_ICONS.get(lvl, "") 
        if "Recovery↓" in detail:
            hint += " rec-decline"
        print(f"{date:<12} {wd:<4} {str(rhr or '-'):>5} "
              f"{str(d.get('avg_sleep_hrv') or '-'):>5} {temp_s:>6} "
              f"{str(d.get('recovery_index') or '-'):>5} "
              f"{str(d.get('sleep_score') or '-'):>5} {strain:>6.2f}  {hint.strip()}")
        shown += 1

    print()
    print(f"{shown} candidate days above. For each you remember, log:")
    print("  python3 symptom_radar.py --label <fine|rough|sick> "
          "--label-date YYYY-MM-DD --label-note 'remembered'")
    print("Target: 15 sick + 15 rough total (currently "
          f"{sum(1 for _, l, _ in sr.get_labels() if l == 'sick')} sick / "
          f"{sum(1 for _, l, _ in sr.get_labels() if l == 'rough')} rough).")

def bulk_from_memory(text):
    """Parse a natural-language retrospective answer and log all labels.

    Accepts flexible input, e.g.:
      "Jan 15 sick, Feb 18 sick, Mar 26-27 rough"
      "may 22 = sick; jun 18 sick; 2024-06-26 rough, 2024-06-27 rough"
      "don't remember" / "not sure" → no-op (honestly reported)
    Parsing: finds date tokens (YYYY-MM-DD or Mon DD) and the closest
    following label word (fine/rough/sick). Returns list of logged labels.
    """
    import re
    text_l = text.lower()
    if any(w in text_l for w in ("don't remember", "dont remember", "not sure",
                                 "no idea", "forget", "unknown")):
        print("Understood — no retrospective labels added. The nightly cron "
              "keeps collecting from today forward.")
        return []

    # Normalize month names
    months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
              "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

    def expand_dates(date_tokens):
        """Expand 'Mar 26-27' → ['2024-03-26','2024-03-27'] and similar."""
        out = []
        for tok in date_tokens:
            m = re.match(r"(\d{4})-(\d{2})-(\d{2})$", tok)
            if m:
                out.append(tok)
                continue
            m = re.match(r"([a-z]{3})\.?\s+(\d{1,2})(?:-(\d{1,2}))?$", tok)
            if m:
                mon = months[m.group(1)]
                day1 = int(m.group(2))
                out.append(f"2024-{mon:02d}-{day1:02d}")
                if m.group(3):
                    day2 = int(m.group(3))
                    out.extend(f"2024-{mon:02d}-{d:02d}"
                               for d in range(day1, day2 + 1))
        return out

    # Find (date_token, label) pairs by scanning for date tokens then the
    # nearest following label word.
    date_re = re.compile(
        r"(\d{4}-\d{2}-\d{2}|[a-z]{3}\.?\s+\d{1,2}(?:-\d{1,2})?)")
    label_re = re.compile(r"\b(fine|rough|sick)\b")
    date_tokens = date_re.findall(text_l)
    if not date_tokens:
        print(f"Couldn't find any dates in: {text!r}")
        print("Expected e.g.: 'Jan 15 sick, Feb 18 sick, Mar 26-27 rough'")
        return []

    # For each date token, find the label that appears after it (and before
    # the next date token, if any). Dedupe expanded ranges ('Mar 26-27'
    # expands to 26 AND 27, and 'Mar 26' alone also matches — drop repeats).
    segments = []
    positions = [m.start() for m in date_re.finditer(text_l)]
    seen = set()
    for i, tok in enumerate(date_tokens):
        start = positions[i]
        end = positions[i + 1] if i + 1 < len(positions) else len(text_l)
        seg = text_l[start:end]
        lm = label_re.search(seg)
        label = lm.group(1) if lm else None
        for date in expand_dates([tok]):
            if date in seen:
                continue
            seen.add(date)
            segments.append((date, label))

    logged = []
    for date, label in segments:
        if label is None:
            print(f"  ? {date}: no label found next to it — skipped")
            continue
        sr.log_label(date, label, "remembered retrospectively")
        logged.append((date, label))
        print(f"  + {date}: {label}")

    if logged:
        print(f"\nLogged {len(logged)} retrospective labels.")
        print("Re-run: python3 labels.py --status  |  python3 scenario.py")
    else:
        print("Nothing logged.")
    return logged


def status():
    """Print collection progress vs. the classifier's training threshold."""
    labels = sr.get_labels()
    counts = {"fine": 0, "rough": 0, "sick": 0}
    for _, lbl, _ in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    print("Label collection status:")
    print(f"  fine : {counts['fine']}")
    print(f"  rough: {counts['rough']}  (need {MIN_ROUGH} → "
          f"{max(0, MIN_ROUGH - counts['rough'])} more)")
    print(f"  sick : {counts['sick']}  (need {MIN_SICK} → "
          f"{max(0, MIN_SICK - counts['sick'])} more)")
    print()
    ready = counts["sick"] >= MIN_SICK and counts["rough"] >= MIN_ROUGH
    if ready:
        print("✅ Enough labels to train the separation classifier: "
              "python3 train.py")
    else:
        print(f"Not ready to train yet: {max(0, MIN_SICK - counts['sick'])} more "
              f"sick + {max(0, MIN_ROUGH - counts['rough'])} more rough days.")
        print("The nightly 21:00 cron is collecting these automatically.")
    return ready


def run_verify():
    """Run the full sickness-vs-strain measurement chain in one command.

    Equivalent to (in order):
        python3 labels.py --status
        python3 scenario.py
        python3 train.py --force
        python3 evaluate.py --labels
    """
    import subprocess, os
    here = os.path.dirname(os.path.abspath(__file__))
    env = {**os.environ, "SYMPTOM_RADAR_DB": sr.DB_PATH}

    print("=" * 78)
    print("FULL VERIFICATION CHAIN")
    print("=" * 78)
    for step, cmd in [
        ("1/4 label status", f"python3 {here}/labels.py --status"),
        ("2/4 sensitivity",  f"python3 {here}/scenario.py"),
        ("3/4 train",        f"python3 {here}/train.py --force"),
        ("4/4 separation",   f"python3 {here}/evaluate.py --labels"),
    ]:
        print(f"\n--- {step}: {cmd} ---")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
        print(r.stdout.rstrip() if r.stdout else "")
        if r.returncode != 0 and r.stderr:
            print("stderr:", r.stderr.strip()[:500])


def main():
    ap = argparse.ArgumentParser(description="Historical self-report labels")
    ap.add_argument("--seed", action="store_true",
                    help="Insert the known historical labels (idempotent)")
    ap.add_argument("--status", action="store_true",
                    help="Show collection progress vs training threshold")
    ap.add_argument("--review", action="store_true",
                    help="List unlabeled candidate days to label from memory")
    ap.add_argument("--from-memory", metavar="TEXT",
                    help="Log retrospective labels from a natural-language "
                         "answer, e.g. 'Jan 15 sick, Feb 18 sick, Mar 26-27 rough'")
    ap.add_argument("--verify", action="store_true",
                    help="Run the full measurement chain in one command: "
                         "status → scenario → train --force → evaluate --labels")
    args = ap.parse_args()

    if args.verify:
        run_verify()
    elif args.from_memory:
        bulk_from_memory(args.from_memory)
    elif args.review:
        review_candidates()
    elif args.status:
        status()
    elif args.seed:
        n = seed()
        print(f"\nSeeded {n} new labels ({len(SEED_LABELS)} known, "
              f"{len(SEED_LABELS) - n} already present).")
    else:
        labels = sr.get_labels()
        if not labels:
            print("No labels in DB. Run: python3 labels.py --seed")
            return
        print(f"{'Date':<12} {'Label':<8} Confidence   Note")
        by_date = {d: (l, n) for d, l, n in labels}
        for date, label, note, conf in SEED_LABELS:
            print(f"{date:<12} {label:<8} {conf:<12} {note}")
        # any extra labels beyond seeds
        for date, label, note in labels:
            if date not in by_date:
                print(f"{date:<12} {label:<8} (logged)     {note or ''}")


if __name__ == "__main__":
    main()
