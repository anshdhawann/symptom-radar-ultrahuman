from . import config, mcp, report, storage
import argparse

def main():
    parser = argparse.ArgumentParser(description="Symptom Radar for Ultrahuman Ring")
    parser.add_argument("--backfill", type=int, nargs="?",
                        const=35, metavar="DAYS",
                        help="Backfill historical data (default: 35 days)")
    parser.add_argument("--mcp", action="store_true",
                        help="Run as a stdio MCP server (for Claude / Hermes / Cursor)")
    parser.add_argument("--label", metavar="STATUS",
                        choices=storage.LABELS,
                        help="Log today's self-report: fine / rough / sick "
                             "(the training signal that lets the engine learn "
                             "your illness-vs-alcohol distinction)")
    parser.add_argument("--label-date", metavar="YYYY-MM-DD", default=None,
                        help="Date for --label (defaults to today in the wearer timezone)")
    parser.add_argument("--label-note", metavar="TEXT", default=None,
                        help="Optional note for --label (e.g. 'hangover', 'fever')")
    parser.add_argument("--confounders", help="alcohol=3,exercise,travel,sleep_loss (exercise means hard exercise); =false or alcohol=0 records absence")
    parser.add_argument("--labels", action="store_true",
                        help="Print all logged self-report labels")
    args = parser.parse_args()

    if not (args.mcp or args.labels or args.label or args.confounders is not None):
        try:
            config.require_token()
        except config.MissingTokenError as exc:
            parser.exit(1, f"{exc}\n")

    if args.mcp:
        mcp.run_mcp_server()
    elif args.backfill:
        report.backfill(args.backfill)
    elif args.label or args.confounders is not None:
        date_str = args.label_date or config.wearer_now().strftime("%Y-%m-%d")
        try:
            storage.log_label(date_str, args.label, args.label_note, confounders=args.confounders)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"✅ Logged {date_str}: {args.label or 'unlabeled'}"
              + (f" ({args.label_note})" if args.label_note else "")
              + (f" | {args.confounders}" if args.confounders else ""))
    elif args.labels:
        storage.print_labels()
    else:
        digest = report.build_report()
        print(digest)
