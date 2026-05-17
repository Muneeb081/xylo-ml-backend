"""
Generate Monthly Reports — Cron Job Script
==========================================
Runs automatically on the 1st of every month (scheduled in render.yaml).
Reads the previous month's energy_logs from Firestore for every home,
generates ML reports, and writes them back to ml_reports/{YYYY-MM}.

Usage:
    python generate_reports.py                  # previous month (auto)
    python generate_reports.py --year 2026 --month 5   # specific month

Render Cron schedule:  0 2 1 * *  (02:00 UTC on 1st of each month)
"""
import os
import sys
import json
import argparse
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def get_previous_month(ref: date = None) -> tuple[int, int]:
    """Return (year, month) for the previous calendar month."""
    if ref is None:
        ref = date.today()
    if ref.month == 1:
        return ref.year - 1, 12
    return ref.year, ref.month - 1


def init_firebase():
    """Initialize Firebase from env vars or local key file."""
    sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    db_url  = os.environ.get("FIREBASE_DATABASE_URL", "").strip()

    if not sa_json:
        key_path = Path("C:/Users/munee/Downloads/0a3b07ee-5667-404f-8b76-7181bddf1eed")
        if key_path.exists():
            sa_json = key_path.read_text()
            db_url  = "https://xylo-switch-default-rtdb.asia-southeast1.firebasedatabase.app"
        else:
            print("ERROR: FIREBASE_SERVICE_ACCOUNT_JSON not set.")
            sys.exit(1)

    import firebase_admin
    from firebase_admin import credentials

    if firebase_admin._DEFAULT_APP_NAME not in firebase_admin._apps:
        cred = credentials.Certificate(json.loads(sa_json))
        firebase_admin.initialize_app(cred, {"databaseURL": db_url})


def main():
    parser = argparse.ArgumentParser(description="Generate ML monthly reports")
    parser.add_argument("--year",  type=int, default=None)
    parser.add_argument("--month", type=int, default=None)
    parser.add_argument("--home",  type=str, default=None, help="Specific home ID (default: all)")
    args = parser.parse_args()

    # Determine target month
    if args.year and args.month:
        year, month = args.year, args.month
    else:
        year, month = get_previous_month()

    print("=" * 60)
    print(f"  Monthly Report Generator")
    print(f"  Period: {datetime(year, month, 1).strftime('%B %Y')} ({year}-{month:02d})")
    print("=" * 60)

    # Init Firebase
    init_firebase()

    from api.firestore_client   import get_all_home_ids, read_energy_logs, write_ml_report
    from models_src.monthly_report import generate_monthly_report

    # Get homes to process
    if args.home:
        home_ids = [args.home]
    else:
        home_ids = get_all_home_ids()

    if not home_ids:
        print("ERROR: No homes found in Firestore.")
        sys.exit(1)

    print(f"\n  Processing {len(home_ids)} home(s)...\n")
    success, failed = 0, 0

    for home_id in home_ids:
        print(f"  [{home_id[:12]}...]")

        # 1. Read energy logs
        records = read_energy_logs(home_id, year, month)
        print(f"    Records found: {len(records)}")

        if not records:
            print(f"    ! No data — skipping")
            failed += 1
            continue

        # 2. Generate ML report
        report = generate_monthly_report(home_id, year, month, records)
        if not report:
            print(f"    ! Insufficient data — skipping")
            failed += 1
            continue

        # 3. Write to Firestore
        ok = write_ml_report(home_id, year, month, report)
        if ok:
            print(f"    [OK] Report written -> homes/{home_id}/ml_reports/{year}-{month:02d}")
            print(f"       Total: {report['total_kwh']} kWh | "
                  f"Avg: {report['avg_daily_kwh']} kWh/day | "
                  f"Predicted next month: {report['predicted_next_month_kwh']} kWh")
            success += 1
        else:
            print(f"    X Failed to write report")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Done: {success} reports generated, {failed} skipped/failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
