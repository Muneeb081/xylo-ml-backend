"""
Seed Test Data — Firestore energy_logs
=======================================
Creates 3 months of realistic daily energy log documents in Firestore
for two test homes, ready for the ML pipeline to process.

Schema written:
  homes/{homeId}/energy_logs/{YYYY-MM-DD}
    date:        str   "2026-05-01"
    home_id:     str
    total_kwh:   float
    devices_on:  int
    rooms: {
      living_room:  { kwh: float, devices_on: int }
      kitchen:      { kwh: float, devices_on: int }
      bedroom:      { kwh: float, devices_on: int }
      drawing_room: { kwh: float, devices_on: int }
    }

Usage:
    python scripts/seed_test_data.py
"""
import os
import sys
import json
import random
from datetime import date, timedelta
from pathlib import Path

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Firebase init ─────────────────────────────────────────────────────────────
def init():
    sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    db_url  = os.environ.get("FIREBASE_DATABASE_URL", "").strip()

    if not sa_json:
        # Try loading from local key file for dev convenience
        key_path = Path("C:/Users/munee/Downloads/0a3b07ee-5667-404f-8b76-7181bddf1eed")
        if key_path.exists():
            sa_json = key_path.read_text()
            db_url  = "https://xylo-switch-default-rtdb.asia-southeast1.firebasedatabase.app"
        else:
            print("ERROR: Set FIREBASE_SERVICE_ACCOUNT_JSON env var.")
            sys.exit(1)

    import firebase_admin
    from firebase_admin import credentials, firestore

    if firebase_admin._DEFAULT_APP_NAME not in firebase_admin._apps:
        cred = credentials.Certificate(json.loads(sa_json))
        firebase_admin.initialize_app(cred, {"databaseURL": db_url})

    return firestore.client()


# ── Test homes — use your real Firestore home IDs ─────────────────────────────
TEST_HOMES = [
    {"id": "CJBxCqpgKOjlw9LZ1ifC", "name": "hafsa",  "profile": "moderate"},
    {"id": "ULEBIZldjvggtFG0wXGZ",  "name": "home2",  "profile": "heavy_ac"},
]

ROOMS = ["living_room", "kitchen", "bedroom", "drawing_room"]

# ── Usage profiles ────────────────────────────────────────────────────────────
PROFILES = {
    "moderate": {
        "living_room":  (1.5, 3.5),
        "kitchen":      (1.0, 2.5),
        "bedroom":      (0.5, 1.8),
        "drawing_room": (0.2, 1.0),
    },
    "heavy_ac": {
        "living_room":  (3.0, 6.0),
        "kitchen":      (1.2, 2.8),
        "bedroom":      (2.0, 4.5),
        "drawing_room": (0.3, 1.2),
    },
}

# ── Date range: March, April, May 2026 ───────────────────────────────────────
def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def generate_day(day: date, profile: str) -> dict:
    """Generate one day's energy log with realistic variation."""
    ranges = PROFILES[profile]

    # Weekend boost (more home usage)
    is_weekend = day.weekday() >= 5
    boost      = 1.25 if is_weekend else 1.0

    # Random anomaly day (5% chance — spike usage)
    is_anomaly = random.random() < 0.05
    anomaly_mult = random.uniform(1.8, 2.5) if is_anomaly else 1.0

    rooms = {}
    total = 0.0

    for room, (lo, hi) in ranges.items():
        kwh        = round(random.uniform(lo, hi) * boost * anomaly_mult, 3)
        devices_on = random.randint(1, 4)
        rooms[room] = {"kwh": kwh, "devices_on": devices_on}
        total      += kwh

    return {
        "date":       day.isoformat(),
        "total_kwh":  round(total, 3),
        "devices_on": sum(r["devices_on"] for r in rooms.values()),
        "rooms":      rooms,
        "is_anomaly": is_anomaly,
    }


def seed(fs, home_id: str, home_name: str, profile: str):
    """Write energy_logs for Mar–May 2026 for one home."""
    start = date(2026, 3, 1)
    end   = date(2026, 5, 17)   # up to today

    col_ref = fs.collection("homes").document(home_id).collection("energy_logs")
    batch   = fs.batch()
    count   = 0

    print(f"\n  Seeding {home_name} ({home_id}) [{profile}]...")

    for day in date_range(start, end):
        record         = generate_day(day, profile)
        record["home_id"] = home_id
        doc_ref        = col_ref.document(day.isoformat())
        batch.set(doc_ref, record)
        count         += 1

        # Firestore batch limit = 500
        if count % 400 == 0:
            batch.commit()
            batch = fs.batch()
            print(f"    Committed {count} docs so far...")

    batch.commit()
    print(f"  [OK] {count} energy_log docs written for {home_name}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)   # reproducible
    print("=" * 55)
    print("  Seeding Firestore test data (energy_logs)")
    print("=" * 55)

    fs = init()

    for home in TEST_HOMES:
        seed(fs, home["id"], home["name"], home["profile"])

    print("\n[OK] All test data seeded successfully.")
    print("     Path: homes/{homeId}/energy_logs/{YYYY-MM-DD}")
    print("     Range: 2026-03-01 -> 2026-05-17")
    print("     Homes:", [h["id"] for h in TEST_HOMES])
