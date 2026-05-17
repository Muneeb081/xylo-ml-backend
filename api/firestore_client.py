"""
Firestore Client
================
Handles all read/write operations against Cloud Firestore (not Realtime DB).

Schema managed by this module:
  homes/{homeId}/energy_logs/{YYYY-MM-DD}   ← daily usage (pipeline reads)
  homes/{homeId}/ml_reports/{YYYY-MM}       ← monthly ML results (pipeline writes)

Usage:
    from api.firestore_client import init_firestore, read_energy_logs, write_ml_report
"""
from __future__ import annotations

import json
import os
import logging
from datetime import datetime, date
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Lazy Firestore reference ──────────────────────────────────────────────────

def _fs():
    """Return the Firestore client instance."""
    from firebase_admin import firestore
    return firestore.client()


def is_firestore_ready() -> bool:
    """Return True if Firebase Admin has been initialized."""
    try:
        import firebase_admin
        return firebase_admin._DEFAULT_APP_NAME in firebase_admin._apps
    except Exception:
        return False


# ── Initialization ────────────────────────────────────────────────────────────

def init_firestore() -> bool:
    """
    Initialize Firebase Admin SDK from environment variables.
    Works for both Firestore and Realtime DB (same SDK, same credentials).

    Required env vars:
        FIREBASE_SERVICE_ACCOUNT_JSON : full JSON string of service account key
        FIREBASE_DATABASE_URL         : Realtime DB URL (still needed for SDK init)

    Returns True if successful, False if env vars are missing.
    """
    import firebase_admin
    from firebase_admin import credentials

    if firebase_admin._DEFAULT_APP_NAME in firebase_admin._apps:
        logger.info("[Firestore] Already initialized.")
        return True

    sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    db_url  = os.environ.get("FIREBASE_DATABASE_URL", "").strip()

    if not sa_json:
        logger.warning(
            "[Firestore] FIREBASE_SERVICE_ACCOUNT_JSON not set. "
            "Running in local-only mode."
        )
        return False

    try:
        sa_dict = json.loads(sa_json)
        cred    = credentials.Certificate(sa_dict)
        opts    = {"databaseURL": db_url} if db_url else {}
        firebase_admin.initialize_app(cred, opts)
        logger.info("[Firestore] Firebase Admin initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"[Firestore] Initialization failed: {e}")
        return False


# ── READ — Energy Logs ────────────────────────────────────────────────────────

def read_energy_logs(home_id: str, year: int, month: int) -> list[dict]:
    """
    Read all daily energy_log documents for a given home and month.

    Firestore path: homes/{homeId}/energy_logs/{YYYY-MM-DD}

    Returns a list of dicts, each representing one day:
    [
      {
        "date":        "2026-05-01",
        "total_kwh":   12.4,
        "devices_on":  5,
        "rooms": {
          "living_room": {"kwh": 3.2, "devices_on": 2},
          "kitchen":     {"kwh": 2.1, "devices_on": 1},
          ...
        }
      },
      ...
    ]
    """
    if not is_firestore_ready():
        logger.warning("[Firestore] Not initialized — returning empty logs.")
        return []

    try:
        from google.cloud.firestore_v1 import FieldFilter
        prefix = f"{year:04d}-{month:02d}"
        col_ref = _fs().collection("homes").document(home_id).collection("energy_logs")
        docs    = col_ref \
                    .where(filter=FieldFilter("date", ">=", f"{prefix}-01")) \
                    .where(filter=FieldFilter("date", "<=", f"{prefix}-31")) \
                    .order_by("date") \
                    .stream()

        records = []
        for doc in docs:
            d = doc.to_dict()
            d["doc_id"] = doc.id
            records.append(d)

        logger.info(f"[Firestore] Read {len(records)} energy_log docs for home={home_id} {prefix}")
        return records

    except Exception as e:
        logger.error(f"[Firestore] Failed to read energy_logs for {home_id}: {e}")
        return []


def get_all_home_ids() -> list[str]:
    """Return all document IDs in the homes/ collection."""
    if not is_firestore_ready():
        return []
    try:
        docs = _fs().collection("homes").stream()
        ids  = [doc.id for doc in docs]
        logger.info(f"[Firestore] Found {len(ids)} homes.")
        return ids
    except Exception as e:
        logger.error(f"[Firestore] Failed to list homes: {e}")
        return []


# ── WRITE — Monthly ML Report ─────────────────────────────────────────────────

def write_ml_report(home_id: str, year: int, month: int, report: dict) -> bool:
    """
    Write an ML monthly report to Firestore.

    Firestore path: homes/{homeId}/ml_reports/{YYYY-MM}

    Parameters
    ----------
    home_id : str   — Firestore home document ID
    year    : int
    month   : int
    report  : dict  — full report payload from monthly_report.py

    Returns True on success.
    """
    if not is_firestore_ready():
        logger.warning("[Firestore] Not ready — skipping write.")
        return False

    try:
        from firebase_admin import firestore as _fstore
        doc_id  = f"{year:04d}-{month:02d}"
        ref     = _fs().collection("homes").document(home_id) \
                        .collection("ml_reports").document(doc_id)

        report["generated_at"] = _fstore.SERVER_TIMESTAMP
        report["period"]       = doc_id
        report["period_label"] = datetime(year, month, 1).strftime("%B %Y")
        report["status"]       = "complete"

        ref.set(report)
        logger.info(f"[Firestore] ml_report written: homes/{home_id}/ml_reports/{doc_id}")
        return True

    except Exception as e:
        logger.error(f"[Firestore] Failed to write ml_report for {home_id}: {e}")
        return False


def read_ml_report(home_id: str, year: int, month: int) -> Optional[dict]:
    """Read a previously generated ML report from Firestore."""
    if not is_firestore_ready():
        return None
    try:
        doc_id = f"{year:04d}-{month:02d}"
        ref    = _fs().collection("homes").document(home_id) \
                       .collection("ml_reports").document(doc_id)
        snap   = ref.get()
        return snap.to_dict() if snap.exists else None
    except Exception as e:
        logger.error(f"[Firestore] Failed to read ml_report: {e}")
        return None
