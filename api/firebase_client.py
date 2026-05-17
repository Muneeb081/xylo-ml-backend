"""
Firebase Client
===============
Handles all read/write operations against Firebase Realtime Database.

Database Node Structure
-----------------------
/houses/{house_id}/
    live_reading/           <- device writes sensor data here
        timestamp: str
        Usage_kW: float
        ac: float
        kitchen: float
        ...

    predictions/            <- pipeline writes inference results here
        timestamp: str
        predicted_kw: float
        predicted_peak_kw: float
        model: str

    alerts/                 <- anomaly detection results
        timestamp: str
        anomalies: list
        alert_count: int

    recommendations/        <- optimization tips
        timestamp: str
        tips: list[str]

    disaggregation/         <- per-appliance breakdown
        timestamp: str
        total_kw: float
        appliances: list

Usage
-----
    from api.firebase_client import firebase_read_live, firebase_write_results
"""
from __future__ import annotations

import json
import os
import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Lazy import guard — firebase_admin may not be initialized yet ─────────────

def _db():
    """Return a reference to the Firebase Realtime Database."""
    try:
        from firebase_admin import db
        return db
    except ImportError:
        raise RuntimeError("firebase-admin is not installed. Run: pip install firebase-admin")


def is_firebase_ready() -> bool:
    """Return True if Firebase Admin has been successfully initialized."""
    try:
        import firebase_admin
        return firebase_admin._DEFAULT_APP_NAME in firebase_admin._apps
    except Exception:
        return False


# ── Initialization ────────────────────────────────────────────────────────────

def init_firebase() -> bool:
    """
    Initialize Firebase Admin SDK from environment variables.

    Required environment variables:
        FIREBASE_SERVICE_ACCOUNT_JSON : full JSON string of the service account key
        FIREBASE_DATABASE_URL         : e.g. https://your-project-default-rtdb.firebaseio.com

    Returns True if successful, False if env vars are missing (non-fatal for local dev).
    """
    import firebase_admin
    from firebase_admin import credentials

    # Already initialized
    if firebase_admin._DEFAULT_APP_NAME in firebase_admin._apps:
        logger.info("[Firebase] Already initialized.")
        return True

    sa_json  = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    db_url   = os.environ.get("FIREBASE_DATABASE_URL", "").strip()

    if not sa_json or not db_url:
        logger.warning(
            "[Firebase] FIREBASE_SERVICE_ACCOUNT_JSON or FIREBASE_DATABASE_URL not set. "
            "Firebase integration disabled — running in local-only mode."
        )
        return False

    try:
        sa_dict = json.loads(sa_json)
        cred    = credentials.Certificate(sa_dict)
        firebase_admin.initialize_app(cred, {"databaseURL": db_url})
        logger.info(f"[Firebase] Initialized successfully → {db_url}")
        return True
    except Exception as e:
        logger.error(f"[Firebase] Initialization failed: {e}")
        return False


# ── READ — Pull live sensor reading for a house ───────────────────────────────

def firebase_read_live(house_id: int) -> Optional[dict]:
    """
    Read the current live_reading node for a given house.

    Returns the data dict, or None if Firebase is not initialized or node is empty.

    Firebase path: /houses/{house_id}/live_reading
    """
    if not is_firebase_ready():
        return None
    try:
        ref  = _db().reference(f"houses/{house_id}/live_reading")
        data = ref.get()
        if data:
            data["house_id"] = house_id
            logger.debug(f"[Firebase] Read live_reading for house {house_id}: {data}")
        return data
    except Exception as e:
        logger.error(f"[Firebase] Failed to read house {house_id} live_reading: {e}")
        return None


# ── WRITE — Store inference results back to Firebase ──────────────────────────

def firebase_write_results(house_id: int, results: dict) -> bool:
    """
    Write inference results back to the Firebase Realtime Database.

    Writes to four separate child nodes under /houses/{house_id}/:
        predictions/      energy + peak prediction
        alerts/           anomaly detection output
        recommendations/  optimization tips
        disaggregation/   per-appliance breakdown

    Parameters
    ----------
    house_id : int
    results  : dict — the full response payload from /api/v1/predict/all

    Returns True if write succeeded, False otherwise.
    """
    if not is_firebase_ready():
        return False

    ts = results.get("timestamp", datetime.now().isoformat())

    try:
        base_ref = _db().reference(f"houses/{house_id}")

        # ── Predictions node ─────────────────────────────────────────────────
        energy = results.get("energy_prediction", {})
        peak   = results.get("peak_forecast", {})
        base_ref.child("predictions").set({
            "timestamp":          ts,
            "predicted_kw":       energy.get("predicted_kw"),
            "model_energy":       energy.get("model"),
            "predicted_peak_kw":  peak.get("predicted_peak_kw"),
            "model_peak":         peak.get("model"),
        })

        # ── Alerts node ───────────────────────────────────────────────────────
        anomaly = results.get("anomaly_detection", {})
        base_ref.child("alerts").set({
            "timestamp":    ts,
            "anomalies":    anomaly.get("anomalies", []),
            "alert_count":  anomaly.get("alert_count", 0),
            "summary":      anomaly.get("summary", ""),
        })

        # ── Recommendations node ──────────────────────────────────────────────
        base_ref.child("recommendations").set({
            "timestamp": ts,
            "tips":      results.get("recommendations", []),
        })

        # ── Disaggregation node ───────────────────────────────────────────────
        disagg = results.get("disaggregation", {})
        base_ref.child("disaggregation").set({
            "timestamp":  ts,
            "total_kw":   disagg.get("total_kw"),
            "appliances": disagg.get("appliances", []),
        })

        logger.info(f"[Firebase] Results written for house {house_id} at {ts}")
        return True

    except Exception as e:
        logger.error(f"[Firebase] Failed to write results for house {house_id}: {e}")
        return False


# ── WRITE — Store a single live reading (useful for testing) ──────────────────

def firebase_write_live_reading(house_id: int, reading: dict) -> bool:
    """
    Write a sensor reading to Firebase. Used for testing the full loop.
    Firebase path: /houses/{house_id}/live_reading
    """
    if not is_firebase_ready():
        return False
    try:
        reading["timestamp"] = reading.get("timestamp", datetime.now().isoformat())
        _db().reference(f"houses/{house_id}/live_reading").set(reading)
        logger.info(f"[Firebase] Live reading written for house {house_id}")
        return True
    except Exception as e:
        logger.error(f"[Firebase] Failed to write live reading: {e}")
        return False
