"""
FastAPI Application — PRECON Smart Energy Inference API
========================================================
Endpoints:
  GET  /health                                    — server, model & Firebase status
  POST /api/v1/predict/all                        — all tasks in one call
  POST /api/v1/predict/energy                     — Task 1a: XGBoost energy
  POST /api/v1/predict/energy/lstm                — Task 1b: LSTM multi-step forecast
  POST /api/v1/predict/peak                       — Task 2: peak load forecast
  POST /api/v1/disaggregate                       — Task 3: per-appliance disaggregation
  POST /api/v1/anomaly                            — Task 4: room-wise anomaly + alerts
  POST /api/v1/recommendations                    — Task 5: optimization tips
  POST /api/v1/stream/firebase                    — push payload → infer → write to Firestore
  GET  /api/v1/report/{home_id}/{year}/{month}    — generate/fetch monthly ML report
  POST /api/v1/report/generate                    — trigger report generation for all homes
"""
from __future__ import annotations

import sys
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.models_loader    import load_models, get_status, MODELS
from api.preprocessing    import build_feature_row, get_appliance_readings
from api.inference        import (
    predict_energy, predict_energy_lstm, predict_peak,
    disaggregate_appliances, detect_anomalies, get_recommendations,
)
from api.firebase_client  import (
    init_firebase, is_firebase_ready,
    firebase_read_live, firebase_write_results, firebase_write_live_reading,
)
from api.firestore_client import (
    init_firestore, is_firestore_ready,
    read_energy_logs, write_ml_report, read_ml_report, get_all_home_ids,
)

# Feature columns loaded at startup from _pipeline_metrics.json
_FEATURE_COLS: list[str] = []


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _FEATURE_COLS
    print("\n" + "=" * 60)
    print("  PRECON ENERGY INFERENCE API — starting ...")
    print("=" * 60)

    # 1. Load ML models
    load_models()

    # 2. Load feature columns from pipeline metrics
    import json
    metrics_path = ROOT / "_pipeline_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        _FEATURE_COLS = m.get("feature_cols", [])
        print(f"  [OK] Feature cols loaded: {len(_FEATURE_COLS)} features")
    else:
        print("  [WARN] _pipeline_metrics.json not found. Run train_pipeline.py first.")

    # 3. Initialize Firebase Admin SDK (covers Firestore + Realtime DB)
    fb_ok = init_firebase()
    if fb_ok:
        print("  [OK] Firebase Admin SDK connected (Firestore enabled)")
    else:
        print("  [INFO] Firebase not configured — running in local-only mode")
        print("         Set FIREBASE_SERVICE_ACCOUNT_JSON to enable Firestore")

    print("[OK] API ready.\n")
    yield


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PRECON Smart Energy ML API",
    description=(
        "Real-time multi-task energy inference API. "
        "Supports energy prediction (XGBoost + LSTM), appliance disaggregation, "
        "room-wise anomaly detection with human-readable alerts, peak load forecasting, "
        "and optimization recommendations. "
        "Compatible with Firebase JSON streaming."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Request Schema ─────────────────────────────────────────────────────────────

class EnergyInput(BaseModel):
    """
    Flexible input schema — accepts any appliance reading.
    Canonical appliance fields are mapped automatically.
    Any extra appliance key is accepted via `extra_appliances`.
    """
    house_id:   int         = Field(..., description="House identifier (1-42)")
    timestamp:  Optional[str] = Field(None, description="ISO-8601 timestamp")

    # Total usage (if metered directly)
    Usage_kW:   float = Field(0.0, description="Total household power draw (kW)")

    # Known canonical appliance readings
    ac:           float = 0.0
    ups:          float = 0.0
    living_room:  float = 0.0
    kitchen:      float = 0.0
    bedroom:      float = 0.0
    drawing_room: float = 0.0
    water_pump:   float = 0.0
    refrigerator: float = 0.0
    laundry:      float = 0.0
    lights:       float = 0.0
    fans:         float = 0.0

    # House metadata (improves accuracy)
    n_rooms:    float = 0.0
    n_acs:      float = 0.0
    n_people:   float = 0.0
    area_sqft:  float = 0.0

    # Any extra appliances or raw Firebase keys
    extra_appliances: Dict[str, float] = Field(
        default={},
        description="Send any extra device reading: {'AC_DR_kW': 1.2, 'Fan_kW': 0.3}"
    )


class FirebasePayload(BaseModel):
    """Raw Firebase Realtime Database payload — any key structure accepted."""
    house_id:  int
    timestamp: Optional[str] = None
    data:      Dict[str, Any] = Field(..., description="Firebase node data dict")


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status":        "healthy",
        "server_time":   datetime.now().isoformat(),
        "models":        get_status(),
        "firebase":      {
            "connected":     is_firebase_ready(),
            "mode":          "live" if is_firebase_ready() else "local-only",
        },
        "version":       "1.0.0",
    }


# ── Helper ─────────────────────────────────────────────────────────────────────

def _build_payload(data: EnergyInput) -> dict:
    d = data.model_dump()
    # Merge extra_appliances into top-level dict for processing
    d.update(d.pop("extra_appliances", {}))
    return d


def _require_feature_cols():
    if not _FEATURE_COLS:
        raise HTTPException(
            status_code=503,
            detail="Feature columns not loaded. Run train_pipeline.py first.",
        )
    return _FEATURE_COLS


# ── Task 1a — XGBoost Energy ──────────────────────────────────────────────────

@app.post("/api/v1/predict/energy", tags=["Inference"])
def predict_energy_endpoint(data: EnergyInput):
    """Predict current-step household energy consumption (kW) using XGBoost."""
    fc   = _require_feature_cols()
    raw  = _build_payload(data)
    X    = build_feature_row(raw, fc)
    result = predict_energy(X)
    return {"house_id": data.house_id, "timestamp": data.timestamp, "energy_prediction": result}


# ── Task 1b — LSTM ─────────────────────────────────────────────────────────────

@app.post("/api/v1/predict/energy/lstm", tags=["Inference"])
def predict_lstm_endpoint(data: EnergyInput):
    """
    Predict next N steps of energy consumption using LSTM.
    Requires LSTM model to have been trained (without --no-lstm flag).
    """
    if "lstm_energy" not in MODELS:
        raise HTTPException(status_code=503, detail="LSTM model not loaded.")
    fc   = _require_feature_cols()
    raw  = _build_payload(data)
    X    = build_feature_row(raw, fc)

    scaler_X = MODELS.get("scaler_X_lstm")
    if scaler_X is None:
        raise HTTPException(status_code=503, detail="LSTM scaler not loaded")

    from src.config import SEQ_LEN
    import numpy as np
    # Replicate the single row to fill the sequence (proxy for missing history)
    X_scaled = scaler_X.transform(X)
    X_seq    = np.tile(X_scaled, (SEQ_LEN, 1))[np.newaxis, :, :]  # (1, SEQ_LEN, n_feat)
    result   = predict_energy_lstm(X_seq)
    return {"house_id": data.house_id, "lstm_forecast": result}


# ── Task 2 — Peak Forecast ────────────────────────────────────────────────────

@app.post("/api/v1/predict/peak", tags=["Inference"])
def predict_peak_endpoint(data: EnergyInput):
    """Forecast today's daily peak kW using XGBoost."""
    fc  = _require_feature_cols()
    fc_peak = ["house_id"] + fc
    raw = _build_payload(data)
    X_peak = build_feature_row(raw, fc_peak)
    result = predict_peak(X_peak)
    return {"house_id": data.house_id, "peak_forecast": result}


# ── Task 3 — Disaggregation ───────────────────────────────────────────────────

@app.post("/api/v1/disaggregate", tags=["Inference"])
def disaggregate_endpoint(data: EnergyInput):
    """
    Per-appliance energy disaggregation.
    Returns absolute kW and percentage contribution for each appliance group.
    """
    fc    = _require_feature_cols()
    raw   = _build_payload(data)
    X     = build_feature_row(raw, fc)
    energy = predict_energy(X)
    total  = energy.get("predicted_kw", data.Usage_kW or 1.0)
    result = disaggregate_appliances(X, total)
    return {"house_id": data.house_id, "disaggregation": result}


# ── Task 4 — Anomaly Detection ────────────────────────────────────────────────

@app.post("/api/v1/anomaly", tags=["Inference"])
def anomaly_endpoint(data: EnergyInput):
    """
    Room-wise anomaly detection with human-readable alert messages.

    Example alerts:
      '⚠ WARNING — Living Room is consuming unusually high energy compared to
       its normal behavior (Z-score=3.1, current=2.4 kW, normal avg=0.5 kW).'
    """
    raw      = _build_payload(data)
    readings = get_appliance_readings(raw)
    result   = detect_anomalies(
        house_id=data.house_id,
        appliance_readings=readings,
        timestamp=data.timestamp or datetime.now().isoformat(),
    )
    return {"house_id": data.house_id, "anomaly_detection": result}


# ── Task 5 — Recommendations ──────────────────────────────────────────────────

@app.post("/api/v1/recommendations", tags=["Inference"])
def recommendations_endpoint(data: EnergyInput):
    """Generate actionable energy-saving recommendations."""
    fc    = _require_feature_cols()
    raw   = _build_payload(data)
    X     = build_feature_row(raw, fc)
    energy = predict_energy(X)
    readings = get_appliance_readings(raw)
    recs = get_recommendations(
        appliance_readings=readings,
        predicted_total_kw=energy.get("predicted_kw"),
        house_meta={"n_acs": data.n_acs, "n_people": data.n_people},
    )
    return {"house_id": data.house_id, "recommendations": recs}


# ── All tasks combined ────────────────────────────────────────────────────────

@app.post("/api/v1/predict/all", tags=["Inference"])
def predict_all(data: EnergyInput):
    """
    Run ALL 5 tasks in a single call. Most efficient endpoint for dashboards.
    Returns energy prediction, peak forecast, disaggregation,
    anomaly alerts, and optimization recommendations.
    """
    fc       = _require_feature_cols()
    raw      = _build_payload(data)
    X        = build_feature_row(raw, fc)
    readings = get_appliance_readings(raw)
    ts       = data.timestamp or datetime.now().isoformat()

    energy   = predict_energy(X)
    
    fc_peak  = ["house_id"] + fc
    X_peak   = build_feature_row(raw, fc_peak)
    peak     = predict_peak(X_peak)
    
    total_kw = energy.get("predicted_kw", data.Usage_kW or 1.0)
    disagg   = disaggregate_appliances(X, total_kw)
    anomaly  = detect_anomalies(data.house_id, readings, ts)
    recs     = get_recommendations(
        readings, total_kw, {"n_acs": data.n_acs, "n_people": data.n_people}
    )

    return {
        "house_id":          data.house_id,
        "timestamp":         ts,
        "energy_prediction": energy,
        "peak_forecast":     peak,
        "disaggregation":    disagg,
        "anomaly_detection": anomaly,
        "recommendations":   recs,
    }


# ── Firebase: Push payload → infer → write back ───────────────────────────────

@app.post("/api/v1/stream/firebase", tags=["Firebase"])
def firebase_stream(payload: FirebasePayload):
    """
    Accept a Firebase JSON payload, run all 5 inference tasks,
    and automatically write results back to Firebase under
    /houses/{house_id}/predictions|alerts|recommendations|disaggregation.

    Payload format:
      {"house_id": 3, "timestamp": "2024-01-15T14:30:00",
       "data": {"Usage_kW": 2.5, "AC_DR_kW": 1.2, "Kitchen_kW": 0.4, ...}}
    """
    fc  = _require_feature_cols()
    ts  = payload.timestamp or datetime.now().isoformat()

    # Flatten Firebase data into a single dict
    flat = {"house_id": payload.house_id, "timestamp": ts}
    flat.update(payload.data)

    X        = build_feature_row(flat, fc)
    readings = get_appliance_readings(flat)

    energy   = predict_energy(X)
    fc_peak  = ["house_id"] + fc
    X_peak   = build_feature_row(flat, fc_peak)
    peak     = predict_peak(X_peak)
    total_kw = energy.get("predicted_kw", 0.0)
    disagg   = disaggregate_appliances(X, total_kw)
    anomaly  = detect_anomalies(payload.house_id, readings, ts)
    recs     = get_recommendations(readings, total_kw)

    result = {
        "source":            "firebase",
        "house_id":          payload.house_id,
        "timestamp":         ts,
        "energy_prediction": energy,
        "peak_forecast":     peak,
        "disaggregation":    disagg,
        "anomaly_detection": anomaly,
        "recommendations":   recs,
    }

    # Write results back to Firebase (no-op if Firebase not configured)
    firebase_write_results(payload.house_id, result)

    return result


# ── Firebase: Poll house → infer → write back (server-side pull) ──────────────

@app.get("/api/v1/firebase/poll/{house_id}", tags=["Firebase"])
def firebase_poll(house_id: int):
    """
    Server-side Firebase pull:
      1. Reads /houses/{house_id}/live_reading from Firebase
      2. Runs all 5 inference tasks on that data
      3. Writes results back to Firebase under /houses/{house_id}/
      4. Returns the full result payload

    Use this endpoint from a Render Cron job or a scheduler
    to continuously process incoming sensor data.
    """
    if not is_firebase_ready():
        raise HTTPException(
            status_code=503,
            detail="Firebase is not configured. Set FIREBASE_SERVICE_ACCOUNT_JSON and FIREBASE_DATABASE_URL.",
        )

    fc  = _require_feature_cols()
    ts  = datetime.now().isoformat()

    # Step 1: Read live reading from Firebase
    live_data = firebase_read_live(house_id)
    if not live_data:
        raise HTTPException(
            status_code=404,
            detail=f"No live_reading found for house {house_id} in Firebase.",
        )

    live_data.setdefault("timestamp", ts)

    # Step 2: Run inference
    X        = build_feature_row(live_data, fc)
    readings = get_appliance_readings(live_data)

    energy   = predict_energy(X)
    fc_peak  = ["house_id"] + fc
    X_peak   = build_feature_row(live_data, fc_peak)
    peak     = predict_peak(X_peak)
    total_kw = energy.get("predicted_kw", 0.0)
    disagg   = disaggregate_appliances(X, total_kw)
    anomaly  = detect_anomalies(house_id, readings, ts)
    recs     = get_recommendations(
        readings, total_kw,
        {"n_acs": live_data.get("n_acs", 0), "n_people": live_data.get("n_people", 0)}
    )

    result = {
        "source":            "firebase_poll",
        "house_id":          house_id,
        "timestamp":         ts,
        "energy_prediction": energy,
        "peak_forecast":     peak,
        "disaggregation":    disagg,
        "anomaly_detection": anomaly,
        "recommendations":   recs,
    }

    # Step 3: Write results back to Firebase
    firebase_write_results(house_id, result)

    return result


# ── Firebase: Write a test live_reading (dev / testing only) ──────────────────

class FirebaseTestReading(BaseModel):
    """Test payload to push a simulated reading directly into Firebase."""
    house_id:  int
    Usage_kW:  float = 4.2
    ac:        float = 2.1
    kitchen:   float = 1.2
    n_acs:     int   = 2
    n_people:  int   = 4
    timestamp: Optional[str] = None


@app.post("/api/v1/firebase/write_test", tags=["Firebase"])
def firebase_write_test(data: FirebaseTestReading):
    """
    Write a simulated sensor reading to Firebase at
    /houses/{house_id}/live_reading.

    Use this to test the full round-trip without a physical device:
      POST /api/v1/firebase/write_test  →  GET /api/v1/firebase/poll/{house_id}
    """
    if not is_firebase_ready():
        raise HTTPException(
            status_code=503,
            detail="Firebase is not configured. Set FIREBASE_SERVICE_ACCOUNT_JSON and FIREBASE_DATABASE_URL.",
        )

    reading = data.model_dump()
    reading["timestamp"] = reading.get("timestamp") or datetime.now().isoformat()
    ok = firebase_write_live_reading(data.house_id, reading)

    if not ok:
        raise HTTPException(status_code=500, detail="Failed to write to Firebase.")

    return {
        "status":   "written",
        "house_id": data.house_id,
        "path":     f"houses/{data.house_id}/live_reading",
        "data":     reading,
    }


# ── Firestore Monthly Report Endpoints ────────────────────────────────────────

@app.get("/api/v1/report/{home_id}/{year}/{month}", tags=["Monthly Reports"])
def get_monthly_report(home_id: str, year: int, month: int):
    """
    Fetch or generate the monthly ML energy report for a home.

    Steps:
      1. Check if ml_reports/{YYYY-MM} already exists in Firestore → return it
      2. If not, read energy_logs for that month, run ML pipeline, write + return report

    Called by the Flutter app to display the AI Insights screen.
    Path: homes/{home_id}/ml_reports/{YYYY-MM}
    """
    if not is_firebase_ready():
        raise HTTPException(
            status_code=503,
            detail="Firebase not configured. Set FIREBASE_SERVICE_ACCOUNT_JSON.",
        )

    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="month must be 1–12")

    # Check if report already exists
    existing = read_ml_report(home_id, year, month)
    if existing:
        return {"source": "cached", "home_id": home_id, "report": existing}

    # Generate fresh report
    records = read_energy_logs(home_id, year, month)
    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"No energy_logs found for home={home_id} {year}-{month:02d}. "
                   f"Seed data first or check the home ID.",
        )

    from models_src.monthly_report import generate_monthly_report
    report = generate_monthly_report(home_id, year, month, records)

    if not report:
        raise HTTPException(
            status_code=422,
            detail="Insufficient data to generate report (need at least 7 days).",
        )

    write_ml_report(home_id, year, month, report)
    return {"source": "generated", "home_id": home_id, "report": report}


class ReportGenerateRequest(BaseModel):
    """Request body for triggering bulk report generation."""
    year:     int
    month:    int
    home_ids: list[str] = []   # empty = all homes


@app.post("/api/v1/report/generate", tags=["Monthly Reports"])
def trigger_report_generation(req: ReportGenerateRequest):
    """
    Trigger monthly ML report generation for all homes (or specific ones).
    Called by the Render Cron Job or manually from the admin dashboard.

    Returns a summary of which reports were generated vs skipped.
    """
    if not is_firebase_ready():
        raise HTTPException(
            status_code=503,
            detail="Firebase not configured. Set FIREBASE_SERVICE_ACCOUNT_JSON.",
        )

    home_ids = req.home_ids or get_all_home_ids()
    if not home_ids:
        raise HTTPException(status_code=404, detail="No homes found in Firestore.")

    from models_src.monthly_report import generate_monthly_report

    results = {"generated": [], "skipped": [], "failed": []}

    for home_id in home_ids:
        try:
            records = read_energy_logs(home_id, req.year, req.month)
            if not records:
                results["skipped"].append({"home_id": home_id, "reason": "no energy_logs"})
                continue

            report = generate_monthly_report(home_id, req.year, req.month, records)
            if not report:
                results["skipped"].append({"home_id": home_id, "reason": "insufficient data"})
                continue

            write_ml_report(home_id, req.year, req.month, report)
            results["generated"].append({
                "home_id":    home_id,
                "total_kwh":  report["total_kwh"],
                "period":     report["period"],
            })

        except Exception as e:
            results["failed"].append({"home_id": home_id, "error": str(e)})

    return {
        "period":    f"{req.year:04d}-{req.month:02d}",
        "processed": len(home_ids),
        "results":   results,
    }
