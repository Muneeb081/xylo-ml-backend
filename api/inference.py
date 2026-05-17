"""
Inference Engine
================
All 5 task inference functions. Loaded models come from api.models_loader.MODELS.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from api.models_loader import MODELS
from models_src.task3_anomaly    import generate_room_alerts
from models_src.task5_recommendations import generate_recommendations
from src.config import USAGE_COL, CANONICAL_APPLIANCE_COLS


# ── Task 1a — XGBoost Energy Prediction ───────────────────────────────────────

def predict_energy(X: np.ndarray) -> dict:
    """Predict current-step Usage_kW using XGBoost."""
    model    = MODELS.get("xgb_energy")
    scaler_X = MODELS.get("scaler_X_energy")
    scaler_y = MODELS.get("scaler_y_energy")

    if None in (model, scaler_X, scaler_y):
        return {"error": "XGBoost energy model not loaded"}

    X_sc   = scaler_X.transform(X)
    pred_s = model.predict(X_sc)
    pred   = float(scaler_y.inverse_transform(pred_s.reshape(-1, 1))[0][0])
    return {"predicted_kw": round(max(pred, 0.0), 4), "model": "XGBoost"}


# ── Task 1b — LSTM Energy Prediction ──────────────────────────────────────────

def predict_energy_lstm(X_seq: np.ndarray) -> dict:
    """
    Predict next FORECAST_STEPS energy steps using LSTM.
    X_seq : shape (1, SEQ_LEN, n_features) — already scaled
    """
    model    = MODELS.get("lstm_energy")
    scaler_y = MODELS.get("scaler_y_lstm")

    if model is None:
        return {"error": "LSTM model not loaded — run pipeline without --no-lstm first"}
    if scaler_y is None:
        return {"error": "LSTM scaler not loaded"}

    pred_s   = model.predict(X_seq, verbose=0)
    pred_raw = scaler_y.inverse_transform(pred_s[:, :1]).ravel()
    return {
        "predicted_kw_next_steps": [round(max(float(v), 0), 4) for v in pred_raw],
        "model": "LSTM",
    }


# ── Task 2 — Peak Load Forecast ───────────────────────────────────────────────

def predict_peak(X: np.ndarray) -> dict:
    """Predict daily peak kW using XGBoost."""
    model    = MODELS.get("xgb_peak")
    scaler_X = MODELS.get("scaler_X_peak")
    scaler_y = MODELS.get("scaler_y_peak")

    if None in (model, scaler_X, scaler_y):
        return {"error": "Peak model not loaded"}

    X_sc   = scaler_X.transform(X)
    pred_s = model.predict(X_sc)
    pred   = float(scaler_y.inverse_transform(pred_s.reshape(-1, 1))[0][0])
    return {"predicted_peak_kw": round(max(pred, 0.0), 4), "model": "XGBoost"}


# ── Task 3 — Appliance Disaggregation ────────────────────────────────────────

def disaggregate_appliances(
    X: np.ndarray,
    predicted_total_kw: float,
) -> dict:
    """
    Predict per-appliance fraction and absolute kW using trained disaggregators.
    Returns a sorted list of per-appliance records.
    """
    disagg_models = MODELS.get("disagg", {})
    scaler_X      = MODELS.get("scaler_X_energy")

    if not disagg_models or scaler_X is None:
        return {"error": "Disaggregation models not loaded"}

    X_sc = scaler_X.transform(X)
    items = []
    fractions_sum = 0.0

    for appl, model in disagg_models.items():
        frac = float(model.predict(X_sc)[0])
        frac = max(0.0, min(1.0, frac))
        kw   = frac * predicted_total_kw
        fractions_sum += frac
        items.append({
            "appliance":       appl,
            "fraction_pct":    round(frac * 100, 2),
            "absolute_kw":     round(kw, 4),
        })

    # Normalise fractions if they don't sum to 1
    if fractions_sum > 0:
        for item in items:
            item["fraction_pct"]  = round(item["fraction_pct"] / fractions_sum * 100, 2)
            item["absolute_kw"]   = round(item["absolute_kw"]  / fractions_sum, 4)

    items.sort(key=lambda x: x["absolute_kw"], reverse=True)
    return {
        "total_kw":   round(predicted_total_kw, 4),
        "appliances": items,
    }


# ── Task 4 — Room Anomaly Detection ──────────────────────────────────────────

def detect_anomalies(
    house_id: int,
    appliance_readings: dict,
    timestamp: str = "",
) -> dict:
    """
    Detect room-level anomalies and generate human-readable alerts.

    Parameters
    ----------
    house_id           : int
    appliance_readings : {canonical_room: current_kw_value}
    timestamp          : ISO string for alert messages
    """
    training_stats = MODELS.get("training_stats", {})
    if not training_stats:
        return {"error": "Training stats not loaded — anomaly detection unavailable"}

    return generate_room_alerts(
        house_id=house_id,
        current_readings=appliance_readings,
        training_stats=training_stats,
        timestamp=timestamp,
    )


# ── Task 5 — Optimization Recommendations ────────────────────────────────────

def get_recommendations(
    appliance_readings: dict,
    predicted_total_kw: Optional[float] = None,
    house_meta: Optional[dict] = None,
) -> list[str]:
    """Generate actionable energy optimization recommendations."""
    return generate_recommendations(
        appliance_readings=appliance_readings,
        predicted_total_kw=predicted_total_kw,
        house_meta=house_meta,
    )
