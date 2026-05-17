"""
API Model Loader
================
Loads all trained model artefacts once at startup and exposes them
via a shared MODELS dict.
"""
from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any

from src.config import (
    MODELS_DIR, DISAGG_MODEL_DIR, IF_MODEL_DIR, STATS_PATH,
    XGB_ENERGY_PATH, XGB_PEAK_PATH, LSTM_MODEL_PATH,
    SCALER_X_ENERGY, SCALER_Y_ENERGY,
    SCALER_X_PEAK, SCALER_Y_PEAK,
    SCALER_X_LSTM, SCALER_Y_LSTM,
    CANONICAL_APPLIANCE_COLS,
)

MODELS: dict[str, Any] = {}


def load_models() -> None:
    """Load all persisted models into the MODELS registry."""

    # ── XGBoost energy ────────────────────────────────────────────────────────
    _load(XGB_ENERGY_PATH,   "xgb_energy")
    _load(SCALER_X_ENERGY,   "scaler_X_energy")
    _load(SCALER_Y_ENERGY,   "scaler_y_energy")

    # ── XGBoost peak ──────────────────────────────────────────────────────────
    _load(XGB_PEAK_PATH,     "xgb_peak")
    _load(SCALER_X_PEAK,     "scaler_X_peak")
    _load(SCALER_Y_PEAK,     "scaler_y_peak")

    # ── LSTM (optional) ───────────────────────────────────────────────────────
    if LSTM_MODEL_PATH.exists():
        try:
            import tensorflow as tf
            MODELS["lstm_energy"] = tf.keras.models.load_model(str(LSTM_MODEL_PATH))
            _load(SCALER_X_LSTM,  "scaler_X_lstm")
            _load(SCALER_Y_LSTM,  "scaler_y_lstm")
            print("  [OK] LSTM model loaded")
        except Exception as e:
            print(f"  [WARN] LSTM not loaded: {e}")

    # ── Disaggregation models ──────────────────────────────────────────────────
    disagg = {}
    for appl in CANONICAL_APPLIANCE_COLS:
        path = DISAGG_MODEL_DIR / f"disagg_{appl}.joblib"
        if path.exists():
            disagg[appl] = joblib.load(path)
    MODELS["disagg"] = disagg
    print(f"  [OK] Disaggregation models: {len(disagg)} appliances")

    # ── Isolation Forest anomaly models ───────────────────────────────────────
    if_models: dict = {}
    for path in IF_MODEL_DIR.glob("if_h*.joblib"):
        # Filename pattern: if_h{house_id}_{room}.joblib
        stem  = path.stem            # e.g. "if_h1_ac"
        parts = stem.split("_", 2)   # ["if", "h1", "ac"]
        if len(parts) >= 3:
            try:
                house_id = int(parts[1][1:])
                room     = parts[2]
                if_models.setdefault(house_id, {})[room] = joblib.load(path)
            except Exception:
                pass
    MODELS["if_models"] = if_models
    print(f"  [OK] Anomaly IF models: {sum(len(v) for v in if_models.values())} room models")

    # ── Training stats (for Z-score) ─────────────────────────────────────────
    if STATS_PATH.exists():
        MODELS["training_stats"] = joblib.load(STATS_PATH)
        print(f"  [OK] Training stats loaded ({len(MODELS['training_stats'])} houses)")

    print(f"  [OK] Total model keys: {[k for k in MODELS if k not in ('disagg', 'if_models')]}")


def _load(path: Path, key: str) -> None:
    if Path(path).exists():
        MODELS[key] = joblib.load(path)
        print(f"  [OK] {key}")
    else:
        print(f"  [WARN] Not found: {path}")


def get_status() -> dict:
    """Return which models are loaded."""
    return {
        "xgb_energy":      "xgb_energy" in MODELS,
        "xgb_peak":        "xgb_peak" in MODELS,
        "lstm_energy":     "lstm_energy" in MODELS,
        "disagg_models":   len(MODELS.get("disagg", {})),
        "if_models":       sum(len(v) for v in MODELS.get("if_models", {}).values()),
        "training_stats":  "training_stats" in MODELS,
    }
