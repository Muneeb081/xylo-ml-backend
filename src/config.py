"""
Central configuration for the PRECON Energy ML System.
All paths, hyperparameters, room mappings, and feature definitions live here.
"""
import os
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
RAW_DATA_DIR    = BASE_DIR / "PRECON"
METADATA_PATH   = BASE_DIR / "Metadata.csv"
MODELS_DIR      = BASE_DIR / "models"
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
SPLITS_DIR      = BASE_DIR / "data" / "splits"
PLOTS_DIR       = BASE_DIR / "plots"

for d in [MODELS_DIR, PROCESSED_DIR, SPLITS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data settings ─────────────────────────────────────────────────────────────
RESAMPLE_FREQ   = "15min"     # resample raw 1-min data to 15-min
TIMESTAMP_COL   = "timestamp"
USAGE_COL       = "Usage_kW"

# LSTM sequence settings
SEQ_LEN         = 96          # 96 × 15-min = 24 hours of history
FORECAST_STEPS  = 4           # predict next 1 hour (4 × 15-min)
LSTM_EPOCHS     = 50
LSTM_BATCH      = 64
LSTM_UNITS      = [128, 64]   # two LSTM layers
EARLY_STOPPING_PATIENCE = 8

# Train / validation / test ratio (time-based split)
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
# test = remaining 0.15

# ── Canonical room / appliance column mapping ─────────────────────────────────
# Keys = canonical group names; values = regex patterns matched against CSV col names
import re

CANONICAL_ROOM_PATTERNS = {
    "ac":           r"(?i)\bac\b|aircon|air.con",
    "ups":          r"(?i)\bups\b",
    "living_room":  r"(?i)\blr\b|living",
    "kitchen":      r"(?i)kitchen|\bkit\b",
    "bedroom":      r"(?i)\bbr\b|bed.?room",
    "drawing_room": r"(?i)\bdr\b|drawing",
    "water_pump":   r"(?i)\bwp\b|water.?pump|\bpump\b",
    "refrigerator": r"(?i)fridge|refrig",
    "laundry":      r"(?i)laundry|washing|washer",
    "lights":       r"(?i)\blight|bulb",
    "fans":         r"(?i)\bfan\b",
}

# Human-readable display names for rooms
ROOM_DISPLAY = {
    "ac":           "Air Conditioning",
    "ups":          "UPS / Power Backup",
    "living_room":  "Living Room",
    "kitchen":      "Kitchen",
    "bedroom":      "Bedroom",
    "drawing_room": "Drawing Room",
    "water_pump":   "Water Pump",
    "refrigerator": "Refrigerator",
    "laundry":      "Laundry",
    "lights":       "Lighting",
    "fans":         "Fans",
}

# All canonical columns (features fed to models)
CANONICAL_APPLIANCE_COLS = list(CANONICAL_ROOM_PATTERNS.keys())

# Full feature set (appliances + time encodings + metadata)
TIME_FEATURES = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin","month_cos",
    "is_weekend",
]
META_FEATURES = ["n_rooms", "n_acs", "n_people", "area_sqft"]

ALL_FEATURES = CANONICAL_APPLIANCE_COLS + TIME_FEATURES + META_FEATURES

# ── Anomaly detection ─────────────────────────────────────────────────────────
Z_THRESHOLD          = 2.5     # standard deviations
IF_CONTAMINATION     = 0.05    # Isolation Forest contamination
ROLLING_WINDOW_STEPS = 96      # rolling baseline window (24 h)

# ── Optimization thresholds ───────────────────────────────────────────────────
HIGH_IMPACT_PCT  = 30.0
MODERATE_PCT     = 15.0
LOW_MEDIUM_PCT   =  5.0
HIGH_TOTAL_KW    =  5.0       # kW threshold for caution alert

# ── Scaler filenames ──────────────────────────────────────────────────────────
SCALER_X_ENERGY  = MODELS_DIR / "scaler_X_energy.joblib"
SCALER_Y_ENERGY  = MODELS_DIR / "scaler_y_energy.joblib"
SCALER_X_PEAK    = MODELS_DIR / "scaler_X_peak.joblib"
SCALER_Y_PEAK    = MODELS_DIR / "scaler_y_peak.joblib"
SCALER_X_LSTM    = MODELS_DIR / "scaler_X_lstm.joblib"
SCALER_Y_LSTM    = MODELS_DIR / "scaler_y_lstm.joblib"

# ── Model filenames ───────────────────────────────────────────────────────────
LSTM_MODEL_PATH       = MODELS_DIR / "lstm_energy.keras"
XGB_ENERGY_PATH       = MODELS_DIR / "xgb_energy.joblib"
XGB_PEAK_PATH         = MODELS_DIR / "xgb_peak.joblib"
IF_MODEL_DIR          = MODELS_DIR / "anomaly"
DISAGG_MODEL_DIR      = MODELS_DIR / "disaggregation"
STATS_PATH            = MODELS_DIR / "training_stats.joblib"

for d in [IF_MODEL_DIR, DISAGG_MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)
