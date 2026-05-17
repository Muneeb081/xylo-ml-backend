"""
API Preprocessing
=================
Converts incoming JSON (real-time or Firebase) into model-ready NumPy arrays.
Handles schema consistency between training and inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime

from src.config import CANONICAL_APPLIANCE_COLS, CANONICAL_ROOM_PATTERNS, USAGE_COL
from src.preprocessing import get_feature_cols
import re


def _get_season(month: int) -> int:
    return {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
            6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}.get(month, 2)


def _canonicalize_input_dict(data: dict) -> dict:
    """
    Map any incoming appliance keys to canonical group names.
    Non-matching keys are silently ignored for model features
    but kept in 'extra_readings' for disaggregation display.
    """
    canonical = {c: 0.0 for c in CANONICAL_APPLIANCE_COLS}
    extras    = {}

    protected = {"house_id", "timestamp", USAGE_COL, "monthly_average_kw",
                 "year", "month", "n_rooms", "n_acs", "n_people", "area_sqft"}

    for key, val in data.items():
        if key in protected:
            continue
        val = float(val or 0.0)
        matched = False
        for group, pattern in CANONICAL_ROOM_PATTERNS.items():
            if re.search(pattern, key):
                canonical[group] = canonical.get(group, 0.0) + val
                matched = True
                break
        if not matched:
            extras[key] = val

    return canonical, extras


def build_feature_row(data: dict, feature_cols: list[str]) -> np.ndarray:
    """
    Build a 1-row feature vector matching `feature_cols` from an API payload.

    Parameters
    ----------
    data         : raw API payload dict
    feature_cols : ordered list of feature names (from training)

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    ts = data.get("timestamp")
    if ts:
        dt    = pd.to_datetime(str(ts))
        month = dt.month
        hour  = dt.hour
        dow   = dt.dayofweek
    else:
        now   = datetime.now()
        month = now.month
        hour  = now.hour
        dow   = now.weekday()

    canonical, _ = _canonicalize_input_dict(data)
    total_kw     = float(data.get(USAGE_COL) or sum(canonical.values()) or 0.0)

    lookup = {
        **canonical,
        "house_id":     float(data.get("house_id") or 0),
        USAGE_COL:      total_kw,
        "hour_sin":     float(np.sin(2 * np.pi * hour  / 24)),
        "hour_cos":     float(np.cos(2 * np.pi * hour  / 24)),
        "dow_sin":      float(np.sin(2 * np.pi * dow   / 7)),
        "dow_cos":      float(np.cos(2 * np.pi * dow   / 7)),
        "month_sin":    float(np.sin(2 * np.pi * month / 12)),
        "month_cos":    float(np.cos(2 * np.pi * month / 12)),
        "is_weekend":   float(dow >= 5),
        "n_rooms":      float(data.get("n_rooms") or 0),
        "n_acs":        float(data.get("n_acs") or 0),
        "n_people":     float(data.get("n_people") or 0),
        "area_sqft":    float(data.get("area_sqft") or 0),
        # Lag features → use current reading as proxy when history unavailable
        f"{USAGE_COL}_lag_1h":       total_kw,
        f"{USAGE_COL}_lag_24h":      total_kw,
        f"{USAGE_COL}_lag_7d":       total_kw,
        f"{USAGE_COL}_roll_1h_mean": total_kw,
        f"{USAGE_COL}_roll_24h_mean":total_kw,
        f"{USAGE_COL}_roll_24h_std": 0.0,
        f"{USAGE_COL}_roll_1h_max":  total_kw,
    }

    row = np.array([lookup.get(col, 0.0) for col in feature_cols], dtype=np.float64)
    return row.reshape(1, -1)


def get_appliance_readings(data: dict) -> dict:
    """
    Extract all appliance readings (canonical + extra) from the payload.
    Used for disaggregation display and recommendation engine.
    """
    canonical, extras = _canonicalize_input_dict(data)
    result = {k: v for k, v in canonical.items() if v > 0}
    result.update({k: v for k, v in extras.items() if v > 0})
    return result
