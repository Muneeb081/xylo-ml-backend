"""
Monthly Report Generator
========================
Core ML logic for generating monthly energy reports from daily energy_log data.

Takes a list of daily records (from Firestore energy_logs) and returns a
complete report dict ready to be written to Firestore ml_reports/{YYYY-MM}.

Called by:
  - scripts/generate_monthly_reports.py  (Cron job)
  - api/main.py  /api/v1/report/generate/{homeId}/{year}/{month}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Optional
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_monthly_report(
    home_id:      str,
    year:         int,
    month:        int,
    daily_records: list[dict],
) -> Optional[dict]:
    """
    Generate a full monthly ML report from daily energy_log records.

    Parameters
    ----------
    home_id        : Firestore home document ID
    year, month    : report period
    daily_records  : list of dicts from Firestore energy_logs

    Returns dict matching the ml_reports Firestore schema, or None if
    insufficient data (< 7 days).
    """
    if len(daily_records) < 7:
        print(f"  [WARN] Only {len(daily_records)} days — skipping report for {home_id} {year}-{month:02d}")
        return None

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = _build_dataframe(daily_records)

    # ── Basic stats ───────────────────────────────────────────────────────────
    total_kwh    = round(float(df["total_kwh"].sum()), 2)
    avg_daily    = round(float(df["total_kwh"].mean()), 2)
    peak_idx     = df["total_kwh"].idxmax()
    peak_day     = str(df.loc[peak_idx, "date"])
    peak_day_kwh = round(float(df.loc[peak_idx, "total_kwh"]), 2)

    # ── ML Predictions ────────────────────────────────────────────────────────
    predicted_next_month = _predict_next_month(df, total_kwh)
    predicted_peak_day   = _predict_peak_day(df)

    # ── Room breakdown ────────────────────────────────────────────────────────
    room_breakdown = _get_room_breakdown(df, total_kwh)

    # ── Anomaly detection ─────────────────────────────────────────────────────
    anomaly_days, anomaly_count = _detect_anomalies(df)

    # ── Weekly breakdown ──────────────────────────────────────────────────────
    weekly = _get_weekly_breakdown(df, year, month)

    # ── Recommendations ───────────────────────────────────────────────────────
    recommendations = _generate_recommendations(room_breakdown, anomaly_count, df)

    return {
        # Period
        "home_id":    home_id,
        "period":     f"{year:04d}-{month:02d}",
        "period_label": datetime(year, month, 1).strftime("%B %Y"),

        # Energy stats
        "total_kwh":          total_kwh,
        "avg_daily_kwh":      avg_daily,
        "peak_day":           peak_day,
        "peak_day_kwh":       peak_day_kwh,
        "days_in_report":     len(df),

        # ML Predictions
        "predicted_next_month_kwh": predicted_next_month,
        "predicted_peak_day_kwh":   predicted_peak_day,

        # Room breakdown (for pie chart in app)
        "room_breakdown": room_breakdown,

        # Anomaly detection
        "anomaly_days":  anomaly_days,
        "anomaly_count": anomaly_count,

        # Weekly breakdown (for bar chart in app)
        "weekly_breakdown": weekly,

        # Recommendations
        "recommendations": recommendations,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert Firestore records to a clean DataFrame."""
    rows = []
    for r in records:
        row = {
            "date":       r.get("date", ""),
            "total_kwh":  float(r.get("total_kwh", 0)),
            "devices_on": int(r.get("devices_on", 0)),
        }
        # Flatten room data
        rooms = r.get("rooms", {})
        for room, data in rooms.items():
            row[f"{room}_kwh"] = float(data.get("kwh", 0))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Fill missing room columns with 0
    for col in ["living_room_kwh", "kitchen_kwh", "bedroom_kwh", "drawing_room_kwh"]:
        if col not in df.columns:
            df[col] = 0.0

    return df


def _predict_next_month(df: pd.DataFrame, total_kwh: float) -> float:
    """
    Simple trend-based prediction for next month's total kWh.
    Uses a weighted average favouring recent days.
    Falls back to total * 1.0 if insufficient history.
    """
    try:
        # Load trained XGBoost model if available
        import joblib
        model_path  = ROOT / "models" / "xgb_energy.joblib"
        scaler_path = ROOT / "models" / "scaler_X_energy.joblib"

        if model_path.exists() and scaler_path.exists():
            model   = joblib.load(model_path)
            scaler  = joblib.load(scaler_path)
            features = _build_prediction_features(df)

            if features is not None:
                # Only use model if feature dimensions match
                n_expected = scaler.n_features_in_
                n_provided = features.shape[1]
                if n_provided == n_expected:
                    X          = scaler.transform(features)
                    pred_daily = float(model.predict(X)[0])
                    prediction = round(pred_daily * 30, 2)
                    print(f"    [ML] XGBoost prediction: {prediction} kWh/month")
                    return prediction
                else:
                    print(f"    [INFO] Feature dim mismatch ({n_provided} vs {n_expected}) - using trend")

    except Exception as e:
        print(f"    [WARN] XGBoost prediction error: {e} - using trend estimate")

    # Fallback: simple 3% month-over-month growth estimate
    days_in_month = len(df)
    daily_avg     = total_kwh / max(days_in_month, 1)
    return round(daily_avg * 30 * 1.03, 2)


def _build_prediction_features(df: pd.DataFrame):
    """Build a feature vector compatible with the trained XGBoost model."""
    try:
        from src.config import ALL_FEATURES
        import joblib

        last_row = df.iloc[-1]
        ts       = last_row["date"]

        row = {
            "Usage_kW":     last_row["total_kwh"] / 24,
            "hour_sin":     np.sin(2 * np.pi * 12 / 24),
            "hour_cos":     np.cos(2 * np.pi * 12 / 24),
            "dow_sin":      np.sin(2 * np.pi * ts.dayofweek / 7),
            "dow_cos":      np.cos(2 * np.pi * ts.dayofweek / 7),
            "month_sin":    np.sin(2 * np.pi * ts.month / 12),
            "month_cos":    np.cos(2 * np.pi * ts.month / 12),
            "is_weekend":   float(ts.dayofweek >= 5),
            "Usage_kW_lag_1h":      last_row["total_kwh"] / 24,
            "Usage_kW_lag_24h":     last_row["total_kwh"] / 24,
            "Usage_kW_lag_7d":      df["total_kwh"].tail(7).mean() / 24,
            "Usage_kW_roll_1h_mean":  last_row["total_kwh"] / 24,
            "Usage_kW_roll_24h_mean": df["total_kwh"].mean() / 24,
            "Usage_kW_roll_24h_std":  df["total_kwh"].std() / 24,
            "Usage_kW_roll_1h_max":   last_row["total_kwh"] / 24,
            "Usage_kW_roll_24h_max":  df["total_kwh"].max() / 24,
        }

        # Room kWh → canonical appliance columns
        room_map = {
            "ac":           last_row.get("living_room_kwh", 0) * 0.4,
            "kitchen":      last_row.get("kitchen_kwh", 0),
            "bedroom":      last_row.get("bedroom_kwh", 0),
            "living_room":  last_row.get("living_room_kwh", 0) * 0.6,
            "drawing_room": last_row.get("drawing_room_kwh", 0),
            "ups":          0.0,
            "water_pump":   0.0,
            "refrigerator": 0.0,
            "laundry":      0.0,
            "lights":       0.0,
            "fans":         0.0,
        }
        row.update(room_map)
        row.update({"n_rooms": 4, "n_acs": 1, "n_people": 4, "area_sqft": 1200})

        feature_vec = [[row.get(f, 0) for f in ALL_FEATURES]]
        return np.array(feature_vec)

    except Exception as e:
        print(f"    [WARN] Feature build failed: {e}")
        return None


def _predict_peak_day(df: pd.DataFrame) -> float:
    """Predict the peak day kWh for next month."""
    try:
        import joblib
        model_path = ROOT / "models" / "xgb_peak.joblib"
        if model_path.exists():
            # Use rolling max of last 7 days as a feature proxy
            peak_estimate = df["total_kwh"].tail(7).max() * 1.05
            return round(float(peak_estimate), 2)
    except Exception:
        pass
    return round(float(df["total_kwh"].max() * 1.05), 2)


def _get_room_breakdown(df: pd.DataFrame, total_kwh: float) -> list[dict]:
    """Calculate per-room kWh totals and percentages."""
    rooms = {
        "living_room":  "living_room_kwh",
        "kitchen":      "kitchen_kwh",
        "bedroom":      "bedroom_kwh",
        "drawing_room": "drawing_room_kwh",
    }
    breakdown = []
    for room_name, col in rooms.items():
        if col in df.columns:
            room_total = float(df[col].sum())
            pct        = round(room_total / max(total_kwh, 0.01) * 100, 1)
            breakdown.append({
                "room":  room_name,
                "label": room_name.replace("_", " ").title(),
                "kwh":   round(room_total, 2),
                "pct":   pct,
            })

    return sorted(breakdown, key=lambda x: x["kwh"], reverse=True)


def _detect_anomalies(df: pd.DataFrame) -> tuple[list[str], int]:
    """
    Detect anomaly days using Z-score (> 2.0 standard deviations above mean).
    """
    mean   = df["total_kwh"].mean()
    std    = df["total_kwh"].std()
    if std < 0.01:
        return [], 0

    z_scores    = (df["total_kwh"] - mean) / std
    anomaly_mask = z_scores > 2.0
    anomaly_days = df.loc[anomaly_mask, "date"].dt.strftime("%Y-%m-%d").tolist()
    return anomaly_days, len(anomaly_days)


def _get_weekly_breakdown(df: pd.DataFrame, year: int, month: int) -> list[dict]:
    """Split the month into weeks and sum kWh per week."""
    df = df.copy()
    df["week"] = df["date"].apply(lambda d: (d.day - 1) // 7 + 1)
    weekly = []

    for week_num in sorted(df["week"].unique()):
        week_df   = df[df["week"] == week_num]
        first_day = week_df["date"].min().strftime("%d").lstrip("0") or "1"
        last_day  = week_df["date"].max().strftime("%d %b").lstrip("0")
        weekly.append({
            "week":  int(week_num),
            "label": f"{first_day}-{last_day}",
            "kwh":   round(float(week_df["total_kwh"].sum()), 2),
            "days":  len(week_df),
        })

    return weekly


def _generate_recommendations(
    room_breakdown: list[dict],
    anomaly_count:  int,
    df:             pd.DataFrame,
) -> list[str]:
    """Generate human-readable energy-saving recommendations."""
    tips = []

    # High consumption rooms
    for room in room_breakdown:
        if room["pct"] >= 35:
            tips.append(
                f"🔴 {room['label']} accounts for {room['pct']}% of usage "
                f"({room['kwh']:.1f} kWh) — consider reducing runtime."
            )
        elif room["pct"] >= 20:
            tips.append(
                f"🟡 {room['label']} uses {room['pct']}% of total energy. "
                f"Switching off when idle could save ~{room['kwh']*0.15:.1f} kWh/month."
            )

    # Anomaly alert
    if anomaly_count > 0:
        tips.append(
            f"⚠️ {anomaly_count} unusual high-usage day(s) detected this month. "
            f"Check for devices left ON overnight."
        )

    # Weekend usage pattern
    df = df.copy()
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    weekend_avg = df.loc[df["is_weekend"], "total_kwh"].mean()
    weekday_avg = df.loc[~df["is_weekend"], "total_kwh"].mean()
    if weekend_avg > weekday_avg * 1.3:
        tips.append(
            f"📅 Weekend usage ({weekend_avg:.1f} kWh/day) is significantly higher "
            f"than weekdays ({weekday_avg:.1f} kWh/day). Review weekend device habits."
        )

    if not tips:
        tips.append("✅ Energy usage looks normal this month. Keep it up!")

    return tips
