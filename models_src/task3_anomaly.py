"""
Task 3 — Room-Wise Anomaly Detection
=====================================
Primary detector : Isolation Forest (per room, trained on 15-min data)
Secondary signal : Rolling Z-Score (real-time; no model needed)

At inference, both signals are combined to produce:
  - is_anomaly  : bool
  - room_alerts : list of human-readable alert strings
    e.g. "⚠ Living Room is consuming unusually high energy compared to its
          normal behavior (Z-score = 3.1, current = 2.4 kW, normal ≤ 0.8 kW)"
  - anomaly_details : full structured data per room
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from src.config import (
    CANONICAL_APPLIANCE_COLS, ROOM_DISPLAY,
    IF_CONTAMINATION, Z_THRESHOLD, ROLLING_WINDOW_STEPS,
    PLOTS_DIR, IF_MODEL_DIR, STATS_PATH, USAGE_COL,
)


# ── Training ───────────────────────────────────────────────────────────────────

def train_anomaly_detectors(
    combined: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """
    Train one Isolation Forest per (house_id, room_group).
    Also compute per-room rolling statistics for Z-score inference.
    Returns training summary.
    """
    training_stats: dict = {}          # {house_id: {room: {mean, std, if_model_path}}}
    available_rooms = [r for r in CANONICAL_APPLIANCE_COLS if r in combined.columns]

    print(f"\n  > Training Isolation Forest for {combined['house_id'].nunique()} houses "
          f"× {len(available_rooms)} rooms ...")

    for house_id, group in combined.groupby("house_id"):
        group = group.sort_index()
        training_stats[int(house_id)] = {}

        for room in available_rooms:
            col_data = group[room].values
            if col_data.sum() < 1e-6:
                continue   # room has no readings for this house

            # Feature matrix for IF: [room_kw, hour_sin, hour_cos]
            feat_cols = [room] + [c for c in ["hour_sin", "hour_cos", "dow_sin"] if c in group.columns]
            X = group[feat_cols].values.astype(np.float32)

            if_model = IsolationForest(
                contamination=IF_CONTAMINATION,
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            )
            if_model.fit(X)

            model_path = IF_MODEL_DIR / f"if_h{house_id}_{room}.joblib"
            joblib.dump(if_model, model_path)

            # Per-room statistics for Z-score
            mean_val = float(col_data.mean())
            std_val  = float(col_data.std()) or 1.0
            p95      = float(np.percentile(col_data[col_data > 0], 95)) if col_data.sum() > 0 else 0.0

            training_stats[int(house_id)][room] = {
                "mean":       mean_val,
                "std":        std_val,
                "p95":        p95,
                "model_path": str(model_path),
            }

        if verbose:
            n_rooms = len(training_stats[int(house_id)])
            print(f"    House {house_id:2d}: {n_rooms} room models trained")

    joblib.dump(training_stats, STATS_PATH)
    print(f"\n  [OK] Anomaly stats saved → {STATS_PATH}")
    _plot_anomaly_overview(combined, available_rooms)
    return training_stats


# ── Plot ───────────────────────────────────────────────────────────────────────

def _plot_anomaly_overview(combined: pd.DataFrame, rooms: list[str]):
    """Z-score distribution box-plot across all rooms."""
    z_data = {}
    for room in rooms:
        if room not in combined.columns:
            continue
        mu  = combined[room].mean()
        sig = combined[room].std() or 1.0
        z   = (combined[room] - mu) / sig
        z_data[ROOM_DISPLAY.get(room, room)] = z.values

    if not z_data:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(z_data) * 1.2), 6))
    ax.boxplot(
        list(z_data.values()),
        labels=list(z_data.keys()),
        patch_artist=True,
        boxprops=dict(facecolor="#AEC6E8"),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="x", color="#C44E52", markersize=5),
    )
    ax.axhline( Z_THRESHOLD, color="red",    linestyle="--", lw=1.5, label=f"+{Z_THRESHOLD}σ")
    ax.axhline(-Z_THRESHOLD, color="orange", linestyle="--", lw=1.0, label=f"-{Z_THRESHOLD}σ")
    ax.set_title("Z-Score Distribution per Room / Appliance Group")
    ax.set_ylabel("Z-Score"); ax.tick_params(axis="x", rotation=25); ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "anomaly_zscore_rooms.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Real-time alert generation (used by API) ──────────────────────────────────

def generate_room_alerts(
    house_id: int,
    current_readings: dict,      # {room: current_kw_value}
    training_stats: dict,        # loaded from STATS_PATH
    timestamp: str = "",
) -> dict:
    """
    Generate human-readable alerts for a single real-time reading.

    Parameters
    ----------
    house_id         : int
    current_readings : {canonical_room_name: kW_value}
    training_stats   : dict loaded from models/training_stats.joblib
    timestamp        : ISO timestamp string for the alert message

    Returns
    -------
    {
        "is_anomaly"   : bool,
        "alerts"       : [list of human-readable strings],
        "details"      : [{room, value_kw, mean_kw, z_score, severity}],
    }
    """
    house_stats = training_stats.get(int(house_id), {})
    alerts      = []
    details     = []
    is_anomaly  = False

    ts_str = f" at {timestamp}" if timestamp else ""

    for room, val in current_readings.items():
        val = float(val)
        if val < 0:
            continue
        stats = house_stats.get(room)
        if stats is None:
            continue

        mean_val = stats["mean"]
        std_val  = stats["std"]
        p95      = stats["p95"]
        z        = (val - mean_val) / std_val if std_val > 0 else 0.0
        display  = ROOM_DISPLAY.get(room, room.replace("_", " ").title())

        detail = {
            "room":       room,
            "display":    display,
            "value_kw":   round(val, 3),
            "mean_kw":    round(mean_val, 3),
            "p95_kw":     round(p95, 3),
            "z_score":    round(z, 2),
            "severity":   "normal",
        }

        if z > Z_THRESHOLD * 1.5:          # Severe
            detail["severity"] = "critical"
            alerts.append(
                f"🚨 CRITICAL — {display} is consuming dangerously high energy{ts_str}. "
                f"Current: {val:.2f} kW  |  Normal avg: {mean_val:.2f} kW  |  Z-score: {z:.1f}. "
                f"Immediate inspection recommended."
            )
            is_anomaly = True
        elif z > Z_THRESHOLD:               # Warning
            detail["severity"] = "warning"
            alerts.append(
                f"⚠ WARNING — {display} is consuming unusually high energy compared to its "
                f"normal behavior{ts_str}. "
                f"Current: {val:.2f} kW  |  Normal avg: {mean_val:.2f} kW  |  Z-score: {z:.1f}. "
                f"Please check this appliance."
            )
            is_anomaly = True
        elif val > p95 and p95 > 0:         # Above 95th percentile — info
            detail["severity"] = "info"
            alerts.append(
                f"ℹ INFO — {display} is above its 95th-percentile usage level{ts_str} "
                f"({val:.2f} kW > {p95:.2f} kW)."
            )

        details.append(detail)

    return {
        "is_anomaly": is_anomaly,
        "n_alerts":   len([d for d in details if d["severity"] in ("warning", "critical")]),
        "alerts":     alerts,
        "details":    details,
    }
