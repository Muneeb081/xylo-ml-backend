"""
Master Training Pipeline — PRECON Energy ML System
====================================================
Orchestrates all 5 tasks end-to-end:
  Task 1 : Energy consumption prediction (LSTM + XGBoost)
  Task 2 : Appliance energy disaggregation (XGBoost per appliance)
  Task 3 : Room-wise anomaly detection (Isolation Forest + Z-score)
  Task 4 : Peak load forecasting (XGBoost)
  Task 5 : Optimization recommendations (rule engine — no training)

Usage
-----
  python train_pipeline.py                  # full run (all 42 houses)
  python train_pipeline.py --quick          # skip LSTM, use 5 houses
  python train_pipeline.py --houses 10      # use first N houses
  python train_pipeline.py --no-lstm        # skip LSTM, run all other tasks
"""
from __future__ import annotations

import sys, io, argparse, json, time, warnings
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

# ── Project root on sys.path ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    MODELS_DIR, PLOTS_DIR, SCALER_X_ENERGY, SCALER_Y_ENERGY,
    SCALER_X_LSTM, SCALER_Y_LSTM, SCALER_X_PEAK, SCALER_Y_PEAK,
    STATS_PATH,
)
from src.data_ingestion  import load_all_houses
from src.preprocessing   import (
    preprocess_all_houses, temporal_split,
    fit_and_save_scalers, get_feature_cols,
)
from src.feature_store   import build_all

from models_src.task1_energy_lstm   import train_lstm, train_xgb_energy
from models_src.task2_disaggregation import train_disaggregation
from models_src.task3_anomaly        import train_anomaly_detectors
from models_src.task4_peak_forecast  import train_peak_forecast


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="PRECON Energy ML Pipeline")
    parser.add_argument("--quick",    action="store_true", help="Quick mode: 5 houses, no LSTM")
    parser.add_argument("--no-lstm",  action="store_true", help="Skip LSTM training")
    parser.add_argument("--houses",   type=int, default=None, help="Limit number of houses loaded")
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    t0    = time.time()
    limit = 5 if args.quick else args.houses
    skip_lstm = args.quick or args.no_lstm

    print("=" * 70)
    print("  PRECON SMART ENERGY ML PIPELINE")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode    : {'Quick (5 houses, no LSTM)' if args.quick else 'Full'}")
    print("=" * 70)

    # ── SECTION 1: DATA INGESTION ──────────────────────────────────────────────
    print("\n[SECTION 1] Loading raw house data ...")
    houses, metadata = load_all_houses(limit=limit, verbose=True)

    # ── SECTION 2: PREPROCESSING ───────────────────────────────────────────────
    print("\n[SECTION 2] Preprocessing ...")
    combined = preprocess_all_houses(houses, verbose=True)

    # ── SECTION 3: SPLITS & SCALERS ────────────────────────────────────────────
    print("\n[SECTION 3] Temporal train / val / test split ...")
    train, val, test = temporal_split(combined)
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    print("\n  Fitting MinMaxScalers on training set ...")
    scaler_X_energy, scaler_y_energy = fit_and_save_scalers(train, task="energy")
    scaler_X_peak,   scaler_y_peak   = fit_and_save_scalers(train, task="peak")
    scaler_X_lstm,   scaler_y_lstm   = fit_and_save_scalers(train, task="lstm")
    print("  [OK] Scalers saved → models/")

    # ── SECTION 4: FEATURE STORE ───────────────────────────────────────────────
    print("\n[SECTION 4] Building feature store ...")
    feature_cols = get_feature_cols(combined)
    fs_meta = build_all(combined, train, val, test, scaler_X_lstm, scaler_y_lstm)

    # ── SECTION 5: TASK 1 — ENERGY PREDICTION ─────────────────────────────────
    print("\n" + "=" * 70)
    print("[TASK 1] Energy Consumption Prediction")
    print("=" * 70)

    xgb_energy_metrics = train_xgb_energy(train, test, scaler_X_energy, scaler_y_energy, feature_cols)

    lstm_metrics = {}
    if not skip_lstm:
        print("\n  [LSTM] Training sequence model ...")
        lstm_metrics = train_lstm(train, val, scaler_X_lstm, scaler_y_lstm, feature_cols)
    else:
        print("\n  [LSTM] Skipped (--no-lstm or --quick flag)")

    # ── SECTION 6: TASK 2 — DISAGGREGATION ────────────────────────────────────
    print("\n" + "=" * 70)
    print("[TASK 2] Appliance Energy Disaggregation")
    print("=" * 70)
    disagg_metrics = train_disaggregation(train, test, scaler_X_energy)

    # ── SECTION 7: TASK 3 — ANOMALY DETECTION ─────────────────────────────────
    print("\n" + "=" * 70)
    print("[TASK 3] Room-Wise Anomaly Detection")
    print("=" * 70)
    anomaly_stats = train_anomaly_detectors(combined, verbose=True)

    # ── SECTION 8: TASK 4 — PEAK LOAD FORECAST ────────────────────────────────
    print("\n" + "=" * 70)
    print("[TASK 4] Peak Load Forecasting")
    print("=" * 70)
    peak_metrics = train_peak_forecast(train, test, scaler_X_peak, scaler_y_peak)

    # ── TASK 5: No training (rule engine) ─────────────────────────────────────
    print("\n[TASK 5] Optimization Recommendations — rule engine (no training needed)")

    # ── FINAL SUMMARY ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  Houses loaded   : {len(houses)}")
    print(f"  Total rows      : {len(combined):,}  (15-min intervals)")
    print(f"  Feature cols    : {len(feature_cols)}")
    print(f"  LSTM sequences  : {fs_meta['n_sequences']:,}")

    print(f"\n  [TASK 1 — XGBoost Energy]")
    print(f"    MAE  = {xgb_energy_metrics.get('MAE', 'N/A'):.4f} kW")
    print(f"    RMSE = {xgb_energy_metrics.get('RMSE', 'N/A'):.4f} kW")
    print(f"    R²   = {xgb_energy_metrics.get('R2', 'N/A'):.4f}")

    if lstm_metrics:
        print(f"\n  [TASK 1 — LSTM Energy]")
        print(f"    MAE  = {lstm_metrics.get('MAE', 'N/A'):.4f} kW")
        print(f"    RMSE = {lstm_metrics.get('RMSE', 'N/A'):.4f} kW")
        print(f"    R²   = {lstm_metrics.get('R2', 'N/A'):.4f}")

    print(f"\n  [TASK 2 — Disaggregation] {len(disagg_metrics)} appliance models trained")
    for appl, m in disagg_metrics.items():
        print(f"    {appl:20s}  R²={m['R2']:.4f}")

    print(f"\n  [TASK 3 — Anomaly]  Isolation Forests trained for {len(anomaly_stats)} houses")

    print(f"\n  [TASK 4 — Peak Forecast]")
    print(f"    MAE  = {peak_metrics.get('MAE', 'N/A'):.4f} kW")
    print(f"    RMSE = {peak_metrics.get('RMSE', 'N/A'):.4f} kW")
    print(f"    R²   = {peak_metrics.get('R2', 'N/A'):.4f}")

    print(f"\n  [MODELS]")
    for f in sorted(MODELS_DIR.rglob("*.joblib")) + sorted(MODELS_DIR.rglob("*.keras")):
        print(f"    {f.relative_to(MODELS_DIR)}")

    # Export metrics JSON
    metrics_out = {
        "generated_at": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "houses_loaded": len(houses),
        "total_rows": int(len(combined)),
        "n_features": len(feature_cols),
        "task1_xgb_energy": xgb_energy_metrics,
        "task1_lstm_energy": lstm_metrics,
        "task2_disaggregation": disagg_metrics,
        "task3_anomaly_houses": len(anomaly_stats),
        "task4_peak": peak_metrics,
        "feature_cols": feature_cols,
    }
    metrics_path = ROOT / "_pipeline_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics_out, fh, indent=2, default=str)

    print(f"\n  [OK] Metrics exported → {metrics_path}")
    print(f"  [OK] Pipeline completed in {elapsed/60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
