"""
Feature Store
=============
Builds task-specific datasets from the preprocessed combined DataFrame and
persists them as Parquet for fast reloading.

Datasets produced:
  1. tabular_energy.parquet  — XGBoost energy/peak models
  2. sequences_lstm.npz      — LSTM sequence arrays (X, y)
  3. disagg_{appliance}.parquet — per-appliance disaggregation targets
  4. anomaly_{house_id}.parquet — per-house data for Isolation Forest
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import (
    PROCESSED_DIR, SPLITS_DIR,
    CANONICAL_APPLIANCE_COLS, USAGE_COL,
    SEQ_LEN, FORECAST_STEPS,
)
from src.preprocessing import get_feature_cols


# ── 1. Tabular dataset (XGBoost Energy + Peak) ────────────────────────────────

def build_tabular_dataset(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a flat tabular DataFrame with features + targets.
    Target 1: Usage_kW (energy consumption)
    Target 2: daily_peak_kw (computed as max Usage_kW in same day)
    """
    df = combined.copy()

    # Daily peak — computed from 15-min data
    df["date"] = df.index.date
    daily_peak = df.groupby(["house_id", "date"])[USAGE_COL].max().reset_index()
    daily_peak.columns = ["house_id", "date", "daily_peak_kw"]
    df = df.merge(daily_peak, on=["house_id", "date"], how="left")
    df = df.drop(columns=["date"])

    out_path = PROCESSED_DIR / "tabular_energy.parquet"
    df.to_parquet(out_path)
    print(f"  [FeatureStore] Tabular dataset saved → {out_path}  ({len(df):,} rows)")
    return df


# ── 2. LSTM sequence dataset ───────────────────────────────────────────────────

def build_sequence_dataset(
    combined: pd.DataFrame,
    feature_cols: list[str],
    seq_len:  int = SEQ_LEN,
    fcast:    int = FORECAST_STEPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build overlapping sliding-window sequences per house.

    Returns
    -------
    X : shape (N, seq_len, n_features)
    y : shape (N, fcast)           — next `fcast` steps of Usage_kW
    """
    X_list, y_list = [], []

    for house_id, group in combined.groupby("house_id"):
        group = group.sort_index()
        if len(group) < seq_len + fcast:
            continue

        feat = group[feature_cols].values.astype(np.float32)
        tgt  = group[USAGE_COL].values.astype(np.float32)

        for i in range(len(group) - seq_len - fcast + 1):
            X_list.append(feat[i : i + seq_len])
            y_list.append(tgt[i + seq_len : i + seq_len + fcast])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    out_path = PROCESSED_DIR / "sequences_lstm.npz"
    np.savez_compressed(out_path, X=X, y=y)
    print(f"  [FeatureStore] LSTM sequences saved → {out_path}  "
          f"X={X.shape}, y={y.shape}")
    return X, y


# ── 3. Disaggregation dataset ─────────────────────────────────────────────────

def build_disagg_dataset(combined: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    For each canonical appliance, save only the target columns (not full feature matrix)
    to avoid memory errors on large datasets.
    """
    datasets: Dict[str, pd.DataFrame] = {}
    total = combined[USAGE_COL].replace(0, np.nan)

    # Only store house_id, timestamp index, usage, and per-appliance targets
    base = pd.DataFrame({
        "house_id": combined["house_id"],
        USAGE_COL:  combined[USAGE_COL],
    }, index=combined.index)

    for appl in CANONICAL_APPLIANCE_COLS:
        if appl not in combined.columns:
            continue
        df = base.copy()
        df[f"{appl}_kw"]       = combined[appl].values
        df[f"{appl}_fraction"] = (combined[appl] / total).fillna(0).clip(0, 1).values

        out_path = PROCESSED_DIR / f"disagg_{appl}.parquet"
        df.to_parquet(out_path)
        datasets[appl] = df

    print(f"  [FeatureStore] Disaggregation datasets saved for {len(datasets)} appliances")
    return datasets


# ── 4. Per-house anomaly dataset ──────────────────────────────────────────────

def build_anomaly_dataset(combined: pd.DataFrame) -> None:
    """
    Save per-house 15-min data to individual parquet files for Isolation Forest.
    """
    rooms = [c for c in CANONICAL_APPLIANCE_COLS if c in combined.columns]

    for house_id, group in combined.groupby("house_id"):
        cols = [USAGE_COL] + rooms + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        cols = [c for c in cols if c in group.columns]
        out_path = PROCESSED_DIR / f"anomaly_house{house_id}.parquet"
        group[cols].to_parquet(out_path)

    print(f"  [FeatureStore] Anomaly datasets saved for {combined['house_id'].nunique()} houses")


# ── Master builder ─────────────────────────────────────────────────────────────

def build_all(
    combined: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    scaler_X,
    scaler_y,
) -> dict:
    """Build and persist all feature-store datasets. Returns metadata dict."""
    feature_cols = get_feature_cols(combined)

    print("\n[*] Building feature store ...")

    tabular = build_tabular_dataset(combined)

    # Scale features for LSTM sequences
    feat_scaled = scaler_X.transform(combined[feature_cols].values)
    tgt_scaled  = scaler_y.transform(combined[[USAGE_COL]].values).ravel()

    combined_scaled = combined.copy()
    combined_scaled[feature_cols] = feat_scaled
    combined_scaled[USAGE_COL]    = tgt_scaled

    X_seq, y_seq = build_sequence_dataset(combined_scaled, feature_cols)

    build_disagg_dataset(combined)
    build_anomaly_dataset(combined)

    # Save train/val/test splits
    train.to_parquet(SPLITS_DIR / "train.parquet")
    val.to_parquet(SPLITS_DIR / "val.parquet")
    test.to_parquet(SPLITS_DIR / "test.parquet")
    print(f"  [FeatureStore] Splits saved → data/splits/")

    return {
        "n_sequences":   len(X_seq),
        "seq_shape":     X_seq.shape,
        "n_features":    len(feature_cols),
        "feature_cols":  feature_cols,
        "n_train":       len(train),
        "n_val":         len(val),
        "n_test":        len(test),
    }
