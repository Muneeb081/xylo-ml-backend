"""
Preprocessing Layer
===================
- Advanced imputation: ffill → bfill → linear interpolation (no data dropped)
- Time-based feature engineering
- Lag features for sequence context
- MinMaxScaler fit + transform
- Temporal train / val / test split (no data leakage)
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

from src.config import (
    CANONICAL_APPLIANCE_COLS, TIME_FEATURES, ALL_FEATURES,
    SCALER_X_ENERGY, SCALER_Y_ENERGY,
    SCALER_X_LSTM, SCALER_Y_LSTM,
    SCALER_X_PEAK, SCALER_Y_PEAK,
    TRAIN_RATIO, VAL_RATIO,
    USAGE_COL,
)


# ── Imputation ─────────────────────────────────────────────────────────────────

def impute_house(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligent imputation strategy — no rows are dropped.
    Order: forward-fill → backward-fill → linear interpolation → zero.
    Applied per-column to preserve temporal integrity.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Step 1: forward-fill (carry last known value)
    df[numeric_cols] = df[numeric_cols].ffill()

    # Step 2: backward-fill (fill leading NaN at start of series)
    df[numeric_cols] = df[numeric_cols].bfill()

    # Step 3: linear interpolation for isolated gaps
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

    # Step 4: remaining NaN → 0 (completely missing device)
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    return df


# ── Time Feature Engineering ───────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time encodings + boolean flags from the DatetimeIndex."""
    idx = df.index

    df["hour_sin"]  = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(float)

    return df


def add_lag_features(df: pd.DataFrame, col: str = USAGE_COL) -> pd.DataFrame:
    """
    Lag features for temporal context:
      lag_1h  : value 4 steps ago  (1 hour   with 15-min intervals)
      lag_24h : value 96 steps ago (24 hours)
      lag_7d  : value 672 steps ago (7 days)
    Rolling statistics (1h, 24h windows).
    """
    if col not in df.columns:
        return df

    df[f"{col}_lag_1h"]  = df[col].shift(4)
    df[f"{col}_lag_24h"] = df[col].shift(96)
    df[f"{col}_lag_7d"]  = df[col].shift(672)
    df[f"{col}_roll_1h_mean"]  = df[col].rolling(4,   min_periods=1).mean()
    df[f"{col}_roll_24h_mean"] = df[col].rolling(96,  min_periods=1).mean()
    df[f"{col}_roll_24h_std"]  = df[col].rolling(96,  min_periods=1).std().fillna(0)
    df[f"{col}_roll_1h_max"]   = df[col].rolling(4,   min_periods=1).max()
    df[f"{col}_roll_24h_max"]  = df[col].rolling(96,  min_periods=1).max()

    # Fill leading NaNs
    lag_cols = [c for c in df.columns if f"{col}_lag" in c]
    df[lag_cols] = df[lag_cols].bfill().fillna(0)

    return df


# ── Ensure canonical appliance cols exist ──────────────────────────────────────

def ensure_canonical_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing canonical appliance columns as zeros."""
    for col in CANONICAL_APPLIANCE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df


# ── Per-house full preprocessing ───────────────────────────────────────────────

def preprocess_house(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline for one house DataFrame."""
    df = impute_house(df)
    df = ensure_canonical_cols(df)
    df = add_time_features(df)
    df = add_lag_features(df, USAGE_COL)
    return df


# ── Combine all houses ─────────────────────────────────────────────────────────

def preprocess_all_houses(
    houses: Dict[int, pd.DataFrame],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply per-house preprocessing and concatenate into one master DataFrame.
    Adds a 'house_id' column if not present.
    """
    processed = []
    for house_id, df in houses.items():
        df = preprocess_house(df.copy())
        df["house_id"] = house_id
        processed.append(df)
        if verbose:
            print(f"  [Preprocessed] House {house_id:2d}: {len(df):>7,} rows")

    combined = pd.concat(processed, axis=0).sort_index()
    if verbose:
        print(f"\n  Combined DataFrame: {combined.shape}")
    return combined


# ── Train / Val / Test split (temporal, no leakage) ───────────────────────────

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float   = VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered split per house to avoid future-data leakage.
    Each house is split independently so all houses appear in every fold.
    """
    train_parts, val_parts, test_parts = [], [], []

    for house_id, group in df.groupby("house_id"):
        group = group.sort_index()
        n = len(group)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train: n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:])

    train = pd.concat(train_parts).sort_index()
    val   = pd.concat(val_parts).sort_index()
    test  = pd.concat(test_parts).sort_index()

    return train, val, test


# ── Scaling ────────────────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame, task: str = "energy") -> list[str]:
    """Return the ordered feature column list for a given task."""
    lag_cols = [c for c in df.columns if USAGE_COL + "_lag" in c or USAGE_COL + "_roll" in c]
    base = CANONICAL_APPLIANCE_COLS + TIME_FEATURES + lag_cols + ["n_rooms", "n_acs", "n_people", "area_sqft"]
    if task == "peak":
        base = ["house_id"] + base
    # Keep only columns that exist
    return [c for c in base if c in df.columns]


def fit_and_save_scalers(
    train: pd.DataFrame,
    task: str = "energy",   # "energy" | "peak" | "lstm"
) -> Tuple[MinMaxScaler, MinMaxScaler]:
    """Fit X and y scalers on training data and persist them."""
    feature_cols = get_feature_cols(train, task)
    target_col   = USAGE_COL

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_X.fit(train[feature_cols].values)
    scaler_y.fit(train[[target_col]].values)

    paths = {
        "energy": (SCALER_X_ENERGY, SCALER_Y_ENERGY),
        "peak":   (SCALER_X_PEAK,   SCALER_Y_PEAK),
        "lstm":   (SCALER_X_LSTM,   SCALER_Y_LSTM),
    }
    px, py = paths.get(task, (SCALER_X_ENERGY, SCALER_Y_ENERGY))
    joblib.dump(scaler_X, px)
    joblib.dump(scaler_y, py)

    return scaler_X, scaler_y


def scale_data(
    df: pd.DataFrame,
    scaler_X: MinMaxScaler,
    scaler_y: MinMaxScaler,
    task: str = "energy",
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform X and y using pre-fitted scalers."""
    feature_cols = get_feature_cols(df, task)
    X = scaler_X.transform(df[feature_cols].values)
    y = scaler_y.transform(df[[USAGE_COL]].values).ravel()
    return X, y
