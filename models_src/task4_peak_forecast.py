"""
Task 4 — Peak Load Forecasting
================================
XGBoost Regressor predicting daily_peak_kw from time + appliance features.
Trained on 15-min data with daily peak as the target.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.config import (
    MODELS_DIR, PLOTS_DIR, XGB_PEAK_PATH,
    SCALER_X_PEAK, SCALER_Y_PEAK, USAGE_COL,
)
from src.preprocessing import get_feature_cols


def train_peak_forecast(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    scaler_X,
    scaler_y_peak,
) -> dict:
    """Train XGBoost peak load forecasting model."""
    feature_cols = get_feature_cols(train, task="peak")
    feature_cols = [c for c in feature_cols if c in train.columns]

    target = "daily_peak_kw"
    if target not in train.columns:
        train[target] = train.groupby([train.index.date, "house_id"])[USAGE_COL].transform("max")
        test[target]  = test.groupby([test.index.date,  "house_id"])[USAGE_COL].transform("max")

    # Resample to daily maximums for stable daily peak forecasting
    train_daily = train.groupby([train.index.date, "house_id"], as_index=False).max()
    test_daily  = test.groupby([test.index.date, "house_id"], as_index=False).max()

    # Scale X
    X_tr = scaler_X.transform(train_daily[feature_cols].values)
    X_te = scaler_X.transform(test_daily[feature_cols].values)

    # Scale y (peak)
    y_tr = scaler_y_peak.fit_transform(train_daily[[target]].values).ravel()
    y_te_raw = test_daily[target].values

    print("\n  > Training XGBoost Peak Load Forecaster (Daily Aggregated) ...")
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, verbose=False)

    pred_scaled = model.predict(X_te)
    pred        = scaler_y_peak.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    mae  = float(mean_absolute_error(y_te_raw, pred))
    rmse = float(np.sqrt(mean_squared_error(y_te_raw, pred)))
    r2   = float(r2_score(y_te_raw, pred))
    print(f"    Peak Forecast → MAE={mae:.4f} kW  RMSE={rmse:.4f} kW  R²={r2:.4f}")

    joblib.dump(model, XGB_PEAK_PATH)
    joblib.dump(scaler_y_peak, SCALER_Y_PEAK)
    print(f"    [OK] Peak model saved → {XGB_PEAK_PATH}")

    # Plot
    _plot_pred_vs_actual(y_te_raw, pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "model": "XGBoost Peak"}


def _plot_pred_vs_actual(y_true, y_pred):
    sample = min(3000, len(y_true))
    idx    = np.random.choice(len(y_true), sample, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_true[idx], y_pred[idx], alpha=0.4, s=12, color="#4C72B0")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual Peak kW"); axes[0].set_ylabel("Predicted Peak kW")
    axes[0].set_title("XGBoost Peak Forecast — Actual vs Predicted"); axes[0].legend()
    residuals = y_pred[idx] - y_true[idx]
    axes[1].scatter(y_pred[idx], residuals, alpha=0.4, s=12, color="#DD8452")
    axes[1].axhline(0, color="r", lw=1.5, linestyle="--")
    axes[1].set_xlabel("Predicted Peak kW"); axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual Plot")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "xgb_peak_pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
