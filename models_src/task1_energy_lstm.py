"""
Task 1 — Energy Consumption Prediction
======================================
Primary model  : LSTM / GRU (TensorFlow/Keras)
Fallback model : XGBoost Regressor with lag features

Both models predict Usage_kW (the total household power draw).
LSTM operates on 24-hour (96-step) sliding windows at 15-min resolution.
XGBoost uses tabular data with time/lag features.
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

from src.config import (
    MODELS_DIR, PLOTS_DIR,
    LSTM_MODEL_PATH, XGB_ENERGY_PATH,
    SCALER_X_LSTM, SCALER_Y_LSTM,
    SCALER_X_ENERGY, SCALER_Y_ENERGY,
    SEQ_LEN, FORECAST_STEPS,
    LSTM_EPOCHS, LSTM_BATCH, LSTM_UNITS, EARLY_STOPPING_PATIENCE,
    USAGE_COL, PROCESSED_DIR,
)
from src.preprocessing import get_feature_cols


# ── Metrics helper ─────────────────────────────────────────────────────────────

def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    print(f"    {name:25s} → MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


# ── LSTM model ─────────────────────────────────────────────────────────────────

def build_lstm_model(n_features: int, n_out: int = FORECAST_STEPS):
    """
    2-layer LSTM architecture:
      Input  → LSTM(128) → Dropout(0.2) → LSTM(64) → Dense(n_out)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            Input(shape=(SEQ_LEN, n_features)),
            LSTM(LSTM_UNITS[0], return_sequences=True),
            Dropout(0.2),
            LSTM(LSTM_UNITS[1], return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(n_out),
        ])
        model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
        return model
    except ImportError:
        return None


def train_lstm(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    scaler_X,
    scaler_y,
    feature_cols: list[str],
) -> dict:
    """Train the LSTM model on sequence data. Returns metrics dict."""
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    except ImportError:
        print("  [WARN] TensorFlow not found — skipping LSTM training")
        return {}

    # Load pre-built sequences
    seq_path = PROCESSED_DIR / "sequences_lstm.npz"
    if not seq_path.exists():
        print("  [WARN] sequences_lstm.npz not found — skipping LSTM")
        return {}

    data = np.load(seq_path)
    X_all, y_all = data["X"], data["y"]

    if len(X_all) == 0:
        return {}

    # Time-based split using pre-computed index sizes
    n_train = int(len(X_all) * 0.70)
    n_val   = int(len(X_all) * 0.15)
    X_tr, y_tr = X_all[:n_train],            y_all[:n_train]
    X_vl, y_vl = X_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val]
    X_te, y_te = X_all[n_train+n_val:],       y_all[n_train+n_val:]

    n_features = X_tr.shape[2]
    n_out      = y_tr.shape[1]

    print(f"\n  LSTM input shape : {X_tr.shape}")
    print(f"  LSTM output shape: {y_tr.shape}")

    model = build_lstm_model(n_features, n_out)
    if model is None:
        return {}

    callbacks = [
        EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(patience=4, factor=0.5, verbose=0),
        ModelCheckpoint(str(LSTM_MODEL_PATH), save_best_only=True, verbose=0),
    ]

    print(f"\n  > Training LSTM ({LSTM_EPOCHS} epochs max, batch={LSTM_BATCH}) ...")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    y_pred = model.predict(X_te, verbose=0)

    # Denormalize (use only first forecast step for scalar metrics)
    y_true_dn = scaler_y.inverse_transform(y_te[:, :1]).ravel()
    y_pred_dn = scaler_y.inverse_transform(y_pred[:, :1]).ravel()

    metrics = eval_metrics(y_true_dn, y_pred_dn, "LSTM Energy")

    # Plot training history
    _plot_lstm_history(history, "LSTM Energy Training History", "lstm_training_history.png")

    # Plot predictions vs actual
    _plot_pred_vs_actual(y_true_dn, y_pred_dn, "LSTM", "kW", "lstm_pred_vs_actual.png")

    print(f"    [OK] LSTM model saved → {LSTM_MODEL_PATH}")
    return metrics


# ── XGBoost fallback / complementary model ────────────────────────────────────

def train_xgb_energy(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    scaler_X,
    scaler_y,
    feature_cols: list[str],
) -> dict:
    """Train XGBoost energy regressor on scaled tabular data."""
    X_tr = scaler_X.transform(train[feature_cols].values)
    y_tr = scaler_y.transform(train[[USAGE_COL]].values).ravel()
    X_te = scaler_X.transform(test[feature_cols].values)
    y_te = test[USAGE_COL].values

    print("\n  > Training XGBoost Energy Regressor ...")
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(scaler_X.transform(test[feature_cols].values), y_te)],
        verbose=False,
    )

    pred_scaled = model.predict(X_te)
    pred        = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    metrics     = eval_metrics(y_te, pred, "XGBoost Energy")

    joblib.dump(model, XGB_ENERGY_PATH)
    print(f"    [OK] XGBoost energy model saved → {XGB_ENERGY_PATH}")

    _plot_pred_vs_actual(y_te, pred, "XGBoost Energy", "kW", "xgb_energy_pred_vs_actual.png")
    _plot_feature_importance(model.feature_importances_, feature_cols, "XGBoost Energy", "xgb_energy_feat_imp.png")

    return metrics


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_lstm_history(history, title: str, filename: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{title} — Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(history.history.get("mae", []),     label="Train MAE")
    axes[1].plot(history.history.get("val_mae", []), label="Val MAE")
    axes[1].set_title(f"{title} — MAE"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight"); plt.close()


def _plot_pred_vs_actual(y_true, y_pred, model_name: str, unit: str, filename: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sample = min(2000, len(y_true))
    idx = np.random.choice(len(y_true), sample, replace=False)
    axes[0].scatter(y_true[idx], y_pred[idx], alpha=0.4, s=12, color="#4C72B0")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
    axes[0].set_xlabel(f"Actual ({unit})"); axes[0].set_ylabel(f"Predicted ({unit})")
    axes[0].set_title(f"{model_name} — Actual vs Predicted"); axes[0].legend()
    residuals = y_pred[idx] - y_true[idx]
    axes[1].scatter(y_pred[idx], residuals, alpha=0.4, s=12, color="#DD8452")
    axes[1].axhline(0, color="r", lw=1.5, linestyle="--")
    axes[1].set_xlabel(f"Predicted ({unit})"); axes[1].set_ylabel("Residual")
    axes[1].set_title(f"{model_name} — Residual Plot")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight"); plt.close()


def _plot_feature_importance(importances, feature_names, model_name: str, filename: str, top_n: int = 15):
    n   = min(top_n, len(importances))
    idx = np.argsort(importances)[::-1][:n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in idx][::-1], importances[idx][::-1], color="#4C72B0")
    ax.set_xlabel("Importance"); ax.set_title(f"{model_name} — Feature Importances")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight"); plt.close()
