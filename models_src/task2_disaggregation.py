"""
Task 2 — Energy Disaggregation
===============================
Trains one XGBoost regressor per canonical appliance group.
Predicts:
  - absolute_kw  : how many kW this appliance is drawing
  - fraction     : appliance share of total Usage_kW (0–1)

At inference, the outputs give a full per-appliance breakdown that sums to ~100%.
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

from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.config import (
    CANONICAL_APPLIANCE_COLS, DISAGG_MODEL_DIR,
    PLOTS_DIR, PROCESSED_DIR, USAGE_COL,
)
from src.preprocessing import get_feature_cols


def train_disaggregation(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    scaler_X,
) -> dict:
    """
    Train one XGBoost model per appliance.
    Returns metrics dict keyed by appliance name.
    """
    feature_cols = get_feature_cols(train)
    results = {}
    contributions = {}

    print("\n  > Training per-appliance disaggregation models ...")

    total_train = train[USAGE_COL].replace(0, np.nan)
    total_test  = test[USAGE_COL].replace(0, np.nan)

    X_tr_scaled = scaler_X.transform(train[feature_cols].values)
    X_te_scaled = scaler_X.transform(test[feature_cols].values)

    for appl in CANONICAL_APPLIANCE_COLS:
        if appl not in train.columns:
            continue
        if train[appl].abs().sum() < 1e-6:
            continue  # skip zero-everywhere appliance

        # Target: fraction of total
        y_tr = (train[appl] / total_train).fillna(0).clip(0, 1).values
        y_te = (test[appl]  / total_test ).fillna(0).clip(0, 1).values

        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_tr_scaled, y_tr, verbose=False)
        pred = model.predict(X_te_scaled).clip(0, 1)

        mae = float(mean_absolute_error(y_te, pred))
        r2  = float(r2_score(y_te, pred))
        print(f"    {appl:20s} → MAE={mae:.4f}  R²={r2:.4f}")

        # Save model
        model_path = DISAGG_MODEL_DIR / f"disagg_{appl}.joblib"
        joblib.dump(model, model_path)

        results[appl] = {"MAE": mae, "R2": r2}
        contributions[appl] = float(train[appl].mean())

    # Contribution pie chart
    _plot_contribution_pie(contributions)
    print(f"    [OK] Disaggregation models saved → {DISAGG_MODEL_DIR}")
    return results


def _plot_contribution_pie(contributions: dict):
    if not contributions:
        return
    filtered = {k: v for k, v in contributions.items() if v > 0}
    if not filtered:
        return

    total = sum(filtered.values())
    labeled = {k: v / total * 100 for k, v in filtered.items()}

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("tab10", len(labeled))
    ax.pie(
        labeled.values(),
        labels=labeled.keys(),
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.82,
    )
    ax.set_title("Mean Appliance Energy Contribution\n(Training Set Average)", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "disagg_contribution_pie.png", dpi=150, bbox_inches="tight")
    plt.close()
