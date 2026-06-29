"""
Baseline models for comparison against the main XGBoost model.

Includes:
1. Climatological Baseline: Uses historical flood rate for the given day-of-year.
2. Logistic Regression: Simple linear physics-informed baseline.
3. Random Forest: Standard ML tree-based benchmark.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from floodai.evaluation.metrics import evaluate, DataProvenance
from floodai.training.threshold import select_f1_optimal_threshold

logger = logging.getLogger("floodai.models.baselines")


class ClimatologicalBaseline:
    """
    Pure calendar/location baseline.
    Predicts probability of flood based on historical frequency
    for a given basin and day of year (smoothed over a +/- 7 day window).
    Uses no meteorological or terrain features.
    """
    def __init__(self, window_days: int = 7):
        self.window_days = window_days
        self.climatology_: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        X must contain 'basin_key' and 'Day_of_Year'.
        """
        if "basin_key" not in X.columns or "Day_of_Year" not in X.columns:
            raise ValueError("Climatological baseline requires 'basin_key' and 'Day_of_Year'.")

        df = X[["basin_key", "Day_of_Year"]].copy()
        df["target"] = np.asarray(y)

        # Calculate raw daily rate per basin
        raw_rates = df.groupby(["basin_key", "Day_of_Year"])["target"].mean().reset_index()
        
        # Smooth with a rolling window (handle wraparound at end of year)
        smoothed_rates = []
        for basin in raw_rates["basin_key"].unique():
            basin_data = raw_rates[raw_rates["basin_key"] == basin].set_index("Day_of_Year")
            # Keep only the numeric target column before reindex
            basin_data = basin_data[["target"]].reindex(range(1, 367)).fillna(0)
            
            # Pad for wraparound
            padded = pd.concat([
                basin_data.iloc[-self.window_days:],
                basin_data,
                basin_data.iloc[:self.window_days]
            ])
            
            smoothed = padded["target"].rolling(window=2*self.window_days+1, center=True).mean()
            smoothed = smoothed.iloc[self.window_days:-self.window_days]
            
            smoothed_df = smoothed.reset_index()
            smoothed_df.columns = ["Day_of_Year", "target"]
            smoothed_df["basin_key"] = basin
            smoothed_rates.append(smoothed_df)

        self.climatology_ = pd.concat(smoothed_rates, ignore_index=True)
        logger.info(f"Fitted ClimatologicalBaseline (window +/- {self.window_days} days).")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.climatology_ is None:
            raise RuntimeError("Model not fitted.")
        
        df = X[["basin_key", "Day_of_Year"]].copy()
        # Merge to get probabilities
        merged = df.merge(
            self.climatology_,
            on=["basin_key", "Day_of_Year"],
            how="left"
        )
        
        # Fill missing with overall basin mean if day not seen, or 0
        basin_means = self.climatology_.groupby("basin_key")["target"].mean()
        
        probs = merged["target"].values
        for i, val in enumerate(probs):
            if pd.isna(val):
                basin = df.iloc[i]["basin_key"]
                probs[i] = basin_means.get(basin, 0.0)

        # Return shape (n_samples, 2) for sklearn compatibility
        return np.vstack((1 - probs, probs)).T


def run_baselines(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    feature_cols: list[str],
    raw_train_df: pd.DataFrame,
    raw_val_df: pd.DataFrame,
    raw_test_df: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """
    Trains and evaluates 3 baseline models on the test set.
    """
    results = {}
    
    # 1. Climatological Baseline
    logger.info("--- Running Climatological Baseline ---")
    climatology = ClimatologicalBaseline()
    climatology.fit(raw_train_df, y_train)
    
    val_probs_clim = climatology.predict_proba(raw_val_df)[:, 1]
    tau_clim = select_f1_optimal_threshold(np.asarray(y_val), val_probs_clim)
    
    test_probs_clim = climatology.predict_proba(raw_test_df)[:, 1]
    metrics_clim = evaluate(np.asarray(y_test), test_probs_clim, tau_clim, "Climatological", provenance=DataProvenance.HELD_OUT)
    results["Climatological"] = metrics_clim


    # 2. Logistic Regression
    logger.info("--- Running Logistic Regression Baseline ---")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train[feature_cols].fillna(0))
    X_val_scaled = scaler.transform(X_val[feature_cols].fillna(0))
    X_test_scaled = scaler.transform(X_test[feature_cols].fillna(0))

    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train_scaled, y_train)
    
    val_probs_lr = lr.predict_proba(X_val_scaled)[:, 1]
    tau_lr = select_f1_optimal_threshold(np.asarray(y_val), val_probs_lr)
    
    test_probs_lr = lr.predict_proba(X_test_scaled)[:, 1]
    metrics_lr = evaluate(np.asarray(y_test), test_probs_lr, tau_lr, "LogisticRegression", provenance=DataProvenance.HELD_OUT)
    results["Logistic Regression"] = metrics_lr


    # 3. Random Forest
    logger.info("--- Running Random Forest Baseline ---")
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    # RF doesn't strictly need scaling, but we use scaled for consistency
    rf.fit(X_train_scaled, y_train)
    
    val_probs_rf = rf.predict_proba(X_val_scaled)[:, 1]
    tau_rf = select_f1_optimal_threshold(np.asarray(y_val), val_probs_rf)
    
    test_probs_rf = rf.predict_proba(X_test_scaled)[:, 1]
    metrics_rf = evaluate(np.asarray(y_test), test_probs_rf, tau_rf, "RandomForest", provenance=DataProvenance.HELD_OUT)
    results["Random Forest"] = metrics_rf

    return results
