"""
Integration test: runs the full pipeline (features -> SMOTE -> tuning ->
threshold selection -> evaluation -> LOBO) on small synthetic data to prove
the modules actually connect, not just that each passes in isolation.

This is intentionally fast (few Optuna trials, small data) — it is a wiring
test, not a performance benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from floodai.evaluation.metrics import DataProvenance, evaluate, report_headline
from floodai.features.pipeline import (
    add_interaction_features,
    add_rainfall_window_features,
    add_scs_cn_runoff,
    add_temporal_features,
)
from floodai.models.xgb_model import build_xgb_classifier, fit_with_validation
from floodai.training.imbalance import resample_training_only
from floodai.training.lobo import run_lobo_cv
from floodai.training.threshold import select_f1_optimal_threshold
from floodai.training.tuning import run_optuna_search


def _build_synthetic_basin_dataset(seed: int = 42) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    basins = ["ganga_bihar", "brahmaputra_assam", "mahanadi_odisha"]
    rows = []
    for basin in basins:
        for point_id in range(8):  # small N for speed
            dates = pd.date_range("2017-01-01", "2024-12-31", freq="D")
            rain = rng.gamma(1.0, 6.0, size=len(dates))
            # Inject a monsoon-season rain bump so the synthetic flood label is learnable
            month = dates.month
            monsoon_boost = np.where(np.isin(month, [7, 8]), rng.gamma(2.0, 15.0, len(dates)), 0)
            rain = rain + monsoon_boost
            elevation = rng.uniform(5, 100)
            curve_number = rng.uniform(65, 90)
            humidity = rng.uniform(60, 95, size=len(dates))
            temp = rng.uniform(20, 35, size=len(dates))

            df_point = pd.DataFrame({
                "Date": dates,
                "Region": f"{basin}_{point_id}",
                "Basin": basin,
                "Rainfall_mm": rain,
                "Elevation_m": elevation,
                "Curve_Number": curve_number,
                "Humidity_pct": humidity,
                "Temperature_C": temp,
            })
            rows.append(df_point)

    df = pd.concat(rows, ignore_index=True).sort_values(["Region", "Date"]).reset_index(drop=True)
    df = add_temporal_features(df)
    df = add_rainfall_window_features(df, group_col="Region")
    df = add_scs_cn_runoff(df)
    df = add_interaction_features(df)

    # Synthetic flood label: driven by 7-day rainfall + CN runoff, with noise,
    # so the model has a real (if synthetic) signal to learn.
    risk_score = (
        0.02 * df["Rainfall_7Day_mm"] + 0.05 * df["CN_Runoff_Q"] + rng.normal(0, 2, len(df))
    )
    threshold = np.percentile(risk_score, 97)  # ~3% positive rate, imbalanced like real data
    df["Flood_Occurred"] = (risk_score > threshold).astype(int)
    df["Year"] = df["Date"].dt.year

    feature_columns = [
        "Month", "Day_of_Year", "Is_Monsoon_Season", "Is_Peak_Monsoon",
        "Month_Sin", "Month_Cos", "Rainfall_3Day_mm", "Rainfall_7Day_mm",
        "Rainfall_14Day_mm", "Rainfall_30Day_mm", "Rainfall_60Day_mm",
        "Heavy_Rain_Days_7D", "Consecutive_Dry_Days", "Soil_Moisture_Proxy",
        "Elevation_m", "Curve_Number", "CN_Runoff_Q",
        "Elevation_Rain_Ratio", "Monsoon_Rain_Interaction",
        "Soil_Monsoon_Interaction",
    ]
    return df, feature_columns


class TestFullPipelineIntegration:
    def test_pipeline_runs_end_to_end_and_produces_valid_headline_metric(self):
        df, feature_columns = _build_synthetic_basin_dataset()
        seed = 42

        train_mask = df["Year"].isin([2017, 2018, 2019, 2020])
        val_mask = df["Year"].isin([2021, 2022])
        test_mask = df["Year"].isin([2023, 2024])

        X_train, y_train = df.loc[train_mask, feature_columns], df.loc[train_mask, "Flood_Occurred"]
        X_val, y_val = df.loc[val_mask, feature_columns], df.loc[val_mask, "Flood_Occurred"]
        X_test, y_test = df.loc[test_mask, feature_columns], df.loc[test_mask, "Flood_Occurred"]

        assert y_train.sum() > 0 and y_val.sum() > 0 and y_test.sum() > 0, (
            "Synthetic label generation produced a split with zero positives; "
            "adjust the synthetic risk_score threshold."
        )

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        X_train_res, y_train_res = resample_training_only(
            X_train_scaled, y_train.values, sampling_strategy=0.10, k_neighbors_max=5, seed=seed
        )

        search_space = {
            "n_estimators": {"low": 50, "high": 80},
            "max_depth": {"low": 3, "high": 4},
            "learning_rate": {"low": 0.05, "high": 0.10, "log": True},
        }
        best_params = run_optuna_search(
            X_train_res, y_train_res, X_val_scaled, y_val.values,
            search_space=search_space, n_trials=3, early_stopping_rounds=10, seed=seed,
        )

        model = build_xgb_classifier(best_params, early_stopping_rounds=10, random_seed=seed)
        model = fit_with_validation(model, X_train_res, y_train_res, X_val_scaled, y_val.values)

        proba_val = model.predict_proba(X_val_scaled)[:, 1]
        threshold = select_f1_optimal_threshold(y_val.values, proba_val)

        proba_test = model.predict_proba(X_test_scaled)[:, 1]
        result = evaluate(
            y_test.values, proba_test, threshold=threshold,
            set_name="test_2023_2024", provenance=DataProvenance.HELD_OUT,
        )
        headlined = report_headline(result)  # must not raise

        assert 0.0 <= headlined.roc_auc <= 1.0
        assert 0.0 <= headlined.f1 <= 1.0
        assert headlined.n_samples == len(y_test)

    def test_lobo_runs_across_synthetic_basins(self):
        df, feature_columns = _build_synthetic_basin_dataset()
        results = run_lobo_cv(
            df, feature_columns, target_column="Flood_Occurred", basin_column="Basin",
            best_params={"n_estimators": 50, "max_depth": 3, "learning_rate": 0.08},
            early_stopping_rounds=10, smote_sampling_strategy=0.10, smote_k_neighbors_max=5,
            seed=42,
        )
        assert len(results) >= 1, "Expected at least one valid LOBO fold from 3 synthetic basins"
        for r in results:
            assert r.provenance.value == "loso_held_out"
