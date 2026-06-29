"""
SHAP analysis module for XGBoost model interpretability.
Requires: pip install shap matplotlib
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("floodai.evaluation.shap")


def run_shap_analysis(
    model: Any, 
    X_test: pd.DataFrame, 
    feature_cols: list[str],
    output_dir: str = "/content/floodai_outputs"
) -> None:
    """
    Computes SHAP values on a subsample of the test set and generates plots.
    """
    try:
        import shap
    except ImportError as exc:
        logger.error("SHAP not installed. Run: !pip install shap matplotlib")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Handle both numpy arrays and DataFrames
    if isinstance(X_test, np.ndarray):
        X_test_df = pd.DataFrame(X_test, columns=feature_cols)
    else:
        X_test_df = X_test[feature_cols].copy()

    # Subsample to speed up SHAP calculation if test set is large
    if len(X_test_df) > 10000:
        logger.info(f"Subsampling test set for SHAP from {len(X_test_df)} to 10000...")
        X_sample = X_test_df.sample(n=10000, random_state=42).fillna(0)
    else:
        X_sample = X_test_df.fillna(0)

    logger.info("Computing SHAP values using TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 1. Global summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Feature Importance (Test Set)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=300)
    plt.close()
    
    # Calculate mean absolute SHAP values for numerical summary
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False)
    
    logger.info(f"Top 5 features by SHAP:\n{importance_df.head(5)}")

    # 2. Terrain vs Rainfall Contribution
    terrain_cols = [c for c in feature_cols if c in [
        "Elevation_m", "Curve_Number", "TWI", "CN_Runoff_Q", 
        "Elevation_Rain_Ratio", "Elevation_Rain30_Ratio", "Low_Elev_Heavy_Rain",
        "CN_Rain_Interaction", "TWI_Rain_Interaction"
    ]]
    rainfall_cols = [c for c in feature_cols if "Rain" in c and c not in terrain_cols]
    
    terrain_idx = [feature_cols.index(c) for c in terrain_cols if c in feature_cols]
    rain_idx = [feature_cols.index(c) for c in rainfall_cols if c in feature_cols]
    
    terrain_impact = mean_abs_shap[terrain_idx].sum() if terrain_idx else 0
    rain_impact = mean_abs_shap[rain_idx].sum() if rain_idx else 0
    total_impact = mean_abs_shap.sum()
    
    logger.info("--- SHAP Feature Group Contributions ---")
    logger.info(f"Terrain/Topography: {terrain_impact / total_impact:.1%}")
    logger.info(f"Rainfall:           {rain_impact / total_impact:.1%}")
    
    print("\n=== SHAP Analysis Complete ===")
    print(f"Summary plot saved to {output_dir}/shap_summary.png")
    print("\nTop 10 features driving predictions:")
    print(importance_df.head(10).to_string(index=False))
