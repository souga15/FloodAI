"""
Step 8b — Baselines, Conformal Prediction & SHAP Analysis
Run this cell AFTER the main pipeline completes and trains the XGBoost model.
"""
import sys
import importlib
import os
sys.path.insert(0, "/content/FloodAI/src")

# Install SHAP if not present
import subprocess
subprocess.run(["pip", "install", "-q", "shap", "matplotlib"], check=True)

import pandas as pd
import numpy as np

os.makedirs("/content/floodai_outputs", exist_ok=True)

# Force-reload modules so git pull changes take effect without session restart
import floodai.models.baselines as _bm
import floodai.evaluation.shap_analysis as _sa
import floodai.evaluation.conformal as _cp
importlib.reload(_bm)
importlib.reload(_sa)
importlib.reload(_cp)

from floodai.models.baselines import run_baselines
from floodai.evaluation.shap_analysis import run_shap_analysis
from floodai.evaluation.conformal import add_conformal_to_results

# ── Baselines ────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  RUNNING BASELINE MODELS")
print("="*50)

train_years = cfg.raw['split']['train_years']
val_years   = cfg.raw['split']['val_years']
test_years  = cfg.raw['split']['test_years']

try:
    raw_train_df = df[df["Date"].dt.year.isin(train_years)].copy()
    raw_val_df   = df[df["Date"].dt.year.isin(val_years)].copy()
    raw_test_df  = df[df["Date"].dt.year.isin(test_years)].copy()

    baseline_results = run_baselines(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_cols,
        raw_train_df, raw_val_df, raw_test_df
    )

    print("\n--- Baseline Results (Test 2023-2024) ---")
    for model_name, res in baseline_results.items():
        print(f"\n{model_name}:")
        print(f"  ROC-AUC    : {res.roc_auc:.4f}")
        print(f"  PR-AUC     : {res.pr_auc:.4f}")
        print(f"  F1 Score   : {res.f1:.4f}")
        print(f"  MCC        : {res.mcc:.4f}")
        print(f"  FAR        : {res.far:.4f}")

    # Compare XGBoost vs best baseline
    best_baseline_auc = max(r.roc_auc for r in baseline_results.values())
    print(f"\n[COMPARISON] XGBoost AUC={headlined.roc_auc:.4f} vs Best Baseline AUC={best_baseline_auc:.4f}")
    delta = headlined.roc_auc - best_baseline_auc
    if delta > 0.02:
        print(f"[OK] XGBoost beats baseline by {delta:.4f} — meaningful improvement.")
    else:
        print(f"[WARNING] Gap is only {delta:.4f} — consider if model is adding real value over climatology.")

except NameError as e:
    print(f"Error: Required variables not found in memory ({e}).")
    print("Please ensure you have run the main pipeline (Step 7) before this cell.")


# ── Conformal Prediction Intervals ──────────────────────────────────────────
print("\n" + "="*50)
print("  CONFORMAL PREDICTION INTERVALS (90% coverage)")
print("="*50)

try:
    interval_df, cal_result = add_conformal_to_results(
        y_val=y_val,
        val_proba=val_proba,
        y_test=y_test,
        test_proba=test_proba,
        alpha=0.10,  # 90% coverage guarantee
    )

    print(f"\nCalibration summary:")
    print(f"  q_hat (interval half-width) : {cal_result.q_hat:.4f}")
    print(f"  Target coverage             : {cal_result.coverage_target:.0%}")
    print(f"  Empirical coverage (val)    : {cal_result.empirical_coverage:.1%}")
    print(f"  Calibration set size        : {cal_result.n_calibration:,}")

    print(f"\nInterval width on test set:")
    print(f"  Mean  : {interval_df['interval_width'].mean():.4f}")
    print(f"  Median: {interval_df['interval_width'].median():.4f}")
    print(f"  (high-risk days p>0.7): mean width = "
          f"{interval_df.loc[interval_df['flood_prob'] > 0.7, 'interval_width'].mean():.4f}")

    interval_df.to_csv("/content/floodai_outputs/conformal_intervals_test.csv", index=False)
    print(f"\nSaved: /content/floodai_outputs/conformal_intervals_test.csv")

except NameError as e:
    print(f"Skipping conformal (missing variable: {e}). Run Step 7 first.")


# ── SHAP Analysis ────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  RUNNING SHAP INTERPRETABILITY ANALYSIS")
print("="*50)

try:
    run_shap_analysis(
        model=best_model,
        X_test=X_test_scaled,
        feature_cols=feature_cols,
        output_dir="/content/floodai_outputs"
    )
except NameError as e:
    print(f"Error running SHAP ({e}). Did you run Step 7 to define 'best_model'?")
