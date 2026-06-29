"""
Step 8b — Baselines & SHAP Analysis
Run this cell AFTER the main pipeline completes and trains the XGBoost model.
It evaluates 3 baseline models and generates SHAP interpretability plots.
"""
import sys
import importlib
sys.path.insert(0, "/content/FloodAI/src")

# Install SHAP if not present
import subprocess
subprocess.run(["pip", "install", "-q", "shap", "matplotlib"], check=True)

import pandas as pd

# Force-reload modules so git pull changes take effect without session restart
import floodai.models.baselines as _bm
import floodai.evaluation.shap_analysis as _sa
importlib.reload(_bm)
importlib.reload(_sa)

from floodai.models.baselines import run_baselines
from floodai.evaluation.shap_analysis import run_shap_analysis

print("\n" + "="*50)
print("  RUNNING BASELINE MODELS")
print("="*50)

# We need the raw dataframes before SMOTE/scaling for the baselines
# (The baselines handle their own scaling/setup)
# Make sure X_train, y_train, X_val, y_val, X_test, y_test, df, and feature_cols are in memory
try:
    raw_train_df = df[df["Year"].isin([2017, 2018, 2019, 2020])].copy()
    raw_val_df = df[df["Year"].isin([2021, 2022])].copy()
    raw_test_df = df[df["Year"].isin([2023, 2024])].copy()
    
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
except NameError as e:
    print(f"Error: Required variables not found in memory ({e}).")
    print("Please ensure you have run the main pipeline (Step 7) before this cell.")


print("\n" + "="*50)
print("  RUNNING SHAP INTERPRETABILITY ANALYSIS")
print("="*50)

try:
    run_shap_analysis(
        model=best_model,  # The trained XGBoost model from Step 7
        X_test=X_test_scaled,
        feature_cols=feature_cols,
        output_dir="/content/floodai_outputs"
    )
except NameError as e:
    print(f"Error running SHAP ({e}). Did you run Step 7 to define 'best_model'?")

