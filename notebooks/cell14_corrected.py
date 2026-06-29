# Steps 5-12: Full pipeline — CORRECTED to use floodai package modules
#
# This replaces a version of this cell that bypassed the floodai package
# entirely and reimplemented SMOTE/training inline. That version had two
# real bugs, both now structurally prevented:
#
#   1. `Year` (a raw integer column from add_temporal_features) was left in
#      feature_cols. With only ~10 verified flood events clustered in
#      certain years, the model could use Year as a lookup key for "which
#      years have a labeled flood" instead of learning rainfall/terrain
#      patterns. This produced LOBO AUC=1.000 / Recall=1.000 identically
#      across three basins with different flood mechanisms -- a sign of
#      leakage, not generalization.
#   2. `SMOTE(random_state=cfg.random_seed)` was called with no
#      `sampling_strategy`, silently defaulting to 50/50 class balancing
#      instead of the ~10% specified in config.yaml.
#
# Both are now caught automatically: select_model_features() excludes Year
# by construction, and resample_training_only() raises if the resulting
# ratio doesn't match what was requested.

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from floodai.evaluation.metrics import DataProvenance, evaluate, report_headline
from floodai.features.governance import assert_no_forbidden_columns, select_model_features
from floodai.models.xgb_model import build_xgb_classifier, fit_with_validation
from floodai.training.imbalance import resample_training_only
from floodai.training.label_sufficiency import check_basin_has_positives, check_split_has_positives
from floodai.training.lobo import run_lobo_cv
from floodai.training.threshold import select_f1_optimal_threshold
from floodai.training.tuning import run_optuna_search

# ── Step 5: Label floods ────────────────────────────────────────────────────
def label_floods(df, flood_events_df):
    df = df.copy()
    df['Flood_Occurred'] = 0
    for _, ev in flood_events_df.iterrows():
        if 'basin_key' in flood_events_df.columns:
            mask = (
                (df['basin_key'] == ev['basin_key']) &
                (df['Date'] >= ev['Start']) &
                (df['Date'] <= ev['End'])
            )
        else:
            mask = (df['Date'] >= ev['Start']) & (df['Date'] <= ev['End'])
        df.loc[mask, 'Flood_Occurred'] = 1
    return df

df = label_floods(df, flood_events_df)
vc = df['Flood_Occurred'].value_counts()
print(f"Flood label distribution:\n{vc}")
print(f"Positive rate: {vc.get(1,0)/len(df)*100:.2f}%")

# ── Step 5b: FAIL FAST if any split or basin lacks enough positive labels ──
# This is the check that would have caught "10 total events, none in the
# test window" immediately, instead of after a 35-minute Optuna run.
check_split_has_positives(
    df, date_col='Date', label_col='Flood_Occurred',
    train_years=cfg.raw['split']['train_years'],
    val_years=cfg.raw['split']['val_years'],
    test_years=cfg.raw['split']['test_years'],
    min_positives_per_split=5,
)
basin_counts = check_basin_has_positives(df, basin_col='basin_key', label_col='Flood_Occurred')
print(f"Per-basin positive counts: {basin_counts}")
print("[OK] All splits have sufficient positive labels. Proceeding.")

# ── Step 6: Temporal train / val / test split ───────────────────────────────
train_years = cfg.raw['split']['train_years']
val_years   = cfg.raw['split']['val_years']
test_years  = cfg.raw['split']['test_years']

df_train = df[df['Date'].dt.year.isin(train_years)].copy()
df_val   = df[df['Date'].dt.year.isin(val_years)].copy()
df_test  = df[df['Date'].dt.year.isin(test_years)].copy()

# EXCLUDE 'temporal' group! If the model has access to the calendar, it will take
# the lazy shortcut and just predict based on season instead of physical rainfall.
feature_groups = ["rainfall_current", "rainfall_windows", "rainfall_anomaly", "terrain_physics", "interaction"]
feature_cols = select_model_features(df, groups=feature_groups)
assert_no_forbidden_columns(feature_cols)
print(f"\nSelected {len(feature_cols)} governed features (Year and Temporal excluded by design):")
print(feature_cols)

X_train, y_train = df_train[feature_cols].values, df_train['Flood_Occurred'].values
X_val,   y_val   = df_val[feature_cols].values,   df_val['Flood_Occurred'].values
X_test,  y_test  = df_test[feature_cols].values,  df_test['Flood_Occurred'].values

print(f"\nSplit sizes — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ── Step 7: RobustScaler — fit on train ONLY ────────────────────────────────
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
print("RobustScaler fitted on train set only.")

# ── Step 8: SMOTE on training set only ─────────────────────────────────────
# CORRECTED: uses resample_training_only(), which enforces the configured
# sampling_strategy and raises if the actual post-SMOTE ratio drifts from it.
imb_cfg = cfg.raw['imbalance']
X_train_res, y_train_res = resample_training_only(
    X_train_scaled, y_train,
    sampling_strategy=imb_cfg['sampling_strategy'],
    k_neighbors_max=imb_cfg['k_neighbors_max'],
    seed=cfg.random_seed,
)
print(f"After SMOTE — X_train_res: {X_train_res.shape}, "
      f"positives: {y_train_res.sum()} / {len(y_train_res)} "
      f"({y_train_res.mean():.2%}, target was {imb_cfg['sampling_strategy']/(1+imb_cfg['sampling_strategy']):.2%})")

# ── Step 9: Optuna hyperparameter search (config-driven search space) ──────
search_space = cfg.raw['model']['optuna']['search_space']
n_trials = cfg.raw['model']['optuna']['n_trials']
early_stopping_rounds = cfg.raw['model']['early_stopping_rounds']

best_params = run_optuna_search(
    X_train_res, y_train_res, X_val_scaled, y_val,
    search_space=search_space, n_trials=n_trials,
    early_stopping_rounds=early_stopping_rounds, seed=cfg.random_seed,
)
print(f"\nBest params: {best_params}")

best_model = build_xgb_classifier(best_params, early_stopping_rounds, cfg.random_seed)
best_model = fit_with_validation(best_model, X_train_res, y_train_res, X_val_scaled, y_val)

# ── Step 10: F1-optimal threshold on validation set ────────────────────────
val_proba = best_model.predict_proba(X_val_scaled)[:, 1]
tau_star = select_f1_optimal_threshold(y_val, val_proba)

# ── Step 11: Evaluate on held-out test set (provenance-tagged) ────────────
test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
result = evaluate(
    y_test, test_proba, threshold=tau_star,
    set_name=f"test_{test_years[0]}_{test_years[-1]}",
    provenance=DataProvenance.HELD_OUT,
)
headlined = report_headline(result)  # raises if provenance isn't HELD_OUT — see evaluation/metrics.py

print("\n" + "="*55)
print("  HELD-OUT TEST SET RESULTS (headline-approved)")
print("="*55)
print(f"  ROC-AUC   : {headlined.roc_auc:.4f}")
print(f"  PR-AUC    : {headlined.pr_auc:.4f}")
print(f"  F1 Score  : {headlined.f1:.4f}")
print(f"  MCC       : {headlined.mcc:.4f}")
print(f"  Bal. Acc  : {headlined.balanced_accuracy:.4f}")
print(f"  FAR       : {headlined.far:.4f}")
print(f"  CSI       : {headlined.csi:.4f}")
print(f"  Threshold : {headlined.threshold:.4f}")
tn, fp, fn, tp = headlined.confusion
print(f"  TN={tn:>6}  FP={fp:>6}  FN={fn:>6}  TP={tp:>6}")
print("="*55)

# ── Step 12: Leave-One-Basin-Out (LOBO) cross-validation ───────────────────
# CORRECTED: uses run_lobo_cv() from the package, which (a) calls
# assert_no_forbidden_columns() internally, and (b) routes all SMOTE calls
# through resample_training_only(), so the Year-leakage and SMOTE-ratio bugs
# cannot recur in this step either.
print("\nRunning LOBO cross-validation (governed)...")
lobo_results = run_lobo_cv(
    df, feature_columns=feature_cols, target_column='Flood_Occurred',
    basin_column='basin_key', best_params=best_params,
    early_stopping_rounds=early_stopping_rounds,
    smote_sampling_strategy=imb_cfg['sampling_strategy'],
    smote_k_neighbors_max=imb_cfg['k_neighbors_max'],
    seed=cfg.random_seed,
)

lobo_df = pd.DataFrame([{
    'held_out_basin': r.set_name.replace('LOBO_held_out_', ''),
    'ROC_AUC': r.roc_auc, 'F1': r.f1, 'MCC': r.mcc, 'n_positive': r.n_positive,
} for r in lobo_results])
print("\nLOBO Summary (provenance=loso_held_out — NOT a headline metric, see metrics.py):")
print(lobo_df.to_string(index=False))
if len(lobo_df) > 0:
    print(f"\nMean LOBO ROC-AUC : {lobo_df['ROC_AUC'].mean():.4f} +/- {lobo_df['ROC_AUC'].std():.4f}")
    print(f"Mean LOBO F1      : {lobo_df['F1'].mean():.4f} +/- {lobo_df['F1'].std():.4f}")
    print("\nNOTE: If ROC_AUC == 1.000 for every basin again, STOP and investigate")
    print("      before reporting -- that pattern previously indicated leakage,")
    print("      not genuine cross-basin generalization. Check feature_cols")
    print("      above for anything that could identify a specific basin/year")
    print("      rather than encode rainfall/terrain physics.")
